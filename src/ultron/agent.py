"""The agent loop.

    goal -> cheap router -> plan -> pick tool -> policy check -> Docker sandbox
         -> verify -> save to memory -> next step | finish

Cost discipline baked into the loop:

* **Money is only spent at two points**: the optional router escalation and the
  optional judge/synthesis. Every other stage is deterministic and free.
* **Budgets are a kill-switch**, not a warning: USD, wall-clock, step count and
  LLM-call count are checked before and after every stage, and breaching any of
  them stops the run immediately (`BudgetExceeded`).
* **Cache and recall short-circuit before anything else.** A repeat goal returns
  the remembered answer with zero tool calls and zero tokens.
* **A step that fails verification is never cached and never remembered**, so a
  flaky tool cannot poison later runs.
* **Refusals are results too.** A policy denial ends the run with status
  ``denied`` and a human-readable explanation, rather than an exception.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .breaker import BreakerVerifier
from .cache import Cache, make_key
from .config import Settings, get_settings
from .errors import (
    BudgetExceeded,
    HumanApprovalRequired,
    PolicyDenied,
    SandboxError,
    SandboxUnavailable,
)
from .forge import ForgeEngine
from .ledger import FailureLedger
from .llm import LLMClient
from .memory import Memory, RunRecord
from .planner import Plan, Planner, PlanStep
from .policy import PolicyDecision, PolicyGate, PolicyRequest
from .provenance import ProvenanceEnvelope
from .registry import Registry
from .router import RouteDecision, Router
from .sandbox import Sandbox, SandboxResult
from .verifier import VerificationResult, Verifier

RunStatus = Literal[
    "ok",
    "denied",
    "failed",
    "budget_exceeded",
    "verification_failed",
    "ungrounded_rejection",
    "no_plan",
]
AnswerSource = Literal["tools", "memory", "direct", "synthesis", "none"]


# --------------------------------------------------------------------- budgets
@dataclass(slots=True)
class Budget:
    """Hard spend/step/time ceilings for a single run."""

    max_usd: float = 0.05
    max_steps: int = 6
    max_seconds: float = 120.0
    max_llm_calls: int = 12

    spent_usd: float = 0.0
    steps: int = 0
    llm_calls: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    #: Set when a limit is breached; the loop stops at the next check.
    breached: str = ""

    def charge(self, *, usd: float = 0.0, llm_calls: int = 0, steps: int = 0) -> None:
        self.spent_usd += max(0.0, usd)
        self.llm_calls += max(0, llm_calls)
        self.steps += max(0, steps)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at

    def check(self) -> None:
        if self.spent_usd > self.max_usd:
            self.breached = "usd"
            raise BudgetExceeded(
                f"budget exceeded: spent ${self.spent_usd:.6f} > ${self.max_usd:.6f}",
                limit="usd",
                spent=self.spent_usd,
            )
        if self.llm_calls > self.max_llm_calls:
            self.breached = "llm_calls"
            raise BudgetExceeded(
                f"budget exceeded: {self.llm_calls} LLM calls > {self.max_llm_calls}",
                limit="llm_calls",
                spent=float(self.llm_calls),
            )
        if self.steps > self.max_steps:
            self.breached = "steps"
            raise BudgetExceeded(
                f"budget exceeded: {self.steps} steps > {self.max_steps}",
                limit="steps",
                spent=float(self.steps),
            )
        if self.elapsed_s > self.max_seconds:
            self.breached = "seconds"
            raise BudgetExceeded(
                f"budget exceeded: {self.elapsed_s:.1f}s > {self.max_seconds:.1f}s",
                limit="seconds",
                spent=self.elapsed_s,
            )

    def warn_near(self) -> list[str]:
        warnings: list[str] = []
        if self.spent_usd > 0.8 * self.max_usd:
            warnings.append(f"cost at {self.spent_usd / self.max_usd:.0%} of budget")
        if self.elapsed_s > 0.8 * self.max_seconds:
            warnings.append(f"time at {self.elapsed_s / self.max_seconds:.0%} of budget")
        return warnings

    def as_dict(self) -> dict[str, Any]:
        return {
            "spent_usd": round(self.spent_usd, 6),
            "max_usd": self.max_usd,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "elapsed_s": round(self.elapsed_s, 3),
            "max_seconds": self.max_seconds,
        }


# ---------------------------------------------------------------------- reports
@dataclass(slots=True)
class StepReport:
    index: int
    tool: str
    version: str
    inputs: dict[str, Any]
    risk: str
    policy_action: str
    policy_reason: str
    network: str
    ok: bool = False
    cached: bool = False
    duration_s: float = 0.0
    cost_usd: float = 0.0
    verification: VerificationResult | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    provenance: list[ProvenanceEnvelope] = field(default_factory=list)
    #: Why the step failed, so the run status is honest:
    #: policy (refused), sandbox (environment/execution), tool (bad output),
    #: verification (output rejected), breaker (ungrounded) or none.
    failure_kind: Literal["none", "policy", "sandbox", "tool", "verification", "breaker"] = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "version": self.version,
            "inputs": self.inputs,
            "risk": self.risk,
            "policy_action": self.policy_action,
            "policy_reason": self.policy_reason,
            "network": self.network,
            "ok": self.ok,
            "cached": self.cached,
            "duration_s": round(self.duration_s, 4),
            "cost_usd": self.cost_usd,
            "verification": self.verification.as_dict() if self.verification else None,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "result": self.result,
            "provenance": [item.as_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class AgentResult:
    run_id: str
    goal: str
    status: RunStatus
    ok: bool
    answer: str = ""
    answer_source: AnswerSource = "none"
    route: RouteDecision | None = None
    plan: Plan | None = None
    #: Ordered capability chain selected by the planner, if any.
    chain: list[str] = field(default_factory=list)
    provenance: list[ProvenanceEnvelope] = field(default_factory=list)
    steps: list[StepReport] = field(default_factory=list)
    budget: Budget | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    cost_basis: str = "actual"
    notes: list[str] = field(default_factory=list)
    recalled_from: str | None = None
    refused: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return bool(self.steps) and all(s.verification and s.verification.ok for s in self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status,
            "ok": self.ok,
            "answer": self.answer,
            "answer_source": self.answer_source,
            "route": self.route.as_dict() if self.route else None,
            "plan": self.plan.as_dict() if self.plan else None,
            "chain": self.chain,
            "provenance": [item.as_dict() for item in self.provenance],
            "steps": [s.as_dict() for s in self.steps],
            "budget": self.budget.as_dict() if self.budget else None,
            "cost_usd": round(self.cost_usd, 8),
            "cost_basis": self.cost_basis,
            "latency_s": round(self.latency_s, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "llm_calls": self.llm_calls,
            "llm_cache_hits": self.llm_cache_hits,
            "verified": self.verified,
            "recalled_from": self.recalled_from,
            "refused": self.refused,
            "notes": self.notes,
        }

    def summary(self) -> str:
        parts = [
            f"{self.status.upper()} {self.run_id}",
            f"cost=${self.cost_usd:.6f}",
            f"{self.latency_s:.2f}s",
            f"steps={len(self.steps)}",
            f"cache={self.cache_hits}/{self.cache_hits + self.cache_misses}",
        ]
        return " | ".join(parts)


# ------------------------------------------------------------------------ agent
class Agent:
    """Wires the pieces together. Every dependency is injectable for tests."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        registry: Registry | None = None,
        cache: Cache | None = None,
        memory: Memory | None = None,
        llm: LLMClient | None = None,
        gate: PolicyGate | None = None,
        sandbox: Sandbox | None = None,
        breaker: BreakerVerifier | None = None,
        forge_engine: ForgeEngine | None = None,
        run_id: str | None = None,
        interactive: bool = False,
        use_memory_recall: bool = True,
        sandbox_backend: Literal["docker", "local"] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or Registry(self.settings).load()
        self.run_id = run_id or Memory.new_run_id()
        self.cache = cache or Cache(self.settings, run_id=self.run_id)
        self.memory = memory or Memory(self.settings)
        self.ledger = FailureLedger(self.settings.state_dir)
        self.forge_engine = forge_engine
        self._forge_attempted: set[str] = set()
        self.llm = llm or LLMClient(self.cache, self.settings)
        self.breaker = breaker or BreakerVerifier()
        self.gate = gate or PolicyGate(self.settings, run_id=self.run_id)
        self.sandbox = sandbox or Sandbox(
            self.settings, backend=sandbox_backend, cache=self.cache, run_id=self.run_id
        )
        self.router = Router(self.registry, cache=self.cache, llm=self.llm, settings=self.settings)
        self.planner = Planner(
            self.registry, cache=self.cache, llm=self.llm, settings=self.settings
        )
        self.verifier = Verifier(
            cache=self.cache,
            llm=self.llm,
            settings=self.settings,
            breaker=self.breaker,
        )
        self.interactive = interactive
        self.use_memory_recall = use_memory_recall

    # --------------------------------------------------------------------- run
    def run(self, goal: str, *, max_steps: int | None = None) -> AgentResult:
        started = time.perf_counter()
        goal = (goal or "").strip()
        budget = Budget(
            max_usd=self.settings.budget_max_usd,
            max_steps=max_steps or self.settings.budget_max_steps,
            max_seconds=self.settings.budget_max_seconds,
            max_llm_calls=self.settings.budget_max_llm_calls,
        )
        result = AgentResult(
            run_id=self.run_id, goal=goal, status="failed", ok=False, budget=budget
        )

        # --- 0. memory recall: a verified answer to the same goal is free ----
        if self.use_memory_recall and goal:
            recall = self.memory.recall(goal, self.registry.fingerprint)
            if recall is not None:
                result.status = "ok"
                result.ok = True
                result.answer = recall["answer"]
                result.answer_source = "memory"
                result.recalled_from = recall["run_id"]
                result.notes.append(
                    f"recalled verified answer from {recall['run_id']} "
                    f"(age {recall['age_s']:.0f}s, original cost ${recall['prior_cost_usd']:.6f})"
                )
                result.latency_s = time.perf_counter() - started
                return self._finalize(result, recalled=True)

        run_id = self.memory.start_run(
            goal,
            run_id=self.run_id,
            registry_fingerprint=self.registry.fingerprint,
        )

        try:
            # --- 1. route ---------------------------------------------------
            budget.check()
            route = self.router.route(goal)
            result.route = route
            if not route.cached and route.cost_usd:
                budget.charge(usd=route.cost_usd, llm_calls=1)
            elif route.source == "llm":
                budget.charge(llm_calls=1)
            budget.check()
            self.memory.set_note("last_route", route.as_dict())
            if route.cheap_path:
                result.notes.append(f"cheap path: routing cost $0 ({route.reason})")

            # --- 2. plan ----------------------------------------------------
            plan = self.planner.plan(goal, route)
            result.plan = plan
            result.chain = list(plan.chain)
            result.notes.extend(plan.notes)
            if plan.cost_usd:
                budget.charge(usd=plan.cost_usd, llm_calls=1)
            budget.check()

            # --- 3. execute steps -------------------------------------------
            if plan.is_empty:
                self.ledger.record_gap(
                    goal,
                    expected_outputs={"answer": "string"},
                    suggested_provides=["answer.text"],
                    failure_reason="planner found no executable tool or capability chain",
                )
                result.notes.append("missing capability recorded in the failure ledger")
                if self.forge_engine is not None and goal not in self._forge_attempted:
                    self._forge_attempted.add(goal)
                    forged = self.forge_engine.auto_forge_from_ledger(top_n=1)
                    if forged:
                        self.registry = self.forge_engine.registry
                        self.router.registry = self.registry
                        self.planner.registry = self.registry
                        result.notes.append(
                            f"forged {', '.join(item.key for item in forged)}; retrying"
                        )
                        return self.run(goal, max_steps=max_steps)
                return self._finish_without_tools(result, run_id, started, goal)

            for step in plan.steps:
                budget.charge(steps=1)
                budget.check()
                executable_step = self._materialize_chain_step(step, result.steps)
                report = self._execute_step(run_id, executable_step, goal, budget)
                result.steps.append(report)
                result.provenance.extend(report.provenance)
                if not report.ok:
                    if report.failure_kind == "policy":
                        result.status = "denied"
                        result.refused.append(
                            {
                                "tool": f"{report.tool}@{report.version}",
                                "risk": report.risk,
                                "action": report.policy_action,
                                "reason": report.policy_reason,
                            }
                        )
                    elif report.failure_kind == "breaker":
                        result.status = "ungrounded_rejection"
                    elif report.failure_kind == "verification":
                        result.status = "verification_failed"
                    else:  # sandbox / tool failure: an execution problem, not a policy one
                        result.status = "failed"
                    result.notes.append(f"stopping after step {step.index}: {report.error}")
                    result.answer = self._compose_answer(result, partial=True)
                    result.answer_source = "tools" if report.result else "none"
                    return self._finalize(result, run_id=run_id, started=started)

            # --- 4. answer --------------------------------------------------
            budget.check()
            answer, source = self._answer(result, goal, budget)
            result.answer = answer
            result.answer_source = source
            final_provenance = list(result.provenance)
            if source == "synthesis":
                final_provenance.append(
                    ProvenanceEnvelope.llm_output(
                        f"llm:{make_key(goal, answer)[:32]}",
                        answer,
                    )
                )
            final_breaker = self.breaker.verify({"answer": answer}, final_provenance)
            if not final_breaker.ok:
                result.status = "ungrounded_rejection"
                result.ok = False
                result.notes.append(f"breaker rejected final answer: {final_breaker.reason}")
                result.provenance = final_provenance
                return self._finalize(result, run_id=run_id, started=started)
            result.provenance = final_provenance
            result.status = "ok"
            result.ok = True
            budget.check()
            return self._finalize(result, run_id=run_id, started=started)

        except BudgetExceeded as exc:
            result.status = "budget_exceeded"
            result.ok = False
            result.notes.append(str(exc))
            result.answer = self._compose_answer(result, partial=True)
            return self._finalize(result, run_id=run_id, started=started)
        except (SandboxUnavailable, SandboxError) as exc:
            result.status = "failed"
            result.ok = False
            result.notes.append(f"sandbox: {exc}")
            return self._finalize(result, run_id=run_id, started=started)

    # -------------------------------------------------------------------- steps
    def _materialize_chain_step(self, step: PlanStep, completed: list[StepReport]) -> PlanStep:
        """Pass the previous verified result into a composed step when possible.

        Manifests deliberately describe semantic types, not an unsafe executable
        wiring language. For the common one-input case, map the previous result
        to the declared field immediately before policy validation; multi-input
        tools keep any fields that can be copied by name and otherwise fail closed
        at the normal input-schema check.
        """
        if step.chain_input is None or step.chain_input >= len(completed):
            return step
        previous = completed[step.chain_input].result or {}
        manifest = self.registry.get(step.tool, step.version)
        if not manifest.inputs:
            return replace(step, inputs={})

        inputs = dict(step.inputs)
        matching = {name: previous[name] for name in manifest.inputs if name in previous}
        inputs.update(matching)
        if len(manifest.inputs) == 1:
            name, typ = next(iter(manifest.inputs.items()))
            if name not in inputs:
                if typ in {"dict", "any"}:
                    inputs[name] = previous
                elif typ == "string":
                    value = previous.get("text", previous.get("summary", previous))
                    inputs[name] = (
                        value if isinstance(value, str) else json.dumps(value, sort_keys=True)
                    )
        return replace(step, inputs=inputs)

    def _execute_step(self, run_id: str, step: PlanStep, goal: str, budget: Budget) -> StepReport:
        manifest = self.registry.get(step.tool, step.version)
        report = StepReport(
            index=step.index,
            tool=manifest.name,
            version=manifest.version,
            inputs=step.inputs,
            risk=manifest.risk.value,
            policy_action="none",
            policy_reason="",
            network="none",
        )
        request = PolicyRequest(
            tool=manifest,
            inputs=step.inputs,
            run_id=run_id,
            goal=goal,
            step=step.index,
            action_type="network_egress" if manifest.wants_network else "tool_execution",
        )

        # --- policy gate: the choke point ----------------------------------
        try:
            decision: PolicyDecision = self.gate.evaluate(request, interactive=self.interactive)
        except (PolicyDenied, HumanApprovalRequired) as exc:
            # peek, not check: this path only reports what the gate decided. It must
            # not claim a stored approval as a side effect of logging.
            decision = self.gate.peek(request)
            report.policy_action = decision.action
            report.policy_reason = decision.reason
            report.error = str(exc)
            report.network = "none"
            report.failure_kind = "policy"
            self.memory.record_step(
                run_id,
                step_index=step.index,
                tool=manifest.name,
                version=manifest.version,
                inputs=step.inputs,
                risk=manifest.risk.value,
                policy_action=decision.action,
                ok=False,
                error=str(exc),
            )
            return report

        report.policy_action = decision.action
        report.policy_reason = decision.reason
        report.network = decision.network

        # --- sandbox -------------------------------------------------------
        start = time.perf_counter()
        try:
            outcome: SandboxResult = self.sandbox.run(
                manifest,
                step.inputs,
                decision,
                defer_cache_write=True,
            )
        except (SandboxError, SandboxUnavailable) as exc:
            report.error = f"sandbox error: {exc}"
            report.failure_kind = "sandbox"
            report.duration_s = time.perf_counter() - start
            self.memory.record_step(
                run_id,
                step_index=step.index,
                tool=manifest.name,
                version=manifest.version,
                inputs=step.inputs,
                risk=manifest.risk.value,
                policy_action=decision.action,
                network=decision.network,
                ok=False,
                error=report.error,
            )
            return report

        report.cached = outcome.cached
        report.duration_s = outcome.duration_s
        report.result = outcome.result
        report.error = outcome.error
        report.provenance = list(outcome.provenance)

        # --- verify --------------------------------------------------------
        verification: VerificationResult | None = None
        if outcome.ok and outcome.result is not None:
            verification = self.verifier.verify(
                manifest,
                outcome.result,
                goal=goal,
                run_id=run_id,
                provenance=outcome.provenance,
            )
            report.verification = verification
            if verification.cost_usd:
                budget.charge(usd=verification.cost_usd, llm_calls=1)
        report.ok = bool(outcome.ok and (verification.ok if verification else False))
        if report.ok:
            for item in outcome.provenance:
                if item.origin in {"sandbox_tool", "web_fetch"}:
                    item.verified = True
            self.sandbox.commit_cache(outcome)
        if not report.ok:
            if (
                verification is not None
                and verification.breaker is not None
                and not verification.breaker.ok
            ):
                report.failure_kind = "breaker"
                report.error = report.error or verification.breaker.reason
            elif verification is not None and not verification.ok:
                report.failure_kind = "verification"
                report.error = report.error or verification.reason
            elif not report.error:
                report.failure_kind = "tool"
                report.error = "tool reported failure"

        self.memory.record_step(
            run_id,
            step_index=step.index,
            tool=manifest.name,
            version=manifest.version,
            inputs=step.inputs,
            risk=manifest.risk.value,
            policy_action=decision.action,
            network=decision.network,
            ok=report.ok,
            cached=report.cached,
            duration_s=report.duration_s,
            cost_usd=report.cost_usd,
            verified=bool(verification and verification.ok),
            error=report.error or "",
        )
        if report.duration_s:
            # Wall-clock is not money; only the timeout ceiling matters here.
            budget.check()
        return report

    # ------------------------------------------------------------------- answer
    def _answer(self, result: AgentResult, goal: str, budget: Budget) -> tuple[str, AnswerSource]:
        deterministic = self._compose_answer(result, partial=False)
        if not self.settings.enable_synthesis or self.llm.offline:
            return deterministic, "tools"
        response = self.llm.complete(
            "synth",
            [
                {
                    "role": "system",
                    "content": (
                        "Write the final answer for the user from the tool results. "
                        "Use ONLY the given facts. Cite source URLs inline. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Goal: {goal}\n\nTool results:\n{deterministic[:4000]}",
                },
            ],
            max_tokens=400,
        )
        budget.charge(usd=response.cost_usd, llm_calls=0 if response.cached else 1)
        budget.check()
        result.notes.append(
            f"synthesis via {response.model} (${response.cost_usd:.6f}{', cached' if response.cached else ''})"
        )
        return (response.text or deterministic), "synthesis"

    def _compose_answer(self, result: AgentResult, *, partial: bool) -> str:
        """Deterministic answer composition: no model, no spend, no invention."""
        blocks: list[str] = []
        for report in result.steps:
            if not report.result:
                continue
            payload = {k: v for k, v in report.result.items() if not k.startswith("_")}
            if "summary" in payload:
                header = f"[{report.tool}]"
                body = str(payload["summary"])
                sources = payload.get("sources") or []
                confidence = payload.get("confidence")
                line = f"{header} {body}"
                if sources:
                    line += "\nSources: " + ", ".join(str(s) for s in sources)
                if isinstance(confidence, (int, float)):
                    line += f"\nConfidence: {confidence}"
                blocks.append(line)
            elif "result" in payload and len(payload) <= 3:
                expr = payload.get("expression", "")
                blocks.append(f"[{report.tool}] {expr} = {payload['result']}".strip())
            else:
                blocks.append(f"[{report.tool}] {json.dumps(payload, default=str)[:1200]}")

        if not blocks:
            if result.refused:
                refusal = result.refused[-1]
                return (
                    f"I did not run {refusal.get('tool')} because policy refused it: "
                    f"{refusal.get('reason')}. Nothing was executed."
                )
            if partial:
                return "No tool produced a usable result for this goal."
            return "No suitable tool was available for this goal; nothing was executed."

        text = "\n\n".join(blocks)
        if result.refused:
            text += "\n\nRefused steps: " + "; ".join(
                f"{r.get('tool')} ({r.get('reason')})" for r in result.refused
            )
        return text

    def _finish_without_tools(
        self, result: AgentResult, run_id: str, started: float, goal: str
    ) -> AgentResult:
        """Depth-0 goals with no matching tool: answer without spending anything."""
        recall = (
            self.memory.recall(goal, self.registry.fingerprint, min_overlap=0.8) if goal else None
        )
        if recall:
            result.answer = recall["answer"]
            result.answer_source = "memory"
            result.recalled_from = recall["run_id"]
        else:
            result.answer = (
                "This goal needs no tool call and no model call in Phase 1 "
                f"(route depth {result.route.plan_depth if result.route else 0}). "
                "Add a tool manifest that matches the goal to extend coverage."
            )
            result.answer_source = "direct"
        result.status = "no_plan"
        result.ok = False  # honest: nothing was executed or verified
        result.notes.append("no executable plan; answered deterministically at zero cost")
        return self._finalize(result, run_id=run_id, started=started)

    # ----------------------------------------------------------------- finalize
    def _finalize(
        self,
        result: AgentResult,
        *,
        run_id: str | None = None,
        started: float | None = None,
        recalled: bool = False,
    ) -> AgentResult:
        result.latency_s = round(time.perf_counter() - (started or time.perf_counter()), 4)
        result.cache_hits = self.cache.hits
        result.cache_misses = self.cache.misses
        result.llm_calls = self.llm.usage.calls
        result.llm_cache_hits = self.llm.usage.cache_hits
        # Single source of truth for spend: the budget is charged at every
        # billable point (router, planner, judge, synthesis), so it never
        # double-counts step cost the way summing LLM usage would.
        result.cost_usd = round((result.budget.spent_usd if result.budget else 0.0), 8)
        result.cost_basis = self._cost_basis()

        if run_id and not recalled and result.status == "ungrounded_rejection":
            self.memory.discard_run(run_id)
        elif run_id and not recalled:
            self.memory.finish_run(
                RunRecord(
                    run_id=run_id,
                    goal=result.goal,
                    status=result.status,
                    success=result.ok and result.verified,
                    verified=result.verified,
                    difficulty=result.route.difficulty if result.route else "",
                    plan_depth=result.route.plan_depth if result.route else 0,
                    steps_planned=len(result.plan.steps) if result.plan else 0,
                    steps_executed=sum(1 for s in result.steps if s.ok),
                    cost_usd=result.cost_usd,
                    latency_s=result.latency_s,
                    cache_hits=result.cache_hits,
                    cache_misses=result.cache_misses,
                    llm_calls=result.llm_calls,
                    answer=result.answer,
                    error=next((s.error for s in result.steps if s.error), ""),
                    chain=list(result.chain),
                )
            )
        return result

    def _cost_basis(self) -> str:
        """How to read ``cost_usd``: real spend, a priced simulation, or none."""
        usage = self.llm.usage
        if usage.calls == 0:
            return "no_llm"
        if usage.cache_hits == usage.calls:
            return "cache_hit"  # every call was served from the LLM cache: $0
        if usage.stub_calls:
            priced = self.settings.cost_simulate or self.settings.llm_mode == "stub"
            return "simulated" if priced else "offline_stub"
        return "actual"
