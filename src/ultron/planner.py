"""Planner: turn a goal + route into a concrete, cost-annotated step list.

Phase 1 ships a **deterministic planner**. It maps goals onto registry tools with
regex/keyword rules and a small cost model, which means:

* planning is free (no tokens) for ~all Phase 1 goals,
* plans are perfectly cacheable and reproducible,
* nothing can be "invented" — a step either names a real manifest that exists in
  ``tools/`` or planning fails closed ("unresolved").

The optional LLM planner (``ULTRON_PLANNER_USE_LLM=1``, cheap ``planner_model``)
only ever *reorders or selects* from the manifests we already have; its output is
validated against the registry and against the input schema, and any invalid step
discards the whole model plan in favour of the deterministic one. A hallucinated
tool name can therefore never reach the sandbox.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .cache import Cache, make_key
from .config import Settings, get_settings
from .llm import LLMClient
from .registry import TYPE_MAP, Registry, RiskTier, ToolManifest
from .router import RouteDecision

URL_RE = re.compile(r"https?://[^\s\"')>]+", re.I)
ARITH_RE = re.compile(r"(?:^|\b)(?:calc(?:ulate)?|compute|evaluate|what\s+is)\b[:\s]*(.+)$", re.I)
NUMBER_EXPR_RE = re.compile(r"[-+*/%^().\d\s]{3,}")

PLANNER_SYSTEM = (
    "You are a planner for a cost-optimized agent. Choose the FEWEST steps from the "
    "given tool list. Prefer deterministic, free tools. Never invent tools.\n"
    'Reply with ONLY JSON: {"steps":[{"tool":"<name>","inputs":{...},"why":"<8 words>"}],'
    '"rationale":"<15 words>"}'
)


@dataclass(slots=True)
class PlanStep:
    index: int
    tool: str
    version: str
    inputs: dict[str, Any]
    risk: str
    reason: str
    expected_outputs: dict[str, str] = field(default_factory=dict)
    est_cost_usd: float = 0.0
    deterministic: bool = True
    #: Index of the preceding step whose verified result supplies this step.
    #: Inputs are materialized by Agent immediately before the policy gate.
    chain_input: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "version": self.version,
            "risk": self.risk,
            "reason": self.reason,
            "inputs": self.inputs,
            "est_cost_usd": self.est_cost_usd,
            "deterministic": self.deterministic,
            "chain_input": self.chain_input,
        }


@dataclass(slots=True)
class Plan:
    goal: str
    depth: int
    steps: list[PlanStep]
    rationale: str = ""
    created_by: Literal["deterministic", "llm", "cache", "none"] = "deterministic"
    cached: bool = False
    cost_usd: float = 0.0
    resolved: bool = True
    notes: list[str] = field(default_factory=list)
    #: Ordered capability chain, empty for ordinary single-tool plans.
    chain: list[str] = field(default_factory=list)

    @property
    def est_cost_usd(self) -> float:
        return round(sum(s.est_cost_usd for s in self.steps), 8)

    @property
    def is_empty(self) -> bool:
        return not self.steps

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "depth": self.depth,
            "rationale": self.rationale,
            "created_by": self.created_by,
            "cached": self.cached,
            "cost_usd": self.cost_usd,
            "resolved": self.resolved,
            "notes": self.notes,
            "chain": self.chain,
            "est_cost_usd": self.est_cost_usd,
            "steps": [s.as_dict() for s in self.steps],
        }


class Planner:
    """Deterministic-first planner with an opt-in, validated LLM path."""

    def __init__(
        self,
        registry: Registry,
        *,
        cache: Cache | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        risk_ceiling: RiskTier | str = RiskTier.LOW,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry
        self.cache = cache or Cache(self.settings)
        self.llm = llm or LLMClient(self.cache, self.settings)
        self.use_llm = bool(_flag("ULTRON_PLANNER_USE_LLM"))
        self.risk_ceiling = _risk_value(risk_ceiling)

    # --------------------------------------------------------------- entry point
    def plan(
        self,
        goal: str,
        route: RouteDecision | None = None,
        *,
        start_types: Iterable[str] | None = None,
        goal_types: Iterable[str] | None = None,
        risk_ceiling: RiskTier | str | None = None,
    ) -> Plan:
        depth = route.plan_depth if route else 1
        explicit_start = tuple(sorted(set(start_types or ())))
        explicit_goal = tuple(sorted(set(goal_types or ())))
        ceiling = _risk_value(risk_ceiling) if risk_ceiling is not None else self.risk_ceiling
        inferred_start, inferred_goal = self._infer_capability_types(goal)
        effective_start = explicit_start or tuple(sorted(inferred_start))
        effective_goal = explicit_goal or tuple(sorted(inferred_goal))
        cache_key = make_key(
            "plan",
            "capability-graph-v1",
            goal.strip().lower(),
            depth,
            self.registry.fingerprint,
            effective_start,
            effective_goal,
            ceiling,
        )
        entry = self.cache.get("plan", cache_key)
        if entry is not None:
            payload = dict(entry.value)
            payload["cached"] = True
            payload["created_by"] = "cache"
            payload["cost_usd"] = 0.0
            payload["steps"] = [PlanStep(**s) for s in payload["steps"]]
            payload.setdefault("chain", [])
            return Plan(**payload)

        plan = self._deterministic(
            goal,
            depth,
            start_types=effective_start,
            goal_types=effective_goal,
            risk_ceiling=ceiling,
            explicit_capabilities=bool(explicit_start or explicit_goal),
        )
        if self.use_llm and depth >= 2 and not self.llm.offline and not plan.chain:
            plan = self._maybe_llm_upgrade(goal, plan)

        self.cache.set(
            "plan",
            cache_key,
            {
                "goal": plan.goal,
                "depth": plan.depth,
                "rationale": plan.rationale,
                "created_by": plan.created_by,
                "resolved": plan.resolved,
                "notes": plan.notes,
                "chain": plan.chain,
                "steps": [s.as_dict() for s in plan.steps],
            },
            ttl_s=self.settings.cache_ttl_llm,
        )
        return plan

    # ----------------------------------------------------------- deterministic
    def _deterministic(
        self,
        goal: str,
        depth: int,
        *,
        start_types: Iterable[str] = (),
        goal_types: Iterable[str] = (),
        risk_ceiling: str = "low",
        explicit_capabilities: bool = False,
    ) -> Plan:
        text = (goal or "").strip()
        urls = URL_RE.findall(text)
        notes: list[str] = []

        if depth == 0:
            steps = self._zero_step_plan(text)
            return Plan(
                goal=text,
                depth=0,
                steps=steps,
                rationale="trivial goal: deterministic short-circuit, no model spend",
                notes=notes,
            )

        steps: list[PlanStep] = []

        # 1. Arithmetic -> calc (free, offline, cacheable forever).
        expr = self._extract_expression(text)
        if expr and "calc" in self.registry:
            calc = self.registry.get("calc")
            steps.append(
                self._step(len(steps), calc, {"expression": expr}, "arithmetic extracted from goal")
            )

        # 2. Explicit URL -> http_fetch (medium risk -> policy gate will ask).
        if urls and "http_fetch" in self.registry:
            fetch = self.registry.get("http_fetch")
            steps.append(
                self._step(
                    len(steps), fetch, {"url": urls[0], "max_bytes": 50_000}, "goal names a URL"
                )
            )

        # 3. Otherwise (or in addition) -> best registry match for research.
        if not steps or (depth >= 2 and not urls):
            candidates = self.registry.search(text, limit=3)
            chosen = self._pick(candidates, exclude={s.tool for s in steps})
            if chosen is not None:
                inputs = self._inputs_for(chosen, text)
                if inputs is not None:
                    steps.append(
                        self._step(len(steps), chosen, inputs, f"registry match: {chosen.name}")
                    )
                else:
                    notes.append(f"could not derive inputs for {chosen.name}; skipped")
            elif not steps:
                notes.append("no registry tool matched the goal")

        # A single-tool plan remains exactly as before. Capability composition is
        # only a fallback (or an explicitly requested capability plan), so adding
        # optional semantic metadata cannot perturb Phase 1 routing.
        chain: list[str] = []
        if goal_types and (not steps or explicit_capabilities):
            chain = self.registry.find_chain(
                start_types,
                goal_types,
                risk_ceiling=risk_ceiling,
            )
            if chain:
                steps = self._chain_steps(chain, text)
                notes = [
                    note for note in notes if not note.startswith("could not derive inputs for ")
                ]
                notes.append(f"capability chain selected: {' -> '.join(chain)}")
            elif explicit_capabilities:
                unrestricted = self.registry.find_chain(start_types, goal_types)
                if unrestricted:
                    notes.append(f"capability chain refused: exceeds {risk_ceiling} risk ceiling")
                else:
                    notes.append("no capability chain exists for the requested types")

        resolved = bool(steps)
        if not resolved:
            notes.append("plan is unresolved: the agent will answer without tools")
        return Plan(
            goal=text,
            depth=depth,
            steps=steps,
            rationale=f"deterministic plan from registry ({len(steps)} step(s)), zero planner tokens",
            resolved=resolved,
            notes=notes,
            chain=chain,
        )

    def _zero_step_plan(self, goal: str) -> list[PlanStep]:
        expr = self._extract_expression(goal)
        if expr and "calc" in self.registry:
            return [
                self._step(
                    0,
                    self.registry.get("calc"),
                    {"expression": expr},
                    "trivial arithmetic: one free deterministic tool call",
                )
            ]
        return []

    # ------------------------------------------------------------------ helpers
    def _step(
        self,
        index: int,
        manifest: ToolManifest,
        inputs: dict[str, Any],
        reason: str,
        *,
        chain_input: int | None = None,
    ) -> PlanStep:
        return PlanStep(
            index=index,
            tool=manifest.name,
            version=manifest.version,
            inputs=inputs,
            risk=manifest.risk.value,
            reason=reason,
            expected_outputs=dict(manifest.outputs),
            est_cost_usd=float(manifest.price_estimate_usd),
            deterministic=manifest.deterministic,
            chain_input=chain_input,
        )

    def _chain_steps(self, chain: list[str], goal: str) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for index, name in enumerate(chain):
            manifest = self.registry.get(name)
            inputs = self._inputs_for(manifest, goal) or {}
            steps.append(
                self._step(
                    index,
                    manifest,
                    inputs,
                    f"capability chain step {index + 1}/{len(chain)}",
                    chain_input=index - 1 if index else None,
                )
            )
        return steps

    def _infer_capability_types(self, goal: str) -> tuple[set[str], set[str]]:
        """Infer capability endpoints from declared types without inventing names.

        Exact ``from pdf.bytes to table.csv`` syntax is preferred. For ordinary
        language, match a declared type's meaningful components (``pdf`` and
        ``table``) against the source and target sides of ``to``/``into``/``as``.
        Initial types are biased toward values no registered tool produces; this
        keeps ``pdf.bytes`` from being confused with the intermediate ``pdf.text``.
        If the wording is still ambiguous, return no endpoints and fail closed.
        """
        manifests = self.registry.latest()
        declared_requires = {
            semantic_type for manifest in manifests for semantic_type in manifest.requires
        }
        declared_provides = {
            semantic_type for manifest in manifests for semantic_type in manifest.provides
        }
        declared = declared_requires | declared_provides
        mentioned = set(re.findall(r"[a-z][a-z0-9_-]*\.[a-z][a-z0-9_.-]*", goal.lower()))
        exact = mentioned.intersection(declared)
        direction = re.search(r"\b(?:to|into|as)\b", goal.lower())
        reverse_direction = re.search(r"\bfrom\b", goal.lower())
        if direction:
            source_text, target_text = goal[: direction.start()], goal[direction.end() :]
        elif reverse_direction:
            # "extract a table from a pdf" names the desired output first.
            target_text, source_text = (
                goal[: reverse_direction.start()],
                goal[reverse_direction.end() :],
            )
        else:
            source_text = target_text = goal

        def matches(semantic_type: str, text: str) -> bool:
            parts = [part for part in re.split(r"[._-]+", semantic_type.lower()) if len(part) > 2]
            words = set(re.findall(r"[a-z0-9]+", text.lower()))
            return bool(parts) and any(part in words for part in parts)

        if exact:
            # Preserve explicit endpoints when both sides are named.
            starts = {value for value in exact if value in declared_requires}
            goals = {value for value in exact if value in declared_provides}
            if starts and goals:
                return starts, goals

        starts = {
            value
            for value in declared_requires
            if matches(value, source_text) and value not in declared_provides
        }
        goals = {value for value in declared_provides if matches(value, target_text)}
        if len(starts) == 1 and len(goals) == 1:
            return starts, goals
        return set(), set()

    def _pick(self, candidates: list[ToolManifest], *, exclude: set[str]) -> ToolManifest | None:
        for manifest in candidates:
            if manifest.name in exclude:
                continue
            if manifest.risk.value == "high":
                continue  # never plan a HIGH risk step unattended
            return manifest
        return None

    def _inputs_for(self, manifest: ToolManifest, text: str) -> dict[str, Any] | None:
        """Derive manifest inputs from free text using per-tool conventions."""
        if manifest.name == "web_research":
            query = URL_RE.sub(" ", text)
            query = re.sub(
                r"^\s*(please\s+)?(research|find|look up|summari[sz]e|investigate)\b",
                "",
                query,
                flags=re.I,
            )
            query = query.strip(" .,:;") or text
            return {"query": query[:400], "max_sources": 3}
        if manifest.name == "http_fetch":
            urls = URL_RE.findall(text)
            return {"url": urls[0], "max_bytes": 50_000} if urls else None
        if manifest.name == "calc":
            expr = self._extract_expression(text)
            return {"expression": expr} if expr else None
        # Generic single-string-input tools: pass the goal through verbatim.
        if list(manifest.inputs.values()) == ["string"]:
            key = next(iter(manifest.inputs))
            return {key: text[:1000]}
        return None

    @staticmethod
    def _extract_expression(text: str) -> str | None:
        """Pull an arithmetic expression out of natural language, safely."""
        match = ARITH_RE.search(text)
        candidate = match.group(1) if match else text
        candidate = candidate.strip().rstrip("?=. ").strip()
        # Accept the three spellings of multiplication a human might type.
        candidate = candidate.replace("^", "**").replace("x", "*").replace("\u00d7", "*")
        if not candidate:
            return None
        match = NUMBER_EXPR_RE.match(candidate)
        if not match:
            return None
        if not any(c.isdigit() for c in candidate):
            return None
        if not all(ch in "0123456789+-*/%^(). " for ch in candidate):
            # Words mixed in: keep only the numeric expression prefix.
            stripped = re.match(r"^[-+*/%^().\d\s]+", candidate)
            if not stripped:
                return None
            candidate = stripped.group(0).strip()
        return candidate or None

    # --------------------------------------------------------------------- llm
    def _maybe_llm_upgrade(self, goal: str, deterministic: Plan) -> Plan:
        """Ask the cheap model to improve the plan, then validate it hard."""
        tools = [
            {
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "inputs": m.inputs,
                "risk": m.risk.value,
            }
            for m in self.registry.latest()
        ]
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Tools: {tools}\nGoal: {goal[:1500]}"},
        ]
        try:
            response = self.llm.complete(
                "plan", messages, max_tokens=300, response_format_json=True
            )
        except Exception as exc:
            deterministic.notes.append(f"llm planner failed: {type(exc).__name__}")
            return deterministic

        if response.cached:
            deterministic.notes.append("llm planner result was cached (no spend)")
            return deterministic

        data = response.json or {}
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            deterministic.notes.append(
                "llm planner returned no usable steps; using deterministic plan"
            )
            return deterministic

        validated: list[PlanStep] = []
        for index, raw in enumerate(raw_steps[:4]):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("tool", "")).strip()
            if name not in self.registry:  # hallucination guard
                deterministic.notes.append(f"llm planner proposed unknown tool {name!r}; ignored")
                continue
            manifest = self.registry.get(name)
            if manifest.risk.value == "high":
                deterministic.notes.append(f"llm planner proposed HIGH risk tool {name!r}; ignored")
                continue
            inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
            problem = _validate_inputs(manifest, inputs)
            if problem:
                deterministic.notes.append(f"llm step {name!r} rejected: {problem}")
                continue
            validated.append(
                self._step(index, manifest, inputs, str(raw.get("why", "llm choice"))[:120])
            )

        if not validated:
            deterministic.notes.append(
                "llm planner produced no valid steps; using deterministic plan"
            )
            return deterministic

        return Plan(
            goal=goal,
            depth=deterministic.depth,
            steps=validated,
            rationale=str(data.get("rationale", "llm plan"))[:200],
            created_by="llm",
            cost_usd=response.cost_usd,
            resolved=True,
            notes=[*deterministic.notes, "llm plan validated against registry + input schemas"],
        )


def _validate_inputs(manifest: ToolManifest, inputs: dict[str, Any]) -> str | None:
    """Same contract the policy gate enforces — checked here so bad steps die early."""
    if sorted(inputs) != sorted(manifest.inputs):
        return f"fields {sorted(inputs)} != manifest {sorted(manifest.inputs)}"
    for name, typ in manifest.inputs.items():
        expected = TYPE_MAP[typ]
        value = inputs[name]
        if expected is int and isinstance(value, bool):
            return f"{name} must be int"
        if expected is not object and not isinstance(value, expected):
            return f"{name} must be {typ}, got {type(value).__name__}"
    return None


def _flag(name: str) -> bool:
    import os

    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _risk_value(risk: RiskTier | str) -> str:
    value = risk.value if isinstance(risk, RiskTier) else str(risk).lower()
    if value not in {"low", "medium", "high"}:
        raise ValueError(f"unknown risk ceiling {risk!r}")
    return value
