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
from dataclasses import dataclass, field
from typing import Any, Literal

from .cache import Cache, make_key
from .config import Settings, get_settings
from .llm import LLMClient
from .registry import TYPE_MAP, Registry, ToolManifest
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
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry
        self.cache = cache or Cache(self.settings)
        self.llm = llm or LLMClient(self.cache, self.settings)
        self.use_llm = bool(_flag("ULTRON_PLANNER_USE_LLM"))

    # --------------------------------------------------------------- entry point
    def plan(self, goal: str, route: RouteDecision | None = None) -> Plan:
        depth = route.plan_depth if route else 1
        cache_key = make_key("plan", goal.strip().lower(), depth, self.registry.fingerprint)
        entry = self.cache.get("plan", cache_key)
        if entry is not None:
            payload = dict(entry.value)
            payload["cached"] = True
            payload["created_by"] = "cache"
            payload["cost_usd"] = 0.0
            payload["steps"] = [PlanStep(**s) for s in payload["steps"]]
            return Plan(**payload)

        plan = self._deterministic(goal, depth)
        if self.use_llm and depth >= 2 and not self.llm.offline:
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
                "steps": [s.as_dict() for s in plan.steps],
            },
            ttl_s=self.settings.cache_ttl_llm,
        )
        return plan

    # ----------------------------------------------------------- deterministic
    def _deterministic(self, goal: str, depth: int) -> Plan:
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
        self, index: int, manifest: ToolManifest, inputs: dict[str, Any], reason: str
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
        )

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
