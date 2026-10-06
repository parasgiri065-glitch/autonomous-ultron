"""Cheap router: classify a goal's difficulty and pick a planning depth.

Cost ladder — spend the least that can possibly work:

  1. **Cache**  identical goal + identical registry -> $0, 0 ms (SQLite lookup).
  2. **Rules**  deterministic regexes resolve the overwhelming majority of goals
     ($0, ~microseconds). Single-tool lookups, arithmetic and URL fetches are
     *classified without any model call at all*.
  3. **Model**  only ambiguous goals escalate to a small/cheap model
     (``ULTRON_ROUTER_MODEL``, JSON mode, ``max_tokens`` capped). The reply is
     cached forever and validated; on any doubt we fall back to the rules answer.

The router never chooses a tool by name — it only decides *how much thinking*
the goal deserves. Tool choice belongs to the planner/registry.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .adapters.zero_auth import AdapterResponse, PollinationsAdapter, ZeroAuthUnavailable
from .cache import Cache, make_key
from .config import Settings, get_settings
from .ledger import FailureLedger
from .llm import LLMClient, extract_json
from .provenance import ProvenanceEnvelope
from .registry import Registry

Difficulty = Literal["trivial", "easy", "medium", "hard"]

DIFFICULTY_DEPTH: dict[str, int] = {"trivial": 0, "easy": 1, "medium": 2, "hard": 3}

URL_RE = re.compile(r"https?://[^\s\"')>]+", re.I)
ARITHMETIC_VERB_RE = re.compile(
    r"^\s*(?:please\s+)?(?:calc(?:ulate)?|compute|evaluate|solve|what\s+is|what's|whats)\b[:\s]*",
    re.I,
)
PURE_EXPRESSION_RE = re.compile(r"^[\s\d+\-*/%^().,]+$")


def looks_arithmetic(text: str) -> bool:
    """True when the goal is a bare expression, optionally behind a verb.

    ``"calculate 12*(3+4)"``, ``"what is 18 % 5"`` and ``"12*(3+4)"`` are all
    arithmetic; ``"calculate the cost of living"`` is not (it has no pure
    expression left after the verb is stripped).
    """
    stripped = ARITHMETIC_VERB_RE.sub("", text, count=1).strip().rstrip("?=. ").strip()
    if not stripped or not any(ch.isdigit() for ch in stripped):
        return False
    return bool(PURE_EXPRESSION_RE.match(stripped))


HARD_MARKERS = (
    "then",
    "after that",
    "pipeline",
    "workflow",
    "multi-step",
    "all of",
    "compare and contrast",
    "and verify",
    "cross-check",
    "reconcile",
)
MEDIUM_MARKERS = (
    " and ",
    "compare",
    "versus",
    " vs ",
    "summarize",
    "research",
    "analyze",
    "evaluate",
    "why",
    "how does",
    "explain",
)
TRIVIAL_MARKERS = ("hello", "hi ", "ping", "what is", "define ", "who is")

SYSTEM_PROMPT = (
    "You are a router for a cost-optimized tool-using agent. "
    "Classify the user's goal so the harness can spend as little as possible.\n"
    "Reply with ONLY compact JSON: "
    '{"difficulty":"trivial|easy|medium|hard","plan_depth":0-3,"needs_tools":true|false,'
    '"reason":"<12 words max>"}.\n'
    "trivial = no tools needed or one deterministic step. easy = one tool call. "
    "medium = 2 tool calls or one tool plus synthesis. hard = 3+ dependent steps."
)


@dataclass(slots=True)
class RouteDecision:
    difficulty: Difficulty
    plan_depth: int
    needs_tools: bool
    reason: str
    source: Literal["cache", "rules", "llm", "waterfall", "fallback"] = "rules"
    provider: str = ""
    cached: bool = False
    cost_usd: float = 0.0
    model: str = ""
    suggested_tools: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def cheap_path(self) -> bool:
        """True when this goal can be served without any LLM spend at all."""
        return self.source in {"cache", "rules"} and self.plan_depth <= 1

    @property
    def meta_loop_eligible(self) -> bool:
        """Whether an unresolved rules-first plan may enter the JIT fall-through."""
        return bool(self.needs_tools)

    def as_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "plan_depth": self.plan_depth,
            "needs_tools": self.needs_tools,
            "reason": self.reason,
            "source": self.source,
            "cached": self.cached,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "provider": self.provider,
            "suggested_tools": self.suggested_tools,
            "signals": self.signals,
        }


class Router:
    """Difficulty classifier with a rules-first, model-second strategy."""

    def __init__(
        self,
        registry: Registry,
        *,
        cache: Cache | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        waterfall: FreeWaterfallRouter | None = None,
        providers: list[Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry
        self.cache = cache or Cache(self.settings)
        self.llm = llm or LLMClient(self.cache, self.settings)
        self.waterfall = waterfall or FreeWaterfallRouter(
            self.settings, providers=providers, ledger=FailureLedger(self.settings.state_dir)
        )

    def waterfall_complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        goal: str = "",
        kind: str = "generic",
    ) -> AdapterResponse | None:
        """Try enabled free providers; ``None`` is the deterministic fallback."""
        return self.waterfall.complete(messages, goal=goal, kind=kind)

    def route(self, goal: str) -> RouteDecision:
        goal = (goal or "").strip()
        if not goal:
            return RouteDecision("trivial", 0, False, "empty goal", source="rules")

        cache_key = make_key("route", goal.lower(), self.registry.fingerprint)
        entry = self.cache.get("router", cache_key)
        if entry is not None:
            decision = RouteDecision(**entry.value)
            decision.cached = True
            decision.source = "cache"
            decision.cost_usd = 0.0
            return decision

        decision, confident = self._rules(goal, cache_key)
        if confident:
            self._store(cache_key, decision)
            return decision

        decision = self._escalate(goal, decision, cache_key)
        self._store(cache_key, decision)
        return decision

    # -------------------------------------------------------------------- rules
    def _rules(self, goal: str, cache_key: str) -> tuple[RouteDecision, bool]:
        """Deterministic classification. ``confident=False`` asks the model."""
        lowered = f" {goal.lower().strip()} "
        words = len(goal.split())
        signals: dict[str, Any] = {"words": words}

        urls = URL_RE.findall(goal)
        signals["urls"] = len(urls)
        signals["arithmetic"] = looks_arithmetic(goal)

        hard_hits = [m for m in HARD_MARKERS if m in lowered]
        medium_hits = [m for m in MEDIUM_MARKERS if m in lowered]
        signals["hard_markers"] = hard_hits
        signals["medium_markers"] = medium_hits

        candidates = self.registry.search(goal, limit=5)
        suggested = [m.name for m in candidates]
        signals["registry_matches"] = suggested

        # Hard evidence: long multi-clause goals, or 3+ clauses of dependency.
        if len(hard_hits) >= 2 or words > 45:
            return (
                RouteDecision(
                    "hard",
                    DIFFICULTY_DEPTH["hard"],
                    True,
                    f"multi-step signals: {', '.join(hard_hits) or f'{words} words'}",
                    signals=signals,
                    suggested_tools=suggested,
                ),
                True,
            )

        # One deterministic tool, no ambiguity -> never touch a model.
        if signals["arithmetic"]:
            return (
                RouteDecision(
                    "trivial", 0, True, "pure arithmetic: deterministic tool", signals=signals
                ),
                True,
            )
        if urls and words <= 25:
            return (
                RouteDecision(
                    "easy", 1, True, "single URL fetch", signals=signals, suggested_tools=suggested
                ),
                True,
            )
        if hard_hits or len(medium_hits) >= 2:
            return (
                RouteDecision(
                    "medium",
                    DIFFICULTY_DEPTH["medium"],
                    True,
                    f"compound goal ({', '.join(medium_hits[:2]) or hard_hits[0]})",
                    signals=signals,
                    suggested_tools=suggested,
                ),
                True,
            )
        if any(lowered.strip().startswith(m.strip()) for m in TRIVIAL_MARKERS) and words <= 8:
            return (
                RouteDecision("trivial", 0, False, "chitchat/simple definition", signals=signals),
                True,
            )
        if len(candidates) == 1 and words <= 20:
            return (
                RouteDecision(
                    "easy",
                    1,
                    True,
                    f"single registry match: {suggested[0]}",
                    signals=signals,
                    suggested_tools=suggested,
                ),
                True,
            )

        # Ambiguous -> escalate (fallback value is used if the model is offline).
        return (
            RouteDecision(
                "easy",
                1,
                True,
                "ambiguous: escalating to cheap classifier",
                source="fallback",
                signals=signals,
                suggested_tools=suggested,
            ),
            False,
        )

    # ------------------------------------------------------------------- model
    def _escalate(self, goal: str, fallback: RouteDecision, cache_key: str) -> RouteDecision:
        """Use the free waterfall before any paid/keyed LLM call."""
        if self.waterfall.enabled:
            registry_hint = ", ".join(
                f"{m.name}({m.description[:40]})" for m in self.registry.latest()[:8]
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Available tools: {registry_hint or 'none'}\\nGoal: {goal[:1200]}",
                },
            ]
            response = self.waterfall.complete(messages, goal=goal, kind="route")
            if response is not None:
                parsed = response.json or extract_json(response.text)
                if parsed:
                    decision = self._decision_from_provider(parsed, fallback, response)
                    if decision is not None:
                        return decision
            # A provider outage or malformed result is fail-open to rules. Do
            # not silently fall through to a paid provider after a free miss.
            fallback.reason = f"{fallback.reason} (waterfall unavailable: rules answer kept)"
            fallback.source = "rules"
            return fallback

        if self.llm.offline:
            fallback.reason = f"{fallback.reason} (offline: rules answer kept)"
            fallback.source = "rules"
            return fallback

        registry_hint = ", ".join(
            f"{m.name}({m.description[:40]})" for m in self.registry.latest()[:8]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Available tools: {registry_hint or 'none'}\nGoal: {goal[:1200]}",
            },
        ]
        try:
            response = self.llm.complete(
                "route",
                messages,
                max_tokens=120,
                use_cache=True,
                response_format_json=True,
            )
        except Exception as exc:
            fallback.reason = f"{fallback.reason} (router error: {type(exc).__name__})"
            fallback.source = "rules"
            return fallback

        if response.cached:
            fallback.cached = True
            fallback.source = "cache"
            fallback.cost_usd = 0.0
            fallback.reason = f"{fallback.reason} (cached classification)"
            return fallback

        data = response.json or {}
        difficulty = str(data.get("difficulty", "")).lower()
        if difficulty not in DIFFICULTY_DEPTH:
            fallback.reason = f"{fallback.reason} (unparseable model reply)"
            fallback.source = "rules"
            fallback.cost_usd = response.cost_usd
            fallback.model = response.model
            return fallback

        try:
            depth = int(data.get("plan_depth", DIFFICULTY_DEPTH[difficulty]))
        except (TypeError, ValueError):
            depth = DIFFICULTY_DEPTH[difficulty]
        depth = max(0, min(3, depth))

        return RouteDecision(
            difficulty=difficulty,  # type: ignore[arg-type]
            plan_depth=depth,
            needs_tools=bool(data.get("needs_tools", True)),
            reason=str(data.get("reason", "model classification"))[:200],
            source="llm",
            cost_usd=response.cost_usd,
            model=response.model,
            suggested_tools=fallback.suggested_tools,
            signals=fallback.signals,
        )

    @staticmethod
    def _decision_from_provider(
        data: dict[str, Any], fallback: RouteDecision, response: AdapterResponse
    ) -> RouteDecision | None:
        difficulty = str(data.get("difficulty", "")).lower()
        if difficulty not in DIFFICULTY_DEPTH:
            return None
        try:
            depth = int(data.get("plan_depth", DIFFICULTY_DEPTH[difficulty]))
        except (TypeError, ValueError):
            depth = DIFFICULTY_DEPTH[difficulty]
        return RouteDecision(
            difficulty=difficulty,  # type: ignore[arg-type]
            plan_depth=max(0, min(3, depth)),
            needs_tools=bool(data.get("needs_tools", True)),
            reason=str(data.get("reason", "waterfall classification"))[:200],
            source="waterfall",
            cost_usd=0.0,
            model=response.model,
            provider=response.provider,
            suggested_tools=fallback.suggested_tools,
            signals=fallback.signals,
        )

    # ------------------------------------------------------------------ caching
    def _store(self, cache_key: str, decision: RouteDecision) -> None:
        payload = decision.as_dict()
        for volatile in ("cached", "cost_usd", "source"):
            payload.pop(volatile, None)
        payload["reason"] = payload["reason"].split(" (")[0]
        self.cache.set("router", cache_key, payload, ttl_s=self.settings.cache_ttl_llm)


@dataclass(slots=True)
class ProviderFailure:
    provider: str
    reason: str
    status: int | None = None


class KeyPoolAdapter:
    """Small keyed-provider adapter backed by LiteLLM's provider routing."""

    def __init__(self, provider: str, env_name: str, model: str) -> None:
        self.provider = provider
        self.env_name = env_name
        self.model = model

    def complete(
        self, messages: Iterable[dict[str, str]], *, timeout: float = 15.0, **_: Any
    ) -> AdapterResponse:
        if not os.environ.get(self.env_name):
            raise ZeroAuthUnavailable(f"{self.env_name} is not configured")
        try:
            import litellm

            completion = litellm.completion(
                model=self.model,
                messages=[dict(message) for message in messages],
                temperature=0.0,
                max_tokens=512,
                timeout=timeout,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise ZeroAuthUnavailable(f"{self.provider} returned an empty completion")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            raise ZeroAuthUnavailable(
                f"{self.provider} unavailable: {type(exc).__name__}", status=status
            ) from exc
        response = AdapterResponse(text=text, provider=self.provider, model=self.model)
        response.provenance = [
            ProvenanceEnvelope.create(
                "llm_generated",
                f"llm:{self.provider}:{uuid.uuid4().hex[:16]}",
                text,
                metadata={"provider": self.provider, "model": self.model},
            )
        ]
        response.verify_with_breaker()
        return response


class FreeWaterfallRouter:
    """Tiered provider cascade with a deterministic, no-network default."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        providers: list[Any] | None = None,
        ledger: FailureLedger | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.timeout = timeout
        self.ledger = ledger
        self._injected = providers
        self.failures: list[ProviderFailure] = []

    @property
    def enabled(self) -> bool:
        return bool(self._injected is not None or self.providers())

    def providers(self) -> list[Any]:
        if self._injected is not None:
            return list(self._injected)
        providers: list[Any] = []
        if self.settings.allow_zero_auth or os.environ.get("ULTRON_ALLOW_ZERO_AUTH") == "1":
            providers.append(
                PollinationsAdapter(
                    timeout=self.timeout,
                    model=os.environ.get("ULTRON_ZERO_AUTH_MODEL", "llama"),
                )
            )
        keyed = (
            (
                "openrouter",
                "OPENROUTER_API_KEY",
                os.environ.get("ULTRON_OPENROUTER_MODEL", "openrouter/free"),
            ),
            (
                "sambanova",
                "SAMBANOVA_API_KEY",
                os.environ.get("ULTRON_SAMBANOVA_MODEL", "sambanova/Meta-Llama-3.1-8B-Instruct"),
            ),
            (
                "groq",
                "GROQ_API_KEY",
                os.environ.get("ULTRON_GROQ_MODEL", "groq/llama-3.1-8b-instant"),
            ),
            (
                "cerebras",
                "CEREBRAS_API_KEY",
                os.environ.get("ULTRON_CEREBRAS_MODEL", "cerebras/llama3.1-8b"),
            ),
            (
                "gemini",
                "GEMINI_API_KEY",
                os.environ.get("ULTRON_GEMINI_MODEL", "gemini/gemini-2.0-flash"),
            ),
        )
        providers.extend(KeyPoolAdapter(*entry) for entry in keyed if os.environ.get(entry[1]))
        return providers

    def complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        goal: str = "",
        kind: str = "generic",
    ) -> AdapterResponse | None:
        self.failures = []
        providers = self.providers()
        if not providers:
            if self.ledger is not None and goal:
                self.ledger.record_gap(
                    goal,
                    expected_outputs={"answer": "string"},
                    failure_reason="no zero-auth or free-key providers configured",
                )
            return None
        message_list = [dict(message) for message in messages]
        for provider in providers:
            name = getattr(provider, "provider", type(provider).__name__)
            try:
                try:
                    response = provider.complete(message_list, timeout=self.timeout, kind=kind)
                except TypeError:
                    response = provider.complete(message_list)
                if response is not None:
                    return response
            except Exception as exc:
                status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
                self.failures.append(ProviderFailure(name, str(exc), status))
                continue
        if self.ledger is not None and goal:
            self.ledger.record_gap(
                goal,
                expected_outputs={"answer": "string"},
                failure_reason="; ".join(
                    f"{item.provider}: {item.reason}" for item in self.failures
                ),
            )
        return None
