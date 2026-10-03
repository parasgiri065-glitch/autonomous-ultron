"""Thin, cache-first LLM client over LiteLLM.

Cost rules enforced here (not in callers):
  1. Every call is content-addressed and cached -> repeated prompts cost $0.
  2. ``offline`` mode returns a deterministic stub with ``cost_usd == 0`` so the
     whole harness (tests, CI eval) runs with no keys and no network.
  3. Temperature defaults to 0 and ``max_tokens`` is capped: determinism makes
     the cache actually hit.
  4. Cost comes from LiteLLM's own accounting where available, else a small
     built-in price table. ``cost_simulate`` prices stub calls with the real
     table so CI cost gates are not vacuous (reports say ``cost_basis``).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from .cache import Cache, make_key
from .config import Settings, get_settings
from .errors import LLMUnavailable

LLMKind = Literal["route", "plan", "judge", "synth", "generic"]

#: USD per 1M tokens (input, output). Conservative fallbacks live in DEFAULT_PRICE.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-haiku": (0.25, 1.25),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}
DEFAULT_PRICE = (0.50, 1.50)

PROVIDER_KEY_ENVS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ULTRON_LLM_API_KEY",
)


def providers_configured() -> bool:
    """True when at least one provider credential (or a local base URL) exists."""
    if any(os.environ.get(k) for k in PROVIDER_KEY_ENVS):
        return True
    return bool(os.environ.get("ULTRON_LLM_API_BASE"))


def price_for(model: str) -> tuple[float, float]:
    bare = model.split("/")[-1].lower()
    for name, price in MODEL_PRICES.items():
        if bare.startswith(name):
            return price
    return DEFAULT_PRICE


def estimate_tokens(text: str) -> int:
    """Cheap heuristic: ~4 chars/token. Only used for simulated accounting."""
    return max(1, len(text) // 4)


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    kind: LLMKind
    cost_usd: float
    latency_s: float
    cached: bool
    stub: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_basis: str = "actual"

    @property
    def json(self) -> dict[str, Any] | None:
        return extract_json(self.text)


@dataclass(slots=True)
class LLMUsage:
    """Aggregate accounting for one agent run."""

    calls: int = 0
    cache_hits: int = 0
    stub_calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_kind: dict[str, float] = field(default_factory=dict)

    def add(self, resp: LLMResponse) -> None:
        self.calls += 1
        self.cache_hits += int(resp.cached)
        self.stub_calls += int(resp.stub)
        self.cost_usd += resp.cost_usd
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.by_kind[resp.kind] = round(self.by_kind.get(resp.kind, 0.0) + resp.cost_usd, 8)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "stub_calls": self.stub_calls,
            "cost_usd": round(self.cost_usd, 8),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "by_kind": self.by_kind,
        }


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an LLM reply (handles ```json fences)."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        cleaned = cleaned.removeprefix("json").strip()
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


class LLMClient:
    """Cache-first LiteLLM wrapper. One instance per run (shares usage totals)."""

    def __init__(
        self,
        cache: Cache | None = None,
        settings: Settings | None = None,
        *,
        mode: Literal["auto", "live", "offline", "stub"] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or Cache(self.settings)
        self.mode = mode or self.settings.llm_mode
        self.usage = LLMUsage()

    # ------------------------------------------------------------------ state
    def resolve_mode(self) -> Literal["live", "offline", "stub"]:
        """``offline`` never calls a model; ``stub`` runs every call site but
        returns deterministic canned answers priced with the real price table
        (so CI can exercise cost accounting without a provider key)."""
        if self.mode == "offline":
            return "offline"
        if self.mode == "stub":
            return "stub"
        if self.mode == "live":
            if not providers_configured():
                raise LLMUnavailable(
                    "ULTRON_LLM_MODE=live but no provider credentials are set. "
                    "Set a provider API key or use ULTRON_LLM_MODE=offline."
                )
            return "live"
        return "live" if providers_configured() else "offline"

    @property
    def offline(self) -> bool:
        """True only in true offline mode. Stub mode deliberately reports False
        so call sites that guard on ``offline`` (router escalation, judge,
        synthesis) still execute and get priced."""
        return self.resolve_mode() == "offline"

    # ------------------------------------------------------------------- calls
    def complete(
        self,
        kind: LLMKind,
        messages: Iterable[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        use_cache: bool = True,
        response_format_json: bool = False,
    ) -> LLMResponse:
        msgs = [dict(m) for m in messages]
        model = model or self._default_model(kind)
        mode = self.resolve_mode()

        cache_key = make_key("llm", kind, model, msgs, temperature, max_tokens, mode)
        if use_cache:
            entry = self.cache.get("llm", cache_key)
            if entry is not None:
                value = dict(entry.value)
                value.update(cached=True, cost_usd=0.0, cost_basis="cache_hit", latency_s=0.0)
                resp = LLMResponse(**value)
                self.usage.add(resp)
                return resp
        else:
            self.cache._miss("llm", record=False)

        start = time.perf_counter()
        if mode in {"offline", "stub"}:
            resp = self._stub(kind, msgs, model, priced=(mode == "stub"))
        else:
            resp = self._live(kind, msgs, model, max_tokens, temperature, response_format_json)
        resp.latency_s = round(time.perf_counter() - start, 4)

        if use_cache:
            self.cache.set(
                "llm",
                cache_key,
                {
                    "text": resp.text,
                    "model": resp.model,
                    "kind": resp.kind,
                    "stub": resp.stub,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "cost_usd": resp.cost_usd,
                    "cost_basis": resp.cost_basis,
                },
                ttl_s=self.settings.cache_ttl_llm,
            )
        self.usage.add(resp)
        return resp

    def _default_model(self, kind: LLMKind) -> str:
        return {
            "route": self.settings.router_model,
            "plan": self.settings.planner_model,
            "judge": self.settings.judge_model,
            "synth": self.settings.synth_model,
            "generic": self.settings.router_model,
        }[kind]

    # -------------------------------------------------------------------- live
    def _live(
        self,
        kind: LLMKind,
        msgs: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        response_format_json: bool,
    ) -> LLMResponse:
        import litellm  # imported lazily: tests never need the network

        litellm.drop_params = True
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.settings.llm_timeout_s,
        }
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.llm_api_base:
            kwargs["api_base"] = self.settings.llm_api_base
            kwargs["api_key"] = os.environ.get("ULTRON_LLM_API_KEY", "not-needed")

        completion = litellm.completion(**kwargs)
        text = (completion.choices[0].message.content or "").strip()
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = self._actual_cost(completion, model, text, prompt_tokens, completion_tokens)
        return LLMResponse(
            text=text,
            model=model,
            kind=kind,
            cost_usd=cost,
            latency_s=0.0,
            cached=False,
            stub=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_basis="actual",
        )

    @staticmethod
    def _actual_cost(
        completion: Any, model: str, text: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        try:  # preferred: LiteLLM knows the real prices
            import litellm

            cost = litellm.completion_cost(completion_response=completion)
            if cost:
                return round(float(cost), 8)
        except Exception:
            pass
        price_in, price_out = price_for(model)
        pt = prompt_tokens or estimate_tokens("".join(m.get("content", "") for m in []))
        ct = completion_tokens or estimate_tokens(text)
        return round((pt * price_in + ct * price_out) / 1_000_000, 8)

    # ------------------------------------------------------------------ stubs
    def _stub(
        self, kind: LLMKind, msgs: list[dict[str, str]], model: str, *, priced: bool = False
    ) -> LLMResponse:
        """Deterministic offline behaviour. Never invents tools or facts."""
        text = self._stub_text(kind, msgs)
        prompt_chars = sum(len(m.get("content", "")) for m in msgs)
        prompt_tokens = estimate_tokens("x" * prompt_chars)
        completion_tokens = estimate_tokens(text)
        cost = 0.0
        basis = "offline_stub"
        if priced or self.settings.cost_simulate:
            price_in, price_out = price_for(model)
            cost = round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 8)
            basis = "simulated"
        return LLMResponse(
            text=text,
            model=model,
            kind=kind,
            cost_usd=cost,
            latency_s=0.0,
            cached=False,
            stub=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_basis=basis,
        )

    @staticmethod
    def _stub_text(kind: LLMKind, msgs: list[dict[str, str]]) -> str:
        user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        if kind == "route":
            lowered = user.lower()
            hard_markers = (" then ", " and ", "compare", "step", "pipeline", "multi", "all of")
            depth = 2 if sum(m in lowered for m in hard_markers) >= 2 else 1
            return json.dumps({"difficulty": "medium", "plan_depth": depth, "needs_tools": True})
        if kind == "plan":
            # No tool invention offline: the planner falls back to its deterministic
            # shortlist when this returns an empty step list.
            return json.dumps(
                {"steps": [], "rationale": "offline stub: defer to deterministic planner"}
            )
        if kind == "judge":
            return json.dumps({"grounded": True, "issues": [], "confidence": 0.5})
        return json.dumps({"note": "offline stub", "ok": True})
