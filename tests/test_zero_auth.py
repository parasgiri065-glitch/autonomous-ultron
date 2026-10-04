from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultron.adapters.zero_auth import AdapterResponse, PollinationsAdapter, ZeroAuthUnavailable
from ultron.config import REPO_ROOT, load_settings
from ultron.registry import Registry
from ultron.router import FreeWaterfallRouter, Router


def _settings(tmp_path: Path, **overrides):
    return load_settings(
        state_dir=tmp_path,
        cache_path=tmp_path / "cache.db",
        memory_path=tmp_path / "memory.db",
        approvals_file=tmp_path / "approvals.json",
        audit_log=tmp_path / "audit.jsonl",
        llm_mode="offline",
        repo_root=REPO_ROOT,
        **overrides,
    )


def test_pollinations_mock_returns_tagged_completion_and_breaker_result():
    seen = {}

    def transport(request, timeout):
        seen["timeout"] = timeout
        seen["payload"] = json.loads(request.data)
        return {"choices": [{"message": {"content": '{"difficulty":"easy"}'}}]}

    response = PollinationsAdapter(transport=transport).complete(
        [{"role": "user", "content": "route this"}]
    )
    assert response is not None
    assert response.origin == "llm_generated"
    assert response.provenance[0].origin == "llm_generated"
    assert response.breaker is not None
    assert seen["timeout"] == 15.0
    assert seen["payload"]["stream"] is False


def test_pollinations_timeout_is_a_graceful_provider_failure():
    def timeout(_request, _timeout):
        raise TimeoutError("offline fixture")

    with pytest.raises(ZeroAuthUnavailable):
        PollinationsAdapter(transport=timeout).complete([])


def test_waterfall_429_fails_over_to_next_provider():
    class RateLimited:
        provider = "pollinations"

        def complete(self, _messages):
            raise ZeroAuthUnavailable("rate limited", status=429)

    class Working:
        provider = "gemini"

        def complete(self, _messages):
            return AdapterResponse(
                text='{"difficulty":"easy","plan_depth":1,"needs_tools":true}',
                provider="gemini",
                model="gemini-free",
            )

    waterfall = FreeWaterfallRouter(
        _settings(Path("/tmp/ultron-waterfall-test")), providers=[RateLimited(), Working()]
    )
    result = waterfall.complete([{"role": "user", "content": "x"}], goal="x")
    assert result is not None
    assert result.provider == "gemini"
    assert waterfall.failures[0].status == 429


def test_router_zero_key_default_never_calls_network(tmp_path, monkeypatch):
    settings = _settings(tmp_path, allow_zero_auth=False)
    registry = Registry(settings).load(strict=True)
    router = Router(registry, settings=settings)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("zero-auth network must be disabled by default")

    monkeypatch.setattr(PollinationsAdapter, "complete", forbidden)
    decision = router.route("an ambiguous goal about an unfamiliar topic")
    assert decision.cost_usd == 0.0
    assert decision.source in {"rules", "cache"}
