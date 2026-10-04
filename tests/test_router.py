from __future__ import annotations

from ultron.adapters.zero_auth import AdapterResponse, ZeroAuthUnavailable
from ultron.config import REPO_ROOT, load_settings
from ultron.router import FreeWaterfallRouter


def test_free_waterfall_is_empty_without_flags_or_keys(tmp_path, monkeypatch):
    for key in (
        "ULTRON_ALLOW_ZERO_AUTH",
        "OPENROUTER_API_KEY",
        "SAMBANOVA_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(
        state_dir=tmp_path,
        llm_mode="offline",
        repo_root=REPO_ROOT,
    )
    waterfall = FreeWaterfallRouter(settings)
    assert waterfall.providers() == []
    assert waterfall.complete([{"role": "user", "content": "x"}]) is None


def test_free_waterfall_continues_after_timeout_then_returns_success(tmp_path):
    class Down:
        provider = "pollinations"

        def complete(self, _messages):
            raise ZeroAuthUnavailable("timeout")

    class Up:
        provider = "openrouter"

        def complete(self, _messages):
            return AdapterResponse("ok", "openrouter", "free-model")

    waterfall = FreeWaterfallRouter(
        load_settings(state_dir=tmp_path, llm_mode="offline", repo_root=REPO_ROOT),
        providers=[Down(), Up()],
    )
    result = waterfall.complete([{"role": "user", "content": "x"}])
    assert result is not None and result.provider == "openrouter"
    assert waterfall.failures[0].provider == "pollinations"
