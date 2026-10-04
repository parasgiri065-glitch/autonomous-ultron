from __future__ import annotations

from pathlib import Path

from ultron.agent import AgentResult, Budget
from ultron.config import REPO_ROOT, load_settings
from ultron.interfaces.telegram import TelegramBotClient, TelegramCockpit, TelegramPrompter
from ultron.ledger import FailureLedger
from ultron.policy import ApprovalPrompt


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path / "state",
        tools_dir=REPO_ROOT / "tools",
        cache_path=tmp_path / "state" / "cache.db",
        memory_path=tmp_path / "state" / "memory.db",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        llm_mode="offline",
        telegram_bot_token="test-token",
        telegram_allowed_user_ids=[42],
        repo_root=REPO_ROOT,
    )


def test_telegram_drops_unauthorized_users(caplog, tmp_path):
    sent = []

    def transport(method, _url, kwargs):
        sent.append((method, kwargs))
        return {"ok": True, "result": {}}

    client = TelegramBotClient("token", [42], transport=transport, settings=_settings(tmp_path))
    cockpit = TelegramCockpit(client, settings=client.settings)
    assert (
        cockpit.handle_update(
            {"update_id": 1, "message": {"from": {"id": 99}, "chat": {"id": 99}, "text": "/status"}}
        )
        is None
    )
    assert sent == []
    assert "unauthorized" in caplog.text


def test_text_message_runs_agent_and_sends_status_and_final(tmp_path):
    sent: list[str] = []

    def transport(method, _url, kwargs):
        if method == "sendMessage":
            sent.append(kwargs["json"]["text"])
        return {"ok": True, "result": {}}

    class FakeAgent:
        def run(self, goal):
            return AgentResult(
                run_id="r-test",
                goal=goal,
                status="ok",
                ok=True,
                answer="grounded answer",
                budget=Budget(),
            )

    settings = _settings(tmp_path)
    client = TelegramBotClient("token", [42], transport=transport, settings=settings)
    cockpit = TelegramCockpit(client, settings=settings, agent_factory=lambda _chat: FakeAgent())
    result = cockpit.handle_update(
        {
            "update_id": 2,
            "message": {"from": {"id": 42}, "chat": {"id": 7}, "text": "calculate 2+2"},
        }
    )
    assert isinstance(result, AgentResult)
    assert sent == ["Planning…", "grounded answer"]


def test_telegram_prompter_sends_keyboard_and_accepts_callback(tmp_path):
    sent_payloads = []
    callback_data = {"value": ""}

    def transport(method, _url, kwargs):
        if method == "sendMessage":
            payload = kwargs["json"]
            sent_payloads.append(payload)
            callback_data["value"] = payload["reply_markup"]["inline_keyboard"][0][0][
                "callback_data"
            ]
            return {"ok": True, "result": {"message_id": 5}}
        if method == "getUpdates":
            return {
                "ok": True,
                "result": [
                    {
                        "update_id": 9,
                        "callback_query": {
                            "id": "callback-1",
                            "from": {"id": 42},
                            "data": callback_data["value"],
                        },
                    }
                ],
            }
        return {"ok": True, "result": {}}

    settings = _settings(tmp_path)
    client = TelegramBotClient("token", [42], transport=transport, settings=settings)
    prompter = TelegramPrompter(client, 42, timeout_s=1, poll_timeout_s=0)
    prompt = ApprovalPrompt(
        tool="http_fetch",
        version="1.0.0",
        risk="medium",
        reason="network access",
        run_id="r-1",
        goal="fetch a page",
        content_hash="abc123",
        input_digest="def456",
        inputs_preview={"url": "https://example.test"},
        network="https",
        permissions=["network:https"],
    )
    assert prompter(prompt) is True
    assert "inline_keyboard" in sent_payloads[0]["reply_markup"]
    assert "✅ Approve" in sent_payloads[0]["reply_markup"]["inline_keyboard"][0][0]["text"]


def test_gaps_and_status_commands_return_expected_payloads(tmp_path):
    sent: list[str] = []

    def transport(method, _url, kwargs):
        if method == "sendMessage":
            sent.append(kwargs["json"]["text"])
        return {"ok": True, "result": {}}

    settings = _settings(tmp_path)
    FailureLedger(settings.state_dir).record_gap("reverse a string")
    client = TelegramBotClient("token", [42], transport=transport, settings=settings)
    cockpit = TelegramCockpit(client, settings=settings)
    gaps = cockpit.handle_command("/gaps", 42)
    status = cockpit.handle_command("/status", 42)
    assert gaps["command"] == "/gaps"
    assert "reverse a string" in gaps["text"]
    assert status["command"] == "/status"
    assert "Capability graph nodes" in status["text"]
    assert status["data"]["tools"]
    assert len(sent) == 2
