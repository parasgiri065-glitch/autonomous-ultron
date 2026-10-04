"""Offline Forge Engine and failure-ledger coverage."""

from __future__ import annotations

import json
from pathlib import Path

from ultron.config import REPO_ROOT, load_settings
from ultron.forge import ForgeEngine
from ultron.ledger import FailureLedger
from ultron.main import main
from ultron.registry import Registry

REVERSE_CODE = """
from tools._io import main_guard, require

def run(payload):
    value = require(payload, "text", str)
    return {"reversed": value[::-1]}

if __name__ == "__main__":
    main_guard(run)
"""


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path / "state",
        tools_dir=tmp_path / "tools",
        cache_path=tmp_path / "state" / "cache.db",
        memory_path=tmp_path / "state" / "memory.db",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        sandbox_backend="local",
        allow_local_sandbox=True,
        llm_mode="offline",
        env_allowlist=[],
        repo_root=REPO_ROOT,
    )


def _spec(ledger: FailureLedger, goal: str = "reverse a string"):
    return ledger.record_gap(
        goal,
        required_inputs={"text": "string"},
        expected_outputs={"reversed": "string"},
        suggested_provides=["text.reversed"],
        suggested_requires=["text.raw"],
    )


def _engine(tmp_path: Path):
    settings = _settings(tmp_path)
    ledger = FailureLedger(settings.state_dir)
    registry = Registry(settings).load(strict=True)
    engine = ForgeEngine(settings, registry=registry, ledger=ledger, backend="local")
    return engine, ledger, settings


def test_ledger_records_and_increments_a_missing_gap(tmp_path):
    ledger = FailureLedger(tmp_path)
    first = ledger.record_gap("reverse a string", required_inputs={"text": "string"})
    second = ledger.record_gap("reverse a string", required_inputs={"text": "string"})
    assert first.frequency == 1
    assert second.frequency == 2
    assert ledger.read()[0].frequency == 2
    assert ledger.read()[0].attempt_count == 0


def test_ultron_gaps_outputs_frequency_sorted_table(tmp_path, monkeypatch, capsys):
    state = tmp_path / "state"
    ledger = FailureLedger(state)
    ledger.record_gap("rare", expected_outputs={"x": "string"})
    ledger.record_gap("common", expected_outputs={"x": "string"})
    ledger.record_gap("common", expected_outputs={"x": "string"})
    monkeypatch.setenv("ULTRON_STATE_DIR", str(state))
    assert main(["gaps"]) == 0
    output = capsys.readouterr().out
    assert "Missing capability gaps" in output
    assert output.index("common") < output.index("rare")
    assert main(["gaps", "--clear"]) == 0
    assert ledger.read() == []


def test_forge_reverse_tool_sandboxes_verifies_and_registers(tmp_path):
    engine, ledger, settings = _engine(tmp_path)
    spec = _spec(ledger)
    engine.add_template(
        spec.goal,
        REVERSE_CODE,
        {
            "name": "reverse_text",
            "version": "0.1.0",
            "risk": "low",
            "inputs": {"text": "string"},
            "outputs": {"reversed": "string"},
            "permissions": [],
        },
        test_inputs={"text": "drawer"},
        expected_schema={"reversed": "string"},
    )
    forged = engine.auto_forge_from_ledger()
    assert [manifest.name for manifest in forged] == ["reverse_text"]
    registered = engine.registry.get("reverse_text")
    assert registered.risk.value == "medium"
    assert registered.provides == ["text.reversed"]
    assert "reverse_text" in engine.registry.graph()
    assert (Path(settings.tools_dir) / "dynamic" / "reverse_text.py").exists()
    final_manifest = Path(settings.tools_dir) / "dynamic" / "reverse_text.json"
    assert "source_path" not in json.loads(final_manifest.read_text(encoding="utf-8"))
    assert engine.registry.find_chain(["text.raw"], ["text.reversed"]) == ["reverse_text"]


def test_auto_forge_skips_three_failed_attempts(tmp_path):
    engine, ledger, _ = _engine(tmp_path)
    spec = _spec(ledger)
    ledger.record_attempt(spec)
    ledger.record_attempt(spec)
    ledger.record_attempt(spec)
    assert ledger.read()[0].attempt_count == 3
    engine.add_template(spec.goal, REVERSE_CODE, {"name": "never_used"})
    assert engine.auto_forge_from_ledger() == []
    assert engine.last_report["skipped"][0]["reason"] == "attempt_count >= 3"
