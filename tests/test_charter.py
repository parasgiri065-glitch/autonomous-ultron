"""Autonomy Charter tier and PolicyGate integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultron.charter import DEFAULT_TIERS, Charter
from ultron.config import REPO_ROOT, load_settings
from ultron.errors import PolicyDenied
from ultron.policy import PolicyGate, PolicyRequest
from ultron.registry import Registry, RiskTier


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path,
        cache_path=tmp_path / "cache.db",
        memory_path=tmp_path / "memory.db",
        approvals_file=tmp_path / "approvals.json",
        audit_log=tmp_path / "audit.jsonl",
        sandbox_backend="local",
        allow_local_sandbox=True,
        llm_mode="offline",
        repo_root=REPO_ROOT,
    )


def test_default_charter_tiers_and_custom_override(tmp_path):
    charter = Charter(tmp_path)
    assert charter.evaluate("sandbox_execution") == "green"
    assert charter.evaluate("tool_registration") == "yellow"
    assert charter.evaluate("paid_action") == "red"
    assert all(charter.evaluate(action) == tier for action, tier in DEFAULT_TIERS.items())

    config = tmp_path / "custom.yaml"
    config.write_text("green:\n  - paid_action\nred:\n  - sandbox_execution\n", encoding="utf-8")
    custom = Charter(tmp_path, config_path=config)
    assert custom.evaluate("paid_action") == "green"
    assert custom.evaluate("sandbox_execution") == "red"

    compact = tmp_path / "compact.json"
    compact.write_text('{"forge_test": "yellow"}', encoding="utf-8")
    assert Charter(tmp_path, config_path=compact).evaluate("forge_test") == "yellow"


def test_charter_yellow_logs_and_continues(tmp_path):
    charter = Charter(tmp_path)
    assert charter.allows("repo_push") is True
    lines = (tmp_path / "charter.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["tier"] == "yellow"


def test_policy_gate_green_forge_test_bypasses_prompt(tmp_path):
    settings = _settings(tmp_path)
    registry = Registry(load_settings()).load(strict=True)
    manifest = registry.get("calc").model_copy(update={"risk": RiskTier.MEDIUM})
    gate = PolicyGate(settings)
    decision = gate.evaluate(
        PolicyRequest(
            tool=manifest,
            inputs={"expression": "1+1"},
            action_type="forge_test",
        ),
        interactive=False,
    )
    assert decision.allowed is True
    assert decision.reason.startswith("charter GREEN")


def test_policy_gate_red_halts(tmp_path):
    settings = _settings(tmp_path)
    manifest = Registry(load_settings()).load(strict=True).get("calc")
    gate = PolicyGate(settings)
    with pytest.raises(PolicyDenied):
        gate.evaluate(
            PolicyRequest(tool=manifest, inputs={"expression": "1+1"}, action_type="paid_action"),
            interactive=False,
        )
