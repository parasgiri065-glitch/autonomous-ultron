"""Offline coverage for provenance and the Breaker boundary."""

from __future__ import annotations

import json
from pathlib import Path

from ultron.agent import Agent
from ultron.breaker import BreakerVerifier
from ultron.config import REPO_ROOT, load_settings
from ultron.policy import PolicyDecision
from ultron.provenance import ProvenanceEnvelope
from ultron.registry import Registry
from ultron.sandbox import Sandbox, SandboxResult


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
        enable_llm_judge=False,
        env_allowlist=[],
        web_mock=str(REPO_ROOT / "eval" / "fixtures" / "web_mock.json"),
        eval_live=False,
        policy_network_low_auto=True,
    )


def test_provenance_tag_is_applied_to_tool_output(tmp_path):
    settings = _settings(tmp_path)
    registry = Registry(settings).load(strict=True)
    manifest = registry.get("calc")
    sandbox = Sandbox(settings, backend="local")
    decision = PolicyDecision(
        action="allow",
        reason="test",
        risk="low",
        tool=manifest.name,
        version=manifest.version,
        network="none",
        granted=True,
        limits={"timeout_s": 10, "memory": "256m", "cpus": "1", "pids": 64},
    )
    outcome = sandbox.run(manifest, {"expression": "2+2"}, decision, use_cache=False)
    output = next(item for item in outcome.provenance if item.origin == "sandbox_tool")
    assert output.source_id == f"{manifest.key}#{manifest.content_hash}"
    assert output.metadata["tool_key"] == manifest.key
    assert output.metadata["content_hash"] == manifest.content_hash
    assert output.verified is False
    assert output.payload == outcome.result


def test_breaker_catches_hallucinated_number_and_entity():
    raw = ProvenanceEnvelope.tool_output(
        "calc@0.1.0",
        "hash",
        {"answer": "Berlin has 3 million residents."},
        metadata={
            "raw_stdout": '{"ok":true,"result":{"answer":"Berlin has 3 million residents."}}'
        },
    )
    verdict = BreakerVerifier().verify(
        {"answer": "Paris has 99 million residents."},
        [raw],
    )
    assert verdict.ok is False
    assert "99" in verdict.unsupported_numbers
    assert "Paris" in verdict.unsupported_entities


def test_breaker_approves_fully_grounded_output():
    raw = ProvenanceEnvelope.tool_output(
        "web_research@0.1.0",
        "hash",
        {"summary": "Paris has 2 million residents."},
    )
    verdict = BreakerVerifier().verify(
        {"answer": "Paris has 2 million residents."},
        [raw],
    )
    assert verdict.ok is True
    assert verdict.untrusted is False


def test_breaker_rejection_never_writes_tool_cache_or_memory(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    agent = Agent(settings=settings, sandbox_backend="local", use_memory_recall=False)
    manifest = agent.registry.get("calc")
    raw = ProvenanceEnvelope.tool_output(
        manifest.key,
        manifest.content_hash,
        {"result": 2.0, "expression": "1+1", "normalized": "1 + 1"},
        metadata={
            "ok": True,
            "raw_stdout": json.dumps(
                {"ok": True, "result": {"result": 2.0, "expression": "1+1", "normalized": "1 + 1"}}
            ),
        },
    )
    bad_result = {"result": 999.0, "expression": "1+1", "normalized": "1 + 1"}
    key = "rejected-cache-key"

    def fake_run(*_args, **_kwargs):
        return SandboxResult(
            tool=manifest.name,
            version=manifest.version,
            ok=True,
            result=bad_result,
            backend="local",
            run_id=agent.run_id,
            provenance=[
                ProvenanceEnvelope.user_input({"expression": "1+1"}),
                raw,
            ],
            cache_key=key,
            cache_ttl_s=600,
        )

    monkeypatch.setattr(agent.sandbox, "run", fake_run)
    result = agent.run("calculate 1+1")
    assert result.status == "ungrounded_rejection"
    assert result.ok is False
    assert agent.cache.get("tool", key) is None
    assert agent.memory.recent_runs() == []
