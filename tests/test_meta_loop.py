from __future__ import annotations

from pathlib import Path

from ultron.config import REPO_ROOT, load_settings
from ultron.engine.dag import CapabilityGraph, GoalDecomposer
from ultron.engine.runtime import RuntimeEngine
from ultron.registry import Registry


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path / "state",
        tools_dir=REPO_ROOT / "tools",
        cache_path=tmp_path / "state" / "cache.db",
        memory_path=tmp_path / "state" / "memory.db",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        sandbox_backend="local",
        allow_local_sandbox=True,
        llm_mode="offline",
        web_mock=str(REPO_ROOT / "eval" / "fixtures" / "web_mock.json"),
    )


def test_dag_plans_typed_dependencies_topologically_and_flags_jit(tmp_path):
    settings = _settings(tmp_path)
    registry = Registry(settings).load(strict=True)
    decomposer = GoalDecomposer(registry)
    dag = decomposer.plan(
        "collect then summarize",
        {
            "nodes": [
                {
                    "id": "summarize",
                    "goal": "summarize collected data",
                    "required_capability": "summarize_text",
                    "inputs": {"text": "string"},
                    "outputs": {"summary": "string"},
                    "dependencies": ["collect"],
                },
                {
                    "id": "collect",
                    "goal": "collect records",
                    "required_capability": "collect_records",
                    "outputs": {"records": "list[string]"},
                },
            ],
            "available_capabilities": ["collect_records"],
        },
    )
    assert dag.topological_order() == ["collect", "summarize"]
    assert dag.nodes["summarize"].jit_target
    assert dag.missing_capabilities == ["summarize_text"]
    assert dag.as_dict()["order"] == ["collect", "summarize"]


def test_jit_harvest_refine_test_register_execute_once(tmp_path):
    settings = _settings(tmp_path)
    registry = Registry(settings).load(strict=True)
    events: list[str] = []

    class Harvester:
        def search(self, goal):
            events.append(f"harvest:{goal}")
            return [
                {
                    "code": "def run(payload): return {'answer': 'fixture'}",
                    "manifest": {
                        "name": "fixture_jit",
                        "entrypoint": "python -c pass",
                        "risk": "low",
                        "outputs": {"answer": "string"},
                        "provides": ["fixture.answer"],
                    },
                    "test_inputs": {},
                }
            ]

    class Scavenger:
        def search(self, goal):
            events.append(f"scavenge:{goal}")
            return []

    class Refinery:
        def refine(self, code, manifest):
            events.append("refine")
            return code, manifest

    class Forge:
        def test_tool(self, manifest, inputs, expected):
            events.append("sandbox-test-breaker")
            return True

    def execute(manifest, inputs):
        events.append(f"execute:{manifest.name}")
        return {"answer": "fixture"}

    runtime = RuntimeEngine(
        settings,
        registry=registry,
        capability_graph=CapabilityGraph(registry),
        harvester=Harvester(),
        scavenger=Scavenger(),
        refinery=Refinery(),
        forge_engine=Forge(),
        executor=execute,
        verifier=lambda manifest, result, goal, provenance: True,
    )
    result = runtime.run(
        "answer with fixture",
        {
            "nodes": [
                {
                    "id": "answer",
                    "goal": "answer with fixture",
                    "required_capability": "fixture.answer",
                    "outputs": {"answer": "string"},
                }
            ]
        },
    )
    assert result.ok
    assert result.outputs == {"answer": "fixture"}
    assert events == [
        "harvest:answer with fixture",
        "scavenge:answer with fixture",
        "refine",
        "sandbox-test-breaker",
        "execute:fixture_jit",
    ]
    assert "fixture_jit" in registry


def test_breaker_failure_enters_bounded_repair_loop(tmp_path):
    settings = _settings(tmp_path)
    registry = Registry(settings).load(strict=True)
    manifest_path = Path(settings.tools_dir) / "repairable.0.1.0.json"
    # Use an existing manifest shape by writing a tiny test-only tool beside the
    # real registry is unnecessary: an injected registry is clearer below.
    from ultron.registry import ToolManifest

    manifest = ToolManifest(
        name="repairable",
        version="0.1.0",
        entrypoint="python -c pass",
        risk="low",
        provides=["repair.value"],
        outputs={"value": "string"},
    )
    registry.register(manifest)
    calls = {"breaker": 0, "repair": 0}

    class Breaker:
        def verify(self, result, provenance):
            calls["breaker"] += 1
            return calls["breaker"] > 1

    class Repair:
        def repair(self, **kwargs):
            calls["repair"] += 1
            return {"inputs": {}}

    runtime = RuntimeEngine(
        settings,
        registry=registry,
        capability_graph=CapabilityGraph(registry),
        executor=lambda manifest, inputs: {"value": "stable"},
        breaker=Breaker(),
        verifier=lambda manifest, result, goal, provenance: True,
        repair_engine=Repair(),
        max_repair_attempts=3,
    )
    result = runtime.run(
        "repair this",
        {"nodes": [{"id": "repair", "goal": "repair this", "required_capability": "repair.value"}]},
    )
    assert result.ok
    assert result.nodes[0].attempts == 2
    assert result.nodes[0].repaired == 1
    assert calls == {"breaker": 2, "repair": 1}
    assert not manifest_path.exists()
