from __future__ import annotations

from pathlib import Path

from ultron.agent import Agent
from ultron.config import REPO_ROOT, load_settings
from ultron.engine.dag import CapabilityGraph
from ultron.engine.runtime import RuntimeResult
from ultron.registry import Registry

GOAL = "Parse the cron expression '0 3 * * *' and give me the next 5 run times in UTC"


def _settings(tmp_path: Path, *, jit_enabled: bool = True):
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
        jit_enabled=jit_enabled,
    )


def test_no_plan_jit_enabled_invokes_goal_decomposer_and_marks_targets(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    registry = Registry(settings).load(strict=True)
    captured = {}

    class FakeRuntime:
        def __init__(self, settings, **kwargs):
            captured["runtime"] = self
            self.decomposer = kwargs["decomposer"]

        def plan(self, goal, context):
            captured["context"] = context
            dag = self.decomposer.plan(goal, context)
            captured["dag"] = dag
            return dag

        def run(self, goal, context):
            assert context["_execution_dag"] is captured["dag"]
            return RuntimeResult(goal=goal, ok=True, dag=captured["dag"], answer="fixture answer")

    import ultron.engine.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "RuntimeEngine", FakeRuntime)
    result = Agent(settings=settings, registry=registry).run(GOAL)

    assert result.ok
    assert result.answer == "fixture answer"
    assert "router -> meta-loop: 1 nodes, 1 JIT" in result.notes
    assert captured["dag"].jit_targets == ["goal"]
    assert captured["dag"].missing_capabilities == ["cron.run_times"]


def test_no_plan_jit_disabled_preserves_old_behavior(tmp_path, monkeypatch):
    settings = _settings(tmp_path, jit_enabled=False)
    registry = Registry(settings).load(strict=True)

    class ExplodingRuntime:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("meta-loop must not run when ULTRON_JIT=0")

    import ultron.engine.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "RuntimeEngine", ExplodingRuntime)
    result = Agent(settings=settings, registry=registry).run(GOAL)

    assert result.status == "no_plan"
    assert not result.ok
    assert "Add a tool manifest" in result.answer
    assert not any("router -> meta-loop" in note for note in result.notes)


def test_cron_like_candidate_is_forged_registered_executed_and_verified(tmp_path):
    settings = _settings(tmp_path)
    registry = Registry(settings).load(strict=True)
    events: list[str] = []

    class Harvester:
        def search(self, goal):
            events.append(f"harvest:{goal}")
            return [
                {
                    "code": "def run(payload): return {'run_times': ['2030-01-01T03:00:00+00:00']}",
                    "manifest": {
                        "name": "cron_fixture",
                        "risk": "low",
                        "inputs": {"expression": "string"},
                        "outputs": {"run_times": "list[string]"},
                        "provides": ["cron.run_times"],
                    },
                    "test_inputs": {"expression": "0 3 * * *"},
                    "expected_schema": {"run_times": "list[string]"},
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
            events.append("sandbox")
            return True

    class Breaker:
        def verify(self, result, provenance):
            events.append("breaker")
            return True

    def execute(manifest, inputs):
        events.append(f"execute:{manifest.name}")
        return {"run_times": ["2030-01-01T03:00:00+00:00"]}

    from ultron.engine.runtime import RuntimeEngine

    runtime = RuntimeEngine(
        settings,
        registry=registry,
        capability_graph=CapabilityGraph(registry),
        harvester=Harvester(),
        scavenger=Scavenger(),
        refinery=Refinery(),
        forge_engine=Forge(),
        breaker=Breaker(),
        executor=execute,
        verifier=lambda manifest, result, goal, provenance: True,
    )
    result = runtime.run(
        GOAL,
        {
            "nodes": [
                {
                    "id": "cron",
                    "goal": GOAL,
                    "required_capability": "cron.run_times",
                    "inputs": {"expression": "string"},
                    "outputs": {"run_times": "list[string]"},
                }
            ],
            "inputs": {"expression": "0 3 * * *"},
        },
    )

    assert result.ok
    assert result.verified
    assert result.outputs == {"run_times": ["2030-01-01T03:00:00+00:00"]}
    assert "cron_fixture" in registry
    assert events == [
        f"harvest:{GOAL}",
        f"scavenge:{GOAL}",
        "refine",
        "sandbox",
        "execute:cron_fixture",
        "breaker",
    ]
