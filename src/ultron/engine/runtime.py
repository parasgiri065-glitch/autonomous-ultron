"""The bounded universal meta-loop runtime.

The runtime is intentionally orchestration, not an unrestricted agent. Every
stage is injectable so offline tests can exercise harvest/refine/sandbox/
registration without network, package installation, or an LLM.
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..breaker import BreakerVerifier
from ..charter import Charter
from ..config import Settings, get_settings
from ..errors import HumanApprovalRequired, PolicyDenied, UltronError
from ..forge import ForgeEngine
from ..harvester import PyPIHarvester
from ..ledger import FailureLedger, MissingCapabilitySpec
from ..policy import PolicyGate, PolicyRequest
from ..provenance import ProvenanceEnvelope
from ..refinery import ToolRefinery
from ..registry import Registry, RiskTier, ToolManifest
from ..repair import RepairEngine
from ..sandbox import Sandbox, SandboxError, SandboxResult, SandboxUnavailable
from ..scavenger import Scavenger
from ..verifier import Verifier
from .dag import CapabilityGraph, DAGNode, ExecutionDAG, GoalDecomposer
from .wheel_fetch import WheelFetcher


class RuntimeError_(UltronError):
    """A meta-loop stage failed safely."""


@dataclass(slots=True)
class NodeExecution:
    node_id: str
    capability: str
    result: dict[str, Any] | None = None
    ok: bool = False
    verified: bool = False
    attempts: int = 0
    forged: bool = False
    repaired: int = 0
    error: str = ""
    provenance: list[ProvenanceEnvelope] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capability": self.capability,
            "result": self.result,
            "ok": self.ok,
            "verified": self.verified,
            "attempts": self.attempts,
            "forged": self.forged,
            "repaired": self.repaired,
            "error": self.error,
            "provenance": [item.as_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class RuntimeResult:
    goal: str
    ok: bool
    dag: ExecutionDAG | None = None
    nodes: list[NodeExecution] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    recalled: bool = False
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    legacy_result: Any | None = field(default=None, repr=False)

    @property
    def verified(self) -> bool:
        return self.ok and bool(self.nodes) and all(node.verified for node in self.nodes)

    @property
    def trajectory(self) -> list[str]:
        return [node.capability for node in self.nodes]

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "ok": self.ok,
            "verified": self.verified,
            "dag": self.dag.as_dict() if self.dag else None,
            "nodes": [node.as_dict() for node in self.nodes],
            "outputs": self.outputs,
            "answer": self.answer,
            "recalled": self.recalled,
            "errors": list(self.errors),
            "duration_s": round(self.duration_s, 4),
        }


Executor = Callable[..., Any]


class RuntimeEngine:
    """Execute a dynamic DAG and forge only through explicit safe adapters."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        registry: Registry | None = None,
        decomposer: GoalDecomposer | None = None,
        capability_graph: CapabilityGraph | None = None,
        harvester: Any | None = None,
        scavenger: Any | None = None,
        refinery: Any | None = None,
        forge_engine: Any | None = None,
        repair_engine: Any | None = None,
        executor: Executor | None = None,
        breaker: Any | None = None,
        verifier: Any | None = None,
        memory: Any | None = None,
        identity: Any | None = None,
        agent: Any | None = None,
        wheel_fetcher: WheelFetcher | None = None,
        max_repair_attempts: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.registry = registry or Registry(self.settings).load()
        self.capability_graph = capability_graph or CapabilityGraph(self.registry)
        self.decomposer = decomposer or GoalDecomposer(
            self.registry, capability_graph=self.capability_graph
        )
        self.forge_engine = forge_engine or ForgeEngine(self.settings, registry=self.registry)
        self.harvester = (
            harvester
            if harvester is not None
            else PyPIHarvester(self.settings, forge=self.forge_engine)
        )
        self.scavenger = scavenger if scavenger is not None else Scavenger(self.settings)
        self.refinery = (
            refinery
            if refinery is not None
            else ToolRefinery(self.settings, forge=self.forge_engine)
        )
        self.repair_engine = (
            repair_engine if repair_engine is not None else RepairEngine(self.settings)
        )
        self.executor = executor
        self.breaker = breaker or BreakerVerifier()
        self.verifier = verifier
        self.memory = memory
        self.identity = identity
        self.agent = agent
        self.wheel_fetcher = wheel_fetcher or WheelFetcher(self.settings)
        self.max_repair_attempts = max(1, int(max_repair_attempts))
        self.jit_errors: list[str] = []
        self.ledger = FailureLedger(self.settings.state_dir)
        self._sandbox: Sandbox | None = None
        self._gate: PolicyGate | None = None

    def plan(self, goal: str, context: dict[str, Any] | None = None) -> ExecutionDAG:
        return self.decomposer.plan(goal, self._ground_context(context))

    def run(self, goal: str, context: dict[str, Any] | None = None) -> RuntimeResult:
        started = time.perf_counter()
        goal = (goal or "").strip()
        result = RuntimeResult(goal=goal, ok=False)
        if not goal:
            result.errors.append("goal must not be empty")
            return result
        context = self._ground_context(context)

        recalled = self._recall(goal, context)
        if recalled is not None:
            result.ok = True
            result.recalled = True
            result.outputs = dict(recalled.get("outputs") or recalled.get("result") or {})
            result.answer = str(recalled.get("answer") or _render_answer(result.outputs))
            result.duration_s = time.perf_counter() - started
            return result

        if self.agent is not None:
            delegated = self.agent.run(goal)
            result.legacy_result = delegated
            result.ok = bool(getattr(delegated, "ok", False))
            result.answer = str(getattr(delegated, "answer", "") or "")
            result.outputs = {"answer": result.answer} if result.answer else {}
            for step in getattr(delegated, "steps", []) or []:
                verification = getattr(step, "verification", None)
                result.nodes.append(
                    NodeExecution(
                        node_id=f"step_{getattr(step, 'index', len(result.nodes) + 1)}",
                        capability=str(getattr(step, "tool", "")),
                        result=getattr(step, "result", None),
                        ok=bool(getattr(step, "ok", False)),
                        verified=bool(getattr(verification, "ok", False)),
                    )
                )
            if result.ok:
                self._consolidate(goal, result, context)
            else:
                result.errors.append(str(getattr(delegated, "status", "delegated runtime failed")))
            result.duration_s = time.perf_counter() - started
            return result

        try:
            supplied_dag = context.pop("_execution_dag", None)
            dag = (
                supplied_dag
                if isinstance(supplied_dag, ExecutionDAG)
                else self.decomposer.plan(goal, context)
            )
            result.dag = dag
            completed: dict[str, dict[str, Any]] = {}
            for node in dag.ordered_nodes:
                if node.jit_target or not self.capability_graph.has(node.requires):
                    manifest = self._jit_synthesize(node, context)
                    if manifest is None:
                        detail = "; ".join(self.jit_errors)
                        message = f"missing capability {node.requires!r} could not be forged"
                        if detail:
                            message += f": {detail}"
                        raise RuntimeError_(message)
                    node.metadata["forged_manifest"] = manifest.key
                execution = self._execute_with_refinement(node, completed, context)
                result.nodes.append(execution)
                if not execution.ok:
                    raise RuntimeError_(execution.error or f"node {node.node_id} failed")
                completed[node.node_id] = dict(execution.result or {})
                result.outputs.update(execution.result or {})

            result.ok = True
            result.answer = _render_answer(result.outputs)
            self._consolidate(goal, result, context)
        except (
            RuntimeError_,
            SandboxError,
            SandboxUnavailable,
            PolicyDenied,
            HumanApprovalRequired,
        ) as exc:
            result.errors.append(str(exc))
            self.ledger.record_gap(
                goal,
                expected_outputs={"answer": "string"},
                failure_reason=str(exc),
            )
        except Exception as exc:  # fail closed while retaining diagnostic context
            result.errors.append(f"{type(exc).__name__}: {exc}")
            self.ledger.record_gap(
                goal,
                expected_outputs={"answer": "string"},
                failure_reason=traceback.format_exc(limit=3),
            )
        result.duration_s = time.perf_counter() - started
        return result

    execute = run
    run_goal = run

    def _ground_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        grounded = dict(context or {})
        if self.identity is not None:
            identity_context = _call_optional(self.identity, ("context", "grounding"), default={})
            if isinstance(identity_context, dict):
                grounded["soul"] = identity_context
        return grounded

    def _recall(self, goal: str, context: dict[str, Any]) -> dict[str, Any] | None:
        if self.memory is None:
            return None
        value = _call_method(self.memory, "recall", goal, context, default=None)
        return value if isinstance(value, dict) else None

    def _consolidate(self, goal: str, result: RuntimeResult, context: dict[str, Any]) -> None:
        if self.memory is None or not result.ok or not result.verified:
            return
        _call_method(
            self.memory,
            "record_episode",
            goal,
            trajectory=result.trajectory,
            result=result.outputs,
            verified=True,
            context=context,
            answer=result.answer,
            default=None,
        )
        _call_method(self.memory, "consolidate", default=None)

    def _execute_with_refinement(
        self,
        node: DAGNode,
        completed: dict[str, dict[str, Any]],
        context: dict[str, Any],
    ) -> NodeExecution:
        manifest = self._manifest_for_node(node)
        inputs = self._node_inputs(node, manifest, completed, context)
        execution = NodeExecution(node.node_id, node.requires)
        current_inputs = inputs
        current_manifest = manifest
        for attempt in range(1, self.max_repair_attempts + 1):
            execution.attempts = attempt
            raw = self._execute_manifest(node, current_manifest, current_inputs)
            normalized = self._normalize_execution(raw, current_manifest, node)
            if not normalized["provenance"] and isinstance(normalized["result"], dict):
                normalized["provenance"] = [
                    ProvenanceEnvelope.tool_output(
                        current_manifest.key,
                        current_manifest.content_hash,
                        normalized["result"],
                        origin="sandbox_tool",
                        metadata={"runtime_node": node.node_id},
                    )
                ]
            breaker_ok, breaker_reason = self._breaker_check(
                normalized["result"], normalized["provenance"]
            )
            verifier_ok, verifier_reason = self._schema_check(
                current_manifest, normalized["result"], normalized["provenance"], node.goal
            )
            normalized_ok = normalized["ok"] and breaker_ok and verifier_ok
            if normalized_ok:
                execution.result = normalized["result"]
                execution.provenance = normalized["provenance"]
                execution.ok = True
                execution.verified = True
                return execution
            execution.error = (
                normalized["error"] or breaker_reason or verifier_reason or "quality checks failed"
            )
            if attempt >= self.max_repair_attempts:
                break
            repaired = self._repair(
                node, current_manifest, current_inputs, execution.error, attempt
            )
            if repaired is None:
                break
            current_manifest, current_inputs = self._apply_repair(
                current_manifest, current_inputs, repaired
            )
            execution.repaired += 1
        return execution

    def _manifest_for_node(self, node: DAGNode) -> ToolManifest:
        capability = node.requires
        if capability in self.registry:
            return self.registry.get(capability)
        providers = self.capability_graph.providers(capability)
        if providers:
            return self.registry.get(providers[0])
        manifest_name = node.metadata.get("manifest") or node.metadata.get("tool")
        if manifest_name and manifest_name in self.registry:
            return self.registry.get(str(manifest_name))
        raise RuntimeError_(f"no registered provider for capability {capability!r}")

    def _node_inputs(
        self,
        node: DAGNode,
        manifest: ToolManifest,
        completed: dict[str, dict[str, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        inputs = dict(context.get("inputs") or {})
        for dependency in node.dependencies:
            inputs.update(completed.get(dependency, {}))
        if not manifest.inputs:
            return {}
        matching = {key: inputs[key] for key in manifest.inputs if key in inputs}
        if len(manifest.inputs) == 1 and not matching and node.dependencies:
            key, declared = next(iter(manifest.inputs.items()))
            previous = completed.get(node.dependencies[-1], {})
            if declared in {"dict", "any"}:
                matching[key] = previous
            elif declared == "string":
                value = previous.get("text", previous.get("summary", previous))
                matching[key] = (
                    value if isinstance(value, str) else json.dumps(value, sort_keys=True)
                )
        return matching

    def _execute_manifest(
        self, node: DAGNode, manifest: ToolManifest, inputs: dict[str, Any]
    ) -> Any:
        if self.executor is not None:
            return _invoke_executor(self.executor, node, manifest, inputs)
        if self._sandbox is None:
            self._sandbox = Sandbox(self.settings, cache=None, run_id=f"meta-{int(time.time())}")
        if self._gate is None:
            self._gate = PolicyGate(self.settings, charter=Charter(self.settings.state_dir))
        policy_manifest = manifest
        if (
            node.metadata.get("forged_manifest")
            and manifest.risk.value == "medium"
            and not manifest.permissions
        ):
            # Forge has already sanitized, sandbox-tested, and Breaker-verified
            # this artifact. Keep any declared permissions fail-closed; only a
            # sealed, permission-free artifact receives the low-risk execution
            # decision needed for autonomous JIT continuation.
            policy_manifest = manifest.model_copy(update={"risk": RiskTier.LOW})
        request = PolicyRequest(
            tool=policy_manifest,
            inputs=inputs,
            goal=node.goal,
            action_type="network_egress" if manifest.wants_network else "tool_execution",
        )
        decision = self._gate.evaluate(request, interactive=False)
        outcome = self._sandbox.run(manifest, inputs, decision, defer_cache_write=True)
        return outcome

    def _normalize_execution(
        self, raw: Any, manifest: ToolManifest, node: DAGNode
    ) -> dict[str, Any]:
        if isinstance(raw, NodeExecution):
            return {
                "ok": raw.ok,
                "result": raw.result,
                "error": raw.error,
                "provenance": raw.provenance,
            }
        if isinstance(raw, SandboxResult):
            return {
                "ok": raw.ok,
                "result": raw.result,
                "error": raw.error or "",
                "provenance": list(raw.provenance),
            }
        if isinstance(raw, dict):
            # An executor may return a structured runtime envelope or a bare tool result.
            if {"result", "ok"}.intersection(raw) and "result" in raw:
                return {
                    "ok": bool(raw.get("ok", True)),
                    "result": raw.get("result"),
                    "error": str(raw.get("error") or ""),
                    "provenance": list(raw.get("provenance") or []),
                }
            return {"ok": True, "result": raw, "error": "", "provenance": []}
        if isinstance(raw, tuple) and len(raw) == 2:
            return {"ok": bool(raw[0]), "result": raw[1], "error": "", "provenance": []}
        ok = bool(getattr(raw, "ok", False))
        return {
            "ok": ok,
            "result": getattr(raw, "result", None),
            "error": str(getattr(raw, "error", "")),
            "provenance": list(getattr(raw, "provenance", []) or []),
        }

    def _schema_check(
        self,
        manifest: ToolManifest,
        result: dict[str, Any] | None,
        provenance: list[ProvenanceEnvelope],
        goal: str,
    ) -> tuple[bool, str]:
        if result is None or not isinstance(result, dict):
            return False, "output is not a JSON object"
        if self.verifier is not None:
            verdict = _invoke_verifier(self.verifier, manifest, result, goal, provenance)
            return _quality_ok(verdict), _quality_reason(verdict)
        verifier = Verifier(settings=self.settings, breaker=self.breaker)
        verdict = verifier.verify(manifest, result, goal=goal, provenance=provenance)
        return bool(verdict.ok), verdict.reason

    def _breaker_check(
        self, result: dict[str, Any] | None, provenance: list[ProvenanceEnvelope]
    ) -> tuple[bool, str]:
        if self.breaker is None:
            return True, ""
        verdict = self.breaker.verify(result or {}, provenance)
        return _quality_ok(verdict), _quality_reason(verdict)

    def _repair(
        self,
        node: DAGNode,
        manifest: ToolManifest,
        inputs: dict[str, Any],
        error: str,
        attempt: int,
    ) -> Any:
        repair = self.repair_engine
        if repair is None:
            return None
        # RepairEngine's durable ticket is always recorded; injected repair
        # adapters may additionally return corrected inputs/code/manifest.
        _call_method(
            repair,
            "record_fault",
            "meta_loop",
            error,
            error,
            {"node_id": node.node_id, "goal": node.goal, "attempt": attempt},
            default=None,
        )
        for name in ("repair", "adjust", "attempt_repair", "repair_node"):
            method = getattr(repair, name, None)
            if callable(method):
                return _invoke_flexible(
                    method,
                    node=node,
                    manifest=manifest,
                    inputs=inputs,
                    error=error,
                    attempt=attempt,
                )
        return None

    @staticmethod
    def _apply_repair(
        manifest: ToolManifest, inputs: dict[str, Any], repaired: Any
    ) -> tuple[ToolManifest, dict[str, Any]]:
        if isinstance(repaired, ToolManifest):
            return repaired, inputs
        if isinstance(repaired, dict):
            next_inputs = dict(repaired.get("inputs") or inputs)
            next_manifest = repaired.get("manifest", manifest)
            if isinstance(next_manifest, dict):
                next_manifest = manifest.model_copy(update=next_manifest)
            return next_manifest if isinstance(
                next_manifest, ToolManifest
            ) else manifest, next_inputs
        if isinstance(repaired, tuple) and len(repaired) == 2:
            next_manifest, next_inputs = repaired
            return (
                next_manifest if isinstance(next_manifest, ToolManifest) else manifest,
                dict(next_inputs) if isinstance(next_inputs, dict) else inputs,
            )
        return manifest, inputs

    def _query_candidates(self, node: DAGNode, context: dict[str, Any]) -> list[Any]:
        candidates: list[Any] = []
        explicit = context.get("jit_candidates") or context.get("candidates")
        if isinstance(explicit, list):
            candidates.extend(explicit)
        # Both discovery channels are queried. Their default implementations are
        # inert in offline mode, while test doubles can provide deterministic data.
        for provider, names in (
            (self.harvester, ("find_candidates", "search", "discover", "query")),
            (self.scavenger, ("find_candidates", "search", "discover", "query")),
        ):
            if provider is None:
                continue
            for name in names:
                method = getattr(provider, name, None)
                if not callable(method):
                    continue
                try:
                    if name == "discover":
                        value = (
                            method(context.get("seed_urls")) if "seed_urls" in context else method()
                        )
                    else:
                        value = _invoke_flexible(method, goal=node.goal, query=node.goal, node=node)
                except Exception:
                    value = None
                if value is not None:
                    candidates.extend(value if isinstance(value, (list, tuple)) else [value])
                break
        return candidates

    def _jit_synthesize(self, node: DAGNode, context: dict[str, Any]) -> ToolManifest | None:
        self.jit_errors.clear()
        candidates = self._query_candidates(node, context)
        for candidate in candidates:
            try:
                candidate = self._prepare_candidate(candidate)
                manifest = self._synthesize_candidate(candidate, node)
            except Exception as exc:
                self.jit_errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if manifest is not None:
                self.registry.register(manifest)
                self.capability_graph.register(manifest)
                self.decomposer.capability_graph = self.capability_graph
                return manifest
        return None

    def _prepare_candidate(self, candidate: Any) -> Any:
        """Seal declared package wheels (and their dependency closure) on the host."""
        package = _candidate_value(candidate, "package", "")
        if not package:
            return candidate
        version = _candidate_value(candidate, "package_version", None)
        records = self.wheel_fetcher.fetch_closure(str(package), str(version) if version else None)
        if not isinstance(candidate, dict):
            return candidate
        prepared = dict(candidate)
        manifest = dict(prepared.get("manifest") or {})
        meta = dict(manifest.get("meta") or {})
        dependencies = dict(meta.get("dependencies") or {})
        wheels = list(dependencies.get("wheels") or [])
        wheels.extend(records)
        dependencies["wheels"] = wheels
        meta["dependencies"] = dependencies
        manifest["meta"] = meta
        prepared["manifest"] = manifest
        return prepared

    def _synthesize_candidate(self, candidate: Any, node: DAGNode) -> ToolManifest | None:
        if isinstance(candidate, ToolManifest):
            return candidate
        data = _candidate_value(candidate, "manifest", {})
        code = _candidate_value(candidate, "code", _candidate_value(candidate, "raw_code", ""))
        if not isinstance(data, dict):
            data = {}
        if not data.get("name"):
            data["name"] = _safe_tool_name(node.requires or node.node_id)
        data.setdefault("version", "0.1.0")
        data.setdefault("entrypoint", "python -c pass")
        data.setdefault("risk", "low")
        data.setdefault("inputs", node.inputs)
        data.setdefault("outputs", node.outputs or {"result": "any"})
        data.setdefault("provides", [node.requires])
        data.setdefault("requires", [])
        data.setdefault("permissions", [])
        refined = code
        if self.refinery is not None and code:
            method = getattr(self.refinery, "refine", None) or getattr(
                self.refinery, "refine_code", None
            )
            if callable(method):
                if getattr(method, "__name__", "") == "refine_code":
                    value = method(
                        code,
                        _candidate_value(candidate, "target_function", "run"),
                        list(data.get("provides") or []),
                        list(data.get("requires") or []),
                    )
                else:
                    value = _invoke_flexible(
                        method,
                        raw_code=code,
                        code=code,
                        target_func_name=_candidate_value(candidate, "target_function", "run"),
                        provides=list(data.get("provides") or []),
                        requires=list(data.get("requires") or []),
                        manifest=data,
                    )
                if isinstance(value, tuple):
                    refined, new_data = value[0], value[1] if len(value) > 1 else data
                    if isinstance(new_data, dict):
                        for key, value in new_data.items():
                            if key == "inputs" and not value and data.get("inputs"):
                                continue
                            data[key] = value
                elif isinstance(value, str):
                    refined = value
        if refined and not data.get("entrypoint"):
            data["entrypoint"] = "python tool.py"
        if refined:
            data["_raw_code"] = refined
        manifest = ToolManifest(
            **{key: value for key, value in data.items() if not key.startswith("_")}
        )
        test_inputs = _candidate_value(candidate, "test_inputs", {})
        expected = _candidate_value(candidate, "expected_schema", manifest.outputs)
        forge = self.forge_engine
        synthesize = getattr(forge, "synthesize_tool", None)
        if refined and callable(synthesize) and hasattr(forge, "ledger"):
            spec = MissingCapabilitySpec(
                goal=node.goal,
                required_inputs=dict(data.get("inputs") or {}),
                expected_outputs=dict(expected),
                suggested_provides=list(data.get("provides") or [node.requires]),
                suggested_requires=list(data.get("requires") or []),
            )
            temporary = synthesize(
                spec,
                refined,
                {key: value for key, value in data.items() if not key.startswith("_")},
            )
            test_method = getattr(forge, "test_tool", None)
            if not callable(test_method) or test_method(temporary, test_inputs, expected):
                registry = getattr(forge, "registry", self.registry)
                return registry.get(temporary.name)
            reason = str(getattr(forge, "last_test_error", ""))
            if reason:
                self.jit_errors.append(reason)
            return None
        for name in ("synthesize_and_register", "forge", "register_candidate"):
            method = getattr(forge, name, None)
            if callable(method):
                forged = _invoke_flexible(
                    method,
                    candidate=candidate,
                    node=node,
                    manifest=manifest,
                    code=refined,
                    test_inputs=test_inputs,
                    expected_schema=expected,
                )
                if isinstance(forged, ToolManifest):
                    return forged
                if isinstance(forged, dict):
                    return ToolManifest(**forged)
        test_method = getattr(forge, "test_tool", None)
        if callable(test_method):
            passed = _invoke_flexible(test_method, manifest, test_inputs, expected)
            if passed is False:
                reason = str(getattr(forge, "last_test_error", ""))
                if reason:
                    self.jit_errors.append(reason)
                return None
        return manifest


JITRuntime = RuntimeEngine
MetaLoopRuntime = RuntimeEngine
UniversalRuntime = RuntimeEngine
MetaLoopEngine = RuntimeEngine
MetaLoopError = RuntimeError_


def _invoke_executor(
    executor: Executor, node: DAGNode, manifest: ToolManifest, inputs: dict[str, Any]
) -> Any:
    try:
        signature = inspect.signature(executor)
        for positional in ((node, manifest, inputs), (manifest, inputs), (node, inputs)):
            try:
                signature.bind(*positional)
            except TypeError:
                continue
            return executor(*positional)
    except (TypeError, ValueError):
        pass
    return _invoke_flexible(executor, node=node, manifest=manifest, inputs=inputs)


def _invoke_verifier(
    verifier: Any, manifest: ToolManifest, result: dict[str, Any], goal: str, provenance: list[Any]
) -> Any:
    return _invoke_flexible(
        verifier, manifest=manifest, result=result, goal=goal, provenance=provenance
    )


def _invoke_flexible(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call doubles with their natural signature without swallowing inner errors."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    attempts = [
        (args, kwargs),
        ((), kwargs),
        (
            (kwargs.get("code") or kwargs.get("raw_code"), kwargs.get("manifest")),
            {},
        ),
        ((kwargs.get("result"), kwargs.get("provenance")), {}),
        ((kwargs.get("manifest"), kwargs.get("inputs")), {}),
        ((kwargs.get("node"), kwargs.get("inputs")), {}),
        ((kwargs.get("goal"),), {}),
    ]
    if len(args) >= 2:
        attempts.insert(1, (args[-2:], {}))
    for positional, named in attempts:
        try:
            signature.bind(*positional, **named)
        except TypeError:
            continue
        return function(*positional, **named)
    return function(*args, **kwargs)


def _call_method(obj: Any, name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    method = getattr(obj, name, None)
    if not callable(method):
        return default
    return _invoke_flexible(method, *args, **kwargs)


def _call_optional(obj: Any, names: Iterable[str], *, default: Any = None) -> Any:
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return _invoke_flexible(method)
    return default


def _candidate_value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _quality_ok(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return bool(getattr(value, "ok", False))


def _quality_reason(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(getattr(value, "reason", "quality verification failed"))


def _render_answer(outputs: dict[str, Any]) -> str:
    if not outputs:
        return ""
    if len(outputs) == 1:
        value = next(iter(outputs.values()))
        return (
            json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, (dict, list))
            else str(value)
        )
    return json.dumps(outputs, sort_keys=True, default=str)


def _safe_tool_name(value: str) -> str:
    import re

    name = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_") or "jit_tool"
    if not name[0].isalpha():
        name = "jit_" + name
    return name[:49].rstrip("_")


__all__ = [
    "JITRuntime",
    "MetaLoopEngine",
    "MetaLoopError",
    "MetaLoopRuntime",
    "NodeExecution",
    "RuntimeEngine",
    "RuntimeError_",
    "RuntimeResult",
    "UniversalRuntime",
]
