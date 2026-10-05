"""Deterministic goal decomposition and dependency ordering for Phase 5."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..errors import UltronError
from ..registry import Registry, ToolManifest


class DAGError(UltronError):
    """The requested execution graph is malformed or cyclic."""


@dataclass(slots=True)
class DAGNode:
    """One typed unit of work in an :class:`ExecutionDAG`."""

    node_id: str
    goal: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    required_capability: str = ""
    dependencies: list[str] = field(default_factory=list)
    capability: str = ""
    jit_target: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.node_id

    @property
    def requires(self) -> str:
        return self.required_capability or self.capability

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "goal": self.goal,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "required_capability": self.required_capability,
            "dependencies": list(self.dependencies),
            "capability": self.capability,
            "jit_target": self.jit_target,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ExecutionDAG:
    """A validated, typed, dependency-aware plan."""

    goal: str
    nodes: dict[str, DAGNode] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    missing_capabilities: list[str] = field(default_factory=list)
    jit_targets: list[str] = field(default_factory=list)
    _order: list[str] = field(default_factory=list, repr=False)

    @property
    def ordered_nodes(self) -> list[DAGNode]:
        return [self.nodes[node_id] for node_id in self.topological_order()]

    @property
    def is_complete(self) -> bool:
        return not self.missing_capabilities

    def topological_order(self) -> list[str]:
        """Return a stable Kahn ordering and reject cycles/missing references."""
        if self._order and len(self._order) == len(self.nodes):
            return list(self._order)
        indegree = dict.fromkeys(self.nodes, 0)
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for node in self.nodes.values():
            for dependency in node.dependencies:
                if dependency not in self.nodes:
                    raise DAGError(f"node {node.node_id!r} depends on unknown node {dependency!r}")
                indegree[node.node_id] += 1
                outgoing[dependency].append(node.node_id)
        ready = deque(sorted(node_id for node_id, count in indegree.items() if count == 0))
        order: list[str] = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for successor in sorted(outgoing[current]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if len(order) != len(self.nodes):
            cyclic = sorted(node_id for node_id, count in indegree.items() if count)
            raise DAGError(f"dependency cycle detected involving: {', '.join(cyclic)}")
        self._order = order
        return list(order)

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "nodes": [self.nodes[node_id].as_dict() for node_id in self.topological_order()],
            "order": self.topological_order(),
            "missing_capabilities": list(self.missing_capabilities),
            "jit_targets": list(self.jit_targets),
            "context": dict(self.context),
        }


class CapabilityGraph:
    """Small capability view over the existing registry graph/index."""

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        capabilities: Iterable[str] | None = None,
    ) -> None:
        self.registry = registry
        self._capabilities = {str(item) for item in (capabilities or ())}
        self._providers: dict[str, list[str]] = {}
        if registry is not None:
            self.refresh()

    def refresh(self) -> CapabilityGraph:
        if self.registry is None:
            return self
        self._providers.clear()
        for manifest in self.registry.latest():
            self._add_manifest(manifest)
        return self

    def _add_manifest(self, manifest: ToolManifest) -> None:
        self._capabilities.add(manifest.name)
        for capability in manifest.provides:
            self._capabilities.add(capability)
            self._providers.setdefault(capability, []).append(manifest.name)

    def register(self, manifest: ToolManifest) -> None:
        self._add_manifest(manifest)
        if self.registry is not None and manifest.name not in self.registry:
            self.registry.register(manifest)

    def has(self, capability: str) -> bool:
        return str(capability) in self._capabilities

    def providers(self, capability: str) -> list[str]:
        return sorted(self._providers.get(str(capability), []))

    def snapshot(self) -> dict[str, list[str]]:
        return {key: self.providers(key) for key in sorted(self._capabilities)}

    def __contains__(self, capability: str) -> bool:
        return self.has(capability)


class GoalDecomposer:
    """Build typed execution DAGs without inventing tool outputs."""

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        capability_graph: CapabilityGraph | dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        if isinstance(capability_graph, dict):
            available = set(capability_graph)
            for value in capability_graph.values():
                if isinstance(value, (list, tuple, set)):
                    available.update(str(item) for item in value)
            self.capability_graph = CapabilityGraph(capabilities=available)
        else:
            self.capability_graph = capability_graph or CapabilityGraph(registry)

    def plan(self, goal: str, context: dict[str, Any] | None = None) -> ExecutionDAG:
        goal = (goal or "").strip()
        if not goal:
            raise DAGError("goal must not be empty")
        context = dict(context or {})
        raw_nodes = context.get("nodes") or context.get("steps") or context.get("decomposition")
        if raw_nodes is None:
            raw_nodes = self._heuristic_nodes(goal, context)
        if not isinstance(raw_nodes, (list, tuple)) or not raw_nodes:
            raise DAGError("goal decomposition must contain at least one node")

        nodes: dict[str, DAGNode] = {}
        for index, raw in enumerate(raw_nodes):
            if isinstance(raw, DAGNode):
                node = raw
            elif isinstance(raw, str):
                node = DAGNode(
                    node_id=f"step_{index + 1}",
                    goal=raw,
                    required_capability=self._infer_capability(raw, context),
                )
            elif isinstance(raw, dict):
                node = self._node_from_dict(raw, index, context)
            else:
                raise DAGError(f"unsupported decomposition node at index {index}")
            if node.node_id in nodes:
                raise DAGError(f"duplicate node id: {node.node_id}")
            nodes[node.node_id] = node

        self._infer_dependency_edges(nodes)
        available = {str(item) for item in context.get("available_capabilities", ())}
        available.update(str(item) for item in context.get("inputs", {}))
        context_capabilities = context.get("capabilities", {})
        if isinstance(context_capabilities, dict):
            available.update(str(item) for item in context_capabilities)
            for value in context_capabilities.values():
                if isinstance(value, (list, tuple, set)):
                    available.update(str(item) for item in value)
        elif isinstance(context_capabilities, (list, tuple, set)):
            available.update(str(item) for item in context_capabilities)
        missing: list[str] = []
        jit_targets: list[str] = []
        for node in nodes.values():
            capability = node.requires
            if not capability:
                continue
            if capability in available or self.capability_graph.has(capability):
                continue
            if capability not in missing:
                missing.append(capability)
            node.jit_target = True
            jit_targets.append(node.node_id)

        dag = ExecutionDAG(
            goal=goal,
            nodes=nodes,
            context=context,
            missing_capabilities=sorted(missing),
            jit_targets=jit_targets,
        )
        dag.topological_order()
        return dag

    def _node_from_dict(self, raw: dict[str, Any], index: int, context: dict[str, Any]) -> DAGNode:
        node_id = str(raw.get("id") or raw.get("node_id") or raw.get("name") or f"step_{index + 1}")
        node_goal = str(raw.get("goal") or raw.get("task") or node_id).strip()
        capability = str(
            raw.get("required_capability")
            or raw.get("capability")
            or raw.get("requires_capability")
            or self._infer_capability(node_goal, context)
        )
        inputs = _typed_mapping(raw.get("inputs") or raw.get("input_schema") or {})
        outputs = _typed_mapping(raw.get("outputs") or raw.get("output_schema") or {})
        dependencies = [
            str(item) for item in (raw.get("dependencies") or raw.get("depends_on") or [])
        ]
        metadata = dict(raw.get("metadata") or {})
        return DAGNode(
            node_id=node_id,
            goal=node_goal,
            inputs=inputs,
            outputs=outputs,
            required_capability=capability,
            dependencies=dependencies,
            capability=capability,
            metadata=metadata,
        )

    def _heuristic_nodes(self, goal: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        chunks = [
            item.strip()
            for item in re.split(r"\s*(?:;|->|\bthen\b|\bfollowed by\b)\s*", goal, flags=re.I)
            if item.strip()
        ]
        if not chunks:
            chunks = [goal]
        output_types = context.get("outputs") if isinstance(context.get("outputs"), dict) else {}
        return [
            {
                "id": f"step_{index + 1}",
                "goal": chunk,
                "required_capability": self._infer_capability(chunk, context),
                "outputs": output_types if index == len(chunks) - 1 else {},
                "dependencies": [f"step_{index}"] if index else [],
            }
            for index, chunk in enumerate(chunks)
        ]

    def _infer_capability(self, text: str, context: dict[str, Any]) -> str:
        explicit = context.get("capability")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        if self.registry is not None:
            matches = self.registry.search(text, limit=1)
            if matches:
                return matches[0].name
        return _capability_slug(text)

    def _infer_dependency_edges(self, nodes: dict[str, DAGNode]) -> None:
        prior: list[DAGNode] = []
        for node in nodes.values():
            if node.dependencies:
                prior.append(node)
                continue
            for candidate in prior:
                if node.requires and node.requires in candidate.outputs:
                    node.dependencies.append(candidate.node_id)
                    break
            prior.append(node)


ExecutionNode = DAGNode
DAG = ExecutionDAG


def _typed_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, declared in value.items():
        if isinstance(declared, dict):
            declared = declared.get("type", "any")
        result[str(key)] = str(declared)
    return result


def _capability_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_.]+", "_", value.casefold()).strip("_")[:96] or "unknown_capability"


__all__ = [
    "DAG",
    "CapabilityGraph",
    "DAGError",
    "DAGNode",
    "ExecutionDAG",
    "ExecutionNode",
    "GoalDecomposer",
]
