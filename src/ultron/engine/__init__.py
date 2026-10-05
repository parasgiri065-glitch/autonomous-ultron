"""Phase 5 universal meta-loop execution primitives."""

from .dag import DAG, CapabilityGraph, DAGNode, ExecutionDAG, ExecutionNode, GoalDecomposer
from .runtime import (
    JITRuntime,
    MetaLoopEngine,
    MetaLoopError,
    MetaLoopRuntime,
    NodeExecution,
    RuntimeEngine,
    RuntimeResult,
    UniversalRuntime,
)

__all__ = [
    "DAG",
    "CapabilityGraph",
    "DAGNode",
    "ExecutionDAG",
    "ExecutionNode",
    "GoalDecomposer",
    "JITRuntime",
    "MetaLoopEngine",
    "MetaLoopError",
    "MetaLoopRuntime",
    "NodeExecution",
    "RuntimeEngine",
    "RuntimeResult",
    "UniversalRuntime",
]
