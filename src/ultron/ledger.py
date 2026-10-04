"""Persistent, append-only record of goals the current registry cannot solve."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MissingCapabilitySpec:
    """Description of a missing capability and its forge history."""

    goal: str
    required_inputs: dict[str, str] = field(default_factory=dict)
    expected_outputs: dict[str, str] = field(default_factory=dict)
    suggested_provides: list[str] = field(default_factory=list)
    suggested_requires: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    frequency: int = 0
    attempt_count: int = 0
    failure_reason: str = ""
    # Compatibility aliases make specs convenient to construct from planner
    # output while the canonical serialized names remain explicit.
    inputs: dict[str, str] | None = None
    outputs: dict[str, str] | None = None
    provides: list[str] | None = None
    requires: list[str] | None = None
    missing_capability: str | None = None

    def __post_init__(self) -> None:
        if self.inputs is not None and not self.required_inputs:
            self.required_inputs = dict(self.inputs)
        if self.outputs is not None and not self.expected_outputs:
            self.expected_outputs = dict(self.outputs)
        if self.provides is not None and not self.suggested_provides:
            self.suggested_provides = list(self.provides)
        if self.requires is not None and not self.suggested_requires:
            self.suggested_requires = list(self.requires)
        self.inputs = self.required_inputs
        self.outputs = self.expected_outputs
        self.provides = self.suggested_provides
        self.requires = self.suggested_requires
        if self.missing_capability is None:
            self.missing_capability = self.goal

    @property
    def attempts(self) -> int:
        """Readable alias for integrations that call the field ``attempts``."""
        return self.attempt_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "required_inputs": self.required_inputs,
            "expected_outputs": self.expected_outputs,
            "suggested_provides": self.suggested_provides,
            "suggested_requires": self.suggested_requires,
            "timestamp": self.timestamp,
            "frequency": self.frequency,
            "attempt_count": self.attempt_count,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissingCapabilitySpec:
        # ``attempt_count`` was the only counter in the earliest v0 JSONL
        # shape. Treat it as frequency when reading that shape, without losing
        # compatibility with a state directory created by a prior checkout.
        frequency = int(data.get("frequency", data.get("attempt_count", 0)))
        return cls(
            goal=str(data.get("goal", "")),
            required_inputs=dict(data.get("required_inputs") or {}),
            expected_outputs=dict(data.get("expected_outputs") or {}),
            suggested_provides=list(data.get("suggested_provides") or []),
            suggested_requires=list(data.get("suggested_requires") or []),
            timestamp=float(data.get("timestamp", 0.0)),
            frequency=frequency,
            attempt_count=int(data.get("attempt_count", 0)),
            failure_reason=str(data.get("failure_reason", "")),
        )


class FailureLedger:
    """JSONL ledger with latest-record compaction by exact goal."""

    def __init__(self, state_dir: Path | str, *, path: Path | str | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(path) if path else self.state_dir / "failure_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _all(self) -> list[MissingCapabilitySpec]:
        if not self.path.exists():
            return []
        rows: list[MissingCapabilitySpec] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("goal"):
                    rows.append(MissingCapabilitySpec.from_dict(row))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return rows

    def read(self) -> list[MissingCapabilitySpec]:
        """Return one latest row per goal, sorted by request frequency."""
        latest: dict[str, MissingCapabilitySpec] = {}
        for item in self._all():
            latest[item.goal] = item
        return sorted(latest.values(), key=lambda item: (-item.frequency, item.goal))

    def record(
        self, spec: MissingCapabilitySpec, *, failure_reason: str | None = None
    ) -> MissingCapabilitySpec:
        """Record one missing-capability observation and increment frequency."""
        previous = next((item for item in self.read() if item.goal == spec.goal), None)
        current = MissingCapabilitySpec(
            goal=spec.goal,
            required_inputs=dict(
                spec.required_inputs or (previous.required_inputs if previous else {})
            ),
            expected_outputs=dict(
                spec.expected_outputs or (previous.expected_outputs if previous else {})
            ),
            suggested_provides=list(
                spec.suggested_provides or (previous.suggested_provides if previous else [])
            ),
            suggested_requires=list(
                spec.suggested_requires or (previous.suggested_requires if previous else [])
            ),
            timestamp=time.time(),
            frequency=(previous.frequency if previous else spec.frequency) + 1,
            attempt_count=previous.attempt_count if previous else spec.attempt_count,
            failure_reason=(failure_reason if failure_reason is not None else spec.failure_reason),
        )
        self._append(current)
        return current

    def record_attempt(
        self, spec: MissingCapabilitySpec, *, failure_reason: str = ""
    ) -> MissingCapabilitySpec:
        """Record a forge attempt without treating it as another user request."""
        previous = next((item for item in self.read() if item.goal == spec.goal), None)
        current = MissingCapabilitySpec(
            goal=spec.goal,
            required_inputs=dict(
                spec.required_inputs or (previous.required_inputs if previous else {})
            ),
            expected_outputs=dict(
                spec.expected_outputs or (previous.expected_outputs if previous else {})
            ),
            suggested_provides=list(
                spec.suggested_provides or (previous.suggested_provides if previous else [])
            ),
            suggested_requires=list(
                spec.suggested_requires or (previous.suggested_requires if previous else [])
            ),
            timestamp=time.time(),
            frequency=previous.frequency if previous else max(1, spec.frequency),
            attempt_count=(previous.attempt_count if previous else spec.attempt_count) + 1,
            failure_reason=failure_reason,
        )
        self._append(current)
        return current

    def _append(self, current: MissingCapabilitySpec) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(current.as_dict(), sort_keys=True) + "\n")

    def record_gap(
        self,
        goal: str,
        *,
        required_inputs: dict[str, str] | None = None,
        expected_outputs: dict[str, str] | None = None,
        suggested_provides: list[str] | None = None,
        suggested_requires: list[str] | None = None,
        failure_reason: str = "",
    ) -> MissingCapabilitySpec:
        return self.record(
            MissingCapabilitySpec(
                goal=goal,
                required_inputs=required_inputs or {},
                expected_outputs=expected_outputs or {},
                suggested_provides=suggested_provides or [],
                suggested_requires=suggested_requires or [],
                failure_reason=failure_reason,
            )
        )

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def top(self, limit: int = 3) -> list[MissingCapabilitySpec]:
        return self.read()[: max(0, limit)]

    def __len__(self) -> int:
        return len(self.read())
