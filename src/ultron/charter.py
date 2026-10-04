"""Configurable autonomy charter for actions outside ordinary tool semantics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

Tier = Literal["green", "yellow", "red"]

DEFAULT_TIERS: dict[str, Tier] = {
    "sandbox_execution": "green",
    "cache_read": "green",
    "cache_write": "green",
    "registry_operation": "green",
    "capability_graph_query": "green",
    "failure_ledger_write": "green",
    "forge_test": "green",
    "eval_run": "green",
    "memory_read": "green",
    "memory_write": "green",
    "tool_registration": "yellow",
    "network_egress": "yellow",
    "github_pr_create": "yellow",
    "repo_push": "yellow",
    "tool_execution": "yellow",
    "paid_action": "red",
    "external_account": "red",
    "external_delete": "red",
    "irreversible": "red",
    "cost": "red",
    # Short action names used by CLI/tests and integrations.
    "sandbox": "green",
    "cache": "green",
    "registry": "green",
    "capability_graph": "green",
    "failure_ledger": "green",
    "forge": "green",
    "eval": "green",
    "memory": "green",
    "network": "yellow",
    "github": "yellow",
    "push": "yellow",
    "cost_usd": "red",
    "external": "red",
    "delete": "red",
    "read": "green",
    "write": "yellow",
    "sandbox_test": "green",
}


class Charter:
    """Resolve action types to autonomy tiers and audit yellow actions."""

    def __init__(self, state_dir: Path | str, *, config_path: Path | str | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path) if config_path else self.state_dir / "charter.yaml"
        self.log_path = self.state_dir / "charter.jsonl"
        self.tiers = dict(DEFAULT_TIERS)
        self._load_overrides()

    def _load_overrides(self) -> None:
        if not self.config_path.exists():
            return
        text = self.config_path.read_text(encoding="utf-8")
        data: Any
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _parse_simple_yaml(text)
        if not isinstance(data, dict):
            return
        for tier in ("green", "yellow", "red"):
            actions = data.get(tier, [])
            if isinstance(actions, dict):
                for action, configured_tier in actions.items():
                    if str(configured_tier).lower() in {"green", "yellow", "red"}:
                        self.tiers[str(action)] = str(configured_tier).lower()  # type: ignore[assignment]
                continue
            if isinstance(actions, str):
                actions = [actions]
            if isinstance(actions, list):
                for action in actions:
                    self.tiers[str(action)] = tier  # type: ignore[assignment]
        # Also accept the compact mapping form: {forge_test: yellow}.
        for action, configured_tier in data.items():
            if str(configured_tier).lower() in {"green", "yellow", "red"}:
                self.tiers[str(action)] = str(configured_tier).lower()  # type: ignore[assignment]

    def evaluate(self, action_type: str) -> Tier:
        """Return the configured tier; unknown actions fail closed as red."""
        return self.tiers.get(str(action_type), "red")

    def log(self, action_type: str, *, tier: Tier | None = None, detail: str = "") -> None:
        resolved = tier or self.evaluate(action_type)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "action_type": action_type,
                        "tier": resolved,
                        "detail": detail,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def allows(self, action_type: str) -> bool:
        tier = self.evaluate(action_type)
        if tier == "yellow":
            self.log(action_type, tier=tier, detail="notify-after; continued")
            return True
        if tier == "red":
            self.log(action_type, tier=tier, detail="halted")
            return False
        return True


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small list-valued YAML subset used by charter.yaml.

    JSON is accepted too; this fallback avoids making PyYAML a runtime
    dependency for a config file containing only ``tier: [action, ...]`` lists.
    """
    result: dict[str, Any] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            current = line[:-1].strip()
            result.setdefault(current, [])
            continue
        if ":" in line and not line.startswith("-"):
            key, value = line.split(":", 1)
            current = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                result[current] = [
                    item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()
                ]
            elif value:
                result[current] = value.strip("'\"")
            else:
                result[current] = []
            continue
        if line.startswith("-") and current:
            if not isinstance(result.get(current), list):
                result[current] = []
            result[current].append(line[1:].strip().strip("'\""))
    return result
