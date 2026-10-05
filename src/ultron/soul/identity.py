"""Persistent, operator-controlled soul identity."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings

DEFAULT_PROFILE: dict[str, Any] = {
    "version": 1,
    "identity": {"name": "Ultron", "user": "", "mission": "be useful without fabrication"},
    "preferences": {},
    "assets": [],
    "lore": [],
}


class SoulIdentity:
    """Load and atomically persist the user's identity/profile."""

    def __init__(self, settings: Settings | None = None, *, path: Path | str | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path or self.settings.state_dir / "soul_profile.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _copy_profile(DEFAULT_PROFILE)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _copy_profile(DEFAULT_PROFILE)
        if not isinstance(data, dict):
            return _copy_profile(DEFAULT_PROFILE)
        return _merge_profile(DEFAULT_PROFILE, data)

    def reload(self) -> dict[str, Any]:
        self.profile = self._load()
        return self.snapshot()

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="soul-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.profile, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.path

    def update(self, values: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
        if not isinstance(values, dict):
            raise TypeError("soul profile update must be an object")
        self.profile = _merge_profile(self.profile, values)
        if persist:
            self.save()
        return self.snapshot()

    def set_identity(self, **values: Any) -> dict[str, Any]:
        return self.update({"identity": values})

    def add_preference(self, key: str, value: Any) -> dict[str, Any]:
        preferences = dict(self.profile.get("preferences") or {})
        preferences[str(key)] = value
        return self.update({"preferences": preferences})

    def add_lore(self, entry: str) -> dict[str, Any]:
        lore = list(self.profile.get("lore") or [])
        lore.append(str(entry))
        return self.update({"lore": lore})

    def context(self) -> dict[str, Any]:
        """Return only grounding context, never an instruction to override policy."""
        identity = dict(self.profile.get("identity") or {})
        return {
            "identity": identity,
            "preferences": dict(self.profile.get("preferences") or {}),
            "assets": list(self.profile.get("assets") or []),
            "lore": list(self.profile.get("lore") or []),
        }

    grounding = context

    def ground_goal(self, goal: str) -> str:
        identity = self.context().get("identity", {})
        user = identity.get("user") or identity.get("name")
        if not user:
            return goal
        return f"For {user}: {goal}"

    def snapshot(self) -> dict[str, Any]:
        return _copy_profile(self.profile)

    def inspect(self) -> dict[str, Any]:
        return {"path": str(self.path), "profile": self.snapshot()}

    def reset(self) -> Path:
        self.profile = _copy_profile(DEFAULT_PROFILE)
        return self.save()


def _copy_profile(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=str))


def _merge_profile(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = _copy_profile(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_profile(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = ["DEFAULT_PROFILE", "SoulIdentity"]
