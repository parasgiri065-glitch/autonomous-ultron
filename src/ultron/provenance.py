"""Small, serializable provenance envelopes for data crossing tool boundaries.

The envelope is deliberately boring: provenance is metadata, not a second data
model. The payload remains the exact value produced by the user, tool, cache, web
fetch, or LLM, while the source id and timestamp make the boundary auditable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .cache import make_key

Origin = Literal["user", "sandbox_tool", "cache", "web_fetch", "local_file", "llm_generated"]


@dataclass(slots=True)
class ProvenanceEnvelope:
    """A value plus its origin, stable source identifier, and trust state."""

    origin: Origin
    source_id: str
    timestamp: float = field(default_factory=time.time)
    verified: bool = False
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        origin: Origin,
        source_id: str,
        payload: Any,
        *,
        verified: bool = False,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceEnvelope:
        return cls(
            origin=origin,
            source_id=source_id,
            timestamp=time.time() if timestamp is None else float(timestamp),
            verified=bool(verified),
            payload=payload,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def user_input(cls, payload: Any, *, source_id: str | None = None) -> ProvenanceEnvelope:
        return cls.create(
            "user",
            source_id or f"user:{make_key(payload)[:32]}",
            payload,
            verified=True,
        )

    @classmethod
    def tool_output(
        cls,
        tool_key: str,
        content_hash: str,
        payload: Any,
        *,
        verified: bool = False,
        origin: Literal["sandbox_tool", "web_fetch", "local_file"] = "sandbox_tool",
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceEnvelope:
        """Tag output with both the manifest key and its content hash."""
        source_id = f"{tool_key}#{content_hash}"
        details = {"tool_key": tool_key, "content_hash": content_hash, **(metadata or {})}
        return cls.create(origin, source_id, payload, verified=verified, metadata=details)

    @classmethod
    def cache_hit(
        cls,
        cache_key: str,
        payload: Any,
        *,
        verified: bool = False,
        upstream: list[ProvenanceEnvelope] | None = None,
    ) -> ProvenanceEnvelope:
        metadata: dict[str, Any] = {"cache_key": cache_key}
        if upstream:
            metadata["upstream"] = [item.as_dict() for item in upstream]
        return cls.create("cache", cache_key, payload, verified=verified, metadata=metadata)

    @classmethod
    def llm_output(cls, source_id: str, payload: Any) -> ProvenanceEnvelope:
        return cls.create("llm_generated", source_id, payload, verified=False)

    @property
    def data(self) -> Any:
        """Alias for callers that call the wrapped value ``data``."""
        return self.payload

    def with_verified(self, verified: bool = True) -> ProvenanceEnvelope:
        return replace(self, verified=bool(verified))

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "verified": self.verified,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProvenanceEnvelope:
        origin = value.get("origin")
        if origin not in {
            "user",
            "sandbox_tool",
            "cache",
            "web_fetch",
            "local_file",
            "llm_generated",
        }:
            raise ValueError(f"unknown provenance origin: {origin!r}")
        return cls(
            origin=origin,
            source_id=str(value.get("source_id", "")),
            timestamp=float(value.get("timestamp", 0.0)),
            verified=bool(value.get("verified", False)),
            payload=value.get("payload"),
            metadata=dict(value.get("metadata") or {}),
        )


def serialize_envelopes(items: list[ProvenanceEnvelope]) -> list[dict[str, Any]]:
    return [item.as_dict() for item in items]


def deserialize_envelopes(items: Any) -> list[ProvenanceEnvelope]:
    if not isinstance(items, list):
        return []
    out: list[ProvenanceEnvelope] = []
    for item in items:
        if isinstance(item, dict):
            try:
                out.append(ProvenanceEnvelope.from_dict(item))
            except (TypeError, ValueError):
                continue
    return out
