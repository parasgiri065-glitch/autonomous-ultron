"""stdin/stdout contract shared by every tool. Keep this dependency-free."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from typing import Any


def read_input() -> dict[str, Any]:
    """Parse the input JSON object from stdin."""
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("tool input must be a JSON object")
    return data


def require(payload: dict[str, Any], field: str, typ: type | tuple[type, ...]) -> Any:
    """Fetch a required input field, failing loudly if it is missing/wrong."""
    if field not in payload:
        raise ValueError(f"missing required input {field!r}")
    value = payload[field]
    if typ is int and isinstance(value, bool):
        raise ValueError(f"input {field!r} must be int, got bool")
    if not isinstance(value, typ):
        raise ValueError(f"input {field!r} must be {typ}, got {type(value).__name__}")
    return value


def optional(payload: dict[str, Any], field: str, typ: type, default: Any) -> Any:
    if field not in payload or payload[field] is None:
        return default
    value = payload[field]
    return value if isinstance(value, typ) else default


def emit_ok(result: dict[str, Any], **meta: Any) -> None:
    _emit({"ok": True, "result": result, "meta": {"tool_pid": True, **meta}})


def emit_error(message: str, **meta: Any) -> None:
    _emit({"ok": False, "result": None, "error": message, "meta": meta})
    sys.exit(1)


def _emit(envelope: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(envelope, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main_guard(fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Run ``fn`` with the standard envelope, timing and failure handling."""
    start = time.perf_counter()
    try:
        payload = read_input()
        result = fn(payload)
    except Exception as exc:
        emit_error(f"{type(exc).__name__}: {exc}")
        return
    emit_ok(result, elapsed_ms=round((time.perf_counter() - start) * 1000, 2))


def text_sentences(text: str) -> list[str]:
    """Very small sentence splitter (no nltk dependency in the sandbox)."""
    out: list[str] = []
    current: list[str] = []
    for chunk in text.replace("\n", " ").split(". "):
        current.append(chunk.strip())
        joined = ". ".join(c for c in current if c)
        if len(joined) > 40:
            out.append(joined)
            current = []
    if current:
        tail = ". ".join(c for c in current if c).strip()
        if tail:
            out.append(tail)
    return out
