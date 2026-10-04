"""SQLite-backed cache: the cost architecture's first line of defence.

Four namespaces, one file, one connection:

  ``router``  cheap-model difficulty classifications
  ``plan``    planner output keyed on (goal, registry fingerprint)
  ``llm``     every raw LLM response (so re-runs of the same prompt cost $0)
  ``web``     HTTP fetches (so we never refetch a page we already paid for)
  ``tool``    sandbox executions keyed on (tool, version, inputs)
  ``judge``   LLM judge verdicts

Rule: a cache **hit skips both execution and the LLM call**. Callers therefore
treat ``CacheEntry.hit is True`` as "do not spend anything".

Concurrency note: SQLite is opened in WAL mode with a busy timeout, and each
call opens/closes its own connection so the cache is usable from threads and
from separate processes (eval runner, CI jobs).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import Settings, get_settings

Namespace = Literal["router", "plan", "llm", "web", "tool", "judge"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    ns          TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    expires_at  REAL,
    hits        INTEGER NOT NULL DEFAULT 0,
    last_hit_at REAL,
    PRIMARY KEY (ns, key)
);
CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache (expires_at);

CREATE TABLE IF NOT EXISTS cache_events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    run_id TEXT,
    ns     TEXT NOT NULL,
    hit    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_events_run ON cache_events (run_id);
"""


def canonical_json(obj: Any) -> str:
    """Deterministic JSON so equal inputs always hash equal."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def make_key(*parts: Any) -> str:
    """Stable content hash for cache keys. Never includes a timestamp."""
    payload = canonical_json(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> str:
    """Hash a file's bytes (used to key the registry into plans)."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


@dataclass(slots=True)
class CacheEntry:
    key: str
    ns: str
    value: Any
    hit: bool
    age_s: float = 0.0


@dataclass(slots=True)
class CacheStats:
    entries: int
    by_ns: dict[str, int]
    hits: int
    misses: int
    hit_rate: float
    bytes_estimate: int
    expiring_soon: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "by_ns": self.by_ns,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "bytes_estimate": self.bytes_estimate,
            "expiring_soon": self.expiring_soon,
        }


class Cache:
    """Namespaced TTL cache with per-run hit/miss accounting."""

    def __init__(self, settings: Settings | None = None, *, run_id: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(self.settings.cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._hits = 0
        self._misses = 0
        self._ensure_schema()

    # ---------------------------------------------------------------- plumbing
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------- reads
    def get(self, ns: Namespace, key: str, *, record: bool = True) -> CacheEntry | None:
        """Return the entry if present and unexpired, else ``None``."""
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, created_at, expires_at FROM cache WHERE ns=? AND key=?", (ns, key)
            ).fetchone()
            if row is None:
                self._miss(ns, record=record)
                return None
            if row["expires_at"] is not None and row["expires_at"] < now:
                conn.execute("DELETE FROM cache WHERE ns=? AND key=?", (ns, key))
                self._miss(ns, record=record)
                return None
            conn.execute(
                "UPDATE cache SET hits = hits + 1, last_hit_at = ? WHERE ns=? AND key=?",
                (now, ns, key),
            )
        self._hit(ns, record=record)
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError:  # pragma: no cover - corrupted row
            return None
        return CacheEntry(key=key, ns=ns, value=value, hit=True, age_s=now - row["created_at"])

    def get_many(self, ns: Namespace, keys: list[str]) -> dict[str, Any]:  # pragma: no cover
        out: dict[str, Any] = {}
        for k in keys:
            entry = self.get(ns, k, record=False)
            if entry is not None:
                out[k] = entry.value
        return out

    # ------------------------------------------------------------------ writes
    def set(
        self,
        ns: Namespace,
        key: str,
        value: Any,
        *,
        ttl_s: int | float | None = None,
    ) -> None:
        now = time.time()
        expires_at = None if ttl_s is None else now + float(ttl_s)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache (ns, key, value, created_at, expires_at, hits, last_hit_at)
                VALUES (?, ?, ?, ?, ?, 0, NULL)
                ON CONFLICT(ns, key) DO UPDATE SET
                    value=excluded.value,
                    created_at=excluded.created_at,
                    expires_at=excluded.expires_at
                """,
                (ns, key, canonical_json(value), now, expires_at),
            )

    def set_many(
        self, ns: Namespace, items: dict[str, Any], *, ttl_s: int | float | None = None
    ) -> None:
        for key, value in items.items():
            self.set(ns, key, value, ttl_s=ttl_s)

    def delete(self, ns: Namespace, key: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM cache WHERE ns=? AND key=?", (ns, key))
        return cur.rowcount > 0

    def clear(self, ns: Namespace | None = None) -> int:
        with self._connect() as conn:
            if ns is None:
                cur = conn.execute("DELETE FROM cache")
            else:
                cur = conn.execute("DELETE FROM cache WHERE ns=?", (ns,))
        return cur.rowcount

    def prune_expired(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?", (time.time(),)
            )
        return cur.rowcount

    # -------------------------------------------------------------- accounting
    def _hit(self, ns: Namespace, *, record: bool) -> None:
        self._hits += 1
        if record:
            self._log_event(ns, True)

    def _miss(self, ns: Namespace, *, record: bool) -> None:
        self._misses += 1
        if record:
            self._log_event(ns, False)

    def _log_event(self, ns: Namespace, hit: bool) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO cache_events (ts, run_id, ns, hit) VALUES (?, ?, ?, ?)",
                    (time.time(), self.run_id, ns, int(hit)),
                )
        except sqlite3.Error:  # pragma: no cover - never fail a run over metrics
            pass

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total else 0.0

    def reset_counters(self) -> None:
        self._hits = 0
        self._misses = 0

    def stats(self) -> CacheStats:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(SUM(LENGTH(value)),0) AS b FROM cache"
            ).fetchone()
            rows = conn.execute("SELECT ns, COUNT(*) AS c FROM cache GROUP BY ns").fetchall()
            soon = conn.execute(
                "SELECT COUNT(*) AS c FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (time.time() + 300,),
            ).fetchone()
        by_ns = {r["ns"]: r["c"] for r in rows}
        return CacheStats(
            entries=int(total["c"]),
            by_ns=by_ns,
            hits=self._hits,
            misses=self._misses,
            hit_rate=self.hit_rate,
            bytes_estimate=int(total["b"]),
            expiring_soon=int(soon["c"]),
        )

    # ------------------------------------------------------------------ helpers
    def ttl_for(self, ns: Namespace) -> int:
        return {
            "llm": self.settings.cache_ttl_llm,
            "router": self.settings.cache_ttl_llm,
            "judge": self.settings.cache_ttl_llm,
            "plan": self.settings.cache_ttl_llm,
            "web": self.settings.cache_ttl_web,
            "tool": self.settings.cache_ttl_tool,
        }[ns]
