"""URL-keyed page cache for the network tools, used **only** in live mode.

Why this exists
---------------
The harness caches a whole tool *envelope* keyed by (tool, inputs). That handles
"same question twice", but not "two different questions that fetch the same
page": a search for A and a search for B can both pull ``example.org/report``,
and the second one should not pay for it again. This cache is keyed by URL.

Contract
--------
* It speaks the **same SQLite schema** as :mod:`ultron.cache` (namespace
  ``web``), so the harness can inspect it with the ordinary ``ultron cache``
  tooling and both writers agree on what the table looks like.
* The key is the URL itself (fragment stripped, as a fragment never changes the
  bytes a server returns). Storing it verbatim keeps ``select key from cache
  where ns='web'`` readable during review, and a URL is not a secret.
* TTL comes from ``Settings.cache_ttl_web`` (``ULTRON_CACHE_TTL_WEB``), passed in
  by the sandbox as an env var; ``ttl <= 0`` disables caching entirely.
* Everything here is stdlib on purpose: this module is imported **inside the
  tool sandbox**, which has no third-party packages beyond what the image ships.

It is never used in fixture (mock) mode — those documents are already local and
deterministic — and never when live mode is off, so the default posture is
unchanged.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

NS = "web"

#: Mirrors ``ultron.cache.SCHEMA`` so either process may create the file first.
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

_DEFAULT_TTL_S = 900


def live_enabled() -> bool:
    """True only when the harness explicitly opted into live egress."""
    return os.environ.get("ULTRON_EVAL_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_url(url: str) -> str:
    """Stable cache key for a URL: drop the fragment, keep everything else."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def cache_path_from_env() -> str:
    return os.environ.get("ULTRON_WEB_CACHE", "").strip()


def ttl_from_env(default: int = _DEFAULT_TTL_S) -> int:
    raw = os.environ.get("ULTRON_CACHE_TTL_WEB", "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


class WebCache:
    """URL -> page text, with a TTL. Opens one short-lived connection per call.

    Short-lived connections (like ``ultron.cache``) keep this usable from a
    process that may exit at any moment, and keep the tool free of teardown
    requirements. Failure to open the cache is never fatal: a tool that cannot
    cache still works, it just costs more.
    """

    def __init__(self, path: str, ttl_s: int) -> None:
        self.path = path
        self.ttl_s = int(ttl_s)
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return conn

    def _record(self, conn: sqlite3.Connection, hit: bool) -> None:
        conn.execute(
            "INSERT INTO cache_events (ts, run_id, ns, hit) VALUES (?, ?, ?, ?)",
            (time.time(), os.environ.get("ULTRON_RUN_ID") or None, NS, 1 if hit else 0),
        )

    # --------------------------------------------------------------------- api
    def get(self, url: str) -> str | None:
        if self.ttl_s <= 0:
            return None
        key = normalize_url(url)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE ns=? AND key=?", (NS, key)
                ).fetchone()
                if row is None:
                    self.misses += 1
                    self._record(conn, hit=False)
                    return None
                if row["expires_at"] is not None and row["expires_at"] < time.time():
                    conn.execute("DELETE FROM cache WHERE ns=? AND key=?", (NS, key))
                    self.misses += 1
                    self._record(conn, hit=False)
                    return None
                conn.execute(
                    "UPDATE cache SET hits = hits + 1, last_hit_at = ? WHERE ns=? AND key=?",
                    (time.time(), NS, key),
                )
                self._record(conn, hit=True)
                self.hits += 1
                try:
                    # The harness stores JSON in this column; writing plain text
                    # would make the row unreadable to `ultron cache get web ...`.
                    return str(json.loads(row["value"]))
                except (TypeError, ValueError):  # pragma: no cover - corrupted row
                    return None
        except sqlite3.Error:
            self.misses += 1
            return None

    def set(self, url: str, text: str, *, ttl_s: int | None = None) -> None:
        ttl = self.ttl_s if ttl_s is None else int(ttl_s)
        if ttl <= 0:
            return
        key = normalize_url(url)
        now = time.time()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO cache (ns, key, value, created_at, expires_at, hits)
                    VALUES (?, ?, ?, ?, ?, 0)
                    ON CONFLICT(ns, key) DO UPDATE SET
                        value = excluded.value,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at
                    """,
                    (NS, key, json.dumps(text), now, now + ttl),
                )
        except sqlite3.Error:
            return

    def stats(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "path": self.path, "ttl_s": self.ttl_s}


def maybe_cache() -> WebCache | None:
    """Build a cache if live mode is on and a path was handed to us."""
    if not live_enabled():
        return None
    path = cache_path_from_env()
    if not path:
        return None
    try:
        cache = WebCache(path, ttl_from_env())
        with cache._connect():  # fail fast on an unusable path, before any fetch
            pass
    except (sqlite3.Error, OSError):
        return None
    return cache
