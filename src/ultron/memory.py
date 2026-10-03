"""Episodic memory + the metrics the eval harness reports on.

Everything is SQLite (no vector DB, no embedding spend). Retrieval is a
deterministic token-overlap match against prior runs, which is enough for the
Phase 1 cost trick that matters most: **an identical goal against an identical
registry is recalled, not recomputed**. Recall returns the previous final answer
at zero token cost, and the recall itself is recorded so the metrics stay honest.

Notable rule: only *verified* runs are written as successes. A failed or
unverified run is stored but is never offered by :meth:`Memory.recall`.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import make_key
from .config import Settings, get_settings

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DENIED = "denied"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    goal                TEXT NOT NULL,
    goal_digest         TEXT NOT NULL,
    goal_tokens         TEXT NOT NULL,
    registry_fingerprint TEXT NOT NULL,
    difficulty          TEXT,
    plan_depth          INTEGER,
    status              TEXT NOT NULL,
    success             INTEGER NOT NULL,
    verified            INTEGER NOT NULL DEFAULT 0,
    steps_planned       INTEGER DEFAULT 0,
    steps_executed      INTEGER DEFAULT 0,
    cost_usd            REAL DEFAULT 0,
    latency_s           REAL DEFAULT 0,
    cache_hits          INTEGER DEFAULT 0,
    cache_misses        INTEGER DEFAULT 0,
    llm_calls           INTEGER DEFAULT 0,
    answer              TEXT,
    error               TEXT,
    created_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_digest ON runs (goal_digest, registry_fingerprint);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at);

CREATE TABLE IF NOT EXISTS steps (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    step_index     INTEGER NOT NULL,
    tool           TEXT NOT NULL,
    version        TEXT NOT NULL,
    inputs_digest  TEXT,
    inputs_json    TEXT,
    risk           TEXT,
    policy_action  TEXT,
    network        TEXT,
    ok             INTEGER NOT NULL,
    cached         INTEGER NOT NULL DEFAULT 0,
    duration_s     REAL DEFAULT 0,
    cost_usd       REAL DEFAULT 0,
    verified       INTEGER DEFAULT 0,
    error          TEXT,
    created_at     REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps (run_id);

CREATE TABLE IF NOT EXISTS tool_stats (
    tool           TEXT NOT NULL,
    version        TEXT NOT NULL,
    calls          INTEGER NOT NULL DEFAULT 0,
    successes      INTEGER NOT NULL DEFAULT 0,
    failures       INTEGER NOT NULL DEFAULT 0,
    denials        INTEGER NOT NULL DEFAULT 0,
    cache_hits     INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    total_ms       REAL NOT NULL DEFAULT 0,
    last_used_at   REAL,
    PRIMARY KEY (tool, version)
);

CREATE TABLE IF NOT EXISTS notes (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


@dataclass(slots=True)
class RunRecord:
    run_id: str
    goal: str
    status: str
    success: bool
    verified: bool
    difficulty: str = ""
    plan_depth: int = 0
    steps_planned: int = 0
    steps_executed: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    answer: str = ""
    error: str = ""


@dataclass(slots=True)
class MemoryStats:
    runs: int = 0
    successes: int = 0
    success_rate: float = 0.0
    avg_cost_usd: float = 0.0
    avg_latency_s: float = 0.0
    avg_steps: float = 0.0
    cache_hit_rate: float = 0.0
    total_cost_usd: float = 0.0
    by_tool: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "avg_latency_s": round(self.avg_latency_s, 4),
            "avg_steps": round(self.avg_steps, 3),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "by_tool": self.by_tool,
        }


class Memory:
    """SQLite episodic memory for runs, steps, tool statistics and notes."""

    def __init__(self, settings: Settings | None = None, *, path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path or self.settings.memory_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    # -------------------------------------------------------------------- runs
    @staticmethod
    def new_run_id() -> str:
        return f"r-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def start_run(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        registry_fingerprint: str = "",
        difficulty: str = "",
        plan_depth: int = 0,
    ) -> str:
        run_id = run_id or self.new_run_id()
        tokens = " ".join(sorted(_tokens(goal)))
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, goal, goal_digest, goal_tokens, registry_fingerprint, difficulty,
                    plan_depth, status, success, verified, created_at)
                   VALUES (?,?,?,?,?,?,?,?,0,0,?)""",
                (
                    run_id,
                    goal,
                    make_key("goal", goal.strip().lower())[:32],
                    tokens,
                    registry_fingerprint,
                    difficulty,
                    plan_depth,
                    "running",
                    time.time(),
                ),
            )
        return run_id

    def finish_run(self, record: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs SET status=?, success=?, verified=?, steps_planned=?, steps_executed=?,
                   cost_usd=?, latency_s=?, cache_hits=?, cache_misses=?, llm_calls=?, answer=?, error=?
                   WHERE run_id=?""",
                (
                    record.status,
                    int(record.success),
                    int(record.verified),
                    record.steps_planned,
                    record.steps_executed,
                    record.cost_usd,
                    record.latency_s,
                    record.cache_hits,
                    record.cache_misses,
                    record.llm_calls,
                    (record.answer or "")[:8000],
                    (record.error or "")[:2000],
                    record.run_id,
                ),
            )

    def record_step(
        self,
        run_id: str,
        *,
        step_index: int,
        tool: str,
        version: str,
        inputs: dict[str, Any],
        risk: str = "",
        policy_action: str = "",
        network: str = "none",
        ok: bool = False,
        cached: bool = False,
        duration_s: float = 0.0,
        cost_usd: float = 0.0,
        verified: bool = False,
        error: str = "",
    ) -> None:
        digest = make_key("inputs", inputs)[:32]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO steps (run_id, step_index, tool, version, inputs_digest, inputs_json,
                   risk, policy_action, network, ok, cached, duration_s, cost_usd, verified, error, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    step_index,
                    tool,
                    version,
                    digest,
                    json.dumps(inputs, default=str)[:4000],
                    risk,
                    policy_action,
                    network,
                    int(ok),
                    int(cached),
                    duration_s,
                    cost_usd,
                    int(verified),
                    (error or "")[:1000],
                    time.time(),
                ),
            )
            self._bump_tool_stats(
                conn,
                tool,
                version,
                ok=ok,
                cached=cached,
                denied=policy_action == "deny",
                cost_usd=cost_usd,
                duration_ms=duration_s * 1000,
            )

    @staticmethod
    def _bump_tool_stats(
        conn: sqlite3.Connection,
        tool: str,
        version: str,
        *,
        ok: bool,
        cached: bool,
        denied: bool,
        cost_usd: float,
        duration_ms: float,
    ) -> None:
        conn.execute(
            """INSERT INTO tool_stats (tool, version, calls, successes, failures, denials, cache_hits,
                                       total_cost_usd, total_ms, last_used_at)
               VALUES (?,?,1,?,?,?,?,?,?,?)
               ON CONFLICT(tool, version) DO UPDATE SET
                 calls = calls + 1,
                 successes = successes + excluded.successes,
                 failures = failures + excluded.failures,
                 denials = denials + excluded.denials,
                 cache_hits = cache_hits + excluded.cache_hits,
                 total_cost_usd = total_cost_usd + excluded.total_cost_usd,
                 total_ms = total_ms + excluded.total_ms,
                 last_used_at = excluded.last_used_at""",
            (
                tool,
                version,
                int(ok),
                int(not ok and not denied),
                int(denied),
                int(cached),
                cost_usd,
                duration_ms,
                time.time(),
            ),
        )

    # ------------------------------------------------------------------ recall
    def recall(
        self, goal: str, registry_fingerprint: str, *, min_overlap: float = 1.0
    ) -> dict[str, Any] | None:
        """Return a previously verified answer for an equivalent goal, if any.

        ``min_overlap=1.0`` means "exactly the same goal tokens" — conservative on
        purpose: a near-miss recall could serve stale facts. Lower it only with a
        freshness policy in place (Phase 2).
        """
        tokens = _tokens(goal)
        if not tokens:
            return None
        digest = make_key("goal", goal.strip().lower())[:32]
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT run_id, goal, goal_tokens, answer, cost_usd, latency_s, success, verified, created_at
                   FROM runs
                   WHERE goal_digest=? AND registry_fingerprint=? AND success=1 AND verified=1
                   ORDER BY created_at DESC LIMIT 5""",
                (digest, registry_fingerprint),
            ).fetchall()
        for row in rows:
            prior = set((row["goal_tokens"] or "").split())
            overlap = len(tokens & prior) / max(len(tokens | prior), 1)
            if overlap >= min_overlap and row["answer"]:
                return {
                    "run_id": row["run_id"],
                    "answer": row["answer"],
                    "age_s": time.time() - row["created_at"],
                    "prior_cost_usd": row["cost_usd"],
                    "prior_latency_s": row["latency_s"],
                    "overlap": overlap,
                }
        return None

    # ------------------------------------------------------------------- stats
    def stats(self, *, since: float | None = None) -> MemoryStats:
        where = "WHERE created_at >= ?" if since else ""
        params: tuple[Any, ...] = (since,) if since else ()
        with self._connect() as conn:
            row = conn.execute(
                f"""SELECT COUNT(*) AS runs, COALESCE(SUM(success),0) AS successes,
                           COALESCE(AVG(cost_usd),0) AS avg_cost, COALESCE(AVG(latency_s),0) AS avg_latency,
                           COALESCE(AVG(steps_executed),0) AS avg_steps,
                           COALESCE(SUM(cache_hits),0) AS hits, COALESCE(SUM(cache_misses),0) AS misses,
                           COALESCE(SUM(cost_usd),0) AS total_cost
                    FROM runs {where}""",
                params,
            ).fetchone()
            tools = conn.execute(
                """SELECT tool, version, calls, successes, failures, denials, cache_hits,
                          total_cost_usd, total_ms
                   FROM tool_stats ORDER BY calls DESC"""
            ).fetchall()
        runs = int(row["runs"] or 0)
        hits, misses = int(row["hits"] or 0), int(row["misses"] or 0)
        return MemoryStats(
            runs=runs,
            successes=int(row["successes"] or 0),
            success_rate=(int(row["successes"] or 0) / runs) if runs else 0.0,
            avg_cost_usd=float(row["avg_cost"] or 0.0),
            avg_latency_s=float(row["avg_latency"] or 0.0),
            avg_steps=float(row["avg_steps"] or 0.0),
            cache_hit_rate=(hits / (hits + misses)) if (hits + misses) else 0.0,
            total_cost_usd=float(row["total_cost"] or 0.0),
            by_tool=[dict(t) for t in tools],
        )

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT run_id, goal, status, success, verified, cost_usd, latency_s, cache_hits,
                          cache_misses, steps_executed, created_at
                   FROM runs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def forget_runs(self, before: float | None = None) -> int:
        """Prune history (used by CI to keep the eval store small)."""
        with self._connect() as conn:
            if before is None:
                cur = conn.execute("DELETE FROM runs")
                conn.execute("DELETE FROM steps")
            else:
                cur = conn.execute("DELETE FROM runs WHERE created_at < ?", (before,))
                conn.execute("DELETE FROM steps WHERE created_at < ?", (before,))
        return cur.rowcount

    # ------------------------------------------------------------------- notes
    def set_note(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO notes (key, value, updated_at) VALUES (?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key, json.dumps(value, default=str), time.time()),
            )

    def get_note(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM notes WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:  # pragma: no cover
            return default

    # ------------------------------------------------------------------ reports
    def failure_modes(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT tool, error, COUNT(*) AS n FROM steps
                   WHERE ok=0 AND error IS NOT NULL AND error != ''
                   GROUP BY tool, error ORDER BY n DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def _tokens(text: str) -> set[str]:
    import re

    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2}
