"""SQLite episodic memory and a compact semantic knowledge graph."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    goal_tokens TEXT NOT NULL,
    trajectory TEXT NOT NULL,
    result TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '{}',
    verified INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_soul_episode_goal ON episodes(goal);
CREATE INDEX IF NOT EXISTS idx_soul_episode_verified ON episodes(verified, created_at);
CREATE TABLE IF NOT EXISTS semantic_nodes (
    node_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    value TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_soul_semantic_label ON semantic_nodes(kind, label);
CREATE TABLE IF NOT EXISTS semantic_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1,
    PRIMARY KEY(source_id, target_id, relation)
);
"""


class MemoryEngine:
    """Persist successful trajectories and consolidate their facts."""

    def __init__(self, path: Path | str | None = None, *, settings: Any | None = None) -> None:
        if path is None:
            if settings is None:
                raise TypeError("MemoryEngine requires a database path or settings")
            path = Path(settings.state_dir) / "soul_memory.db"
        candidate = Path(path)
        self.path = (
            candidate / "soul_memory.db" if candidate.exists() and candidate.is_dir() else candidate
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def record_episode(
        self,
        goal: str,
        trajectory: Iterable[str] | None = None,
        result: dict[str, Any] | None = None,
        *,
        verified: bool = False,
        context: dict[str, Any] | None = None,
        answer: str = "",
    ) -> str:
        episode_id = f"episode-{uuid.uuid4().hex[:12]}"
        trajectory_value = list(trajectory or [])
        result_value = dict(result or {})
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO episodes
                   (episode_id, goal, goal_tokens, trajectory, result, answer, context, verified, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    episode_id,
                    str(goal),
                    " ".join(sorted(_tokens(goal))),
                    json.dumps(trajectory_value, default=str, sort_keys=True),
                    json.dumps(result_value, default=str, sort_keys=True),
                    str(answer or ""),
                    json.dumps(context or {}, default=str, sort_keys=True),
                    int(verified),
                    time.time(),
                ),
            )
        if verified:
            self.consolidate(episode_id)
        return episode_id

    remember = record_episode

    def recall(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        *,
        min_overlap: float = 1.0,
    ) -> dict[str, Any] | None:
        tokens = _tokens(goal)
        if not tokens:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM episodes WHERE verified=1 ORDER BY created_at DESC LIMIT 100"""
            ).fetchall()
        best: sqlite3.Row | None = None
        best_overlap = 0.0
        for row in rows:
            prior = set(str(row["goal_tokens"]).split())
            overlap = len(tokens & prior) / max(len(tokens | prior), 1)
            if overlap >= min_overlap and overlap >= best_overlap:
                best = row
                best_overlap = overlap
        if best is None:
            return None
        return {
            "episode_id": best["episode_id"],
            "run_id": best["episode_id"],
            "goal": best["goal"],
            "trajectory": _json_value(best["trajectory"], []),
            "result": _json_value(best["result"], {}),
            "outputs": _json_value(best["result"], {}),
            "answer": best["answer"],
            "context": _json_value(best["context"], {}),
            "overlap": best_overlap,
            "age_s": time.time() - float(best["created_at"]),
        }

    def consolidate(self, episode_id: str | None = None) -> int:
        with self._connect() as conn:
            query = "SELECT * FROM episodes WHERE verified=1"
            params: tuple[Any, ...] = ()
            if episode_id:
                query += " AND episode_id=?"
                params = (episode_id,)
            episodes = conn.execute(query, params).fetchall()
            count = 0
            for episode in episodes:
                episode_node = self._node(
                    conn, "episode", episode["episode_id"], episode["episode_id"]
                )
                result = _json_value(episode["result"], {})
                for key, value in result.items() if isinstance(result, dict) else ():
                    label = f"{key}:{_scalar(value)}"
                    fact = self._node(conn, "fact", label, value)
                    conn.execute(
                        """INSERT INTO semantic_edges(source_id,target_id,relation,weight)
                           VALUES(?,?,?,1) ON CONFLICT(source_id,target_id,relation)
                           DO UPDATE SET weight=weight+1""",
                        (episode_node, fact, "contains"),
                    )
                    count += 1
                for capability in _json_value(episode["trajectory"], []):
                    capability_node = self._node(conn, "capability", str(capability), capability)
                    conn.execute(
                        """INSERT INTO semantic_edges(source_id,target_id,relation,weight)
                           VALUES(?,?,?,1) ON CONFLICT(source_id,target_id,relation)
                           DO UPDATE SET weight=weight+1""",
                        (episode_node, capability_node, "used"),
                    )
            return count

    def reflect(self, query: str | None = None, *, limit: int = 20) -> dict[str, Any]:
        """Return deterministic semantic reflection, suitable for grounding."""
        with self._connect() as conn:
            if query:
                tokens = _tokens(query)
                rows = conn.execute(
                    "SELECT kind,label,value,weight FROM semantic_nodes ORDER BY weight DESC, updated_at DESC"
                ).fetchall()
                selected = [row for row in rows if tokens & _tokens(str(row["label"]))]
            else:
                selected = conn.execute(
                    "SELECT kind,label,value,weight FROM semantic_nodes ORDER BY weight DESC, updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            edges = conn.execute(
                "SELECT source_id,target_id,relation,weight FROM semantic_edges ORDER BY weight DESC LIMIT ?",
                (limit * 2,),
            ).fetchall()
        return {
            "query": query,
            "nodes": [dict(row) for row in selected[:limit]],
            "edges": [dict(row) for row in edges],
        }

    reflection = reflect

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            episodes = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(verified),0) AS v FROM episodes"
            ).fetchone()
            nodes = conn.execute("SELECT COUNT(*) AS n FROM semantic_nodes").fetchone()
            edges = conn.execute("SELECT COUNT(*) AS n FROM semantic_edges").fetchone()
        return {
            "episodes": int(episodes["n"]),
            "verified_episodes": int(episodes["v"]),
            "semantic_nodes": int(nodes["n"]),
            "semantic_edges": int(edges["n"]),
        }

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT episode_id,goal,trajectory,verified,created_at FROM episodes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def forget(self, *, episode_id: str | None = None) -> int:
        with self._connect() as conn:
            if episode_id:
                cursor = conn.execute("DELETE FROM episodes WHERE episode_id=?", (episode_id,))
            else:
                cursor = conn.execute("DELETE FROM episodes")
            conn.execute("DELETE FROM semantic_edges")
            conn.execute("DELETE FROM semantic_nodes")
        return cursor.rowcount

    @staticmethod
    def _node(conn: sqlite3.Connection, kind: str, label: str, value: Any) -> str:
        existing = conn.execute(
            "SELECT node_id FROM semantic_nodes WHERE kind=? AND label=?", (kind, label)
        ).fetchone()
        now = time.time()
        if existing:
            conn.execute(
                "UPDATE semantic_nodes SET weight=weight+1,value=?,updated_at=? WHERE node_id=?",
                (json.dumps(value, default=str), now, existing["node_id"]),
            )
            return str(existing["node_id"])
        node_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        conn.execute(
            "INSERT INTO semantic_nodes(node_id,kind,label,value,weight,updated_at) VALUES(?,?,?,?,1,?)",
            (node_id, kind, label, json.dumps(value, default=str), now),
        )
        return node_id


def _tokens(value: str) -> set[str]:
    return {item for item in re.split(r"[^a-z0-9]+", str(value).casefold()) if len(item) > 2}


def _scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _json_value(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return default


__all__ = ["MemoryEngine"]
