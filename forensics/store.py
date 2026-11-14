"""Trace storage: full traces as JSON files, metadata indexed in SQLite.

traces/
  index.db                      <- queryable metadata (traces + spans tables)
  2026-11-14/<trace_id>.json    <- complete, human-readable trace
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .tracing import Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    pipeline     TEXT NOT NULL,
    doc_id       TEXT,
    status       TEXT NOT NULL,
    final_score  REAL,
    failed_step  TEXT,
    error        TEXT,
    total_tokens INTEGER,
    latency_ms   REAL,
    model        TEXT,
    json_path    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    span_id      TEXT PRIMARY KEY,
    trace_id     TEXT NOT NULL REFERENCES traces(trace_id),
    name         TEXT NOT NULL,
    status       TEXT NOT NULL,
    confidence   INTEGER,
    latency_ms   REAL,
    tokens       INTEGER,
    error_type   TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name, confidence);
"""


class TraceStore:
    def __init__(self, root: str | Path = "traces"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, trace: Trace) -> Path:
        rel_path = Path(trace.started_at.strftime("%Y-%m-%d")) / f"{trace.trace_id}.json"
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")

        doc_id = trace.input.get("doc_id") if isinstance(trace.input, dict) else None
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trace.trace_id, trace.started_at.isoformat(), trace.pipeline, doc_id,
                    trace.status, trace.final_score, trace.failed_step, trace.error,
                    trace.total_tokens, trace.latency_ms, trace.metadata.get("model"), str(rel_path),
                ),
            )
            conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
            conn.executemany(
                "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)",
                [
                    (s.span_id, s.trace_id, s.name, s.status, s.confidence, s.latency_ms, s.tokens,
                     s.error.type if s.error else None)
                    for s in trace.spans
                ],
            )
        return path

    def load(self, trace_id: str) -> Trace:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT json_path FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            raise KeyError(trace_id)
        return Trace.model_validate_json((self.root / row["json_path"]).read_text(encoding="utf-8"))

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
