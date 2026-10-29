"""SQLite persistence for request-level pipeline traces."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("TRACE_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "traces.db"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def initialize() -> None:
    with _connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            question TEXT NOT NULL,
            intent TEXT,
            rewritten_query TEXT,
            retrieved_chunks TEXT NOT NULL DEFAULT '[]',
            sql TEXT,
            repairs INTEGER NOT NULL DEFAULT 0,
            node_latency_ms TEXT NOT NULL DEFAULT '[]',
            total_latency_ms REAL NOT NULL DEFAULT 0,
            grounded INTEGER NOT NULL DEFAULT 0
        )""")


def save_trace(question: str, result: dict[str, Any], total_ms: float) -> None:
    initialize()
    trace = result.get("trace") or []
    repairs = sum(1 for item in trace if item.get("node") == "sql_repair")
    with _connect() as con:
        con.execute("""INSERT INTO traces
            (question, intent, rewritten_query, retrieved_chunks, sql, repairs,
             node_latency_ms, total_latency_ms, grounded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (question, result.get("intent"), result.get("rewritten"),
             json.dumps(result.get("chunks", []), ensure_ascii=False), result.get("sql"),
             repairs, json.dumps(trace, ensure_ascii=False), total_ms,
             int(bool(result.get("grounded", False)))))


def recent_traces(limit: int = 50) -> list[dict[str, Any]]:
    initialize()
    bounded = max(1, min(int(limit), 500))
    with _connect() as con:
        rows = con.execute("SELECT * FROM traces ORDER BY id DESC LIMIT ?", (bounded,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        for key in ("retrieved_chunks", "node_latency_ms"):
            item[key] = json.loads(item[key])
        item["grounded"] = bool(item["grounded"])
        result.append(item)
    return result
