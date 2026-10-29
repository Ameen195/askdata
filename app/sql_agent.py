"""Schema loading, SQL validation (sqlglot) and read-only execution on DuckDB."""
import os
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

DB_PATH = os.getenv("DUCKDB_PATH", "data/olist.duckdb")
_BANNED = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Command)


def _con():
    db_path = Path(DB_PATH)
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))
    return duckdb.connect(str(db_path), read_only=True)


def get_schema() -> str:
    rows = _con().execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema='main' ORDER BY table_name, ordinal_position").fetchall()
    tables = {}
    for t, c, d in rows:
        tables.setdefault(t, []).append(f"{c} {d}")
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items())


def get_samples(n: int = 3) -> str:
    con, out = _con(), []
    for (t,) in con.execute("SHOW TABLES").fetchall():
        out.append(f"-- {t}\n{con.execute(f'SELECT * FROM {t} LIMIT {n}').fetchdf().to_string()}")
    return "\n".join(out)


def validate_sql(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="duckdb")
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT statements are allowed")
    if any(tree.find(b) for b in _BANNED):
        raise ValueError("Forbidden statement type")
    if isinstance(tree, exp.Select) and not tree.args.get("limit"):
        tree = tree.limit(100)
    return tree.sql(dialect="duckdb")


def run_sql(sql: str) -> list[dict]:
    return _con().execute(sql).fetchdf().to_dict("records")
