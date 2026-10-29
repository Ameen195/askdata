"""Load every CSV in data/raw into a same-named DuckDB table."""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "olist.duckdb"


def load_olist(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> int:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for csv_path in files:
            table = re.sub(r"\W+", "_", csv_path.stem).strip("_").lower()
            escaped = str(csv_path).replace("'", "''")
            con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM read_csv_auto(\'{escaped}\', header=true)')
    finally:
        con.close()
    return len(files)


if __name__ == "__main__":
    print(f"Loaded {load_olist()} CSV files into {DB_PATH}")
