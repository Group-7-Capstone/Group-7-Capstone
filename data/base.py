"""Shared local-database access for the data layer.

All modules in this layer (download.py, reader.py) read and write the
same local cache through this module, so the backend can be swapped
later (e.g. to Postgres or GCP) without touching callers.

Raw pulls are stored as one zstd Parquet file per table next to `DB_PATH`.
The DuckDB file only holds `_provenance` and any views built on top of
the raw tables; every connection exposes each Parquet file as a
temporary view named after its table, so SQL can query it by name.
"""

from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent / "raw" / "nyc_open_data.duckdb"

PROVENANCE_TABLE = "_provenance"


def parquet_path(table_name: str) -> Path:
    """Return where a raw table's Parquet file lives."""
    return DB_PATH.parent / f"{table_name}.parquet"


def register_parquet_views(con: duckdb.DuckDBPyConnection) -> None:
    """Expose every raw Parquet file to `con` as a view named after its table."""
    for path in sorted(DB_PATH.parent.glob("*.parquet")):
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW {path.stem} AS "
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        )


def get_connection() -> duckdb.DuckDBPyConnection:
    """Open a connection to the local cache, creating the file if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PROVENANCE_TABLE} (
            table_name VARCHAR,
            dataset_id VARCHAR,
            query VARCHAR,
            pulled_at TIMESTAMP
        )
        """
    )
    register_parquet_views(con)
    return con
