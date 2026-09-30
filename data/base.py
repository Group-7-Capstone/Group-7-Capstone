"""Shared local-database access for the data layer.

All modules in this layer (download.py, reader.py) read and write the
same local DuckDB file through this module, so the backend can be
swapped later (e.g. to Postgres or GCP) without touching callers.
"""

from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent / "raw" / "nyc_open_data.duckdb"

PROVENANCE_TABLE = "_provenance"


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
    return con
