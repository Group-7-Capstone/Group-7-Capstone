"""Return cached raw data as DataFrames. No transformations happen here."""

import pandas as pd

from data.base import get_connection


def read_table(table_name: str) -> pd.DataFrame:
    """Return a raw table from the local cache."""
    con = get_connection()
    try:
        return con.execute(f"SELECT * FROM {table_name}").fetchdf()
    finally:
        con.close()
