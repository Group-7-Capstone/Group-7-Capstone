"""Return cached raw data as DataFrames. No transformations happen here."""

from pathlib import Path

import pandas as pd

from data.base import get_connection

CAMERA_DATA_DIR = Path(__file__).resolve().parent / "camera_data"


def read_table(table_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a raw table from the local cache, optionally only some columns."""
    select = "*" if columns is None else ", ".join(f'"{c}"' for c in columns)
    con = get_connection()
    try:
        return con.execute(f"SELECT {select} FROM {table_name}").fetchdf()
    finally:
        con.close()


def read_camera_data(name: str) -> pd.DataFrame:
    """Return a team-built camera dataset from `data/camera_data/<name>.csv` as-is."""
    return pd.read_csv(CAMERA_DATA_DIR / f"{name}.csv")
