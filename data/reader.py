"""Return cached raw data as DataFrames. No transformations happen here."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from data.base import get_connection

CAMERA_DATA_DIR = Path(__file__).resolve().parent / "camera_data"
# DCP's LION street centerline file geodatabase, downloaded by hand (git-ignored).
LION_PATH = Path(__file__).resolve().parent / "lion" / "lion.gdb"


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


def read_lion(layer: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a LION layer as-is ("lion" segments or "node" points, EPSG:2263).

    Layers with geometry come back as a GeoDataFrame.
    """
    return gpd.read_file(LION_PATH, layer=layer, columns=columns, engine="pyogrio")
