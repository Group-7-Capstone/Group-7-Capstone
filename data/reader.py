"""Return cached raw data as DataFrames. No transformations happen here."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from data.base import NYCWALKS_DIR, get_connection, lion_release_dir

CAMERA_DATA_DIR = Path(__file__).resolve().parent / "camera_data"
# DCP's LION street centerline file geodatabase, release 26c, downloaded by hand
# (git-ignored). Other releases come from `download.download_lion`.
LION_PATH = Path(__file__).resolve().parent / "lion" / "lion.gdb"
NYCWALKS_PATH = NYCWALKS_DIR / "SHP" / "NYC_pednetwork_estimates_counts_2018-2019.shp"


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


def read_lion(
    layer: str, columns: list[str] | None = None, release: str | None = None
) -> pd.DataFrame:
    """Return a LION layer as-is ("lion" segments or "node" points, EPSG:2263).

    `release` (e.g. "22c") reads a release from `download.download_lion`; None
    reads `LION_PATH`. Layers with geometry come back as a GeoDataFrame.
    """
    path = (
        LION_PATH
        if release is None
        else lion_release_dir(release) / "lion" / "lion.gdb"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run data.download.download_lion({release!r})"
        )
    return gpd.read_file(path, layer=layer, columns=columns, engine="pyogrio")


def read_nycwalks(columns: list[str] | None = None) -> gpd.GeoDataFrame:
    """Return the NYC Walks sidewalk and crosswalk network as-is (EPSG:6538, metres).

    One row per segment with modeled 2018-19 peak-hour pedestrian volumes, e.g.
    `predwkdyAM`, `predwkdyMD`, `predwkdyPM` (weekday 8-9 AM, 12:30-1:30 PM, 5-6 PM).
    """
    if not NYCWALKS_PATH.exists():
        raise FileNotFoundError(
            f"{NYCWALKS_PATH} not found; run data.download.download_nycwalks()"
        )
    return gpd.read_file(NYCWALKS_PATH, columns=columns, engine="pyogrio")
