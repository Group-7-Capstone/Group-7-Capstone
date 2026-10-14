"""Return cached raw data as DataFrames. No transformations happen here."""

import pandas as pd
import hashlib
import io
import json
from pathlib import Path
import subprocess
import pyarrow.parquet as pq

from data.base import get_connection


def read_table(table_name: str) -> pd.DataFrame:
    """Return a raw table from the local cache."""
    con = get_connection()
    try:
        return con.execute(f"SELECT * FROM {table_name}").fetchdf()
    finally:
        con.close()


CAMERA_SOURCE_COMMIT = "743e9c7e27f6a07a6567611406857b8443a2b3f4"
CAMERA_SOURCE_PATH = "data/camera_data/camera_activation_dates.csv"


def read_eric_registry():
    """Read the exact Git snapshot, independent of local newline conversion."""
    root = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output(["git", "show", f"{CAMERA_SOURCE_COMMIT}:{CAMERA_SOURCE_PATH}"], cwd=root)
    return pd.read_csv(io.BytesIO(raw)), {
        "commit": CAMERA_SOURCE_COMMIT, "path": CAMERA_SOURCE_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "url": f"https://github.com/Group-7-Capstone/Group-7-Capstone/blob/{CAMERA_SOURCE_COMMIT}/{CAMERA_SOURCE_PATH}",
    }


def read_snapshot_inventory(cache):
    """Return raw source receipts; interpretation belongs to downstream layers."""
    return [json.loads(p.read_text()) for p in sorted(Path(cache).glob("fy*/manifest.json"))]


def iter_camera_ticket_batches(cache, batch_size=100000):
    """Yield projected raw records with fiscal-year provenance, without cleaning."""
    for path in sorted(Path(cache).glob("fy*/*/*.parquet")):
        fiscal_year = int(path.parents[1].name[2:])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield fiscal_year, batch.to_pandas()
