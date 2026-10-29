"""Return cached raw data as DataFrames. No transformations happen here."""

from pathlib import Path
import hashlib
import json
import pyarrow.parquet as pq

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


def read_snapshot_inventory(cache):
    """Return raw source receipts; interpretation belongs to downstream layers."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(cache).glob("fy*/manifest.json"))]


def iter_camera_ticket_batches(cache, batch_size=100000, columns=None):
    """Yield projected raw records with fiscal-year provenance, without cleaning."""
    for path in sorted(Path(cache).glob("fy*/*/*.parquet")):
        fiscal_year = int(path.parents[1].name[2:])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            yield fiscal_year, batch.to_pandas()




def read_lion_road_attributes(path):
    """Read supplied LION records without cleaning; pin every geodatabase file."""
    import pyogrio
    from datetime import datetime, timezone
    path = Path(path)
    columns = ["Street", "SAFStreetName", "Join_ID", "SegmentID", "FeatureTyp",
               "RB_Layer", "RW_TYPE", "TrafDir", "LBoro", "RBoro", "NodeIDFrom",
               "NodeIDTo", "NodeLevelF", "NodeLevelT", "StreetWidth_Min",
               "StreetWidth_Max", "Number_Travel_Lanes", "Number_Park_Lanes",
               "Number_Total_Lanes"]
    columns += ["XFrom", "YFrom", "XTo", "YTo"]
    roads = pyogrio.read_dataframe(path, layer="lion", columns=columns)
    aliases = pyogrio.read_dataframe(path, layer="altnames")
    files = []
    for item in sorted(path.iterdir()):
        if item.is_file() and not item.name.endswith(".lock"):
            with item.open("rb") as source:
                checksum = hashlib.file_digest(source, "sha256").hexdigest()
            files.append({"name": item.name, "bytes": item.stat().st_size,
                          "sha256": checksum})
    return roads, aliases, {"dataset": "Supplied NYC DCP LION geodatabase",
        "layers": ["lion", "altnames"], "columns": columns, "filter": "None at read time",
        "read_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "release": "Not independently established; supplied snapshot",
        "historical_note": "A single road-network snapshot, not road conditions at inferred activation."}


def camera_data_receipt(name):
    """Record raw file bytes and a newline-independent content hash; no Git needed."""
    path = CAMERA_DATA_DIR / f"{name}.csv"
    raw = path.read_bytes()
    return {"path": f"data/camera_data/{name}.csv", "sha256": hashlib.sha256(raw).hexdigest(),
            "lf_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}
