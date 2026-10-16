"""Clean raw crashes and flag killed-or-seriously-injured (KSI) crashes.

Each step takes and returns a DataFrame and logs how many rows it drops or changes.
"""

import logging

import numpy as np
import pandas as pd

from data.reader import read_table
from preprocessing.config import CrashConfig

log = logging.getLogger(__name__)

COUNT_COLUMNS = [
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_pedestrians_killed",
    "number_of_cyclist_injured",
    "number_of_cyclist_killed",
    "number_of_motorist_injured",
    "number_of_motorist_killed",
]
CRASH_COLUMNS = [
    "collision_id",
    "crash_date",
    "latitude",
    "longitude",
    "on_street_name",
    *COUNT_COLUMNS,
]
PERSON_COLUMNS = ["collision_id", "person_injury", "complaint", "emotional_status"]


def clean_crashes(raw: pd.DataFrame, config: CrashConfig) -> pd.DataFrame:
    """Cast SODA strings to types and keep crashes on or after `config.start_date`.

    Missing injury/death counts stay NA; they are not filled with zero.
    """
    df = raw.copy()
    df["collision_id"] = pd.to_numeric(df["collision_id"]).astype("int64")
    df["crash_date"] = pd.to_datetime(df["crash_date"]).dt.normalize()
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in COUNT_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    keep = df["crash_date"] >= pd.Timestamp(config.start_date)
    log.info(
        "clean_crashes: dropped %d crashes before %s", (~keep).sum(), config.start_date
    )
    return df[keep].reset_index(drop=True)


def drop_ungeocoded(crashes: pd.DataFrame, config: CrashConfig) -> pd.DataFrame:
    """Drop crashes with missing coordinates or coordinates outside NYC."""
    valid = crashes["latitude"].between(*config.lat_bounds) & crashes[
        "longitude"
    ].between(*config.lon_bounds)
    log.info(
        "drop_ungeocoded: dropped %d of %d crashes (%.1f%%) with missing or out-of-NYC coordinates",
        (~valid).sum(),
        len(crashes),
        100 * (~valid).mean(),
    )
    return crashes[valid].reset_index(drop=True)


def flag_ksi(
    crashes: pd.DataFrame, persons: pd.DataFrame, config: CrashConfig
) -> pd.DataFrame:
    """Add `n_killed`, `n_severe_injured` and boolean `ksi` to each crash.

    `n_killed` is the larger of the crash table's count and the person table's
    count, since the two tables occasionally disagree. `n_severe_injured` counts
    injured persons whose complaint or emotional status is in the config's
    severe lists. Crashes with no person records get zero severe injuries.
    """
    p = persons.copy()
    p["collision_id"] = pd.to_numeric(p["collision_id"]).astype("int64")
    p["killed"] = p["person_injury"].eq("Killed")
    p["severe"] = p["person_injury"].eq("Injured") & (
        p["complaint"].isin(config.severe_complaints)
        | p["emotional_status"].isin(config.severe_emotional_status)
    )
    per_crash = p.groupby("collision_id")[["killed", "severe"]].sum()

    out = crashes.merge(per_crash, left_on="collision_id", right_index=True, how="left")
    out["n_killed"] = np.fmax(
        out["number_of_persons_killed"].astype("float").fillna(0),
        out["killed"].fillna(0),
    ).astype("int64")
    out["n_severe_injured"] = out["severe"].fillna(0).astype("int64")
    out["ksi"] = (out["n_killed"] > 0) | (out["n_severe_injured"] > 0)
    out = out.drop(columns=["killed", "severe"])

    log.info(
        "flag_ksi: %d KSI crashes (%d with a death); %d crashes had no person records",
        out["ksi"].sum(),
        (out["n_killed"] > 0).sum(),
        (~out["collision_id"].isin(per_crash.index)).sum(),
    )
    return out


def flag_highway(crashes: pd.DataFrame, config: CrashConfig) -> pd.DataFrame:
    """Add boolean `highway` from `on_street_name`; a missing street name is False."""
    highway = (
        crashes["on_street_name"]
        .str.upper()
        .str.contains(config.highway_pattern, na=False)
    )
    log.info(
        "flag_highway: %d of %d crashes flagged as highway", highway.sum(), len(crashes)
    )
    return crashes.assign(highway=highway)


def load_crashes(config: CrashConfig | None = None) -> pd.DataFrame:
    """Read raw crashes and persons, then clean, drop ungeocoded and flag highway and KSI."""
    config = config or CrashConfig()
    crashes = clean_crashes(read_table("crashes", CRASH_COLUMNS), config)
    crashes = drop_ungeocoded(crashes, config)
    crashes = flag_highway(crashes, config)
    return flag_ksi(crashes, read_table("persons", PERSON_COLUMNS), config)
