
"""Describe cyclist exposure at each crash site using nearby bicycle counters.

For each intersection site:
- Find the nearest bicycle counter within a specified radius.
- Calculate the counter's average daily bicycle volume for each year.
- Associate that annual volume with the intersection site.
- Return site-year features that can be merged into the crash panel.

Input data must contain daily bicycle counts, counter coordinates,
a date, and optionally a counter ID.

Coordinates are transformed into SiteConfig.crs before spatial matching.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from preprocessing.config import SiteConfig

log = logging.getLogger(__name__)

# Model-ready features created by add_bicycle_features.
BICYCLE_FEATURES = [
    "log1p_bike_volume",
    "bicycle_missing",
]


def load_bicycle_counts(
    path: str,
    site_config: SiteConfig | None = None,
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    count_col: str = "count",
    date_col: str = "date",
    counter_id_col: str | None = "counter_id",
    source_crs: str = "EPSG:4326",
) -> pd.DataFrame:
    """Read and clean daily bicycle counter observations.

    Returns a GeoDataFrame with standardized columns:
    counter_id, date, year, count, geometry.
    
    Column names can be changed to match the downloaded dataset.
    """
    site_config = site_config or SiteConfig()

    raw = pd.read_csv(path)

    required = [longitude_col, latitude_col, count_col, date_col]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing bicycle data columns: {missing}")

    out = pd.DataFrame({
        "longitude": pd.to_numeric(raw[longitude_col], errors="coerce"),
        "latitude": pd.to_numeric(raw[latitude_col], errors="coerce"),
        "count": pd.to_numeric(raw[count_col], errors="coerce"),
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
    })

    if counter_id_col and counter_id_col in raw.columns:
        out["counter_id"] = raw[counter_id_col].astype(str)
    else:
        # Fallback when the source has no counter identifier.
        out["counter_id"] = (
            out["longitude"].astype(str) + "_" + out["latitude"].astype(str)
        )

    out = out.dropna(
        subset=["longitude", "latitude", "count", "date"]
    )
    out = out[out["count"] >= 0].copy()
    out["year"] = out["date"].dt.year

    points = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["longitude"], out["latitude"]),
        crs=source_crs,
    )

    points = points.to_crs(site_config.crs)

    log.info(
        "load_bicycle_counts: loaded %d observations from %d counters",
        len(points),
        points["counter_id"].nunique(),
    )

    return points


def match_bicycle_counters(
    sites: pd.DataFrame,
    counters: gpd.GeoDataFrame,
    radius_ft: float = 1000,
) -> pd.DataFrame:
    """Match each intersection to its nearest bicycle counter.

    Sites and counters must use the same projected coordinate system,
    with x and y measured in feet.
    """
    if counters.empty:
        return pd.DataFrame(columns=["site_id", "counter_id", "distance_ft"])

    counter_xy = np.column_stack([
        counters.geometry.x,
        counters.geometry.y,
    ])

    site_xy = sites[["x", "y"]].to_numpy()

    tree = cKDTree(counter_xy)
    distance, index = tree.query(
        site_xy,
        distance_upper_bound=radius_ft,
    )

    found = np.isfinite(distance)

    matched = pd.DataFrame({
        "site_id": sites["site_id"].to_numpy()[found],
        "counter_id": counters.iloc[index[found]]["counter_id"].to_numpy(),
        "distance_ft": distance[found],
    })

    log.info(
        "match_bicycle_counters: matched %d of %d sites within %.0f ft",
        len(matched),
        len(sites),
        radius_ft,
    )

    return matched


def site_bicycle_features(
    sites: pd.DataFrame,
    counters: gpd.GeoDataFrame,
    radius_ft: float = 1000,
) -> pd.DataFrame:
    """Calculate annual bicycle exposure features for each site.

    Uses the nearest counter and its average daily count for each year.
    Assumes the input observations represent daily totals, not hourly counts.
    """
    matched = match_bicycle_counters(sites, counters, radius_ft)

    if matched.empty:
        return pd.DataFrame(
            columns=["site_id", "year", "bike_volume"]
        )

    # Average daily bicycle volume at each counter, by year.
    annual = (
        counters.groupby(["counter_id", "year"])
        .agg(
            bike_volume=("count", "mean"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )

    out = matched.merge(annual, on="counter_id", how="left")

    log.info(
        "site_bicycle_features: created %d site-year observations",
        len(out),
    )

    return out[["site_id", "year", "bike_volume"]]


def load_bicycle_features(
    sites: pd.DataFrame,
    path: str,
    site_config: SiteConfig | None = None,
    radius_ft: float = 1000,
    **column_args,
) -> pd.DataFrame:
    """Load bicycle counter data and return annual site-level features."""
    counters = load_bicycle_counts(
        path,
        site_config=site_config,
        **column_args,
    )

    return site_bicycle_features(
        sites,
        counters,
        radius_ft=radius_ft,
    )


def add_bicycle_features(
    panel: pd.DataFrame,
    bicycle: pd.DataFrame,
) -> pd.DataFrame:
    """Add model-ready bicycle exposure features to the crash panel."""

    key = ["site_id", "year"]

    out = panel.merge(
        bicycle,
        on=key,
        how="left",
        validate="many_to_one",
    )

    missing = out["bike_volume"].isna()

    median_volume = out["bike_volume"].median()

    filled = out["bike_volume"].fillna(median_volume)

    out["log1p_bike_volume"] = np.log1p(
        filled.clip(lower=0)
    )

    out["bicycle_missing"] = missing.astype(float)

    log.info(
        "add_bicycle_features: %d of %d panel rows missing bicycle volume",
        missing.sum(),
        len(out),
    )

    return out
