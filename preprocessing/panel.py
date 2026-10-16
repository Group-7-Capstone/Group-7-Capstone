"""Build the site x year panel: KSI outcome plus lagged crash-history and camera features.

Each row is one site in one prediction year t. The outcome `ksi` is the number of KSI
crashes at the site in year t; every `*_lag{k}` feature sums years t-k..t-1, so no
feature uses information from the outcome year.
"""

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data.reader import read_camera_data
from preprocessing.config import CrashConfig, PanelConfig, SiteConfig
from preprocessing.crashes import load_crashes
from preprocessing.sites import (
    add_site_lonlat,
    assign_sites,
    build_site_points,
    project_coordinates,
    summarize_sites,
)

log = logging.getLogger(__name__)

COUNTS = ["ksi", "fatal", "injury", "crashes", "ped_injured", "cyc_injured"]
CAMERA_TYPES = {"Red Light": "rlc_active", "Speed": "speed_active"}


def count_site_years(
    crashes: pd.DataFrame, sites: pd.DataFrame, config: PanelConfig
) -> pd.DataFrame:
    """Yearly counts on a complete site x year grid (zero where a site had no crashes).

    `injury` counts crashes with any injury or death; `ped_injured`/`cyc_injured`
    count pedestrians/cyclists injured or killed. Missing person counts are
    treated as zero (logged), since they cannot be attributed to a road user.
    """
    c = crashes[crashes["site_id"].notna()]
    c = c[c["crash_date"].dt.year.between(config.first_year, config.last_year)]
    na = c["number_of_persons_injured"].isna().sum()
    log.info("count_site_years: %d crashes with missing injury counts treated as 0", na)

    def n(col):
        return c[col].fillna(0).astype("int64")

    yearly = (
        pd.DataFrame(
            {
                "site_id": c["site_id"].astype("int64"),
                "year": c["crash_date"].dt.year,
                "ksi": c["ksi"].astype("int64"),
                "fatal": (c["n_killed"] > 0).astype("int64"),
                "injury": ((n("number_of_persons_injured") > 0) | c["ksi"]).astype(
                    "int64"
                ),
                "crashes": 1,
                "ped_injured": n("number_of_pedestrians_injured")
                + n("number_of_pedestrians_killed"),
                "cyc_injured": n("number_of_cyclist_injured")
                + n("number_of_cyclist_killed"),
            }
        )
        .groupby(["site_id", "year"])[COUNTS]
        .sum()
    )
    grid = pd.MultiIndex.from_product(
        [sorted(sites["site_id"]), range(config.first_year, config.last_year + 1)],
        names=["site_id", "year"],
    )
    return yearly.reindex(grid, fill_value=0).reset_index()


def append_scoring_year(counts: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    """Add a row per site for year last_year + 1 with unknown (NA) counts.

    After `add_lag_features` these rows hold the features for forecasting the
    year after the data ends; their outcome stays NA.
    """
    nxt = counts[counts["year"] == config.last_year].assign(year=config.last_year + 1)
    nxt[COUNTS] = np.nan
    both = pd.concat([counts, nxt]).sort_values(["site_id", "year"])
    return both.reset_index(drop=True)


def add_lag_features(counts: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    """Add `<count>_lag{k}` = sum over the k years before each row's year.

    Keeps only years with a full k-year history, i.e. from the first year + k on.
    Expects a complete grid sorted by site and year, as from `count_site_years`.
    """
    k = config.lag_years
    first_year = counts["year"].min()
    n_years = counts["year"].nunique()
    out = counts.copy()
    for col in COUNTS:
        arr = counts[col].to_numpy(dtype=float).reshape(-1, n_years)
        cum = np.concatenate([np.zeros((len(arr), 1)), arr.cumsum(axis=1)], axis=1)
        lag = np.full(arr.shape, np.nan)
        lag[:, k:] = cum[:, k:-1] - cum[:, : n_years - k]
        out[f"{col}_lag{k}"] = lag.ravel()
    return out[out["year"] >= first_year + k].reset_index(drop=True)


def add_log1p_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `log1p_<col>` for every lagged count, the form the count models use."""
    lags = [c for c in panel if "_lag" in c and not c.startswith("log1p_")]
    return panel.assign(**{f"log1p_{c}": np.log1p(panel[c]) for c in lags})


def site_highway_share(crashes: pd.DataFrame, site_config: SiteConfig) -> pd.Series:
    """Share of each site's crashes before `build_end` that are on a highway."""
    train = crashes[crashes["crash_date"] < pd.Timestamp(site_config.build_end)]
    return train.groupby("site_id")["highway"].mean().rename("highway_share")


def assign_cameras(
    cameras: pd.DataFrame,
    sites: pd.DataFrame,
    site_config: SiteConfig,
    config: PanelConfig,
) -> pd.DataFrame:
    """Add `site_id` to each camera: the nearest site centroid within `camera_radius_ft`."""
    cams = project_coordinates(
        cameras.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}),
        site_config,
    )
    dist, idx = cKDTree(sites[["x", "y"]].to_numpy()).query(
        cams[["x", "y"]].to_numpy(), distance_upper_bound=config.camera_radius_ft
    )
    matched = np.isfinite(dist)
    site_id = pd.Series(pd.NA, index=cams.index, dtype="Int64")
    site_id[matched] = sites["site_id"].to_numpy()[idx[matched]]
    log.info(
        "assign_cameras: %d of %d cameras matched to a site within %.0f ft",
        matched.sum(),
        len(cams),
        config.camera_radius_ft,
    )
    return cams.assign(
        site_id=site_id,
        activation_date=pd.to_datetime(cams["inferred_activation_date"]),
    )


def add_camera_flags(panel: pd.DataFrame, cameras: pd.DataFrame) -> pd.DataFrame:
    """Add per-type `*_active` (a camera activated before Jan 1 of the row's year)
    and `camera_site` (the site has any camera, at any time)."""
    cams = cameras[cameras["site_id"].notna()]
    out = panel.copy()
    for camera_type, col in CAMERA_TYPES.items():
        of_type = cams[cams["camera_type"] == camera_type]
        first_year = of_type.groupby("site_id")["activation_date"].min().dt.year
        out[col] = out["site_id"].map(first_year.to_dict()) < out["year"]
    out["camera_site"] = out["site_id"].isin(cams["site_id"])
    return out


def load_panel(
    crash_config: CrashConfig | None = None,
    site_config: SiteConfig | None = None,
    config: PanelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all preprocessing steps.

    Returns (panel, scoring, sites): `panel` has one row per site and prediction
    year up to `last_year` with the observed outcome; `scoring` has the features
    for year last_year + 1, whose outcome is not yet known.
    """
    site_config = site_config or SiteConfig()
    config = config or PanelConfig()
    crashes = project_coordinates(load_crashes(crash_config), site_config)
    site_points = build_site_points(crashes, site_config)
    sites = add_site_lonlat(summarize_sites(site_points), site_config)
    crashes = assign_sites(crashes, site_points, site_config)

    counts = append_scoring_year(count_site_years(crashes, sites, config), config)
    rows = add_log1p_features(add_lag_features(counts, config))
    rows = rows.merge(
        site_highway_share(crashes, site_config), on="site_id", how="left"
    )
    cameras = assign_cameras(
        read_camera_data("camera_activation_dates"), sites, site_config, config
    )
    rows = add_camera_flags(rows, cameras)
    for col in COUNTS:
        rows[col] = rows[col].astype("Int64")
    scoring = rows["year"] > config.last_year
    panel = rows[~scoring].reset_index(drop=True)
    log.info(
        "load_panel: %d site-years for %d sites", len(panel), panel["site_id"].nunique()
    )
    return panel, rows[scoring].reset_index(drop=True), sites
