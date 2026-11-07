"""Build the site x year panel: KSI outcome plus lagged crash-history and camera features.

Each row is one site in one prediction year t. The outcome `ksi` is the number of KSI
crashes at the site in year t; every `*_lag{k}` feature sums years t-k..t-1, so no
feature uses information from the outcome year.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data.reader import read_camera_data, read_lion
from preprocessing.config import CrashConfig, PanelConfig, RoadwayConfig, SiteConfig
from preprocessing.crashes import load_crashes
from preprocessing.roadway import (
    ROADWAY_FEATURES,
    SEGMENT_COLUMNS,
    clean_segments,
    find_intersections,
)
from preprocessing.sites import (
    add_site_borough,
    add_site_lonlat,
    assign_nearest_site,
    assign_sites,
    assign_unmatched_to_blocks,
    build_intersection_sites,
    build_midblock_sites,
    build_site_points,
    project_coordinates,
    summarize_sites,
)

log = logging.getLogger(__name__)

RAW_COUNTS = ["ksi", "fatal", "injury", "crashes", "ped_injured", "cyc_injured"]
# Crash types (overlapping, not tiers): red-light running, rear-end and angle.
CRASH_TYPES = ["rlr", "rear_end", "angle"]
# Mutually exclusive severity tiers: each crash counts in exactly one, unlike the
# nested ksi/injury/crashes counts (fatal is already its own tier).
SEVERITY_TIERS = ["pdo", "injury_nonsevere", "ksi_nonfatal"]
# Crashes with a known time, at night (`PanelConfig.night_hours`) and on weekends.
TIME_COUNTS = ["known_time", "night", "weekend"]
COUNTS = RAW_COUNTS + CRASH_TYPES + SEVERITY_TIERS + TIME_COUNTS
CAMERA_TYPES = {"Red Light": "rlc_active", "Speed": "speed_active"}
# Site-type columns from `add_site_features` (LION sites only).
SITE_FEATURES = ["midblock", "log_block_length"]

# Standard model feature sets, so every notebook fits the same columns. History
# and crash types are log1p sums over the `PanelConfig.lag_years` (3) years before.
HISTORY_FEATURES = [
    f"log1p_{c}_lag3" for c in [*SEVERITY_TIERS, "fatal", "ped_injured", "cyc_injured"]
]
CRASH_TYPE_FEATURES = [f"log1p_{c}_lag3" for c in CRASH_TYPES]
# From `add_recent_change` and `add_temporal_features` (Fei's night, weekend and
# recent-change features, fei/temporal_features.py).
TEMPORAL_FEATURES = ["night_share", "weekend_share", "recent_change", "no_history"]
FEATURE_SETS = {
    "history": HISTORY_FEATURES,
    "crash_types": CRASH_TYPE_FEATURES,
    # After `roadway.add_roadway_features`. Bike lanes are left out: lanes are often
    # built where serious crashes happened, so they're tested separately (#48).
    "road": [
        f for f in ROADWAY_FEATURES if f not in ("bike_lane", "protected_bike_lane")
    ],
    "bike": ["bike_lane", "protected_bike_lane"],
    "site": SITE_FEATURES,
    "temporal": TEMPORAL_FEATURES,
}


def count_site_years(
    crashes: pd.DataFrame, sites: pd.DataFrame, config: PanelConfig
) -> pd.DataFrame:
    """Yearly counts on a complete site x year grid (zero where a site had no crashes).

    `injury` counts crashes with any injury or death; `ped_injured`/`cyc_injured`
    count pedestrians/cyclists injured or killed. Missing person counts are
    treated as zero (logged), since they cannot be attributed to a road user.
    The severity tiers split `crashes` into property-damage-only (`pdo`), injury
    but not KSI (`injury_nonsevere`), KSI without a death (`ksi_nonfatal`) and `fatal`.
    """
    c = crashes[crashes["site_id"].notna()]
    c = c[c["crash_date"].dt.year.between(config.first_year, config.last_year)]
    na = c["number_of_persons_injured"].isna().sum()
    log.info("count_site_years: %d crashes with missing injury counts treated as 0", na)

    hour = c["crash_hour"]
    known = hour.notna()
    start, end = config.night_hours
    night = (known & ((hour >= start) | (hour < end))).fillna(False)

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
                **{t: c[t].astype("int64") for t in CRASH_TYPES},
                "known_time": known.astype("int64"),
                "night": night.astype("int64"),
                "weekend": (c["crash_date"].dt.dayofweek >= 5).astype("int64"),
            }
        )
        .groupby(["site_id", "year"])[RAW_COUNTS + CRASH_TYPES + TIME_COUNTS]
        .sum()
    )
    grid = pd.MultiIndex.from_product(
        [sorted(sites["site_id"]), range(config.first_year, config.last_year + 1)],
        names=["site_id", "year"],
    )
    counts = yearly.reindex(grid, fill_value=0).reset_index()
    return counts.assign(
        pdo=counts["crashes"] - counts["injury"],
        injury_nonsevere=counts["injury"] - counts["ksi"],
        ksi_nonfatal=counts["ksi"] - counts["fatal"],
    )


def append_scoring_year(counts: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    """Add a row per site for year last_year + 1 with unknown (NA) counts.

    After `add_lag_features` these rows hold the features for forecasting the
    year after the data ends; their outcome stays NA.
    """
    nxt = counts[counts["year"] == config.last_year].assign(year=config.last_year + 1)
    nxt[COUNTS] = np.nan
    both = pd.concat([counts, nxt]).sort_values(["site_id", "year"])
    return both.reset_index(drop=True)


def add_recent_change(counts: pd.DataFrame) -> pd.DataFrame:
    """Add `recent_change`: log1p(crashes in year t-1) minus log1p(mean crashes in
    t-2 and t-3). Positive when crashes rose last year. NA without 3 prior years
    (those rows are dropped by `add_lag_features`). Expects rows sorted by site, year."""
    crashes = counts.groupby("site_id")["crashes"]
    earlier = (crashes.shift(2) + crashes.shift(3)) / 2
    return counts.assign(recent_change=np.log1p(crashes.shift(1)) - np.log1p(earlier))


def add_temporal_features(panel: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    """Add shares of the lag window's crashes at night (of those with a known time)
    and on weekends, 0 where the denominator is 0, and `no_history` (no crashes in
    the window) so a 0 share with no crashes isn't read as a real 0."""
    k = f"_lag{config.lag_years}"

    def share(num, den):
        den = panel[f"{den}{k}"]
        return (panel[f"{num}{k}"] / den.where(den > 0)).fillna(0.0)

    return panel.assign(
        night_share=share("night", "known_time"),
        weekend_share=share("weekend", "crashes"),
        no_history=(panel[f"crashes{k}"] == 0).astype(int),
    )


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


def drop_mixed_definition_years(
    panel: pd.DataFrame, config: PanelConfig
) -> pd.DataFrame:
    """Keep prediction years whose lag window starts on or after `ksi_definition_start`."""
    keep = panel["year"] - config.lag_years >= config.ksi_definition_start
    log.info(
        "drop_mixed_definition_years: dropped %d site-years before %d",
        (~keep).sum(),
        config.ksi_definition_start + config.lag_years,
    )
    return panel[keep].reset_index(drop=True)


def add_log1p_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `log1p_<col>` for every lagged count, the form the count models use."""
    lags = [c for c in panel if "_lag" in c and not c.startswith("log1p_")]
    return panel.assign(**{f"log1p_{c}": np.log1p(panel[c]) for c in lags})


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


def load_street_segments(site_config: SiteConfig) -> gpd.GeoDataFrame:
    """Drivable LION street segments, with `SegmentID`, in `site_config.crs`."""
    segments = read_lion("lion", SEGMENT_COLUMNS + ["SegmentID"])
    return clean_segments(segments, RoadwayConfig()).to_crs(site_config.crs)


def load_intersections(
    site_config: SiteConfig, segments: gpd.GeoDataFrame | None = None
) -> pd.DataFrame:
    """LION intersection nodes (`node_id`, `x`, `y`, `legs`) in `site_config.crs`."""
    if segments is None:
        segments = load_street_segments(site_config)
    nodes = read_lion("node", ["NODEID"]).to_crs(site_config.crs)
    return find_intersections(segments, nodes, RoadwayConfig())


def build_lion_sites(
    crashes: pd.DataFrame, site_config: SiteConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LION sites and the crashes with their `site_id`.

    Intersection sites come first; with `site_config.midblock`, crashes not near
    an intersection go to the nearest street segment (generic centerline, one per
    street), which becomes a mid-block site. `midblock` marks the site type.
    """
    segments = load_street_segments(site_config)
    sites = build_intersection_sites(
        load_intersections(site_config, segments), site_config
    ).assign(midblock=False)
    crashes = assign_nearest_site(crashes, sites, site_config)
    if site_config.midblock:
        generic = segments[
            segments["SegmentTyp"].isin(RoadwayConfig().generic_segment_types)
        ]
        blocks = build_midblock_sites(generic, int(sites["site_id"].max()) + 1)
        crashes = assign_unmatched_to_blocks(crashes, blocks, site_config)
        sites = pd.concat(
            [
                sites,
                pd.DataFrame(blocks.drop(columns="geometry")).assign(midblock=True),
            ],
            ignore_index=True,
        )
    return add_site_lonlat(sites, site_config), crashes


def add_site_features(panel: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Add `SITE_FEATURES` by `site_id`: `midblock` (0/1) and `log_block_length`
    (log feet, at least 1 ft; 0 for intersections, so it only acts mid-block).
    Intersection-only sites (`SiteConfig.midblock=False`) have no `length_ft`."""
    site = sites.set_index("site_id")
    midblock = site["midblock"]
    length = site.get("length_ft", pd.Series(np.nan, index=site.index)).clip(lower=1)
    out = panel.assign(midblock=panel["site_id"].map(midblock).astype(int))
    log_length = np.log(panel["site_id"].map(length)).fillna(0.0)
    return out.assign(log_block_length=log_length.where(out["midblock"] == 1, 0.0))


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
    if site_config.source == "lion":
        sites, crashes = build_lion_sites(crashes, site_config)
    else:
        site_points = build_site_points(crashes, site_config)
        sites = add_site_lonlat(summarize_sites(site_points), site_config)
        crashes = assign_sites(crashes, site_points, site_config)
    sites = add_site_borough(sites, crashes, site_config)

    counts = append_scoring_year(count_site_years(crashes, sites, config), config)
    rows = add_lag_features(add_recent_change(counts), config)
    rows = add_log1p_features(drop_mixed_definition_years(rows, config))
    rows = add_temporal_features(rows, config)
    cameras = assign_cameras(
        read_camera_data("camera_activation_dates"), sites, site_config, config
    )
    rows = add_camera_flags(rows, cameras)
    if "midblock" in sites:
        rows = add_site_features(rows, sites)
    for col in COUNTS:
        rows[col] = rows[col].astype("Int64")
    scoring = rows["year"] > config.last_year
    panel = rows[~scoring].reset_index(drop=True)
    log.info(
        "load_panel: %d site-years for %d sites", len(panel), panel["site_id"].nunique()
    )
    return panel, rows[scoring].reset_index(drop=True), sites
