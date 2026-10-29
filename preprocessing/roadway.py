"""Describe each crash site by the roads around it, from DCP's LION street network.

Every LION street segment within `RoadwayConfig.radius_ft` of a site centroid
describes the site:

- Intersection geometry: legs (generic street segments meeting) at the nearest
  intersection node, and how many intersection nodes fall inside the radius.
- Cross-section: travel lanes and curb-to-curb width of the biggest street at the
  site, summed across a street's roadbeds (a pedestrian crossing a divided avenue
  crosses both), and the median estimated lane width.
- Posted speed limit, one-way streets, highways/ramps, bike lanes and truck routes.

Sites with no LION street within the radius get NA site features;
`add_roadway_features` turns them into model-ready columns.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data.reader import read_lion
from preprocessing.config import RoadwayConfig, SiteConfig

log = logging.getLogger(__name__)

SEGMENT_COLUMNS = [
    "FeatureTyp",
    "SegmentTyp",
    "RW_TYPE",
    "TrafDir",
    "StreetCode",
    "NodeIDFrom",
    "NodeIDTo",
    "POSTED_SPEED",
    "Number_Travel_Lanes",
    "Number_Park_Lanes",
    "StreetWidth_Min",
    "StreetWidth_Max",
    "BikeLane",
    "TRUCK_ROUTE_TYPE",
]
# LION stores these as zero-padded text; 0 means "not recorded".
CODED_NUMBERS = ["RW_TYPE", "POSTED_SPEED", "Number_Travel_Lanes", "Number_Park_Lanes"]
ZERO_IS_MISSING = [
    "POSTED_SPEED",
    "Number_Travel_Lanes",
    "StreetWidth_Min",
    "StreetWidth_Max",
]

# Model-ready columns from `add_roadway_features`.
ROADWAY_FEATURES = [
    "legs_3",
    "legs_4",
    "legs_5plus",
    "n_intersections",
    "lanes",
    "width_10ft",
    "lane_width_ft",
    "speed_20_or_less",
    "speed_30_35",
    "speed_40_plus",
    "one_way",
    "highway",
    "bike_lane",
    "truck_route",
    "roadway_missing",
]


def clean_segments(segments: pd.DataFrame, config: RoadwayConfig) -> pd.DataFrame:
    """Keep street segments cars drive on, with blank codes as NA and numbers parsed."""
    out = segments.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) and col != "geometry":
            out[col] = out[col].str.strip().replace("", pd.NA)
    for col in CODED_NUMBERS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ZERO_IS_MISSING:
        out[col] = out[col].mask(out[col] == 0)
    for col in ["NodeIDFrom", "NodeIDTo"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    keep = (
        out["FeatureTyp"].isin(config.street_feature_types)
        & out["RW_TYPE"].notna()
        & ~out["RW_TYPE"].isin(config.non_road_rw_types)
        & out["TrafDir"].ne("P")  # pedestrian-only streets
    )
    log.info(
        "clean_segments: kept %d of %d LION segments as drivable streets",
        keep.sum(),
        len(out),
    )
    return out[keep].reset_index(drop=True)


def find_intersections(
    segments: pd.DataFrame, nodes: pd.DataFrame, config: RoadwayConfig
) -> pd.DataFrame:
    """Nodes where at least `min_intersection_legs` generic street segments meet.

    Returns `x`, `y` (feet) and `legs`. Nodes with two segments are bends or
    breaks along one street, not intersections.
    """
    generic = segments[segments["SegmentTyp"].isin(config.generic_segment_types)]
    legs = pd.concat([generic["NodeIDFrom"], generic["NodeIDTo"]]).value_counts()
    out = pd.DataFrame(
        {
            "node_id": pd.to_numeric(nodes["NODEID"], errors="coerce").astype("Int64"),
            "x": nodes.geometry.x,
            "y": nodes.geometry.y,
        }
    )
    out["legs"] = out["node_id"].map(legs).fillna(0).astype(int)
    out = out[out["legs"] >= config.min_intersection_legs].reset_index(drop=True)
    log.info(
        "find_intersections: %d of %d nodes are intersections", len(out), len(nodes)
    )
    return out


def site_intersection_features(
    sites: pd.DataFrame, intersections: pd.DataFrame, config: RoadwayConfig
) -> pd.DataFrame:
    """Per site: `legs` at the nearest intersection within the radius (0 if none)
    and `n_intersections` within the radius."""
    xy = sites[["x", "y"]].to_numpy()
    tree = cKDTree(intersections[["x", "y"]].to_numpy())
    dist, idx = tree.query(xy, distance_upper_bound=config.radius_ft)
    found = np.isfinite(dist)
    legs = np.zeros(len(sites), dtype=int)
    legs[found] = intersections["legs"].to_numpy()[idx[found]]
    n_within = [len(i) for i in tree.query_ball_point(xy, config.radius_ft)]
    log.info(
        "site_intersection_features: %d of %d sites have an intersection within %.0f ft",
        found.sum(),
        len(sites),
        config.radius_ft,
    )
    return pd.DataFrame(
        {
            "site_id": sites["site_id"].to_numpy(),
            "legs": legs,
            "n_intersections": n_within,
        }
    )


def match_segments(
    sites: pd.DataFrame, segments: gpd.GeoDataFrame, config: RoadwayConfig
) -> pd.DataFrame:
    """One row per (site, roadbed segment within the radius), with the segment's attributes."""
    roadbeds = segments[
        segments["SegmentTyp"].isin(config.roadbed_segment_types)
    ].reset_index(drop=True)
    points = gpd.GeoSeries(gpd.points_from_xy(sites["x"], sites["y"]), crs=roadbeds.crs)
    site_i, seg_i = roadbeds.sindex.query(
        points, predicate="dwithin", distance=config.radius_ft
    )
    pairs = pd.DataFrame(roadbeds.drop(columns="geometry").iloc[seg_i]).reset_index(
        drop=True
    )
    pairs.insert(0, "site_id", sites["site_id"].to_numpy()[site_i])
    log.info(
        "match_segments: %d site-segment pairs for %d of %d sites",
        len(pairs),
        pairs["site_id"].nunique(),
        len(sites),
    )
    return pairs


def street_cross_section(pairs: pd.DataFrame, col: str) -> pd.Series:
    """`col` summed across each street's roadbeds, per (site_id, StreetCode).

    Two-way roadbeds (TrafDir "T") already span the street; one-way roadbeds
    ("W"/"A", e.g. the two sides of a divided avenue) are added together. If a
    street has both near a site, the larger of the two totals is used.
    """
    by_dir = pairs.groupby(["site_id", "StreetCode", "TrafDir"])[col].max().unstack()
    one_way = by_dir.drop(columns="T", errors="ignore").sum(axis=1, min_count=1)
    two_way = by_dir["T"] if "T" in by_dir else pd.Series(np.nan, index=by_dir.index)
    return pd.concat([two_way, one_way], axis=1).max(axis=1)


def site_segment_features(pairs: pd.DataFrame, config: RoadwayConfig) -> pd.DataFrame:
    """Aggregate matched roadbed segments to one row per site."""
    # Blank parking lanes are read as none.
    lane_width = (
        pairs["StreetWidth_Min"]
        - config.parking_lane_ft * pairs["Number_Park_Lanes"].fillna(0)
    ) / pairs["Number_Travel_Lanes"]
    low, high = config.lane_width_bounds_ft
    pairs = pairs.assign(
        lane_width_ft=lane_width.where(lane_width.between(low, high)),
        one_way=pairs["SegmentTyp"].eq("U") & pairs["TrafDir"].isin(["W", "A"]),
        highway=pairs["RW_TYPE"].isin(config.highway_rw_types),
        bike_lane=pairs["BikeLane"].notna(),
        truck_route=pairs["TRUCK_ROUTE_TYPE"].notna(),
    )
    by_site = pairs.groupby("site_id")
    out = by_site.agg(
        n_streets=("StreetCode", "nunique"),
        lane_width_ft=("lane_width_ft", "median"),
        speed_limit=("POSTED_SPEED", "max"),
        one_way=("one_way", "any"),
        highway=("highway", "any"),
        bike_lane=("bike_lane", "any"),
        truck_route=("truck_route", "any"),
    )
    out["lanes"] = (
        street_cross_section(pairs, "Number_Travel_Lanes").groupby("site_id").max()
    )
    out["width_ft"] = (
        street_cross_section(pairs, "StreetWidth_Max").groupby("site_id").max()
    )
    return out.reset_index()


def load_roadway_features(
    sites: pd.DataFrame,
    config: RoadwayConfig | None = None,
    site_config: SiteConfig | None = None,
) -> pd.DataFrame:
    """Read LION and return one row of roadway features per site in `sites`.

    `sites` needs `site_id` and projected `x`, `y` in `SiteConfig.crs`, as
    returned by `preprocessing.panel.load_panel`.
    """
    config = config or RoadwayConfig()
    site_config = site_config or SiteConfig()
    segments = clean_segments(read_lion("lion", SEGMENT_COLUMNS), config)
    segments = segments.to_crs(site_config.crs)
    nodes = read_lion("node", ["NODEID"]).to_crs(site_config.crs)

    intersections = site_intersection_features(
        sites, find_intersections(segments, nodes, config), config
    )
    roads = site_segment_features(match_segments(sites, segments, config), config)
    out = intersections.merge(roads, on="site_id", how="left")
    log.info(
        "load_roadway_features: %d of %d sites have no LION street within %.0f ft",
        out["n_streets"].isna().sum(),
        len(out),
        config.radius_ft,
    )
    return out


def add_roadway_features(panel: pd.DataFrame, roadway: pd.DataFrame) -> pd.DataFrame:
    """Add the model-ready `ROADWAY_FEATURES` to `panel` by `site_id`.

    Missing numbers are filled with the median over sites and flagged together in
    `roadway_missing`, so no site-year is dropped. Legs and speed limit become
    indicators: legs against "no intersection" (fewer than 3), speed against 25 mph,
    the citywide default.
    """
    numeric = ["lanes", "width_ft", "lane_width_ft", "speed_limit"]
    site = roadway.set_index("site_id")
    missing = site[numeric].isna().any(axis=1)
    filled = site[numeric].fillna(site[numeric].median())
    speed = filled["speed_limit"]
    features = pd.DataFrame(
        {
            "legs_3": site["legs"].eq(3),
            "legs_4": site["legs"].eq(4),
            "legs_5plus": site["legs"].ge(5),
            "n_intersections": site["n_intersections"],
            "lanes": filled["lanes"],
            "width_10ft": filled["width_ft"] / 10,
            "lane_width_ft": filled["lane_width_ft"],
            "speed_20_or_less": speed.le(20),
            "speed_30_35": speed.between(30, 35),
            "speed_40_plus": speed.ge(40),
            "one_way": site["one_way"].fillna(False).astype(bool),
            "highway": site["highway"].fillna(False).astype(bool),
            "bike_lane": site["bike_lane"].fillna(False).astype(bool),
            "truck_route": site["truck_route"].fillna(False).astype(bool),
            "roadway_missing": missing,
        }
    ).astype(float)
    log.info(
        "add_roadway_features: %d of %d sites had a missing value filled",
        missing.sum(),
        len(site),
    )
    return panel.merge(features, left_on="site_id", right_index=True, how="left")
