"""Every LION intersection as a unit, including those that never had a crash.

Q1's sites are built from crash locations, so places without crashes are missing
from them. Here the units are fixed by the street network instead: LION nodes where
3+ streets meet (`roadway.find_intersections`), with nodes within
`IntersectionConfig.merge_ft` of each other merged into one intersection (a
divided road meets a cross street at several nodes). Each crash is assigned to its
nearest intersection within `crash_radius_ft`, and the panel steps from
`preprocessing.panel` turn the counts into an intersection x year panel whose
`site_id` is the intersection id.
"""

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from data.reader import read_lion
from preprocessing.config import (
    CrashConfig,
    IntersectionConfig,
    PanelConfig,
    RoadwayConfig,
    SiteConfig,
)
from preprocessing.crashes import load_crashes
from preprocessing.panel import (
    add_lag_features,
    add_log1p_features,
    count_site_years,
    drop_mixed_definition_years,
)
from preprocessing.roadway import SEGMENT_COLUMNS, clean_segments, find_intersections
from preprocessing.sites import add_site_lonlat, assign_sites, project_coordinates

log = logging.getLogger(__name__)


def merge_intersection_nodes(
    nodes: pd.DataFrame, config: IntersectionConfig
) -> pd.DataFrame:
    """One row per intersection: nodes within `merge_ft` of each other, chained.

    `nodes` has `node_id`, `x`, `y` (from `find_intersections`). Returns `site_id`,
    `x`, `y` (mean of the member nodes) and `node_ids` (a tuple).
    """
    xy = nodes[["x", "y"]].to_numpy()
    pairs = cKDTree(xy).query_pairs(config.merge_ft, output_type="ndarray")
    graph = sp.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy))
    )
    _, label = connected_components(graph, directed=False)
    out = (
        nodes.assign(site_id=label)
        .groupby("site_id")
        .agg(x=("x", "mean"), y=("y", "mean"), node_ids=("node_id", tuple))
        .reset_index()
    )
    log.info(
        "merge_intersection_nodes: %d nodes -> %d intersections", len(nodes), len(out)
    )
    return out


def add_street_names(
    intersections: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    """Add `streets`: the names of the streets meeting there, " & "-joined."""
    ends = pd.concat(
        [
            segments[["NodeIDFrom", "Street"]].rename(columns={"NodeIDFrom": "node"}),
            segments[["NodeIDTo", "Street"]].rename(columns={"NodeIDTo": "node"}),
        ]
    ).dropna()
    names = ends.groupby("node")["Street"].agg(lambda s: sorted(set(s.str.strip())))
    members = intersections[["site_id", "node_ids"]].explode("node_ids")
    streets = (
        members.assign(name=members["node_ids"].map(names))
        .dropna(subset=["name"])
        .explode("name")
        .groupby("site_id")["name"]
        .agg(lambda s: " & ".join(sorted(set(s))))
    )
    return intersections.assign(streets=intersections["site_id"].map(streets))


def load_intersection_panel(
    crash_config: CrashConfig | None = None,
    site_config: SiteConfig | None = None,
    config: PanelConfig | None = None,
    intersection_config: IntersectionConfig | None = None,
    roadway_config: RoadwayConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, intersections).

    `panel` has one row per intersection and year, with the same counts, lags and
    log features as `load_panel`; intersections with no crashes have zeros.
    `intersections` has `site_id`, `x`, `y`, `longitude`, `latitude`, `streets`.
    """
    site_config = site_config or SiteConfig()
    config = config or PanelConfig()
    intersection_config = intersection_config or IntersectionConfig()
    roadway_config = roadway_config or RoadwayConfig()

    segments = clean_segments(
        read_lion("lion", SEGMENT_COLUMNS + ["Street"]), roadway_config
    ).to_crs(site_config.crs)
    nodes = read_lion("node", ["NODEID"]).to_crs(site_config.crs)
    intersections = merge_intersection_nodes(
        find_intersections(segments, nodes, roadway_config), intersection_config
    )
    intersections = add_street_names(
        add_site_lonlat(intersections, site_config), segments
    )

    crashes = project_coordinates(load_crashes(crash_config), site_config)
    crashes = assign_sites(
        crashes,
        intersections.assign(is_center=True),
        replace(site_config, radius_ft=intersection_config.crash_radius_ft),
    )
    counts = count_site_years(crashes, intersections, config)
    panel = add_log1p_features(
        drop_mixed_definition_years(add_lag_features(counts, config), config)
    )
    log.info(
        "load_intersection_panel: %d intersection-years for %d intersections",
        len(panel),
        len(intersections),
    )
    return panel, intersections.drop(columns="node_ids")
