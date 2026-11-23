"""Use the team's camera locator, roadway features and NYC Walks for #58."""
import numpy as np
import pandas as pd

from data.reader import read_lion
from preprocessing.base import DIRECTION_PREFIX, normalize_street
from preprocessing.camera_locations import (
    locate_cameras,
    node_name_index,
    split_location,
)
from preprocessing.config import CameraDirectionConfig, RoadwayConfig, SiteConfig
from preprocessing.pedestrians import load_pedestrian_features
from preprocessing.roadway import (
    SEGMENT_COLUMNS,
    clean_segments,
    load_roadway_features,
    match_segments,
)


def named_direction_pairs(located, node_names, pairs, config=None):
    """Two named streets at verified intersections, using the shared name parser.

    A two-way generic segment takes precedence over divided one-way roadbeds.
    Missing or mixed street directions stay unknown, never default to two-way.
    """
    config = config or CameraDirectionConfig()
    names = node_name_index(node_names)
    by_name = names.groupby("name").node_id.agg(set).to_dict()
    by_node = names.groupby("node_id").name.agg(set).to_dict()
    pairs = pairs.assign(name=normalize_street(pairs.Street.fillna("")))
    direction = pairs.groupby(["site_id", "name"]).TrafDir.agg(set)
    texts = normalize_street(located.search_query.str.replace(DIRECTION_PREFIX, "", regex=True))
    result = pd.Series(np.nan, index=located.site_id, dtype="object", name="direction_pair")
    for row, text in zip(located.itertuples(), texts):
        on, crosses = split_location(text, by_name, by_node, config.min_street_name_length)
        if len(crosses) != 1 or row.geocode_method != "lion_intersection" or row.geocode_confidence != "high":
            continue
        labels = []
        for name in (on, crosses[0]):
            dirs = direction.get((row.site_id, name), set())
            labels.append("two-way" if "T" in dirs else "one-way" if dirs and dirs <= {"A", "W"} else None)
        if all(labels):
            result.loc[row.site_id] = " × ".join(sorted(labels))
    return result


def load_ticket_site_features(summary):
    """One row per camera; a fixed LION/2018-19 pedestrian snapshot, not history."""
    direction_config, road_config, site_config = CameraDirectionConfig(), RoadwayConfig(), SiteConfig()
    cameras = summary.loc[summary.camera_type.eq("Speed")].copy().reset_index(drop=True)
    node_names = read_lion("node_stname")
    located = locate_cameras(cameras, node_names, read_lion("node", ["NODEID"]).to_crs(site_config.crs), direction_config, site_config)
    located["site_id"] = located.camera_id
    roads = load_roadway_features(located, road_config, site_config)
    ped = load_pedestrian_features(located, site_config=site_config)
    segments = clean_segments(read_lion("lion", SEGMENT_COLUMNS + ["Street", "LBoro", "RBoro"]), road_config).to_crs(site_config.crs)
    pairs = match_segments(located, segments, road_config)
    # Include generic two-way centerlines when classifying divided streets.
    from preprocessing.base import sites_within
    si, ri = sites_within(located, segments, road_config.radius_ft)
    direction_pairs = pd.DataFrame(segments.drop(columns="geometry").iloc[ri]).reset_index(drop=True)
    direction_pairs["site_id"] = located.site_id.to_numpy()[si]
    named = named_direction_pairs(located, node_names, direction_pairs, direction_config)
    located["direction_pair"] = located.site_id.map(named)
    borough = pd.concat([pairs[["site_id", "LBoro"]].rename(columns={"LBoro": "borough"}),
                         pairs[["site_id", "RBoro"]].rename(columns={"RBoro": "borough"})])
    borough["borough"] = pd.to_numeric(borough.borough, errors="coerce")
    borough = borough.loc[borough.borough.between(1, 5)].groupby("site_id").borough.agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
    located["borough"] = located.site_id.map(borough).map({1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"})
    cols = ["camera_id", "site_id", "borough", "direction_pair", "latitude", "longitude", "geocode_method", "geocode_confidence", "geocode_shift_ft"]
    out = located[cols].merge(roads, on="site_id", validate="one_to_one").merge(ped, on="site_id", validate="one_to_one")
    # Unverified/far-ambiguous locations remain in the outcome directory but not feature claims.
    unknown = ~out.geocode_confidence.eq("high")
    out.loc[unknown, ["borough", "direction_pair", "lanes", "width_ft", "ped_volume"]] = np.nan
    out["direction_pair_method"] = "shared_LION_locator_named_street_directions"
    out["direction_pair_confidence"] = np.where(out.direction_pair.notna(), "high", "unresolved")
    return out.drop(columns="site_id")
