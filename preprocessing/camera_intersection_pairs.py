"""Assign unordered road pairs at a unique LION node shared by both named streets."""
import re

import numpy as np
import pandas as pd
from pyproj import Transformer

from preprocessing.camera_road_features import (
    GROUPS, ROAD_CONFIG, UNKNOWN, consensus_attributes, match_camera_roads,
)

PAIR_CONFIG = {**ROAD_CONFIG, "narrow_width_m": 15.0}
PAIR_DIMENSIONS = ("direction", "width", "lanes")


def reverse_intersection(address):
    streets = re.split(r"\s+AND\s+|\s*&\s*", address.upper().split(",")[0].strip(), maxsplit=1)
    if len(streets) != 2:
        raise ValueError(f"Expected two named streets: {address}")
    return " AND ".join(reversed(streets))


def unordered_pair(left, right, order):
    if left not in order or right not in order:
        return UNKNOWN
    return " × ".join(sorted((left, right), key=order.index))


def width_class(segments, config):
    low = pd.to_numeric(segments.width_min_ft, errors="coerce") * 0.3048
    high = pd.to_numeric(segments.width_max_ft, errors="coerce") * 0.3048
    threshold = config["narrow_width_m"]
    if not (low.ge(config["minimum_width_m"]) & low.le(high) & high.le(config["maximum_width_m"])).all():
        return UNKNOWN
    if high.lt(threshold).all():
        return f"<{threshold:g} m"
    if low.ge(threshold).all():
        return f"≥{threshold:g} m"
    return UNKNOWN


def endpoints_for(roads, segment_ids, config):
    subset = roads.loc[roads.SegmentID.isin(segment_ids)]
    project = Transformer.from_crs(roads.crs, config["crs"], always_xy=True)
    endpoints = {}
    for side, lev in [("From", "F"), ("To", "T")]:
        xx, yy = project.transform(subset["X"+side].to_numpy(), subset["Y"+side].to_numpy())
        for seg, node, level, x, y in zip(subset.SegmentID, subset["NodeID"+side],
                                           subset["NodeLevel"+lev], xx, yy):
            node, level = str(node or "").strip(), str(level or "").strip()
            if node in ("", "0", "0000000", "nan", "None") or level in ("", "*", "nan", "None"):
                continue
            endpoints.setdefault(seg, []).append(((node, level), x, y))
    return endpoints


def near_nodes(ids, endpoints, x, y, radius):
    nodes = {}
    for segment in ids:
        for node, nx, ny in endpoints.get(segment, []):
            if np.hypot(nx-x, ny-y) <= radius:
                nodes.setdefault(node, set()).add(segment)
    return nodes


def classify_pairs(features, first_segments, roads, aliases, config=PAIR_CONFIG):
    """Keep the frozen first-street match, and apply the same criteria to the cross street."""
    if not features.camera_id.is_unique:
        raise ValueError("Repeated camera identifier in road features")
    keep = ["camera_id", "camera_type", "search_query", "Latitude", "Longitude", "county_labels",
            "inferred_activation_date", "detection_confidence", "first_street_key", "cross_street_key",
            "match_status", "strict_match"]
    result = features[keep].copy().set_index("camera_id")
    result["pair_status"] = result.match_status.map(lambda reason: "First street: "+reason)
    for dim in PAIR_DIMENSIONS:
        result[dim+"_pair"] = UNKNOWN
    for field in ["node_id", "node_level", "intersection_pair_id", "first_segments", "cross_segments",
                  "first_streets", "cross_streets", "first_direction", "cross_direction",
                  "first_width", "cross_width", "first_lanes", "cross_lanes"]:
        result[field] = ""
    result["method"] = "Both named streets; single same-level LION node within radius; segment consensus"
    result["pair_confidence"] = "Unresolved"
    valid = features.loc[features.strict_match].copy()
    if valid.empty:
        return result.reset_index()
    valid["search_query"] = valid.search_query.map(reverse_intersection)
    reverse, other_segments = match_camera_roads(valid, roads, aliases, config)
    reverse = reverse.set_index("camera_id")
    result.loc[reverse.index, "pair_status"] = "Cross street: "+reverse.match_status
    accepted = reverse.index[reverse.strict_match]
    if not len(accepted):
        return result.reset_index()
    first = first_segments.loc[first_segments.camera_id.isin(accepted)]
    other = other_segments.loc[other_segments.camera_id.isin(accepted)]
    if set(accepted) != set(first.camera_id) or set(accepted) != set(other.camera_id):
        raise ValueError("A strict street match is missing its segment audit")
    raw = pd.concat([first, other], ignore_index=True)
    attrs = raw.drop_duplicates("segment_id").set_index("segment_id")
    endpoints = endpoints_for(roads, attrs.index, config)
    by_first = first.groupby("camera_id").segment_id.agg(set)
    by_other = other.groupby("camera_id").segment_id.agg(set)
    camera_xy = Transformer.from_crs(4326, config["crs"], always_xy=True)
    cache = {}
    def street(ids):
        key = tuple(sorted(ids))
        if key not in cache:
            rows = attrs.loc[list(key)]
            common = consensus_attributes(rows.rename(columns={
                "travel_lanes":"Number_Travel_Lanes", "traffic_direction":"TrafDir",
                "width_min_ft":"StreetWidth_Min", "width_max_ft":"StreetWidth_Max"}), config)
            cache[key] = {"segments":";".join(key), "streets":";".join(sorted(set(rows.street))),
                          "direction":common["direction_group"], "lanes":common["lanes_group"],
                          "width":width_class(rows, config)}
        return cache[key]
    orders = {"direction":GROUPS["direction"][:-1], "lanes":GROUPS["lanes"][:-1],
              "width":[f'<{config["narrow_width_m"]:g} m', f'≥{config["narrow_width_m"]:g} m']}
    for camera in accepted:
        x, y = camera_xy.transform(result.at[camera, "Longitude"], result.at[camera, "Latitude"])
        main = near_nodes(by_first[camera], endpoints, x, y, config["radius_m"])
        cross = near_nodes(by_other[camera], endpoints, x, y, config["radius_m"])
        shared = main.keys() & cross.keys()
        if len(shared) != 1:
            result.at[camera, "pair_status"] = "Multiple common nodes" if shared else "No common node"
            continue
        node = next(iter(shared))
        if main[node] & cross[node]:
            result.at[camera, "pair_status"] = "Overlapping street segment IDs"
            continue
        a, b = street(main[node]), street(cross[node])
        result.at[camera, "node_id"], result.at[camera, "node_level"] = node
        result.at[camera, "intersection_pair_id"] = "|".join([*node, *sorted((a["segments"], b["segments"]))])
        result.at[camera, "pair_status"] = "Two streets matched"
        for side, road in [("first", a), ("cross", b)]:
            for field, value in road.items():
                result.at[camera, side+"_"+field] = value
        for dim, order in orders.items():
            result.at[camera, dim+"_pair"] = unordered_pair(a[dim], b[dim], order)
    result["pair_confidence"] = np.where(result.pair_status.eq("Two streets matched"),
                                         "Both streets and node matched", "Unresolved")
    return result.reset_index()
