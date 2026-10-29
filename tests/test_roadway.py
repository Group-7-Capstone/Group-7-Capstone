"""Tests for LION roadway features in preprocessing/roadway.py."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from preprocessing.config import RoadwayConfig
from preprocessing.roadway import (
    ROADWAY_FEATURES,
    add_roadway_features,
    clean_segments,
    find_intersections,
    match_segments,
    site_intersection_features,
    site_segment_features,
)

CONFIG = RoadwayConfig()


def _segment(street, seg_type, traf_dir, nodes, line, lanes="  ", width=None, **kw):
    return {
        "Street": street,
        "FeatureTyp": kw.get("feature", "0"),
        "SegmentTyp": seg_type,
        "RW_TYPE": kw.get("rw", " 1"),
        "TrafDir": traf_dir,
        "StreetCode": street,
        "NodeIDFrom": f"{nodes[0]:07d}",
        "NodeIDTo": f"{nodes[1]:07d}",
        "POSTED_SPEED": kw.get("speed", "25"),
        "Number_Travel_Lanes": lanes,
        "Number_Park_Lanes": kw.get("park", "  "),
        "StreetWidth_Min": width,
        "StreetWidth_Max": width,
        "BikeLane": kw.get("bike", "  "),
        "TRUCK_ROUTE_TYPE": " ",
        "geometry": LineString(line),
    }


def _raw_segments():
    """A two-way street (A) crossing a divided avenue (B) at node 2, at (0, 0)."""
    rows = [
        # A St: undivided, two-way, generic and roadbed at once
        _segment("A", "U", "T", (1, 2), [(-500, 0), (0, 0)], " 2", 34.0, park=" 2"),
        _segment("A", "U", "T", (2, 3), [(0, 0), (500, 0)], " 2", 34.0, park=" 2"),
        # B Ave: generic centerline plus one roadbed per direction
        _segment("B", "G", "T", (4, 2), [(0, -500), (0, 0)]),
        _segment("B", "G", "T", (2, 5), [(0, 0), (0, 500)]),
        _segment(
            "B", "R", "W", (6, 7), [(-20, -500), (-20, 500)], " 3", 36.0, speed="30"
        ),
        _segment(
            "B", "R", "A", (8, 9), [(20, -500), (20, 500)], " 3", 36.0, speed="30"
        ),
        # not drivable: a park path and a shoreline
        _segment("P", "U", "T", (2, 10), [(0, 0), (-300, -300)], rw=" 6"),
        _segment("S", "U", " ", (11, 12), [(0, 50), (50, 50)], feature="2"),
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:2263")


def _nodes():
    xy = {
        1: (-500, 0),
        2: (0, 0),
        3: (500, 0),
        4: (0, -500),
        5: (0, 500),
        10: (-300, -300),
    }
    return gpd.GeoDataFrame(
        {"NODEID": [str(n) for n in xy]},
        geometry=[Point(p) for p in xy.values()],
        crs="EPSG:2263",
    )


def _sites():
    # 0 at the intersection, 1 midblock on A St, 2 far from any street
    return pd.DataFrame(
        {"site_id": [0, 1, 2], "x": [10.0, 400.0, 5000.0], "y": [10.0, 5.0, 5000.0]}
    )


def test_clean_segments_keeps_drivable_streets_and_parses_codes():
    seg = clean_segments(_raw_segments(), CONFIG)
    assert set(seg["Street"]) == {"A", "B"}
    assert seg["Number_Travel_Lanes"].dtype.kind == "f"
    assert seg.loc[seg["SegmentTyp"] == "G", "Number_Travel_Lanes"].isna().all()
    assert seg["NodeIDFrom"].iloc[0] == 1


def test_find_intersections_counts_generic_legs():
    seg = clean_segments(_raw_segments(), CONFIG)
    found = find_intersections(seg, _nodes(), CONFIG)
    # only node 2 has 3+ generic legs; the path to node 10 was dropped
    assert found["node_id"].tolist() == [2]
    assert found["legs"].tolist() == [4]


def test_site_intersection_features_uses_radius():
    seg = clean_segments(_raw_segments(), CONFIG)
    out = site_intersection_features(
        _sites(), find_intersections(seg, _nodes(), CONFIG), CONFIG
    )
    assert out["legs"].tolist() == [4, 0, 0]
    assert out["n_intersections"].tolist() == [1, 0, 0]


def test_site_segment_features_sums_divided_roadbeds():
    seg = clean_segments(_raw_segments(), CONFIG)
    out = site_segment_features(
        match_segments(_sites(), seg, CONFIG), CONFIG
    ).set_index("site_id")
    assert 2 not in out.index  # no street within the radius
    # at the intersection the divided avenue is the biggest street: 3 + 3 lanes, 36 + 36 ft
    assert out.loc[0, "lanes"] == 6
    assert out.loc[0, "width_ft"] == 72
    assert out.loc[0, "speed_limit"] == 30
    assert out.loc[0, "n_streets"] == 2
    # midblock on A St: two-way, 2 lanes, (34 - 2 parking * 8) / 2 = 9 ft lanes
    assert out.loc[1, "lanes"] == 2
    assert out.loc[1, "lane_width_ft"] == 9
    assert not out.loc[1, "one_way"]


def test_site_segment_features_drops_implausible_lane_widths():
    raw = _raw_segments()
    raw.loc[0:1, "StreetWidth_Min"] = 200.0  # 100 ft lanes
    seg = clean_segments(raw, CONFIG)
    out = site_segment_features(
        match_segments(_sites(), seg, CONFIG), CONFIG
    ).set_index("site_id")
    assert pd.isna(out.loc[1, "lane_width_ft"])


def test_add_roadway_features_fills_and_flags_missing():
    seg = clean_segments(_raw_segments(), CONFIG)
    sites = _sites()
    roadway = site_intersection_features(
        sites, find_intersections(seg, _nodes(), CONFIG), CONFIG
    ).merge(
        site_segment_features(match_segments(sites, seg, CONFIG), CONFIG), how="left"
    )
    panel = pd.DataFrame({"site_id": [0, 0, 1, 2], "year": [2024, 2025, 2024, 2024]})
    out = add_roadway_features(panel, roadway)
    assert len(out) == len(panel)
    assert out[ROADWAY_FEATURES].notna().all().all()
    assert out["roadway_missing"].tolist() == [0, 0, 0, 1]
    assert out["legs_4"].tolist() == [1, 1, 0, 0]
    assert out["speed_30_35"].tolist() == [1, 1, 0, 0]
