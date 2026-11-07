"""Tests for camera approach directions parsed from ticket text in preprocessing/."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from preprocessing.base import normalize_street
from preprocessing.camera_directions import (
    add_bearings,
    camera_search_query,
    combine_locations,
    enforced_approaches,
    parse_direction,
    summarize_locations,
)
from preprocessing.config import CameraDirectionConfig

CONFIG = CameraDirectionConfig()


def test_parse_direction_handles_prefix_parentheses_and_split_fields():
    street = pd.Series(
        [
            "ASTORIA BLVD S (E/B)",
            "BEACH CHANNEL DR (W/",
            "NB OCEAN PKWY @ AVEN",
            "WB N CONDUIT AVE @ S",
            "NWB REMSEN AVE @ LIN",
            "HORACE HARDING EXPWY",
            "W BROADWAY @ CANAL S",
            "PIKE ST",
        ]
    )
    cross = pd.Series(
        [
            "@ 77TH ST",
            "     B) @ B 140TH ST",
            "UE I",
            "PRINGFIELD BLVD",
            "DEN BLVD",
            "(E/B) @ 173RD ST",
            "T",
            None,
        ]
    )
    assert parse_direction(street, cross).tolist()[:6] == [
        "E",
        "W",
        "N",
        "W",
        "NW",
        "E",
    ]
    # "W BROADWAY" is a street name, not westbound
    assert parse_direction(street, cross).iloc[6:].isna().all()


def test_camera_search_query_matches_activation_notebook_key():
    street = pd.Series(["NB UTOPIA PKWY @ 17T", "NORTHERN BLVD (E/B)"])
    cross = pd.Series(["H AVE", "@ DOUGLASTON PKWY"])
    assert camera_search_query(street, cross).tolist() == [
        "UTOPIA PKWY AND 17T H AVE, NEW YORK, NY",
        "NORTHERN BLVD AND DOUGLASTON PKWY, NEW YORK, NY",
    ]


def _tickets():
    return pd.DataFrame(
        {
            "violation_code": ["7", "7", "7", "36", "36", "36"],
            "street_name": [
                "MAIN ST (E/B)",
                "MAIN ST (E/B)",
                "MAIN ST (W/B)",
                "NB 1 AVE @ E 7 ST",
                "NB 1 AVE @ E 7 ST",
                None,
            ],
            "intersecting_street": ["@ 1 AVE", "@ 1 AVE", "@ 1 AVE", "", "", "X"],
            "issue_date": [
                "2020-01-05",
                "2021-03-01",
                "2022-07-01",
                "2015-06-01",
                "2019-02-01",
                "2019-02-01",
            ],
        }
    )


def test_summarize_locations_counts_and_drops_early_or_blank_rows():
    out = summarize_locations(_tickets(), CONFIG).set_index("street_name")
    assert out.loc["MAIN ST (E/B)", "tickets"] == 2
    assert out.loc["MAIN ST (E/B)", "first_ticket_date"] == pd.Timestamp("2020-01-05")
    # the 2015 ticket is before start_date and the blank street is dropped
    assert out.loc["NB 1 AVE @ E 7 ST", "tickets"] == 1
    assert len(out) == 3


def test_combine_locations_merges_tables():
    part = summarize_locations(_tickets(), CONFIG)
    out = combine_locations([part, part])
    assert out["tickets"].sum() == 2 * part["tickets"].sum()
    assert len(out) == len(part)


def test_enforced_approaches_one_row_per_direction_with_confidence():
    locations = summarize_locations(_tickets(), CONFIG)
    out = enforced_approaches(locations, CameraDirectionConfig(min_direction_share=0.4))
    red = out[out["violation_code"] == 7].set_index("direction")
    assert red["search_query"].unique().tolist() == ["MAIN ST AND 1 AVE, NEW YORK, NY"]
    assert red.loc["E", "tickets"] == 2
    assert red.loc["E", "direction_share"] == 2 / 3
    assert red.loc["E", "direction_confidence"] == "high"
    assert red.loc["W", "direction_confidence"] == "low"  # 1/3 < 0.4
    assert (out["direction_method"] == "ticket_location_text").all()
    assert out.loc[out["violation_code"] == 36, "camera_type"].tolist() == ["Speed"]


def test_normalize_street_makes_lion_and_ticket_spellings_equal():
    lion = pd.Series(
        ["EAST 137 STREET", "UTOPIA PARKWAY", "3 AVENUE", "BEACH 140 STREET"]
    )
    ticket = pd.Series(["E 137 TH ST", "UTOPIA PKWY", "THIRD AVE", "B 140TH ST"])
    assert normalize_street(lion).tolist() == normalize_street(ticket).tolist()


def _approaches():
    # cameras at (0, 0): one on a north-south avenue tilted 30 degrees east of north
    # (like Manhattan), one on a geocode that landed on the wrong street
    return pd.DataFrame(
        {
            "search_query": [
                "MAIN AVE AND 1 ST, NEW YORK, NY",
                "MAIN AVE AND 1 ST, NEW YORK, NY",
                "ELM ST AND 2 AVE, NEW YORK, NY",
            ],
            "direction": ["N", "S", "E"],
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
        }
    )


def _segments():
    tilt = np.radians(30)
    dx, dy = 500 * np.sin(tilt), 500 * np.cos(tilt)
    return gpd.GeoDataFrame(
        {
            "Street": ["MAIN AVENUE", "1 STREET"],
            "SegmentID": ["0000001", "0000002"],
        },
        # MAIN AVENUE is drawn from north to south; the cross street runs east-west
        geometry=[
            LineString([(dx, dy), (-dx, -dy)]),
            LineString([(-500, 5), (500, 5)]),
        ],
        crs="EPSG:2263",
    )


def test_add_bearings_orients_the_cameras_own_street():
    out = add_bearings(_approaches(), _segments(), CONFIG)
    north, south, wrong = out.itertuples()
    assert north.lion_street == "MAIN AVENUE"
    assert north.bearing_deg == 30.0
    assert north.bearing_offset_deg == 30.0
    assert north.bearing_confidence == "high"
    assert south.bearing_deg == 210.0
    # "ELM ST" isn't near the camera: no bearing rather than the cross street's
    assert np.isnan(wrong.bearing_deg)
    assert wrong.bearing_confidence == "none"
    assert (out["bearing_method"] == "lion_street_tangent").all()


def test_add_bearings_flags_diagonal_streets_low_and_ignores_kept_prefix():
    tilt = np.radians(70)
    dx, dy = 500 * np.sin(tilt), 500 * np.cos(tilt)
    segments = gpd.GeoDataFrame(
        {"Street": ["BOSTON ROAD"], "SegmentID": ["0000003"]},
        geometry=[LineString([(-dx, -dy), (dx, dy)])],
        crs="EPSG:2263",
    )
    approaches = pd.DataFrame(
        {
            "search_query": ["NWB BOSTON RD AND X AVE, NEW YORK, NY"],
            "direction": ["N"],
            "x": [0.0],
            "y": [0.0],
        }
    )
    out = add_bearings(approaches, segments, CONFIG).iloc[0]
    assert out.lion_street == "BOSTON ROAD"
    assert out.bearing_deg == 70.0
    assert out.bearing_confidence == "low"


def test_normalize_street_keeps_numbered_street_types():
    lion = pd.Series(["48 STREET", "251 STREET", "21 AVENUE", "1 AVENUE"])
    ticket = pd.Series(["48TH S T", "251 ST ST", "21ST AV E", "1ST AVE"])
    assert normalize_street(ticket).tolist() == normalize_street(lion).tolist()
    assert normalize_street(lion).tolist() == ["48ST", "251ST", "21AV", "1AV"]
    # a numbered street followed by more text keeps its type
    assert (
        normalize_street(pd.Series(["65TH ST AND 21ST AVE"])).iloc[0] == "65STAND21AV"
    )
