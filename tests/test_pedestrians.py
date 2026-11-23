"""Tests for preprocessing/pedestrians.py: NYC Walks volumes joined to sites."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from preprocessing.config import PedestrianConfig
from preprocessing.panel import FEATURE_SETS
from preprocessing.pedestrians import (
    PEDESTRIAN_FEATURES,
    add_pedestrian_features,
    segment_volume,
    site_ped_volume,
)

CONFIG = PedestrianConfig(radius_ft=100.0)


def _segments(rows: list[tuple[LineString, float, float, float]]) -> gpd.GeoDataFrame:
    """Raw-style NYC Walks rows: geometry and the three weekday peak hours."""
    am, md, pm = zip(*[r[1:] for r in rows]) if rows else ((), (), ())
    return gpd.GeoDataFrame(
        {"predwkdyAM": am, "predwkdyMD": md, "predwkdyPM": pm},
        geometry=[r[0] for r in rows],
        crs="EPSG:2263",
    )


def _sites() -> pd.DataFrame:
    # An intersection at the origin, a mid-block site and a site far from any segment.
    return pd.DataFrame(
        {"site_id": [1, 2, 3], "x": [0.0, 1000.0, 5000.0], "y": [0.0, 0.0, 5000.0]}
    )


def _volumes() -> pd.DataFrame:
    segments = _segments(
        [
            # Two crosswalks at the intersection; the busier one describes it.
            (LineString([(-30, 20), (30, 20)]), 100, 100, 100),
            (LineString([(20, -30), (20, 30)]), 300, 200, 100),
            # A busy segment just beyond the radius doesn't count.
            (LineString([(130, -30), (130, 30)]), 5000, 5000, 5000),
            # A sidewalk along the mid-block site, 40 ft off the centerline.
            (LineString([(900, 40), (1100, 40)]), 50, 60, 70),
        ]
    )
    return site_ped_volume(_sites(), segment_volume(segments, CONFIG), CONFIG)


def test_site_takes_the_busiest_segment_within_the_radius():
    volume = _volumes().set_index("site_id")["ped_volume"]
    assert volume[1] == 600


def test_midblock_sites_match_their_sidewalks():
    volume = _volumes().set_index("site_id")["ped_volume"]
    assert volume[2] == 180


def test_site_with_no_segment_nearby_is_na():
    volume = _volumes().set_index("site_id")["ped_volume"]
    assert np.isnan(volume[3])


def test_segments_missing_an_hour_are_dropped():
    segments = _segments(
        [
            (LineString([(0, 0), (10, 0)]), 1, 2, 3),
            (LineString([(0, 5), (10, 5)]), np.nan, 2, 3),
        ]
    )
    out = segment_volume(segments, CONFIG)
    assert out["ped_volume"].tolist() == [6]


def test_missing_sites_get_the_median_and_a_flag():
    ped = pd.DataFrame({"site_id": [1, 2, 3], "ped_volume": [600.0, 180.0, np.nan]})
    panel = pd.DataFrame({"site_id": [1, 1, 2, 3], "year": [2022, 2023, 2022, 2022]})

    out = add_pedestrian_features(panel, ped)

    assert len(out) == len(panel)
    assert out["log_ped_volume"].tolist() == pytest.approx(
        np.log1p([600, 600, 180, 390]).tolist()
    )
    assert out["ped_missing"].tolist() == [0, 0, 0, 1]


def test_pedestrian_feature_set_lists_the_model_columns():
    assert FEATURE_SETS["pedestrian"] == PEDESTRIAN_FEATURES
