"""Tests for traffic volume features in preprocessing/traffic_volume.py."""

import numpy as np
import pandas as pd

from preprocessing.config import TrafficVolumeConfig
from preprocessing.traffic_volume import (
    add_traffic_volume,
    daily_counts,
    segment_daily_volume,
    site_traffic_volume,
)

CONFIG = TrafficVolumeConfig(intervals_per_day=4)


def _day(segment, direction, day, vols, request=1):
    """Raw SODA-style rows (all strings) for one count day, one row per interval."""
    return [
        {
            "requestid": str(request),
            "segmentid": str(segment),
            "direction": direction,
            "yr": "2024",
            "m": "5",
            "d": str(day),
            "hh": str(i // 4),
            "mm": str(15 * (i % 4)),
            "vol": str(v),
        }
        for i, v in enumerate(vols)
    ]


def test_daily_counts_keeps_complete_nonzero_days_once():
    rows = (
        _day(10, "NB", 1, [1, 2, 3, 4])
        + _day(10, "NB", 1, [1])  # the first interval recorded twice
        + _day(10, "NB", 2, [5, 5])  # partial day
        + _day(10, "NB", 3, [0, 0, 0, 0])  # failed counter
    )
    out = daily_counts(pd.DataFrame(rows), CONFIG)
    assert out[["segmentid", "direction", "volume"]].values.tolist() == [[10, "NB", 10]]


def test_segment_daily_volume_medians_days_then_adds_directions():
    daily = pd.DataFrame(
        {
            "segmentid": [10, 10, 10, 10, 20],
            "direction": ["NB", "NB", "NB", "SB", "EB"],
            "volume": [100, 200, 900, 50, 70],
        }
    )
    out = segment_daily_volume(daily)
    assert out.to_dict() == {10: 250, 20: 70}


def test_site_traffic_volume_adds_divided_roadbeds_and_takes_busiest_street():
    pairs = pd.DataFrame(
        {
            "site_id": [1, 1, 1, 1, 2],
            "StreetCode": ["AV", "AV", "ST", "ST", "ST"],
            "TrafDir": ["W", "A", "T", "T", "T"],
            "SegmentID": ["0000001", "0000002", "0000003", "0000004", "0000004"],
        }
    )
    volume = pd.Series({1: 300.0, 2: 400.0, 3: 500.0})  # segment 4 not counted
    out = site_traffic_volume(pairs, volume)
    # site 1: avenue roadbeds 300 + 400 beat the street's 500; site 2 has no count
    assert out.values.tolist() == [[1, 700.0]]


def test_add_traffic_volume_leaves_uncounted_sites_na():
    panel = pd.DataFrame({"site_id": [1, 2], "year": [2024, 2024]})
    out = add_traffic_volume(
        panel, pd.DataFrame({"site_id": [1], "daily_volume": [8.0]})
    )
    assert out["log2_volume"].iloc[0] == 3
    assert np.isnan(out["log2_volume"].iloc[1])
