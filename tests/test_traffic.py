"""Tests for preprocessing/traffic.py: traffic counts to site volumes."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from preprocessing.config import TrafficConfig
from preprocessing.traffic import (
    add_traffic_features,
    daily_volumes,
    impute_log_traffic,
    segment_volume,
    site_traffic_volume,
)

CONFIG = TrafficConfig()


def _day(requestid, segmentid, direction, date, vol_per_interval, n=96):
    """Raw-style rows (all strings, as SODA returns them) for one count day."""
    ts = pd.Timestamp(date) + pd.to_timedelta(15 * np.arange(n), unit="min")
    return pd.DataFrame(
        {
            "requestid": str(requestid),
            "segmentid": str(segmentid),
            "direction": direction,
            "yr": ts.year.astype(str),
            "m": ts.month.astype(str),
            "d": ts.day.astype(str),
            "hh": ts.hour.astype(str),
            "mm": ts.minute.astype(str),
            "vol": str(vol_per_interval),
        }
    )


def test_daily_volumes_keeps_only_complete_weekdays():
    counts = pd.concat(
        [
            _day(1, 10, "NB", "2023-06-05", 10),  # Monday, complete
            _day(1, 10, "NB", "2023-06-06", 10, n=50),  # partial day
            _day(1, 10, "NB", "2023-06-10", 10),  # Saturday
        ]
    )
    out = daily_volumes(counts, CONFIG)
    assert out["date"].tolist() == [pd.Timestamp("2023-06-05")]
    assert out["vol"].tolist() == [960]


def test_daily_volumes_sums_duplicate_intervals():
    counts = pd.concat(
        [_day(1, 10, "NB", "2023-06-05", 10), _day(1, 10, "NB", "2023-06-05", 5)]
    )
    assert daily_volumes(counts, CONFIG)["vol"].tolist() == [96 * 15]


def test_segment_volume_sums_directions_and_takes_median_of_studies():
    daily = pd.DataFrame(
        {
            "requestid": [1, 1, 1, 2, 3],
            "segmentid": [10, 10, 10, 10, 10],
            "direction": ["NB", "NB", "SB", "NB", "NB"],
            "date": pd.to_datetime(
                ["2023-06-05", "2023-06-06", "2023-06-05", "2020-01-06", "2019-01-07"]
            ),
            "vol": [100.0, 300.0, 50.0, 1000.0, 10.0],
        }
    )
    out = segment_volume(daily).set_index("segmentid")
    # Study 1: NB mean 200 + SB 50 = 250; studies 2 and 3: 1000 and 10.
    assert out.loc[10, "adt"] == 250
    assert out.loc[10, "n_studies"] == 3


def test_site_takes_the_busiest_counted_segment_within_the_radius():
    segments = gpd.GeoDataFrame(
        {"adt": [5000.0, 20000.0, 90000.0]},
        geometry=[
            LineString([(-100, 0), (100, 0)]),
            LineString([(0, -100), (0, 100)]),
            LineString([(300, -100), (300, 100)]),  # beyond the radius
        ],
        crs="EPSG:2263",
    )
    sites = pd.DataFrame({"site_id": [1, 2], "x": [0.0, 5000.0], "y": [0.0, 0.0]})
    out = site_traffic_volume(sites, segments, CONFIG).set_index("site_id")["adt"]
    assert out[1] == 20000
    assert np.isnan(out[2])


def test_add_traffic_features_flags_uncounted_and_zero_sites():
    traffic = pd.DataFrame({"site_id": [1, 2, 3], "adt": [1000.0, np.nan, 0.0]})
    panel = pd.DataFrame({"site_id": [1, 1, 2, 3], "year": [2022, 2023, 2022, 2022]})
    out = add_traffic_features(panel, traffic)
    assert len(out) == len(panel)
    assert out["log_traffic"].iloc[0] == pytest.approx(np.log(1000))
    assert out["log_traffic"].iloc[2:].isna().all()
    assert out["traffic_counted"].tolist() == [1, 1, 0, 0]


def test_impute_log_traffic_keeps_counts_and_fills_the_rest_from_the_fit():
    panel = pd.DataFrame(
        {"lanes": [1.0, 2.0, 3.0, 4.0], "log_traffic": [7.0, 8.0, 9.0, np.nan]}
    )
    out = impute_log_traffic(panel, ["lanes"])
    assert out["log_traffic_filled"].tolist() == pytest.approx([7, 8, 9, 10])
    assert out["log_traffic_imputed"].tolist() == pytest.approx([7, 8, 9, 10])
