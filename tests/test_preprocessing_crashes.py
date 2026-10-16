"""Tests for crash cleaning, KSI flagging and site clustering in preprocessing/."""

import pandas as pd

from preprocessing.config import CrashConfig, SiteConfig
from preprocessing.crashes import clean_crashes, drop_ungeocoded, flag_ksi
from preprocessing.sites import (
    assign_sites,
    build_site_points,
    project_coordinates,
    summarize_sites,
)

CONFIG = CrashConfig()


def _raw_crashes():
    return pd.DataFrame(
        {
            "collision_id": ["1", "2", "3", "4"],
            "crash_date": [
                "2016-12-31T00:00:00.000",
                "2017-01-01T00:00:00.000",
                "2020-06-01T00:00:00.000",
                "2024-03-01T00:00:00.000",
            ],
            "latitude": ["40.75", "40.75", "0", None],
            "longitude": ["-73.98", "-73.98", "0", None],
            "number_of_persons_killed": ["0", "1", None, "0"],
        }
    )


def test_clean_crashes_types_and_cutoff():
    out = clean_crashes(_raw_crashes(), CONFIG)

    assert out["collision_id"].tolist() == [2, 3, 4]
    assert pd.api.types.is_datetime64_any_dtype(out["crash_date"])
    assert out["number_of_persons_killed"].isna().sum() == 1  # NA kept, not zero


def test_drop_ungeocoded_removes_zero_and_missing():
    out = drop_ungeocoded(clean_crashes(_raw_crashes(), CONFIG), CONFIG)

    assert out["collision_id"].tolist() == [2]


def test_flag_ksi_uses_deaths_and_severe_proxy():
    crashes = pd.DataFrame(
        {
            "collision_id": [1, 2, 3, 4],
            "number_of_persons_killed": pd.array([1, 0, 0, 0], dtype="Int64"),
        }
    )
    persons = pd.DataFrame(
        {
            "collision_id": ["2", "2", "3", "4"],
            "person_injury": ["Injured", "Unspecified", "Injured", "Injured"],
            "complaint": ["Internal", "Does Not Apply", "Whiplash", "Whiplash"],
            "emotional_status": ["Conscious", "Does Not Apply", "Conscious", "Unconscious"],
        }
    )

    out = flag_ksi(crashes, persons, CONFIG).set_index("collision_id")

    assert out["ksi"].to_dict() == {1: True, 2: True, 3: False, 4: True}
    assert out.loc[1, "n_killed"] == 1  # no person rows: crash-table count used
    assert out.loc[2, "n_severe_injured"] == 1


def _located(feet_offsets, dates):
    """Crashes placed east of a fixed point by the given offsets in feet."""
    base = project_coordinates(
        pd.DataFrame({"latitude": [40.75], "longitude": [-73.98]}), SiteConfig()
    )
    return pd.DataFrame(
        {
            "x": base["x"].iloc[0] + pd.Series(feet_offsets, dtype=float),
            "y": base["y"].iloc[0],
            "crash_date": pd.to_datetime(dates),
        }
    )


def test_sites_cluster_nearby_points_and_freeze_before_build_end():
    config = SiteConfig(eps_ft=80, build_end="2024-01-01")
    crashes = _located(
        [0, 50, 1000, 0, 2000],
        ["2018-01-01", "2019-01-01", "2020-01-01", "2024-06-01", "2024-06-01"],
    )

    points = build_site_points(crashes, config)
    sites = summarize_sites(points)
    assigned = assign_sites(crashes, points, config)

    assert len(points) == 3  # the 2024 crashes do not define sites
    assert len(sites) == 2
    ids = assigned["site_id"]
    assert ids[0] == ids[1] == ids[3]  # 2024 crash joins the existing site
    assert ids[2] != ids[0]
    assert pd.isna(ids[4])  # new location after build_end is unassigned
    assert sites["span_ft"].max() == 25.0
