"""Tests for crash cleaning, KSI flagging and site clustering in preprocessing/."""

import pandas as pd

from preprocessing.config import CrashConfig, PanelConfig, SiteConfig
from preprocessing.crashes import clean_crashes, drop_ungeocoded, flag_highway, flag_ksi
from preprocessing.panel import (
    add_camera_flags,
    add_lag_features,
    add_log1p_features,
    count_site_years,
)
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
            "emotional_status": [
                "Conscious",
                "Does Not Apply",
                "Conscious",
                "Unconscious",
            ],
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


def _site_crashes(rows):
    """rows: (site_id, date, ksi, persons_injured, peds_injured)."""
    df = pd.DataFrame(rows, columns=["site_id", "crash_date", "ksi", "inj", "ped"])
    return pd.DataFrame(
        {
            "site_id": pd.array(df["site_id"], dtype="Int64"),
            "crash_date": pd.to_datetime(df["crash_date"]),
            "ksi": df["ksi"],
            "n_killed": 0,
            "number_of_persons_injured": pd.array(df["inj"], dtype="Int64"),
            "number_of_pedestrians_injured": pd.array(df["ped"], dtype="Int64"),
            "number_of_pedestrians_killed": pd.array([0] * len(df), dtype="Int64"),
            "number_of_cyclist_injured": pd.array([0] * len(df), dtype="Int64"),
            "number_of_cyclist_killed": pd.array([0] * len(df), dtype="Int64"),
        }
    )


def test_panel_lags_sum_prior_years_only():
    config = PanelConfig(first_year=2017, last_year=2021, lag_years=2)
    crashes = _site_crashes(
        [
            (0, "2017-05-01", True, 1, 1),
            (0, "2018-05-01", False, 0, 0),
            (0, "2019-05-01", True, 2, 0),
            (0, "2020-05-01", False, None, 0),
            (None, "2019-05-01", True, 1, 0),  # unassigned: ignored
        ]
    )
    sites = pd.DataFrame({"site_id": [0, 1]})

    counts = count_site_years(crashes, sites, config)
    panel = add_lag_features(counts, config).set_index(["site_id", "year"])

    assert len(counts) == 10  # complete 2 sites x 5 years grid
    assert panel.index.get_level_values("year").min() == 2019
    assert panel.loc[(0, 2019), "ksi"] == 1
    assert panel.loc[(0, 2019), "ksi_lag2"] == 1  # 2017 + 2018
    assert panel.loc[(0, 2020), "ksi_lag2"] == 1  # 2018 + 2019, not 2020
    assert panel.loc[(0, 2021), "crashes_lag2"] == 2
    assert panel.loc[(0, 2019), "ped_injured_lag2"] == 1
    assert panel.loc[(1, 2021)].filter(like="_lag2").sum() == 0


def test_camera_flags_require_activation_before_the_year():
    panel = pd.DataFrame({"site_id": [0, 0, 1], "year": [2020, 2021, 2021]})
    cameras = pd.DataFrame(
        {
            "site_id": pd.array([0, pd.NA], dtype="Int64"),
            "camera_type": ["Red Light", "Speed"],
            "activation_date": pd.to_datetime(["2020-06-01", "2016-01-01"]),
        }
    )

    out = add_camera_flags(panel, cameras)

    assert out["rlc_active"].tolist() == [False, True, False]
    assert not out["speed_active"].any()  # unmatched camera ignored
    assert out["camera_site"].tolist() == [True, True, False]


def test_flag_highway_matches_whole_words():
    crashes = pd.DataFrame(
        {
            "on_street_name": [
                "BELT PARKWAY",
                "BQE EXPWY",
                "PARKWAY PLACE X",
                "PARKWAYS",
                None,
            ]
        }
    )

    out = flag_highway(crashes, CONFIG)

    assert out["highway"].tolist() == [True, True, True, False, False]


def test_add_log1p_features_only_transforms_lags():
    panel = pd.DataFrame({"ksi": [3], "ksi_lag3": [0.0], "crashes_lag3": [9.0]})

    out = add_log1p_features(panel)

    assert out["log1p_ksi_lag3"].tolist() == [0.0]
    assert abs(out["log1p_crashes_lag3"].iloc[0] - 2.302585) < 1e-6
    assert "log1p_ksi" not in out
