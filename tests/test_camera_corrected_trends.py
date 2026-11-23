"""Numerical controls for schedule boundaries, calendar shocks and holdout leakage."""
import numpy as np
import pandas as pd
import pytest

from data.reader import read_camera_data
from preprocessing.camera_corrected_trends import (
    calendar_hours,
    correct_calendar_weeks,
    first_year_outcomes,
)
from preprocessing.camera_shared_trends import DEFAULTS
from preprocessing.config import TICKET_REVISION_CONFIG
from validation.camera_corrected_trends import (
    camera_halves,
    compare_feature_groups,
    discover_clusters,
    feature_rules,
)

CFG = {**DEFAULTS, **TICKET_REVISION_CONFIG, "minimum_references": 1}


def test_schedule_daily_boundaries_and_unknown_dates():
    schedule = read_camera_data("speed_camera_schedule")
    dates = pd.to_datetime(["2013-12-23", "2016-06-27", "2018-07-02", "2019-07-01", "2022-08-01"])
    result = calendar_hours(dates, schedule)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].tolist() == [40, 0, 80, 168]
    schedule.loc[1, "end_date"] = "2022-08-02"
    schedule.loc[2, "start_date"] = "2022-08-03"
    assert calendar_hours(["2022-08-01"], schedule).iloc[0] == 152
    with pytest.raises(ValueError, match="Monday"):
        calendar_hours(["2022-08-02"], schedule)


def inputs():
    summary = pd.DataFrame({"camera_id": ["old1", "old2", "new", "holdout"], "camera_type": "Speed",
        "inferred_activation_date": ["2020-01-06", "2020-01-06", "2023-01-02", "2020-01-06"],
        "activation_confidence": "high", "left_censored": False})
    weekly = pd.DataFrame([{"camera_id": name, "week_start": pd.Timestamp("2023-01-02") + pd.Timedelta(weeks=t),
        "observed_ticket_count": count * (1 if t == 0 else .5), "is_complete_week": True, "source_complete": True}
        for name, count in zip(summary.camera_id, [840, 840, 1680, 840000]) for t in range(2)])
    return summary, weekly


def test_common_shock_cancels_and_holdout_never_enters_reference():
    summary, weekly = inputs()
    corrected = correct_calendar_weeks(summary, weekly, read_camera_data("speed_camera_schedule"), ["old1", "old2"], CFG)
    assert corrected.loc[corrected.camera_id.eq("new"), "corrected_rate"].tolist() == [2, 2]
    # A reference camera's own count is left out.
    assert corrected.loc[corrected.camera_id.eq("old1"), "reference_count"].tolist() == [1, 1]
    weekly.loc[weekly.camera_id.eq("holdout"), "observed_ticket_count"] *= 100
    changed = correct_calendar_weeks(summary, weekly, read_camera_data("speed_camera_schedule"), ["old1", "old2"], CFG)
    pd.testing.assert_series_equal(corrected.reference_mean, changed.reference_mean)


def test_zero_reference_and_missing_source_remain_missing():
    summary, weekly = inputs()
    weekly.loc[weekly.camera_id.str.startswith("old"), "observed_ticket_count"] = 0
    result = correct_calendar_weeks(summary, weekly, read_camera_data("speed_camera_schedule"), ["old1", "old2"], CFG)
    assert result.corrected_rate.isna().all()
    summary, weekly = inputs()
    weekly.loc[weekly.camera_id.eq("new"), "source_complete"] = False
    result = correct_calendar_weeks(summary, weekly, read_camera_data("speed_camera_schedule"), ["old1", "old2"], CFG)
    assert result.loc[result.camera_id.eq("new"), "corrected_rate"].isna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        correct_calendar_weeks(summary, pd.concat([weekly, weekly.iloc[:1]]), read_camera_data("speed_camera_schedule"), [], CFG)


def test_decline_window_censoring_and_coverage():
    summary = pd.DataFrame({"camera_id": ["eligible", "censored", "missing", "zero"], "camera_type": "Speed",
        "inferred_activation_date": "2023-01-02", "activation_confidence": "high", "left_censored": [False, True, False, False]})
    rows = pd.DataFrame([{"camera_id": name, "week_start": pd.Timestamp("2023-01-02") + pd.Timedelta(weeks=t),
        "corrected_rate": (0 if name == "zero" and t < 13 else 2 if t < 13 else 1),
        "is_complete_week": not (name == "missing" and t == 20), "source_complete": True}
        for name in summary.camera_id for t in range(52)])
    outcomes, _ = first_year_outcomes(summary, rows, CFG)
    assert outcomes.loc["eligible", "decline_52w_pct"] == 50
    assert outcomes.outcome_status.tolist() == ["eligible", "left_censored", "incomplete_first_year", "zero_baseline"]
    assert outcomes.decline_52w_pct.notna().sum() == 1


def test_split_is_order_invariant_and_disjoint():
    ids = [f"c{i}" for i in range(31)]
    a = camera_halves(ids, 42)
    pd.testing.assert_series_equal(a.sort_index(), camera_halves(ids[::-1], 42).sort_index())
    assert a.value_counts().to_dict() == {"B": 16, "A": 15}
    with pytest.raises(ValueError):
        camera_halves(["same", "same"], 42)


def test_cluster_selection_sees_only_A(monkeypatch):
    import validation.camera_corrected_trends as module
    ids = pd.Index(["a", "b"])
    outcomes = pd.DataFrame({"outcome_status": "eligible"}, index=ids)
    panel = pd.DataFrame([np.arange(1, 53), np.arange(52, 0, -1)], index=ids)
    split = pd.Series(["A", "B"], index=ids)
    seen = []
    monkeypatch.setattr(module, "select_shape_model", lambda x, config: (seen.append(x.copy()) or None, pd.DataFrame()))
    _, _, info = discover_clusters(outcomes, panel, split, CFG)
    panel.loc["b"] *= 100
    discover_clusters(outcomes, panel, split, CFG)
    pd.testing.assert_frame_equal(seen[0], seen[1])
    assert seen[0].index.tolist() == ["a"]
    assert info["mode"] == "decline_52w_pct"


def test_frozen_bins_and_within_borough_year_comparisons():
    rows = []
    for half in ["A", "B"]:
        for borough, baseline in [("Manhattan", 10), ("Queens", 50)]:
            for group in ["one-way", "two-way"]:
                for i in range(12):
                    rows.append({"split": half, "borough": borough, "activation_year": 2020, "direction_pair": group,
                                 "lanes": i + 1, "width_ft": 30 + i, "ped_volume": 100 * i,
                                 "decline_52w_pct": baseline + (5 if group == "two-way" else 0) + (i % 3)})
    frame = pd.DataFrame(rows)
    rules = feature_rules(frame.loc[frame.split.eq("A")], CFG)
    assert rules["lanes"]["cutpoint"] == 6.5
    result, detail = compare_feature_groups(frame, {"direction_pair": rules["direction_pair"]}, CFG)
    assert result.iloc[0].A_difference_pp == pytest.approx(5)
    assert result.iloc[0].B_difference_pp == pytest.approx(5)
    assert result.iloc[0].replicated
    assert len(detail) == 4
    # Completely absent B support must not masquerade as a validated contrast.
    empty, _ = compare_feature_groups(frame.loc[frame.split.eq("A")], {"direction_pair": rules["direction_pair"]}, CFG)
    assert empty.iloc[0].strata == 0 and not empty.iloc[0].replicated


def test_cluster_labels_are_ordered_on_A_decline_and_frozen_for_B(monkeypatch):
    from types import SimpleNamespace

    import validation.camera_corrected_trends as module

    ids = pd.Index(["a1", "a2", "b1", "b2"])
    outcomes = pd.DataFrame({"outcome_status": "eligible", "decline_52w_pct": [50, -10, -99, 99]}, index=ids)
    panel = pd.DataFrame([np.arange(1, 53)] * 4, index=ids)
    split = pd.Series(["A", "A", "B", "B"], index=ids)
    model = SimpleNamespace(n_clusters=2, labels_=np.array([1, 0]), predict=lambda x: np.array([1, 0, 1, 0]))
    monkeypatch.setattr(module, "select_shape_model", lambda x, config: (model, pd.DataFrame()))
    labels, _, _ = discover_clusters(outcomes, panel, split, CFG)
    assert labels.tolist() == [2, 1, 2, 1]


def test_shared_locator_direction_pairs_preserve_ambiguity_and_divided_roads():
    from preprocessing.camera_trend_sites import named_direction_pairs

    located = pd.DataFrame({"site_id": ["valid", "ambiguous", "midblock"],
        "search_query": "MAIN AV AND OAK ST, NEW YORK, NY",
        "geocode_method": ["lion_intersection", "lion_intersection", "lion_midblock"],
        "geocode_confidence": ["high", "medium", "high"]})
    names = pd.DataFrame({"NODEID": [1, 1], "STNAME": ["MAIN AVENUE", "OAK STREET"]})
    pairs = pd.DataFrame({"site_id": ["valid"] * 3, "Street": ["MAIN AVENUE", "MAIN AVENUE", "OAK STREET"], "TrafDir": ["T", "A", "W"]})
    result = named_direction_pairs(located, names, pairs)
    assert result.loc["valid"] == "one-way × two-way"
    assert result.loc[["ambiguous", "midblock"]].isna().all()
