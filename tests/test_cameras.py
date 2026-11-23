"""Tests for schedule-aware camera activation dates (preprocessing/cameras.py)."""

import numpy as np
import pandas as pd

from preprocessing.cameras import (
    FALLBACK_METHOD,
    SUSTAINED_METHOD,
    add_activation_confidence,
    drop_ticket_fragments,
    infer_activations,
    permitted_hours_per_week,
    permitted_months,
    speed_enforced_weeks,
    weekly_ticket_counts,
)
from preprocessing.config import CameraActivationConfig

CONFIG = CameraActivationConfig()


def _schedule():
    return pd.DataFrame(
        {
            "start_date": ["2014-01-01", "2019-07-01", "2022-08-01"],
            "end_date": ["2019-06-30", "2022-07-31", None],
            "hours_of_opp": ["7am-5pm", "6am-10pm", "All"],
            "days_of_opp": ["Mon-Fri", "Mon-Fri", "All"],
            "months_of_opp": ["September-June", "All", "All"],
        }
    )


def _weekly(code, query, weeks, tickets):
    return pd.DataFrame(
        {
            "violation_code": code,
            "search_query": query,
            "week": pd.to_datetime(weeks),
            "tickets": tickets,
        }
    )


def _enforced(weeks, hours=50.0):
    return pd.DataFrame(
        {"week": pd.to_datetime(weeks), "enforced": True, "permitted_hours": hours}
    )


def test_weekly_ticket_counts_skips_tickets_already_counted():
    tickets = pd.DataFrame(
        {
            "summons_number": ["1", "2", "3", "4"],
            "violation_code": ["36", "36", "36", "36"],
            "street_name": ["NB MAIN ST", "NB MAIN ST", "NB MAIN ST", None],
            "intersecting_street": ["@ 1 AV", "@ 1 AV", "@ 1 AV", "@ 1 AV"],
            # Wed and Sun of one week, then the next Monday
            "issue_date": ["2020-01-08", "2020-01-12", "2020-01-13", "2020-01-08"],
        }
    )
    counts, seen = weekly_ticket_counts(tickets, np.array([1]), CONFIG)
    assert counts["week"].tolist() == [
        pd.Timestamp("2020-01-06"),
        pd.Timestamp("2020-01-13"),
    ]
    assert counts["tickets"].tolist() == [1, 1]
    assert counts["search_query"].iloc[0] == "MAIN ST AND 1 AV, NEW YORK, NY"
    assert set(seen) == {1, 2, 3, 4}


def test_schedule_text_parses_to_months_and_weekly_hours():
    assert permitted_months("September-June") == {9, 10, 11, 12, 1, 2, 3, 4, 5, 6}
    assert permitted_months("All") == set(range(1, 13))
    assert permitted_hours_per_week("7am-5pm", "Mon-Fri") == 50
    assert permitted_hours_per_week("6am-10pm", "Mon-Fri") == 80
    assert permitted_hours_per_week("All", "All") == 168


def test_speed_enforced_weeks_skip_summer_and_holidays():
    weeks = pd.date_range("2018-06-04", "2018-09-24", freq="7D")
    tickets = [100] * len(weeks)
    tickets[list(weeks).index(pd.Timestamp("2018-09-10"))] = 5  # a holiday week
    out = speed_enforced_weeks(
        _weekly(36, "A", weeks, tickets), _schedule(), CONFIG
    ).set_index("week")
    assert out.loc["2018-06-04", "enforced"]
    assert out.loc["2018-06-04", "permitted_hours"] == 50
    assert not out.loc["2018-07-16", "enforced"]  # summer, before July 2019
    assert not out.loc["2018-09-10", "enforced"]
    assert out.loc["2018-09-17", "enforced"]


def test_a_summer_break_restarts_a_speed_camera_only_without_the_schedule():
    june = pd.date_range("2018-06-11", periods=2, freq="7D")
    sept = pd.date_range("2018-09-03", periods=3, freq="7D")
    weekly = _weekly(36, "A", june.append(sept), 50)
    # the summer weeks are simply not in the enforced timeline
    out = infer_activations(weekly, _enforced(june.append(sept)), CONFIG).iloc[0]
    assert out["inferred_activation_date"] == pd.Timestamp("2018-06-11")
    assert out["activation_date_unadjusted"] == pd.Timestamp("2018-09-03")
    assert out["activation_method"] == SUSTAINED_METHOD


def test_hours_scaled_date_needs_more_tickets_in_longer_weeks():
    weeks = pd.date_range("2023-01-02", periods=6, freq="7D")
    # 10 a week clears 5, but not the 168-hour bar of 5 * 168 / 50 = 16.8
    tickets = [10, 10, 10, 20, 20, 20]
    out = infer_activations(
        _weekly(36, "A", weeks, tickets), _enforced(weeks, hours=168.0), CONFIG
    ).iloc[0]
    assert out["inferred_activation_date"] == pd.Timestamp("2023-01-02")
    assert out["activation_date_hours_scaled"] == pd.Timestamp("2023-01-23")


def test_a_camera_without_a_sustained_run_falls_back_to_its_first_ticket():
    weekly = _weekly(7, "B", ["2020-03-02", "2020-03-16"], [9, 9])
    out = infer_activations(weekly, _enforced([]), CONFIG).iloc[0]
    assert out["camera_type"] == "Red Light"
    assert out["inferred_activation_date"] == pd.Timestamp("2020-03-02")
    assert out["activation_method"] == FALLBACK_METHOD


def test_drop_ticket_fragments_keeps_sustained_and_busy_locations():
    activations = pd.DataFrame(
        {
            "activation_method": [FALLBACK_METHOD, FALLBACK_METHOD, SUSTAINED_METHOD],
            "total_tickets_issued": [12, 500, 12],
        }
    )
    out = drop_ticket_fragments(activations, CONFIG)
    assert out["total_tickets_issued"].tolist() == [500, 12]


def test_activation_confidence_flags_censoring_and_schedule_changes():
    activations = pd.DataFrame(
        {
            "violation_code": [36, 36, 7, 36, 36],
            "inferred_activation_date": pd.to_datetime(
                ["2016-06-13", "2019-07-08", "2019-07-08", "2020-03-02", "2020-03-02"]
            ),
            "activation_method": [SUSTAINED_METHOD] * 4 + [FALLBACK_METHOD],
        }
    )
    out = add_activation_confidence(activations, _schedule(), CONFIG)
    assert out["left_censored"].tolist() == [True, False, False, False, False]
    # only speed cameras are affected by the speed-camera schedule
    assert out["activation_confidence"].tolist() == [
        "high",
        "medium",
        "high",
        "high",
        "low",
    ]
