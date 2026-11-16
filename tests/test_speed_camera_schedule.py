"""Checks on DOT's speed-camera operating schedule in data/camera_data/."""

import pandas as pd

from data.reader import read_camera_data


def test_schedule_periods_are_contiguous_and_open_ended():
    schedule = read_camera_data("speed_camera_schedule")
    start = pd.to_datetime(schedule["start_date"])
    end = pd.to_datetime(schedule["end_date"])
    order = start.argsort()
    start, end = start.iloc[order].reset_index(drop=True), end.iloc[order].reset_index(drop=True)

    # Each period ends the day before the next starts, and only the latest is open.
    assert (end.iloc[:-1] + pd.Timedelta(days=1)).tolist() == start.iloc[1:].tolist()
    assert end.iloc[:-1].notna().all() and pd.isna(end.iloc[-1])
