"""Calendar correction and fixed-year outcomes for #58; no fitting or splits."""
import logging

import numpy as np
import pandas as pd

from preprocessing.cameras import permitted_hours_per_week, permitted_months
from preprocessing.config import TICKET_REVISION_CONFIG

log = logging.getLogger(__name__)


def calendar_hours(weeks, schedule):
    """Sum permitted hours day by day, including boundary and summer weeks.

    These are scheduled opportunities, not device uptime. Unknown schedule days
    stay missing; weekends and out-of-season days within a known period are zero.
    Reuses #23's parsing without its outcome-based holiday classification.
    """
    weeks = pd.DatetimeIndex(pd.to_datetime(weeks)).unique().sort_values()
    if weeks.empty or (weeks.dayofweek != 0).any():
        raise ValueError("Supply nonempty Monday-start calendar weeks")
    days = pd.date_range(weeks.min(), weeks.max() + pd.Timedelta(days=6))
    hours = pd.Series(np.nan, index=days)
    for row in schedule.itertuples(index=False):
        end = days.max() if pd.isna(row.end_date) else pd.Timestamp(row.end_date)
        in_period = (days >= pd.Timestamp(row.start_date)) & (days <= end)
        if hours.loc[in_period].notna().any():
            raise ValueError("Overlapping schedule periods")
        if row.days_of_opp not in ("All", "Mon-Fri"):
            raise ValueError("Unsupported schedule weekdays")
        n_days = 7 if row.days_of_opp == "All" else 5
        daily = permitted_hours_per_week(row.hours_of_opp, row.days_of_opp) / n_days
        allowed = days.month.isin(permitted_months(row.months_of_opp))
        allowed &= (days.dayofweek < 5) | (n_days == 7)
        hours.loc[in_period] = np.where(allowed[in_period], daily, 0)
    return hours.groupby(days.to_period("W-SUN").start_time).sum(min_count=7).reindex(weeks)


def correct_calendar_weeks(summary, weekly, schedule, reference_ids, config=TICKET_REVISION_CONFIG):
    """Normalize against established A cameras only, leaving each camera out of its own mean."""
    if not summary.camera_id.is_unique or weekly.duplicated(["camera_id", "week_start"]).any():
        raise ValueError("Duplicate camera or camera-week keys")
    meta = summary.loc[summary.camera_type.eq("Speed")].copy()
    meta["activation"] = pd.to_datetime(meta.inferred_activation_date)
    cols = ["camera_id", "activation", "activation_confidence"]
    out = weekly.merge(meta[cols], on="camera_id", validate="many_to_one")
    out["week_start"] = pd.to_datetime(out.week_start)
    hours = calendar_hours(out.week_start, schedule)
    out["permitted_hours"] = out.week_start.map(hours)
    valid = out.is_complete_week.fillna(False) & out.source_complete.fillna(False)
    out["tickets_per_hour"] = (out.observed_ticket_count / out.permitted_hours.where(out.permitted_hours.gt(0))).where(valid)
    age = (out.week_start - out.activation).dt.days / 7
    reference = (out.camera_id.isin(reference_ids) & age.ge(config["established_weeks"])
                 & out.activation_confidence.isin(config["activation_confidences"]))
    refs = out.tickets_per_hour.where(reference)
    sums = refs.groupby(out.week_start).transform("sum")
    counts = refs.groupby(out.week_start).transform("count")
    out["reference_count"] = counts - refs.notna().astype(int)
    out["reference_mean"] = (sums - refs.fillna(0)) / out.reference_count.where(out.reference_count.gt(0))
    enough = out.reference_count.ge(config["minimum_references"]) & out.reference_mean.gt(0)
    out["corrected_rate"] = (out.tickets_per_hour / out.reference_mean).where(enough)
    log.info("Calendar correction: %d/%d camera-weeks usable; %d reference cameras",
             out.corrected_rate.notna().sum(), len(out), len(reference_ids))
    return out


def first_year_outcomes(summary, corrected, config=TICKET_REVISION_CONFIG):
    """Percent decline: mean of weeks 40-52 vs 1-13, not a noisy single week.

    Full source coverage is required through week 52. Non-enforced weeks remain
    missing; each endpoint window needs the configured number of usable weeks.
    """
    meta = summary.loc[summary.camera_type.eq("Speed")].copy().set_index("camera_id")
    date = pd.to_datetime(meta.inferred_activation_date)
    meta["anchor"] = date + pd.to_timedelta((-date.dt.dayofweek) % 7, unit="D")
    meta["activation_year"] = date.dt.year
    rows = corrected.assign(relative_week=((corrected.week_start - corrected.camera_id.map(meta.anchor)).dt.days // 7) + 1)
    h, w = config["horizon_weeks"], config["window_weeks"]
    rows = rows.loc[rows.relative_week.between(1, h)]
    columns = range(1, h + 1)
    panel = rows.pivot(index="camera_id", columns="relative_week", values="corrected_rate").reindex(index=meta.index, columns=columns)
    covered = rows.assign(covered=rows.is_complete_week & rows.source_complete).pivot(index="camera_id", columns="relative_week", values="covered").reindex(index=meta.index, columns=columns)
    early, late = panel.iloc[:, :w], panel.iloc[:, -w:]
    meta["early_weeks"], meta["late_weeks"] = early.count(axis=1), late.count(axis=1)
    meta["early_mean"], meta["late_mean"] = early.mean(axis=1), late.mean(axis=1)
    meta["outcome_status"] = np.select(
        [meta.left_censored, ~meta.activation_confidence.isin(config["activation_confidences"]),
         ~covered.fillna(False).all(axis=1),
         meta.early_weeks.lt(config["minimum_window_weeks"]) | meta.late_weeks.lt(config["minimum_window_weeks"]),
         ~meta.early_mean.gt(0)],
        ["left_censored", "low_activation_confidence", "incomplete_first_year", "insufficient_corrected_weeks", "zero_baseline"], default="eligible")
    meta["decline_52w_pct"] = (100 * (1 - meta.late_mean / meta.early_mean)).where(meta.outcome_status.eq("eligible"))
    log.info("First-year outcomes: %s", meta.outcome_status.value_counts().to_dict())
    return meta, panel
