"""When each camera started enforcing, inferred from its tickets (#23).

There is no public install date, so a camera's activation is the first week of
sustained ticketing: `min_consecutive_weeks` weeks in a row with at least
`min_weekly_tickets` tickets. This replaces eric/camera_activation_inference.ipynb
and adds DOT's speed-camera schedule (`speed_camera_schedule.csv`):

- Speed cameras are only judged on weeks they were allowed to ticket. Before July
  2019 they ran September-June, so a quiet summer isn't a shutdown and the return
  in September isn't a new camera. School holidays, when citywide speed tickets
  collapse, are skipped the same way.
- Cameras already running when the ticket data starts (June 2016) are flagged
  `left_censored` instead of being treated as 2016 activations.
- Speed-camera activations just after a schedule change (July 2019, August 2022)
  get "medium" confidence: a jump in tickets there may be more hours, not a new camera.
- A ticket listed in two fiscal-year tables is counted once, and locations with no
  sustained run and almost no tickets (text typos, not cameras) are dropped.

`inferred_activation_date` is the schedule-aware date. Two alternatives are kept
for stress tests (as in Nguyen 2026, Tables 7-8): `activation_date_unadjusted`, the
notebook's rule with no schedule, and `activation_date_hours_scaled`, with the
weekly threshold scaled by permitted hours. `first_ticket_date` is the third.

Red light cameras run around the clock, so the schedule doesn't apply to them.
"""

import logging
import re

import numpy as np
import pandas as pd

from data.reader import read_camera_data, read_table
from preprocessing.camera_directions import CAMERA_TYPES, camera_search_query
from preprocessing.config import CameraActivationConfig

log = logging.getLogger(__name__)

TICKET_COLUMNS = [
    "summons_number",
    "violation_code",
    "street_name",
    "intersecting_street",
    "issue_date",
]
SPEED = 36
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]  # fmt: skip
SUSTAINED_METHOD = "first_sustained_run_on_enforced_weeks"
FALLBACK_METHOD = "first_ticket_week"


def weekly_ticket_counts(
    tickets: pd.DataFrame, seen: np.ndarray, config: CameraActivationConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """Tickets per (violation_code, search_query, week), skipping summons in `seen`.

    Weeks start on Monday. Rows with no street text or an issue date before
    `start_date` are dropped. Returns the counts and `seen` plus this table's summons.
    """
    summons = pd.to_numeric(tickets["summons_number"], errors="coerce")
    issued = pd.to_datetime(tickets["issue_date"], errors="coerce")
    duplicate = summons.isin(seen)
    keep = (
        tickets["street_name"].notna()
        & tickets["intersecting_street"].notna()
        & (issued >= pd.Timestamp(config.start_date))
        & ~duplicate
    )
    log.info(
        "weekly_ticket_counts: dropped %d of %d tickets (%d already counted, the "
        "rest no street text or before %s)",
        (~keep).sum(),
        len(tickets),
        duplicate.sum(),
        config.start_date,
    )
    kept = tickets[keep]
    counts = (
        pd.DataFrame(
            {
                "violation_code": pd.to_numeric(kept["violation_code"]).astype(int),
                "search_query": camera_search_query(
                    kept["street_name"], kept["intersecting_street"]
                ),
                "week": issued[keep].dt.to_period("W-SUN").dt.start_time,
            }
        )
        .groupby(["violation_code", "search_query", "week"])
        .size()
        .rename("tickets")
        .reset_index()
    )
    seen = np.union1d(seen, summons.dropna().to_numpy(dtype="int64"))
    return counts, seen


def permitted_months(text: str) -> set[int]:
    """Month numbers from the schedule's `months_of_opp`: "All" or "September-June"."""
    if text.strip().lower() == "all":
        return set(range(1, 13))
    first, last = (MONTHS.index(m.strip()) + 1 for m in text.split("-"))
    span = (last - first) % 12 + 1
    return {(first - 1 + i) % 12 + 1 for i in range(span)}


def _clock_hour(text: str) -> int:
    """24-hour clock hour from "7am" or "10pm"."""
    hour, half = re.fullmatch(r"(\d+)\s*(am|pm)", text.strip().lower()).groups()
    return int(hour) % 12 + (12 if half == "pm" else 0)


def permitted_hours_per_week(hours: str, days: str) -> float:
    """Hours a week a camera may ticket, from `hours_of_opp` ("6am-10pm" or "All")
    and `days_of_opp` ("Mon-Fri" or "All")."""
    if hours.strip().lower() == "all":
        per_day = 24
    else:
        start, end = (_clock_hour(h) for h in hours.split("-"))
        per_day = end - start
    per_week = 7 if days.strip().lower() == "all" else 5
    return float(per_day * per_week)


def speed_enforced_weeks(
    weekly: pd.DataFrame, schedule: pd.DataFrame, config: CameraActivationConfig
) -> pd.DataFrame:
    """Every week from the first speed ticket on, with `period_start` (the schedule
    period it falls in), its `permitted_hours` and `enforced`: the schedule permits
    its month and citywide speed tickets reach `min_citywide_share` of the period's
    median week."""
    totals = weekly[weekly["violation_code"] == SPEED].groupby("week")["tickets"].sum()
    weeks = pd.DataFrame(
        {"week": pd.date_range(totals.index.min(), totals.index.max(), freq="7D")}
    )
    weeks["citywide"] = weeks["week"].map(totals).fillna(0)
    starts = pd.to_datetime(schedule["start_date"]).sort_values()
    period = np.searchsorted(starts.to_numpy(), weeks["week"].to_numpy(), "right") - 1
    rows = schedule.loc[starts.index].reset_index(drop=True).iloc[period]
    weeks["period_start"] = starts.to_numpy()[period]
    weeks["permitted_hours"] = [
        permitted_hours_per_week(h, d)
        for h, d in zip(rows["hours_of_opp"], rows["days_of_opp"])
    ]
    months = rows["months_of_opp"].map(permitted_months).to_numpy()
    in_season = pd.Series(
        [w.month in m for w, m in zip(weeks["week"], months)], index=weeks.index
    )
    median = weeks[in_season].groupby("period_start")["citywide"].transform("median")
    weeks["enforced"] = in_season & (
        weeks["citywide"] >= config.min_citywide_share * median.reindex(weeks.index)
    )
    log.info(
        "speed_enforced_weeks: %d of %d weeks enforced (%d out of season, %d holidays)",
        weeks["enforced"].sum(),
        len(weeks),
        (~in_season).sum(),
        (in_season & ~weeks["enforced"]).sum(),
    )
    return weeks


def first_sustained_week(
    tickets: pd.Series, threshold, config: CameraActivationConfig
) -> pd.Timestamp | None:
    """Start of the first run of `min_consecutive_weeks` active weeks, or None.

    `tickets` is indexed by week and holds every week the camera is judged on, with
    zeros for weeks without tickets. A week is active at `threshold` tickets or
    more (a number, or a Series aligned with `tickets`).
    """
    n = config.min_consecutive_weeks
    active = (tickets >= threshold).astype(int)
    run = active.rolling(n).sum() == n
    if not run.any():
        return None
    return tickets.index[run.to_numpy().argmax() - n + 1]


def infer_activations(
    weekly: pd.DataFrame, enforced: pd.DataFrame, config: CameraActivationConfig
) -> pd.DataFrame:
    """One row per (violation_code, search_query) with its activation dates.

    - `inferred_activation_date`: speed cameras judged on enforced weeks only, red
      light cameras on every week from the first red light ticket.
    - `activation_date_unadjusted`: every week, no schedule (the notebook's rule).
    - `activation_date_hours_scaled`: as the first, with the speed threshold scaled
      by permitted hours per week relative to `reference_hours`.

    A camera with no sustained run falls back to its first ticket week.
    """
    speed = enforced[enforced["enforced"]].set_index("week")
    scaled = config.min_weekly_tickets * speed["permitted_hours"] / config.reference_hours
    first_week = weekly.groupby("violation_code")["week"].min()
    rows = []
    for (code, query), g in weekly.groupby(["violation_code", "search_query"]):
        counts = g.set_index("week")["tickets"]
        first, last = counts.index.min(), counts.index.max()
        every_week = pd.date_range(first_week[code], last, freq="7D")
        judged = speed.index[speed.index <= last] if code == SPEED else every_week
        on = counts.reindex(judged, fill_value=0)
        dates = {
            "inferred_activation_date": first_sustained_week(
                on, config.min_weekly_tickets, config
            ),
            "activation_date_unadjusted": first_sustained_week(
                counts.reindex(every_week, fill_value=0),
                config.min_weekly_tickets,
                config,
            ),
            "activation_date_hours_scaled": first_sustained_week(
                on,
                scaled.reindex(judged) if code == SPEED else config.min_weekly_tickets,
                config,
            ),
        }
        rows.append(
            {
                "violation_code": code,
                "search_query": query,
                **{k: first if v is None else v for k, v in dates.items()},
                "first_ticket_date": first,
                "last_ticket_date": last,
                "total_tickets_issued": counts.sum(),
                "activation_method": FALLBACK_METHOD
                if dates["inferred_activation_date"] is None
                else SUSTAINED_METHOD,
            }
        )
    out = pd.DataFrame(rows)
    out.insert(1, "camera_type", out["violation_code"].map(CAMERA_TYPES))
    return out


def drop_ticket_fragments(
    activations: pd.DataFrame, config: CameraActivationConfig
) -> pd.DataFrame:
    """Drop locations with no sustained run and fewer than `min_fallback_tickets`
    tickets in all: misspelled or mis-split location text, not cameras."""
    fragment = (activations["activation_method"] == FALLBACK_METHOD) & (
        activations["total_tickets_issued"] < config.min_fallback_tickets
    )
    log.info(
        "drop_ticket_fragments: dropped %d of %d locations (%d tickets)",
        fragment.sum(),
        len(activations),
        activations.loc[fragment, "total_tickets_issued"].sum(),
    )
    return activations[~fragment].reset_index(drop=True)


def add_activation_confidence(
    activations: pd.DataFrame, schedule: pd.DataFrame, config: CameraActivationConfig
) -> pd.DataFrame:
    """Add `left_censored` and `activation_confidence`.

    "low": no sustained run (first ticket week used). "medium": a speed camera
    activated within `schedule_change_weeks` after a schedule change. "high": the
    rest. Left-censored cameras keep their date but are flagged, so before/after
    comparisons can leave them out.
    """
    date = activations["inferred_activation_date"]
    changes = pd.to_datetime(schedule["start_date"])
    changes = changes[changes > date.min()]
    window = pd.Timedelta(weeks=config.schedule_change_weeks)
    near_change = np.zeros(len(activations), dtype=bool)
    for change in changes:
        near_change |= ((date >= change) & (date < change + window)).to_numpy()
    near_change &= (activations["violation_code"] == SPEED).to_numpy()
    out = activations.assign(
        left_censored=date < pd.Timestamp(config.data_start_cutoff),
        activation_confidence=np.select(
            [activations["activation_method"] == FALLBACK_METHOD, near_change],
            ["low", "medium"],
            "high",
        ),
    )
    log.info(
        "add_activation_confidence: %d cameras left-censored; confidence %s",
        out["left_censored"].sum(),
        out["activation_confidence"].value_counts().to_dict(),
    )
    return out


def add_geocodes(activations: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    """Add the original `Latitude`/`Longitude` by `search_query`; drop locations that
    were never geocoded (as the activation notebook did)."""
    points = coordinates[["search_query", "Latitude", "Longitude"]].drop_duplicates(
        "search_query"
    )
    out = activations.merge(points, on="search_query", how="inner")
    log.info(
        "add_geocodes: %d of %d camera locations have a geocode",
        len(out),
        len(activations),
    )
    return out


OUTPUT_COLUMNS = [
    "search_query",
    "camera_type",
    "violation_code",
    "Latitude",
    "Longitude",
    "inferred_activation_date",
    "activation_date_unadjusted",
    "activation_date_hours_scaled",
    "first_ticket_date",
    "last_ticket_date",
    "total_tickets_issued",
    "activation_method",
    "activation_confidence",
    "left_censored",
    "activation_year",
]


def load_camera_activations(
    config: CameraActivationConfig | None = None,
) -> pd.DataFrame:
    """Read every ticket table and return one row per camera location with its
    activation dates, method and confidence (the `camera_activation_dates` columns).

    Ticket tables are read one at a time, so only one fiscal year is in memory.
    """
    config = config or CameraActivationConfig()
    schedule = read_camera_data("speed_camera_schedule")
    seen = np.array([], dtype="int64")
    parts = []
    for table in config.ticket_tables:
        counts, seen = weekly_ticket_counts(
            read_table(table, TICKET_COLUMNS), seen, config
        )
        parts.append(counts)
    weekly = (
        pd.concat(parts)
        .groupby(["violation_code", "search_query", "week"], as_index=False)["tickets"]
        .sum()
    )
    enforced = speed_enforced_weeks(weekly, schedule, config)
    activations = drop_ticket_fragments(
        infer_activations(weekly, enforced, config), config
    )
    activations = add_activation_confidence(activations, schedule, config)
    out = add_geocodes(activations, read_camera_data("yearly_camera_coordinates"))
    out["activation_year"] = out["inferred_activation_date"].dt.year
    return out[OUTPUT_COLUMNS].sort_values(["search_query", "violation_code"])


if __name__ == "__main__":
    # Rebuild the shared file:
    # uv run python -m preprocessing.cameras
    from data.reader import CAMERA_DATA_DIR

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = load_camera_activations()
    path = CAMERA_DATA_DIR / "camera_activation_dates.csv"
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(out):,} camera locations to {path}")
