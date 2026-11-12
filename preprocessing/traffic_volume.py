"""Daily traffic volume at crash sites, from DOT's Automated Traffic Volume Counts.

DOT counts traffic in 15-minute intervals on a few thousand LION street segments,
usually for a week at a time. Each counted segment gets a typical daily volume:
the median full-day count in each direction, added across directions. A site's
volume is that of its busiest counted street within `RoadwayConfig.radius_ft`,
with a divided road's roadbeds added together as in `roadway.street_cross_section`.

Only a small share of sites has a counted street nearby, and DOT chooses where to
count, so volume is an attribute of that subset, not of all sites.
"""

import logging

import numpy as np
import pandas as pd

from data.reader import read_lion, read_table
from preprocessing.config import RoadwayConfig, SiteConfig, TrafficVolumeConfig
from preprocessing.roadway import (
    SEGMENT_COLUMNS,
    clean_segments,
    match_segments,
    street_cross_section,
)

log = logging.getLogger(__name__)

INTERVAL_KEY = ["requestid", "segmentid", "direction", "date", "hh", "mm"]


def daily_counts(counts: pd.DataFrame, config: TrafficVolumeConfig) -> pd.DataFrame:
    """Total vehicles per (segment, direction, day), for complete non-zero days only.

    A 15-minute interval recorded twice in the same study is counted once. A day
    with no vehicles at all is treated as a failed counter.
    """
    out = counts[["requestid", "segmentid", "direction", "hh", "mm", "vol"]].copy()
    for col in ["requestid", "segmentid", "hh", "mm", "vol"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = pd.to_datetime(
        {"year": counts["yr"], "month": counts["m"], "day": counts["d"]},
        errors="coerce",
    )
    out = out.dropna().drop_duplicates(INTERVAL_KEY)
    days = out.groupby(INTERVAL_KEY[:4])["vol"].agg(["size", "sum"])
    keep = days["size"].eq(config.intervals_per_day) & days["sum"].gt(0)
    log.info(
        "daily_counts: kept %d of %d count days (complete and non-zero)",
        keep.sum(),
        len(days),
    )
    return (
        days.loc[keep, "sum"].rename("volume").reset_index().astype({"segmentid": int})
    )


def segment_daily_volume(daily: pd.DataFrame) -> pd.Series:
    """Typical daily volume per LION segment id, both directions together."""
    by_direction = daily.groupby(["segmentid", "direction"])["volume"].median()
    out = by_direction.groupby("segmentid").sum().rename("daily_volume")
    log.info("segment_daily_volume: %d counted segments", len(out))
    return out


def site_traffic_volume(pairs: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
    """`daily_volume` of the busiest counted street at each site.

    `pairs` is `roadway.match_segments` output with `SegmentID`; sites with no
    counted segment nearby are left out.
    """
    pairs = pairs.assign(
        daily_volume=pd.to_numeric(pairs["SegmentID"], errors="coerce").map(volume)
    ).dropna(subset=["daily_volume"])
    by_street = street_cross_section(pairs, "daily_volume")
    out = by_street.groupby("site_id").max().rename("daily_volume").reset_index()
    log.info("site_traffic_volume: %d sites have a counted street", len(out))
    return out


def load_traffic_volume(
    sites: pd.DataFrame,
    config: TrafficVolumeConfig | None = None,
    roadway_config: RoadwayConfig | None = None,
    site_config: SiteConfig | None = None,
) -> pd.DataFrame:
    """Read the counts and LION and return `site_id`, `daily_volume` for counted sites.

    `sites` needs `site_id` and projected `x`, `y`, as returned by `load_panel`.
    """
    config = config or TrafficVolumeConfig()
    roadway_config = roadway_config or RoadwayConfig()
    site_config = site_config or SiteConfig()
    volume = segment_daily_volume(daily_counts(read_table(config.table), config))
    segments = clean_segments(
        read_lion("lion", SEGMENT_COLUMNS), roadway_config
    ).to_crs(site_config.crs)
    return site_traffic_volume(match_segments(sites, segments, roadway_config), volume)


def add_traffic_volume(panel: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Add `daily_volume` and `log2_volume` (a one-unit step is a doubling) by site.

    Sites without a counted street get NA, so the volume subset is
    `panel["daily_volume"].notna()`.
    """
    out = panel.merge(volume, on="site_id", how="left")
    out["log2_volume"] = np.log2(out["daily_volume"])
    return out
