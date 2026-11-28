"""Describe each site by counted vehicle traffic, from DOT's Automated Traffic Volume
Counts (`7ym2-wayt`), and fill in uncounted sites from road features (#35).

The counts are short (usually 1-2 week) studies of 15-minute volumes by direction on
LION street segments. A segment's volume is its average weekday daily traffic:

1. Rows for the same study, segment, direction and 15 minutes are summed (about
   5,000 duplicates, which look like two counters on one direction).
2. Only complete weekdays are kept (`TrafficConfig`), and each direction's days
   are averaged.
3. A study's volume is the sum of the directions it counted; a segment counted by
   several studies takes their median. Count years differ (2000 on), so this is a
   fixed site feature, not change over time.

A site takes the busiest counted segment within `TrafficConfig.radius_ft`, as for
NYC Walks. Only about 4.5% of sites have one, so `impute_log_traffic` fills the rest
from a linear fit on road features. `experiments/traffic_exposure.ipynb` tests both.
"""

import logging

import numpy as np
import pandas as pd

from data.reader import read_lion, read_table
from preprocessing.base import sites_within
from preprocessing.config import SiteConfig, TrafficConfig

log = logging.getLogger(__name__)

COUNT_COLUMNS = ["requestid", "yr", "m", "d", "hh", "mm", "vol", "segmentid"]


def daily_volumes(counts: pd.DataFrame, config: TrafficConfig) -> pd.DataFrame:
    """One row per complete weekday of a study, segment and direction, with `vol`."""
    c = counts.assign(**{col: pd.to_numeric(counts[col]) for col in COUNT_COLUMNS})
    c["date"] = pd.to_datetime({"year": c["yr"], "month": c["m"], "day": c["d"]})
    keys = ["requestid", "segmentid", "direction", "date"]
    interval = c.groupby([*keys, "hh", "mm"])["vol"].sum().reset_index()
    log.info(
        "daily_volumes: summed %d duplicate 15-minute rows", len(c) - len(interval)
    )
    day = interval.groupby(keys)["vol"].agg(["sum", "size"]).reset_index()
    keep = (day["size"] == config.intervals_per_day) & day["date"].dt.dayofweek.isin(
        config.weekdays
    )
    log.info("daily_volumes: kept %d of %d count days", keep.sum(), len(day))
    return day.loc[keep, keys + ["sum"]].rename(columns={"sum": "vol"})


def segment_volume(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per LION segment: `segmentid`, weekday `adt` and `n_studies`."""
    direction = daily.groupby(["requestid", "segmentid", "direction"])["vol"].mean()
    study = direction.groupby(["requestid", "segmentid"]).sum()
    return study.groupby("segmentid").agg(adt="median", n_studies="size").reset_index()


def site_traffic_volume(
    sites: pd.DataFrame, segments, config: TrafficConfig
) -> pd.DataFrame:
    """One row per site: the largest `adt` among segments within the radius (NA if none).

    `segments` is a GeoDataFrame of counted segments with `adt`, in the sites' CRS.
    """
    site_i, seg_i = sites_within(sites, segments, config.radius_ft)
    busiest = (
        pd.DataFrame(
            {
                "site_id": sites["site_id"].to_numpy()[site_i],
                "adt": segments["adt"].to_numpy()[seg_i],
            }
        )
        .groupby("site_id")["adt"]
        .max()
    )
    out = pd.DataFrame({"site_id": sites["site_id"].to_numpy()})
    out["adt"] = out["site_id"].map(busiest)
    log.info(
        "site_traffic_volume: %d of %d sites have a counted segment within %.0f ft",
        out["adt"].notna().sum(),
        len(out),
        config.radius_ft,
    )
    return out


def load_traffic_features(
    sites: pd.DataFrame,
    config: TrafficConfig | None = None,
    site_config: SiteConfig | None = None,
) -> pd.DataFrame:
    """Read the counts and LION; return `site_id`, `adt` for every site in `sites`.

    Counts use LION v14 segment IDs; segments missing from the current LION are
    dropped (logged).
    """
    config = config or TrafficConfig()
    site_config = site_config or SiteConfig()
    volume = segment_volume(daily_volumes(read_table("traffic_counts"), config))
    lion = read_lion("lion", ["SegmentID"]).to_crs(site_config.crs)
    lion["segmentid"] = pd.to_numeric(lion["SegmentID"])
    segments = lion.drop_duplicates("segmentid").merge(volume, on="segmentid")
    log.info(
        "load_traffic_features: %d of %d counted segments are in LION",
        len(segments),
        len(volume),
    )
    return site_traffic_volume(sites, segments, config)


def add_traffic_features(panel: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Add `log_traffic` (log of counted weekday ADT, NA if uncounted or zero) and
    `traffic_counted` to `panel` by `site_id`."""
    adt = traffic.set_index("site_id")["adt"]
    counted = adt > 0
    features = pd.DataFrame(
        {
            "log_traffic": np.log(adt.where(counted)),
            "traffic_counted": counted.astype(float),
        }
    )
    return panel.merge(features, left_on="site_id", right_index=True, how="left")


def impute_log_traffic(panel: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Add `log_traffic_imputed`, a least-squares fit of `log_traffic` on `predictors`
    over counted site-years, and `log_traffic_filled` (counted where known, else
    imputed). Only site features are used, never crash outcomes."""
    X = np.column_stack([np.ones(len(panel)), panel[predictors].astype(float)])
    known = panel["log_traffic"].notna().to_numpy()
    beta = np.linalg.lstsq(X[known], panel["log_traffic"].to_numpy()[known])[0]
    imputed = X @ beta
    log.info("impute_log_traffic: filled %d of %d site-years", (~known).sum(), len(X))
    return panel.assign(
        log_traffic_imputed=imputed,
        log_traffic_filled=panel["log_traffic"].fillna(pd.Series(imputed, panel.index)),
    )
