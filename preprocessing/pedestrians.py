"""Describe each site by modeled foot traffic, from MIT City Form Lab's NYC Walks.

NYC Walks (Sevtsuk et al., Nature Cities) estimates peak-hour pedestrian volumes on
every sidewalk, crosswalk and footpath segment for a typical 2018-19 weekday. It is
one pre-COVID snapshot, so it is a fixed site feature, not change over time. Its
published estimates are used as-is; its calibrated models are not rerun.

A segment's volume is the sum of the weekday peak hours in
`PedestrianConfig.volume_columns`. A site takes the largest volume among segments
within `PedestrianConfig.radius_ft` of its centroid, which works the same for
intersections (crosswalks) and mid-block sites (sidewalks) and doesn't depend on
how finely the network is cut.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from data.reader import read_nycwalks
from preprocessing.base import sites_within
from preprocessing.config import PedestrianConfig, SiteConfig

log = logging.getLogger(__name__)

# Model-ready columns from `add_pedestrian_features`.
PEDESTRIAN_FEATURES = ["log_ped_volume", "ped_missing"]


def segment_volume(
    segments: gpd.GeoDataFrame, config: PedestrianConfig
) -> gpd.GeoDataFrame:
    """Each segment's geometry and `ped_volume`; segments missing any hour are dropped."""
    cols = list(config.volume_columns)
    volume = segments[cols].sum(axis=1, min_count=len(cols))
    out = gpd.GeoDataFrame(
        {"ped_volume": volume}, geometry=segments.geometry, crs=segments.crs
    )
    log.info(
        "segment_volume: dropped %d of %d segments missing a volume",
        volume.isna().sum(),
        len(out),
    )
    return out[volume.notna()].reset_index(drop=True)


def site_ped_volume(
    sites: pd.DataFrame, segments: gpd.GeoDataFrame, config: PedestrianConfig
) -> pd.DataFrame:
    """One row per site: the largest `ped_volume` within the radius (NA if none)."""
    site_i, seg_i = sites_within(sites, segments, config.radius_ft)
    pairs = pd.DataFrame(
        {
            "site_id": sites["site_id"].to_numpy()[site_i],
            "ped_volume": segments["ped_volume"].to_numpy()[seg_i],
        }
    )
    busiest = pairs.groupby("site_id")["ped_volume"].max()
    out = pd.DataFrame({"site_id": sites["site_id"].to_numpy()})
    out["ped_volume"] = out["site_id"].map(busiest)
    log.info(
        "site_ped_volume: %d of %d sites have no NYC Walks segment within %.0f ft",
        out["ped_volume"].isna().sum(),
        len(out),
        config.radius_ft,
    )
    return out


def load_pedestrian_features(
    sites: pd.DataFrame,
    config: PedestrianConfig | None = None,
    site_config: SiteConfig | None = None,
) -> pd.DataFrame:
    """Read NYC Walks and return `site_id`, `ped_volume` for every site in `sites`.

    `sites` needs `site_id` and projected `x`, `y` in `SiteConfig.crs`, as returned
    by `preprocessing.panel.load_panel`.
    """
    config = config or PedestrianConfig()
    site_config = site_config or SiteConfig()
    segments = read_nycwalks(list(config.volume_columns)).to_crs(site_config.crs)
    return site_ped_volume(sites, segment_volume(segments, config), config)


def add_pedestrian_features(panel: pd.DataFrame, ped: pd.DataFrame) -> pd.DataFrame:
    """Add `PEDESTRIAN_FEATURES` to `panel` by `site_id` (the snapshot has no year).

    `log_ped_volume` is log1p of the weekday peak-hour volume. Sites with no segment
    nearby get the median over sites and `ped_missing` = 1, so no site-year is dropped.
    """
    site = ped.set_index("site_id")["ped_volume"]
    missing = site.isna()
    features = pd.DataFrame(
        {
            "log_ped_volume": np.log1p(site.fillna(site.median())),
            "ped_missing": missing.astype(float),
        }
    )
    log.info(
        "add_pedestrian_features: %d of %d sites had the median filled in",
        missing.sum(),
        len(site),
    )
    return panel.merge(features, left_on="site_id", right_index=True, how="left")
