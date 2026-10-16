"""Group crash locations into sites with DBSCAN and assign crashes to them.

Sites are built from the unique coordinates of crashes before `SiteConfig.build_end`.
Every crash (any year) is then assigned to the site of the nearest training
coordinate within `eps_ft`; crashes farther than that stay unassigned (NA).
For training crashes this reproduces the DBSCAN labels exactly.
"""

import logging

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from preprocessing.config import SiteConfig

log = logging.getLogger(__name__)


def project_coordinates(crashes: pd.DataFrame, config: SiteConfig) -> pd.DataFrame:
    """Add projected `x`, `y` (feet) from `longitude`, `latitude`."""
    transformer = Transformer.from_crs("EPSG:4326", config.crs, always_xy=True)
    x, y = transformer.transform(
        crashes["longitude"].to_numpy(), crashes["latitude"].to_numpy()
    )
    return crashes.assign(x=x, y=y)


def build_site_points(crashes: pd.DataFrame, config: SiteConfig) -> pd.DataFrame:
    """Cluster unique crash coordinates before `build_end` into sites.

    Returns one row per unique coordinate: `x`, `y`, `n_crashes`, `site_id`.
    """
    train = crashes[crashes["crash_date"] < pd.Timestamp(config.build_end)]
    points = train.groupby(["x", "y"]).size().rename("n_crashes").reset_index()
    labels = DBSCAN(eps=config.eps_ft, min_samples=1).fit_predict(
        points[["x", "y"]].to_numpy()
    )
    points["site_id"] = labels
    log.info(
        "build_site_points: %d crashes before %s at %d unique points -> %d sites",
        len(train),
        config.build_end,
        len(points),
        points["site_id"].nunique(),
    )
    return points


def summarize_sites(site_points: pd.DataFrame) -> pd.DataFrame:
    """One row per site: crash-weighted centroid, crash count and span.

    `span_ft` is the largest distance from the centroid to a member point; a large
    span means DBSCAN chained many locations (e.g. along a corridor) into one site.
    """
    w = site_points["n_crashes"]
    sums = (
        site_points.assign(wx=site_points["x"] * w, wy=site_points["y"] * w)
        .groupby("site_id")[["wx", "wy", "n_crashes"]]
        .sum()
    )
    sites = pd.DataFrame(
        {
            "x": sums["wx"] / sums["n_crashes"],
            "y": sums["wy"] / sums["n_crashes"],
            "n_crashes": sums["n_crashes"],
            "n_points": site_points.groupby("site_id").size(),
        }
    )
    centroid = sites.loc[site_points["site_id"], ["x", "y"]].to_numpy()
    dist = np.hypot(*(site_points[["x", "y"]].to_numpy() - centroid).T)
    sites["span_ft"] = pd.Series(dist).groupby(site_points["site_id"].to_numpy()).max()
    return sites.reset_index()


def add_site_lonlat(sites: pd.DataFrame, config: SiteConfig) -> pd.DataFrame:
    """Add `longitude`, `latitude` of each site centroid, for mapping."""
    transformer = Transformer.from_crs(config.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(sites["x"].to_numpy(), sites["y"].to_numpy())
    return sites.assign(longitude=lon, latitude=lat)


def assign_sites(
    crashes: pd.DataFrame, site_points: pd.DataFrame, config: SiteConfig
) -> pd.DataFrame:
    """Add `site_id` to each crash: the site of the nearest training point within `eps_ft`."""
    tree = cKDTree(site_points[["x", "y"]].to_numpy())
    dist, idx = tree.query(
        crashes[["x", "y"]].to_numpy(), distance_upper_bound=config.eps_ft
    )
    matched = np.isfinite(dist)
    site_id = pd.Series(pd.NA, index=crashes.index, dtype="Int64")
    site_id[matched] = site_points["site_id"].to_numpy()[idx[matched]]

    after = crashes["crash_date"] >= pd.Timestamp(config.build_end)
    log.info(
        "assign_sites: %d crashes unassigned (%d of %d on/after %s)",
        (~matched).sum(),
        (~matched & after).sum(),
        after.sum(),
        config.build_end,
    )
    return crashes.assign(site_id=site_id)
