"""Group crash locations into compact sites and assign crashes to them.

Site centers are chosen from the unique coordinates of crashes before
`SiteConfig.build_end` by peak-radius clustering: the busiest unclaimed point
becomes a center and claims every point within `radius_ft`, repeated until all
points are claimed. Unlike DBSCAN this cannot chain locations along a corridor,
so no site extends more than `radius_ft` from its center.

Every crash (any year) is then assigned to its nearest center within
`radius_ft`; crashes farther than that stay unassigned (NA).
"""

import logging

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

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
    """Pick site centers among unique crash coordinates before `build_end`.

    Returns one row per unique coordinate: `x`, `y`, `n_crashes`, `is_center` and
    `site_id` (the nearest center, which is always within `radius_ft`).
    """
    train = crashes[crashes["crash_date"] < pd.Timestamp(config.build_end)]
    points = (
        train.groupby(["x", "y"])
        .size()
        .rename("n_crashes")
        .reset_index()
        .sort_values(["n_crashes", "x", "y"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    xy = points[["x", "y"]].to_numpy()
    tree = cKDTree(xy)
    claimed = np.zeros(len(points), dtype=bool)
    is_center = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):  # busiest first
        if not claimed[i]:
            is_center[i] = True
            claimed[tree.query_ball_point(xy[i], config.radius_ft)] = True
    points["is_center"] = is_center

    centers = np.flatnonzero(is_center)
    _, nearest = cKDTree(xy[centers]).query(xy)
    points["site_id"] = nearest
    log.info(
        "build_site_points: %d crashes before %s at %d unique points -> %d sites",
        len(train),
        config.build_end,
        len(points),
        len(centers),
    )
    return points


def summarize_sites(site_points: pd.DataFrame) -> pd.DataFrame:
    """One row per site: crash-weighted centroid, crash count and span.

    `span_ft` is the largest distance from the centroid to a member point.
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


def add_site_borough(
    sites: pd.DataFrame, crashes: pd.DataFrame, config: SiteConfig
) -> pd.DataFrame:
    """Add `borough`: the borough most often reported by the site's crashes before
    `build_end`. NA if none of them report one (the field is often blank)."""
    train = crashes[
        (crashes["crash_date"] < pd.Timestamp(config.build_end))
        & crashes["site_id"].notna()
        & crashes["borough"].notna()
    ]
    borough = train.groupby("site_id")["borough"].agg(lambda b: b.mode().iloc[0])
    return sites.assign(borough=sites["site_id"].map(borough))


def assign_sites(
    crashes: pd.DataFrame, site_points: pd.DataFrame, config: SiteConfig
) -> pd.DataFrame:
    """Add `site_id` to each crash: its nearest site center within `radius_ft`."""
    centers = site_points[site_points["is_center"]]
    dist, idx = cKDTree(centers[["x", "y"]].to_numpy()).query(
        crashes[["x", "y"]].to_numpy(), distance_upper_bound=config.radius_ft
    )
    matched = np.isfinite(dist)
    site_id = pd.Series(pd.NA, index=crashes.index, dtype="Int64")
    site_id[matched] = centers["site_id"].to_numpy()[idx[matched]]

    after = crashes["crash_date"] >= pd.Timestamp(config.build_end)
    log.info(
        "assign_sites: %d crashes unassigned (%d of %d on/after %s)",
        (~matched).sum(),
        (~matched & after).sum(),
        after.sum(),
        config.build_end,
    )
    return crashes.assign(site_id=site_id)
