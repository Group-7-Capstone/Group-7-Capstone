"""Place each camera location on LION using the street names in its own text.

The geocoded points in `camera_activation_dates.csv` put roughly 1 in 8 speed
cameras far from the street their tickets name. Here every location
(`search_query`) is placed on LION's street network instead:

- The camera's street is the longest LION street name that starts the text.
- Cross streets are other names, at nodes on the camera's street, that appear
  later in the text.
- One cross street ("ON AND CROSS"): the node where the two streets meet.
- Two cross streets ("ON. CROSS1 - CROSS2", a midblock speed camera): the midpoint
  of the two intersections.

Where a street pair meets at nodes more than `max_node_spread_ft` apart (a name
repeated in two boroughs), the node nearest the original geocode is used, with
"medium" confidence. Locations whose streets can't be found in LION keep the
original geocode ("unverified").
"""

import logging

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import pdist

from preprocessing.base import DIRECTION_PREFIX, normalize_street
from preprocessing.config import CameraDirectionConfig, SiteConfig

log = logging.getLogger(__name__)


def node_name_index(node_names: pd.DataFrame) -> pd.DataFrame:
    """Normalized street names at each LION node (`node_id`, `name`)."""
    out = pd.DataFrame(
        {
            "node_id": pd.to_numeric(node_names["NODEID"], errors="coerce"),
            "name": normalize_street(node_names["STNAME"].fillna("")),
        }
    )
    return (
        out[(out["name"] != "") & out["node_id"].notna()]
        .astype({"node_id": int})
        .drop_duplicates()
        .reset_index(drop=True)
    )


def split_location(
    text: str,
    nodes_by_name: dict[str, set[int]],
    names_by_node: dict[int, set[str]],
    min_name_length: int,
) -> tuple[str | None, list[str]]:
    """(camera street, up to two cross streets) named in normalized location text."""
    on = next(
        (
            text[:k]
            for k in range(len(text), min_name_length - 1, -1)
            if text[:k] in nodes_by_name
        ),
        None,
    )
    if on is None:
        return None, []
    rest = text[len(on) :]
    names = {n for node in nodes_by_name[on] for n in names_by_node[node]} - {on}
    found = sorted(
        (rest.find(n), -len(n), n)
        for n in names
        if len(n) >= min_name_length and n in rest
    )
    crosses, end = [], -1
    for pos, _, name in found:  # earliest first, longest first at a tie
        if pos >= end:  # skip names inside one already taken ("E13" in "E137ST")
            crosses.append(name)
            end = pos + len(name)
    return on, crosses[:2]


def locate_cameras(
    cameras: pd.DataFrame,
    node_names: pd.DataFrame,
    nodes: pd.DataFrame,
    config: CameraDirectionConfig,
    site_config: SiteConfig,
) -> pd.DataFrame:
    """Add LION-based `latitude`, `longitude`, `x`, `y` to each geocoded camera.

    `cameras` has `search_query`, `Latitude`, `Longitude` (the original geocode);
    `nodes` has `NODEID` and point geometry in `site_config.crs`. Adds
    `geocode_method` ("lion_intersection", "lion_midblock" or "original_geocode"),
    `geocode_confidence` ("high", "medium" or "unverified") and `geocode_shift_ft`,
    the distance from the original geocode.
    """
    index = node_name_index(node_names)
    nodes_by_name = index.groupby("name")["node_id"].agg(set).to_dict()
    names_by_node = index.groupby("node_id")["name"].agg(set).to_dict()
    node_xy = dict(
        zip(
            pd.to_numeric(nodes["NODEID"], errors="coerce"),
            zip(nodes.geometry.x, nodes.geometry.y),
        )
    )
    to_feet = Transformer.from_crs("EPSG:4326", site_config.crs, always_xy=True)
    orig_x, orig_y = to_feet.transform(
        cameras["Longitude"].to_numpy(), cameras["Latitude"].to_numpy()
    )
    texts = normalize_street(
        cameras["search_query"].str.replace(DIRECTION_PREFIX, "", regex=True)
    )

    rows = []
    for text, ox, oy in zip(texts, orig_x, orig_y):
        on, crosses = split_location(
            text, nodes_by_name, names_by_node, config.min_street_name_length
        )
        points, confidence = [], "high"
        for cross in crosses:
            shared = nodes_by_name[on] & nodes_by_name[cross]
            xy = np.array([node_xy[n] for n in shared if n in node_xy])
            if len(xy) == 0:
                continue
            if len(xy) == 1 or pdist(xy).max() <= config.max_node_spread_ft:
                points.append(xy.mean(axis=0))
            else:
                points.append(xy[np.argmin(np.hypot(*(xy - (ox, oy)).T))])
                confidence = "medium"
        if points:
            x, y = np.mean(points, axis=0)
            method = "lion_intersection" if len(points) == 1 else "lion_midblock"
        else:
            x, y, method, confidence = ox, oy, "original_geocode", "unverified"
        rows.append(
            (x, y, method, confidence, round(float(np.hypot(x - ox, y - oy)), 1))
        )

    located = pd.DataFrame(
        rows,
        columns=["x", "y", "geocode_method", "geocode_confidence", "geocode_shift_ft"],
        index=cameras.index,
    )
    lon, lat = Transformer.from_crs(
        site_config.crs, "EPSG:4326", always_xy=True
    ).transform(located["x"].to_numpy(), located["y"].to_numpy())
    out = cameras.assign(latitude=lat, longitude=lon, **located)
    log.info(
        "locate_cameras: %s; %d moved more than 500 ft from the original geocode",
        out["geocode_method"].value_counts().to_dict(),
        (out["geocode_shift_ft"] > 500).sum(),
    )
    return out
