"""Which approach each camera enforces, parsed from the ticket location text.

Camera tickets store the location as fixed-width text split across
`street_name` and `intersecting_street`, sometimes mid-word, e.g.
"ASTORIA BLVD S (E/B)" + "@ 77TH ST" or "WB N CONDUIT AVE @ S" + "PRINGFIELD BLVD".
The direction code ("E/B", "WB", "NEB") is the travel direction of the ticketed
traffic, i.e. the approach the camera enforces. It is a street-grid direction
(Manhattan "north" runs about 29 degrees east of true north), not a compass
bearing, and not necessarily the way the lens points.

Locations are keyed by the same `search_query` the activation notebook
(eric/camera_activation_inference.ipynb) geocoded, so they join to
`data/camera_data/camera_activation_dates.csv` on (search_query, violation_code)
for coordinates. The grid direction is then turned into a compass bearing: the
direction of travel along the camera's own street in LION, at the camera.
"""

import logging
from difflib import SequenceMatcher

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from data.reader import read_camera_data, read_lion, read_table
from preprocessing.base import DIRECTION_PAREN, DIRECTION_PREFIX, normalize_street
from preprocessing.camera_locations import locate_cameras
from preprocessing.config import CameraDirectionConfig, RoadwayConfig, SiteConfig
from preprocessing.roadway import SEGMENT_COLUMNS, clean_segments

log = logging.getLogger(__name__)

TICKET_COLUMNS = ["violation_code", "street_name", "intersecting_street", "issue_date"]
CAMERA_TYPES = {7: "Red Light", 36: "Speed"}
DIRECTION_METHOD = "ticket_location_text"
BEARING_METHOD = "lion_street_tangent"
# Compass bearing (degrees clockwise from north) of each grid direction code.
GRID_BEARINGS = {
    "N": 0,
    "NE": 45,
    "E": 90,
    "SE": 135,
    "S": 180,
    "SW": 225,
    "W": 270,
    "NW": 315,
}


def summarize_locations(
    tickets: pd.DataFrame, config: CameraDirectionConfig
) -> pd.DataFrame:
    """Ticket count and first/last issue date per raw location text.

    Rows with no street text or an unparseable or too-early issue date are dropped.
    """
    issued = pd.to_datetime(tickets["issue_date"], errors="coerce")
    keep = (
        tickets["street_name"].notna()
        & tickets["intersecting_street"].notna()
        & (issued >= pd.Timestamp(config.start_date))
    )
    log.info(
        "summarize_locations: dropped %d of %d tickets (no street text or date before %s)",
        (~keep).sum(),
        len(tickets),
        config.start_date,
    )
    return (
        tickets[keep]
        .assign(
            violation_code=pd.to_numeric(tickets["violation_code"]).astype(int),
            issue_date=issued,
        )
        .groupby(["violation_code", "street_name", "intersecting_street"])["issue_date"]
        .agg(tickets="size", first_ticket_date="min", last_ticket_date="max")
        .reset_index()
    )


def combine_locations(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-table summaries from `summarize_locations` (a location can span years)."""
    return (
        pd.concat(parts)
        .groupby(["violation_code", "street_name", "intersecting_street"])
        .agg(
            tickets=("tickets", "sum"),
            first_ticket_date=("first_ticket_date", "min"),
            last_ticket_date=("last_ticket_date", "max"),
        )
        .reset_index()
    )


def parse_direction(
    street_name: pd.Series, intersecting_street: pd.Series
) -> pd.Series:
    """Direction code (N, S, E, W, NE, NW, SE, SW) from the joined location text, or NA."""
    text = (street_name.fillna("") + intersecting_street.fillna("")).str.upper()
    prefix = text.str.extract(DIRECTION_PREFIX, expand=False)
    return prefix.fillna(text.str.extract(DIRECTION_PAREN, expand=False))


def camera_search_query(
    street_name: pd.Series, intersecting_street: pd.Series
) -> pd.Series:
    """The geocoding key built in eric/camera_activation_inference.ipynb, reproduced exactly.

    Directions and anything in parentheses are removed and "@" becomes "AND".
    """
    raw = street_name.fillna("") + " " + intersecting_street.fillna("")
    return (
        raw.str.upper()
        .str.replace(r"\b(NB|SB|EB|WB|N/B|S/B|E/B|W/B)\b", "", regex=True)
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace("@", " AND ")
        .str.replace(r"^\s+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        + ", NEW YORK, NY"
    )


def enforced_approaches(
    locations: pd.DataFrame, config: CameraDirectionConfig
) -> pd.DataFrame:
    """One row per (violation_code, search_query, direction) with its tickets and dates.

    `direction_share` is the direction's share of the location's tickets. Every
    row carries `direction_method` and `direction_confidence`: "high" for a parsed
    direction with at least `min_direction_share`, "low" below it, "none" when no
    direction was found.
    """
    located = locations.assign(
        direction=parse_direction(
            locations["street_name"], locations["intersecting_street"]
        ),
        search_query=camera_search_query(
            locations["street_name"], locations["intersecting_street"]
        ),
    )
    keys = ["violation_code", "search_query"]
    out = (
        located.groupby(keys + ["direction"], dropna=False)
        .agg(
            tickets=("tickets", "sum"),
            first_ticket_date=("first_ticket_date", "min"),
            last_ticket_date=("last_ticket_date", "max"),
        )
        .reset_index()
    )
    out["direction_share"] = out["tickets"] / out.groupby(keys)["tickets"].transform(
        "sum"
    )
    out["direction_confidence"] = np.select(
        [out["direction"].isna(), out["direction_share"] < config.min_direction_share],
        ["none", "low"],
        "high",
    )
    out.insert(1, "camera_type", out["violation_code"].map(CAMERA_TYPES))
    out["direction_method"] = DIRECTION_METHOD
    log.info(
        "enforced_approaches: %d approaches at %d locations; %d without a direction, "
        "%d below %.0f%% of their location's tickets",
        len(out),
        out[keys].drop_duplicates().shape[0],
        (out["direction_confidence"] == "none").sum(),
        (out["direction_confidence"] == "low").sum(),
        config.min_direction_share * 100,
    )
    return out.sort_values(keys + ["direction"]).reset_index(drop=True)


LOCATION_COLUMNS = [
    "latitude",
    "longitude",
    "x",
    "y",
    "geocode_method",
    "geocode_confidence",
    "geocode_shift_ft",
]


def add_camera_points(approaches: pd.DataFrame, located: pd.DataFrame) -> pd.DataFrame:
    """Add each location's point from `camera_locations.locate_cameras`.

    Approaches whose location was never geocoded keep NA coordinates.
    """
    keys = ["search_query", "violation_code"]
    out = approaches.merge(located[keys + LOCATION_COLUMNS], on=keys, how="left")
    log.info(
        "add_camera_points: %d of %d approaches have no camera point",
        out["x"].isna().sum(),
        len(out),
    )
    return out


def _tangent_bearing(line, point, half_length: float) -> float:
    """Bearing (degrees from north) of `line` from start to end, near `point`."""
    line = shapely.line_merge(line)
    if line.geom_type == "MultiLineString":
        line = min(line.geoms, key=point.distance)
    d = line.project(point)
    a = line.interpolate(max(d - half_length, 0))
    b = line.interpolate(min(d + half_length, line.length))
    return float(np.degrees(np.arctan2(b.x - a.x, b.y - a.y)) % 360)


def _angle_between(a, b):
    """Smallest absolute difference between two bearings, in degrees."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def match_camera_streets(
    approaches: pd.DataFrame, segments: gpd.GeoDataFrame, config: CameraDirectionConfig
) -> pd.DataFrame:
    """Candidate (approach, LION segment) pairs on the camera's own street.

    A segment within `street_radius_ft` is on the camera's street if its normalized
    name starts the normalized `search_query` ("exact") or is at least
    `min_street_name_similarity` similar to the query's start ("fuzzy"). A leading
    direction code the key kept ("NWB REMSEN AVE ...") is ignored.
    """
    has_point = approaches["x"].notna() & approaches["direction"].notna()
    located = approaches[has_point]
    points = gpd.GeoSeries(
        gpd.points_from_xy(located["x"], located["y"]), crs=segments.crs
    )
    app_i, seg_i = segments.sindex.query(
        points, predicate="dwithin", distance=config.street_radius_ft
    )
    pairs = pd.DataFrame(
        {
            "approach": located.index.to_numpy()[app_i],
            "segment": seg_i,
            "query": normalize_street(
                located["search_query"].str.replace(DIRECTION_PREFIX, "", regex=True)
            ).to_numpy()[app_i],
            "name": normalize_street(segments["Street"].fillna("")).to_numpy()[seg_i],
        }
    )
    exact = np.array([q.startswith(n) for q, n in zip(pairs["query"], pairs["name"])])
    similarity = np.array(
        [
            SequenceMatcher(None, n, q[: len(n)]).ratio()
            for q, n in zip(pairs["query"], pairs["name"])
        ]
    )
    pairs["name_match"] = np.where(exact, "exact", "fuzzy")
    keep = (pairs["name"] != "") & (
        exact | (similarity >= config.min_street_name_similarity)
    )
    log.info(
        "match_camera_streets: %d of %d located approaches have their street in LION "
        "within %.0f ft",
        pairs.loc[keep, "approach"].nunique(),
        len(located),
        config.street_radius_ft,
    )
    return pairs[keep].reset_index(drop=True)


def add_bearings(
    approaches: pd.DataFrame, segments: gpd.GeoDataFrame, config: CameraDirectionConfig
) -> pd.DataFrame:
    """Add the compass bearing of the enforced traffic at each camera.

    Each candidate segment from `match_camera_streets` is measured at the camera and
    oriented the way closest to the ticket's grid direction; the candidate closest
    to that direction wins. Adds `bearing_deg`, `bearing_offset_deg` (bearing minus
    grid direction), `lion_segment_id`, `lion_street`, `bearing_method` and
    `bearing_confidence`: "high"/"medium" (see `CameraDirectionConfig`), "low" for a
    street more than `max_bearing_offset_deg` off its grid direction, "none" when the
    camera's street isn't within `street_radius_ft` (often a misplaced geocode).
    """
    pairs = match_camera_streets(approaches, segments, config)
    geometry, point = (
        segments.geometry.to_numpy(),
        shapely.points(approaches["x"].to_numpy(), approaches["y"].to_numpy()),
    )
    pairs["distance_ft"] = shapely.distance(
        geometry[pairs["segment"]], point[pairs["approach"]]
    )
    line_bearing = np.array(
        [
            _tangent_bearing(geometry[s], point[a], config.tangent_half_length_ft)
            for a, s in zip(pairs["approach"], pairs["segment"])
        ]
    )
    grid = approaches["direction"].map(GRID_BEARINGS).to_numpy()[pairs["approach"]]
    reversed_line = (line_bearing + 180) % 360
    forward = _angle_between(line_bearing, grid) <= _angle_between(reversed_line, grid)
    pairs["bearing_deg"] = np.where(forward, line_bearing, reversed_line)
    pairs["offset"] = (pairs["bearing_deg"] - grid + 180) % 360 - 180
    best = (
        pairs.assign(abs_offset=lambda d: d["offset"].abs())
        .sort_values(["approach", "abs_offset", "distance_ft"])
        .drop_duplicates("approach")
        .set_index("approach")
    )
    high = (
        (best["name_match"] == "exact")
        & (best["distance_ft"] <= config.high_confidence_distance_ft)
        & (best["abs_offset"] <= 45)
    )
    medium = best["abs_offset"] <= config.max_bearing_offset_deg
    out = approaches.copy()
    out["bearing_deg"] = best["bearing_deg"].round(1)
    out["bearing_offset_deg"] = best["offset"].round(1)
    chosen = segments.iloc[best["segment"]].set_index(best.index)
    out["lion_segment_id"] = chosen["SegmentID"]
    out["lion_street"] = chosen["Street"]
    out["bearing_method"] = BEARING_METHOD
    out["bearing_confidence"] = "none"
    out.loc[best.index, "bearing_confidence"] = np.select(
        [high, medium], ["high", "medium"], "low"
    )
    log.info(
        "add_bearings: %s",
        out["bearing_confidence"].value_counts().to_dict(),
    )
    return out


def load_camera_directions(
    config: CameraDirectionConfig | None = None,
    roadway_config: RoadwayConfig | None = None,
    site_config: SiteConfig | None = None,
) -> pd.DataFrame:
    """Read every camera ticket table, the geocoded cameras and LION, and return one
    row per enforced approach with its grid direction, LION-based location and
    compass bearing.

    Ticket tables are summarized one at a time, so only one fiscal year is in memory.
    """
    config = config or CameraDirectionConfig()
    roadway_config = roadway_config or RoadwayConfig()
    site_config = site_config or SiteConfig()
    parts = [
        summarize_locations(read_table(table, TICKET_COLUMNS), config)
        for table in config.ticket_tables
    ]
    located = locate_cameras(
        read_camera_data("camera_activation_dates"),
        read_lion("node_stname"),
        read_lion("node", ["NODEID"]).to_crs(site_config.crs),
        config,
        site_config,
    )
    approaches = add_camera_points(
        enforced_approaches(combine_locations(parts), config), located
    )
    segments = clean_segments(
        read_lion("lion", SEGMENT_COLUMNS + ["Street", "SegmentID"]), roadway_config
    ).to_crs(site_config.crs)
    roadbeds = segments[
        segments["SegmentTyp"].isin(roadway_config.roadbed_segment_types)
    ].reset_index(drop=True)
    return add_bearings(approaches, roadbeds, config)


if __name__ == "__main__":
    # Share the result with teammates who don't have the raw tickets:
    # uv run python -m preprocessing.camera_directions
    from data.reader import CAMERA_DATA_DIR

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = load_camera_directions().drop(columns=["x", "y"])
    path = CAMERA_DATA_DIR / "camera_directions.csv"
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(out):,} approaches to {path}")
