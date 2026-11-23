"""Tests for placing cameras on LION from their location text in preprocessing/."""

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from preprocessing.camera_locations import (
    locate_cameras,
    node_name_index,
    split_location,
)
from preprocessing.config import CameraDirectionConfig, SiteConfig

CONFIG = CameraDirectionConfig()
SITE = SiteConfig()

# Node street names along MAIN AVE: 1 ST (node 1), 2 ST (node 2), E 137 ST (node 3)
# and E 13 ST (node 5); node 4 is MAIN AVE & 1 ST again, far away (the same names
# in another borough).
NODE_NAMES = pd.DataFrame(
    {
        "NODEID": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"],
        "STNAME": [
            "MAIN AVENUE",
            "1 STREET",
            "MAIN AVENUE",
            "2 STREET",
            "MAIN AVENUE",
            "EAST 137 STREET",
            "MAIN AVENUE",
            "1 STREET",
            "MAIN AVENUE",
            "EAST 13 STREET",
        ],
    }
)
XY = {1: (0, 0), 2: (0, 300), 3: (0, 600), 4: (50000, 0), 5: (0, 900)}
NODES = gpd.GeoDataFrame(
    {"NODEID": [str(n) for n in XY]},
    geometry=[Point(p) for p in XY.values()],
    crs=SITE.crs,
)


def _index():
    index = node_name_index(NODE_NAMES)
    return (
        index.groupby("name")["node_id"].agg(set).to_dict(),
        index.groupby("node_id")["name"].agg(set).to_dict(),
    )


def test_split_location_finds_camera_and_cross_streets():
    by_name, by_node = _index()
    assert split_location("MAINAVAND1ST", by_name, by_node, 4) == (
        "MAINAV",
        [],
    )  # "1ST" shorter than 4
    assert split_location("MAINAVANDE137ST", by_name, by_node, 3) == (
        "MAINAV",
        ["E137ST"],
    )
    # a midblock "ON. CROSS1 - CROSS2" camera names two cross streets
    assert split_location("MAINAVE13STE137ST", by_name, by_node, 3) == (
        "MAINAV",
        ["E13ST", "E137ST"],
    )
    assert split_location("ELMSTAND2ST", by_name, by_node, 3) == (None, [])


def _cameras(queries, lon_lat):
    to_deg = Transformer.from_crs(SITE.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_deg.transform(*zip(*lon_lat))
    return pd.DataFrame({"search_query": queries, "Latitude": lat, "Longitude": lon})


def test_locate_cameras_moves_misplaced_geocodes_to_the_named_intersection():
    cameras = _cameras(
        [
            "MAIN AVE AND E 137 TH ST, NEW YORK, NY",  # geocoded 2 miles away
            "MAIN AVE. E 13 ST - E 137 ST, NEW YORK, NY",  # midblock
            "ELM ST AND OAK AVE, NEW YORK, NY",  # not in LION
        ],
        [(10000, 10000), (0, 700), (123, 456)],
    )
    out = locate_cameras(cameras, NODE_NAMES, NODES, CONFIG, SITE)
    first, midblock, unknown = out.itertuples()
    assert (round(first.x), round(first.y)) == (0, 600)
    assert first.geocode_method == "lion_intersection"
    assert first.geocode_confidence == "high"
    assert first.geocode_shift_ft > 10000
    assert (round(midblock.x), round(midblock.y)) == (0, 750)
    assert midblock.geocode_method == "lion_midblock"
    assert (round(unknown.x), round(unknown.y)) == (123, 456)
    assert unknown.geocode_confidence == "unverified"


def test_locate_cameras_picks_the_nearest_of_repeated_street_pairs():
    names = NODE_NAMES.assign(
        STNAME=NODE_NAMES["STNAME"].replace("1 STREET", "OAK STREET")
    )
    cameras = _cameras(["MAIN AVE AND OAK ST, NEW YORK, NY"], [(49000, 100)])
    out = locate_cameras(cameras, names, NODES, CONFIG, SITE).iloc[0]
    # nodes 1 and 4 both join MAIN AVE and OAK ST, 50,000 ft apart
    assert (round(out.x), round(out.y)) == (50000, 0)
    assert out.geocode_confidence == "medium"
