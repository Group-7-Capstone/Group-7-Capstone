"""Tests for the intersection units in preprocessing/intersections.py."""

import pandas as pd

from preprocessing.config import IntersectionConfig
from preprocessing.intersections import add_street_names, merge_intersection_nodes


def _nodes():
    # 1 and 2: two nodes of one divided-avenue crossing, 40 ft apart; 3 a block away
    return pd.DataFrame(
        {"node_id": [1, 2, 3], "x": [0.0, 40.0, 600.0], "y": [0.0, 0.0, 0.0]}
    )


def test_merge_intersection_nodes_joins_nearby_nodes_only():
    out = merge_intersection_nodes(_nodes(), IntersectionConfig(merge_ft=50))
    assert out["node_ids"].tolist() == [(1, 2), (3,)]
    assert out["x"].tolist() == [20.0, 600.0]


def test_add_street_names_lists_each_street_once():
    intersections = merge_intersection_nodes(_nodes(), IntersectionConfig(merge_ft=50))
    segments = pd.DataFrame(
        {
            "NodeIDFrom": [1, 2, 9, 3],
            "NodeIDTo": [8, 7, 1, 6],
            "Street": ["BROADWAY ", "BROADWAY", "W 96 ST", "W 97 ST"],
        }
    )
    out = add_street_names(intersections, segments)
    assert out["streets"].tolist() == ["BROADWAY & W 96 ST", "W 97 ST"]
