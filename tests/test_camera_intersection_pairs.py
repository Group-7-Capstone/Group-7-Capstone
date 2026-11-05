"""Regression checks for unordered street pairs and ambiguous geometry."""
import pandas as pd

from preprocessing.camera_intersection_pairs import (
    PAIR_CONFIG, classify_pairs, near_nodes, unordered_pair, width_class,
)
from preprocessing.camera_road_features import UNKNOWN, match_camera_roads
from test_camera_road_features import fixture_roads


def test_reversing_street_names_yields_one_intersection_and_identical_pair():
    summary, roads, aliases = fixture_roads()
    roads.loc[0, ["TrafDir", "StreetWidth_Min", "StreetWidth_Max"]] = ["W", 50, 55]
    summary = pd.concat([summary, summary.assign(camera_id="y", search_query="SECOND ST & FIRST AVE")])
    first, segments = match_camera_roads(summary, roads, aliases)
    result = classify_pairs(first, segments, roads, aliases)
    assert result.intersection_pair_id.nunique() == 1
    assert result.direction_pair.tolist() == ["One-way × Two-way"] * 2
    assert result.width_pair.tolist() == ["<15 m × ≥15 m"] * 2
    assert result.first_direction.tolist() == ["One-way", "Two-way"]


def test_divided_cross_street_is_not_silently_classified_as_one_way():
    summary, roads, aliases = fixture_roads()
    roads.loc[1, "RB_Layer"] = "R"
    first, segments = match_camera_roads(summary, roads, aliases)
    assert first.strict_match.all()
    row = classify_pairs(first, segments, roads, aliases).iloc[0]
    assert row.direction_pair == UNKNOWN and "Divided road" in row.pair_status


def test_a_single_alias_segment_does_not_make_an_intersection():
    summary, roads, aliases = fixture_roads()
    roads = roads.iloc[:1].copy()
    roads["SAFStreetName"] = "SECOND STREET"
    first, segments = match_camera_roads(summary, roads, aliases)
    row = classify_pairs(first, segments, roads, aliases).iloc[0]
    assert row.pair_status == "Overlapping street segment IDs" and row.lanes_pair == UNKNOWN


def test_ambiguous_width_and_remote_nodes_are_kept_out_of_pair_classes():
    rows = pd.DataFrame({"width_min_ft":[45, 46], "width_max_ft":[45, 51]})
    assert width_class(rows, PAIR_CONFIG) == UNKNOWN
    rows.loc[1, "width_max_ft"] = None
    assert width_class(rows, PAIR_CONFIG) == UNKNOWN
    assert unordered_pair("One-way", UNKNOWN, ["One-way","Two-way"]) == UNKNOWN
    ends = {"s":[(("n","M"), 5, 5), (("distant","M"), 80, 0)]}
    assert near_nodes(["s"], ends, 0, 0, 60) == {("n","M"):{"s"}}
