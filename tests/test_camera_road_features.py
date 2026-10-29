import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from preprocessing.camera_road_features import (
    UNKNOWN, street_key, consensus_attributes, match_camera_roads, road_cohort_ids,
)


def test_split_tokens_ordinals_and_directional_names():
    assert street_key('SPRIN GFIELD BLVD') == street_key('SPRINGFIELD BOULEVARD')
    assert street_key('E 9 4TH ST') == street_key('EAST 94 STREET')
    assert street_key('85TH R D') == street_key('85 ROAD')
    assert street_key('BEV ERLEY RD') != street_key('BEVERLY RD')  # No fuzzy guess.
    assert street_key('QUEENS BLVD SR') != street_key('QUEENS BLVD')


def candidates(lanes=('2','2'), low=(34,34), high=(34,34), directions=('T','T')):
    return pd.DataFrame({'Number_Travel_Lanes':lanes,'StreetWidth_Min':low,
                         'StreetWidth_Max':high,'TrafDir':directions})


def test_consensus_preserves_ambiguity_and_units():
    values = consensus_attributes(candidates())
    assert values['width_group'] == '10–<15 m'
    assert np.isclose(values['width_min_m'],10.3632)
    assert values['lanes_group'] == '2 lanes'
    assert values['direction_group'] == 'Two-way'
    assert consensus_attributes(candidates(lanes=('2','3')))['lanes_group'] == UNKNOWN
    assert consensus_attributes(candidates(low=(30,34)))['width_group'] == UNKNOWN
    assert consensus_attributes(candidates(directions=('W','T')))['direction_group'] == UNKNOWN
    assert consensus_attributes(candidates(lanes=('','2')))['lanes_group'] == UNKNOWN
    assert consensus_attributes(candidates(high=(0,34)))['width_group'] == UNKNOWN


def test_explicit_config_is_honored_without_mutating_input():
    from preprocessing.camera_road_features import ROAD_CONFIG
    frame = candidates()
    original = frame.copy(deep=True)
    assert consensus_attributes(frame, {**ROAD_CONFIG, 'maximum_lanes': 1})['lanes_group'] == UNKNOWN
    pd.testing.assert_frame_equal(frame, original)


def test_width_boundaries_and_unmarked_lanes_not_parking():
    r = consensus_attributes(candidates(lanes=('4','5'),low=(50,50),high=(55,55)))
    assert r['lanes_group'] == '4+ lanes'
    assert r['width_group'] == '15–<20 m'
    assert consensus_attributes(candidates(lanes=('0','0')))['lanes_group'] == UNKNOWN


def fixture_roads():
    base = dict(FeatureTyp='0',RB_Layer='B',RW_TYPE='1',TrafDir='T',LBoro=1,RBoro=1,
                NodeLevelF='M',NodeLevelT='M',StreetWidth_Min=34,StreetWidth_Max=34,
                Number_Travel_Lanes='2',Number_Park_Lanes='2',Number_Total_Lanes='4',SAFStreetName='')
    roads = gpd.GeoDataFrame([
        {**base,'Street':'FIRST AVENUE','Join_ID':'a','SegmentID':'1','NodeIDFrom':'a','NodeIDTo':'j','geometry':LineString([(-73.99,40.75),(-73.99,40.751)])},
        {**base,'Street':'SECOND STREET','Join_ID':'b','SegmentID':'2','NodeIDFrom':'j','NodeIDTo':'b','geometry':LineString([(-73.99,40.751),(-73.989,40.751)])}
    ],crs=4326)
    roads['XFrom'] = roads.geometry.map(lambda line: line.coords[0][0])
    roads['YFrom'] = roads.geometry.map(lambda line: line.coords[0][1])
    roads['XTo'] = roads.geometry.map(lambda line: line.coords[-1][0])
    roads['YTo'] = roads.geometry.map(lambda line: line.coords[-1][1])
    summary = pd.DataFrame([dict(camera_id='x',search_query='FIRST AVE AND SECOND ST, NEW YORK, NY',camera_type='Speed',county_labels='MN',detection_confidence='high_sustained',Latitude=40.751,Longitude=-73.99,inferred_activation_date='2020-01-01')])
    return summary, roads, pd.DataFrame(columns=['Join_ID','Street'])


def test_join_requires_borough_connected_node_and_excludes_divided_road():
    s,r,a = fixture_roads()
    f,audit = match_camera_roads(s,r,a)
    assert f.strict_match.iloc[0] and f.candidate_segments.iloc[0] == 1
    assert len(audit) == 1
    f,_ = match_camera_roads(s.assign(county_labels='QN'),r,a)
    assert not f.any_match.iloc[0]
    disconnected = r.copy();disconnected.loc[1,'NodeIDFrom']='different'
    f,_ = match_camera_roads(s,disconnected,a)
    assert f.match_status.iloc[0] == 'Cross street / connected-node mismatch'
    split = r.copy();split.loc[0,'RB_Layer']='R'
    f,_ = match_camera_roads(s,split,a)
    assert f.match_status.iloc[0] == 'Divided road or generic-roadbed ambiguity'
    assert f.direction_group.iloc[0] == UNKNOWN


def test_remote_coordinates_and_road_levels_not_silently_snapped():
    s,r,a = fixture_roads()
    f,_ = match_camera_roads(s.assign(Longitude=-74.1),r,a)
    assert not f.any_match.iloc[0]
    missing = r.copy();missing.loc[0,'NodeIDTo']='';missing.loc[1,'NodeIDFrom']=''
    f,_ = match_camera_roads(s,missing,a)
    assert not f.any_match.iloc[0]
    elevated = r.copy();elevated.loc[1,'NodeLevelF']='Q'
    f,_ = match_camera_roads(s,elevated,a)
    assert not f.any_match.iloc[0]


def test_cohorts_keep_unknown_without_double_counting_and_skip_missing_weeks():
    features = pd.DataFrame({'camera_id':['a','b','c','d'],'camera_type':['Speed']*4,
                             'lanes_group':['2 lanes']*4,'strict_match':[True,False,True,True],
                             'any_match':[True,True,True,True]})
    primary = pd.DataFrame([[2]*26,[10]*26,[0]*26,[3]*25+[np.nan]],index=['a','b','c','d'],columns=range(1,27))
    assert road_cohort_ids(features,primary,'Speed',26,'lanes','2 lanes','intersection_only',0).tolist() == ['a','c']
    assert road_cohort_ids(features,primary,'Speed',26,'lanes',UNKNOWN,'intersection_only',0).tolist() == ['b']
    assert road_cohort_ids(features,primary,'Speed',26,'lanes','2 lanes','include_street_only',5).tolist() == ['b']


def test_nearby_roads_that_join_far_away_are_not_an_intersection_match():
    s,r,a = fixture_roads()
    s['Latitude'] = 40.75
    r.at[0,'geometry'] = LineString([(-73.99,40.75),(-73.99,40.752)])
    r.at[1,'geometry'] = LineString([(-73.99,40.752),(-73.9899,40.75)])
    r['XFrom'] = r.geometry.map(lambda line:line.coords[0][0])
    r['YFrom'] = r.geometry.map(lambda line:line.coords[0][1])
    r['XTo'] = r.geometry.map(lambda line:line.coords[-1][0])
    r['YTo'] = r.geometry.map(lambda line:line.coords[-1][1])
    f,_ = match_camera_roads(s,r,a)
    assert f.match_status.iloc[0] == 'Cross street / connected-node mismatch'
