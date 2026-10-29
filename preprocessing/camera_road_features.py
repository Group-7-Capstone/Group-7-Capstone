"""Conservative first-named-street attributes; original camera groups stay intact."""
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

ROAD_CONFIG = {"radius_m": 60, "crs": 26918, "width_edges_m": [10, 15, 20],
               "minimum_width_m": 2, "maximum_width_m": 80,
               "maximum_lanes": 20, "small_group_n": 10}
UNKNOWN = "Unknown / ambiguous"
GROUPS = {"lanes": ["1 lane", "2 lanes", "3 lanes", "4+ lanes", UNKNOWN],
          "width": ["<10 m", "10–<15 m", "15–<20 m", "20+ m", UNKNOWN],
          "direction": ["One-way", "Two-way", UNKNOWN]}
BOROUGHS = {"MN": 1, "NY": 1, "BX": 2, "BK": 3, "K": 3, "QN": 4, "Q": 4, "ST": 5, "R": 5}


def street_key(value):
    """Normalize only for this join; collapse split tokens, ordinals and suffixes."""
    if pd.isna(value):
        return ""
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    for long, short in [("EXPRESSWAY", "EXPWY"), ("BOULEVARD", "BLVD"),
                        ("AVENUE", "AV"), ("STREET", "ST"), ("PARKWAY", "PKWY"),
                        ("TERRACE", "TER"), ("DRIVE", "DR"), ("PLACE", "PL"),
                        ("ROAD", "RD"), ("LANE", "LN"), ("COURT", "CT"),
                        ("AVE", "AV"), ("PKY", "PKWY"), ("PY", "PKWY"), ("BV", "BLVD")]:
        text = re.sub(re.escape(long) + r"$", short, text)
    text = re.sub(r"^(EAST|WEST|NORTH|SOUTH)(?=\d)", lambda m: m[1][0], text)
    text = re.sub(r"(\d)(ST|ND|RD|TH)(?=(ST|AV|RD|DR|PL|LN|TER|CT)$)", r"\1", text)
    return text


def address_parts(value):
    text = str(value).upper().split(",")[0].strip()
    parts = re.split(r"\s+AND\s+|\s*&\s*", text, maxsplit=1)
    if len(parts) == 2:
        return street_key(parts[0]), street_key(parts[1]), "Intersection"
    # A period is a strong boundary in Eric's block-range address strings.
    if "." in text:
        return street_key(text.split(".", 1)[0]), "", "Street only"
    return "", "", "Unsupported address"


def lane_band(value, config=ROAD_CONFIG):
    if pd.isna(value) or value < 1 or value % 1 or value > config["maximum_lanes"]:
        return UNKNOWN
    return f"{int(value)} lane" + ("s" if value > 1 else "") if value < 4 else "4+ lanes"


def width_band(value, config=ROAD_CONFIG):
    if pd.isna(value) or not config["minimum_width_m"] <= value <= config["maximum_width_m"]:
        return UNKNOWN
    return GROUPS["width"][int(np.searchsorted(config["width_edges_m"], value, side="right"))]


def consensus_attributes(candidates, config=ROAD_CONFIG):
    """All plausible adjoining segments must agree on a category; never average widths."""
    lanes = pd.to_numeric(candidates.Number_Travel_Lanes, errors="coerce")
    low = pd.to_numeric(candidates.StreetWidth_Min, errors="coerce") * .3048
    high = pd.to_numeric(candidates.StreetWidth_Max, errors="coerce") * .3048
    lane_values = lanes.map(lambda value: lane_band(value, config))
    width_values = pd.concat([low, high]).map(lambda x: width_band(x, config))
    direction_values = candidates.TrafDir.map({"A": "One-way", "W": "One-way", "T": "Two-way"}).fillna(UNKNOWN)
    def agree(values):
        return values.iloc[0] if len(values) and values.nunique() == 1 else UNKNOWN
    return {"lanes_group": agree(lane_values),
            "width_group": agree(width_values) if low.le(high).all() else UNKNOWN,
            "direction_group": agree(direction_values),
            "travel_lanes_min": lanes.min(), "travel_lanes_max": lanes.max(),
            "width_min_m": low.min(), "width_max_m": high.max()}


def match_camera_roads(summary, roads, aliases, config=ROAD_CONFIG):
    """Require borough, street text and distance; intersections also need shared nodes."""
    roads = roads[roads.FeatureTyp.eq("0") & roads.geometry.notna()].copy()
    transform = Transformer.from_crs(roads.crs, config["crs"], always_xy=True)
    roads['from_x'], roads['from_y'] = transform.transform(roads.XFrom, roads.YFrom)
    roads['to_x'], roads['to_y'] = transform.transform(roads.XTo, roads.YTo)
    roads = roads.to_crs(config["crs"])
    for col in ["RB_Layer", "RW_TYPE", "TrafDir", "NodeIDFrom", "NodeIDTo", "NodeLevelF", "NodeLevelT"]:
        roads[col] = roads[col].fillna("").astype(str).str.strip()
    roads = roads.reset_index(drop=True)
    alternate = aliases.groupby("Join_ID").Street.agg(list).to_dict()
    roads["name_keys"] = [set(filter(None, map(street_key, [r.Street, r.SAFStreetName] + alternate.get(r.Join_ID, [])))) for r in roads.itertuples()]
    index = roads.sindex
    cameras = gpd.GeoDataFrame(summary.copy(), geometry=gpd.points_from_xy(summary.Longitude, summary.Latitude), crs=4326).to_crs(config["crs"])
    records, audit = [], []
    for row in cameras.itertuples():
        first, cross, form = address_parts(row.search_query)
        result = {"camera_id": row.camera_id, "search_query": row.search_query, "camera_type": row.camera_type,
                  "county_labels": row.county_labels, "detection_confidence": row.detection_confidence,
                  "Latitude": row.Latitude, "Longitude": row.Longitude,
                  "inferred_activation_date": row.inferred_activation_date,
                  "first_street_key": first, "cross_street_key": cross, "address_form": form,
                  "match_status": "Unsupported address", "method": "Borough + normalized first street + 60 m coordinate neighborhood; shared node for intersections",
                  "match_confidence": "Unresolved", "enforced_approach": "Unverified; directions merged upstream",
                  "candidate_segments": 0, "matched_segment_ids": "", "matched_street_names": "", "match_distance_m": np.nan,
                  "lanes_group": UNKNOWN, "width_group": UNKNOWN, "direction_group": UNKNOWN,
                  "travel_lanes_min": np.nan, "travel_lanes_max": np.nan, "width_min_m": np.nan, "width_max_m": np.nan}
        records.append(result)
        borough = BOROUGHS.get(str(row.county_labels))
        if borough is None:
            result["match_status"] = "Missing or ambiguous ticket borough"
            continue
        if not (-74.3 <= row.Longitude <= -73.65 and 40.45 <= row.Latitude <= 40.95):
            result["match_status"] = "Invalid original coordinate"
            continue
        if not first or first == cross:
            continue
        nearby = roads.iloc[index.query(row.geometry.buffer(config["radius_m"]), predicate="intersects")].copy()
        nearby = nearby[(nearby.LBoro.eq(borough) | nearby.RBoro.eq(borough))]
        nearby["distance_m"] = nearby.geometry.distance(row.geometry)
        nearby = nearby[nearby.distance_m.le(config["radius_m"])]
        nearby['from_near'] = np.hypot(nearby.from_x-row.geometry.x, nearby.from_y-row.geometry.y).le(config['radius_m'])
        nearby['to_near'] = np.hypot(nearby.to_x-row.geometry.x, nearby.to_y-row.geometry.y).le(config['radius_m'])
        main = nearby[nearby.name_keys.map(lambda keys: first in keys)].copy()
        if main.empty:
            result["match_status"] = "Street / coordinate mismatch"
            continue
        if cross:
            other = nearby[nearby.name_keys.map(lambda keys: cross in keys)]
            # IDs and level codes are paired; crossing bridges cannot supply a junction.
            nodes = set(zip(other.loc[other.from_near, 'NodeIDFrom'], other.loc[other.from_near, 'NodeLevelF'])) | set(zip(other.loc[other.to_near, 'NodeIDTo'], other.loc[other.to_near, 'NodeLevelT']))
            nodes = {(node, level) for node, level in nodes if node not in ('', '0', '0000000') and level not in ('', '*')}
            linked = [(r.from_near and (r.NodeIDFrom, r.NodeLevelF) in nodes) or (r.to_near and (r.NodeIDTo, r.NodeLevelT) in nodes) for r in main.itertuples()]
            main = main[linked]
            if main.empty:
                result["match_status"] = "Cross street / connected-node mismatch"
                continue
        # Divided roadbeds cannot be labeled a one-way street without a verified approach.
        if not main.RB_Layer.eq("B").all():
            result["match_status"] = "Divided road or generic-roadbed ambiguity"
            continue
        if not main.RW_TYPE.eq("1").all() or not main.NodeLevelF.eq(main.NodeLevelT).all():
            result["match_status"] = "Nonstandard road type or level change"
            continue
        main = main.drop_duplicates("SegmentID").sort_values(["distance_m", "SegmentID"])
        result.update(consensus_attributes(main, config))
        result.update({"match_status": "Intersection street match" if cross else "Street-only match",
                       "match_confidence": "Text + borough + coordinate + connected node" if cross else "Text + borough + coordinate only",
                       "candidate_segments": len(main), "matched_segment_ids": ";".join(main.SegmentID.astype(str)),
                       "matched_street_names": ";".join(sorted(main.Street.unique())), "match_distance_m": float(main.distance_m.min())})
        for road in main.itertuples():
            audit.append({"camera_id": row.camera_id, "segment_id": road.SegmentID, "street": road.Street,
                          "distance_m": road.distance_m, "rb_layer": road.RB_Layer, "rw_type": road.RW_TYPE,
                          "node_from": road.NodeIDFrom, "node_to": road.NodeIDTo,
                          "level_from": road.NodeLevelF, "level_to": road.NodeLevelT,
                          "travel_lanes": road.Number_Travel_Lanes, "parking_lanes": road.Number_Park_Lanes,
                          "width_min_ft": road.StreetWidth_Min, "width_max_ft": road.StreetWidth_Max, "traffic_direction": road.TrafDir})
    result = pd.DataFrame(records)
    result["strict_match"] = result.match_status.eq("Intersection street match")
    result["any_match"] = result.match_status.isin(["Intersection street match", "Street-only match"])
    return result, pd.DataFrame(audit)


def road_cohort_ids(features, primary, kind, horizon, dimension, group, policy, baseline_min):
    """One camera contributes once; unknown includes unmatched/ambiguous records."""
    rows = features.set_index("camera_id").reindex(primary.index)
    category = rows[f"{dimension}_group"].where(rows["strict_match" if policy == "intersection_only" else "any_match"], UNKNOWN)
    mask = rows.camera_type.eq(kind) & category.eq(group) & primary.loc[:, 1:horizon].notna().all(axis=1)
    if baseline_min:
        mask &= primary.loc[:, 1:13].mean(axis=1).ge(baseline_min)
    return primary.index[mask]
