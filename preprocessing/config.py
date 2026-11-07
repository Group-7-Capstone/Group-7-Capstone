"""Analytic choices for preprocessing, kept in one place so they are visible and auditable."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CrashConfig:
    # Crashes on or after this date are kept.
    start_date: str = "2017-01-01"

    # Coordinates outside this box (incl. 0/0 placeholders) are treated as not geocoded.
    lat_bounds: tuple[float, float] = (40.47, 40.92)
    lon_bounds: tuple[float, float] = (-74.27, -73.68)

    # Public data has no "seriously injured" field, so an injured person counts as
    # severe if their complaint or emotional status is in these lists (KSI proxy).
    severe_complaints: tuple[str, ...] = (
        "Fracture - Distorted - Dislocation",
        "Fracture - Dislocation",
        "Internal",
        "Severe Bleeding",
        "Severe Lacerations",
        "Crush Injuries",
        "Amputation",
        "Paralysis",
        "Moderate Burn",
        "Severe Burn",
        "Concussion",
    )
    severe_emotional_status: tuple[str, ...] = (
        "Unconscious",
        "Semiconscious",
        "Apparent Death",
    )


@dataclass(frozen=True)
class SiteConfig:
    # Projected CRS used for distances (NY State Plane Long Island, feet).
    crs: str = "EPSG:2263"
    # Peak-radius clustering: a site is every crash location within this many
    # feet of its center (the busiest crash location not already claimed).
    radius_ft: float = 100.0
    # Sites are built only from crashes before this date, so held-out years
    # never define which sites exist. Test periods must start on or after it.
    build_end: str = "2024-01-01"


@dataclass(frozen=True)
class PanelConfig:
    # Site x year panel covers these calendar years; the last must be complete.
    first_year: int = 2017
    last_year: int = 2025
    # Features sum the `lag_years` years before each prediction year.
    lag_years: int = 3
    # New York changed its serious-injury coding in 2019 (MMUCC 4th edition): new
    # complaint categories appear and the KSI share of injury crashes jumps. Only
    # prediction years whose whole lag window starts on or after this year are kept,
    # so every KSI-based feature uses one definition.
    ksi_definition_start: int = 2019
    # A camera belongs to the nearest site centroid within this many feet.
    camera_radius_ft: float = 150.0


@dataclass(frozen=True)
class RoadwayConfig:
    # LION segments within this many feet of a site centroid describe the site
    # (same as SiteConfig.radius_ft, so a site's roads span its crash locations).
    radius_ft: float = 100.0
    # LION FeatureTyp "0" is a street; the rest are rail, shoreline, boundaries, etc.
    street_feature_types: tuple[str, ...] = ("0",)
    # RW_TYPE codes that aren't roads a car drives on: boardwalk, path, step street,
    # driveway, non-physical segment, ferry route.
    non_road_rw_types: tuple[int, ...] = (5, 6, 7, 8, 12, 14)
    # Highway (2) and ramp (9) segments.
    highway_rw_types: tuple[int, ...] = (2, 9)
    # Lanes, width and speed are recorded on roadbed-level segments; intersection
    # legs are counted on the generic network (one centerline per street).
    roadbed_segment_types: tuple[str, ...] = ("U", "R", "B", "C", "T", "S", "E")
    generic_segment_types: tuple[str, ...] = ("U", "B", "G")
    # A node where at least this many generic street segments meet is an intersection.
    min_intersection_legs: int = 3
    # Lane width = (curb-to-curb width - parking lanes * this) / travel lanes.
    parking_lane_ft: float = 8.0
    # Estimated lane widths outside this range are treated as data errors (NA).
    lane_width_bounds_ft: tuple[float, float] = (7.0, 20.0)


@dataclass(frozen=True)
class CameraDirectionConfig:
    # Raw camera ticket tables (one per fiscal year), already filtered to camera
    # violation codes at pull time.
    ticket_tables: tuple[str, ...] = tuple(f"tickets_fy{y}" for y in range(2017, 2028))
    # Tickets issued before this date are ignored (same cutoff as the activation
    # notebook, eric/camera_activation_inference.ipynb).
    start_date: str = "2016-01-01"
    # A direction with less than this share of its location's tickets may be a data
    # entry error rather than a separately enforced approach; it gets low confidence.
    min_direction_share: float = 0.05
    # Location: LION street names shorter than this (after normalizing) are ignored as
    # cross streets, so e.g. "AVI" isn't found inside "HAVILAND".
    min_street_name_length: int = 3
    # Nodes where the same street pair meets count as one intersection (divided roads
    # meet at several nodes) if they're all within this many feet of each other.
    max_node_spread_ft: float = 400.0
    # Bearing: LION roadbed segments of the camera's own street within this many feet
    # of the geocoded camera point are candidates.
    street_radius_ft: float = 150.0
    # A LION street name matches the camera's street if it starts the normalized
    # search_query, or failing that, is at least this similar to its start (typos).
    min_street_name_similarity: float = 0.8
    # The street's bearing is measured over this many feet either side of the camera.
    tangent_half_length_ft: float = 20.0
    # A street running more than this far off the ticket's grid direction (a diagonal
    # street labeled N/S or E/W) still gets a bearing, with "low" confidence.
    # Manhattan's grid is about 29 degrees off true north, so this must exceed that.
    max_bearing_offset_deg: float = 60.0
    # "high" bearing confidence needs an exact name match, the camera within this many
    # feet of the street and the street within 45 degrees of the grid direction;
    # anything else within `max_bearing_offset_deg` is "medium".
    high_confidence_distance_ft: float = 50.0
