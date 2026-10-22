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
