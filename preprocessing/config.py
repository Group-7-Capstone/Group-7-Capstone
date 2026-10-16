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
    # DBSCAN radius in feet; with min_samples=1 every crash belongs to a site.
    eps_ft: float = 80.0
    # Sites are built only from crashes before this date, so held-out years
    # never define which sites exist. Test periods must start on or after it.
    build_end: str = "2024-01-01"
