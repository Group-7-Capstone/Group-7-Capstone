"""Exploratory temporal features for Fei's experiment; no model fitting here."""
import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    "Nighttime": ["night_share_lag3", "unknown_time_share_lag3", "night_unavailable", "no_history"],
    "Weekend": ["weekend_share_lag3", "no_history"],
    "Recent change": ["recent_log_change", "no_history"],
}


def temporal_features(crashes, site_ids, first_year=2017, last_year=2025):
    """Use only t-3..t-1 to predict t. Missing times are not daytime events.

    Night means 22:00–05:59 local recorded clock time, not astronomical darkness.
    Zero shares with unavailable histories carry explicit missingness indicators.
    """
    c = crashes.loc[crashes.site_id.notna()].copy()
    if c.collision_id.isna().any() or c.collision_id.duplicated().any():
        raise ValueError("Expected one row per collision_id.")
    dates = pd.to_datetime(c.crash_date, errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid crash dates.")
    c["year"] = dates.dt.year
    c = c[c.year.between(first_year, last_year)].copy()
    dates = dates.loc[c.index]
    parts = c.crash_time.astype("string").str.strip().str.extract(r"^(\d{1,2}):(\d{2})$")
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce")
    known = (hour.between(0, 23) & minute.between(0, 59)).fillna(False)
    c["known_time"] = known.astype(int)
    c["night"] = (known & ((hour >= 22) | (hour < 6))).astype(int)
    c["weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    c["total"] = 1
    grid = pd.MultiIndex.from_product([sorted(site_ids), range(first_year, last_year + 2)], names=["site_id", "year"])
    annual = c.groupby(["site_id", "year"])[["total", "known_time", "night", "weekend"]].sum().reindex(grid, fill_value=0).reset_index()
    for col in ["total", "known_time", "night", "weekend"]:
        annual[col + "_lag3"] = annual.groupby("site_id")[col].transform(lambda x: x.shift(1).rolling(3, min_periods=3).sum())
    previous = annual.groupby("site_id").total.shift(1)
    earlier_mean = (annual.groupby("site_id").total.shift(2) + annual.groupby("site_id").total.shift(3)) / 2
    annual["recent_log_change"] = np.log1p(previous) - np.log1p(earlier_mean)
    out = annual[annual.year >= first_year + 3].copy()
    out["no_history"] = (out.total_lag3 == 0).astype(int)
    out["night_unavailable"] = (out.known_time_lag3 == 0).astype(int)
    out["night_share_lag3"] = (out.night_lag3 / out.known_time_lag3.replace(0, np.nan)).fillna(0)
    out["weekend_share_lag3"] = (out.weekend_lag3 / out.total_lag3.replace(0, np.nan)).fillna(0)
    out["unknown_time_share_lag3"] = (1 - out.known_time_lag3 / out.total_lag3.replace(0, np.nan)).fillna(0)
    return out.drop(columns=["total", "known_time", "night", "weekend"])


def select_features(train, requested):
    """Drop constant or identical columns using training data only."""
    used, dropped = [], {}
    for feature in dict.fromkeys(requested):
        values = train[feature].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training values in {feature}")
        if train[feature].nunique() <= 1:
            dropped[feature] = "constant in training data"
            continue
        duplicate = next((other for other in used
                          if np.array_equal(values, train[other].to_numpy(dtype=float))), None)
        if duplicate is not None:
            dropped[feature] = f"identical to {duplicate} in training data"
        else:
            used.append(feature)
    return used, dropped
