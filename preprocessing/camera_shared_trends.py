"""Comparable observed-ticket panels. No model fitting or validation occurs here."""
import numpy as np
import pandas as pd

from pathlib import Path
import json

DEFAULTS = {
    "cutoff": "2026-08-30", "coverage_start": "2016-07-04",
    "horizons": [26, 52, 104], "primary_horizon": 52,
    "baseline_weeks": 13, "block_weeks": 4, "skip_weeks": 4,
    "seed": 42, "bootstrap_repeats": 500, "k_values": [2, 3, 4, 5, 6],
    "stability_repeats": 12, "stability_fraction": 0.8,
    "minimum_silhouette": 0.2, "minimum_stability": 0.75,
    "minimum_cluster_size": 10, "minimum_cluster_fraction": 0.02,
    "sparse_max_nonzero_weeks": 13, "constant_tolerance": 1e-10,
}


def load_inputs(directory):
    directory = Path(directory)
    return (pd.read_csv(directory / "camera_summary.csv"),
            pd.read_parquet(directory / "camera_weekly_trends.parquet"),
            json.loads((directory / "reconciliation.json").read_text(encoding="utf-8")))


def aligned_panels(summary, weekly, audit, config=DEFAULTS):
    """Anchor once at the first Monday on/after activation, never at first valid data."""
    if weekly.duplicated(["camera_id", "week_start"]).any():
        raise ValueError("Duplicate camera-week input keys")
    meta = summary.copy().set_index("camera_id", drop=False).sort_index()
    activation = pd.to_datetime(meta.inferred_activation_date)
    meta["anchor"] = activation + pd.to_timedelta((-activation.dt.weekday) % 7, unit="D")
    meta["activation_year"] = activation.dt.year
    weeks = weekly.copy()
    weeks["week_start"] = pd.to_datetime(weeks.week_start)
    weeks["relative_week"] = ((weeks.week_start - weeks.camera_id.map(meta.anchor)).dt.days // 7) + 1
    cutoff = pd.Timestamp(config["cutoff"])
    max_weeks = max(int((cutoff - meta.anchor.min()).days // 7) + 1, max(config["horizons"]))
    columns = pd.RangeIndex(1, max_weeks + 1, name="relative_week")
    valid = weeks.relative_week.ge(1)
    # pivot intentionally raises for duplicate camera/week keys.
    observed = weeks.loc[valid].pivot(index="camera_id", columns="relative_week", values="observed_ticket_count").reindex(index=meta.index, columns=columns)
    full = weeks.loc[valid].pivot(index="camera_id", columns="relative_week", values="is_complete_week").reindex(index=meta.index, columns=columns).fillna(False).astype(bool)
    primary = observed.where(full)
    administrative = observed.copy()
    for camera_id, anchor in meta.anchor.items():
        dates = anchor + pd.to_timedelta((columns.to_numpy() - 1) * 7, unit="D")
        covered = (dates >= pd.Timestamp(config["coverage_start"])) & (dates + pd.Timedelta(days=6) <= cutoff)
        if audit["all_sources_complete"]:
            # Different estimand: tickets observed at a registered location through a fixed cutoff.
            administrative.loc[camera_id] = administrative.loc[camera_id].fillna(0).where(covered)
        else:
            administrative.loc[camera_id] = primary.loc[camera_id]
    calendar = weeks[weeks.is_complete_week & weeks.relative_week.ge(1)].pivot(index="camera_id", columns="week_start", values="observed_ticket_count").reindex(meta.index)
    return meta, primary, administrative, calendar


def eligibility_table(meta, primary, administrative, config=DEFAULTS):
    result = meta[["camera_id", "search_query", "camera_type", "detection_confidence", "inferred_activation_date", "anchor", "activation_year", "last_actual_ticket_date", "Latitude", "Longitude"]].copy()
    baseline = primary.loc[:, 1:config["baseline_weeks"]]
    result["baseline_mean"] = baseline.mean(axis=1).where(baseline.notna().all(axis=1))
    result["baseline_status"] = np.select([result.baseline_mean.isna(), result.baseline_mean.eq(0), result.baseline_mean.lt(5)], ["Incomplete baseline", "Zero baseline", "Low positive baseline (<5 tickets/week)"], default="Positive baseline")
    for horizon in config["horizons"]:
        window = primary.loc[:, 1:horizon]
        result[f"eligible_{horizon}w"] = window.notna().all(axis=1)
        result[f"complete_weeks_{horizon}w"] = window.notna().sum(axis=1)
    h = config["primary_horizon"]
    result["weekly_mean_52w"] = primary.loc[:, 1:h].mean(axis=1).where(result[f"eligible_{h}w"])
    result["nonzero_weeks"] = primary.loc[:, 1:h].gt(0).sum(axis=1).where(result[f"eligible_{h}w"])
    result["eligible_administrative_52w"] = administrative.loc[:, 1:h].notna().all(axis=1)
    result["cluster_status"] = np.where(result[f"eligible_{h}w"], "Awaiting shape model", "Incomplete first 52 weeks")
    result["method"] = "Unchanged Eric registry; first full Monday-Sunday week; complete fixed window"
    result["confidence"] = result.detection_confidence
    return result


def normalize_baseline(matrix, baseline_columns):
    baseline = matrix.loc[:, baseline_columns]
    means = baseline.mean(axis=1).where(baseline.notna().all(axis=1) & baseline.mean(axis=1).gt(0))
    return matrix.div(means, axis=0) * 100


def descriptive_rows(matrix):
    counts = matrix.notna().sum(axis=0)
    return pd.DataFrame({"time": matrix.columns, "n": counts.to_numpy(),
                         "mean": matrix.mean(axis=0).to_numpy(),
                         "median": matrix.median(axis=0).to_numpy(),
                         "q25": matrix.quantile(.25, axis=0).to_numpy(),
                         "q75": matrix.quantile(.75, axis=0).to_numpy()})


def shape_features(matrix, config=DEFAULTS):
    """Four-week means, log1p, then within-camera standardization."""
    block = config["block_weeks"]
    counts = matrix.to_numpy(dtype=float)
    if counts.shape[1] % block or not np.isfinite(counts).all():
        raise ValueError("Shape features require a complete window divisible by block length")
    blocks = counts.reshape(len(matrix), -1, block).mean(axis=2)
    logged = np.log1p(blocks)
    scale = logged.std(axis=1)
    good = scale > config["constant_tolerance"]
    standardized = (logged[good] - logged[good].mean(axis=1, keepdims=True)) / scale[good, None]
    return pd.DataFrame(standardized, index=matrix.index[good], columns=range(1, blocks.shape[1]+1)), matrix.index[~good]


def profile_cluster(meta, matrix, features, labels):
    """Readable profiles; labels reflect fitted shapes, not presumed camera effects."""
    memberships, profiles, curves = [], [], []
    raw_labels = np.unique(labels)
    ordering = sorted(raw_labels, key=lambda c: features.to_numpy()[labels == c, -3:].mean() - features.to_numpy()[labels == c, :3].mean())
    for number, label in enumerate(ordering, 1):
        ids = features.index[labels == label]
        counts = matrix.loc[ids]
        shape = features.loc[ids]
        centroid = shape.mean(axis=0).to_numpy()
        change = float(centroid[-3:].mean() - centroid[:3].mean())
        peak = int(np.argmax(centroid)) + 1
        direction = "Declining" if change < -.5 else "Increasing" if change > .5 else "Middle-period peak" if 3 <= peak <= 11 else "Mixed / approximately flat"
        cluster_id = f"{meta.loc[ids[0], 'camera_type'].lower().replace(' ', '-')}-{number}"
        distance = np.linalg.norm(shape.to_numpy() - centroid, axis=1)
        representative = ids[np.argsort(distance)[:3]].tolist()
        early, late = counts.iloc[:, :13].mean(axis=1), counts.iloc[:, -13:].mean(axis=1)
        pct = (late / early.where(early.gt(0)) - 1) * 100
        profiles.append({"camera_type": meta.loc[ids[0], "camera_type"], "cluster_id": cluster_id,
                         "cluster_label": f"C{number}: {direction}", "groups": len(ids),
                         "median_weekly_tickets": float(counts.mean(axis=1).median()),
                         "median_change_pct": float(pct.median()), "zero_week_fraction": float(counts.eq(0).to_numpy().mean()),
                         "high_confidence_groups": int(meta.loc[ids].detection_confidence.eq("high_sustained").sum()),
                         "representative_ids": ";".join(representative)})
        for camera_id, dist in zip(ids, distance):
            memberships.append({"camera_id": camera_id, "cluster_id": cluster_id, "cluster_label": profiles[-1]["cluster_label"], "distance_to_centroid": float(dist), "is_representative": camera_id in representative})
        indexed = normalize_baseline(counts, list(range(1, 14)))
        for week in counts.columns:
            curves.append({"camera_type": meta.loc[ids[0], "camera_type"], "cluster_id": cluster_id,
                           "cluster_label": profiles[-1]["cluster_label"], "week": int(week), "n": len(ids),
                           "mean": float(counts[week].mean()), "median": float(counts[week].median()),
                           "index_mean": float(indexed[week].mean()), "index_median": float(indexed[week].median()),
                           "index_q25": float(indexed[week].quantile(.25)), "index_q75": float(indexed[week].quantile(.75)),
                           "shape_mean": float(centroid[(week-1)//4])})
    return pd.DataFrame(memberships), pd.DataFrame(profiles), pd.DataFrame(curves)
