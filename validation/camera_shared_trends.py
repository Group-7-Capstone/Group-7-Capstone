"""Uncertainty, cluster selection, resampling, and numerical acceptance checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from model.camera_shape import fit_camera_shapes
from preprocessing.camera_shared_trends import descriptive_rows


def bootstrap_mean(matrix, repeats=500, seed=42):
    """Pointwise percentile bounds resampling whole histories, not individual weeks."""
    values = np.asarray(matrix, dtype=float)
    values = values[~np.isnan(values).all(axis=1)]
    if len(values) < 2:
        return np.full((2, values.shape[1]), np.nan)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=repeats).astype(float)
    denominators = weights @ np.isfinite(values)
    totals = weights @ np.nan_to_num(values)
    samples = np.divide(totals, denominators, out=np.full_like(totals, np.nan), where=denominators > 0)
    output = np.full((2, values.shape[1]), np.nan)
    valid = np.isfinite(values).sum(axis=0) >= 2
    output[:, valid] = np.nanpercentile(samples[:, valid], [2.5, 97.5], axis=0)
    return output


def select_shape_model(features, config):
    """Fit candidates; choose highest silhouette among stable, non-tiny solutions."""
    rows, models = [], {}
    n = len(features)
    if n < 2 * config["minimum_cluster_size"]:
        return None, pd.DataFrame()
    for k in config["k_values"]:
        if k >= n:
            continue
        fitted = fit_camera_shapes(features, k, config["seed"])
        labels = fitted.labels_
        sizes = np.bincount(labels, minlength=k)
        silhouette = silhouette_score(features, labels, sample_size=min(n, 2000), random_state=config["seed"]) if len(np.unique(labels)) > 1 else -1
        stability = []
        rng = np.random.default_rng(config["seed"])
        for repeat in range(config["stability_repeats"]):
            idx = rng.choice(n, size=max(k+1, int(n * config["stability_fraction"])), replace=False)
            replica = fit_camera_shapes(features.iloc[idx], k, config["seed"] + repeat + 1, n_init=10)
            stability.append(adjusted_rand_score(labels, replica.predict(features)))
        minimum = max(config["minimum_cluster_size"], int(np.ceil(n * config["minimum_cluster_fraction"])))
        accepted = bool(silhouette >= config["minimum_silhouette"] and np.mean(stability) >= config["minimum_stability"] and sizes.min() >= minimum)
        rows.append({"k": k, "groups": n, "silhouette": float(silhouette), "stability_ari": float(np.mean(stability)), "stability_min_ari": float(min(stability)), "smallest_cluster": int(sizes.min()), "eligible_solution": accepted, "selected": False})
        models[k] = fitted
    diagnostics = pd.DataFrame(rows)
    acceptable = diagnostics[diagnostics.eligible_solution].sort_values(["silhouette", "k"], ascending=[False, True])
    if acceptable.empty:
        return None, diagnostics
    k = int(acceptable.iloc[0].k)
    diagnostics.loc[diagnostics.k.eq(k), "selected"] = True
    return models[k], diagnostics


def subset_stability(features, fitted, subset, seed):
    selected = features.loc[features.index.intersection(subset)]
    if len(selected) < fitted.n_clusters * 10:
        return {"groups": len(selected), "ari": None, "status": "Too few groups for refit"}
    replica = fit_camera_shapes(selected, fitted.n_clusters, seed)
    return {"groups": len(selected), "ari": float(adjusted_rand_score(fitted.predict(selected), replica.labels_)), "status": "Refit on subset"}


def validate_panels(meta, weekly, primary, administrative, eligibility, membership, config):
    checks = []
    def check(name, result):
        checks.append({"check": name, "passed": bool(result)})
    check("Unique original camera identifiers", meta.index.is_unique)
    check("Unique camera-week input keys", not weekly.duplicated(["camera_id", "week_start"]).any())
    check("All original groups retained in eligibility", set(meta.index) == set(eligibility.camera_id))
    check("Activation dates unchanged", eligibility.inferred_activation_date.equals(meta.inferred_activation_date))
    observed = primary.to_numpy()[primary.notna()]
    check("Primary observed counts nonnegative integers", ((observed >= 0) & (observed % 1 == 0)).all())
    full = weekly[weekly.is_complete_week].copy()
    full["relative_week"] = ((pd.to_datetime(full.week_start) - full.camera_id.map(meta.anchor)).dt.days // 7) + 1
    full = full[full.relative_week.ge(1)]
    check("Complete observed counts reconcile", primary.sum().sum() == full.observed_ticket_count.sum())
    check("Administrative panel does not lose covered primary counts", np.allclose(primary.to_numpy()[primary.notna()], administrative.to_numpy()[primary.notna()]))
    h = config["primary_horizon"]
    check("Eligibility requires every anchored week", np.array_equal(eligibility[f"eligible_{h}w"], primary.loc[:, 1:h].notna().all(axis=1)))
    check("Every original group retained in cluster directory", set(membership.camera_id) == set(meta.index))
    check("Cluster assignments unique", membership.camera_id.is_unique)
    check("Incomplete histories never assigned", membership.loc[~membership[f"eligible_{h}w"], "cluster_id"].isna().all())
    return pd.DataFrame(checks)



SCENARIOS = {
    "balanced_26": "Fixed cohort: 26 weeks", "balanced_52": "Fixed cohort: 52 weeks",
    "balanced_104": "Fixed cohort: 104 weeks", "available": "All available histories",
    "high_confidence": "High sustained confidence: 52 weeks",
    "skip_first_four": "Fixed 52-week cohort: omit weeks 1–4",
    "administrative": "Fixed cutoff: 52 weeks", "baseline_at_least_five": "Baseline ≥5 tickets/week: 52 weeks",
}


def describe_matrix(matrix, normalized, kind, scenario, config):
    raw = descriptive_rows(matrix)
    indexed = descriptive_rows(normalized)
    raw_ci = bootstrap_mean(matrix, config["bootstrap_repeats"], config["seed"])
    index_ci = bootstrap_mean(normalized, config["bootstrap_repeats"], config["seed"])
    raw = raw.rename(columns={c: f"raw_{c}" for c in ["mean", "median", "q25", "q75"]})
    for c in ["n", "mean", "median", "q25", "q75"]:
        raw[f"index_{c}"] = indexed[c]
    raw["raw_ci_low"], raw["raw_ci_high"] = raw_ci
    raw["index_ci_low"], raw["index_ci_high"] = index_ci
    raw["raw_mean_4w"] = raw.raw_mean.rolling(4, min_periods=4).mean()
    raw["index_mean_4w"] = raw.index_mean.rolling(4, min_periods=4).mean()
    raw["camera_type"], raw["scenario"] = kind, scenario
    raw["scenario_label"] = SCENARIOS.get(scenario, scenario.replace("cohort_", "Activation year: "))
    raw["cohort_groups"] = len(matrix)
    return raw
