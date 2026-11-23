"""Discovery/holdout splits and frozen feature comparisons for issue #58."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import silhouette_score

from preprocessing.camera_shared_trends import shape_features
from validation.camera_shared_trends import select_shape_model


def camera_halves(camera_ids, seed):
    """Deterministic 50/50 random split, independent of outcomes and input order."""
    ids = pd.Index(camera_ids).sort_values()
    if not ids.is_unique:
        raise ValueError("Repeated camera IDs")
    shuffled = np.random.default_rng(seed).permutation(ids)
    return pd.Series(["A"] * (len(ids) // 2) + ["B"] * (len(ids) - len(ids) // 2), index=shuffled, name="split")


def discover_clusters(outcomes, panel, split, config):
    """Select k and stability on A only; B gets predictions from that frozen fit."""
    valid = outcomes.outcome_status.eq("eligible") & panel.notna().all(axis=1)
    features, constant = shape_features(panel.loc[valid], config)
    discovery = features.loc[split.reindex(features.index).eq("A")]
    model, diagnostics = select_shape_model(discovery, config)
    assignments = pd.Series(pd.NA, index=outcomes.index, dtype="Int64", name="cluster")
    if model is not None:
        # Stable meaning across sensitivity runs: C1 has the least decline in A.
        order = outcomes.loc[discovery.index, "decline_52w_pct"].groupby(model.labels_).mean().sort_values().index
        mapping = {label: i + 1 for i, label in enumerate(order)}
        assignments.loc[features.index] = pd.Series(model.predict(features), index=features.index).map(mapping)
    holdout = features.loc[split.reindex(features.index).eq("B")]
    labels = assignments.reindex(holdout.index).dropna().astype(int)
    holdout_silhouette = None
    if 1 < labels.nunique() < len(labels):
        holdout_silhouette = float(silhouette_score(holdout.loc[labels.index], labels, sample_size=min(len(labels), 2000), random_state=config["seed"]))
    return assignments, diagnostics, {
        "mode": "clusters" if model is not None else "decline_52w_pct",
        "shape_eligible_A": len(discovery),
        "shape_eligible_B": int(split.reindex(features.index).eq("B").sum()),
        "constant_curves": len(constant),
        "selected_k": None if model is None else model.n_clusters,
        "holdout_silhouette": holdout_silhouette,
        "holdout_cluster_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
    }


def feature_rules(discovery, config):
    """Learn numeric cutpoints and categorical reference levels from A only."""
    rules = {}
    for feature in config["numeric_features"]:
        value = discovery[feature].dropna()
        if len(value) and value.nunique() > 1:
            rules[feature] = {"cutpoint": float(value.median()), "reference": "at_or_below", "groups": ["above"]}
    for feature in config["categorical_features"]:
        counts = discovery[feature].dropna().value_counts()
        if len(counts) > 1:
            reference = min(counts.index[counts.eq(counts.max())])
            rules[feature] = {"reference": reference, "groups": sorted(set(counts.index) - {reference})}
    return rules


def apply_feature_rule(values, rule):
    if "cutpoint" in rule:
        return pd.Series(np.where(values.gt(rule["cutpoint"]), "above", "at_or_below"), index=values.index).where(values.notna())
    return values.where(values.isin([rule["reference"], *rule["groups"]]))


def compare_feature_groups(frame, rules, config, response="decline_52w_pct"):
    """Mean decline differences in percentage points, within borough × activation year.

    A defines groups and stratum weights. A and B estimates use the same common
    strata; cells need >=2 cameras per group and split to estimate uncertainty.
    Normal intervals are approximate. All planned contrasts get a Holm-adjusted
    B p-value; no selection or re-binning is done using B outcomes.
    """
    results, details = [], []
    z = norm.ppf(1 - config["alpha"] / 2)
    for feature, rule in rules.items():
        data = frame.assign(group=apply_feature_rule(frame[feature], rule)).dropna(subset=["group", "borough", "activation_year", response])
        for group in rule["groups"]:
            cells = data.loc[data.group.isin([group, rule["reference"]])].groupby(["split", "borough", "activation_year", "group"])[response].agg(["mean", "var", "count"])
            contrasts = {}
            for half in ("A", "B"):
                if half not in cells.index.get_level_values("split"):
                    contrasts[half] = pd.DataFrame()
                    continue
                part = cells.loc[half].reset_index()
                a = part.loc[part.group.eq(group)].drop(columns="group")
                b = part.loc[part.group.eq(rule["reference"])].drop(columns="group")
                paired = a.merge(b, on=["borough", "activation_year"], suffixes=("_g", "_r"))
                paired = paired.loc[paired.count_g.ge(config["minimum_stratum_size"]) & paired.count_r.ge(config["minimum_stratum_size"])].copy()
                paired["effect"] = paired.mean_g - paired.mean_r
                paired["variance"] = paired.var_g / paired.count_g + paired.var_r / paired.count_r
                contrasts[half] = paired.set_index(["borough", "activation_year"])
            common = contrasts["A"].index.intersection(contrasts["B"].index)
            result = {"feature": feature, "group": group, "reference": rule["reference"], "strata": len(common)}
            if len(common):
                a = contrasts["A"].loc[common]
                weights = a.count_g * a.count_r / (a.count_g + a.count_r)
                weights = weights / weights.sum()
                for half in ("A", "B"):
                    part = contrasts[half].loc[common]
                    estimate = float((weights * part.effect).sum())
                    se = float(np.sqrt((weights.pow(2) * part.variance).sum()))
                    n_group, n_reference = int(part.count_g.sum()), int(part.count_r.sum())
                    supported = min(n_group, n_reference) >= config["minimum_group_size"]
                    p = float(2 * norm.sf(abs(estimate / se))) if se > 0 else (1.0 if estimate == 0 else 0.0)
                    result.update({f"{half}_difference_pp": estimate, f"{half}_ci_low": estimate - z * se,
                                   f"{half}_ci_high": estimate + z * se, f"{half}_p": p if supported else np.nan,
                                   f"{half}_n_group": n_group, f"{half}_n_reference": n_reference})
                    details.append(part.reset_index().assign(feature=feature, group=group, reference=rule["reference"], split=half, weight=weights.to_numpy()))
            results.append(result)
    results = pd.DataFrame(results)
    if not results.empty:
        p = results.get("B_p", pd.Series(np.nan, index=results.index)).fillna(1)
        order = p.sort_values(kind="stable").index
        adjusted = np.maximum.accumulate(p.loc[order].to_numpy() * np.arange(len(p), 0, -1)).clip(0, 1)
        results.loc[order, "B_p_holm"] = adjusted
        a = results.get("A_difference_pp", pd.Series(np.nan, index=results.index))
        b = results.get("B_difference_pp", pd.Series(np.nan, index=results.index))
        results["same_direction"] = (a * b).gt(0)
        results["replicated"] = results.same_direction & results.B_p_holm.lt(config["alpha"])
    return results, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def cluster_feature_profiles(frame, rules):
    """Within-cluster distributions, using exactly the A-defined bins in both halves."""
    parts = []
    for feature, rule in rules.items():
        values = frame.assign(group=apply_feature_rule(frame[feature], rule).fillna("Missing"))
        counts = values.groupby(["split", "cluster", "group"]).size().rename("cameras").reset_index()
        counts["share"] = counts.cameras / counts.groupby(["split", "cluster"]).cameras.transform("sum")
        parts.append(counts.assign(feature=feature))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
