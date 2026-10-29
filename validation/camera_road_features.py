"""Independent reconciliation of road cohorts and chart-ready mean values."""
import numpy as np
import pandas as pd
from preprocessing.camera_road_features import GROUPS, UNKNOWN


def validate_road_analysis(summary, primary, features, trends, coverage, memberships, segments):
    checks = []
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
    check("All original camera groups retained once", features.camera_id.is_unique and set(features.camera_id) == set(summary.camera_id))
    original = summary.set_index("camera_id").loc[features.camera_id]
    check("Activation dates unchanged", np.array_equal(original.inferred_activation_date, features.inferred_activation_date))
    check("Coordinates unchanged", np.array_equal(original[['Latitude', 'Longitude']].to_numpy(), features[['Latitude', 'Longitude']].to_numpy()))
    check("Every assigned segment is ordinary roadbed B", segments.rb_layer.eq("B").all() and segments.rw_type.astype(str).str.strip().eq("1").all())
    check("Every assigned segment is within 60 metres", segments.distance_m.le(60 + 1e-8).all())
    keys = ['camera_type', 'horizon', 'policy', 'baseline_min', 'dimension', 'group']
    check("Unique group-week chart keys", not trends.duplicated(keys + ['week']).any())
    check("No camera multiplied within a dimension/cohort", not memberships.duplicated(keys[:-1] + ['camera_id']).any())
    feature_lookup = features.set_index("camera_id")
    membership_groups = memberships.groupby(keys, sort=False)
    coverage_lookup = coverage.set_index(keys)
    for key, rows in trends.groupby(keys, sort=False):
        ids = membership_groups.get_group(key).camera_id
        window = primary.loc[ids, 1:int(key[1])].to_numpy(float)
        raw = window.mean(axis=0)
        base = window[:, :13].mean(axis=1)
        valid = base > 0
        indexed = window[valid] / base[valid, None] * 100
        ordered = rows.sort_values('week')
        index_mean = indexed.mean(axis=0) if valid.any() else np.full(int(key[1]), np.nan)
        prefix = ' / '.join(map(str, key))
        check(prefix + ': complete fixed cohort', np.isfinite(window).all() and len(rows) == int(key[1]))
        check(prefix + ': mean and denominator reconcile', np.allclose(raw, ordered.raw_mean) and ordered.n.eq(len(ids)).all())
        check(prefix + ': index and denominator reconcile', np.allclose(index_mean, ordered.index_mean, equal_nan=True) and ordered.index_n.eq(valid.sum()).all())
        check(prefix + ': baseline mean is 100', not valid.any() or np.isclose(index_mean[:13].mean(), 100))
        check(prefix + ': coverage table matches', coverage_lookup.loc[key, 'cohort_n'] == len(ids))
        if key[-1] != UNKNOWN:
            flag = 'strict_match' if key[2] == 'intersection_only' else 'any_match'
            check(prefix + ': road class backed by eligible match', feature_lookup.loc[ids, flag].all() and feature_lookup.loc[ids, key[4] + '_group'].eq(key[-1]).all())
    # All fixed-history groups enter exactly one category, including unknown.
    for key, rows in coverage.groupby(keys[:4], sort=False):
        mask = feature_lookup.reindex(primary.index).camera_type.eq(key[0]) & primary.loc[:, 1:int(key[1])].notna().all(axis=1)
        if key[3]:
            mask &= primary.loc[:, 1:13].mean(axis=1).ge(key[3])
        for dimension in GROUPS:
            check(f'{key} / {dimension}: known + unknown = full cohort', rows[rows.dimension.eq(dimension)].cohort_n.sum() == mask.sum())
    return pd.DataFrame(checks)
