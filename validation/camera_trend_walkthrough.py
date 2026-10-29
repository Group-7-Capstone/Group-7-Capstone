"""Independent arithmetic and saved-output checks for the teaching notebook."""
import numpy as np
import pandas as pd


def compare_trend_tables(actual, expected, keys, columns, rtol=1e-9, atol=1e-8):
    """Require the same keys, missingness, and values; never silently inner-join."""
    for frame in (actual, expected):
        if frame.duplicated(keys).any():
            raise AssertionError("Duplicate trend keys")
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise AssertionError("Trend key sets differ")
    checks = []
    for column in columns:
        a = pd.to_numeric(left[column]).to_numpy(dtype=float)
        b = pd.to_numeric(right[column]).to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            raise AssertionError(f"Trend values differ: {column}")
        delta = np.abs(a - b)
        checks.append({"field": column, "rows_checked": len(a),
                       "maximum_absolute_difference": float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.,
                       "passed": True})
    return pd.DataFrame(checks)


def verify_fixed_cohort_arithmetic(matrix, indexed, stats, baseline_weeks=13):
    """Check the production summaries against explicit NumPy numerators."""
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise AssertionError("Fixed cohort requires complete nonnegative observations")
    baseline = values[:, :baseline_weeks].sum(axis=1) / baseline_weeks
    valid = baseline > 0
    indices = np.divide(100 * values, baseline[:, None],
                        out=np.full_like(values, np.nan), where=valid[:, None])
    np.testing.assert_allclose(indexed.to_numpy(), indices, equal_nan=True)
    ordered = stats.sort_values("time")
    np.testing.assert_array_equal(ordered.time.to_numpy(), matrix.columns.to_numpy())
    np.testing.assert_array_equal(ordered.n.to_numpy(), np.full(values.shape[1], len(values)))
    np.testing.assert_allclose(ordered.raw_mean, values.sum(axis=0) / len(values))
    np.testing.assert_allclose(ordered.index_mean, indices[valid].sum(axis=0) / valid.sum())
    np.testing.assert_array_equal(ordered.index_n.to_numpy(), np.full(values.shape[1], valid.sum()))
    return {"groups": len(values), "weeks": values.shape[1], "camera_weeks_checked": int(values.size),
            "indexed_groups": int(valid.sum()), "passed": True}
