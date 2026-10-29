import numpy as np
import pandas as pd
import pytest
from validation.camera_trend_walkthrough import compare_trend_tables, verify_fixed_cohort_arithmetic


def test_output_comparison_rejects_missing_or_duplicated_keys():
    reference = pd.DataFrame({"week": [1, 2], "mean": [0., np.nan]})
    with pytest.raises(AssertionError, match="key sets"):
        compare_trend_tables(reference.iloc[:1], reference, ["week"], ["mean"])
    with pytest.raises(AssertionError, match="Duplicate"):
        compare_trend_tables(pd.concat([reference, reference.iloc[:1]]), reference, ["week"], ["mean"])


def test_output_comparison_distinguishes_zero_from_missing_and_tolerates_serialization():
    reference = pd.DataFrame({"week": [1, 2], "mean": [1 / 3, np.nan]})
    serialized = pd.DataFrame({"week": [2, 1], "mean": [np.nan, .3333333333]})
    assert compare_trend_tables(serialized, reference, ["week"], ["mean"]).passed.all()
    with pytest.raises(AssertionError, match="values differ"):
        compare_trend_tables(reference.fillna(0), reference, ["week"], ["mean"])


def test_arithmetic_check_uses_camera_specific_baselines_and_denominators():
    matrix = pd.DataFrame([[100., 80.], [10., 5.], [0., 3.]], columns=[1, 2])
    indexed = pd.DataFrame([[100., 80.], [100., 50.], [np.nan, np.nan]], columns=[1, 2])
    stats = pd.DataFrame({"time": [1, 2], "n": [3, 3], "index_n": [2, 2],
                          "raw_mean": [110 / 3, 88 / 3], "index_mean": [100., 65.]})
    assert verify_fixed_cohort_arithmetic(matrix, indexed, stats, 1)["camera_weeks_checked"] == 6
    stats.loc[1, "index_mean"] = 100 * 88 / 110
    with pytest.raises(AssertionError):
        verify_fixed_cohort_arithmetic(matrix, indexed, stats, 1)
