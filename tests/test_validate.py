"""Tests for temporal validation metrics in validation/."""

import numpy as np
import pandas as pd

from model.count_models import HistoryRanker
from validation.validate import (
    ValidationConfig,
    calibration_table,
    evaluate,
    expanding_window_splits,
    mean_poisson_deviance,
    poisson_interval,
    top_k_capture,
    top_k_totals,
)


def test_top_k_capture_counts_highest_predictions():
    y = pd.Series([5, 3, 0, 2])
    pred = pd.Series([0.9, 0.1, 0.5, 0.2])

    assert top_k_capture(y, pred, 1) == 0.5
    assert top_k_capture(y, pred, 3) == 0.7
    assert top_k_capture(y, pred, 10) == 1.0


def test_top_k_capture_splits_ties_at_the_cutoff():
    y = pd.Series([4, 2, 2])
    pred = pd.Series([1.0, 1.0, 0.0])

    assert top_k_capture(y, pred, 1) == 0.375  # half of the tied pair's 6, out of 8


def test_mean_poisson_deviance_is_zero_for_perfect_predictions():
    y = pd.Series([0, 1, 3])

    assert mean_poisson_deviance(y, y.astype(float)) < 1e-5
    assert mean_poisson_deviance(y, pd.Series([1.0, 1.0, 1.0])) > 0


def test_calibration_table_bins_by_prediction():
    y = pd.Series([0, 0, 1, 1])
    pred = pd.Series([0.1, 0.2, 0.8, 0.9])

    table = calibration_table(y, pred, (0.0, 0.5, 1.0))

    assert table["mean_obs"].tolist() == [0, 1]
    assert np.allclose(table["mean_pred"], [0.15, 0.85])
    assert table["q_high"].tolist() == [0.5, 1.0]
    assert (table["obs_low"] <= table["mean_obs"]).all()
    assert (table["obs_high"] >= table["mean_obs"]).all()


def test_calibration_table_uneven_quantiles_isolate_the_top():
    pred = pd.Series(np.arange(1000, dtype=float))
    y = pd.Series(np.zeros(1000))

    table = calibration_table(y, pred, (0.0, 0.9, 0.99, 1.0))

    assert table["n"].tolist() == [900, 90, 10]
    assert table["mean_pred"].iloc[-1] == 994.5


def test_top_k_totals_compares_predicted_and_observed():
    y = pd.Series([1, 0, 0, 2])
    pred = pd.Series([3.0, 2.0, 1.0, 0.5])

    assert top_k_totals(y, pred, 2) == (5.0, 1.0)


def test_poisson_interval_known_values():
    low, high = poisson_interval([0, 10])

    assert low[0] == 0 and abs(high[0] - 3.689) < 1e-3
    assert abs(low[1] - 4.795) < 1e-3 and abs(high[1] - 18.390) < 1e-3


def _panel():
    return pd.DataFrame(
        {
            "site_id": [0, 1, 0, 1, 0, 1],
            "year": [2023, 2023, 2024, 2024, 2025, 2025],
            "ksi": [1, 0, 2, 0, 1, 1],
            "ksi_lag3": [3, 0, 3, 0, 4, 0],
            "highway_share": [0.0, 0.9, 0.0, 0.9, 0.0, 0.9],
        }
    )


def test_expanding_window_never_trains_on_the_test_year():
    for year, train, test in expanding_window_splits(_panel(), (2024, 2025)):
        assert train["year"].max() < year
        assert (test["year"] == year).all()


def test_evaluate_reports_each_year_and_scope():
    config = ValidationConfig(test_years=(2024, 2025), top_k=(1,))

    metrics, calibration, predictions = evaluate(
        lambda: HistoryRanker("ksi_lag3", 3), _panel(), config
    )

    capture = metrics[metrics["metric"] == "top_k_capture"].set_index(
        ["test_year", "scope"]
    )
    assert capture.loc[(2024, "all"), "value"] == 1.0
    assert capture.loc[(2025, "all"), "value"] == 0.5
    assert capture.loc[(2025, "non_highway"), "value"] == 1.0
    assert len(predictions) == 4
    assert set(calibration["scope"]) == {"all", "non_highway"}
