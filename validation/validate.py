"""Temporal validation of any count model (see `model/base.py`) on the site x year panel.

For each test year T the model is fit on all rows with year < T and scored on
year T (an expanding window). Metrics are reported for all sites and for
non-highway sites, where cameras can actually be placed.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.base import CountModel


@dataclass(frozen=True)
class ValidationConfig:
    # Test years must be on or after the site build cutoff (SiteConfig.build_end).
    test_years: tuple[int, ...] = (2024, 2025)
    top_k: tuple[int, ...] = (50, 100, 250, 500, 1000)
    # Sites with a larger share of highway crashes are dropped from the "non_highway" scope.
    highway_threshold: float = 0.5
    n_calibration_bins: int = 10


def expanding_window_splits(
    panel: pd.DataFrame, test_years: tuple[int, ...]
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (test year, rows before it, rows in it) for each test year."""
    for year in test_years:
        yield year, panel[panel["year"] < year], panel[panel["year"] == year]


def top_k_capture(y: pd.Series, pred: pd.Series, k: int) -> float:
    """Share of the total of `y` that falls in the `k` highest-`pred` rows.

    Rows tied with the k-th prediction are credited proportionally (the expected
    capture under a random tie-break), so ties don't depend on row order.
    """
    groups = (
        pd.DataFrame({"y": np.asarray(y), "pred": np.asarray(pred)})
        .groupby("pred")["y"]
        .agg(["size", "sum"])
        .sort_index(ascending=False)
    )
    n_before = groups["size"].cumsum() - groups["size"]
    take = ((k - n_before) / groups["size"]).clip(0, 1)
    captured = (take * groups["sum"]).sum()
    total = groups["sum"].sum()
    return float(captured / total) if total else np.nan


def calibration_table(y: pd.Series, pred: pd.Series, n_bins: int) -> pd.DataFrame:
    """Mean predicted vs observed count within equal-size bins of predicted risk."""
    rank = pd.Series(np.asarray(pred)).rank(method="first").to_numpy() - 1
    bins = (rank * n_bins // len(rank)).astype(int) + 1
    df = pd.DataFrame({"bin": bins, "pred": np.asarray(pred), "obs": np.asarray(y)})
    return df.groupby("bin").agg(
        n=("obs", "size"), mean_pred=("pred", "mean"), mean_obs=("obs", "mean")
    )


def mean_poisson_deviance(y: pd.Series, pred: pd.Series, eps: float = 1e-6) -> float:
    """Mean Poisson deviance; predictions are floored at `eps` so zero-risk rows score."""
    y, mu = (
        np.asarray(y, dtype=float),
        np.clip(np.asarray(pred, dtype=float), eps, None),
    )
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def _scopes(test: pd.DataFrame, config: ValidationConfig) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=test.index),
        "non_highway": test["highway_share"] <= config.highway_threshold,
    }


def evaluate(
    make_model: Callable[[], CountModel],
    panel: pd.DataFrame,
    config: ValidationConfig,
    y_col: str = "ksi",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a fresh model per test year and score it.

    Returns (metrics, calibration, predictions). Metrics are long-format rows of
    test_year, scope, metric, k, value.
    """
    metrics, calibration, predictions = [], [], []
    for year, train, test in expanding_window_splits(panel, config.test_years):
        model = make_model().fit(train, train[y_col], groups=train["site_id"])
        pred = model.predict(test)
        predictions.append(test[["site_id", "year", y_col]].assign(pred=pred))
        for scope, mask in _scopes(test, config).items():
            y, p = test.loc[mask, y_col], pred[mask]
            row = {"test_year": year, "scope": scope}
            metrics += [
                {
                    **row,
                    "metric": "top_k_capture",
                    "k": k,
                    "value": top_k_capture(y, p, k),
                }
                for k in config.top_k
            ]
            metrics += [
                {
                    **row,
                    "metric": "poisson_deviance",
                    "k": None,
                    "value": mean_poisson_deviance(y, p),
                },
                {
                    **row,
                    "metric": "pred_to_obs_total",
                    "k": None,
                    "value": p.sum() / y.sum(),
                },
            ]
            calibration.append(
                calibration_table(y, p, config.n_calibration_bins).assign(
                    test_year=year, scope=scope
                )
            )
    return (
        pd.DataFrame(metrics),
        pd.concat(calibration).reset_index(),
        pd.concat(predictions),
    )
