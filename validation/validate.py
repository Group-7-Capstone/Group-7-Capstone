"""Temporal validation of any count model (see `model/base.py`) on the site x year panel.

For each test year T the model is fit on all rows with year < T and scored on
year T (an expanding window).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from model.base import CountModel


@dataclass(frozen=True)
class ValidationConfig:
    # Test years must be on or after the site build cutoff (SiteConfig.build_end).
    test_years: tuple[int, ...] = (2024, 2025)
    top_k: tuple[int, ...] = (50, 100, 250, 500, 1000)
    # Calibration bins as quantiles of predicted risk. They narrow toward the top,
    # where watch lists are drawn, so over-predicted top sites can't hide inside a
    # bin dominated by thousands of moderate-risk sites. The last bin is the top 0.1%.
    calibration_quantiles: tuple[float, ...] = (
        0.0,
        0.5,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.999,
        1.0,
    )


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


def top_k_totals(y: pd.Series, pred: pd.Series, k: int) -> tuple[float, float]:
    """(predicted, observed) totals over the `k` highest-`pred` rows.

    Ties at the cutoff are handled as in `top_k_capture`; the predicted total
    needs no tie rule because tied rows share the same prediction.
    """
    predicted = float(np.sort(np.asarray(pred, dtype=float))[::-1][:k].sum())
    return predicted, top_k_capture(y, pred, k) * float(np.sum(y))


def poisson_interval(count, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Exact (Garwood) interval for a Poisson count; works elementwise on arrays."""
    count = np.asarray(count, dtype=float)
    alpha = 1 - level
    low = np.where(count > 0, chi2.ppf(alpha / 2, 2 * count) / 2, 0.0)
    high = chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2
    return low, high


def calibration_table(
    y: pd.Series, pred: pd.Series, quantiles: tuple[float, ...]
) -> pd.DataFrame:
    """Predicted vs observed counts within bins of predicted-risk quantiles.

    Bin b holds the rows ranked between quantiles[b] and quantiles[b+1] of
    predicted risk. `obs_low`/`obs_high` are a 95% Poisson interval on the
    bin's observed mean, so small top bins show how uncertain they are.
    """
    pred = np.asarray(pred, dtype=float)
    pct = (pd.Series(pred).rank(method="first").to_numpy() - 1) / len(pred)
    bins = np.searchsorted(np.asarray(quantiles[1:-1]), pct, side="right")
    df = pd.DataFrame({"bin": bins, "pred": pred, "obs": np.asarray(y, dtype=float)})
    table = df.groupby("bin").agg(
        n=("obs", "size"), pred_total=("pred", "sum"), obs_total=("obs", "sum")
    )
    low, high = poisson_interval(table["obs_total"])
    return table.assign(
        q_low=[quantiles[b] for b in table.index],
        q_high=[quantiles[b + 1] for b in table.index],
        mean_pred=table["pred_total"] / table["n"],
        mean_obs=table["obs_total"] / table["n"],
        obs_low=low / table["n"],
        obs_high=high / table["n"],
    )


def evaluate(
    make_model: Callable[[], CountModel],
    panel: pd.DataFrame,
    config: ValidationConfig,
    y_col: str = "ksi",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a fresh model per test year and score it.

    Returns (capture, calibration, predictions): `capture` has one row per test
    year and k with the top-k capture; `calibration` is `calibration_table` per
    test year; `predictions` holds each test row's prediction.
    """
    capture, calibration, predictions = [], [], []
    for year, train, test in expanding_window_splits(panel, config.test_years):
        model = make_model().fit(train, train[y_col], groups=train["site_id"])
        y, pred = test[y_col], model.predict(test)
        predictions.append(test[["site_id", "year", y_col]].assign(pred=pred))
        capture += [
            {"test_year": year, "k": k, "capture": top_k_capture(y, pred, k)}
            for k in config.top_k
        ]
        calibration.append(
            calibration_table(y, pred, config.calibration_quantiles).assign(
                test_year=year
            )
        )
    return (
        pd.DataFrame(capture),
        pd.concat(calibration).reset_index(),
        pd.concat(predictions),
    )
