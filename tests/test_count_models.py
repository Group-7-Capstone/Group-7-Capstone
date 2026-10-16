"""Tests for the count models in model/."""

import numpy as np
import pandas as pd

from model.count_models import HistoryRanker, NegativeBinomialModel


def _simulated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    year = rng.choice([2020, 2021], size=n)
    mu = np.exp(-1 + 0.7 * x + 0.4 * (year == 2021))
    y = rng.negative_binomial(2, 2 / (2 + mu))  # NB2 with alpha = 0.5
    return pd.DataFrame(
        {"x": x, "year": year, "site_id": np.arange(n) % 500}
    ), pd.Series(y)


def test_nb_recovers_coefficients_and_year_effect():
    X, y = _simulated()

    model = NegativeBinomialModel(["x"]).fit(X, y, groups=X["site_id"])
    coef = model.coefficients()["coef"]

    assert abs(coef["x"] - 0.7) < 0.1
    assert abs(coef["year_2021"] - 0.4) < 0.15
    assert np.isnan(model.coefficients().loc["alpha", "rate_ratio"])


def test_nb_scores_unseen_years_with_latest_training_year():
    X, y = _simulated()
    model = NegativeBinomialModel(["x"]).fit(X, y)
    row = pd.DataFrame({"x": [0.0, 0.0], "year": [2021, 2024]})

    pred = model.predict(row)

    assert pred.iloc[0] == pred.iloc[1]


def test_history_ranker_is_mean_annual_history():
    X = pd.DataFrame({"ksi_lag3": [0, 3, 6]})

    pred = HistoryRanker("ksi_lag3", n_years=3).fit(X, None).predict(X)

    assert pred.tolist() == [0, 1, 2]
