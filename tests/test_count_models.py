"""Tests for the count models in model/."""

import numpy as np
import pandas as pd

from model.count_models import NegativeBinomialModel


def _simulated(n=4000, seed=0):
    """Site-year crash counts where x raises the rate and 2021 is 1.5x 2020."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    year = rng.choice([2020, 2021], size=n)
    mu = np.exp(1 + 0.7 * x + 0.4 * (year == 2021))
    y = rng.negative_binomial(2, 2 / (2 + mu))  # NB2 with alpha = 0.5
    X = pd.DataFrame({"x": x, "year": year, "site_id": np.arange(n) % 500})
    return X, pd.Series(y)


def test_nb_recovers_coefficients_and_year_effect():
    X, y = _simulated()

    model = NegativeBinomialModel(["x"]).fit(X, y, groups=X["site_id"])
    coef = model.coefficients()["coef"]

    assert abs(coef["x"] - 0.7) < 0.1
    assert abs(coef["year_2021"] - 0.4) < 0.15
    assert np.isnan(model.coefficients().loc["alpha", "ratio"])


def test_unseen_years_get_the_latest_training_year_effect():
    X, y = _simulated()
    model = NegativeBinomialModel(["x"]).fit(X, y)

    pred = model.predict(pd.DataFrame({"x": [0.0, 0.0], "year": [2021, 2024]}))

    assert pred.iloc[0] == pred.iloc[1]
