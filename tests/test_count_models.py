"""Tests for the count models in model/."""

import numpy as np
import pandas as pd

from model.count_models import (
    FrequencySeverityModel,
    NegativeBinomialModel,
    SeverityShareModel,
)


def _simulated(n=4000, seed=0):
    """Site-years where x raises crash counts and z raises the KSI share."""
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    year = rng.choice([2020, 2021], size=n)
    mu = np.exp(1 + 0.7 * x + 0.4 * (year == 2021))
    crashes = rng.negative_binomial(2, 2 / (2 + mu))  # NB2 with alpha = 0.5
    share = 1 / (1 + np.exp(-(-2 + 0.8 * z)))
    ksi = rng.binomial(crashes, share)
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "year": year,
            "crashes": crashes,
            "site_id": np.arange(n) % 500,
        }
    )
    return X, pd.Series(ksi)


def test_nb_recovers_coefficients_and_year_effect():
    X, _ = _simulated()

    model = NegativeBinomialModel(["x"]).fit(X, X["crashes"], groups=X["site_id"])
    coef = model.coefficients()["coef"]

    assert abs(coef["x"] - 0.7) < 0.1
    assert abs(coef["year_2021"] - 0.4) < 0.15
    assert np.isnan(model.coefficients().loc["alpha", "ratio"])


def test_unseen_years_get_the_latest_training_year_effect():
    X, _ = _simulated()
    model = NegativeBinomialModel(["x"]).fit(X, X["crashes"])

    pred = model.predict(pd.DataFrame({"x": [0.0, 0.0], "year": [2021, 2024]}))

    assert pred.iloc[0] == pred.iloc[1]


def test_severity_share_recovers_logit_coefficient():
    X, ksi = _simulated()

    model = SeverityShareModel(["z"], year_effects=False).fit(
        X, ksi, groups=X["site_id"]
    )

    assert abs(model.coefficients().loc["z", "coef"] - 0.8) < 0.15


def test_frequency_severity_predicts_crashes_times_share():
    X, ksi = _simulated()

    model = FrequencySeverityModel(["x", "z"]).fit(X, ksi)
    pred = model.predict(X)

    expected = model.frequency.predict(X) * model.severity.predict(X)
    assert np.allclose(pred, expected)
    assert abs(pred.sum() / ksi.sum() - 1) < 0.1
