"""Count models for site-level KSI risk. Fitting and prediction only; no splits or metrics."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _design(
    X: pd.DataFrame, features: list[str], years: list[int] | None
) -> pd.DataFrame:
    """Features plus a constant and, if `years` is given, year fixed effects.

    Years after the last training year get the last training year's effect.
    """
    design = X[features].astype(float)
    if years is not None:
        clipped = X["year"].clip(upper=years[-1])
        for year in years[1:]:
            design[f"year_{year}"] = (clipped == year).astype(float)
    return sm.add_constant(design, has_constant="add")


def _fit_kwargs(groups: pd.Series | None) -> dict:
    """Cluster standard errors on `groups` (each site appears in several years)."""
    if groups is None:
        return {}
    return {"cov_type": "cluster", "cov_kwds": {"groups": np.asarray(groups)}}


def _coefficient_table(result) -> pd.DataFrame:
    """Coefficients with exp(beta) ratios and 95% intervals."""
    ci = result.conf_int()
    return pd.DataFrame(
        {
            "coef": result.params,
            "ratio": np.exp(result.params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p_value": result.pvalues,
        }
    )


class HistoryRanker:
    """Baseline with nothing to fit: expected count = one past-history column / `n_years`.

    With `ksi_lag3` and n_years=3 this is "rank sites by their past KSI",
    the practice the fitted models are compared against.
    """

    def __init__(self, column: str, n_years: int = 1):
        self.column = column
        self.n_years = n_years

    def fit(self, X, y, groups=None) -> "HistoryRanker":
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.column] / self.n_years


class NegativeBinomialModel:
    """NB2 regression (log link) on `features`, optionally with year fixed effects."""

    def __init__(self, features: list[str], year_effects: bool = True):
        self.features = list(features)
        self.year_effects = year_effects

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> "NegativeBinomialModel":
        self.years_ = sorted(X["year"].unique()) if self.year_effects else None
        self.result_ = sm.NegativeBinomial(
            np.asarray(y, dtype=float), _design(X, self.features, self.years_)
        ).fit(disp=0, maxiter=200, **_fit_kwargs(groups))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        design = _design(X, self.features, self.years_)
        return pd.Series(self.result_.predict(design), index=X.index)

    def coefficients(self) -> pd.DataFrame:
        """`ratio` is the rate ratio exp(beta); the NB dispersion `alpha` has none."""
        table = _coefficient_table(self.result_)
        table.loc["alpha", ["ratio", "ci_low", "ci_high"]] = np.nan
        return table


class SeverityShareModel:
    """Binomial regression (logit link) for the share of a site's crashes that are KSI.

    Fit on site-years with at least one crash as (KSI, non-KSI) crash counts out of
    `trials_col`, so a site with 2 crashes and 1 KSI carries far less weight than
    one with 200 crashes and 100 KSI. `predict` returns the expected share.
    """

    def __init__(
        self,
        features: list[str],
        trials_col: str = "crashes",
        year_effects: bool = True,
    ):
        self.features = list(features)
        self.trials_col = trials_col
        self.year_effects = year_effects

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> "SeverityShareModel":
        has = (X[self.trials_col] > 0).to_numpy()
        X, y, trials = (
            X[has],
            np.asarray(y, dtype=float)[has],
            X.loc[has, self.trials_col],
        )
        self.years_ = sorted(X["year"].unique()) if self.year_effects else None
        successes_failures = np.column_stack([y, trials.to_numpy(dtype=float) - y])
        self.result_ = sm.GLM(
            successes_failures,
            _design(X, self.features, self.years_),
            family=sm.families.Binomial(),
        ).fit(**_fit_kwargs(None if groups is None else np.asarray(groups)[has]))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        design = _design(X, self.features, self.years_)
        return pd.Series(self.result_.predict(design), index=X.index)

    def coefficients(self) -> pd.DataFrame:
        """`ratio` is the odds ratio exp(beta)."""
        return _coefficient_table(self.result_)


class FrequencySeverityModel:
    """Expected KSI = expected crashes x expected share of crashes that are KSI.

    `frequency` is a negative binomial model of all crashes (`trials_col`);
    `severity` is a `SeverityShareModel`. Splitting the two shows whether a
    site is risky because it has many crashes or because its crashes are severe.
    """

    def __init__(self, features: list[str], trials_col: str = "crashes"):
        self.frequency = NegativeBinomialModel(features)
        self.severity = SeverityShareModel(features, trials_col)
        self.trials_col = trials_col

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> "FrequencySeverityModel":
        self.frequency.fit(X, X[self.trials_col], groups)
        self.severity.fit(X, y, groups)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.frequency.predict(X) * self.severity.predict(X)
