"""Count models for site-level KSI risk. Fitting and prediction only; no splits or metrics."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    """NB2 regression (log link) on `features` plus year fixed effects.

    Years after the last training year (e.g. a held-out test year) get the last
    training year's effect. With `groups`, standard errors are clustered on them,
    since each site appears in several years.
    """

    def __init__(self, features: list[str]):
        self.features = list(features)

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        design = X[self.features].astype(float)
        clipped = X["year"].clip(upper=self.years_[-1])
        for year in self.years_[1:]:
            design[f"year_{year}"] = (clipped == year).astype(float)
        return sm.add_constant(design, has_constant="add")

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> "NegativeBinomialModel":
        self.years_ = sorted(X["year"].unique())
        cluster = (
            {}
            if groups is None
            else {"cov_type": "cluster", "cov_kwds": {"groups": np.asarray(groups)}}
        )
        self.result_ = sm.NegativeBinomial(
            np.asarray(y, dtype=float), self._design(X)
        ).fit(disp=0, maxiter=200, **cluster)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.result_.predict(self._design(X)), index=X.index)

    def coefficients(self) -> pd.DataFrame:
        """Coefficients with rate ratios exp(beta) and 95% intervals.

        The last row is the NB dispersion `alpha`, which has no rate ratio.
        """
        ci = self.result_.conf_int()
        table = pd.DataFrame(
            {
                "coef": self.result_.params,
                "ratio": np.exp(self.result_.params),
                "ci_low": np.exp(ci[0]),
                "ci_high": np.exp(ci[1]),
                "p_value": self.result_.pvalues,
            }
        )
        table.loc["alpha", ["ratio", "ci_low", "ci_high"]] = np.nan
        return table
