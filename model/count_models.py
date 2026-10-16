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
    """NB2 regression (log link) on `features`, optionally with year fixed effects.

    Years not seen in training (e.g. a held-out test year) get the fixed effect
    of the latest training year. With `groups`, standard errors are clustered
    on them, since each site appears in several years.
    """

    def __init__(self, features: list[str], year_col: str | None = "year"):
        self.features = list(features)
        self.year_col = year_col

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        design = X[self.features].astype(float)
        if self.year_col is not None:
            years = X[self.year_col].clip(upper=self.years_[-1])
            for year in self.years_[1:]:
                design[f"year_{year}"] = (years == year).astype(float)
        return sm.add_constant(design, has_constant="add")

    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> "NegativeBinomialModel":
        if self.year_col is not None:
            self.years_ = sorted(X[self.year_col].unique())
        fit_kwargs = {"disp": 0, "maxiter": 200}
        if groups is not None:
            fit_kwargs |= {
                "cov_type": "cluster",
                "cov_kwds": {"groups": np.asarray(groups)},
            }
        self.result_ = sm.NegativeBinomial(
            np.asarray(y, dtype=float), self._design(X)
        ).fit(**fit_kwargs)
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
                "rate_ratio": np.exp(self.result_.params),
                "ci_low": np.exp(ci[0]),
                "ci_high": np.exp(ci[1]),
                "p_value": self.result_.pvalues,
            }
        )
        table.loc["alpha", ["rate_ratio", "ci_low", "ci_high"]] = np.nan
        return table
