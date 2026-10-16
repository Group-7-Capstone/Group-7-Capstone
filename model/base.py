"""The interface every model in this layer follows, so validation/ can evaluate any of them."""

from typing import Protocol

import pandas as pd


class CountModel(Protocol):
    def fit(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> "CountModel":
        """Fit on feature rows `X` and counts `y`; `groups` marks repeated units."""
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return the expected count for each row of `X`."""
        ...
