"""Rank and normalisation aggregations: RANK, ZSCORE, CV, NORMDEV."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from alphaminer.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("RANK")
class Rank(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).apply(
            lambda w: rankdata(w)[-1] / len(w), raw=True
        )


@register_aggregation("ZSCORE")
class ZScore(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        r = series.rolling(window=window, min_periods=window)
        return (series - r.mean()) / r.std()


@register_aggregation("CV")
class CoefficientOfVariation(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        r = series.rolling(window=window, min_periods=window)
        return r.std() / r.mean()


@register_aggregation("NORMDEV")
class NormalityDeviation(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        r = series.rolling(window=window, min_periods=window)
        return np.abs(r.skew()) + np.abs(r.kurt() - 3)
