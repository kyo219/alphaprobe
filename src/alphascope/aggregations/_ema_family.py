"""Exponential moving average family: EMA, DEMA, TEMA, WMA, EWMSTD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphascope.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("EMA")
class ExponentialMovingAverage(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.ewm(span=window, min_periods=window).mean()


@register_aggregation("DEMA")
class DoubleExponentialMovingAverage(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        ema1 = series.ewm(span=window, min_periods=window).mean()
        ema2 = ema1.ewm(span=window, min_periods=window).mean()
        return 2 * ema1 - ema2


@register_aggregation("TEMA")
class TripleExponentialMovingAverage(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        ema1 = series.ewm(span=window, min_periods=window).mean()
        ema2 = ema1.ewm(span=window, min_periods=window).mean()
        ema3 = ema2.ewm(span=window, min_periods=window).mean()
        return 3 * ema1 - 3 * ema2 + ema3


@register_aggregation("WMA")
class WeightedMovingAverage(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        weights = np.arange(1, window + 1, dtype=np.float64)

        def _wma(w: np.ndarray) -> float:
            return float(np.average(w, weights=weights))

        return series.rolling(window=window, min_periods=window).apply(_wma, raw=True)


@register_aggregation("EWMSTD")
class ExponentialWeightedStd(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.ewm(span=window, min_periods=window).std()
