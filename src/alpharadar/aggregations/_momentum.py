"""Momentum aggregations: MOM, ROC, MEANREV, TRENDSIG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alpharadar.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("MOM")
class Momentum(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series - series.shift(window)


@register_aggregation("ROC")
class RateOfChange(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return (series / series.shift(window) - 1) * 100


@register_aggregation("MEANREV")
class MeanReversion(Aggregation):
    """Negative autocorrelation of deviations within rolling window."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        def _meanrev(w: np.ndarray) -> float:
            dev = w - np.mean(w)
            if len(dev) < 2 or np.std(dev) == 0:
                return np.nan
            return float(np.corrcoef(dev[:-1], dev[1:])[0, 1])

        return series.rolling(window=window, min_periods=window).apply(_meanrev, raw=True)


@register_aggregation("TRENDSIG")
class TrendSignal(Aggregation):
    """T-statistic of rolling linear regression slope."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        def _t_stat(w: np.ndarray) -> float:
            n = len(w)
            x = np.arange(n, dtype=np.float64)
            x_mean = (n - 1) / 2.0
            ss_x = np.sum((x - x_mean) ** 2)
            if ss_x == 0:
                return np.nan
            slope = np.sum((x - x_mean) * (w - np.mean(w))) / ss_x
            y_hat = x_mean + slope * (x - x_mean) + (np.mean(w) - slope * x_mean)
            # Simpler: y_hat = slope * x + (mean(w) - slope * mean(x))
            residuals = w - (slope * x + (np.mean(w) - slope * x_mean))
            if n <= 2:
                return np.nan
            ss_res = np.sum(residuals ** 2)
            if ss_res == 0:
                return float(np.inf) if slope > 0 else float(-np.inf) if slope < 0 else 0.0
            se = np.sqrt(ss_res / (n - 2) / ss_x)
            return float(slope / se)

        return series.rolling(window=window, min_periods=window).apply(_t_stat, raw=True)
