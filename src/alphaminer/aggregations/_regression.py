"""Regression aggregations: LSLOPE, LR2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphaminer.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("LSLOPE")
class LinearSlope(Aggregation):
    """Vectorised rolling linear regression slope using rolling sums."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        import pandas as _pd
        y = series
        x = _pd.Series(np.arange(len(y), dtype=np.float64), index=y.index)
        r = y.rolling(window=window, min_periods=window)
        rx = x.rolling(window=window, min_periods=window)
        sum_x = rx.sum()
        sum_y = r.sum()
        sum_xy = (x * y).rolling(window=window, min_periods=window).sum()
        sum_x2 = (x ** 2).rolling(window=window, min_periods=window).sum()
        n = window
        return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)


@register_aggregation("LR2")
class LinearR2(Aggregation):
    """Rolling R-squared of linear regression."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        def _r2(w: np.ndarray) -> float:
            n = len(w)
            x = np.arange(n, dtype=np.float64)
            ss_x = np.sum((x - x.mean()) ** 2)
            if ss_x == 0:
                return np.nan
            slope = np.sum((x - x.mean()) * (w - w.mean())) / ss_x
            intercept = w.mean() - slope * x.mean()
            y_hat = slope * x + intercept
            ss_res = np.sum((w - y_hat) ** 2)
            ss_tot = np.sum((w - w.mean()) ** 2)
            if ss_tot == 0:
                return np.nan
            return float(1 - ss_res / ss_tot)

        return series.rolling(window=window, min_periods=window).apply(_r2, raw=True)
