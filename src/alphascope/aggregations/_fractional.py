"""Fractional and quantile aggregations: FRACDIFF, QUANTILE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphascope.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


def _fracdiff_coeffs(d: float, window: int) -> np.ndarray:
    """Compute fractional differencing weights."""
    coeffs = np.zeros(window)
    coeffs[0] = 1.0
    for k in range(1, window):
        coeffs[k] = coeffs[k - 1] * (d - k + 1) / k
    return coeffs


@register_aggregation("FRACDIFF")
class FractionalDifference(Aggregation):
    """Fractional differencing. extra = d * 10 (e.g. extra=3 means d=0.3)."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("FRACDIFF requires extra parameter (d*10). Use FRACDIFF_d10_WINDOW format (e.g. FRACDIFF_3_50 for d=0.3).")
        d = extra / 10.0
        coeffs = _fracdiff_coeffs(d, window)

        def _apply_fracdiff(w: np.ndarray) -> float:
            return float(np.dot(coeffs, w[::-1]))

        return series.rolling(window=window, min_periods=window).apply(
            _apply_fracdiff, raw=True
        )


@register_aggregation("QUANTILE")
class Quantile(Aggregation):
    """Rolling quantile. extra = percentile (0-100)."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("QUANTILE requires extra parameter (percentile 0-100). Use QUANTILE_q_WINDOW format (e.g. QUANTILE_25_50).")
        q = extra / 100.0
        return series.rolling(window=window, min_periods=window).quantile(q)
