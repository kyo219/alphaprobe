"""Correlation-based aggregations: ACF, PACF, MI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alpharadar.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


def _acf_at_lag(w: np.ndarray, lag: int) -> float:
    """Autocorrelation at a specific lag."""
    n = len(w)
    if lag >= n:
        return np.nan
    mean = np.mean(w)
    var = np.var(w)
    if var == 0:
        return np.nan
    cov = np.mean((w[: n - lag] - mean) * (w[lag:] - mean))
    return float(cov / var)


def _pacf_at_lag(w: np.ndarray, lag: int) -> float:
    """Partial autocorrelation at a specific lag via Durbin-Levinson."""
    n = len(w)
    if lag >= n or lag < 1:
        return np.nan

    # Compute ACF values needed
    acf_vals = []
    mean = np.mean(w)
    var = np.var(w)
    if var == 0:
        return np.nan
    for k in range(lag + 1):
        if k == 0:
            acf_vals.append(1.0)
        else:
            cov = np.mean((w[: n - k] - mean) * (w[k:] - mean))
            acf_vals.append(cov / var)

    # Durbin-Levinson recursion
    phi = np.zeros((lag + 1, lag + 1))
    phi[1, 1] = acf_vals[1]

    for k in range(2, lag + 1):
        num = acf_vals[k] - sum(phi[k - 1, j] * acf_vals[k - j] for j in range(1, k))
        den = 1 - sum(phi[k - 1, j] * acf_vals[j] for j in range(1, k))
        if den == 0:
            return np.nan
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return float(phi[lag, lag])


def _mutual_information(w_x: np.ndarray, w_y: np.ndarray, n_bins: int) -> float:
    """Mutual information between two arrays using histogram binning."""
    n = len(w_x)
    if n < 2:
        return np.nan

    # Joint histogram
    hist_xy, _, _ = np.histogram2d(w_x, w_y, bins=n_bins)
    p_xy = hist_xy / n
    p_xy = p_xy[p_xy > 0]

    # Marginals
    hist_x, _ = np.histogram(w_x, bins=n_bins)
    hist_y, _ = np.histogram(w_y, bins=n_bins)
    p_x = hist_x / n
    p_y = hist_y / n

    h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
    h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
    h_xy = -np.sum(p_xy * np.log2(p_xy))

    return float(h_x + h_y - h_xy)


@register_aggregation("ACF")
class AutoCorrelation(Aggregation):
    """Rolling autocorrelation at a specific lag. extra = lag."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("ACF requires extra parameter (lag). Use ACF_lag_WINDOW format.")
        lag = extra

        def _acf_window(w: np.ndarray) -> float:
            return _acf_at_lag(w, lag)

        return series.rolling(window=window, min_periods=window).apply(
            _acf_window, raw=True
        )


@register_aggregation("PACF")
class PartialAutoCorrelation(Aggregation):
    """Rolling partial autocorrelation at a specific lag. extra = lag."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("PACF requires extra parameter (lag). Use PACF_lag_WINDOW format.")
        lag = extra

        def _pacf_window(w: np.ndarray) -> float:
            return _pacf_at_lag(w, lag)

        return series.rolling(window=window, min_periods=window).apply(
            _pacf_window, raw=True
        )


@register_aggregation("MI")
class MutualInformation(Aggregation):
    """Rolling mutual information between feature and target. extra = number of bins."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("MI requires extra parameter (number of bins). Use MI_bins_WINDOW format.")
        if target is None:
            raise ValueError("MI (Mutual Information) requires a target series.")
        n_bins = extra
        import pandas as _pd

        result = _pd.Series(np.nan, index=series.index)
        for i in range(window - 1, len(series)):
            s_window = series.iloc[i - window + 1 : i + 1].values
            t_window = target.iloc[i - window + 1 : i + 1].values
            mask = ~np.isnan(s_window) & ~np.isnan(t_window)
            if mask.sum() < 2:
                continue
            result.iloc[i] = _mutual_information(s_window[mask], t_window[mask], n_bins)
        return result
