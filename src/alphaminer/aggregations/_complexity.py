"""Complexity aggregations: LZC, HURST, DFA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphaminer.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


def _lempel_ziv_complexity(w: np.ndarray) -> float:
    """Lempel-Ziv complexity (binary sequence from median split)."""
    binary = (w >= np.median(w)).astype(int)
    s = "".join(map(str, binary))
    n = len(s)
    if n == 0:
        return 0.0

    # LZ76 algorithm
    i, k, l_val = 0, 1, 1
    c = 1
    while l_val + k <= n:
        if s[l_val : l_val + k] in _substrings(s[i:l_val]):
            k += 1
        else:
            c += 1
            l_val += k
            i = 0
            k = 1
    # Normalise by n / log2(n)
    if n <= 1:
        return float(c)
    return float(c * np.log2(n) / n)


def _substrings(s: str) -> set[str]:
    """Return all substrings of s."""
    result = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            result.add(s[i:j])
    return result


def _hurst_rs(w: np.ndarray) -> float:
    """Hurst exponent via rescaled range (R/S) analysis."""
    n = len(w)
    if n < 8:
        return np.nan

    max_k = n // 2
    sizes = []
    rs_values = []

    for size in [n // 4, n // 2, n]:
        if size < 4:
            continue
        n_chunks = n // size
        if n_chunks == 0:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = w[i * size : (i + 1) * size]
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 2:
        return np.nan

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    slope = np.polyfit(log_sizes, log_rs, 1)[0]
    return float(slope)


def _dfa(w: np.ndarray) -> float:
    """Detrended Fluctuation Analysis — scaling exponent alpha."""
    n = len(w)
    if n < 8:
        return np.nan

    # Cumulative sum of demeaned series
    y = np.cumsum(w - np.mean(w))

    scales = []
    fluctuations = []
    for scale in [4, 8, 16, n // 4, n // 2]:
        if scale < 4 or scale > n // 2:
            continue
        n_segments = n // scale
        if n_segments == 0:
            continue
        fluct = []
        for seg in range(n_segments):
            segment = y[seg * scale : (seg + 1) * scale]
            x = np.arange(scale, dtype=np.float64)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            fluct.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if fluct:
            scales.append(scale)
            fluctuations.append(np.mean(fluct))

    if len(scales) < 2:
        return np.nan

    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)
    slope = np.polyfit(log_scales, log_fluct, 1)[0]
    return float(slope)


@register_aggregation("LZC")
class LempelZivComplexity(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).apply(
            _lempel_ziv_complexity, raw=True
        )


@register_aggregation("HURST")
class HurstExponent(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).apply(
            _hurst_rs, raw=True
        )


@register_aggregation("DFA")
class DetrendedFluctuationAnalysis(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).apply(
            _dfa, raw=True
        )
