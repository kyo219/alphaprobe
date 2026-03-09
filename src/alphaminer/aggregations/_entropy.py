"""Entropy aggregations: ENTROPY, SPECENT, SAMPEN, APEN, PERMEN."""

from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import TYPE_CHECKING

import numpy as np

from alphaminer.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


def _shannon_entropy(w: np.ndarray) -> float:
    """Shannon entropy of binned distribution."""
    n_bins = max(2, int(np.sqrt(len(w))))
    counts, _ = np.histogram(w, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _spectral_entropy(w: np.ndarray) -> float:
    """Normalised spectral entropy from FFT power spectrum."""
    fft_vals = np.fft.rfft(w - np.mean(w))
    psd = np.abs(fft_vals) ** 2
    total = psd.sum()
    if total == 0:
        return 0.0
    psd_norm = psd / total
    psd_norm = psd_norm[psd_norm > 0]
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    max_entropy = np.log2(len(psd))
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def _sample_entropy(w: np.ndarray, m: int, r: float) -> float:
    """Sample entropy with embedding dimension m and tolerance r."""
    n = len(w)
    if n < m + 1:
        return np.nan

    def _count_matches(templates: np.ndarray, tol: float) -> int:
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= tol:
                    count += 1
        return count

    # Build templates of length m and m+1
    templates_m = np.array([w[i : i + m] for i in range(n - m)])
    templates_m1 = np.array([w[i : i + m + 1] for i in range(n - m)])

    b = _count_matches(templates_m, r)
    a = _count_matches(templates_m1, r)

    if b == 0:
        return np.nan
    return float(-np.log(a / b)) if a > 0 else np.nan


def _approx_entropy(w: np.ndarray, m: int, r: float) -> float:
    """Approximate entropy with embedding dimension m and tolerance r."""
    n = len(w)
    if n < m + 1:
        return np.nan

    def _phi(dim: int) -> float:
        templates = np.array([w[i : i + dim] for i in range(n - dim + 1)])
        counts = np.zeros(len(templates))
        for i in range(len(templates)):
            for j in range(len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    counts[i] += 1
        counts /= len(templates)
        return float(np.mean(np.log(counts)))

    return float(_phi(m) - _phi(m + 1))


def _permutation_entropy(w: np.ndarray, order: int) -> float:
    """Permutation entropy of given order."""
    n = len(w)
    if n < order:
        return np.nan

    # Count ordinal patterns
    pattern_counts: dict[tuple[int, ...], int] = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(w[i : i + order]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    total = sum(pattern_counts.values())
    probs = np.array(list(pattern_counts.values())) / total
    probs = probs[probs > 0]
    max_entropy = np.log2(factorial(order))
    if max_entropy == 0:
        return 0.0
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / max_entropy)


@register_aggregation("ENTROPY")
class ShannonEntropy(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).apply(
            _shannon_entropy, raw=True
        )


@register_aggregation("SPECENT")
class SpectralEntropy(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).apply(
            _spectral_entropy, raw=True
        )


@register_aggregation("SAMPEN")
class SampleEntropy(Aggregation):
    """Sample entropy. extra = embedding dimension m. r = 0.2 * std (fixed)."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("SAMPEN requires extra parameter (embedding dimension m). Use SAMPEN_m_WINDOW format.")
        m = extra

        def _sampen_window(w: np.ndarray) -> float:
            r = 0.2 * np.std(w)
            if r == 0:
                return np.nan
            return _sample_entropy(w, m, r)

        return series.rolling(window=window, min_periods=window).apply(
            _sampen_window, raw=True
        )


@register_aggregation("APEN")
class ApproximateEntropy(Aggregation):
    """Approximate entropy. extra = embedding dimension m. r = 0.2 * std (fixed)."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("APEN requires extra parameter (embedding dimension m). Use APEN_m_WINDOW format.")
        m = extra

        def _apen_window(w: np.ndarray) -> float:
            r = 0.2 * np.std(w)
            if r == 0:
                return np.nan
            return _approx_entropy(w, m, r)

        return series.rolling(window=window, min_periods=window).apply(
            _apen_window, raw=True
        )


@register_aggregation("PERMEN")
class PermutationEntropy(Aggregation):
    """Permutation entropy. extra = order."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        if extra is None:
            raise ValueError("PERMEN requires extra parameter (order). Use PERMEN_order_WINDOW format.")
        order = extra

        def _permen_window(w: np.ndarray) -> float:
            return _permutation_entropy(w, order)

        return series.rolling(window=window, min_periods=window).apply(
            _permen_window, raw=True
        )
