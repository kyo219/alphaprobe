from __future__ import annotations

import numpy as np

from alpharadar.correlations._base import Correlation, register_correlation


@register_correlation("chatterjee")
class Chatterjee(Correlation):
    r"""Chatterjee's xi correlation coefficient.

    ξ_n = 1 - 3 * Σ|r_{i+1} - r_i| / (n² - 1)

    where r_i is the rank of Y_{(i)} (Y sorted by X).
    """

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        if n < 2:
            return float("nan")
        # Sort y by x
        order = np.argsort(x)
        y_sorted = y[order]
        # Rank of y_sorted (average ranking)
        ranks = stats_rankdata(y_sorted)
        # Compute xi
        diffs = np.abs(np.diff(ranks))
        xi = 1.0 - 3.0 * np.sum(diffs) / (n * n - 1)
        return float(xi)


def stats_rankdata(a: np.ndarray) -> np.ndarray:
    """Rank data using average method (no scipy dependency for this)."""
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    # Handle ties: average ranks
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense = np.cumsum(obs)[inv]
    # For average ranking, compute count per dense rank
    count = np.bincount(dense)
    cumcount = np.cumsum(count)
    # average rank = (start + end) / 2, 1-based
    avg_rank = np.empty(len(count))
    avg_rank[0] = (cumcount[0] + 1) / 2.0
    for i in range(1, len(count)):
        avg_rank[i] = (cumcount[i - 1] + 1 + cumcount[i]) / 2.0
    return avg_rank[dense]
