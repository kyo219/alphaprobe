from __future__ import annotations

import numpy as np
from scipy import stats

from alphascope.correlations._base import Correlation, register_correlation


@register_correlation("spearman")
class Spearman(Correlation):
    """Spearman rank correlation."""

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2:
            return float("nan")
        r, _ = stats.spearmanr(x, y)
        return float(r)
