from __future__ import annotations

import numpy as np

from alpharadar.correlations._base import Correlation, register_correlation


@register_correlation("pearson")
class Pearson(Correlation):
    """Pearson product-moment correlation."""

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2:
            return float("nan")
        r = np.corrcoef(x, y)[0, 1]
        return float(r)
