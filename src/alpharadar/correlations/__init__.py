"""Built-in correlation methods. Importing this package triggers registration."""

from alpharadar.correlations._base import Correlation, get_correlation, register_correlation
from alpharadar.correlations._chatterjee import Chatterjee
from alpharadar.correlations._pearson import Pearson
from alpharadar.correlations._spearman import Spearman

__all__ = [
    "Correlation",
    "get_correlation",
    "register_correlation",
    "Chatterjee",
    "Pearson",
    "Spearman",
]
