"""Built-in correlation methods. Importing this package triggers registration."""

from alphaprobe.correlations._base import Correlation, get_correlation, register_correlation
from alphaprobe.correlations._chatterjee import Chatterjee
from alphaprobe.correlations._pearson import Pearson
from alphaprobe.correlations._spearman import Spearman

__all__ = [
    "Correlation",
    "get_correlation",
    "register_correlation",
    "Chatterjee",
    "Pearson",
    "Spearman",
]
