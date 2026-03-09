"""Built-in correlation methods. Importing this package triggers registration."""

from alphaminer.correlations._base import Correlation, get_correlation, register_correlation
from alphaminer.correlations._chatterjee import Chatterjee
from alphaminer.correlations._pearson import Pearson
from alphaminer.correlations._spearman import Spearman

__all__ = [
    "Correlation",
    "get_correlation",
    "register_correlation",
    "Chatterjee",
    "Pearson",
    "Spearman",
]
