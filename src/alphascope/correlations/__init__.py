"""Built-in correlation methods. Importing this package triggers registration."""

from alphascope.correlations._base import Correlation, get_correlation, register_correlation
from alphascope.correlations._chatterjee import Chatterjee
from alphascope.correlations._pearson import Pearson
from alphascope.correlations._spearman import Spearman

__all__ = [
    "Correlation",
    "get_correlation",
    "register_correlation",
    "Chatterjee",
    "Pearson",
    "Spearman",
]
