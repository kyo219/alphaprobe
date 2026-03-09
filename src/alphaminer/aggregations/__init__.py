"""Built-in aggregations. Importing this package triggers registration."""

from alphaminer.aggregations._base import Aggregation, get_aggregation, register_aggregation
from alphaminer.aggregations._moving_average import MovingAverage
from alphaminer.aggregations._moving_corr import MovingCorrelation
from alphaminer.aggregations._raw import Raw

__all__ = [
    "Aggregation",
    "get_aggregation",
    "register_aggregation",
    "MovingAverage",
    "MovingCorrelation",
    "Raw",
]
