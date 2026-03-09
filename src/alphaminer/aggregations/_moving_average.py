from __future__ import annotations

from typing import TYPE_CHECKING

from alphaminer.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("MA")
class MovingAverage(Aggregation):
    """Simple moving average."""

    def apply(
        self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None,
    ) -> pd.Series:
        return series.rolling(window=window, min_periods=window).mean()
