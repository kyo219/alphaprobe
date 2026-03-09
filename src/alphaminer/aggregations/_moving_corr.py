from __future__ import annotations

from typing import TYPE_CHECKING

from alphaminer.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("MC")
class MovingCorrelation(Aggregation):
    """Rolling Pearson correlation between a feature and the target."""

    def apply(
        self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None,
    ) -> pd.Series:
        if target is None:
            raise ValueError("MC (Moving Correlation) requires a target series.")
        return series.rolling(window=window, min_periods=window).corr(target)
