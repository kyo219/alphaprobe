from __future__ import annotations

from typing import TYPE_CHECKING

from alphascope.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("RAW")
class Raw(Aggregation):
    """No-op aggregation — returns the series as-is."""

    def apply(
        self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None,
    ) -> pd.Series:
        return series
