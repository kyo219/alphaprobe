"""Basic rolling window aggregations: SUM, MEDIAN, STD, VAR, MAX, MIN, RANGE, SKEW, KURT."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alpharadar.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("SUM")
class Sum(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).sum()


@register_aggregation("MEDIAN")
class Median(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).median()


@register_aggregation("STD")
class Std(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).std()


@register_aggregation("VAR")
class Var(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).var()


@register_aggregation("MAX")
class Max(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).max()


@register_aggregation("MIN")
class Min(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).min()


@register_aggregation("RANGE")
class Range(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        r = series.rolling(window=window, min_periods=window)
        return r.max() - r.min()


@register_aggregation("SKEW")
class Skew(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).skew()


@register_aggregation("KURT")
class Kurt(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.rolling(window=window, min_periods=window).kurt()
