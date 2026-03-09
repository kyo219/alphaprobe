"""Technical indicator aggregations: RSI, BPOS, RVOL, ARCH."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphaprobe.aggregations._base import Aggregation, register_aggregation

if TYPE_CHECKING:
    import pandas as pd


@register_aggregation("RSI")
class RelativeStrengthIndex(Aggregation):
    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=window, min_periods=window).mean()
        avg_loss = loss.ewm(span=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)


@register_aggregation("BPOS")
class BollingerPosition(Aggregation):
    """Position within Bollinger Bands: (x - MA) / (2 * STD)."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        r = series.rolling(window=window, min_periods=window)
        return (series - r.mean()) / (2 * r.std())


@register_aggregation("RVOL")
class RealisedVolatility(Aggregation):
    """Rolling standard deviation of returns (diff)."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return series.diff().rolling(window=window, min_periods=window).std()


@register_aggregation("ARCH")
class ArchEffect(Aggregation):
    """Rolling mean of squared returns — proxy for ARCH effect."""

    def apply(self, series: pd.Series, window: int, *, target: pd.Series | None = None, extra: int | None = None) -> pd.Series:
        return (series.diff() ** 2).rolling(window=window, min_periods=window).mean()
