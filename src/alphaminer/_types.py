from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AggSpec:
    """Parsed aggregation specification, e.g. AggSpec("MA", 5) or AggSpec("ACF", 50, extra=3)."""

    name: str
    window: int
    extra: int | None = None

    def __str__(self) -> str:
        if self.extra is not None:
            return f"{self.name}_{self.extra}_{self.window}"
        return f"{self.name}_{self.window}"


@dataclass(frozen=True)
class CorrResult:
    """Single correlation result for one (feature, agg, lag) combination."""

    feature: str
    agg: str
    lag: int
    correlation: float


class ExploreResult:
    """Container for explore() output. Supports plotting and DataFrame export."""

    def __init__(
        self,
        results: list[CorrResult],
        feature_cols: list[str],
        agg_labels: list[str],
        corr_method: str,
    ) -> None:
        self.results = results
        self.feature_cols = feature_cols
        self.agg_labels = agg_labels
        self.corr_method = corr_method

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a tidy DataFrame."""
        return pd.DataFrame(
            [
                {
                    "feature": r.feature,
                    "agg": r.agg,
                    "lag": r.lag,
                    "correlation": r.correlation,
                }
                for r in self.results
            ]
        )

    def plot(self, figsize: tuple[float, float] | None = None) -> None:
        """Render ACF-style subplot grid."""
        from alphaminer._plot import plot_results

        plot_results(self, figsize=figsize)
