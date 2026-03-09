"""AlphaMiner - Lag-correlation explorer for low S/N time-series data."""

from alphaminer._engine import explore
from alphaminer._types import AggSpec, CorrResult, ExploreResult

__all__ = ["explore", "AggSpec", "CorrResult", "ExploreResult"]
