"""AlphaScope - Lag-correlation explorer for low S/N time-series data."""

from alpharadar._engine import explore
from alpharadar._types import AggSpec, CorrResult, ExploreResult

__version__ = "0.1.1"
__all__ = ["explore", "AggSpec", "CorrResult", "ExploreResult", "__version__"]
