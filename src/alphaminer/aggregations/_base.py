from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

_REGISTRY: dict[str, type[Aggregation]] = {}


def register_aggregation(name: str):
    """Decorator to register an Aggregation subclass under *name*."""

    def decorator(cls: type[Aggregation]) -> type[Aggregation]:
        _REGISTRY[name.upper()] = cls
        return cls

    return decorator


def get_aggregation(name: str) -> Aggregation:
    """Instantiate a registered Aggregation by name (case-insensitive)."""
    key = name.upper()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown aggregation {name!r}. Available: {available}")
    return _REGISTRY[key]()


class Aggregation(ABC):
    """Base class for aggregation plugins."""

    @abstractmethod
    def apply(
        self,
        series: pd.Series,
        window: int,
        *,
        target: pd.Series | None = None,
        extra: int | None = None,
    ) -> pd.Series:
        """Apply the aggregation to *series* with the given *window*.

        Parameters
        ----------
        series : pd.Series
            Feature column.
        window : int
            Rolling window size.
        target : pd.Series | None
            Target column, required for cross-feature aggregations like MC.
        extra : int | None
            Extra parameter for aggregations that need it (e.g. lag for ACF).
        """
