from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_REGISTRY: dict[str, type[Correlation]] = {}


def register_correlation(name: str):
    """Decorator to register a Correlation subclass under *name*."""

    def decorator(cls: type[Correlation]) -> type[Correlation]:
        _REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_correlation(name: str) -> Correlation:
    """Instantiate a registered Correlation by name (case-insensitive)."""
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown correlation method {name!r}. Available: {available}")
    return _REGISTRY[key]()


class Correlation(ABC):
    """Base class for correlation method plugins."""

    @abstractmethod
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute correlation between two 1-D arrays (NaN-free)."""
