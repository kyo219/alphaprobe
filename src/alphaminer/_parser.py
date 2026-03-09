from __future__ import annotations

import re

from alphaminer._types import AggSpec

_PATTERN_2 = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_(\d+)$")
_PATTERN_3 = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_(\d+)_(\d+)$")

_EXTRA_PARAM_AGGS = frozenset(
    {"ACF", "PACF", "MI", "SAMPEN", "APEN", "PERMEN", "FRACDIFF", "QUANTILE"}
)


def parse_agg(spec: str) -> AggSpec:
    """Parse an aggregation spec string into an :class:`AggSpec`.

    Supports two formats:
    - ``"MA_5"`` → ``AggSpec("MA", 5)``
    - ``"ACF_3_50"`` → ``AggSpec("ACF", 50, extra=3)``

    Raises
    ------
    ValueError
        If the string does not match the expected format, or if extra is
        missing for aggregations that require it.
    """
    m3 = _PATTERN_3.match(spec)
    if m3 is not None:
        name = m3.group(1).upper()
        extra = int(m3.group(2))
        window = int(m3.group(3))
        return AggSpec(name=name, window=window, extra=extra)

    m2 = _PATTERN_2.match(spec)
    if m2 is not None:
        name = m2.group(1).upper()
        window = int(m2.group(2))
        if name in _EXTRA_PARAM_AGGS:
            raise ValueError(
                f"Aggregation {name!r} requires an extra parameter. "
                f"Expected format: {name}_EXTRA_WINDOW (e.g. '{name}_3_{window}')"
            )
        return AggSpec(name=name, window=window)

    raise ValueError(
        f"Invalid agg spec {spec!r}. Expected format: NAME_WINDOW (e.g. 'MA_5') "
        f"or NAME_EXTRA_WINDOW (e.g. 'ACF_3_50')"
    )
