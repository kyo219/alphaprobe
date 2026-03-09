from __future__ import annotations

import re

from alphaminer._types import AggSpec

_PATTERN = re.compile(r"^([A-Za-z]+)_(\d+)$")


def parse_agg(spec: str) -> AggSpec:
    """Parse an aggregation spec string like ``"MA_5"`` into an :class:`AggSpec`.

    Raises
    ------
    ValueError
        If the string does not match the expected ``NAME_WINDOW`` format.
    """
    m = _PATTERN.match(spec)
    if m is None:
        raise ValueError(
            f"Invalid agg spec {spec!r}. Expected format: NAME_WINDOW (e.g. 'MA_5', 'MC_30')"
        )
    return AggSpec(name=m.group(1).upper(), window=int(m.group(2)))
