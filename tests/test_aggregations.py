import numpy as np
import pandas as pd
import pytest

from alphaminer.aggregations import get_aggregation


class TestMovingAverage:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = get_aggregation("MA")
        result = ma.apply(s, window=3)
        assert result.isna().sum() == 2  # first 2 are NaN
        np.testing.assert_almost_equal(result.iloc[2], 2.0)
        np.testing.assert_almost_equal(result.iloc[4], 4.0)

    def test_case_insensitive(self):
        assert get_aggregation("ma") is not None
        assert get_aggregation("Ma") is not None


class TestMovingCorrelation:
    def test_requires_target(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mc = get_aggregation("MC")
        with pytest.raises(ValueError, match="requires a target"):
            mc.apply(s, window=3)

    def test_perfect_correlation(self):
        s = pd.Series(range(20), dtype=float)
        target = pd.Series(range(20), dtype=float)
        mc = get_aggregation("MC")
        result = mc.apply(s, window=5, target=target)
        # Perfect correlation for identical series
        np.testing.assert_almost_equal(result.dropna().iloc[0], 1.0)


class TestRaw:
    def test_passthrough(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        raw = get_aggregation("RAW")
        result = raw.apply(s, window=999)
        pd.testing.assert_series_equal(result, s)


class TestRegistry:
    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown aggregation"):
            get_aggregation("NONEXISTENT")
