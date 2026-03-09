import numpy as np
import pytest

from alphaminer.correlations import get_correlation


class TestPearson:
    def test_perfect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        r = get_correlation("pearson").compute(x, y)
        np.testing.assert_almost_equal(r, 1.0)

    def test_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        r = get_correlation("pearson").compute(x, y)
        np.testing.assert_almost_equal(r, -1.0)

    def test_short(self):
        r = get_correlation("pearson").compute(np.array([1.0]), np.array([2.0]))
        assert np.isnan(r)


class TestSpearman:
    def test_monotonic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        r = get_correlation("spearman").compute(x, y)
        np.testing.assert_almost_equal(r, 1.0)


class TestChatterjee:
    def test_perfect_monotone(self):
        x = np.arange(100, dtype=float)
        y = np.arange(100, dtype=float)
        xi = get_correlation("chatterjee").compute(x, y)
        # Should be close to 1 for a perfect monotone relationship
        assert xi > 0.9

    def test_independent(self):
        np.random.seed(0)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        xi = get_correlation("chatterjee").compute(x, y)
        # Should be close to 0 for independent data
        assert abs(xi) < 0.1


class TestRegistry:
    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown correlation"):
            get_correlation("nonexistent")
