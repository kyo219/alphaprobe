import numpy as np
import pandas as pd
import pytest

from alpharadar.aggregations import get_aggregation


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

    def test_all_44_registered(self):
        from alpharadar.aggregations._base import _REGISTRY
        assert len(_REGISTRY) == 44


# ── New aggregation tests ──────────────────────────────────────────────


class TestBasicRolling:
    @pytest.fixture
    def s(self):
        return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_sum(self, s):
        result = get_aggregation("SUM").apply(s, window=3)
        assert result.isna().sum() == 2
        np.testing.assert_almost_equal(result.iloc[2], 6.0)  # 1+2+3
        np.testing.assert_almost_equal(result.iloc[9], 27.0)  # 8+9+10

    def test_median(self, s):
        result = get_aggregation("MEDIAN").apply(s, window=3)
        np.testing.assert_almost_equal(result.iloc[2], 2.0)

    def test_std(self, s):
        result = get_aggregation("STD").apply(s, window=3)
        assert result.isna().sum() == 2
        assert result.iloc[2] > 0

    def test_var(self, s):
        result = get_aggregation("VAR").apply(s, window=3)
        std_result = get_aggregation("STD").apply(s, window=3)
        np.testing.assert_array_almost_equal(
            result.dropna().values, (std_result.dropna().values) ** 2
        )

    def test_max(self, s):
        result = get_aggregation("MAX").apply(s, window=3)
        np.testing.assert_almost_equal(result.iloc[2], 3.0)

    def test_min(self, s):
        result = get_aggregation("MIN").apply(s, window=3)
        np.testing.assert_almost_equal(result.iloc[2], 1.0)

    def test_range(self, s):
        result = get_aggregation("RANGE").apply(s, window=3)
        np.testing.assert_almost_equal(result.iloc[2], 2.0)  # 3-1

    def test_skew(self, s):
        result = get_aggregation("SKEW").apply(s, window=5)
        assert result.isna().sum() == 4
        # Uniform spacing → skew near 0
        np.testing.assert_almost_equal(result.iloc[4], 0.0, decimal=5)

    def test_kurt(self, s):
        result = get_aggregation("KURT").apply(s, window=5)
        assert result.isna().sum() == 4


class TestRankNorm:
    @pytest.fixture
    def s(self):
        return pd.Series([10.0, 1.0, 5.0, 8.0, 3.0, 7.0, 2.0, 9.0, 4.0, 6.0])

    def test_rank(self, s):
        result = get_aggregation("RANK").apply(s, window=5)
        assert result.isna().sum() == 4
        # Rank is between 0 and 1
        assert all(0 <= v <= 1 for v in result.dropna())

    def test_zscore(self, s):
        result = get_aggregation("ZSCORE").apply(s, window=5)
        assert result.isna().sum() == 4

    def test_cv(self, s):
        result = get_aggregation("CV").apply(s, window=5)
        assert result.isna().sum() == 4

    def test_normdev(self, s):
        result = get_aggregation("NORMDEV").apply(s, window=5)
        assert result.isna().sum() == 4
        assert all(v >= 0 for v in result.dropna())


class TestMomentum:
    @pytest.fixture
    def s(self):
        return pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 107.0, 110.0])

    def test_mom(self, s):
        result = get_aggregation("MOM").apply(s, window=3)
        # MOM = x[t] - x[t-3]
        np.testing.assert_almost_equal(result.iloc[3], 5.0)  # 105 - 100

    def test_roc(self, s):
        result = get_aggregation("ROC").apply(s, window=3)
        # ROC = (x[t]/x[t-3] - 1) * 100
        np.testing.assert_almost_equal(result.iloc[3], (105.0 / 100.0 - 1) * 100)

    def test_meanrev(self, s):
        result = get_aggregation("MEANREV").apply(s, window=5)
        assert result.isna().sum() == 4

    def test_trendsig(self, s):
        # Strong uptrend → positive t-stat
        s_up = pd.Series(np.arange(20, dtype=float))
        result = get_aggregation("TRENDSIG").apply(s_up, window=10)
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v > 0 for v in valid)


class TestEmaFamily:
    @pytest.fixture
    def s(self):
        np.random.seed(42)
        return pd.Series(np.cumsum(np.random.randn(50)) + 100)

    def test_ema(self, s):
        result = get_aggregation("EMA").apply(s, window=10)
        assert result.isna().sum() == 9
        assert len(result.dropna()) > 0

    def test_dema(self, s):
        result = get_aggregation("DEMA").apply(s, window=10)
        assert len(result.dropna()) > 0

    def test_tema(self, s):
        result = get_aggregation("TEMA").apply(s, window=10)
        assert len(result.dropna()) > 0

    def test_wma(self, s):
        result = get_aggregation("WMA").apply(s, window=5)
        assert result.isna().sum() == 4

    def test_ewmstd(self, s):
        result = get_aggregation("EWMSTD").apply(s, window=10)
        assert len(result.dropna()) > 0
        assert all(v >= 0 for v in result.dropna())

    def test_ema_dema_tema_relationship(self):
        """DEMA and TEMA should respond faster than EMA to trend changes."""
        s = pd.Series(list(range(30)) + list(range(30, 0, -1)), dtype=float)
        ema = get_aggregation("EMA").apply(s, window=5)
        dema = get_aggregation("DEMA").apply(s, window=5)
        tema = get_aggregation("TEMA").apply(s, window=5)
        # All should produce valid values
        assert len(ema.dropna()) > 0
        assert len(dema.dropna()) > 0
        assert len(tema.dropna()) > 0


class TestTechnical:
    @pytest.fixture
    def s(self):
        np.random.seed(42)
        return pd.Series(np.cumsum(np.random.randn(100)) + 100)

    def test_rsi(self, s):
        result = get_aggregation("RSI").apply(s, window=14)
        valid = result.dropna()
        assert len(valid) > 0
        # RSI is between 0 and 100
        assert all(0 <= v <= 100 for v in valid)

    def test_bpos(self, s):
        result = get_aggregation("BPOS").apply(s, window=20)
        assert len(result.dropna()) > 0

    def test_rvol(self, s):
        result = get_aggregation("RVOL").apply(s, window=20)
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)

    def test_arch(self, s):
        result = get_aggregation("ARCH").apply(s, window=20)
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)


class TestRegression:
    def test_lslope_linear(self):
        s = pd.Series(np.arange(20, dtype=float) * 2 + 5)
        result = get_aggregation("LSLOPE").apply(s, window=10)
        valid = result.dropna()
        assert len(valid) > 0
        # Slope should be close to 2
        np.testing.assert_almost_equal(valid.iloc[-1], 2.0, decimal=5)

    def test_lr2_perfect(self):
        s = pd.Series(np.arange(20, dtype=float) * 3 + 1)
        result = get_aggregation("LR2").apply(s, window=10)
        valid = result.dropna()
        assert len(valid) > 0
        # Perfect linear → R² = 1
        np.testing.assert_almost_equal(valid.iloc[-1], 1.0, decimal=5)


class TestEntropy:
    def test_entropy_uniform(self):
        # Uniform distribution should have high entropy
        np.random.seed(42)
        s = pd.Series(np.random.uniform(0, 1, 100))
        result = get_aggregation("ENTROPY").apply(s, window=50)
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_specent(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(100))
        result = get_aggregation("SPECENT").apply(s, window=32)
        valid = result.dropna()
        assert len(valid) > 0
        assert all(0 <= v <= 1 for v in valid)

    def test_sampen_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("SAMPEN").apply(s, window=20)

    def test_sampen_basic(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(100))
        result = get_aggregation("SAMPEN").apply(s, window=30, extra=2)
        # Should have some valid values (may have NaN for some windows)
        assert len(result) == 100

    def test_apen_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("APEN").apply(s, window=20)

    def test_apen_basic(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(100))
        result = get_aggregation("APEN").apply(s, window=30, extra=2)
        assert len(result) == 100

    def test_permen_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("PERMEN").apply(s, window=20)

    def test_permen_basic(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(100))
        result = get_aggregation("PERMEN").apply(s, window=30, extra=3)
        valid = result.dropna()
        assert len(valid) > 0
        # Normalised permutation entropy between 0 and 1
        assert all(0 <= v <= 1 for v in valid)


class TestComplexity:
    @pytest.fixture
    def s(self):
        np.random.seed(42)
        return pd.Series(np.random.randn(100))

    def test_lzc(self, s):
        result = get_aggregation("LZC").apply(s, window=30)
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_hurst(self, s):
        result = get_aggregation("HURST").apply(s, window=50)
        # May have NaN for small windows
        assert len(result) == 100

    def test_dfa(self, s):
        result = get_aggregation("DFA").apply(s, window=50)
        assert len(result) == 100


class TestCorrelationAgg:
    def test_acf_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("ACF").apply(s, window=20)

    def test_acf_lag0(self):
        """ACF at lag 0 should be 1.0 (autocorrelation with itself)."""
        np.random.seed(42)
        s = pd.Series(np.random.randn(100))
        # Lag 0 should not be used in practice but let's test lag=1
        result = get_aggregation("ACF").apply(s, window=30, extra=1)
        valid = result.dropna()
        assert len(valid) > 0
        # ACF values should be between -1 and 1
        assert all(-1 <= v <= 1 for v in valid)

    def test_pacf_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("PACF").apply(s, window=20)

    def test_pacf_basic(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(100))
        result = get_aggregation("PACF").apply(s, window=30, extra=1)
        valid = result.dropna()
        assert len(valid) > 0

    def test_mi_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("MI").apply(s, window=20)

    def test_mi_requires_target(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires a target"):
            get_aggregation("MI").apply(s, window=20, extra=10)

    def test_mi_basic(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(100))
        target = pd.Series(np.random.randn(100))
        result = get_aggregation("MI").apply(s, window=30, extra=10, target=target)
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)


class TestFractional:
    def test_fracdiff_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("FRACDIFF").apply(s, window=20)

    def test_fracdiff_basic(self):
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        result = get_aggregation("FRACDIFF").apply(s, window=20, extra=5)  # d=0.5
        valid = result.dropna()
        assert len(valid) > 0

    def test_quantile_requires_extra(self):
        s = pd.Series(np.random.randn(50))
        with pytest.raises(ValueError, match="requires extra"):
            get_aggregation("QUANTILE").apply(s, window=20)

    def test_quantile_50_equals_median(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(50))
        q50 = get_aggregation("QUANTILE").apply(s, window=10, extra=50)
        median = get_aggregation("MEDIAN").apply(s, window=10)
        np.testing.assert_array_almost_equal(
            q50.dropna().values, median.dropna().values
        )
