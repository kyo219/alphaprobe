import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

import alphascope as am
from alphascope._parser import parse_agg
from alphascope._types import AggSpec


class TestParser:
    def test_valid_2segment(self):
        assert parse_agg("MA_5") == AggSpec("MA", 5)
        assert parse_agg("MC_30") == AggSpec("MC", 30)
        assert parse_agg("ema_10") == AggSpec("EMA", 10)
        assert parse_agg("STD_20") == AggSpec("STD", 20)

    def test_valid_3segment(self):
        assert parse_agg("ACF_3_50") == AggSpec("ACF", 50, extra=3)
        assert parse_agg("PACF_5_50") == AggSpec("PACF", 50, extra=5)
        assert parse_agg("MI_10_50") == AggSpec("MI", 50, extra=10)
        assert parse_agg("SAMPEN_2_50") == AggSpec("SAMPEN", 50, extra=2)
        assert parse_agg("FRACDIFF_3_50") == AggSpec("FRACDIFF", 50, extra=3)
        assert parse_agg("QUANTILE_25_50") == AggSpec("QUANTILE", 50, extra=25)

    def test_3segment_str_roundtrip(self):
        spec = parse_agg("ACF_3_50")
        assert str(spec) == "ACF_3_50"
        spec2 = parse_agg("FRACDIFF_5_20")
        assert str(spec2) == "FRACDIFF_5_20"

    def test_extra_required_aggs_reject_2segment(self):
        for name in ["ACF", "PACF", "MI", "SAMPEN", "APEN", "PERMEN", "FRACDIFF", "QUANTILE"]:
            with pytest.raises(ValueError, match="requires an extra parameter"):
                parse_agg(f"{name}_50")

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid agg spec"):
            parse_agg("bad")
        with pytest.raises(ValueError, match="Invalid agg spec"):
            parse_agg("MA5")
        with pytest.raises(ValueError, match="Invalid agg spec"):
            parse_agg("_5")


class TestExplore:
    def test_basic(self, sample_df):
        result = am.explore(
            sample_df,
            target_col="return_1d",
            time_col="date",
            feature_cols=["close"],
            agg=["MA_5"],
            lags=[0, 1, 2],
            corr_method="pearson",
            show_progress=False,
        )
        df_out = result.to_dataframe()
        assert len(df_out) == 3  # 1 feature * 1 agg * 3 lags
        assert set(df_out.columns) == {"feature", "agg", "lag", "correlation"}
        assert list(sorted(df_out["lag"].unique())) == [0, 1, 2]

    def test_multiple_features_aggs(self, sample_df):
        result = am.explore(
            sample_df,
            target_col="return_1d",
            time_col="date",
            feature_cols=["close", "volume"],
            agg=["MA_5", "MA_10", "MC_30"],
            lags=list(range(5)),
            corr_method="pearson",
            show_progress=False,
        )
        df_out = result.to_dataframe()
        # 2 features * 3 aggs * 5 lags = 30
        assert len(df_out) == 30

    def test_spearman(self, sample_df):
        result = am.explore(
            sample_df,
            target_col="return_1d",
            time_col="date",
            feature_cols=["close"],
            agg=["MA_5"],
            lags=[0],
            corr_method="spearman",
            show_progress=False,
        )
        assert result.corr_method == "spearman"
        assert len(result.results) == 1

    def test_chatterjee(self, sample_df):
        result = am.explore(
            sample_df,
            target_col="return_1d",
            time_col="date",
            feature_cols=["close"],
            agg=["MA_5"],
            lags=[0],
            corr_method="chatterjee",
            show_progress=False,
        )
        assert result.corr_method == "chatterjee"

    def test_missing_column(self, sample_df):
        with pytest.raises(KeyError, match="not found"):
            am.explore(
                sample_df,
                target_col="nonexistent",
                time_col="date",
                feature_cols=["close"],
                agg=["MA_5"],
                lags=[0],
            )

    def test_plot_does_not_raise(self, sample_df):
        result = am.explore(
            sample_df,
            target_col="return_1d",
            time_col="date",
            feature_cols=["close"],
            agg=["MA_5"],
            lags=[0, 1],
            show_progress=False,
        )
        # plot() was already called by explore(); calling again should not raise
        result.plot()

    def test_new_aggs_e2e(self, sample_df):
        """E2E test with a selection of new aggregation types."""
        result = am.explore(
            sample_df,
            target_col="return_1d",
            time_col="date",
            feature_cols=["close"],
            agg=["STD_10", "RSI_14", "EMA_5", "RANK_20"],
            lags=[0, 1, 2],
            corr_method="pearson",
            show_progress=False,
        )
        df_out = result.to_dataframe()
        # 1 feature * 4 aggs * 3 lags = 12
        assert len(df_out) == 12

    def test_3segment_agg_e2e(self, sample_df):
        """E2E test with 3-segment aggregation specs."""
        result = am.explore(
            sample_df,
            target_col="return_1d",
            time_col="date",
            feature_cols=["close"],
            agg=["ACF_1_30", "QUANTILE_50_20"],
            lags=[0, 1],
            corr_method="pearson",
            show_progress=False,
        )
        df_out = result.to_dataframe()
        # 1 feature * 2 aggs * 2 lags = 4
        assert len(df_out) == 4
        assert "ACF_1_30" in df_out["agg"].values
        assert "QUANTILE_50_20" in df_out["agg"].values
