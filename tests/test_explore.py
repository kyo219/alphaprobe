import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

import alphaminer as am
from alphaminer._parser import parse_agg
from alphaminer._types import AggSpec


class TestParser:
    def test_valid(self):
        assert parse_agg("MA_5") == AggSpec("MA", 5)
        assert parse_agg("MC_30") == AggSpec("MC", 30)
        assert parse_agg("ema_10") == AggSpec("EMA", 10)

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
