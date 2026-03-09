"""AlphaMiner Demo — 全44種Aggregationのラグ相関を可視化."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import alphaminer as am


def main():
    # ── 1. サンプルデータ作成 (3000点) ──────────────────────────────────────

    np.random.seed(42)
    n = 3000

    df = pd.DataFrame({
        "timestamp": pd.date_range("2015-01-01", periods=n, freq="h"),
        "feature_1": np.cumsum(np.random.randn(n) * 0.5) + 100,       # ランダムウォーク（価格っぽい）
        "feature_2": np.abs(np.random.randn(n)) * 10 + 50,             # 非負ノイズ（出来高っぽい）
        "feature_3": np.sin(np.linspace(0, 20 * np.pi, n)) * 3 + np.random.randn(n) * 2,  # 周期 + ノイズ
        "target": np.random.randn(n) * 0.02,                           # 低S/Nのリターン
    })

    print(f"Sample data: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df.head())
    print()

    # ── 2. 全Aggregationで探索 & PNG保存 ──────────────────────────────────

    result = am.explore(
        df,
        target_col="target",
        time_col="timestamp",
        feature_cols=["feature_1", "feature_2", "feature_3"],
        agg=[
            # Basic Rolling
            "RAW_1",
            "MA_5", "MA_20",
            "SUM_20", "MEDIAN_20", "STD_20", "VAR_20",
            "MAX_20", "MIN_20", "RANGE_20", "SKEW_20", "KURT_20",
            # Rank & Normalisation
            "RANK_20", "ZSCORE_20", "CV_20", "NORMDEV_20",
            # Momentum
            "MOM_10", "ROC_10", "MEANREV_20", "TRENDSIG_20",
            # EMA Family
            "EMA_20", "DEMA_20", "TEMA_20", "WMA_20", "EWMSTD_20",
            # Technical
            "RSI_14", "BPOS_20", "RVOL_20", "ARCH_20",
            # Regression
            "LSLOPE_20", "LR2_20",
            # Correlation
            "MC_30",
            "ACF_1_50", "PACF_1_50", "MI_10_50",
            # Entropy
            "ENTROPY_50", "SPECENT_32",
            "SAMPEN_2_50", "APEN_2_50", "PERMEN_3_50",
            # Complexity
            "LZC_50", "HURST_50", "DFA_50",
            # Fractional & Quantile
            "FRACDIFF_5_20", "QUANTILE_25_50",
        ],
        lags=list(range(11)),
        corr_method="pearson",
        show_progress=True,
    )

    # PNG保存
    output_path = "sample/output/lag_correlation_grid.png"
    result.plot(save_path=output_path)
    print(f"\nSaved: {output_path}")

    # DataFrame表示
    df_out = result.to_dataframe()
    print(f"\nResults: {len(df_out)} rows")
    print(df_out.head(20))


if __name__ == "__main__":
    main()
