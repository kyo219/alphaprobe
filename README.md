# AlphaMiner

Visualize lag-correlations between a target column and aggregated features for low signal-to-noise ratio time-series data. Produces an ACF-style subplot grid to help discover useful feature transformations.

## Install

```bash
pip install -e .
```

## Quick Start

```python
import pandas as pd, numpy as np
import alphaminer as am

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=n),
    "close": np.cumsum(np.random.randn(n)) + 100,
    "volume": np.abs(np.random.randn(n)) * 1000,
    "return_1d": np.random.randn(n) * 0.02,
})

result = am.explore(
    df,
    target_col="return_1d",
    time_col="date",
    feature_cols=["close", "volume"],
    agg=["MA_5", "STD_10", "RSI_14", "EMA_20", "ACF_1_30"],
    lags=list(range(11)),
    corr_method="pearson",
)

result.to_dataframe()  # tidy DataFrame output
result.plot()          # re-render the subplot grid
```

## API

### `am.explore(df, *, target_col, time_col, feature_cols, agg, lags, corr_method="pearson", max_workers=None, show_progress=True)`

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Input data |
| `target_col` | `str` | Target column (e.g. 1-period forward return) |
| `time_col` | `str` | Time/date column for sorting |
| `feature_cols` | `list[str]` | Feature columns to explore |
| `agg` | `list[str]` | Aggregation specs (see format below) |
| `lags` | `list[int]` | Lag values to compute |
| `corr_method` | `str` | `"pearson"`, `"spearman"`, or `"chatterjee"` |
| `max_workers` | `int \| None` | Process pool size (defaults to CPU count) |
| `show_progress` | `bool` | Show rich progress bar |

Returns an `ExploreResult` with `.plot()` and `.to_dataframe()` methods.

## Aggregation Spec Format

Two formats are supported:

| Format | Example | Description |
|---|---|---|
| `NAME_WINDOW` | `MA_5`, `RSI_14` | Standard aggregations with a rolling window |
| `NAME_EXTRA_WINDOW` | `ACF_3_50`, `FRACDIFF_5_20` | Aggregations that require an extra parameter |

## Built-in Aggregations (44 types)

### Basic Rolling (9)

| Code | Name | Formula |
|---|---|---|
| `MA` | Moving Average | `rolling(W).mean()` |
| `SUM` | Rolling Sum | `rolling(W).sum()` |
| `MEDIAN` | Rolling Median | `rolling(W).median()` |
| `STD` | Rolling Std Dev | `rolling(W).std()` |
| `VAR` | Rolling Variance | `rolling(W).var()` |
| `MAX` | Rolling Max | `rolling(W).max()` |
| `MIN` | Rolling Min | `rolling(W).min()` |
| `RANGE` | Rolling Range | `rolling(W).max() - rolling(W).min()` |
| `SKEW` | Rolling Skewness | `rolling(W).skew()` |
| `KURT` | Rolling Kurtosis | `rolling(W).kurt()` |

### Rank & Normalisation (4)

| Code | Name | Formula |
|---|---|---|
| `RANK` | Rolling Rank | Percentile rank of latest value in window |
| `ZSCORE` | Rolling Z-Score | `(x - rolling.mean()) / rolling.std()` |
| `CV` | Coefficient of Variation | `rolling.std() / rolling.mean()` |
| `NORMDEV` | Normality Deviation | `abs(skew) + abs(kurt - 3)` |

### Momentum (4)

| Code | Name | Formula |
|---|---|---|
| `MOM` | Momentum | `x[t] - x[t-W]` |
| `ROC` | Rate of Change | `(x[t] / x[t-W] - 1) * 100` |
| `MEANREV` | Mean Reversion | Negative autocorrelation of deviations |
| `TRENDSIG` | Trend Signal | T-statistic of rolling linear regression |

### EMA Family (5)

| Code | Name | Formula |
|---|---|---|
| `EMA` | Exponential MA | `ewm(span=W).mean()` |
| `DEMA` | Double EMA | `2*EMA - EMA(EMA)` |
| `TEMA` | Triple EMA | `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))` |
| `WMA` | Weighted MA | Linearly weighted moving average |
| `EWMSTD` | EWM Std Dev | `ewm(span=W).std()` |

### Technical (4)

| Code | Name | Formula |
|---|---|---|
| `RSI` | Relative Strength Index | Gain/loss ratio with EWM smoothing |
| `BPOS` | Bollinger Position | `(x - MA) / (2 * STD)` |
| `RVOL` | Realised Volatility | `diff().rolling(W).std()` |
| `ARCH` | ARCH Effect | `(diff()²).rolling(W).mean()` |

### Regression (2)

| Code | Name | Formula |
|---|---|---|
| `LSLOPE` | Linear Slope | Vectorised rolling OLS slope |
| `LR2` | Linear R² | Rolling R-squared of linear regression |

### Correlation-based (3) — requires `extra`

| Code | Name | Extra | Example |
|---|---|---|---|
| `MC` | Moving Correlation | — | `MC_30` (requires target) |
| `ACF` | Autocorrelation | lag | `ACF_3_50` → lag=3, window=50 |
| `PACF` | Partial Autocorrelation | lag | `PACF_5_50` → lag=5, window=50 |
| `MI` | Mutual Information | bins | `MI_10_50` → 10 bins, window=50 (requires target) |

### Entropy (5)

| Code | Name | Extra | Example |
|---|---|---|---|
| `ENTROPY` | Shannon Entropy | — | `ENTROPY_50` |
| `SPECENT` | Spectral Entropy | — | `SPECENT_32` |
| `SAMPEN` | Sample Entropy | m (embed dim) | `SAMPEN_2_50` → m=2, window=50 |
| `APEN` | Approximate Entropy | m (embed dim) | `APEN_2_50` → m=2, window=50 |
| `PERMEN` | Permutation Entropy | order | `PERMEN_3_50` → order=3, window=50 |

### Complexity (3)

| Code | Name | Formula |
|---|---|---|
| `LZC` | Lempel-Ziv Complexity | Binary sequence complexity |
| `HURST` | Hurst Exponent | Rescaled range (R/S) analysis |
| `DFA` | Detrended Fluctuation Analysis | DFA scaling exponent |

### Fractional & Quantile (2) — requires `extra`

| Code | Name | Extra | Example |
|---|---|---|---|
| `FRACDIFF` | Fractional Differencing | d×10 | `FRACDIFF_3_50` → d=0.3, window=50 |
| `QUANTILE` | Rolling Quantile | percentile | `QUANTILE_25_50` → 25th %ile, window=50 |

### Identity (1)

| Code | Name | Formula |
|---|---|---|
| `RAW` | Raw (no-op) | Returns series as-is |

## Built-in Correlation Methods

| Name | Description |
|---|---|
| `pearson` | Pearson product-moment correlation |
| `spearman` | Spearman rank correlation |
| `chatterjee` | Chatterjee's xi coefficient (detects non-linear dependence) |

## Adding a Custom Aggregation

Create a file and use the `@register_aggregation` decorator:

```python
# src/alphaminer/aggregations/_my_agg.py
from alphaminer.aggregations._base import Aggregation, register_aggregation

@register_aggregation("MYAGG")
class MyAggregation(Aggregation):
    def apply(self, series, window, *, target=None, extra=None):
        return series.rolling(window).mean()  # your logic here
```

Then add the import in `aggregations/__init__.py`. Use it as `"MYAGG_10"`.

## Architecture

- **Shared-memory zero-copy parallel engine**: All aggregated arrays are packed into a single `multiprocessing.SharedMemory` block. Worker processes access data via numpy views — no serialization, no copies.
- **Plugin pattern**: Aggregations and correlation methods are registered via decorators. Adding a new one = one file + one import.
- **Slice-based lag**: Instead of `pd.Series.shift()`, lags are computed via `agg[:n-lag]` / `target[lag:]` numpy slicing (zero-copy views).

---

# AlphaMiner (日本語)

S/N比が低い時系列データ（金融データ等）において、ターゲット列と特徴量のアグリゲーション（移動平均・移動相関等）のラグ相関を可視化するパッケージです。ACFプロットライクなsubplotグリッドで、特徴量設計のヒントを得られます。

## インストール

```bash
pip install -e .
```

## 使い方

```python
import pandas as pd, numpy as np
import alphaminer as am

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=n),
    "close": np.cumsum(np.random.randn(n)) + 100,
    "volume": np.abs(np.random.randn(n)) * 1000,
    "return_1d": np.random.randn(n) * 0.02,
})

result = am.explore(
    df,
    target_col="return_1d",
    time_col="date",
    feature_cols=["close", "volume"],
    agg=["MA_5", "STD_10", "RSI_14", "EMA_20", "ACF_1_30"],
    lags=list(range(11)),
    corr_method="pearson",  # "spearman", "chatterjee" も可
)

result.to_dataframe()  # DataFrameとして取得
result.plot()          # 再描画
```

## アグリゲーション引数フォーマット

| 形式 | 例 | 説明 |
|---|---|---|
| `NAME_WINDOW` | `MA_5`, `RSI_14` | ローリングウィンドウのみ |
| `NAME_EXTRA_WINDOW` | `ACF_3_50`, `FRACDIFF_5_20` | 追加パラメータが必要なアグリゲーション |

## 組み込みアグリゲーション (全44種)

### 基本ローリング (9)

| コード | 名前 | 計算式 |
|---|---|---|
| `MA` | 移動平均 | `rolling(W).mean()` |
| `SUM` | ローリング合計 | `rolling(W).sum()` |
| `MEDIAN` | ローリング中央値 | `rolling(W).median()` |
| `STD` | ローリング標準偏差 | `rolling(W).std()` |
| `VAR` | ローリング分散 | `rolling(W).var()` |
| `MAX` / `MIN` | ローリング最大/最小 | `rolling(W).max()` / `.min()` |
| `RANGE` | ローリングレンジ | `max - min` |
| `SKEW` / `KURT` | ローリング歪度/尖度 | `rolling(W).skew()` / `.kurt()` |

### ランク・正規化 (4): `RANK`, `ZSCORE`, `CV`, `NORMDEV`

### モメンタム (4): `MOM`, `ROC`, `MEANREV`, `TRENDSIG`

### EMAファミリー (5): `EMA`, `DEMA`, `TEMA`, `WMA`, `EWMSTD`

### テクニカル (4): `RSI`, `BPOS`, `RVOL`, `ARCH`

### 回帰 (2): `LSLOPE`, `LR2`

### 相関ベース (3): `ACF_lag_W`, `PACF_lag_W`, `MI_bins_W`

### エントロピー (5): `ENTROPY`, `SPECENT`, `SAMPEN_m_W`, `APEN_m_W`, `PERMEN_order_W`

### 複雑度 (3): `LZC`, `HURST`, `DFA`

### 分数・分位 (2): `FRACDIFF_d10_W`, `QUANTILE_q_W`

### ID (1): `RAW`

詳細は英語セクションの表を参照してください。

## 組み込み相関手法

| 名前 | 説明 |
|---|---|
| `pearson` | ピアソン積率相関 |
| `spearman` | スピアマン順位相関 |
| `chatterjee` | Chatterjeeのξ係数（非線形依存も検出） |

## アーキテクチャ

- **共有メモリ・ゼロコピー並列エンジン**: 全アグリゲーション結果を1つの `SharedMemory` ブロックにパック。ワーカープロセスはnumpyビューでアクセス — シリアライズもコピーも一切なし。
- **プラグインパターン**: デコレータで登録。新しいAgg/Corrの追加 = ファイル1つ + import 1行。
- **スライスベースlag**: `pd.Series.shift()` ではなく `agg[:n-lag]` / `target[lag:]` のnumpyスライス（ゼロコピービュー）で実装。
