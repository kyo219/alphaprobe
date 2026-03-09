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
    agg=["MA_5", "MA_10", "MC_30"],
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
| `agg` | `list[str]` | Aggregation specs: `"NAME_WINDOW"` (e.g. `"MA_5"`, `"MC_30"`) |
| `lags` | `list[int]` | Lag values to compute |
| `corr_method` | `str` | `"pearson"`, `"spearman"`, or `"chatterjee"` |
| `max_workers` | `int \| None` | Process pool size (defaults to CPU count) |
| `show_progress` | `bool` | Show rich progress bar |

Returns an `ExploreResult` with `.plot()` and `.to_dataframe()` methods.

## Built-in Aggregations

| Code | Name | Description |
|---|---|---|
| `MA` | Moving Average | `series.rolling(window).mean()` |
| `MC` | Moving Correlation | `series.rolling(window).corr(target)` |

## Built-in Correlation Methods

| Name | Description |
|---|---|
| `pearson` | Pearson product-moment correlation |
| `spearman` | Spearman rank correlation |
| `chatterjee` | Chatterjee's xi coefficient (detects non-linear dependence) |

## Adding a Custom Aggregation

Create a file and use the `@register_aggregation` decorator:

```python
# src/alphaminer/aggregations/_ema.py
from alphaminer.aggregations._base import Aggregation, register_aggregation

@register_aggregation("EMA")
class ExponentialMovingAverage(Aggregation):
    def apply(self, series, window, *, target=None):
        return series.ewm(span=window).mean()
```

Then add the import in `aggregations/__init__.py`. Use it as `"EMA_10"`.

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
    agg=["MA_5", "MA_10", "MC_30"],
    lags=list(range(11)),
    corr_method="pearson",  # "spearman", "chatterjee" も可
)

result.to_dataframe()  # DataFrameとして取得
result.plot()          # 再描画
```

## 組み込みアグリゲーション

| コード | 名前 | 説明 |
|---|---|---|
| `MA` | 移動平均 | `series.rolling(window).mean()` |
| `MC` | 移動相関 | `series.rolling(window).corr(target)` |

## 組み込み相関手法

| 名前 | 説明 |
|---|---|
| `pearson` | ピアソン積率相関 |
| `spearman` | スピアマン順位相関 |
| `chatterjee` | Chatterjeeのξ係数（非線形依存も検出） |

## カスタムアグリゲーションの追加

`@register_aggregation` デコレータ付きクラスを1ファイル作成し、`__init__.py` にimportを追加するだけ:

```python
@register_aggregation("EMA")
class ExponentialMovingAverage(Aggregation):
    def apply(self, series, window, *, target=None):
        return series.ewm(span=window).mean()
```

## アーキテクチャ

- **共有メモリ・ゼロコピー並列エンジン**: 全アグリゲーション結果を1つの `SharedMemory` ブロックにパック。ワーカープロセスはnumpyビューでアクセス — シリアライズもコピーも一切なし。
- **プラグインパターン**: デコレータで登録。新しいAgg/Corrの追加 = ファイル1つ + import 1行。
- **スライスベースlag**: `pd.Series.shift()` ではなく `agg[:n-lag]` / `target[lag:]` のnumpyスライス（ゼロコピービュー）で実装。
