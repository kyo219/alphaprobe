# AlphaProbe

[![PyPI version](https://img.shields.io/pypi/v/alphaprobe)](https://pypi.org/project/alphaprobe/)
[![Python](https://img.shields.io/pypi/pyversions/alphaprobe)](https://pypi.org/project/alphaprobe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**時系列の特徴量エンジニアリングを加速するラグ相関探索ライブラリ。**

44 種類の組み込みアグリゲーション（移動平均・RSI・エントロピー・フラクタル次元など）をターゲット変数に対してラグ付きで相関分析し、どの特徴量変換が予測に効くかを一目で把握できます。共有メモリ・ゼロコピー並列エンジンで大規模な探索も高速に実行します。

---

## Output Example

`demo.py` の実行結果（3 features × 45 aggs × 11 lags）:

![demo_output](demo_output.png)

- **灰色 + 点線バー**: lag=0（同時相関）
- **色付きバー**: lag≥1、`RdBu_r` カラーマップ（正→赤、負→青）
- **Y 軸**: 全 subplot で統一スケール

---

## Install

```bash
pip install alphaprobe
```

開発用:

```bash
git clone https://github.com/kyo219/alphaprobe.git
cd alphaprobe
uv sync --all-extras
```

---

## Quick Start

```python
import numpy as np
import pandas as pd
import alphaprobe as ap

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=n),
    "close": np.cumsum(np.random.randn(n)) + 100,
    "volume": np.abs(np.random.randn(n)) * 1000,
    "return_1d": np.random.randn(n) * 0.02,
})

result = ap.explore(
    df,
    target_col="return_1d",
    time_col="date",
    feature_cols=["close", "volume"],
    agg=["MA_5", "STD_10", "RSI_14", "EMA_20", "ACF_1_30"],
    lags=list(range(11)),
    corr_method="pearson",
)

# DataFrame として取得
df_out = result.to_dataframe()
print(df_out.head())
#   feature    agg  lag  correlation
# 0   close   MA_5    0    -0.012345
# 1   close   MA_5    1    -0.008901
# ...

# PNG 保存
result.plot(save_path="output.png")
```

---

## API

### `ap.explore()`

```python
ap.explore(
    df,
    *,
    target_col: str,
    time_col: str,
    feature_cols: list[str],
    agg: list[str],
    lags: list[int],
    corr_method: str = "pearson",
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ExploreResult
```

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | 入力データ |
| `target_col` | `str` | ターゲット列（例: 1 期先リターン） |
| `time_col` | `str` | 時間軸列（ソートに使用） |
| `feature_cols` | `list[str]` | 探索する特徴量列 |
| `agg` | `list[str]` | アグリゲーション指定（後述） |
| `lags` | `list[int]` | 計算するラグ値のリスト |
| `corr_method` | `str` | `"pearson"` / `"spearman"` / `"chatterjee"` |
| `max_workers` | `int \| None` | 並列ワーカー数（デフォルト: CPU 数） |
| `show_progress` | `bool` | Rich プログレスバー表示 |

### `ExploreResult`

| Method | Description |
|---|---|
| `.to_dataframe()` | `feature`, `agg`, `lag`, `correlation` 列の tidy DataFrame を返す |
| `.plot(figsize=None, save_path=None)` | bar plot グリッドを描画。`save_path` 指定で PNG 保存 |

---

## Aggregation Spec Format

2 つのフォーマットをサポート:

| Format | Example | Description |
|---|---|---|
| `NAME_WINDOW` | `MA_5`, `RSI_14` | ローリングウィンドウのみ |
| `NAME_EXTRA_WINDOW` | `ACF_3_50`, `FRACDIFF_5_20` | 追加パラメータが必要 |

---

## Built-in Aggregations (44 types)

### Basic Rolling (10)

| Code | Name | Formula |
|---|---|---|
| `RAW` | Raw (no-op) | 入力そのまま |
| `MA` | Moving Average | `rolling(W).mean()` |
| `SUM` | Rolling Sum | `rolling(W).sum()` |
| `MEDIAN` | Rolling Median | `rolling(W).median()` |
| `STD` | Rolling Std Dev | `rolling(W).std()` |
| `VAR` | Rolling Variance | `rolling(W).var()` |
| `MAX` | Rolling Max | `rolling(W).max()` |
| `MIN` | Rolling Min | `rolling(W).min()` |
| `RANGE` | Rolling Range | `max - min` |
| `SKEW` | Rolling Skewness | `rolling(W).skew()` |
| `KURT` | Rolling Kurtosis | `rolling(W).kurt()` |

### Rank & Normalisation (4)

| Code | Name | Formula |
|---|---|---|
| `RANK` | Rolling Rank | ウィンドウ内パーセンタイル順位 |
| `ZSCORE` | Rolling Z-Score | `(x - mean) / std` |
| `CV` | Coefficient of Variation | `std / mean` |
| `NORMDEV` | Normality Deviation | `|skew| + |kurt - 3|` |

### Momentum (4)

| Code | Name | Formula |
|---|---|---|
| `MOM` | Momentum | `x[t] - x[t-W]` |
| `ROC` | Rate of Change | `(x[t] / x[t-W] - 1) * 100` |
| `MEANREV` | Mean Reversion | 偏差の負の自己相関 |
| `TRENDSIG` | Trend Signal | 線形回帰スロープの T 統計量 |

### EMA Family (5)

| Code | Name | Formula |
|---|---|---|
| `EMA` | Exponential MA | `ewm(span=W).mean()` |
| `DEMA` | Double EMA | `2*EMA - EMA(EMA)` |
| `TEMA` | Triple EMA | `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))` |
| `WMA` | Weighted MA | 線形加重移動平均 |
| `EWMSTD` | EWM Std Dev | `ewm(span=W).std()` |

### Technical (4)

| Code | Name | Formula |
|---|---|---|
| `RSI` | Relative Strength Index | Gain/Loss 比率（EWM 平滑化） |
| `BPOS` | Bollinger Position | `(x - MA) / (2 * STD)` |
| `RVOL` | Realised Volatility | `diff().rolling(W).std()` |
| `ARCH` | ARCH Effect | `(diff()²).rolling(W).mean()` |

### Regression (2)

| Code | Name | Formula |
|---|---|---|
| `LSLOPE` | Linear Slope | ベクトル化ローリング OLS スロープ |
| `LR2` | Linear R² | ローリング決定係数 |

### Correlation-based (4)

| Code | Name | Extra | Example |
|---|---|---|---|
| `MC` | Moving Correlation | — | `MC_30`（target 必要） |
| `ACF` | Autocorrelation | lag | `ACF_3_50` → lag=3, window=50 |
| `PACF` | Partial Autocorrelation | lag | `PACF_5_50` → lag=5, window=50 |
| `MI` | Mutual Information | bins | `MI_10_50` → 10 bins, window=50（target 必要） |

### Entropy (5)

| Code | Name | Extra | Example |
|---|---|---|---|
| `ENTROPY` | Shannon Entropy | — | `ENTROPY_50` |
| `SPECENT` | Spectral Entropy | — | `SPECENT_32` |
| `SAMPEN` | Sample Entropy | m (embed dim) | `SAMPEN_2_50` |
| `APEN` | Approximate Entropy | m (embed dim) | `APEN_2_50` |
| `PERMEN` | Permutation Entropy | order | `PERMEN_3_50` |

### Complexity (3)

| Code | Name | Formula |
|---|---|---|
| `LZC` | Lempel-Ziv Complexity | バイナリ列の LZ76 複雑度 |
| `HURST` | Hurst Exponent | R/S 解析 |
| `DFA` | Detrended Fluctuation Analysis | DFA スケーリング指数 |

### Fractional & Quantile (2)

| Code | Name | Extra | Example |
|---|---|---|---|
| `FRACDIFF` | Fractional Differencing | d×10 | `FRACDIFF_5_20` → d=0.5, window=20 |
| `QUANTILE` | Rolling Quantile | percentile | `QUANTILE_25_50` → 25th %ile, window=50 |

---

## Correlation Methods

| Name | Description |
|---|---|
| `pearson` | ピアソン積率相関 |
| `spearman` | スピアマン順位相関 |
| `chatterjee` | Chatterjee の ξ 係数（非線形依存も検出） |

---

## Architecture

```
explore()
  │
  ├── Phase 1: Aggregation (SharedMemory + ProcessPoolExecutor)
  │     入力を共有メモリにパック → ワーカーが読み取り → 結果を共有メモリに直接書き込み
  │
  └── Phase 2: Correlation (SharedMemory + ProcessPoolExecutor)
        Phase 1 の出力共有メモリをそのまま再利用 → ゼロコピーでラグ相関計算
```

- **全フェーズ共有メモリ**: Phase 1 / Phase 2 ともに pickle シリアライズなし
- **プラグインパターン**: `@register_aggregation` / `@register_correlation` デコレータで拡張可能
- **スライスベース lag**: `agg[:n-lag]` / `target[lag:]` の numpy ビュー

---

## Adding a Custom Aggregation

```python
from alphaprobe.aggregations._base import Aggregation, register_aggregation

@register_aggregation("MYAGG")
class MyAggregation(Aggregation):
    def apply(self, series, window, *, target=None, extra=None):
        return series.rolling(window).mean()  # your logic
```

`aggregations/__init__.py` に import を追加すれば `"MYAGG_10"` として使えます。

---

## Release

main ブランチへの push で自動的にリリースされます:

1. GitHub Actions がテストを実行
2. テスト通過後、patch バージョンを自動インクリメント（例: `0.1.3` → `0.1.4`）
3. バージョンバンプを commit & tag して push
4. `uv build` → PyPI に Trusted Publishing で自動公開

デプロイせずに push したい場合は、コミットメッセージに `[skip deploy]` を含めてください。テストのみ実行されます。

---

## Development

```bash
# セットアップ
git clone https://github.com/kyo219/alphaprobe.git
cd alphaprobe
uv sync --all-extras

# テスト
uv run pytest tests/ -v

# デモ実行（demo_output.png を生成）
uv run python demo.py
```

---

## License

[MIT](LICENSE)
