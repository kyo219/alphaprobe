# AlphaMiner

[![PyPI version](https://img.shields.io/pypi/v/alphaminer)](https://pypi.org/project/alphaminer/)
[![Python](https://img.shields.io/pypi/pyversions/alphaminer)](https://pypi.org/project/alphaminer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**低 S/N 時系列データのラグ相関を可視化し、有効な特徴量変換を発見するツール。**

ターゲット列（例: 将来リターン）と、44 種類の特徴量アグリゲーション（移動平均・RSI・エントロピー等）のラグ相関を一括計算し、bar plot グリッドで可視化します。共有メモリ・ゼロコピー並列エンジンにより、大量の組み合わせを高速に探索できます。

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
pip install alphaminer
```

開発用:

```bash
git clone https://github.com/kyo219/alphaminer.git
cd alphaminer
uv sync --all-extras
```

---

## Quick Start

```python
import numpy as np
import pandas as pd
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

### `am.explore()`

```python
am.explore(
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
  ├── Phase 1: Aggregation (ProcessPoolExecutor)
  │     各 (feature, agg) ペアを並列計算
  │
  └── Phase 2: Correlation (SharedMemory + ProcessPoolExecutor)
        結果を SharedMemory にパック → ゼロコピーでワーカーが参照
        numpy スライスでラグ計算（shift() 不使用）
```

- **共有メモリ・ゼロコピー**: シリアライズもコピーも一切なし
- **プラグインパターン**: `@register_aggregation` / `@register_correlation` デコレータで拡張可能
- **スライスベース lag**: `agg[:n-lag]` / `target[lag:]` の numpy ビュー

---

## Adding a Custom Aggregation

```python
from alphaminer.aggregations._base import Aggregation, register_aggregation

@register_aggregation("MYAGG")
class MyAggregation(Aggregation):
    def apply(self, series, window, *, target=None, extra=None):
        return series.rolling(window).mean()  # your logic
```

`aggregations/__init__.py` に import を追加すれば `"MYAGG_10"` として使えます。

---

## Build

```bash
# ビルド
uv build

# dist/ に以下が生成される
#   alphaminer-X.Y.Z.tar.gz
#   alphaminer-X.Y.Z-py3-none-any.whl
```

## Release (自動)

main ブランチへの push（PR マージ含む）で自動的にリリースされます:

1. GitHub Actions がテストを実行
2. テスト通過後、patch バージョンを自動インクリメント（例: `0.1.0` → `0.1.1`）
3. バージョンバンプを commit & tag して push
4. `uv build` → PyPI に Trusted Publishing で自動公開

### 初回セットアップ（1 回だけ必要）

1. [PyPI](https://pypi.org) でアカウント作成
2. **Publishing** → **Add a new pending publisher**:
   - Owner: `kyo219`
   - Repository: `alphaminer`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. GitHub リポジトリの **Settings** → **Environments** で `pypi` environment を作成

### 手動リリース（任意）

```bash
# バージョンを手動で上げる場合
# pyproject.toml と src/alphaminer/__init__.py の version を更新
git tag v0.2.0
git push origin main --tags
```

---

## Development

```bash
# セットアップ
git clone https://github.com/kyo219/alphaminer.git
cd alphaminer
uv sync --all-extras

# テスト
uv run pytest tests/ -v

# デモ実行（demo_output.png を生成）
uv run python demo.py
```

---

## License

[MIT](LICENSE)
