from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from alphaminer._types import ExploreResult


def plot_results(
    result: ExploreResult,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> None:
    """Render ACF-style stem-plot grid: rows=features, cols=aggs."""
    features = result.feature_cols
    aggs = result.agg_labels
    nrows = len(features)
    ncols = len(aggs)

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True
    )

    # Build lookup: (feature, agg) -> sorted list of (lag, corr)
    lookup: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for r in result.results:
        key = (r.feature, r.agg)
        lookup.setdefault(key, []).append((r.lag, r.correlation))

    for i, feat in enumerate(features):
        for j, agg_label in enumerate(aggs):
            ax = axes[i][j]
            data = lookup.get((feat, agg_label), [])
            data.sort(key=lambda t: t[0])
            lags_vals = [d[0] for d in data]
            corrs = [d[1] for d in data]

            # Stem plot (ACF style)
            ax.axhline(0, color="grey", linewidth=0.5)
            markerline, stemlines, baseline = ax.stem(
                lags_vals, corrs, linefmt="-", markerfmt="o", basefmt=" "
            )
            markerline.set_markersize(4)

            # lag 0 と 1 の間に区切り線を入れる
            if 0 in lags_vals and len(lags_vals) > 1:
                first_nonzero = min(l for l in lags_vals if l > 0)
                ax.axvline(
                    (0 + first_nonzero) / 2,
                    color="red", linewidth=0.8, linestyle="--", alpha=0.5,
                )

            ax.set_title(f"{feat} | {agg_label}", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel("Lag")
            if j == 0:
                ax.set_ylabel("Correlation")

    fig.suptitle(
        f"Lag-Correlation Grid (method={result.corr_method})",
        fontsize=12,
        fontweight="bold",
    )
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
