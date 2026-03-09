from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from alphascope._types import ExploreResult


def plot_results(
    result: ExploreResult,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> None:
    """Render bar-plot grid: rows=features, cols=aggs.

    lag=0 is shown as a gray dashed-edge bar on the left.
    lag>=1 bars are colored by value using RdBu_r.
    """
    features = result.feature_cols
    aggs = result.agg_labels
    nrows = len(features)
    ncols = len(aggs)

    # Build lookup: (feature, agg) -> sorted list of (lag, corr)
    lookup: dict[tuple[str, str], list[tuple[int, float]]] = {}
    y_max = 0.0
    for r in result.results:
        key = (r.feature, r.agg)
        lookup.setdefault(key, []).append((r.lag, r.correlation))
        y_max = max(y_max, abs(r.correlation))

    y_max = y_max or 1.0
    norm = Normalize(vmin=-y_max, vmax=y_max)
    cmap = plt.get_cmap("RdBu_r")

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True
    )

    for i, feat in enumerate(features):
        for j, agg_label in enumerate(aggs):
            ax = axes[i][j]
            data = lookup.get((feat, agg_label), [])
            data.sort(key=lambda t: t[0])

            for lag, corr in data:
                if lag == 0:
                    # lag=0: gray + dashed edge
                    ax.bar(
                        lag, corr, width=0.7,
                        color="lightgrey", edgecolor="grey",
                        linestyle="--", linewidth=1.0,
                    )
                else:
                    # lag>=1: colored by value
                    ax.bar(
                        lag, corr, width=0.7,
                        color=cmap(norm(corr)), edgecolor="none",
                    )

            ax.axhline(0, color="grey", linewidth=0.5)
            ax.set_ylim(-y_max, y_max)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            ax.set_title(f"{feat} | {agg_label}", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel("Lag")
            if j == 0:
                ax.set_ylabel("Correlation")

    fig.suptitle(
        f"Lag-Correlation Grid (method={result.corr_method})",
        fontsize=12, fontweight="bold",
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
