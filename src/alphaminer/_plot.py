from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from alphaminer._types import ExploreResult


def plot_results(
    result: ExploreResult,
    figsize: tuple[float, float] | None = None,
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
    plt.show()
