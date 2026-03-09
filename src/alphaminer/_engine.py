"""Parallel computation engine with shared-memory zero-copy architecture."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from alphaminer._parser import parse_agg
from alphaminer._shared import SharedBlock, create_shared_block, open_shared_block
from alphaminer._types import CorrResult, ExploreResult
from alphaminer.aggregations import get_aggregation
from alphaminer.correlations import get_correlation

# ── Per-worker globals (set once via ProcessPoolExecutor initializer) ───
_w_shm = None
_w_buf: np.ndarray | None = None


def _init_worker(meta: SharedBlock) -> None:
    """Open shared memory once per worker process."""
    global _w_shm, _w_buf
    _w_shm, _w_buf = open_shared_block(meta)


def _compute_agg(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    agg_name: str,
    window: int,
    extra: int | None,
) -> np.ndarray:
    """Worker function for parallel Phase 1 aggregation."""
    series = pd.Series(feature_values)
    target = pd.Series(target_values)
    agg_impl = get_aggregation(agg_name)
    result = agg_impl.apply(series, window, target=target, extra=extra)
    return result.to_numpy(dtype=np.float64)


def _compute_group(
    feat: str,
    agg_label: str,
    agg_index: int,
    lags: list[int],
    n: int,
    corr_method: str,
) -> list[CorrResult]:
    """Compute correlations for one (feature, agg) pair across all lags.

    All data is read from the process-global shared-memory view — zero-copy.
    Lag is implemented via array slicing (views, not copies).
    """
    target = _w_buf[0]  # row 0 = target; zero-copy view
    agg = _w_buf[agg_index]  # zero-copy view
    corr_fn = get_correlation(corr_method)

    results: list[CorrResult] = []
    for lag in lags:
        if 0 < lag < n:
            # corr(agg[t-lag], target[t]) == corr(agg[:n-lag], target[lag:])
            x_raw = agg[: n - lag]  # view — zero-copy
            y_raw = target[lag:]  # view — zero-copy
        elif lag == 0:
            x_raw = agg
            y_raw = target
        else:
            results.append(CorrResult(feat, agg_label, lag, float("nan")))
            continue

        # NaN mask + fancy-index are the only allocations (unavoidable, small)
        mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        x = x_raw[mask]
        y = y_raw[mask]
        corr = corr_fn.compute(x, y) if len(x) > 1 else float("nan")
        results.append(CorrResult(feat, agg_label, lag, corr))

    return results


# ── Public API ──────────────────────────────────────────────────────────


def explore(
    df: pd.DataFrame,
    *,
    target_col: str,
    time_col: str,
    feature_cols: list[str],
    agg: list[str],
    lags: list[int],
    corr_method: str = "pearson",
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ExploreResult:
    """Compute lag-correlations between target and aggregated features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing *target_col*, *time_col*, and all *feature_cols*.
    target_col, time_col : str
        Column names for target and time axis.
    feature_cols : list[str]
        Feature column names.
    agg : list[str]
        Aggregation specs (e.g. ``["MA_5", "MC_30"]``).
    lags : list[int]
        Lag values to evaluate.
    corr_method : str
        ``"pearson"`` | ``"spearman"`` | ``"chatterjee"``.
    max_workers : int | None
        Process pool size. Defaults to ``min(cpu_count, n_groups)``.
    show_progress : bool
        Show a ``rich`` progress bar.

    Returns
    -------
    ExploreResult
        Call ``.plot()`` to visualise or ``.to_dataframe()`` to export.
    """
    # ── Validate ────────────────────────────────────────────────────
    for col in [target_col, time_col, *feature_cols]:
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found in DataFrame")

    # ── Sort & extract target ───────────────────────────────────────
    df = df.sort_values(time_col).reset_index(drop=True)
    target_series = df[target_col]
    target_np = target_series.to_numpy(dtype=np.float64)
    n = len(target_np)

    # ── Phase 1: Aggregation (parallel) ─────────────────────────────
    agg_specs = [parse_agg(a) for a in agg]

    # Pre-assign indices so order is deterministic
    tasks: list[tuple[str, str, int, str, int, int | None]] = []
    index_map: dict[tuple[str, str], int] = {}
    idx = 1
    for feat in feature_cols:
        for spec in agg_specs:
            spec_str = str(spec)
            index_map[(feat, spec_str)] = idx
            tasks.append((feat, spec_str, idx, spec.name, spec.window, spec.extra))
            idx += 1

    n_agg_tasks = len(tasks)
    n_workers = max_workers if max_workers is not None else min(os.cpu_count() or 4, n_agg_tasks)
    arrays: list[np.ndarray | None] = [target_np] + [None] * n_agg_tasks

    if show_progress:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            tid = progress.add_task("Aggregating", total=n_agg_tasks)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _compute_agg,
                        df[feat].to_numpy(dtype=np.float64),
                        target_np,
                        agg_name,
                        window,
                        extra,
                    ): task_idx
                    for feat, _spec_str, task_idx, agg_name, window, extra in tasks
                }
                for fut in as_completed(futures):
                    task_idx = futures[fut]
                    arrays[task_idx] = fut.result()
                    progress.advance(tid)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _compute_agg,
                    df[feat].to_numpy(dtype=np.float64),
                    target_np,
                    agg_name,
                    window,
                    extra,
                ): task_idx
                for feat, _spec_str, task_idx, agg_name, window, extra in tasks
            }
            for fut in as_completed(futures):
                task_idx = futures[fut]
                arrays[task_idx] = fut.result()

    # ── Phase 2: Pack into shared memory & correlate in parallel ────
    shm, meta = create_shared_block(arrays)  # type: ignore[arg-type]
    del arrays  # free temporary list; data lives in shm now

    agg_labels = [str(s) for s in agg_specs]
    try:
        results = _run_pool(
            feature_cols,
            agg_labels,
            index_map,
            lags,
            n,
            corr_method,
            meta,
            max_workers,
            show_progress,
        )
    finally:
        shm.close()
        shm.unlink()

    result = ExploreResult(results, feature_cols, agg_labels, corr_method)
    result.plot()
    return result


# ── Internal parallel runner ────────────────────────────────────────────


def _run_pool(
    feature_cols: list[str],
    agg_labels: list[str],
    index_map: dict[tuple[str, str], int],
    lags: list[int],
    n: int,
    corr_method: str,
    meta: SharedBlock,
    max_workers: int | None,
    show_progress: bool,
) -> list[CorrResult]:
    """Submit (feature, agg) groups to a process pool.

    Each group computes all lags in a single worker call, minimising
    scheduling overhead and maximising cache locality.
    """
    groups = [
        (feat, agg_label, index_map[(feat, agg_label)])
        for feat in feature_cols
        for agg_label in agg_labels
    ]
    n_groups = len(groups)
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, n_groups)

    pool_kw = dict(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(meta,),
    )
    results: list[CorrResult] = []

    if not show_progress:
        with ProcessPoolExecutor(**pool_kw) as pool:
            futs = {
                pool.submit(
                    _compute_group, feat, agg_label, agg_idx, lags, n, corr_method
                ): None
                for feat, agg_label, agg_idx in groups
            }
            for fut in as_completed(futs):
                results.extend(fut.result())
        return results

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        tid = progress.add_task("Computing correlations", total=n_groups)
        with ProcessPoolExecutor(**pool_kw) as pool:
            futs = {
                pool.submit(
                    _compute_group, feat, agg_label, agg_idx, lags, n, corr_method
                ): None
                for feat, agg_label, agg_idx in groups
            }
            for fut in as_completed(futs):
                results.extend(fut.result())
                progress.advance(tid)
    return results
