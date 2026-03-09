"""Parallel computation engine with shared-memory zero-copy architecture.

Both Phase 1 (aggregation) and Phase 2 (correlation) use shared memory
to avoid pickle serialisation of large arrays across process boundaries.

Phase 1 layout:
  input  shm: [target, feature_1, feature_2, ...]   (read-only by workers)
  output shm: [target, agg_result_1, agg_result_2, ...]  (written by workers)

Phase 2 reuses the output shm from Phase 1 directly — no extra copy.
"""

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
from alphaminer._shared import (
    SharedBlock,
    create_empty_shared_block,
    create_shared_block,
    open_shared_block,
)
from alphaminer._types import CorrResult, ExploreResult
from alphaminer.aggregations import get_aggregation
from alphaminer.correlations import get_correlation

# ── Per-worker globals (set once via ProcessPoolExecutor initializer) ───
_w_shm_in = None
_w_buf_in: np.ndarray | None = None
_w_shm_out = None
_w_buf_out: np.ndarray | None = None


def _init_agg_worker(input_meta: SharedBlock, output_meta: SharedBlock) -> None:
    """Open input & output shared memory for Phase 1 aggregation workers."""
    global _w_shm_in, _w_buf_in, _w_shm_out, _w_buf_out
    _w_shm_in, _w_buf_in = open_shared_block(input_meta)
    _w_shm_out, _w_buf_out = open_shared_block(output_meta)


def _init_corr_worker(meta: SharedBlock) -> None:
    """Open shared memory for Phase 2 correlation workers."""
    global _w_shm_in, _w_buf_in
    _w_shm_in, _w_buf_in = open_shared_block(meta)


def _compute_agg(
    feat_row: int,
    output_row: int,
    agg_name: str,
    window: int,
    extra: int | None,
) -> None:
    """Worker: read from input shm, compute aggregation, write to output shm.

    Input arrays are copied once for pandas safety (rolling may create
    internal views).  Result is written directly to the output shared
    memory block — no pickle return.
    """
    series = pd.Series(_w_buf_in[feat_row].copy())
    target = pd.Series(_w_buf_in[0].copy())
    agg_impl = get_aggregation(agg_name)
    result = agg_impl.apply(series, window, target=target, extra=extra)
    np.copyto(_w_buf_out[output_row], result.to_numpy(dtype=np.float64))


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
    target = _w_buf_in[0]  # row 0 = target; zero-copy view
    agg = _w_buf_in[agg_index]  # zero-copy view
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
        Process pool size. Defaults to ``min(cpu_count, n_tasks)``.
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
    target_np = df[target_col].to_numpy(dtype=np.float64)
    n = len(target_np)

    # ── Parse agg specs ─────────────────────────────────────────────
    agg_specs = [parse_agg(a) for a in agg]
    agg_labels = [str(s) for s in agg_specs]

    # index_map: (feature, agg_label) → output row (1-based; row 0 = target)
    index_map: dict[tuple[str, str], int] = {}
    idx = 1
    for feat in feature_cols:
        for label in agg_labels:
            index_map[(feat, label)] = idx
            idx += 1
    n_agg_tasks = len(index_map)

    # ── Phase 1 input shm: [target, feat_1, feat_2, ...] ───────────
    input_arrays = [target_np] + [
        df[feat].to_numpy(dtype=np.float64) for feat in feature_cols
    ]
    input_shm, input_meta = create_shared_block(input_arrays)
    del input_arrays

    # feat_name → row in input shm
    feat_row = {feat: i + 1 for i, feat in enumerate(feature_cols)}

    # ── Output shm: [target, agg_1, agg_2, ...] (Phase 1 writes, Phase 2 reads)
    n_total = 1 + n_agg_tasks
    output_shm, output_meta, output_buf = create_empty_shared_block(n_total, n)
    np.copyto(output_buf[0], target_np)  # row 0 = target

    # ── Build agg task list: (input_feat_row, output_row, name, window, extra)
    agg_tasks = [
        (feat_row[feat], index_map[(feat, label)], spec.name, spec.window, spec.extra)
        for feat in feature_cols
        for spec, label in zip(agg_specs, agg_labels)
    ]

    # ── Phase 1: Aggregation (shm → shm, parallel) ─────────────────
    try:
        try:
            _run_agg_pool(
                agg_tasks, input_meta, output_meta, max_workers, show_progress,
            )
        finally:
            input_shm.close()
            input_shm.unlink()

        # ── Phase 2: Correlation (from output shm, parallel) ────────
        results = _run_corr_pool(
            feature_cols, agg_labels, index_map, lags, n,
            corr_method, output_meta, max_workers, show_progress,
        )
    finally:
        output_shm.close()
        output_shm.unlink()

    result = ExploreResult(results, feature_cols, agg_labels, corr_method)
    result.plot()
    return result


# ── Internal pool runners ──────────────────────────────────────────────


def _run_agg_pool(
    tasks: list[tuple[int, int, str, int, int | None]],
    input_meta: SharedBlock,
    output_meta: SharedBlock,
    max_workers: int | None,
    show_progress: bool,
) -> None:
    """Run Phase 1 aggregation tasks in a process pool."""
    n_tasks = len(tasks)
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, n_tasks)

    pool_kw = dict(
        max_workers=max_workers,
        initializer=_init_agg_worker,
        initargs=(input_meta, output_meta),
    )

    if show_progress:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            tid = progress.add_task("Aggregating", total=n_tasks)
            with ProcessPoolExecutor(**pool_kw) as pool:
                futs = [
                    pool.submit(_compute_agg, fr, out, name, win, ext)
                    for fr, out, name, win, ext in tasks
                ]
                for fut in as_completed(futs):
                    fut.result()  # propagate exceptions
                    progress.advance(tid)
    else:
        with ProcessPoolExecutor(**pool_kw) as pool:
            futs = [
                pool.submit(_compute_agg, fr, out, name, win, ext)
                for fr, out, name, win, ext in tasks
            ]
            for fut in as_completed(futs):
                fut.result()


def _run_corr_pool(
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
    """Run Phase 2 correlation tasks in a process pool."""
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
        initializer=_init_corr_worker,
        initargs=(meta,),
    )
    results: list[CorrResult] = []

    if show_progress:
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
    else:
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
