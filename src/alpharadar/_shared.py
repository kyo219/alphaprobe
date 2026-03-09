"""Shared-memory utilities for zero-copy parallel computation."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np


@dataclass(frozen=True)
class SharedBlock:
    """Metadata for a packed shared memory block.

    Layout: contiguous row-major 2-D array of shape ``(n_arrays, n_rows)``
    stored in a single ``SharedMemory`` segment.  Each row is one 1-D
    float64 array (target or an aggregated feature series).
    """

    shm_name: str
    n_rows: int
    n_arrays: int
    dtype: str  # e.g. "float64"


def create_empty_shared_block(
    n_arrays: int, n_rows: int
) -> tuple[SharedMemory, SharedBlock, np.ndarray]:
    """Allocate a shared memory block without copying data.

    Returns ``(shm_handle, metadata, buf)`` where ``buf`` is a writable
    2-D numpy view of shape ``(n_arrays, n_rows)``.  The caller **must**
    call ``shm.close()`` and ``shm.unlink()`` when finished.
    """
    dtype = np.float64
    itemsize = np.dtype(dtype).itemsize
    shm = SharedMemory(create=True, size=n_arrays * n_rows * itemsize)
    buf = np.ndarray((n_arrays, n_rows), dtype=dtype, buffer=shm.buf)
    meta = SharedBlock(shm.name, n_rows, n_arrays, np.dtype(dtype).str)
    return shm, meta, buf


def create_shared_block(arrays: list[np.ndarray]) -> tuple[SharedMemory, SharedBlock]:
    """Pack equal-length 1-D arrays into a single shared memory segment.

    Returns ``(shm_handle, metadata)``.  The caller **must** call
    ``shm.close()`` and ``shm.unlink()`` when finished.
    """
    n_arrays = len(arrays)
    n_rows = len(arrays[0])
    dtype = np.float64
    itemsize = np.dtype(dtype).itemsize

    shm = SharedMemory(create=True, size=n_arrays * n_rows * itemsize)
    buf = np.ndarray((n_arrays, n_rows), dtype=dtype, buffer=shm.buf)
    for i, arr in enumerate(arrays):
        np.copyto(buf[i], arr)  # single memcpy per array

    return shm, SharedBlock(shm.name, n_rows, n_arrays, np.dtype(dtype).str)


def open_shared_block(meta: SharedBlock) -> tuple[SharedMemory, np.ndarray]:
    """Open an existing shared block and return a **zero-copy** 2-D view.

    Returns ``(shm_handle, buf)`` where ``buf.shape == (n_arrays, n_rows)``.
    The caller should call ``shm.close()`` when the view is no longer needed.
    """
    shm = SharedMemory(name=meta.shm_name, create=False)
    buf = np.ndarray(
        (meta.n_arrays, meta.n_rows),
        dtype=np.dtype(meta.dtype),
        buffer=shm.buf,
    )
    return shm, buf
