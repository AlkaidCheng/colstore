"""Shared builders and spies for the multi-record kernel and routing tests.

Consolidates the ``_layout`` / ``_write_store`` / ``_spy`` helpers that grew
near-identical copies across the per-feature test files. Builders return
self-consistent layouts (tests compare kernel output against the returned
``column``), so exact random values are irrelevant as long as each call is
reproducible from its seed.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

import colstore


class Layout(NamedTuple):
    """Synthetic single-column multi-record byte layout (irregular records).

    ``prefix`` is the per-row byte count of simulated preceding columns;
    it is the ``col_prefix_bytes`` argument the kernels expect.
    """

    buf: np.ndarray
    column: np.ndarray
    rsr: np.ndarray
    rsb: np.ndarray
    nrr: np.ndarray
    prefix: int
    total: int


class UniformLayout(NamedTuple):
    """Like :class:`Layout`, but with the constant body ``stride`` the
    uniform kernels take; the final record may be partial."""

    buf: np.ndarray
    column: np.ndarray
    rsr: np.ndarray
    rsb: np.ndarray
    nrr: np.ndarray
    stride: int
    prefix: int
    total: int


def _random_column(rng: np.random.Generator, total: int, dtype) -> np.ndarray:
    if np.issubdtype(np.dtype(dtype), np.floating):
        return rng.standard_normal(total).astype(dtype)
    return rng.integers(-100, 100, total).astype(dtype)


def build_layout(rows_per_record, dtype, col_prefix_rows: int = 0, seed: int = 0) -> Layout:
    """Build an irregular multi-record layout with packed record bodies."""
    rng = np.random.default_rng(seed)
    itemsize = np.dtype(dtype).itemsize
    nrr = np.asarray(rows_per_record, dtype=np.int64)
    n_records = nrr.shape[0]
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    body = nrr * (col_prefix_rows + itemsize)
    rsb = np.zeros(n_records, dtype=np.int64)
    rsb[1:] = np.cumsum(body)[:-1]
    total = int(rsr[-1])
    column = _random_column(rng, total, dtype)
    buf = np.zeros(int(body.sum()), dtype=np.uint8)
    for r in range(n_records):
        off = int(rsb[r]) + col_prefix_rows * int(nrr[r])
        rows = column[int(rsr[r]) : int(rsr[r + 1])]
        buf[off : off + rows.nbytes] = rows.view(np.uint8)
    return Layout(buf, column, rsr, rsb, nrr, col_prefix_rows, total)


def build_uniform_layout(
    n_records: int,
    rows: int,
    dtype,
    last_rows: int | None = None,
    col_prefix_rows: int = 0,
    seed: int = 0,
) -> UniformLayout:
    """Build a uniform layout (optional partial tail) at a constant stride.

    The stride is computed from FULL records, as the packed format implies
    for equal row counts.
    """
    rng = np.random.default_rng(seed)
    itemsize = np.dtype(dtype).itemsize
    last = rows if last_rows is None else last_rows
    per_record_rows = [rows] * (n_records - 1) + [last]
    total = sum(per_record_rows)
    column = _random_column(rng, total, dtype)
    stride = rows * (col_prefix_rows + itemsize)
    rsb = np.arange(n_records, dtype=np.int64) * stride
    buf = np.zeros(int(rsb[-1]) + last * (col_prefix_rows + itemsize), dtype=np.uint8)
    nrr = np.asarray(per_record_rows, dtype=np.int64)
    rsr = np.zeros(n_records + 1, dtype=np.int64)
    rsr[1:] = np.cumsum(nrr)
    start_row = 0
    for r, rec_rows in enumerate(per_record_rows):
        off = int(rsb[r]) + col_prefix_rows * rec_rows
        chunk = column[start_row : start_row + rec_rows]
        buf[off : off + chunk.nbytes] = chunk.view(np.uint8)
        start_row += rec_rows
    return UniformLayout(buf, column, rsr, rsb, nrr, stride, col_prefix_rows, total)


def standard_columns(total: int, seed: int) -> dict[str, np.ndarray]:
    """The f8/f4/i2 column trio used by the routing stores."""
    rng = np.random.default_rng(seed)
    return {
        "f8": rng.standard_normal(total),
        "f4": rng.standard_normal(total).astype(np.float32),
        "i2": rng.integers(-(2**14), 2**14, total).astype(np.int16),
    }


def write_records(path: Path, columns: dict[str, np.ndarray], rows_per_record) -> None:
    """Stream ``columns`` into ``path`` as one record per entry of
    ``rows_per_record`` (slicing each column in order)."""
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in columns.items()})
            offset += rows


def write_standard_store(tmp_path, rows_per_record, seed: int, name: str = "store"):
    """Write a multi-record store of :func:`standard_columns`.

    Returns ``(path, full_columns, total_rows)``.
    """
    total = sum(rows_per_record)
    full = standard_columns(total, seed)
    path = tmp_path / f"{name}.cstore"
    write_records(path, full, rows_per_record)
    return path, full, total


def kernel_spy(monkeypatch, names) -> list[str]:
    """Record the names of spied ``colstore._gather`` entries, in call order.

    The wrapped kernels still run; routing tests assert on the recorded
    sequence (these are pinned routing contracts).
    """
    from colstore import _gather

    calls: list[str] = []
    for name in names:
        original = getattr(_gather, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_gather, name, wrapper)
    return calls


@contextlib.contextmanager
def opened(path):
    """Open a store and guarantee it is closed (replaces try/finally close)."""
    reader = colstore.open(path)
    try:
        yield reader
    finally:
        reader.close()
