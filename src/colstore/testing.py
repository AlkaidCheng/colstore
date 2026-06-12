"""Synthetic data and store builders for tests, checks, and benchmarks.

A single, namespaced source of truth for the synthetic artifacts the project
(and downstream users) need when exercising colstore: parameterized column
sets and multi-record stores, sized by row count, column count, record count,
and dtype. Centralizing them keeps every test, check, and benchmark building
inputs the same way, and pairs directly with the canonical benchmark options
(``--rows`` / ``--cols`` / ``--record-counts`` / ``--dtype``) and with
:mod:`colstore.profiling`.

Generation is fully reproducible: the same ``(rows, cols, dtype, seed)`` always
yields identical data, so a correctness check can recover ground truth with a
second :func:`make_columns` call rather than reading it back::

    import colstore
    from colstore import testing

    expected = testing.make_columns(1000, 2, dtype="float32", seed=7)
    with testing.make_store(path, rows=1000, cols=2, dtype="float32", seed=7) as ds:
        assert (ds[:, "c0"].array() == expected["c0"]).all()

Value distributions are a stable part of the contract: float columns are drawn
from a standard normal, integer columns uniformly from the central half of the
dtype's range. Columns are named ``c0`` .. ``c{cols-1}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import api
from .reader import ColStoreReader

__all__ = [
    "make_columns",
    "make_store",
    "uniform_record_rows",
    "write_columns",
]


def _random_column(rng: np.random.Generator, rows: int, dt: np.dtype[Any]) -> NDArray[Any]:
    """A length-``rows`` column of ``dt`` with representative random values."""
    if dt.kind == "f":
        return rng.standard_normal(rows).astype(dt)
    if dt.kind in ("i", "u"):
        info = np.iinfo(dt)
        return rng.integers(info.min // 2, info.max // 2, size=rows, dtype=np.int64).astype(dt)
    raise ValueError(f"unsupported dtype {dt!r}: only float and integer columns are generated")


def _resolve_dtypes(dtype: str | Sequence[str], cols: int) -> list[np.dtype[Any]]:
    """Resolve ``dtype`` to one validated np.dtype per column, cycling a sequence."""
    specs = [dtype] if isinstance(dtype, str) else list(dtype)
    if not specs:
        raise ValueError("dtype sequence must be non-empty")
    resolved = [np.dtype(spec) for spec in specs]
    return [resolved[i % len(resolved)] for i in range(cols)]


def make_columns(
    rows: int,
    cols: int,
    *,
    dtype: str | Sequence[str] = "float64",
    seed: int = 0,
    names: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, NDArray[Any]]:
    """Build ``cols`` reproducible columns of ``rows`` rows each.

    ``dtype`` is a single dtype applied to every column, or a sequence cycled
    across the columns (column ``i`` uses ``dtype[i % len(dtype)]``), so mixed
    layouts like ``("f8", "f4", "i4", "i2")`` are a single call.

    Columns are named ``c0`` .. ``c{cols-1}`` unless ``names`` overrides them
    (which must have length ``cols`` and be unique). The data is drawn from
    ``rng`` when given -- so a caller can thread one generator across several
    calls -- otherwise from ``numpy.random.default_rng(seed)`` (``seed`` is
    ignored when ``rng`` is supplied).
    """
    if rows < 0:
        raise ValueError("rows must be >= 0")
    if cols < 1:
        raise ValueError("cols must be >= 1")
    if names is None:
        column_names = [f"c{i}" for i in range(cols)]
    else:
        column_names = list(names)
        if len(column_names) != cols:
            raise ValueError(f"names has {len(column_names)} entries, expected cols={cols}")
        if len(set(column_names)) != cols:
            raise ValueError("names must be unique")
    dtypes = _resolve_dtypes(dtype, cols)
    generator = rng if rng is not None else np.random.default_rng(seed)
    return {column_names[i]: _random_column(generator, rows, dtypes[i]) for i in range(cols)}


def uniform_record_rows(total: int, records: int) -> list[int]:
    """Split ``total`` rows into ``records`` near-equal record sizes.

    Every record gets ``total // records`` rows; the last absorbs the
    remainder. When ``records > total`` the surplus records are empty (a valid
    stress shape).
    """
    if records < 1:
        raise ValueError("records must be >= 1")
    if total < 0:
        raise ValueError("total must be >= 0")
    per = total // records
    rows = [per] * (records - 1)
    rows.append(total - per * (records - 1))
    return rows


def _resolve_records(records: int | Sequence[int], total: int) -> list[int]:
    """Resolve ``records`` to a per-record row-count list summing to ``total``."""
    if isinstance(records, int):
        return uniform_record_rows(total, records)
    rows_per_record = list(records)
    if any(n < 0 for n in rows_per_record):
        raise ValueError("record row counts must be >= 0")
    if sum(rows_per_record) != total:
        raise ValueError(f"records sum to {sum(rows_per_record)}, expected {total}")
    return rows_per_record


def write_columns(
    path: Path | str,
    columns: dict[str, NDArray[Any]],
    *,
    records: int | Sequence[int] = 1,
) -> ColStoreReader:
    """Write caller-provided ``columns`` as a multi-record store; return a reader.

    Unlike :func:`make_store` (which *generates* the data), this writes the
    arrays you pass -- use it for crafted or non-random layouts. All columns
    must share a length; ``records`` is a record *count* (rows split uniformly
    via :func:`uniform_record_rows`) or an explicit per-record row-count
    sequence summing to that length. The caller owns the returned reader.
    """
    if not columns:
        raise ValueError("columns must be non-empty")
    lengths = {len(column) for column in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"all columns must share a length; got {sorted(lengths)}")
    rows_per_record = _resolve_records(records, lengths.pop())
    target = Path(path)
    offset = 0
    with api.create(target) as writer:
        for n in rows_per_record:
            writer.write({name: column[offset : offset + n] for name, column in columns.items()})
            offset += n
    return api.open(target)


def make_store(
    path: Path | str,
    *,
    rows: int,
    cols: int = 1,
    records: int | Sequence[int] = 1,
    dtype: str | Sequence[str] = "float64",
    seed: int = 0,
) -> ColStoreReader:
    """Write a multi-record store of synthetic data and return an open reader.

    The columns are :func:`make_columns` ``(rows, cols, dtype, seed)``, written
    via :func:`write_columns`. ``records`` is either a record *count* (rows
    split uniformly via :func:`uniform_record_rows`) or an explicit sequence of
    per-record row counts summing to ``rows``. The caller owns the returned
    reader and should close it (``with testing.make_store(...) as ds:``).
    """
    columns = make_columns(rows, cols, dtype=dtype, seed=seed)
    return write_columns(path, columns, records=records)
