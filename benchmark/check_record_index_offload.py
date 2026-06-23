"""A/B the vectorized record-index build against the per-record header walk at open.

``read_record_index`` builds the per-record index -- cumulative rows, body byte
offsets, and per-record row counts -- that the reader needs before any data read.
The baseline walks all ``R`` 32-byte headers one at a time in Python
(seek/read/unpack/crc per record): the one data-scaling Python-interpreter loop
in the open path, ~linear in record count. The fast path
(``format._VECTORIZED_RECORD_INDEX``) reads every header in a single strided pass
over a read-only mmap and builds the three int64 arrays with NumPy, keeping only
the CRC check per-record. It assumes a uniform record stride (the stride of
record 0) and falls back to the walk for any non-uniform or corrupt file.

Two layouts, both of which the fast path handles because records 0..R-2 share a
size: a *uniform* store (near-equal records, the last absorbing the remainder)
and a *partial-final* store (full fixed-size records with a smaller last record,
a streaming writer's common shape). The cost scales with record *count*, so the
store is built with many small records; ``--records`` is the cost driver.

The correctness gate asserts the vectorized output is byte-identical to the walk
(all three arrays) and that the fast path actually engaged, before any timing.
The speedup is Python-interpreter self-time (hardware-independent), so a local
A/B is indicative; confirm on the deployment node with ``--tmpdir`` on the
parallel filesystem.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import _common as _c
import numpy as np

import colstore
import colstore.format as _fmt
from colstore import testing

_ARRAY_NAMES = ("record_starts_rows", "record_starts_bytes", "n_rows_per_record")


@dataclass(frozen=True)
class Shape:
    key: str
    label: str
    # (rows, n_records) -> (total_rows, records-arg for write_columns)
    layout: Callable[[int, int], tuple[int, int | list[int]]]


def _uniform(rows: int, n_records: int) -> tuple[int, int]:
    """Near-equal records; the last absorbs the remainder (``uniform_record_rows``)."""
    return rows, min(n_records, max(1, rows))


def _partial_final(rows: int, n_records: int) -> tuple[int, list[int]]:
    """``n_records`` fixed-size records with a strictly smaller final record."""
    n = min(n_records, max(2, rows))
    full = max(2, rows // n)
    last = max(1, full // 2)
    return full * (n - 1) + last, [full] * (n - 1) + [last]


_SHAPES = (
    Shape("uniform", "uniform records (last absorbs remainder)", _uniform),
    Shape("partial_final", "fixed records + smaller final (partial last batch)", _partial_final),
)


def _build(path: Path, total_rows: int, records: int | list[int], cols: int, dtype: str) -> None:
    path.unlink(missing_ok=True)  # --tmpdir persists between runs, unlike a TemporaryDirectory
    columns = testing.make_columns(total_rows, cols, dtype=dtype, seed=0)
    testing.write_columns(path, columns, records=records).close()


def _index_args(path: Path) -> tuple[int, int, list[int]]:
    """Recover the ``read_record_index`` arguments the reader computes at open."""
    reader = colstore.open(path)
    try:
        record_starts_bytes = np.asarray(reader._record_starts_bytes)
        data_offset = int(record_starts_bytes[0]) - _fmt._RECORD_HEADER_SIZE
        n_records = len(np.asarray(reader._n_rows_per_record))
        itemsizes = [np.dtype(dt).itemsize for dt in reader.dtypes.values()]
        return data_offset, n_records, itemsizes
    finally:
        reader.close()


def _open_with(path: Path, vectorized: bool) -> None:
    _fmt._VECTORIZED_RECORD_INDEX = vectorized
    colstore.open(path).close()


def _index_with(
    path: Path, data_offset: int, n_records: int, itemsizes: list[int], vectorized: bool
) -> None:
    _fmt._VECTORIZED_RECORD_INDEX = vectorized
    _fmt.read_record_index(path, data_offset, n_records, itemsizes)


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    for shape in _SHAPES:
        total_rows, records = shape.layout(args.rows, args.records)
        path = directory / f"correctness_{shape.key}.cstore"
        _build(path, total_rows, records, args.cols, args.dtype)
        data_offset, n_records, itemsizes = _index_args(path)
        _fmt._VECTORIZED_RECORD_INDEX = False
        walk = _fmt.read_record_index(path, data_offset, n_records, itemsizes)
        _fmt._VECTORIZED_RECORD_INDEX = True
        vectorized = _fmt.read_record_index(path, data_offset, n_records, itemsizes)
        engaged = (
            _fmt._read_record_index_uniform(path, data_offset, n_records, itemsizes) is not None
        )
        _fmt._VECTORIZED_RECORD_INDEX = False
        for got, expected, name in zip(vectorized, walk, _ARRAY_NAMES, strict=True):
            _c.check_equal(got, expected, f"{shape.key}:{name} vectorized vs walk")
        if not engaged:
            raise AssertionError(f"{shape.key}: fast path did not engage on a uniform layout")
        path.unlink(missing_ok=True)
    print("  CORRECTNESS OK (vectorized == walk byte-for-byte; fast path engaged)\n")


def run_bench(directory: Path, args: argparse.Namespace) -> None:
    print("Environment:")
    print(f"  rows={args.rows:,}  records={args.records:,}  cols={args.cols}  dtype={args.dtype}")
    print(f"  repeat={args.repeat}  warmup={args.warmup}  store dir={directory}\n")
    for shape in _SHAPES:
        total_rows, records = shape.layout(args.rows, args.records)
        path = directory / f"src_{shape.key}.cstore"
        _build(path, total_rows, records, args.cols, args.dtype)
        data_offset, n_records, itemsizes = _index_args(path)
        print(f"=== {shape.label}  ({total_rows:,} rows, {n_records:,} records) ===")
        print("  open() end-to-end:")
        _c.compare(
            [
                ("walk-open ", lambda p=path: _open_with(p, False)),
                ("vec-open  ", lambda p=path: _open_with(p, True)),
            ],
            repeat=args.repeat,
            warmup=args.warmup,
            baseline=0,
        )
        print("  read_record_index (isolated):")
        _c.compare(
            [
                (
                    "walk-index",
                    lambda p=path, d=data_offset, n=n_records, s=itemsizes: _index_with(
                        p, d, n, s, False
                    ),
                ),
                (
                    "vec-index ",
                    lambda p=path, d=data_offset, n=n_records, s=itemsizes: _index_with(
                        p, d, n, s, True
                    ),
                ),
            ],
            repeat=args.repeat,
            warmup=args.warmup,
            baseline=0,
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=7,
        warmup=2,
        rows=10_000_000,
        cols=8,
        dtype="float64",
        threads=True,
        scale=True,
        tmpdir=True,
    )
    parser.add_argument(
        "--records", type=int, default=200_000, help="records in the source store (the cost driver)"
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    previous = _fmt._VECTORIZED_RECORD_INDEX
    try:
        if args.tmpdir is not None:
            directory = Path(args.tmpdir)
            directory.mkdir(parents=True, exist_ok=True)
            if not args.skip_correctness:
                check_correctness(directory, args)
            if not args.skip_bench:
                run_bench(directory, args)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                if not args.skip_correctness:
                    check_correctness(directory, args)
                if not args.skip_bench:
                    run_bench(directory, args)
    finally:
        _fmt._VECTORIZED_RECORD_INDEX = previous


if __name__ == "__main__":
    main()
