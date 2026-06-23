"""A/B the C++ record-index kernel against the pure-Python header walk at open.

``read_record_index`` builds the per-record index (cumulative rows, body byte
offsets, per-record rows) the reader needs before any data read. The Python walk
(``colstore.format._read_record_index_walk``) reads each 32-byte header one at a
time and validates magic / sequential index / CRC32 -- the one data-scaling
Python-interpreter loop in the open path, ~linear in record count. The C++ kernel
(``colstore._gather.read_record_index``) performs the identical walk natively,
reading headers through a reused sliding buffer (no whole-file mmap, so no
page-fault churn on a large file).

The kernel's ``read_chunk`` is swept here. A chunk of 32 reads each header in
isolation (one positional read per record); larger chunks amortize the syscall
count across more records by reading the interleaved bodies as warm collateral.
The cost is a syscall-count vs bytes-read tradeoff that depends on the machine
and the page-cache state, so the sweep exists to pick (and re-confirm on the
deployment node) the buffer size.

Two layouts span the records-0..R-2-equal case the index walk sees in practice:
a uniform store and a partial-final store (fixed-size records with a smaller
last record). The cost scales with record *count*, so the store is built with
many small records; ``--records`` is the cost driver.

The correctness gate asserts every variant is byte-identical to the Python walk
before any timing. Run on the deployment node with ``--tmpdir`` on the parallel
filesystem; local numbers are indicative only.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import _common as _c
import numpy as np

import colstore.format as _fmt
from colstore import kernels, testing

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
    """The ``(data_offset, n_records, itemsizes)`` the reader passes to the index build."""
    manifest, data_offset = _fmt.read_header(path)
    n_records = int(manifest["n_records"])
    itemsizes = [np.dtype(col["dtype"]).itemsize for col in manifest["columns"]]
    return data_offset, n_records, itemsizes


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    for shape in _SHAPES:
        total_rows, records = shape.layout(args.rows, args.records)
        path = directory / f"correctness_{shape.key}.cstore"
        _build(path, total_rows, records, args.cols, args.dtype)
        data_offset, n_records, itemsizes = _index_args(path)
        walk = _fmt._read_record_index_walk(path, data_offset, n_records, itemsizes)
        itemsize_sum = sum(itemsizes)
        for chunk in args.chunks:
            got = kernels._cpp_module.read_record_index(
                str(path), data_offset, n_records, itemsize_sum, chunk
            )
            for array, expected, name in zip(got, walk, _ARRAY_NAMES, strict=True):
                _c.check_equal(array, expected, f"{shape.key}:{name} chunk={chunk} vs walk")
        path.unlink(missing_ok=True)
    print("  CORRECTNESS OK (every chunk byte-for-byte == the Python walk)\n")


def run_bench(directory: Path, args: argparse.Namespace) -> None:
    print("Environment:")
    print(f"  rows={args.rows:,}  records={args.records:,}  cols={args.cols}  dtype={args.dtype}")
    print(f"  chunks={args.chunks}")
    print(f"  repeat={args.repeat}  warmup={args.warmup}  store dir={directory}\n")
    for shape in _SHAPES:
        total_rows, records = shape.layout(args.rows, args.records)
        path = directory / f"src_{shape.key}.cstore"
        _build(path, total_rows, records, args.cols, args.dtype)
        data_offset, n_records, itemsizes = _index_args(path)
        itemsize_sum = sum(itemsizes)
        print(f"=== {shape.label}  ({total_rows:,} rows, {n_records:,} records) ===")
        variants: list[tuple[str, Callable[[], object]]] = [
            (
                "walk(py)  ",
                lambda p=path, d=data_offset, n=n_records, s=itemsizes: (
                    _fmt._read_record_index_walk(p, d, n, s)
                ),
            )
        ]
        for chunk in args.chunks:
            variants.append(
                (
                    f"cpp{_chunk_label(chunk)}",
                    lambda p=path, d=data_offset, n=n_records, s=itemsize_sum, c=chunk: (
                        kernels._cpp_module.read_record_index(str(p), d, n, s, c)
                    ),
                )
            )
        _c.compare(variants, repeat=args.repeat, warmup=args.warmup, baseline=0)
        print()


def _chunk_label(chunk: int) -> str:
    if chunk >= 1 << 20:
        return f"@{chunk >> 20}M"
    if chunk >= 1 << 10:
        return f"@{chunk >> 10}K"
    return f"@{chunk}B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=9,
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
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[32, 65536, 1 << 20, 1 << 22],
        help="read_chunk sizes (bytes) to sweep; 32 reads each header in isolation",
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    if not kernels.cpp_available():
        raise SystemExit("the compiled extension is required for this benchmark")

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


if __name__ == "__main__":
    main()
