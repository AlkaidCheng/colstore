"""Benchmark the streaming fill order: row-major (production) vs column-major.

The commit fills the preallocated file row-major -- an outer loop over row
ranges, an inner loop over columns, one memo shared across the columns of a
range. The alternative is column-major: fill each output column completely
before moving to the next, with no memo shared between columns. The two trade
off against each other:

  row-major     shares a subexpression across columns within a range (one
                compute per range), but writes scatter across the per-column
                regions of the file as it cycles columns each range.
  column-major  writes each column's region contiguously (sequential), but a
                shared subexpression is recomputed for every column that uses it.

So row-major is expected to win when columns share work, and column-major when
they do not and write locality is all that is left. This script measures both,
on a shared graph and an independent one, to make that trade-off visible. The
column-major fill lives here, in the benchmark, because it is a rejected design
that does not belong in the library. Run on the deployment hardware:

    PYTHONPATH=src python benchmark/check_edit_layout.py
    PYTHONPATH=src python benchmark/check_edit_layout.py --skip-bench
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import _edit_workload as _w
import numpy as np

import colstore
from colstore.format import (
    _fill_streaming,
    _resolve_streaming_layout,
    record_body_size,
    write_header,
    write_record_header,
)
from colstore.frame import evaluate


def _frame_file(path, specs, n_rows):
    """Write header + record header and preallocate the body; return layout."""
    columns_meta, on_disk_dtypes, bytes_per_row = _resolve_streaming_layout(specs)
    names = list(specs)
    with open(path, "wb") as handle:
        write_header(handle, columns_meta, n_records=1, committed_rows=n_rows)
        write_record_header(handle, record_index=0, n_rows=n_rows)
        body_offset = handle.tell()
        itemsizes = [on_disk_dtypes[name].itemsize for name in names]
        handle.truncate(body_offset + record_body_size(n_rows, itemsizes))
    return names, on_disk_dtypes, body_offset, bytes_per_row


def _fill_column_major(path, specs, names, on_disk_dtypes, body_offset, n_rows, batch_rows):
    """Column-major counterpart to _fill_streaming: each column filled in full.

    Outer loop over columns, inner loop over row ranges, a fresh memo per range
    so nothing is shared between columns -- the deliberate inverse of the
    production fill.
    """
    views = {}
    offset = body_offset
    try:
        for name in names:
            dtype = on_disk_dtypes[name]
            views[name] = np.memmap(path, dtype=dtype, mode="r+", offset=offset, shape=(n_rows,))
            offset += n_rows * dtype.itemsize
        for name in names:
            for start in range(0, n_rows, batch_rows):
                stop = min(start + batch_rows, n_rows)
                views[name][start:stop] = evaluate(specs[name], start, stop, {})
        for name in names:
            views[name].flush()
    finally:
        for view in views.values():
            mmap_obj = getattr(view, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        views.clear()


def _read_columns(path, names):
    reader = colstore.open(path)
    try:
        return reader.dict()
    finally:
        reader.close()


def _run_scenario(label, specs, n, budget, args, tmp):
    path = Path(tmp) / f"{label}.cstore"
    names, on_disk_dtypes, body_offset, bytes_per_row = _frame_file(path, specs, n)
    batch_rows = max(1, min(n, budget // bytes_per_row))
    print(f"\n=== {label}: {len(names)} columns x {n:,} rows, batch_rows={batch_rows:,} ===")

    def row_major():
        _fill_streaming(str(path), specs, names, on_disk_dtypes, body_offset, n, batch_rows)

    def column_major():
        _fill_column_major(str(path), specs, names, on_disk_dtypes, body_offset, n, batch_rows)

    if not getattr(args, "skip_correctness", False):
        expected = {name: evaluate(specs[name], 0, n, {}) for name in names}
        row_major()
        got = _read_columns(path, names)
        for name in names:
            _c.check_equal(got[name], expected[name], f"{label} row-major[{name}]")
        column_major()
        got = _read_columns(path, names)
        for name in names:
            _c.check_equal(got[name], expected[name], f"{label} column-major[{name}]")
        print("  correctness: both fills reproduce the expected columns")

    if args.skip_bench:
        return
    _c.compare(
        [("row-major   (shared memo)", row_major), ("column-major (per column)", column_major)],
        repeat=args.repeat,
        warmup=args.warmup,
        baseline=0,
        throughput_rows=n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=2, rows=4_000_000, cols=12)
    parser.add_argument(
        "--budget", type=int, default=32 * 1024 * 1024, help="memory budget in bytes (batch sizing)"
    )
    args = parser.parse_args()

    n, k = args.rows, args.cols
    with tempfile.TemporaryDirectory() as tmp:
        _run_scenario("shared", _w.shared_graph(n, k), n, args.budget, args, tmp)
        _run_scenario("independent", _w.independent_graph(n, k), n, args.budget, args, tmp)


if __name__ == "__main__":
    main()
