"""Verify the bin-reuse multi-column gather: correctness and timing.

For a multi-column **unsorted fancy** read of a multi-record store, the
per-index record binning (a branchless binary search, measured 87-93% of the
fused gather kernel's cost on the target hardware) is identical for every
column, so the reader computes it once (``gather_segment_bins``) and
reuses it for the remaining columns (``gather_segment_withbins``). This
script checks the route end-to-end through the public reader API and times
it against the per-column path it replaces.

Run on the deployment hardware:

    python benchmark/check_multicolumn_gather.py
    python benchmark/check_multicolumn_gather.py --skip-bench   # correctness only

Expected shape (from the standalone premise check on the same hardware):
~1.9-2.5x at realistic thread caps, growing with the column count and the
record count. Sorted reads and single-column reads are unaffected by design.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import testing
from colstore.reader import ColStoreReader


def _build_store(directory: Path, n_records: int, rows: int, n_cols: int):
    total = n_records * rows
    full = testing.make_columns(total, n_cols, seed=0)
    names = list(full)
    path = directory / f"r{n_records}_c{n_cols}.cstore"
    testing.write_columns(path, full, records=n_records).close()
    return path, names, full


def _disable_route(monkey_target=ColStoreReader):
    original = monkey_target._gather_many_bin_reuse
    monkey_target._gather_many_bin_reuse = lambda self, names, indexer: None  # type: ignore[assignment]
    return original


def _restore_route(original, monkey_target=ColStoreReader):
    monkey_target._gather_many_bin_reuse = original  # type: ignore[assignment]


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, names, full = _build_store(Path(tmp), 50, 800, 4)
        dataset = colstore.open(path)
        indices = np.random.default_rng(5).integers(0, 50 * 800, size=10_000).astype(np.int64)
        routed = dataset[indices, names].dict()
        original = _disable_route()
        try:
            fallback = dataset[indices, names].dict()
        finally:
            _restore_route(original)
        for name in names:
            assert np.array_equal(routed[name], full[name][indices]), name
            assert np.array_equal(routed[name], fallback[name]), name
        # Sorted selector must agree too (route declines it by design).
        sorted_indices = np.sort(indices)
        sorted_read = dataset[sorted_indices, names].dict()
        for name in names:
            assert np.array_equal(sorted_read[name], full[name][sorted_indices]), name
        dataset.close()
    print("  ALL CORRECTNESS CHECKS PASSED (routed == per-column == ground truth)\n")


def _read_per_column(dataset, indices, names):
    """One multi-column read forced onto the per-column fallback path."""
    original = _disable_route()
    try:
        return dataset[indices, names].dict()
    finally:
        _restore_route(original)


def run_bench(args: argparse.Namespace) -> None:
    for n_records in args.record_counts:
        rows = args.rows // n_records
        total = rows * n_records
        with tempfile.TemporaryDirectory() as tmp:
            path, names, _ = _build_store(Path(tmp), n_records, rows, args.cols)
            dataset = colstore.open(path)
            indices = (
                np.random.default_rng(1).integers(0, total, size=args.indices).astype(np.int64)
            )
            dataset[indices[:1000], names].dict()  # warm mmap + route
            print(f"R={n_records:<7} rows/rec={rows:<8} C={args.cols} K={args.indices:,}")
            _c.compare(
                [
                    ("per-column", lambda d=dataset, i=indices, n=names: _read_per_column(d, i, n)),
                    ("bin-reuse", lambda d=dataset, i=indices, n=names: d[i, n].dict()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
            print()
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        rows=20_000_000,
        record_counts=[1_000, 10_000],
        cols=4,
        indices=1_000_000,
        threads=True,
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if not args.skip_correctness:
        check_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
