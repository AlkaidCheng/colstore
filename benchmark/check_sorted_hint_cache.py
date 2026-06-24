"""Verify the per-read sortedness-hint cache: correctness and timing.

A multi-column fancy read tests whether the selector is sorted to choose between
the cursor walk (fast for a sorted selector) and the search kernel. That test is
a full O(K) serial pass for a sorted selector, so a C-column read that recomputes
it per column pays it C times over identical indices. The reader instead resolves
it once per read and threads it to each column. This script times a sorted
multi-column read with the cache against the per-column recompute baseline, and
checks both equal the ground truth.

The baseline is forced through the same internal seam the reader uses: stripping
the ``indices_sorted`` hint so each column recomputes it. The removed work is a
full O(K) serial pass per column; it is the largest share at moderate index
counts and shrinks at very large K, where materializing the output dominates the
read. The cache helps only sorted selectors -- an unsorted selector rejects in
the sampled pass, and single-column reads have nothing to amortize.

Run on the deployment hardware:

    python benchmark/check_sorted_hint_cache.py
    python benchmark/check_sorted_hint_cache.py --skip-bench   # correctness only
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


def _force_recompute():
    """Strip the per-read sortedness hint so each column recomputes it (the baseline)."""
    original = ColStoreReader._gather_one

    def no_hint(self, column_name, row_indexer, thread_cap=None, out=None, indices_sorted=None):
        return original(self, column_name, row_indexer, thread_cap, out=out, indices_sorted=None)

    ColStoreReader._gather_one = no_hint  # type: ignore[assignment]
    return original


def _restore(original) -> None:
    ColStoreReader._gather_one = original  # type: ignore[assignment]


def _read_recompute(reader, indices, names):
    """One sorted multi-column read with the hint stripped (per-column recompute)."""
    original = _force_recompute()
    try:
        return reader[indices, names].dict()
    finally:
        _restore(original)


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, names, full = _build_store(Path(tmp), 50, 800, 4)
        reader = colstore.open(path)
        indices = np.sort(np.random.default_rng(5).integers(0, 50 * 800, size=10_000)).astype(
            np.int64
        )
        cached = reader[indices, names].dict()
        baseline = _read_recompute(reader, indices, names)
        for name in names:
            assert np.array_equal(cached[name], full[name][indices]), name
            assert np.array_equal(cached[name], baseline[name]), name
        # An unsorted selector must agree too (the cache only changes the cost split).
        unsorted = np.random.default_rng(6).integers(0, 50 * 800, size=10_000).astype(np.int64)
        unsorted_read = reader[unsorted, names].dict()
        for name in names:
            assert np.array_equal(unsorted_read[name], full[name][unsorted]), name
        reader.close()
    print("  ALL CORRECTNESS CHECKS PASSED (cache == recompute == ground truth)\n")


def run_bench(args: argparse.Namespace) -> None:
    rows = args.rows // args.records
    with tempfile.TemporaryDirectory() as tmp:
        path, names, _ = _build_store(Path(tmp), args.records, rows, args.cols)
        reader = colstore.open(path)
        for k in args.index_counts:
            indices = np.sort(np.random.default_rng(1).integers(0, reader.n_rows, size=k)).astype(
                np.int64
            )
            reader[indices[:1000], names].dict()  # warm mmaps + segment-table memo
            print(f"K={k:,} C={args.cols} records={args.records} (sorted selector)")
            _c.compare(
                [
                    (
                        "recompute/col",
                        lambda r=reader, i=indices, n=names: _read_recompute(r, i, n),
                    ),
                    ("cache once", lambda r=reader, i=indices, n=names: r[i, n].dict()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
            print()
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, rows=10_000_000, cols=6, threads=True)
    parser.add_argument("--records", type=int, default=2_000, help="records in the synthetic store")
    parser.add_argument(
        "--index-counts",
        type=int,
        nargs="+",
        default=[200_000, 1_000_000, 5_000_000],
        help="sorted-selector index counts to sweep (the O(K) check the cache removes)",
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    if not args.skip_correctness:
        check_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
