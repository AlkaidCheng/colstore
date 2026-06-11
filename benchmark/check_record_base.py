"""Verify record-base precompute on the irregular multi-column route.

After the first column of an irregular multi-column unsorted read fills the
record bins, the generic withbins kernel still pays three per-record
metadata loads (record_starts_bytes[r], n_rows_per_record[r],
record_starts_rows[r]) and the prefix arithmetic per element. The
record-base route folds all of it into one per-record scalar built once per
column (an O(R) vectorized pass), so the per-element address becomes
record_base[bins[i]] + indices[i] * itemsize -- one metadata load, one
multiply-add. The route is gated on the read being large enough to amortize
the O(R) build (indices-per-record ratio); uniform-record files are
unaffected (they take the arithmetic-binning pair).

This script checks the route end-to-end against ground truth and against
the generic withbins route (toggled via the gate constant,
``reader._RBASE_MIN_INDICES_PER_RECORD``), then times multi-column unsorted
reads on irregular layouts.

Run on the deployment hardware (quiet compute node), both thread regimes:

    python benchmark/check_record_base.py
    OMP_NUM_THREADS=8 python benchmark/check_record_base.py

The reported speedup is for the whole C-column read; the kernel-side saving
applies to columns 2..C only, so the ceiling grows with the column count.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import reader as reader_mod


class _force_generic:
    def __enter__(self):
        self._original = reader_mod._RBASE_MIN_INDICES_PER_RECORD
        reader_mod._RBASE_MIN_INDICES_PER_RECORD = float("inf")
        return self

    def __exit__(self, *exc):
        reader_mod._RBASE_MIN_INDICES_PER_RECORD = self._original
        return False


def _build_irregular_store(directory: Path, n_records: int, mean_rows: int, n_columns: int):
    """Record sizes drawn around the mean so detection stays irregular."""
    rng = np.random.default_rng(0)
    rows = rng.integers(mean_rows // 2, mean_rows + mean_rows // 2, n_records)
    total = int(rows.sum())
    full = {f"c{i}": rng.standard_normal(total) for i in range(n_columns)}
    path = directory / f"r{n_records}.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rec_rows in rows.tolist():
            writer.write({k: v[offset : offset + rec_rows] for k, v in full.items()})
            offset += rec_rows
    return path, full, total


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, full, total = _build_irregular_store(Path(tmp), 500, 300, 4)
        dataset = colstore.open(path)
        assert dataset._uniform_record_layout() is None
        indices = np.random.default_rng(1).integers(0, total, 50_000).astype(np.int64)
        cols = list(full)
        via_rbase = dataset[indices, cols].dict()
        dataset.close()
        with _force_generic():
            dataset = colstore.open(path)
            via_generic = dataset[indices, cols].dict()
            dataset.close()
        for name in cols:
            assert np.array_equal(via_rbase[name], full[name][indices]), name
            assert np.array_equal(via_rbase[name], via_generic[name]), name
    print("  ALL CORRECTNESS CHECKS PASSED (rbase route == generic route == ground truth)\n")


def _read_generic(dataset, indices, cols):
    """One multi-column read forced through the generic withbins route."""
    with _force_generic():
        return dataset[indices, cols].dict()


def run_bench(args: argparse.Namespace) -> None:
    cols = [f"c{i}" for i in range(args.cols)]
    for n_records in args.record_counts:
        mean_rows = args.rows // n_records
        with tempfile.TemporaryDirectory() as tmp:
            path, _, total = _build_irregular_store(Path(tmp), n_records, mean_rows, args.cols)
            dataset = colstore.open(path)
            idx = np.random.default_rng(2).integers(0, total, args.indices).astype(np.int64)
            dataset[idx, cols].dict()  # fault pages first
            print(f"R={n_records:<7} ~rows/rec={mean_rows:<7} C={args.cols} K={args.indices:,}")
            _c.compare(
                [
                    ("generic", lambda d=dataset, i=idx, c=cols: _read_generic(d, i, c)),
                    ("rbase", lambda d=dataset, i=idx, c=cols: d[i, c].dict()),
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
        record_counts=[1_000, 10_000, 100_000],
        cols=4,
        indices=10_000_000,
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
