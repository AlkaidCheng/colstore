"""Verify the uniform-record fast path: correctness and timing.

On uniform-record files (every record the same row count, final record
possibly partial, constant body stride) the unsorted fancy gather's record
bin is computable arithmetically -- one integer division -- instead of the
branchless binary search, and the byte address needs no per-record metadata
loads: full records share one affine formula and the final record one
guarded base. The binning was measured at 87-93% of the fused kernel's
cost on the target hardware, so this targets the dominant term directly.
On multi-column reads the int32 bins array (whose only purpose was to
amortize the search across columns) is skipped entirely.

This script checks the route end-to-end against ground truth and against
the generic route (toggled by monkeypatching the detection seam,
``ColStoreReader._detect_uniform_record_layout``), then times both for
single-column and 4-column unsorted reads.

Run on the deployment hardware (quiet compute node), both thread regimes:

    python benchmark/check_uniform_multirecord.py
    OMP_NUM_THREADS=8 python benchmark/check_uniform_multirecord.py

Expected shape of the result: the win grows with the record count R (the
search it removes deepens with log R) and applies only to files the
detection accepts; irregular files keep the generic kernels unchanged.
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
        self._original = reader_mod.ColStoreReader._detect_uniform_record_layout
        reader_mod.ColStoreReader._detect_uniform_record_layout = lambda self: None
        return self

    def __exit__(self, *exc):
        reader_mod.ColStoreReader._detect_uniform_record_layout = self._original
        return False


def _build_store(directory: Path, n_records: int, rows: int, n_columns: int):
    rng = np.random.default_rng(0)
    total = n_records * rows
    full = {f"c{i}": rng.standard_normal(total) for i in range(n_columns)}
    path = directory / f"r{n_records}.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({k: v[r * rows : (r + 1) * rows] for k, v in full.items()})
    return path, full


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Partial final record: the guarded-base case must hold end to end.
        rng = np.random.default_rng(1)
        rows_per_record = [300] * 199 + [123]
        total = sum(rows_per_record)
        full = {"a": rng.standard_normal(total), "b": rng.standard_normal(total)}
        path = Path(tmp) / "u.cstore"
        offset = 0
        with colstore.create(path) as writer:
            for rows in rows_per_record:
                writer.write({k: v[offset : offset + rows] for k, v in full.items()})
                offset += rows
        indices = rng.integers(0, total, 50_000).astype(np.int64)
        dataset = colstore.open(path)
        assert dataset._uniform_record_layout() is not None
        one = dataset[indices, "a"].array()
        many = dataset[indices, ["a", "b"]].dict()
        dataset.close()
        with _force_generic():
            dataset = colstore.open(path)
            assert dataset._uniform_record_layout() is None
            one_generic = dataset[indices, "a"].array()
            many_generic = dataset[indices, ["a", "b"]].dict()
            dataset.close()
        assert np.array_equal(one, full["a"][indices])
        assert np.array_equal(one, one_generic)
        for name in ("a", "b"):
            assert np.array_equal(many[name], full[name][indices]), name
            assert np.array_equal(many[name], many_generic[name]), name
    print(
        "  ALL CORRECTNESS CHECKS PASSED"
        " (uniform route == generic route == ground truth, partial tail)\n"
    )


def run_bench(args: argparse.Namespace) -> None:
    cols = [f"c{i}" for i in range(args.cols)]
    for n_records in args.record_counts:
        rows = args.rows // n_records
        total = rows * n_records
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _build_store(Path(tmp), n_records, rows, args.cols)
            # Routing is fixed at open time, so a generic-opened reader stays on
            # the generic route without the patch held during the timed reads.
            dataset = colstore.open(path)
            with _force_generic():
                generic = colstore.open(path)
            idx = np.random.default_rng(2).integers(0, total, args.indices).astype(np.int64)
            dataset[idx, "c0"].array()  # fault this index pattern's pages
            print(f"R={n_records:<7} rows/rec={rows:<7} K={args.indices:,}")
            _c.compare(
                [
                    ("generic  1-col", lambda d=generic, i=idx: d[i, "c0"].array()),
                    ("uniform  1-col", lambda d=dataset, i=idx: d[i, "c0"].array()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
            _c.compare(
                [
                    (f"generic {args.cols}-col", lambda d=generic, i=idx: d[i, cols].dict()),
                    (f"uniform {args.cols}-col", lambda d=dataset, i=idx: d[i, cols].dict()),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
            )
            print()
            dataset.close()
            generic.close()


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
