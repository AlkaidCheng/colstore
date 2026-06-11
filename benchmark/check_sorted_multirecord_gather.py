"""Verify the native sorted multi-record gather: correctness and timing.

The sorted multi-record fancy path previously ran a NumPy pipeline: an
O(R log K) boundary partition, a per-record Python loop materializing a
K-sized ``byte_offsets`` array, and the raw byte-offset kernel.
Decomposition shows the Python-side machinery grows with the record count --
13% of the path at R=100, 79% at R=10^4, 97% at R=10^5 -- and the per-record
loop is serial. The walk kernel replaces all of it: each thread
binary-searches the record of its chunk's first index, then advances the
record cursor monotonically, computing offsets in registers.

This script checks the route end-to-end through the public reader API and
times it against the partition pipeline it replaces (forced via the
non-native fallback, which is the identical pipeline on a little-endian
host).

Run on the deployment hardware:

    python benchmark/check_sorted_multirecord_gather.py
    python benchmark/check_sorted_multirecord_gather.py --skip-bench

The win should grow strongly with the record count and be largest exactly
where event-organized physics data lives (10^4-10^5 records per file).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import reader as reader_mod


def _build_store(directory: Path, n_records: int, rows: int):
    rng = np.random.default_rng(0)
    full = rng.standard_normal(n_records * rows)
    path = directory / f"r{n_records}.cstore"
    with colstore.create(path) as writer:
        for r in range(n_records):
            writer.write({"value": full[r * rows : (r + 1) * rows]})
    return path, full


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, full = _build_store(Path(tmp), 200, 300)
        dataset = colstore.open(path)
        rng = np.random.default_rng(5)
        for size in (1, 7, 5_000, 60_000):
            indices = np.sort(rng.integers(0, 200 * 300, size=size).astype(np.int64))
            via_kernel = dataset[indices, "value"].array()
            assert np.array_equal(via_kernel, full[indices]), size
        dataset.close()
        # Forced partition pipeline (the pre-change path) must agree.
        original = reader_mod._dtype_is_native
        reader_mod._dtype_is_native = lambda dtype: False  # type: ignore[assignment]
        try:
            dataset = colstore.open(path)
            indices = np.sort(rng.integers(0, 200 * 300, size=20_000).astype(np.int64))
            via_pipeline = dataset[indices, "value"].array()
            dataset.close()
        finally:
            reader_mod._dtype_is_native = original  # type: ignore[assignment]
        dataset = colstore.open(path)
        assert np.array_equal(dataset[indices, "value"].array(), via_pipeline)
        dataset.close()
    print("  ALL CORRECTNESS CHECKS PASSED (walk kernel == partition pipeline == ground truth)\n")


def _read_pipeline(dataset, indices):
    """One sorted read forced through the pre-change partition pipeline
    (the non-native fallback, identical to the old path on a LE host)."""
    original = reader_mod._dtype_is_native
    reader_mod._dtype_is_native = lambda dtype: False  # type: ignore[assignment]
    try:
        return dataset[indices, "value"].array()
    finally:
        reader_mod._dtype_is_native = original  # type: ignore[assignment]


def run_bench(args: argparse.Namespace) -> None:
    for n_records in args.record_counts:
        rows = args.rows // n_records
        total = rows * n_records
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _build_store(Path(tmp), n_records, rows)
            dataset = colstore.open(path)
            indices = np.sort(
                np.random.default_rng(1).integers(0, total, size=args.indices).astype(np.int64)
            )
            dataset[indices[:1000], "value"].array()  # warm mmap
            print(f"R={n_records:<7} rows/rec={rows:<8} K={args.indices:,}")
            _c.compare(
                [
                    ("partition pipeline", lambda d=dataset, i=indices: _read_pipeline(d, i)),
                    ("walk kernel", lambda d=dataset, i=indices: d[i, "value"].array()),
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
        repeat=5,
        warmup=1,
        rows=20_000_000,
        record_counts=[100, 1_000, 10_000, 100_000],
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
