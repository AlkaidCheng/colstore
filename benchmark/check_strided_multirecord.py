"""Verify the native strided multi-record range read: correctness and timing.

Multi-record slices with ``step != 1`` previously materialized a full int64
index array (``np.arange``), ran the O(K) sortedness check, and went through
the fancy-gather machinery -- the sorted walk kernel for positive steps, the
per-element binary-search kernel for negative steps (a descending stream
fails the sortedness check). The strided kernel synthesizes the row stream
arithmetically: no index array, no sortedness pass, no index-read bandwidth,
and an O(K + R) monotone walk in either direction.

This script checks the route end-to-end through the public reader API
against ground truth and against the replaced route, then times both. The
baseline is reproduced exactly by monkeypatching the routing seam
(``Reader._read_strided_range_multi_record``) with the pre-change logic:
arange + sortedness check + sorted-or-unsorted fancy kernel.

Run on the deployment hardware (quiet compute node):

    python benchmark/check_strided_multirecord.py
    python benchmark/check_strided_multirecord.py --skip-bench

Expected shape of the result: modest wins for positive steps (the replaced
path was already kernel-bound; the savings are arange + sortedness check +
index bandwidth) and large wins for negative steps (binary-search binning
becomes a linear walk).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import _gather, config, testing
from colstore import reader as reader_mod

LAYOUTS = ((1_000, 20_000), (10_000, 2_000), (100_000, 200))
STEPS = (2, 10, 100, -1, -3)


def _baseline_strided(self, start, stop, step, disk_dtype, native_dtype, col_prefix, thread_cap):
    """The pre-change route for native-dtype strided slices, verbatim.

    Materialize the indices, pay the sortedness check, dispatch to the
    sorted walk kernel (ascending) or the fused binary-search kernel
    (descending) -- identical to the old ``_gather_one_multi_record`` flow.
    """
    indices = np.arange(start, stop, step, dtype=np.int64)
    n = indices.shape[0]
    if n == 0:
        return np.empty(0, dtype=native_dtype)
    output = np.empty(n, dtype=disk_dtype)
    cap = config.get_gather_thread_cap() if thread_cap is None else max(1, thread_cap)
    if n > 1 and bool(np.all(indices[1:] >= indices[:-1])):
        _gather.gather_multirecord_sorted(
            self._file_mmap,
            indices,
            output,
            self._record_starts_rows,
            self._record_starts_bytes,
            self._n_rows_per_record,
            int(col_prefix),
            cap,
            config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=True),
        )
    else:
        _gather.gather_multirecord(
            self._file_mmap,
            indices,
            output,
            self._record_starts_rows,
            self._record_starts_bytes,
            self._n_rows_per_record,
            int(col_prefix),
            cap,
            config.resolve_prefetch_distance(self._file_mmap.nbytes, indices_sorted=False),
        )
    return output.astype(native_dtype, copy=False)


class _force_baseline:
    def __enter__(self):
        self._original = reader_mod.ColStoreReader._read_strided_range_multi_record
        reader_mod.ColStoreReader._read_strided_range_multi_record = _baseline_strided
        return self

    def __exit__(self, *exc):
        reader_mod.ColStoreReader._read_strided_range_multi_record = self._original
        return False


def _build_store(directory: Path, n_records: int, rows: int):
    total = n_records * rows
    full = testing.make_columns(total, 1, names=("value",), seed=0)["value"]
    path = directory / f"r{n_records}.cstore"
    testing.write_columns(path, {"value": full}, records=n_records).close()
    return path, full


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, full = _build_store(Path(tmp), 200, 300)
        total = full.shape[0]
        slices = [
            slice(None, None, 2),
            slice(7, None, 3),
            slice(None, None, 997),
            slice(None, None, -1),
            slice(total - 5, 10, -7),
            slice(50, 50, 2),
        ]
        dataset = colstore.open(path)
        via_kernel = [dataset[s, "value"].array() for s in slices]
        dataset.close()
        with _force_baseline():
            dataset = colstore.open(path)
            via_baseline = [dataset[s, "value"].array() for s in slices]
            dataset.close()
        for s, kern, base in zip(slices, via_kernel, via_baseline, strict=True):
            assert np.array_equal(kern, full[s]), s
            assert np.array_equal(kern, base), s
    print("  ALL CORRECTNESS CHECKS PASSED (strided kernel == arange route == ground truth)\n")


def _read_baseline(dataset, s):
    """One strided read forced through the pre-change arange route."""
    with _force_baseline():
        return dataset[s, "value"].array()


def run_bench(args: argparse.Namespace) -> None:
    for n_records in args.record_counts:
        rows = args.rows // n_records
        total = rows * n_records
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _build_store(Path(tmp), n_records, rows)
            dataset = colstore.open(path)
            for step in STEPS:
                s = slice(None, None, step)
                k = len(range(*s.indices(total)))
                dataset[s, "value"].array()  # fault this slice's pages first
                print(f"R={n_records:<7} rows/rec={rows:<7} step={step:<4} K={k:,}")
                _c.compare(
                    [
                        ("arange route", lambda d=dataset, sl=s: _read_baseline(d, sl)),
                        ("strided", lambda d=dataset, sl=s: d[sl, "value"].array()),
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
