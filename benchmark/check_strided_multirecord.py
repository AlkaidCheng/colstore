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
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import colstore
from colstore import _gather, config
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


def _time_read(dataset, s, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        dataset[s, "value"].array()
        best = min(best, time.perf_counter() - start)
    return best


def run_bench(repeat: int) -> None:
    header = f"{'layout':<30}{'step':>6}{'K':>10}{'arange route':>14}{'strided':>10}{'speedup':>9}"
    print(header)
    for n_records, rows in LAYOUTS:
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _build_store(Path(tmp), n_records, rows)
            total = n_records * rows
            dataset = colstore.open(path)
            for step in STEPS:
                s = slice(None, None, step)
                k = len(range(*s.indices(total)))
                dataset[s, "value"].array()  # fault this slice's pages before either side
                t_kernel = _time_read(dataset, s, repeat)
                with _force_baseline():
                    t_baseline = _time_read(dataset, s, repeat)
                print(
                    f"  R={n_records:<8} rows/rec={rows:<8}"
                    f"{step:>6}{k:>10}"
                    f"{t_baseline * 1e3:11.2f}ms{t_kernel * 1e3:8.2f}ms"
                    f"{t_baseline / t_kernel:8.2f}x"
                )
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()
    check_correctness()
    if not args.skip_bench:
        run_bench(args.repeat)


if __name__ == "__main__":
    main()
