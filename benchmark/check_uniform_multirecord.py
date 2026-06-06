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
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import colstore
from colstore import reader as reader_mod

LAYOUTS = ((1_000, 20_000), (10_000, 2_000), (100_000, 200))
K_SIZES = (2_000_000, 10_000_000)
N_COLUMNS = 4


class _force_generic:
    def __enter__(self):
        self._original = reader_mod.ColStoreReader._detect_uniform_record_layout
        reader_mod.ColStoreReader._detect_uniform_record_layout = lambda self: None
        return self

    def __exit__(self, *exc):
        reader_mod.ColStoreReader._detect_uniform_record_layout = self._original
        return False


def _build_store(directory: Path, n_records: int, rows: int, n_columns: int = N_COLUMNS):
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


def _best(f, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return best


def run_bench(repeat: int) -> None:
    cols = [f"c{i}" for i in range(N_COLUMNS)]
    head = f"{'layout':<28}{'K':>10}{'read':>16}{'generic':>10}{'uniform':>10}{'speedup':>9}"
    print(head)
    for n_records, rows in LAYOUTS:
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _build_store(Path(tmp), n_records, rows)
            total = n_records * rows
            dataset = colstore.open(path)
            for k in K_SIZES:
                indices = np.random.default_rng(2).integers(0, total, k).astype(np.int64)
                dataset[indices, "c0"].array()  # fault pages before either side
                dataset[indices, cols].dict()
                t_new_1 = _best(lambda idx=indices, ds=dataset: ds[idx, "c0"].array(), repeat)
                t_new_c = _best(lambda idx=indices, ds=dataset: ds[idx, cols].dict(), repeat)
                with _force_generic():
                    generic = colstore.open(path)
                    t_old_1 = _best(lambda idx=indices, ds=generic: ds[idx, "c0"].array(), repeat)
                    t_old_c = _best(lambda idx=indices, ds=generic: ds[idx, cols].dict(), repeat)
                    generic.close()
                print(
                    f"  R={n_records:<7} rows/rec={rows:<7}{k:>10}{'1 col':>16}"
                    f"{t_old_1 * 1e3:8.1f}ms{t_new_1 * 1e3:8.1f}ms{t_old_1 / t_new_1:8.2f}x"
                )
                print(
                    f"{'':38}{f'{N_COLUMNS} col bins':>16}"
                    f"{t_old_c * 1e3:8.1f}ms{t_new_c * 1e3:8.1f}ms{t_old_c / t_new_c:8.2f}x"
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
