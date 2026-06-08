"""Correctness + perf check for the native fused multi-record fancy gather.

Self-contained. Run directly to (1) assert the native fused kernel
(``ColStoreReader._gather_one_multi_record`` unsorted/native path) is
byte-identical to the NumPy searchsorted pipeline and to ground truth across
record layouts, dtypes (including a non-native byte order that must take the
NumPy fallback), and index shapes (unsorted, duplicates, n=1, full coverage,
out-of-order spanning all records), then (2) time native vs the NumPy pipeline.

    PYTHONPATH=src python benchmark/check_multirecord_fancy_gather.py

The unsorted multi-record fancy path was dominated by ``np.searchsorted``
record-binning (~75-85% of cost, measured), not by the byte_offsets
materialization. The fused kernel bins each index with a branchless binary
search and loads in one pass, replacing searchsorted + the K-sized int64
temporaries + the raw gather. The sorted path is left untouched (already
~4-11x faster than unsorted) and is exercised here only for correctness.

Single thread: searchsorted is single-threaded in NumPy, so the single-thread
comparison understates the win on multi-core hosts, where the kernel's binning
also scales with thread count. Run with OMP_NUM_THREADS>1 on a multi-core box
to see the parallel component.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import _gather as _cpp_module  # type: ignore[attr-defined]
from colstore import kernels
from colstore.reader import ColStoreReader, _dtype_is_native


def _build(path: Path, dtype: np.dtype, rows_per_record: list[int]) -> np.ndarray:
    rng = np.random.default_rng(2024)
    total = sum(rows_per_record)
    if dtype.kind == "f":
        full = rng.standard_normal(total).astype(dtype)
    else:
        info = np.iinfo(dtype)
        full = rng.integers(info.min // 2, info.max // 2, size=total).astype(dtype)
    with colstore.create(path) as writer:
        off = 0
        for n in rows_per_record:
            writer.write({"a": full[off : off + n]})
            off += n
    return full


def _numpy_pipeline(ds: ColStoreReader, indices: np.ndarray, disk_dtype: np.dtype) -> np.ndarray:
    """Reference: the pre-Stage-2 unsorted pipeline (searchsorted + gather_bytes)."""
    nd = disk_dtype.newbyteorder("=")
    its = disk_dtype.itemsize
    cp = int(ds._column_prefix_bytes["a"])
    rsr = ds._record_starts_rows
    rsb = ds._record_starts_bytes
    nrr = ds._n_rows_per_record
    record_id = np.searchsorted(rsr, indices, side="right") - 1
    within = indices - rsr[record_id]
    byte_offsets = rsb[record_id] + cp * nrr[record_id] + within * its
    out = np.empty(indices.shape[0], dtype=nd)
    _cpp_module.gather_bytes(ds._file_mmap, byte_offsets, out, 0)
    return out


def _check_one(dtype_str: str, rows_per_record: list[int]) -> None:
    dtype = np.dtype(dtype_str)
    native = _dtype_is_native(dtype)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.cstore"
        full = _build(path, dtype, rows_per_record)
        total = sum(rows_per_record)
        ground = full.astype(full.dtype.newbyteorder("="))

        ds = ColStoreReader(path)
        if len(rows_per_record) > 1:
            assert ds._is_multi_record

        rng = np.random.default_rng(99)
        index_sets = {
            "unsorted_random": rng.integers(0, total, size=min(5000, total * 3)).astype(np.int64),
            "with_duplicates": (
                rng.integers(0, total, size=total * 2).astype(np.int64)
                if total > 1
                else np.array([0], np.int64)
            ),
            "reversed_full": np.arange(total - 1, -1, -1, dtype=np.int64),
            "single": np.array([rng.integers(0, total)], dtype=np.int64),
            "two_unsorted": np.array([min(total - 1, 3), 0], dtype=np.int64),
            "all_one_record": np.array([total - 1] * 7 if total > 0 else [], dtype=np.int64),
        }
        for label, idx in index_sets.items():
            if idx.size == 0:
                continue
            got = ds[idx, "a"].array()
            exp = ground[idx]
            assert got.dtype == exp.dtype, f"{dtype_str} {label}: dtype {got.dtype} != {exp.dtype}"
            assert np.array_equal(got, exp), f"{dtype_str} {label}: value mismatch native={native}"

            # Native path must match the NumPy pipeline byte-for-byte.
            if native and ds._is_multi_record:
                ref = _numpy_pipeline(ds, idx, ds._column_dtypes["a"])
                assert np.array_equal(got, ref), f"{dtype_str} {label}: native != numpy pipeline"
        ds.close()
    print(
        f"  ok: dtype={dtype_str:>6} native={native!s:>5} "
        f"records={len(rows_per_record):>3} total={total}"
    )


def run_correctness() -> None:
    print("Correctness: fused native gather vs numpy pipeline vs ground truth")
    print(f"  cpp_available = {kernels.cpp_available()}")
    configs: list[tuple[str, list[int]]] = [
        ("<f8", [10, 10, 10, 10, 10]),
        ("<f4", [7, 3, 11, 1, 20, 5]),
        ("<i8", [1, 1, 1, 1, 1, 1, 1, 1]),
        ("<i4", [1000, 1, 500, 250, 1, 1, 333]),
        ("<i2", [64, 64, 64, 64]),
        ("i1", [50, 50, 50]),
        ("<f8", [500000, 500000, 500000, 500000]),  # larger, fewer records
        (">f8", [10, 10, 10, 10]),  # non-native -> numpy fallback path
        (">i4", [33, 17, 100, 5]),  # non-native -> numpy fallback path
    ]
    for dtype_str, rpr in configs:
        _check_one(dtype_str, rpr)
    print("  ALL CORRECTNESS CHECKS PASSED\n")


def _time(fn, *, repeat: int, warmup: int = 3) -> float:
    return _c.best_time(fn, repeat=repeat, warmup=warmup)


def _bench(total_rows: int, n_records: int, k: int, dtype_str: str, repeat: int) -> None:
    dtype = np.dtype(dtype_str)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "b.cstore"
        rng = np.random.default_rng(7)
        full = rng.standard_normal(total_rows).astype(dtype)
        per = total_rows // n_records
        with colstore.create(path) as writer:
            off = 0
            for r in range(n_records):
                n = per if r < n_records - 1 else total_rows - off
                writer.write({"a": full[off : off + n]})
                off += n

        ds = ColStoreReader(path)
        assert ds._is_multi_record
        disk_dtype = ds._column_dtypes["a"]
        indices = rng.integers(0, total_rows, size=k).astype(np.int64)  # unsorted

        native_out = ds[indices, "a"].array()
        ref = _numpy_pipeline(ds, indices, disk_dtype)
        assert np.array_equal(native_out, ref)

        t_native = _time(lambda: ds[indices, "a"].array(), repeat=repeat)
        t_numpy = _time(lambda: _numpy_pipeline(ds, indices, disk_dtype), repeat=repeat)
        ds.close()
        speedup = t_numpy / t_native if t_native > 0 else float("nan")
        print(
            f"  R={n_records:>5} K={k:>9} dtype={dtype_str} | "
            f"numpy={t_numpy * 1e3:8.2f} ms  native={t_native * 1e3:8.2f} ms  "
            f"speedup={speedup:5.2f}x"
        )


def run_bench(repeat: int) -> None:
    print("Timing: native fused gather vs numpy searchsorted pipeline")
    print(f"  max_threads = {kernels.max_threads()} (set OMP_NUM_THREADS>1 for the parallel part)")
    for n_records in (100, 1000):
        for k in (200_000, 1_000_000):
            _bench(2_000_000, n_records, k, "<f8", repeat)
    _bench(2_000_000, 1000, 1_000_000, "<i4", repeat)


def main() -> None:
    _c.run_script(
        correctness=run_correctness,
        bench=run_bench,
        default_repeat=20,
        skip_correctness_flag=True,
        description=__doc__,
    )


if __name__ == "__main__":
    main()
