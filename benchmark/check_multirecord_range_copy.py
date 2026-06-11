"""Correctness + perf check for the native contiguous multi-record range copy.

Self-contained: run it directly to (1) assert the native C++ range-copy path
(``ColStoreReader._read_contiguous_range_multi_record``) is byte-identical to
the NumPy per-record fallback and to ground truth across many record layouts,
dtypes (including a non-native byte order that must take the fallback), and
range shapes, then (2) time native vs fallback on a spread of record counts.

    PYTHONPATH=src python benchmark/check_multirecord_range_copy.py

The native path runs the per-record copy loop entirely in C++ (one ``memcpy``
per overlapping record, record membership found by an in-kernel binary search),
so the speedup over the fallback grows with the number of records the range
spans. For a range covering few large records the two converge: a big memcpy
dominates and there is no per-record overhead left to remove.

This is a work-reduction change (fewer Python calls, fewer temporaries), not a
parallelism change, so single-thread timing is the relevant measurement.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

# Allow running directly without installing the package.
import _common as _c
import numpy as np

import colstore
from colstore import kernels
from colstore.reader import ColStoreReader, _dtype_is_native

# ---- shared file builder ------------------------------------------------


def _build_multirecord(
    path: Path, dtype: np.dtype, rows_per_record: list[int], *, second_col: bool
) -> dict[str, np.ndarray]:
    """Stream a multi-record file; return the full ground-truth columns."""
    rng = np.random.default_rng(12345)
    total = sum(rows_per_record)
    if dtype.kind == "f":
        full_a = rng.standard_normal(total).astype(dtype)
    elif dtype.kind in ("i", "u"):
        info = np.iinfo(dtype)
        full_a = rng.integers(info.min // 2, info.max // 2, size=total).astype(dtype)
    else:
        raise ValueError(f"unsupported dtype for builder: {dtype}")
    full_b = np.arange(total, dtype=np.int64)

    cols: dict[str, np.ndarray] = {"a": full_a}
    if second_col:
        cols["b"] = full_b
    with colstore.create(path) as writer:
        off = 0
        for n in rows_per_record:
            rec = {"a": full_a[off : off + n]}
            if second_col:
                rec["b"] = full_b[off : off + n]
            writer.write(rec)
            off += n
    return cols


# ---- correctness --------------------------------------------------------


def _check_one(dtype_str: str, rows_per_record: list[int]) -> None:
    dtype = np.dtype(dtype_str)
    native = _dtype_is_native(dtype)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.cstore"
        cols = _build_multirecord(path, dtype, rows_per_record, second_col=True)
        total = sum(rows_per_record)
        ground = {k: v.astype(v.dtype.newbyteorder("=")) for k, v in cols.items()}

        ds = ColStoreReader(path)
        if len(rows_per_record) > 1:
            assert ds._is_multi_record, "expected a multi-record file"

        boundaries = np.cumsum([0, *rows_per_record])
        ranges = [(0, total), (0, 1), (total - 1, total), (total, total)]
        for i in range(len(rows_per_record)):
            lo, hi = int(boundaries[i]), int(boundaries[i + 1])
            if hi - lo >= 3:
                ranges.append((lo + 1, hi - 1))
            ranges.append((max(0, lo - 2), min(total, hi + 2)))
        ranges.append((total // 4, 3 * total // 4))

        for col in cols:
            for lo, hi in ranges:
                got = ds[lo:hi, col].array()
                exp = ground[col][lo:hi]
                assert (
                    got.dtype == exp.dtype
                ), f"dtype mismatch {dtype_str} {col} [{lo}:{hi}]: {got.dtype} != {exp.dtype}"
                assert np.array_equal(
                    got, exp
                ), f"value mismatch {dtype_str} col={col} range=[{lo}:{hi}] native={native}"

        if native and kernels.cpp_available() and ds._is_multi_record:
            for col in cols:
                disk_dtype = ds._column_dtypes[col]
                nd = disk_dtype.newbyteorder("=")
                cp = int(ds._column_prefix_bytes[col])
                its = disk_dtype.itemsize
                for lo, hi in ranges:
                    n = hi - lo
                    out_native = ds._read_contiguous_range_multi_record(
                        lo, hi, disk_dtype, nd, cp, its
                    )
                    out_py = ds._copy_multirecord_range_python(
                        lo, hi, disk_dtype, cp, its, np.empty(n, dtype=nd)
                    )
                    assert np.array_equal(
                        out_native, out_py
                    ), f"native != python fallback {dtype_str} {col} [{lo}:{hi}]"
        ds.close()
    print(
        f"  ok: dtype={dtype_str:>6} native={native!s:>5} "
        f"records={len(rows_per_record):>3} total={total}"
    )


def run_correctness() -> None:
    print("Correctness: native vs python fallback vs ground truth")
    print(f"  cpp_available = {kernels.cpp_available()}")
    configs: list[tuple[str, list[int]]] = [
        ("<f8", [100]),
        ("<f8", [10, 10, 10, 10, 10]),
        ("<f4", [7, 3, 11, 1, 20, 5]),
        ("<i8", [1, 1, 1, 1, 1, 1, 1, 1]),
        ("<i4", [1000, 1, 500, 250, 1, 1, 333]),
        ("<i2", [64, 64, 64, 64]),
        ("i1", [50, 50, 50]),
        (">f8", [10, 10, 10, 10]),  # non-native -> fallback
        (">i4", [33, 17, 100, 5]),  # non-native -> fallback
    ]
    for dtype_str, rpr in configs:
        _check_one(dtype_str, rpr)
    print("  ALL CORRECTNESS CHECKS PASSED\n")


# ---- timing -------------------------------------------------------------


def _run_one(
    total_rows: int, n_records: int, dtype_str: str, frac: float, repeat: int, warmup: int
) -> None:
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
        nd = disk_dtype.newbyteorder("=")
        cp = int(ds._column_prefix_bytes["a"])
        its = disk_dtype.itemsize

        span = int(total_rows * frac)
        lo = (total_rows - span) // 2
        hi = lo + span
        rec_span = int(
            np.searchsorted(ds._record_starts_rows, hi - 1, "right")
            - np.searchsorted(ds._record_starts_rows, lo, "right")
            + 1
        )

        assert np.array_equal(
            ds._read_contiguous_range_multi_record(lo, hi, disk_dtype, nd, cp, its),
            ds._copy_multirecord_range_python(lo, hi, disk_dtype, cp, its, np.empty(hi - lo, nd)),
        )
        print(
            f"rows={total_rows:>9,} records={n_records:>5} span={frac:>4.0%} "
            f"(~{rec_span} recs span) dtype={dtype_str}"
        )
        _c.compare(
            [
                (
                    "python",
                    lambda: ds._copy_multirecord_range_python(
                        lo, hi, disk_dtype, cp, its, np.empty(hi - lo, dtype=nd)
                    ),
                ),
                (
                    "native",
                    lambda: ds._read_contiguous_range_multi_record(lo, hi, disk_dtype, nd, cp, its),
                ),
            ],
            repeat=repeat,
            warmup=warmup,
            baseline=0,
        )
        ds.close()


def run_bench(args: argparse.Namespace) -> None:
    print("Timing: native vs python contiguous multi-record range copy (single thread)")
    print("The win grows with the record count the range spans; few large records converge to ~1x.")
    for n_records in args.record_counts:
        _run_one(args.rows, n_records, args.dtype, args.frac, args.repeat, args.warmup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=20,
        warmup=2,
        rows=2_000_000,
        record_counts=[10, 2_000, 5_000, 10_000],
        dtype="<f8",
    )
    parser.add_argument(
        "--frac", type=float, default=1.0, help="fraction of total rows the range spans"
    )
    args = parser.parse_args()
    if not args.skip_correctness:
        run_correctness()
    if not args.skip_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
