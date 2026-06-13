"""Cost decomposition of the NumPy pipeline replaced by the fused gather.

Breaks the *pre-fused* unsorted multi-record fancy path (searchsorted ->
record_id -> within_record -> byte_offsets -> gather_bytes) into its
constituent stages and times each one. This measurement is what motivated the
fused kernel: the searchsorted record-binning dominates (~75-90%), not the
byte_offsets materialization (~7-11%), so the kernel folds the binning into
the load rather than merely skipping the offset temporaries.

The ``reader end-to-end`` row goes through the real reader, which uses the
fused kernel -- so it is NOT the sum of the stages above it; the comparison
shows directly how much of the replaced pipeline's cost the kernel removed.

    PYTHONPATH=src python benchmark/profile_multirecord_fancy.py

Stages timed for the REPLACED unsorted pipeline:
  1. sortedness check : np.all(indices[1:] >= indices[:-1])
  2. searchsorted     : record_id = searchsorted(starts, indices) - 1
  3. offset materialize: within_record + byte_offsets (4 K-sized int64 temps)
  4. gather_bytes      : the raw byte-offset gather kernel

Single thread: the setup stages are pure NumPy on one core; the kernel is
capped at 1 thread so the comparison is apples-to-apples on a 1-core box.
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
from colstore import _gather as _cpp_module  # type: ignore[attr-defined]
from colstore.reader import ColStoreReader


def _build(path: Path, total_rows: int, n_records: int, dtype: np.dtype) -> None:
    rng = np.random.default_rng(7)
    full = rng.standard_normal(total_rows).astype(dtype)
    per = total_rows // n_records
    with colstore.create(path) as writer:
        off = 0
        for r in range(n_records):
            n = per if r < n_records - 1 else total_rows - off
            writer.write({"a": full[off : off + n]})
            off += n


def _time(fn, *, repeat: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def profile(total_rows: int, n_records: int, k: int, dtype_str: str, repeat: int) -> None:
    dtype = np.dtype(dtype_str)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.cstore"
        _build(path, total_rows, n_records, dtype)
        ds = ColStoreReader(path)
        assert ds._is_multi_record

        rsr = ds._record_starts_rows
        rsb = ds._record_starts_bytes
        nrr = ds._n_rows_per_record
        cp = int(ds._column_prefix_bytes["a"])
        its = dtype.itemsize
        nd = dtype.newbyteorder("=")

        rng = np.random.default_rng(1)
        indices = rng.integers(0, total_rows, size=k).astype(np.int64)  # unsorted

        # Stage 1: sortedness check.
        t_sorted = _time(lambda: bool(np.all(indices[1:] >= indices[:-1])), repeat=repeat)

        # Stage 2: searchsorted -> record_id.
        def stage_search() -> np.ndarray:
            return np.searchsorted(rsr, indices, side="right") - 1

        t_search = _time(stage_search, repeat=repeat)
        record_id = stage_search()

        # Stage 3: within_record + byte_offsets (the temporaries we want to kill).
        def stage_offsets() -> np.ndarray:
            within_record = indices - rsr[record_id]
            return rsb[record_id] + cp * nrr[record_id] + within_record * its

        t_offsets = _time(stage_offsets, repeat=repeat)
        byte_offsets = stage_offsets()

        # Stage 4: the gather kernel (1 thread).
        output = np.empty(k, dtype=nd)
        t_kernel = _time(
            lambda: _cpp_module.gather_bytes(ds._file_mmap, byte_offsets, output, 1),
            repeat=repeat,
        )

        # End-to-end through the real reader, for reference.
        t_e2e = _time(lambda: ds[indices, "a"].array(), repeat=repeat)
        ds.close()

        setup = t_search + t_offsets + t_sorted
        total = setup + t_kernel
        print(f"  R={n_records:>6} K={k:>9}  dtype={dtype_str}")
        print(
            f"    sortedness : {t_sorted * 1e3:8.3f} ms  ({t_sorted / total:5.1%})\n"
            f"    searchsorted: {t_search * 1e3:8.3f} ms  ({t_search / total:5.1%})\n"
            f"    offsets mat.: {t_offsets * 1e3:8.3f} ms  ({t_offsets / total:5.1%})\n"
            f"    gather_bytes: {t_kernel * 1e3:8.3f} ms  ({t_kernel / total:5.1%})\n"
            f"    -- replaced pipeline total : {total * 1e3:8.3f} ms\n"
            f"    -- reader end-to-end (fused kernel): {t_e2e * 1e3:8.3f} ms"
        )
        print(
            f"    => realized speedup of the fused reader over the replaced "
            f"pipeline: {total / t_e2e:.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    print("Cost decomposition: replaced NumPy pipeline vs fused reader (single thread)\n")
    for n_records in (100, 1000):
        for k in (200_000, 1_000_000):
            profile(2_000_000, n_records, k, "<f8", args.repeat)
    # A small-dtype case: offset math is the same width (int64) but the kernel
    # moves fewer bytes, so setup is an even larger fraction.
    profile(2_000_000, 1000, 1_000_000, "<i4", args.repeat)


if __name__ == "__main__":
    main()
