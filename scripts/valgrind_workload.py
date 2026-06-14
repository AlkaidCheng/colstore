#!/usr/bin/env python3
"""Deterministic allocation-heavy workload for leak-checking colstore under Valgrind.

Run *through* ``run_valgrind.sh`` rather than directly; that wrapper sets the
environment (``PYTHONMALLOC=malloc``, OpenMP knobs) that keeps Memcheck's output
readable. Run standalone only to sanity-check the workload itself.

The workload writes synthetic stores and exercises every native code path the
package can reach -- the writer's ``writev`` assembly, all gather routes
(contiguous, strided in both directions, sorted walk, unsorted branchless
search, bin-reuse multi-column, the uniform reciprocal-divide kernels, the
boolean-mask scan), the zero-copy view path, and the dict/recarray/frame
conversions -- then drops every reference and collects. It repeats for
``--iterations`` so that a per-call leak accumulates into a visible figure
instead of hiding in one-time setup. Sizes are deliberately small: Valgrind runs
20-50x slower than native, so the goal is *coverage of the paths*, not
throughput.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import sys
import tempfile
from pathlib import Path

import numpy as np

import colstore
from colstore import testing


def _exercise_reads(ds: colstore.ColStoreReader, rows: int) -> None:
    """Touch every multi-record gather route on an open reader."""
    rng = np.random.default_rng(1234)
    k = max(1, rows // 4)
    idx = rng.integers(0, rows, size=k, dtype=np.int64)

    # Contiguous range, strided (both directions), per the slice routing.
    ds[: rows // 2, "c0"].array()
    ds[rows // 4 : 3 * rows // 4, "c0"].array()
    ds[::2, "c0"].array()
    ds[::-1, "c0"].array()
    ds[rows::-3, "c0"].array()

    # Unsorted fancy (branchless search / uniform reciprocal kernels) and the
    # sorted linear walk; single- and multi-column (bin-reuse) variants.
    ds[idx, "c0"].array()
    ds[np.sort(idx), "c0"].array()
    ds[idx, ["c0", "c1"]].dict()
    ds[np.sort(idx), ["c0", "c1"]].dict()

    # Boolean mask: dense (mask-native kernel) and sparse (lowers to indices).
    dense = rng.random(rows) < 0.6
    sparse = rng.random(rows) < 0.02
    ds[dense, "c0"].array()
    ds[dense, ["c0", "c1"]].dict()
    ds[sparse, "c0"].array()

    # Whole-store conversions.
    ds.dict()
    ds.recarray()
    try:
        ds.frame()
    except ImportError:
        pass  # pandas optional; skip if absent.


def _exercise_zero_copy(path: Path) -> None:
    """Compact to a single-record store and exercise the copy=False view path."""
    compacted = path.with_suffix(".compact.cstore")
    colstore.compact(path, out=compacted, show_progress=False)
    with colstore.open(compacted) as ds:
        d = ds.dict(copy=False)  # read-only views over the mmap
        half = d["c0"].shape[0] // 2
        _ = sum(float(v.sum()) for v in d.values())
        ds["c0"].array(copy=False)
        ds[:half, "c0"].array(copy=False)
    compacted.unlink(missing_ok=True)


def _one_iteration(tmpdir: Path, rows: int, cols: int, records: int, n: int) -> None:
    # Irregular multi-record store (search / withbins / rbase routes).
    irregular = tmpdir / f"irregular_{n}.cstore"
    with testing.make_store(irregular, rows=rows, cols=cols, records=records, seed=n) as ds:
        _exercise_reads(ds, rows)
        colstore.info(irregular)
        colstore.schema(irregular)

    # Uniform-layout store (reciprocal-divide kernels): rows split evenly.
    uniform = tmpdir / f"uniform_{n}.cstore"
    even_records = max(1, records)
    even_rows = (rows // even_records) * even_records
    with testing.make_store(uniform, rows=even_rows, cols=cols, records=even_records, seed=n) as ds:
        _exercise_reads(ds, even_rows)

    _exercise_zero_copy(irregular)

    irregular.unlink(missing_ok=True)
    uniform.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="rows per store (default 100000)")
    parser.add_argument("--cols", type=int, default=4, help="columns per store (default 4)")
    parser.add_argument("--records", type=int, default=8, help="records per store (default 8)")
    parser.add_argument(
        "--iterations", type=int, default=3, help="repeat the full cycle N times (default 3)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="gather thread cap (default 1: serial kernels keep Memcheck output clean)",
    )
    parser.add_argument(
        "--tmpdir", type=str, default=None, help="scratch dir (default: a temp dir)"
    )
    args = parser.parse_args()

    # A fixed, low cap keeps the kernels serial so leak stacks are unambiguous;
    # the OpenMP runtime's own pool is filtered by the suppression file.
    colstore.set_gather_thread_cap(max(1, args.threads))

    with contextlib.ExitStack() as stack:
        if args.tmpdir is not None:
            base = Path(args.tmpdir)
            base.mkdir(parents=True, exist_ok=True)
        else:
            base = Path(stack.enter_context(tempfile.TemporaryDirectory()))

        for n in range(args.iterations):
            print(f"[workload] iteration {n + 1}/{args.iterations}", file=sys.stderr, flush=True)
            _one_iteration(base, args.rows, args.cols, args.records, n)
            gc.collect()

    gc.collect()
    print("[workload] done", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
