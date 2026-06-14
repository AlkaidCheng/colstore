"""Focused, loopable single-operation driver for ``perf`` sampling.

Unlike ``profile_gather.py`` (which runs each operation once and prints a
Python-level report), this repeats ONE selected operation in a tight loop, so a
``perf record`` or ``perf stat`` window is dominated by that operation's native
kernel and hotspot attribution is clean. All setup -- store creation, index
generation, a warm-up call -- happens once, before the timed loop, so it stays
out of the measured region. It is driven by ``run_perf.sh``; running it directly
is only useful for a quick standalone throughput check.

The default store is multi-record with rows split evenly, which exercises the
uniform-stride gather kernels (including the reciprocal-divide path). Pass an
uneven ``--records`` count to drive the search/record-base kernels instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

import numpy as np

import colstore
from colstore import testing

# op name -> human description (also the list of valid --op choices)
_OPS = {
    "array-unsorted": "scattered fancy gather, 1 column -> ndarray (memory-latency bound)",
    "array-sorted": "sorted fancy gather, 1 column -> ndarray",
    "dict-unsorted": "scattered fancy gather, all columns -> dict",
    "dict-sorted": "sorted fancy gather, all columns -> dict",
    "recarray": "scattered fancy gather, all columns -> structured ndarray",
    "frame": "scattered fancy gather, all columns -> pandas DataFrame",
    "range": "contiguous range, all columns -> dict",
    "strided": "strided range, 1 column -> ndarray",
    "mask": "boolean-mask gather, 1 column -> ndarray",
}


def _build_thunk(
    ds: colstore.ColStoreReader, op: str, rows: int, indices: int
) -> Callable[[], object]:
    """Return a zero-argument callable performing one instance of ``op``."""
    rng = np.random.default_rng(0)
    unsorted = rng.permutation(rows)[:indices].astype(np.int64)
    ordered = np.sort(unsorted)
    col = "c0"

    if op == "array-unsorted":
        return lambda: ds[unsorted, col].array()
    if op == "array-sorted":
        return lambda: ds[ordered, col].array()
    if op == "dict-unsorted":
        return lambda: ds[unsorted].dict()
    if op == "dict-sorted":
        return lambda: ds[ordered].dict()
    if op == "recarray":
        return lambda: ds[ordered].recarray()
    if op == "frame":
        return lambda: ds[ordered].frame()
    if op == "range":
        return lambda: ds[:indices].dict()
    if op == "strided":
        step = max(2, rows // max(indices, 1))
        return lambda: ds[0 : step * indices : step, col].array()
    if op == "mask":
        mask = np.zeros(rows, dtype=bool)
        mask[unsorted] = True
        return lambda: ds[mask, col].array()
    raise ValueError(f"unknown op: {op}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=sorted(_OPS), default="array-unsorted")
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument(
        "--records", type=int, default=16, help="record count; even split drives uniform kernels"
    )
    parser.add_argument("--indices", type=int, default=1_000_000)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--seconds", type=float, default=5.0, help="loop until this much wall time elapses"
    )
    parser.add_argument(
        "--loops", type=int, default=0, help="fixed loop count (overrides --seconds when > 0)"
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="gather thread cap (0 = leave the configured cap)"
    )
    parser.add_argument("--store-path", default="/tmp/perf_workload.cstore")
    parser.add_argument("--keep-store", action="store_true", help="do not delete the store at exit")
    args = parser.parse_args()

    if args.threads > 0:
        colstore.set_gather_thread_cap(args.threads)

    from pathlib import Path

    store = Path(args.store_path)
    if not store.exists():
        print(
            f"creating store: {args.rows:,} rows x {args.cols} cols x {args.records} records "
            f"({args.dtype}) at {store}",
            file=sys.stderr,
        )
        testing.make_store(
            store,
            rows=args.rows,
            cols=args.cols,
            records=args.records,
            dtype=args.dtype,
            seed=0,
        ).close()

    bytes_per_elt = np.dtype(args.dtype).itemsize
    ds = colstore.open(store)
    try:
        thunk = _build_thunk(ds, args.op, args.rows, args.indices)
        thunk()  # warm: fault in pages, build any caches, so the loop is steady-state

        loops = 0
        t0 = time.perf_counter()
        if args.loops > 0:
            for _ in range(args.loops):
                thunk()
            loops = args.loops
        else:
            while time.perf_counter() - t0 < args.seconds:
                thunk()
                loops += 1
        elapsed = time.perf_counter() - t0
    finally:
        ds.close()
        if not args.keep_store:
            store.unlink(missing_ok=True)

    per_loop_ms = elapsed * 1e3 / max(loops, 1)
    gbps = loops * args.indices * bytes_per_elt / max(elapsed, 1e-12) / 1e9
    print(
        f"[perf_workload] op={args.op} loops={loops} "
        f"per-loop={per_loop_ms:.2f} ms throughput={gbps:.2f} GB/s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
