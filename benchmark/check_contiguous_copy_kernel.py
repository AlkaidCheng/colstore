"""Interleaved A/B: single-file contiguous read, copy kernel vs the host path.

``_parallel_copy`` routes a large, native, contiguous copy through the
``colstore_parallel_copy_runs`` kernel; below the size gate, or for a
byteswapping or strided source, it keeps the prior path (a single ``np.array``
copy, or a Python ``ThreadPoolExecutor`` row split above 16 MB). This measures
the kernel against that host path by toggling the kernel off in-process, so both
variants run **interleaved** under one page-cache and NUMA state -- the only
trustworthy comparison on a multi-NUMA-node host. The correctness gate asserts
the two agree before any timing.

    PYTHONPATH=src python benchmark/check_contiguous_copy_kernel.py
    OMP_PROC_BIND=close OMP_PLACES=cores numactl --interleave=all \
        PYTHONPATH=src python benchmark/check_contiguous_copy_kernel.py --scale 4
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _common as _c

import colstore
from colstore import config, kernels, testing


def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def with_kernel_off(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap ``fn`` so it runs with the copy kernel disabled (the host path)."""
    real = kernels.cpp_available

    def wrapped() -> Any:
        kernels.cpp_available = lambda: False  # type: ignore[assignment]
        try:
            return fn()
        finally:
            kernels.cpp_available = real  # type: ignore[assignment]

    return wrapped


def ab(label: str, read: Callable[[], Any], rows: int, repeat: int, warmup: int) -> None:
    """Correctness-gate then interleaved-A/B one read, kernel vs host path."""
    _c.check_equal(read(), with_kernel_off(read)(), label)
    banner(label)
    _c.compare(
        [("kernel", read), ("host (_parallel_copy)", with_kernel_off(read))],
        repeat=repeat,
        warmup=warmup,
        baseline=1,
        throughput_rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=7,
        warmup=3,
        rows=8_000_000,
        cols=4,
        dtype="float64",
        threads=True,
        scale=True,
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    rows = _c.scaled_rows(args.rows, args)

    print("Environment:")
    print(f"  cpp_available     = {_c.cpp_available()}")
    print(f"  gather_thread_cap = {config.get_gather_thread_cap()}  (kernel path needs > 1)")
    print(f"  rows={rows:,}  cols={args.cols}  dtype={args.dtype}")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "single.cstore"
        testing.make_store(path, rows=rows, cols=args.cols, dtype=args.dtype, seed=0).close()
        ds = colstore.open(path)
        try:
            col = ds.columns[0]
            ab(
                f"WHOLE READ, one column ({rows:,} rows)",
                lambda: ds[col].array(),
                rows,
                args.repeat,
                args.warmup,
            )
            lo, hi = rows // 100, rows - rows // 100
            ab(
                f"FORWARD SLICE [{lo:,}:{hi:,}], one column",
                lambda: ds[lo:hi, col].array(),
                hi - lo,
                args.repeat,
                args.warmup,
            )
        finally:
            ds.close()


if __name__ == "__main__":
    main()
