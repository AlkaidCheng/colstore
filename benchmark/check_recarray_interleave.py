"""Interleaved A/B: recarray() with the interleave kernel vs the host path.

``recarray()`` interleaves the columns into the record array with the row-major
``colstore_interleave_records`` kernel, reading each column straight from its
memmap where a zero-copy view exists. This measures it against the column-major
host assignment (``record_array[name] = column``) by toggling the kernel off
in-process, so both variants run **interleaved** under one page-cache and NUMA
state -- the only trustworthy comparison on a multi-NUMA-node host. The
correctness gate asserts the two agree before any timing.

    PYTHONPATH=src python benchmark/check_recarray_interleave.py
    OMP_PROC_BIND=close OMP_PLACES=cores numactl --interleave=all \
        python benchmark/check_recarray_interleave.py --scale 4
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _common as _c
import numpy as np

import colstore
from colstore import config, kernels, testing


def with_kernel_off(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap ``fn`` so it runs with the interleave kernel disabled (host path)."""
    real = kernels.cpp_available

    def wrapped() -> Any:
        kernels.cpp_available = lambda: False  # type: ignore[assignment]
        try:
            return fn()
        finally:
            kernels.cpp_available = real  # type: ignore[assignment]

    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=7,
        warmup=3,
        rows=1_000_000,
        cols=50,
        dtype="float64",
        threads=True,
        scale=True,
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)
    rows = _c.scaled_rows(args.rows, args)

    print("Environment:")
    print(f"  cpp_available     = {_c.cpp_available()}")
    print(f"  gather_thread_cap = {config.get_gather_thread_cap()}")
    print(f"  rows={rows:,}  cols={args.cols}  dtype={args.dtype}")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "store.cstore"
        testing.make_store(path, rows=rows, cols=args.cols, dtype=args.dtype, seed=0).close()
        store = colstore.open(path)
        try:
            read = store.recarray
            _c.check_equal(
                read().view(np.uint8), with_kernel_off(read)().view(np.uint8), "recarray"
            )
            print(f"\n=== RECARRAY, {rows:,} rows x {args.cols} cols ===")
            _c.compare(
                [("kernel", read), ("host (column-major)", with_kernel_off(read))],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=1,
                throughput_rows=rows,
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
