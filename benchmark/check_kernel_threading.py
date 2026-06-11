"""Sanity check: confirm whether the C++ gather kernel is actually serial at cap=1.

Three independent observations are reported for a 10M-element gather:

1. Wall time and user CPU time during the call. If user CPU > wall * 1.5, the
   kernel is running on multiple cores regardless of what the cap claims.

2. The thread count the C++ kernel resolves to internally, via
   ``_gather.thread_count_for(n, cap)``. This mirrors the kernel's own
   ``resolve_thread_count`` so what it reports is what the kernel uses.

3. ``OMP_NUM_THREADS=1`` rerun: if the bench numbers move noticeably when
   the OpenMP runtime is forced to one thread at startup, parallelism was
   leaking in somewhere. They should be identical.
"""

from __future__ import annotations

import argparse
import os

import _common as _c
import numpy as np

from colstore import _gather  # type: ignore[attr-defined]


def report(label: str, fn, repeat: int, warmup: int) -> None:
    # The public profiler captures wall/cpu/threads/faults; cpu/wall ratio is
    # the utilization signal -- above ~1.5 the kernel is using multiple cores.
    result = _c.profile(fn, repeat=repeat, warmup=warmup, label=label)
    flag = "  <-- MULTI-THREADED" if result.cpu_wall_ratio > 1.5 else ""
    print(result.report() + flag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=1, rows=20_000_000, indices=10_000_000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    source = rng.standard_normal(args.rows).astype(np.float32)
    n = args.indices
    sorted_idx = np.sort(rng.choice(args.rows, size=n, replace=False)).astype(np.int64)
    unsorted_idx = rng.permutation(args.rows)[:n].astype(np.int64)

    out_np = np.empty(n, dtype=np.float32)
    out_cpp = np.empty(n, dtype=np.float32)

    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"_gather.max_threads()      = {_gather.max_threads()}")
    print(f"thread_count_for({n}, 1)   = {_gather.thread_count_for(n, 1)}")
    print(f"thread_count_for({n}, 8)   = {_gather.thread_count_for(n, 8)}")
    print()

    for label, idx in (("sorted", sorted_idx), ("unsorted", unsorted_idx)):
        print(f"---- {n:,} {label} ----")
        report("np.take (no out)", lambda i=idx: np.take(source, i), args.repeat, args.warmup)
        report(
            "np.take(out=)", lambda i=idx: np.take(source, i, out=out_np), args.repeat, args.warmup
        )
        report(
            "gather_into cap=1",
            lambda i=idx: _gather.gather_into(source, i, out_cpp, 1),
            args.repeat,
            args.warmup,
        )
        report(
            "gather_into cap=8",
            lambda i=idx: _gather.gather_into(source, i, out_cpp, 8),
            args.repeat,
            args.warmup,
        )
        print()


if __name__ == "__main__":
    main()
