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

import os
import resource
import time

import numpy as np

from colstore import _gather  # type: ignore[attr-defined]


def _cpu_times() -> tuple[float, float]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime, r.ru_stime


def report(label: str, fn, repeats: int = 5) -> None:
    fn()  # warmup
    walls, users, syss = [], [], []
    for _ in range(repeats):
        u0, s0 = _cpu_times()
        w0 = time.perf_counter()
        fn()
        w = time.perf_counter() - w0
        u1, s1 = _cpu_times()
        walls.append(w)
        users.append(u1 - u0)
        syss.append(s1 - s0)
    wall = min(walls)
    user = users[walls.index(wall)]
    sys_t = syss[walls.index(wall)]
    util = (user + sys_t) / wall * 100 if wall > 0 else 0
    flag = " <-- MULTI-THREADED" if util > 150 else ""
    print(
        f"  {label:<32} wall={wall*1000:7.2f} ms  user={user*1000:7.2f} ms  "
        f"sys={sys_t*1000:6.2f} ms  util={util:6.1f}%{flag}"
    )


def main() -> None:
    rng = np.random.default_rng(0)
    source = rng.standard_normal(20_000_000).astype(np.float32)
    n = 10_000_000
    sorted_idx = np.sort(rng.choice(20_000_000, size=n, replace=False)).astype(np.int64)
    unsorted_idx = rng.permutation(20_000_000)[:n].astype(np.int64)

    out_np = np.empty(n, dtype=np.float32)
    out_cpp = np.empty(n, dtype=np.float32)

    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"_gather.max_threads()      = {_gather.max_threads()}")
    print(f"thread_count_for(10M, 1)   = {_gather.thread_count_for(n, 1)}")
    print(f"thread_count_for(10M, 8)   = {_gather.thread_count_for(n, 8)}")
    print()

    print("---- 10M sorted ----")
    report("np.take (no out)", lambda: np.take(source, sorted_idx))
    report("np.take(out=)", lambda: np.take(source, sorted_idx, out=out_np))
    report("_gather.gather_into cap=1", lambda: _gather.gather_into(source, sorted_idx, out_cpp, 1))
    report("_gather.gather_into cap=8", lambda: _gather.gather_into(source, sorted_idx, out_cpp, 8))
    print()
    print("---- 10M unsorted ----")
    u_idx = unsorted_idx
    report("np.take (no out)", lambda: np.take(source, u_idx))
    report("np.take(out=)", lambda: np.take(source, u_idx, out=out_np))
    report(
        "_gather.gather_into cap=1", lambda: _gather.gather_into(source, unsorted_idx, out_cpp, 1)
    )
    report(
        "_gather.gather_into cap=8", lambda: _gather.gather_into(source, unsorted_idx, out_cpp, 8)
    )


if __name__ == "__main__":
    main()
