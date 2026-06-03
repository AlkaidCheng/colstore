"""Comprehensive performance profile of ColStoreReader gather operations.

Captures three layers of measurement for every operation:

1. Wall and CPU times via ``time.perf_counter`` and ``psutil.Process.cpu_times``.
   The ratio (user+sys) / wall gives effective core utilization. A ratio
   significantly below 1 means the workload is stalled (memory, syscalls, I/O,
   GIL); above 1 means real parallel execution on multiple cores.

2. Page faults and disk I/O via ``resource.getrusage`` and ``/proc/[pid]/io``.
   ``ru_majflt`` counts major page faults (disk reads); ``ru_minflt`` counts
   minor (already-cached pages first-touched). ``read_bytes`` from /proc/io is
   the bytes the kernel actually pulled from the block device; ``read_chars``
   includes cached reads.

3. Per-operation derived metrics: throughput in GB/s of output, nanoseconds
   per gathered element, page-fault rate.

Run cold-cache vs warm-cache variants by recreating the store to force the
OS page cache to evict, or by passing --drop-cache (requires root).
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import psutil

import colstore
from colstore import ColStoreReader, cpp_available, max_threads


@dataclass
class Sample:
    label: str
    wall_s: float
    user_s: float
    sys_s: float
    cpu_pct: float
    minflt: int
    majflt: int
    read_bytes: int
    read_chars: int
    nvcsw: int  # voluntary context switches (usually = blocking syscalls)
    nivcsw: int  # involuntary (preempted)
    n_elements: int
    output_bytes: int
    extra: dict = field(default_factory=dict)

    def print(self) -> None:
        ns_per_elt = self.wall_s * 1e9 / max(self.n_elements, 1)
        gbps = self.output_bytes / max(self.wall_s, 1e-12) / 1e9
        cpu_seconds = self.user_s + self.sys_s
        utilization = cpu_seconds / max(self.wall_s, 1e-12)
        print(f"\n{self.label}")
        print(f"  wall          : {self.wall_s * 1000:8.2f} ms")
        print(f"  user cpu      : {self.user_s * 1000:8.2f} ms")
        print(f"  sys  cpu      : {self.sys_s * 1000:8.2f} ms")
        print(
            f"  utilization   : {utilization * 100:8.1f}%   "
            f"(cpu_time / wall; values < 100% mean stalled, > 100% mean multi-core)"
        )
        print(f"  throughput    : {gbps:8.2f} GB/s output")
        print(f"  per-element   : {ns_per_elt:8.1f} ns/elt")
        print(f"  minor faults  : {self.minflt:8d}  (first-touch of pages already in page cache)")
        print(
            f"  major faults  : {self.majflt:8d}  "
            f"(first-touch requiring disk read; HUGE if non-zero)"
        )
        print(
            f"  vol ctx sw    : {self.nvcsw:8d}  "
            f"(blocking syscalls; high means io_wait or lock waits)"
        )
        print(f"  invol ctx sw  : {self.nivcsw:8d}  (preemption; high under contention)")
        print(
            f"  disk read     : {self.read_bytes / 1e6:8.2f} MB"
            f" (logical), {self.read_chars / 1e6:.2f} MB incl. cache"
        )
        for key, value in self.extra.items():
            print(f"  {key:14}: {value}")


def read_proc_io(pid: int) -> dict[str, int]:
    """Return /proc/[pid]/io as a dict of ints, or empty on non-Linux."""
    try:
        out = {}
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                key, _, value = line.partition(":")
                out[key.strip()] = int(value.strip())
        return out
    except (FileNotFoundError, PermissionError):
        return {}


@contextmanager
def measure(label: str, n_elements: int, output_bytes: int) -> Sample:
    """Context manager that captures all three measurement layers around a block."""
    proc = psutil.Process()
    sample = Sample(label, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n_elements, output_bytes)
    gc.collect()
    start_ru = resource.getrusage(resource.RUSAGE_SELF)
    start_cpu = proc.cpu_times()
    start_io = read_proc_io(proc.pid)
    proc.cpu_percent(interval=None)  # baseline
    t0 = time.perf_counter()
    try:
        yield sample
    finally:
        t1 = time.perf_counter()
        end_cpu = proc.cpu_times()
        end_ru = resource.getrusage(resource.RUSAGE_SELF)
        end_io = read_proc_io(proc.pid)
        sample.wall_s = t1 - t0
        sample.user_s = end_cpu.user - start_cpu.user
        sample.sys_s = end_cpu.system - start_cpu.system
        sample.cpu_pct = proc.cpu_percent(interval=None)
        sample.minflt = end_ru.ru_minflt - start_ru.ru_minflt
        sample.majflt = end_ru.ru_majflt - start_ru.ru_majflt
        sample.nvcsw = end_ru.ru_nvcsw - start_ru.ru_nvcsw
        sample.nivcsw = end_ru.ru_nivcsw - start_ru.ru_nivcsw
        if start_io and end_io:
            sample.read_bytes = end_io.get("read_bytes", 0) - start_io.get("read_bytes", 0)
            sample.read_chars = end_io.get("rchar", 0) - start_io.get("rchar", 0)


def make_store(path: str, n_rows: int, n_cols: int, dtype) -> None:
    """Build a store with random data; do nothing if file already exists."""
    if os.path.exists(path):
        print(f"Reusing existing store: {path}")
        return
    print(f"Creating store M={n_rows:,} x N_COLS={n_cols} ({dtype}) at {path}")
    rng = np.random.default_rng(0)
    columns = {f"f{i}": rng.standard_normal(n_rows).astype(dtype) for i in range(n_cols)}
    colstore.store(columns, path, show_progress=False).close()


def evict_file_cache(path: str) -> bool:
    """Evict a single file's pages from the OS page cache without root.

    Uses ``posix_fadvise(POSIX_FADV_DONTNEED)``, which drops the cached pages
    for just this file (after an ``fsync`` to flush any dirty pages). Unlike
    ``/proc/sys/vm/drop_caches`` this needs no privileges and does not disturb
    the rest of the system cache, so it is the right tool for cold-cache
    measurement of a specific store. Returns ``True`` on success.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            fadvise(fd, 0, 0, dontneed)  # offset=0, len=0 -> whole file
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def drop_caches(path: str | None = None) -> bool:
    """Flush the page cache. Prefer per-file eviction; fall back to global.

    If ``path`` is given, evict just that file via ``posix_fadvise`` (no root
    required). Otherwise, or if that is unavailable, attempt the global
    ``/proc/sys/vm/drop_caches`` (requires root). Returns ``True`` on success.
    """
    if path is not None and evict_file_cache(path):
        return True
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, FileNotFoundError):
        return False


def benchmark_workload(
    store_path: str,
    n_rows: int,
    n_cols_total: int,
    n_indices: int,
    bytes_per_elt: int,
    backends: list[str],
) -> None:
    rng = np.random.default_rng(0)
    sorted_indices = np.sort(rng.choice(n_rows, size=n_indices, replace=False))
    unsorted_indices = rng.permutation(n_rows)[:n_indices].astype(np.int64)

    for backend in backends:
        print(f"\n{'=' * 70}\nbackend = {backend!r}")
        ds = ColStoreReader(store_path, backend=backend)

        # Warm-cache run: touch every page once so subsequent measurements
        # exclude major page-fault cost.
        ds[sorted_indices].to_dict()

        with measure(
            f"[{backend}] 1M sorted indices, all {n_cols_total} cols -> to_dict",
            n_indices * n_cols_total,
            n_indices * n_cols_total * bytes_per_elt,
        ) as s:
            ds[sorted_indices].to_dict()
        s.print()

        with measure(
            f"[{backend}] 1M sorted indices, 1 col -> to_array",
            n_indices,
            n_indices * bytes_per_elt,
        ) as s:
            ds[sorted_indices, "f0"].to_array()
        s.print()

        ds[unsorted_indices, "f0"].to_array()  # warm
        with measure(
            f"[{backend}] 1M UNSORTED indices, 1 col -> to_array",
            n_indices,
            n_indices * bytes_per_elt,
        ) as s:
            ds[unsorted_indices, "f0"].to_array()
        s.print()

        ds.close()


def benchmark_thread_sweep(
    store_path: str,
    n_rows: int,
    n_indices: int,
    bytes_per_elt: int,
    thread_caps: list[int],
) -> None:
    """Sweep the gather thread cap on the unsorted single-column workload.

    This is the workload most sensitive to thread count (scattered, memory-
    latency-bound), so it shows the bandwidth-saturation knee most clearly.
    Use it to confirm the configured cap sits just past the knee on this box.
    """
    from colstore import config

    rng = np.random.default_rng(0)
    unsorted_indices = rng.permutation(n_rows)[:n_indices].astype(np.int64)
    ds = ColStoreReader(store_path, backend="cpp")
    ds[unsorted_indices, "f0"].to_array()  # warm

    print(f"\n{'=' * 70}\nthread-cap sweep (cpp, {n_indices:,} unsorted indices, 1 col)")
    original = config.get_gather_thread_cap()
    try:
        for cap in thread_caps:
            config.set_gather_thread_cap(cap)
            # best-of-3 to damp scheduling noise
            best = float("inf")
            for _ in range(3):
                with measure(f"[cpp] thread_cap={cap}", n_indices, n_indices * bytes_per_elt) as s:
                    ds[unsorted_indices, "f0"].to_array()
                best = min(best, s.wall_s)
            gbps = n_indices * bytes_per_elt / max(best, 1e-12) / 1e9
            print(f"  cap={cap:>3}: best {best * 1000:7.2f} ms  ({gbps:5.2f} GB/s)")
    finally:
        config.set_gather_thread_cap(original)
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="/tmp/profile_gather.cstore")
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--indices", type=int, default=1_000_000)
    parser.add_argument("--backends", nargs="+", default=["cpp", "numpy"], choices=["cpp", "numpy"])
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Evict the store from the page cache before measuring (per-file "
        "posix_fadvise, no root needed).",
    )
    parser.add_argument(
        "--thread-sweep",
        nargs="*",
        type=int,
        default=None,
        metavar="CAP",
        help="Sweep these gather thread caps on the unsorted workload (e.g. "
        "--thread-sweep 1 2 4 8 16). With no values, uses 1 2 4 8 16.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run one-time autotune calibration and cache the chosen cap.",
    )
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    print(f"C++ extension: {cpp_available()}, OpenMP threads: {max_threads()}")
    print(
        f"CPU count: physical={psutil.cpu_count(logical=False)}, "
        f"logical={psutil.cpu_count(logical=True)}"
    )
    from colstore import config

    print(f"gather_thread_cap: {config.get_gather_thread_cap()}")
    cpu_freq = psutil.cpu_freq()
    if cpu_freq is not None:
        print(
            f"CPU freq: {cpu_freq.current:.0f} MHz "
            f"(min={cpu_freq.min:.0f}, max={cpu_freq.max:.0f})"
        )

    bytes_per_elt = 4  # float32
    make_store(args.path, args.rows, args.cols, np.float32)
    file_size_gb = os.path.getsize(args.path) / 1e9
    print(f"\nStore file: {file_size_gb:.2f} GB")

    if args.calibrate:
        from colstore import calibrate

        print("\nRunning calibration...")
        calibrate(verbose=True)

    if args.cold:
        if not drop_caches(args.path):
            print(
                "WARNING: cache eviction failed. Cold runs not meaningful "
                "(need posix_fadvise support or root for drop_caches)."
            )
        else:
            print(f"Page cache evicted for {args.path}.")

    if args.thread_sweep is not None:
        caps = args.thread_sweep or [1, 2, 4, 8, 16]
        caps = [c for c in caps if c <= max_threads()] or [1]
        benchmark_thread_sweep(args.path, args.rows, args.indices, bytes_per_elt, caps)

    benchmark_workload(args.path, args.rows, args.cols, args.indices, bytes_per_elt, args.backends)


if __name__ == "__main__":
    main()
