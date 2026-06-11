"""Lightweight, dependency-free profiling helpers for colstore workloads.

A small public surface for measuring reads and writes the way the package's
own benchmarks do, built around its design goals -- maximum throughput,
minimum memory footprint, minimum per-call overhead. The primitives capture,
per timed call: wall-clock time, process CPU time, the peak active thread
count (so intra-call parallelism is observable), and page-fault deltas (so a
cold read is distinguishable from a warm one).

Two entry points cover the common cases:

* :func:`profile` -- best-of-``repeat`` measurement of a single callable,
  with an optional ``setup`` run *outside* the timed region for destructive
  workloads (e.g. rebuilding a store before each write).
* :func:`profile_interleaved` -- A/B/A/B-style measurement of several
  callables at once, which keeps page-cache and scheduler state comparable
  across the variants (running A...A then B...B confounds the comparison).

Both return the metrics of the single least-perturbed run (minimum wall
time): the minimum is the right summary for a like-for-like measurement, as
it is the run least disturbed by the OS scheduler and other tenants.

The module is stdlib-only and degrades gracefully. Page-fault deltas come
from :mod:`resource` and are ``None`` on platforms that lack it (e.g.
Windows); wall time, CPU time (via :func:`time.process_time`), and the peak
thread count are always populated.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any

try:
    import resource as _resource
except ImportError:  # pragma: no cover - exercised only on non-Unix platforms
    _resource = None  # type: ignore[assignment]

_RESOURCE: ModuleType | None = _resource

__all__ = [
    "ProfileResult",
    "peak_thread_watcher",
    "profile",
    "profile_interleaved",
]


@dataclass(frozen=True)
class ProfileResult:
    """Metrics for the least-perturbed (minimum-wall) run of a profiled call.

    ``major_pf`` / ``minor_pf`` are ``None`` on platforms without
    :mod:`resource`. ``repeat`` records how many timed runs the reported run
    was the best of.
    """

    label: str
    wall_ms: float
    cpu_ms: float
    peak_threads: int
    major_pf: int | None
    minor_pf: int | None
    repeat: int

    @property
    def cpu_wall_ratio(self) -> float:
        """CPU time / wall time. Above 1.0 indicates real parallelism."""
        return self.cpu_ms / self.wall_ms if self.wall_ms > 0 else float("nan")

    def throughput(self, n_rows: int) -> float:
        """Rows per second at the measured wall time (``inf`` if wall is 0)."""
        return n_rows / (self.wall_ms / 1000.0) if self.wall_ms > 0 else float("inf")

    def report(self) -> str:
        """A compact one-line human summary."""
        pf = "n/a" if self.major_pf is None else f"{self.major_pf}/{self.minor_pf}"
        label = f"{self.label} " if self.label else ""
        return (
            f"{label}wall={self.wall_ms:.2f}ms cpu={self.cpu_ms:.1f}ms "
            f"ratio={self.cpu_wall_ratio:.2f}x threads={self.peak_threads} pf={pf}"
        )


@dataclass(frozen=True)
class _Sample:
    """One raw measurement, before best-of reduction."""

    wall_ms: float
    cpu_ms: float
    peak_threads: int
    major_pf: int | None
    minor_pf: int | None


@contextmanager
def peak_thread_watcher(interval_s: float = 0.001) -> Iterator[Callable[[], int]]:
    """Track the peak :func:`threading.active_count` over the ``with`` scope.

    Yields a zero-argument callable returning the peak observed so far. A
    daemon poller samples every ``interval_s`` seconds and is joined on exit.
    """
    peak = [threading.active_count()]
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            count = threading.active_count()
            if count > peak[0]:
                peak[0] = count
            time.sleep(interval_s)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    try:
        yield lambda: peak[0]
    finally:
        stop.set()
        watcher.join(timeout=0.5)


def _measure(fn: Callable[[], Any]) -> _Sample:
    """Run ``fn`` once, capturing wall/CPU time, peak threads, and faults."""
    ru_before = _RESOURCE.getrusage(_RESOURCE.RUSAGE_SELF) if _RESOURCE is not None else None
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    with peak_thread_watcher() as peak_fn:
        fn()
        peak = peak_fn()
    wall_ms = (time.perf_counter() - wall_before) * 1000.0
    cpu_ms = (time.process_time() - cpu_before) * 1000.0
    major: int | None
    minor: int | None
    if _RESOURCE is not None and ru_before is not None:
        ru_after = _RESOURCE.getrusage(_RESOURCE.RUSAGE_SELF)
        major = ru_after.ru_majflt - ru_before.ru_majflt
        minor = ru_after.ru_minflt - ru_before.ru_minflt
    else:
        major = None
        minor = None
    return _Sample(wall_ms, cpu_ms, peak, major, minor)


def _best_of(
    fn: Callable[[], Any],
    setup: Callable[[], Any] | None,
    repeat: int,
    warmup: int,
) -> _Sample:
    """Best-of-``repeat`` sample of ``fn``, with ``setup`` outside timing."""
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    best: _Sample | None = None
    for _ in range(repeat):
        if setup is not None:
            setup()
        sample = _measure(fn)
        if best is None or sample.wall_ms < best.wall_ms:
            best = sample
    assert best is not None  # repeat >= 1 guarantees at least one sample
    return best


def profile(
    fn: Callable[[], Any],
    *,
    repeat: int = 5,
    warmup: int = 2,
    setup: Callable[[], Any] | None = None,
    label: str = "",
) -> ProfileResult:
    """Best-of-``repeat`` profile of ``fn`` after ``warmup`` discarded runs.

    Returns the metrics of the single least-perturbed run (minimum wall time).

    ``setup``, if given, runs before each warmup and timed iteration but
    *outside* the timed region -- use it to rebuild destructive state (such as
    a fresh store for a write benchmark) without timing the teardown.

    Raises ``ValueError`` if ``repeat < 1`` or ``warmup < 0``.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    best = _best_of(fn, setup, repeat, warmup)
    return ProfileResult(
        label=label,
        wall_ms=best.wall_ms,
        cpu_ms=best.cpu_ms,
        peak_threads=best.peak_threads,
        major_pf=best.major_pf,
        minor_pf=best.minor_pf,
        repeat=repeat,
    )


def profile_interleaved(
    labels: list[str],
    fns: list[Callable[[], Any]],
    *,
    repeat: int = 5,
    warmup: int = 2,
    setups: list[Callable[[], Any] | None] | None = None,
) -> list[ProfileResult]:
    """A/B/A/B-style profile of several callables; one result per callable.

    Interleaving keeps page-cache and scheduler state comparable across the
    variants. Returns the best-of-``repeat`` :class:`ProfileResult` per
    callable, in input order. ``setups`` (when given) supplies a per-callable
    setup run outside the timed region, ``None`` for callables that need none.

    Raises ``ValueError`` on length mismatches or ``repeat < 1`` /
    ``warmup < 0``.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if len(labels) != len(fns):
        raise ValueError("labels and fns must have equal length")
    if setups is not None and len(setups) != len(fns):
        raise ValueError("setups must match fns length")
    n = len(fns)
    resolved_setups: list[Callable[[], Any] | None] = setups if setups is not None else [None] * n

    def run_setup(index: int) -> None:
        setup = resolved_setups[index]
        if setup is not None:
            setup()

    for _ in range(warmup):
        for i in range(n):
            run_setup(i)
            fns[i]()
    best: list[_Sample | None] = [None] * n
    for _ in range(repeat):
        for i in range(n):
            run_setup(i)
            sample = _measure(fns[i])
            current = best[i]
            if current is None or sample.wall_ms < current.wall_ms:
                best[i] = sample
    results: list[ProfileResult] = []
    for i in range(n):
        chosen = best[i]
        assert chosen is not None  # repeat >= 1 guarantees a sample per fn
        results.append(
            ProfileResult(
                label=labels[i],
                wall_ms=chosen.wall_ms,
                cpu_ms=chosen.cpu_ms,
                peak_threads=chosen.peak_threads,
                major_pf=chosen.major_pf,
                minor_pf=chosen.minor_pf,
                repeat=repeat,
            )
        )
    return results
