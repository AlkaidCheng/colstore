"""Tests for the public ``colstore.profiling`` API."""

from __future__ import annotations

import threading
import time

import pytest

import colstore
from colstore import profiling


def test_profiling_is_namespaced_on_the_package():
    # The agreed surface is ``colstore.profiling.*``, not a top-level export.
    assert colstore.profiling is profiling
    assert not hasattr(colstore, "profile")
    for name in ("profile", "profile_interleaved", "ProfileResult", "peak_thread_watcher"):
        assert name in profiling.__all__


def test_profile_returns_populated_result():
    result = profiling.profile(lambda: sum(range(1000)), repeat=3, warmup=1, label="sum")
    assert result.label == "sum"
    assert result.repeat == 3
    assert result.wall_ms >= 0.0
    assert result.cpu_ms >= 0.0
    assert result.peak_threads >= 1
    # Page-fault fields are populated on Unix, None elsewhere -- never garbage.
    assert result.major_pf is None or result.major_pf >= 0
    assert (result.major_pf is None) == (result.minor_pf is None)


def test_cpu_wall_ratio_and_throughput_math():
    result = profiling.ProfileResult(
        label="x", wall_ms=10.0, cpu_ms=25.0, peak_threads=4, major_pf=0, minor_pf=7, repeat=1
    )
    assert result.cpu_wall_ratio == pytest.approx(2.5)
    # 1000 rows in 10 ms -> 100_000 rows/s.
    assert result.throughput(1000) == pytest.approx(100_000.0)


def test_zero_wall_is_safe():
    result = profiling.ProfileResult(
        label="", wall_ms=0.0, cpu_ms=0.0, peak_threads=1, major_pf=None, minor_pf=None, repeat=1
    )
    import math

    assert math.isnan(result.cpu_wall_ratio)
    assert math.isinf(result.throughput(1000))


def test_report_includes_label_and_handles_missing_faults():
    with_pf = profiling.ProfileResult("read", 1.0, 1.0, 2, 0, 3, 1).report()
    assert "read" in with_pf and "pf=0/3" in with_pf
    without_pf = profiling.ProfileResult("", 1.0, 1.0, 2, None, None, 1).report()
    assert "pf=n/a" in without_pf


def test_setup_runs_each_iteration_outside_timing():
    setup_calls = 0
    fn_calls = 0

    def setup():
        nonlocal setup_calls
        setup_calls += 1

    def fn():
        nonlocal fn_calls
        fn_calls += 1

    profiling.profile(fn, repeat=4, warmup=2, setup=setup)
    # setup precedes every warmup and timed run, paired one-to-one with fn.
    assert setup_calls == 6
    assert fn_calls == 6


def test_setup_can_rebuild_destructive_state():
    # A queue the callable drains; without per-iteration setup the second run
    # would see an empty queue. Proves setup re-primes state before each run.
    box: list[int] = []

    def setup():
        box.clear()
        box.extend([1, 2, 3])

    def consume():
        while box:
            box.pop()

    result = profiling.profile(consume, repeat=3, warmup=1, setup=setup)
    assert result.repeat == 3


def test_profile_interleaved_one_result_per_fn_in_order():
    labels = ["a", "b", "c"]
    fns = [lambda: sum(range(100)), lambda: sum(range(200)), lambda: sum(range(300))]
    results = profiling.profile_interleaved(labels, fns, repeat=2, warmup=1)
    assert [r.label for r in results] == labels
    assert all(r.repeat == 2 for r in results)
    assert all(r.wall_ms >= 0.0 for r in results)


def test_profile_interleaved_per_fn_setups():
    counters = [0, 0]

    def make_setup(i):
        def setup():
            counters[i] += 1

        return setup

    fns = [lambda: None, lambda: None]
    setups = [make_setup(0), None]
    profiling.profile_interleaved(["a", "b"], fns, repeat=3, warmup=1, setups=setups)
    assert counters[0] == 4  # warmup + 3 timed
    assert counters[1] == 0  # None setup never invoked


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repeat": 0},
        {"warmup": -1},
    ],
)
def test_profile_validates_counts(kwargs):
    with pytest.raises(ValueError):
        profiling.profile(lambda: None, **kwargs)


def test_profile_interleaved_validates_lengths():
    with pytest.raises(ValueError):
        profiling.profile_interleaved(["a"], [lambda: None, lambda: None])
    with pytest.raises(ValueError):
        profiling.profile_interleaved(["a", "b"], [lambda: None, lambda: None], setups=[None])


def test_peak_thread_watcher_observes_a_spawned_thread():
    baseline = threading.active_count()
    with profiling.peak_thread_watcher(interval_s=0.0005) as peak:
        barrier = threading.Event()

        def worker():
            barrier.wait(timeout=1.0)

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.01)  # let the poller sample the elevated count
        observed = peak()
        barrier.set()
        t.join(timeout=1.0)
    assert observed >= baseline + 1


def test_profile_measures_a_real_colstore_read(tmp_path):
    import numpy as np

    path = tmp_path / "p.cstore"
    data = np.arange(500_000, dtype=np.float64)
    with colstore.create(path) as writer:
        writer.write({"x": data})
    with colstore.open(path) as ds:
        result = profiling.profile(lambda: ds[:, "x"].array(), repeat=3, warmup=1, label="read")
    assert result.wall_ms > 0.0
    assert result.throughput(500_000) > 0.0
