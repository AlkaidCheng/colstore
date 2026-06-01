"""Tests for gather thread-cap config, kernel thread resolution, and autotune."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import colstore
from colstore import autotune, config
from colstore.kernels import cpp_available


def test_import_does_not_touch_global_threading_env(monkeypatch):
    # colstore must not set process-global OpenMP/BLAS env vars at import.
    # OPENBLAS_NUM_THREADS is never touched; OMP_WAIT_POLICY is opt-in only.
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    import importlib

    import colstore as freshly

    importlib.reload(freshly)
    assert "OPENBLAS_NUM_THREADS" not in os.environ
    assert "OMP_WAIT_POLICY" not in os.environ


def test_use_passive_openmp_wait_is_opt_in(monkeypatch):
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    assert colstore.use_passive_openmp_wait() is True
    assert os.environ["OMP_WAIT_POLICY"] == "passive"
    # Does not override an already-set value.
    monkeypatch.setenv("OMP_WAIT_POLICY", "active")
    assert colstore.use_passive_openmp_wait() is False
    assert os.environ["OMP_WAIT_POLICY"] == "active"


def test_default_thread_cap_is_within_ceiling():
    cap = config.get_gather_thread_cap()
    assert 1 <= cap <= config._GATHER_THREAD_CEILING


def test_set_gather_thread_cap_roundtrips():
    original = config.get_gather_thread_cap()
    try:
        config.set_gather_thread_cap(3)
        assert config.get_gather_thread_cap() == 3
    finally:
        config.set_gather_thread_cap(original)


def test_set_gather_thread_cap_rejects_non_positive():
    with pytest.raises(ValueError, match=">= 1"):
        config.set_gather_thread_cap(0)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_thread_count_resolution_rules():
    from colstore import _gather  # type: ignore[attr-defined]

    # Below the serial threshold -> always 1 thread.
    assert _gather.thread_count_for(1000, 8) == 1
    assert _gather.thread_count_for((1 << 18) - 1, 8) == 1
    # At/above threshold the count is >= 1 and never exceeds the cap.
    big = _gather.thread_count_for(50_000_000, 4)
    assert 1 <= big <= 4


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_gather_correct_under_various_caps(tmp_path):
    store = colstore.ColStore.from_dict(
        {"a": np.arange(5000, dtype=np.float64)},
        tmp_path / "caps.cstore",
        show_progress=False,
        backend="cpp",
    )
    indices = np.array([4999, 0, 2500, 1, 4998])
    expected = indices.astype(np.float64)
    original = config.get_gather_thread_cap()
    try:
        for cap in (1, 2, 8):
            config.set_gather_thread_cap(cap)
            result = store[indices, "a"].to_array()
            assert np.array_equal(result, expected)
    finally:
        config.set_gather_thread_cap(original)
        store.close()


def test_load_cached_cap_absent_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert autotune.load_cached_cap() is None


def test_cache_roundtrip_and_fingerprint_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    autotune._write_cache(4, {1: 1.0, 4: 2.0})
    assert autotune.load_cached_cap() == 4

    # A cache written for different hardware must be ignored.
    path = autotune._cache_path()
    payload = json.loads(path.read_text())
    payload["fingerprint"] = "some-other-machine"
    path.write_text(json.dumps(payload))
    assert autotune.load_cached_cap() is None


def test_apply_cached_cap_if_present(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    original = config.get_gather_thread_cap()
    try:
        assert autotune.apply_cached_cap_if_present() is False
        autotune._write_cache(2, {1: 1.0, 2: 1.5})
        config.set_gather_thread_cap(7)
        assert autotune.apply_cached_cap_if_present() is True
        assert config.get_gather_thread_cap() == 2
    finally:
        config.set_gather_thread_cap(original)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_calibrate_picks_and_caches_a_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Shrink the synthetic workload so the test is fast.
    monkeypatch.setattr(autotune, "_CALIB_SOURCE_ROWS", 1_000_000)
    monkeypatch.setattr(autotune, "_CALIB_N_INDICES", 200_000)
    monkeypatch.setattr(autotune, "_CALIB_REPEATS", 2)
    original = config.get_gather_thread_cap()
    try:
        chosen = autotune.calibrate(persist=True)
        assert chosen >= 1
        assert config.get_gather_thread_cap() == chosen
        assert autotune.load_cached_cap() == chosen
    finally:
        config.set_gather_thread_cap(original)
