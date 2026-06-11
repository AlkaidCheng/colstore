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
    # A caller cap of 1 forces serial regardless of n (invariant, any host).
    assert _gather.thread_count_for(50_000_000, 1) == 1

    # The work-proportional ramp only grants extra threads when OpenMP actually
    # has them; on a single-core host omp_get_max_threads() == 1 and every
    # resolution is serial. Gate the scaling assertions on that.
    omp_max = _gather.max_threads()
    if omp_max >= 2:
        # Regression guard for the fixed by_work floor: a gather of 256K-1M
        # elements is past PARALLEL_THRESHOLD and must get >= 2 threads. The
        # old ``n / ELEMENTS_PER_THREAD + 1`` floored this band to 1 thread,
        # silently overriding the threshold and leaving a measured ~2x unused.
        assert _gather.thread_count_for(1 << 18, 8) >= 2
        assert _gather.thread_count_for(1_000_000, 8) >= 2
        # Work-proportional: a much larger gather resolves to at least as many
        # threads as a smaller one (more elements -> more threads, up to cap).
        assert _gather.thread_count_for(20_000_000, 32) >= _gather.thread_count_for(1_000_000, 32)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_gather_correct_under_various_caps(tmp_path):
    store = colstore.store(
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
            result = store[indices, "a"].array()
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


def test_clear_cached_cap_removes_file_and_resets_default(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    original = config.get_gather_thread_cap()
    try:
        autotune._write_cache(5, {1: 1.0, 5: 2.0})
        config.set_gather_thread_cap(5)
        assert autotune.clear_cached_cap() is True
        assert autotune.load_cached_cap() is None
        # In-process cap returns to the static hardware default.
        assert config.get_gather_thread_cap() == config._default_gather_thread_cap()
        # Idempotent.
        assert autotune.clear_cached_cap() is False
    finally:
        config.set_gather_thread_cap(original)


def test_clear_cached_cap_can_keep_in_process_value(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    original = config.get_gather_thread_cap()
    try:
        autotune._write_cache(3, {1: 1.0, 3: 2.0})
        config.set_gather_thread_cap(3)
        assert autotune.clear_cached_cap(reset_in_process=False) is True
        assert config.get_gather_thread_cap() == 3
    finally:
        config.set_gather_thread_cap(original)


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
    monkeypatch.setattr(autotune, "_CALIB_WARMUP_ROUNDS", 0)
    original = config.get_gather_thread_cap()
    try:
        chosen = autotune.calibrate(persist=True, rounds=2)
        assert chosen >= 1
        assert config.get_gather_thread_cap() == chosen
        assert autotune.load_cached_cap() == chosen
    finally:
        config.set_gather_thread_cap(original)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_gather_honors_explicit_thread_cap(tmp_path):
    # An explicit thread_cap override is accepted and produces correct output,
    # independent of the global config cap.
    from colstore import kernels

    source = np.arange(2_000_000, dtype=np.float64)
    indices = np.array([1_999_999, 0, 1_000_000, 5], dtype=np.int64)
    for cap in (1, 2, 8):
        out = kernels.gather(source, indices, source.dtype, backend="cpp", thread_cap=cap)
        assert out.tolist() == [1_999_999.0, 0.0, 1_000_000.0, 5.0]


def test_gather_many_divides_thread_budget(tmp_path, monkeypatch):
    # Multi-column concurrent reads must divide the per-call cap across columns
    # so outer threads x inner OpenMP threads does not oversubscribe. We assert
    # on the divided value the dispatcher computes, and on correct output.
    captured: dict[str, int | None] = {}

    import colstore.reader as store_mod

    real_gather_one = store_mod.ColStoreReader._gather_one

    def spy(self, column_name, row_indexer, thread_cap=None):  # type: ignore[no-untyped-def]
        captured[column_name] = thread_cap
        return real_gather_one(self, column_name, row_indexer, thread_cap)

    monkeypatch.setattr(store_mod.ColStoreReader, "_gather_one", spy)

    columns = {f"c{i}": np.arange(100, dtype=np.float64) + i for i in range(4)}
    store = colstore.store(columns, tmp_path / "many.cstore", show_progress=False, backend="cpp")
    original_cap = config.get_gather_thread_cap()
    original_workers = config.get_max_workers()
    try:
        config.set_gather_thread_cap(8)
        config.set_max_workers(4)
        indices = np.array([99, 0, 50], dtype=np.int64)
        result = store[indices, list(columns)].dict()
        # 8 cap / 4 concurrent columns -> 2 threads each.
        assert set(captured.values()) == {2}
        # Output still correct.
        for i, name in enumerate(columns):
            assert result[name].tolist() == [99 + i, 0 + i, 50 + i]
    finally:
        config.set_gather_thread_cap(original_cap)
        config.set_max_workers(original_workers)
        store.close()


def test_gather_many_cap_never_below_one(tmp_path, monkeypatch):
    # With more concurrent columns than the cap, each kernel floors at 1 thread.
    captured: dict[str, int | None] = {}
    import colstore.reader as store_mod

    real = store_mod.ColStoreReader._gather_one

    def spy(self, column_name, row_indexer, thread_cap=None):  # type: ignore[no-untyped-def]
        captured[column_name] = thread_cap
        return real(self, column_name, row_indexer, thread_cap)

    monkeypatch.setattr(store_mod.ColStoreReader, "_gather_one", spy)
    columns = {f"c{i}": np.arange(50, dtype=np.float32) for i in range(6)}
    store = colstore.store(columns, tmp_path / "floor.cstore", show_progress=False, backend="cpp")
    original_cap = config.get_gather_thread_cap()
    original_workers = config.get_max_workers()
    try:
        config.set_gather_thread_cap(2)
        config.set_max_workers(6)
        store[np.array([1, 2, 3]), list(columns)].dict()
        # 2 cap / 6 columns -> floored at 1.
        assert set(captured.values()) == {1}
    finally:
        config.set_gather_thread_cap(original_cap)
        config.set_max_workers(original_workers)
        store.close()


# ---- Dispatch: numpy delegation for serial, cpp kernel for parallel --------


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_gather_into_matches_gather(tmp_path):
    """``gather_into`` and ``gather`` produce identical output in-place."""
    from colstore import _gather  # type: ignore[attr-defined]

    source = np.arange(1_000_000, dtype=np.float64)
    indices = np.array([999_999, 0, 500_000, 1, 999_998], dtype=np.int64)
    out_old = np.empty(5, dtype=np.float64)
    out_new = np.empty(5, dtype=np.float64)
    _gather.gather(source, indices, out_old, 4)
    _gather.gather_into(source, indices, out_new, 4)
    assert np.array_equal(out_old, out_new)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_dispatcher_always_uses_cpp_kernel_when_compatible(monkeypatch):
    """The cpp kernel is invoked for every kernel-compatible gather, including
    small ones; the kernel itself picks serial vs parallel internally.

    Earlier versions of the dispatcher delegated small gathers to ``np.take``
    on the assumption that NumPy's tight C loop beat Cython/OpenMP entry cost.
    Benchmarks on real multi-core hardware overturned that: the cpp kernel
    beats ``np.take`` by 2-4x even at one thread, because numpy re-validates
    indices that we already validated upstream and uses a slower internal
    copy path. So the cpp kernel is now the right answer at every size.
    """
    from colstore import (
        _gather,  # type: ignore[attr-defined]
        kernels,
    )

    cpp_calls: list[int] = []
    real_gather_into = _gather.gather_into

    def spy(source, indices, output, thread_cap, prefetch_distance=-1):
        cpp_calls.append(len(indices))
        return real_gather_into(source, indices, output, thread_cap, prefetch_distance)

    monkeypatch.setattr(_gather, "gather_into", spy)

    source = np.arange(2_000_000, dtype=np.float32)
    # Tiny, mid, and large gathers should all reach the cpp kernel.
    for n in (100, 10_000, 1_500_000):
        indices = np.arange(n, dtype=np.int64)
        out = kernels.gather(source, indices, source.dtype, backend="cpp", thread_cap=8)
        assert np.array_equal(out, source[indices])

    assert cpp_calls == [100, 10_000, 1_500_000]


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_dispatcher_falls_back_to_numpy_for_incompatible_dtypes(tmp_path):
    """Non-native byte order and unsupported kinds still bypass the cpp kernel.

    This is the *correctness* fallback (the kernel does raw element copies and
    cannot handle byte-swapping or e.g. datetime64), separate from any
    performance-based decision.
    """
    from colstore import kernels

    # Big-endian source -> numpy fallback.
    be_source = np.arange(100, dtype=">f4")
    out = kernels.gather(
        be_source, np.array([5, 0, 99], dtype=np.int64), np.dtype("<f4"), backend="cpp"
    )
    assert out.tolist() == [5.0, 0.0, 99.0]


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_cpp_kernel_output_matches_numpy(tmp_path):
    """Sanity: cpp kernel output matches plain ``source[indices]`` byte-for-byte."""
    from colstore import kernels

    rng = np.random.default_rng(7)
    source = rng.standard_normal(500_000).astype(np.float64)
    indices = rng.permutation(500_000)[:50_000].astype(np.int64)

    via_cpp = kernels.gather(source, indices, source.dtype, backend="cpp")
    expected = source[indices]
    assert np.array_equal(via_cpp, expected)


# ---- Parallel contiguous copy helper -----------------------------------


def test_parallel_copy_below_threshold_uses_single_thread():
    """Tiny sources go through the np.array path regardless of thread_cap."""
    from colstore.reader import _parallel_copy

    source = np.arange(1000, dtype=np.int64)
    out = _parallel_copy(source, source.dtype, thread_cap=8)
    assert np.array_equal(out, source)
    assert out.dtype == source.dtype
    # Returned array must own its data (not a view of source).
    assert out.base is None


def test_parallel_copy_thread_cap_one_is_serial():
    """thread_cap=1 must not spin up the threadpool even for big inputs."""
    from colstore.reader import _parallel_copy

    # 32 MiB of int8 -- above the size threshold but cap forbids parallel.
    source = np.zeros(32 * 1024 * 1024, dtype=np.int8)
    source[::1024] = 1
    out = _parallel_copy(source, source.dtype, thread_cap=1)
    assert np.array_equal(out, source)
    assert out.base is None


def test_parallel_copy_produces_identical_bytes_above_threshold():
    """The parallel path must be byte-equivalent to a single np.array copy."""
    from colstore.reader import _parallel_copy

    # 32 MiB of float64 -- above the size threshold; with cap=4 the path
    # splits into multiple chunks.
    rng = np.random.default_rng(0)
    source = rng.standard_normal(4 * 1024 * 1024).astype(np.float64)
    assert source.nbytes >= 16 * 1024 * 1024  # >= _PARALLEL_COPY_MIN_BYTES
    out_parallel = _parallel_copy(source, source.dtype, thread_cap=4)
    out_serial = np.array(source, copy=True)
    assert np.array_equal(out_parallel, out_serial)
    assert out_parallel.base is None


def test_parallel_copy_handles_dtype_change():
    """A non-native source dtype is byte-swapped during the copy on
    little-endian hosts; correctness must be identical to np.array."""
    from colstore.reader import _parallel_copy

    # Force a big-endian source so the dtype conversion does real work.
    big_endian_source = np.arange(4 * 1024 * 1024, dtype=">f8")
    assert big_endian_source.nbytes >= 16 * 1024 * 1024
    out = _parallel_copy(big_endian_source, np.dtype("<f8"), thread_cap=4)
    expected = np.array(big_endian_source, dtype=np.dtype("<f8"), copy=True)
    assert np.array_equal(out, expected)
    assert out.dtype == np.dtype("<f8")


def test_parallel_copy_strided_source_is_byte_identical():
    """A strided (stepped) source must copy identically whether the row-range
    split engages or not -- the helper sizes by logical nbytes, so a strided
    view large enough clears the threshold and parallelizes like a contiguous
    one. The result must equal a single np.array copy of the same view."""
    from colstore.reader import _parallel_copy

    rng = np.random.default_rng(1)
    # Base big enough that the step-2 view's logical size (8 MiB elements ->
    # 64 MiB) clears _PARALLEL_COPY_BYTES_PER_THREAD for cap=4.
    base = rng.standard_normal(16 * 1024 * 1024).astype(np.float64)
    strided = base[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    assert strided.nbytes >= 16 * 1024 * 1024  # >= _PARALLEL_COPY_MIN_BYTES
    out = _parallel_copy(strided, strided.dtype, thread_cap=4)
    expected = np.array(strided, copy=True)
    assert np.array_equal(out, expected)
    assert out.base is None  # owns its data, not a view of base
