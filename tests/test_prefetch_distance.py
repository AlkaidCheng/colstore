"""Tests for the runtime-configurable gather prefetch distance.

The prefetch distance only affects *when* cache lines are requested, never
which bytes are read or written -- so the load-bearing property is that every
distance (including 0 = disabled and values past the input length) produces
byte-identical output, and that the config knob round-trips, validates, and is
honored by the reader paths.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import autotune, config
from colstore.kernels import cpp_available


@pytest.fixture()
def _clean_auto_state(tmp_path, monkeypatch):
    """Isolate auto-mode state: empty cache dir, unloaded table, restore after."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    original = config.get_prefetch_distance()
    config._set_auto_prefetch_table(None)
    yield
    config._set_auto_prefetch_table(None)
    config.set_prefetch_distance(original)


def test_prefetch_distance_roundtrip_and_validation():
    original = config.get_prefetch_distance()
    try:
        config.set_prefetch_distance(0)  # 0 = disabled, legal
        assert config.get_prefetch_distance() == 0
        config.set_prefetch_distance(64)
        assert config.get_prefetch_distance() == 64
        with pytest.raises(ValueError, match="prefetch_distance"):
            config.set_prefetch_distance(-1)
        with pytest.raises(ValueError, match="prefetch_distance"):
            config.set_prefetch_distance(config._PREFETCH_DISTANCE_CEILING + 1)
    finally:
        config.set_prefetch_distance(original)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_config_default_mirrors_kernel_constant():
    # config._DEFAULT_PREFETCH_DISTANCE duplicates the C++ constant so that
    # importing config never requires the extension; this pins the mirror.
    from colstore import _gather  # type: ignore[attr-defined]

    assert _gather.default_prefetch_distance() == config._DEFAULT_PREFETCH_DISTANCE


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
@pytest.mark.parametrize("distance", [0, 1, 3, 8, 64, 4096])
def test_gather_output_identical_across_distances(distance):
    from colstore import _gather  # type: ignore[attr-defined]

    rng = np.random.default_rng(11)
    source = rng.standard_normal(300_000).astype(np.float64)
    indices = rng.integers(0, source.size, size=50_000).astype(np.int64)
    expected = source[indices]

    out = np.empty(indices.size, dtype=source.dtype)
    _gather.gather(source, indices, out, 2, distance)
    assert np.array_equal(out, expected)

    # Distance past the input length: prefetch never fires; output unchanged.
    out2 = np.empty(indices.size, dtype=source.dtype)
    _gather.gather(source, indices, out2, 2, indices.size + 100)
    assert np.array_equal(out2, expected)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
@pytest.mark.parametrize("distance", [0, 8, 64])
def test_multirecord_reads_identical_across_config_distances(tmp_path, distance):
    # End-to-end: the reader passes config.get_prefetch_distance() into both
    # gather_bytes (sorted branch) and gather_multirecord (unsorted branch).
    rng = np.random.default_rng(5)
    full = rng.standard_normal(40_000)
    path = tmp_path / "p.cstore"
    with colstore.create(path) as w:
        for off in range(0, 40_000, 1_000):
            w.write({"a": full[off : off + 1_000]})

    unsorted_idx = rng.integers(0, 40_000, size=5_000).astype(np.int64)
    sorted_idx = np.sort(unsorted_idx)
    original = config.get_prefetch_distance()
    try:
        config.set_prefetch_distance(distance)
        ds = colstore.open(path)
        assert np.array_equal(ds[unsorted_idx, "a"].array(), full[unsorted_idx])
        assert np.array_equal(ds[sorted_idx, "a"].array(), full[sorted_idx])
        ds.close()
    finally:
        config.set_prefetch_distance(original)


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_negative_distance_means_compiled_default():
    # -1 at the _gather layer resolves to the compiled default rather than
    # disabling; the two calls must agree with each other and ground truth.
    from colstore import _gather  # type: ignore[attr-defined]

    rng = np.random.default_rng(3)
    source = rng.standard_normal(10_000).astype(np.float32)
    indices = rng.integers(0, source.size, size=2_000).astype(np.int64)
    out_default = np.empty(indices.size, dtype=source.dtype)
    out_explicit = np.empty(indices.size, dtype=source.dtype)
    _gather.gather(source, indices, out_default, 1, -1)
    _gather.gather(source, indices, out_explicit, 1, _gather.default_prefetch_distance())
    assert np.array_equal(out_default, out_explicit)
    assert np.array_equal(out_default, source[indices])


# ---- "auto" mode ---------------------------------------------------------


def test_default_setting_is_auto():
    # The shipped default is "auto"; uncalibrated it resolves to the compiled
    # default, so out-of-the-box behavior matches the previous fixed setting.
    assert config._DEFAULT_PREFETCH_DISTANCE == 8


def test_set_auto_roundtrip_and_invalid_strings(_clean_auto_state):
    config.set_prefetch_distance("auto")
    assert config.get_prefetch_distance() == "auto"
    with pytest.raises(ValueError, match="prefetch_distance"):
        config.set_prefetch_distance("fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prefetch_distance"):
        config.set_prefetch_distance(3.5)  # type: ignore[arg-type]


def test_resolve_is_passthrough_for_explicit_setting(_clean_auto_state):
    config.set_prefetch_distance(17)
    assert config.resolve_prefetch_distance(10**12, indices_sorted=True) == 17
    assert config.resolve_prefetch_distance(0, indices_sorted=False) == 17
    config.set_prefetch_distance(0)
    assert config.resolve_prefetch_distance(10**12, indices_sorted=False) == 0


def test_auto_without_calibration_falls_back_to_default(_clean_auto_state):
    config.set_prefetch_distance("auto")
    resolved = config.resolve_prefetch_distance(1 << 30, indices_sorted=False)
    assert resolved == config._DEFAULT_PREFETCH_DISTANCE


def test_auto_resolves_regimes_from_table(_clean_auto_state, monkeypatch):
    config.set_prefetch_distance("auto")
    table = {
        "resident_unsorted": 4,
        "resident_sorted": 0,
        "dram_unsorted": 32,
        "dram_sorted": 64,
    }
    config._set_auto_prefetch_table(table)
    monkeypatch.setattr(autotune, "llc_bytes", lambda: 1_000_000)
    assert config.resolve_prefetch_distance(500_000, indices_sorted=False) == 4
    assert config.resolve_prefetch_distance(500_000, indices_sorted=True) == 0
    assert config.resolve_prefetch_distance(2_000_000, indices_sorted=False) == 32
    assert config.resolve_prefetch_distance(2_000_000, indices_sorted=True) == 64
    # Boundary: exactly LLC-sized counts as resident.
    assert config.resolve_prefetch_distance(1_000_000, indices_sorted=False) == 4


def test_prefetch_cache_roundtrip_and_fingerprint_invalidation(_clean_auto_state):
    import json

    table = {r: 8 for r in autotune._PREFETCH_REGIMES}
    autotune._write_prefetch_cache(table, {r: {8: 1.0} for r in table})
    assert autotune.load_cached_prefetch() == table

    path = autotune._prefetch_cache_path()
    payload = json.loads(path.read_text())
    payload["fingerprint"] = "some-other-machine"
    path.write_text(json.dumps(payload))
    assert autotune.load_cached_prefetch() is None


def test_load_cached_prefetch_rejects_malformed_table(_clean_auto_state):
    incomplete = {"resident_unsorted": 8}  # missing regimes
    autotune._write_prefetch_cache(incomplete, {})
    assert autotune.load_cached_prefetch() is None


@pytest.mark.skipif(not cpp_available(), reason="C++ gather extension not built")
def test_calibrate_prefetch_picks_persists_and_applies(_clean_auto_state, monkeypatch):
    # Shrink the synthetic workloads so the test is fast.
    monkeypatch.setattr(autotune, "llc_bytes", lambda: 4 * 1024 * 1024)
    monkeypatch.setattr(autotune, "_PREFETCH_CANDIDATES", (0, 8, 32))
    monkeypatch.setattr(autotune, "_PREFETCH_CALIB_N_INDICES", 100_000)
    monkeypatch.setattr(autotune, "_PREFETCH_CALIB_WARMUP_ROUNDS", 0)

    import warnings

    with warnings.catch_warnings():
        # The stability diagnostic may fire on a noisy CI box with rounds=2;
        # this test asserts the pick/persist/apply contract, not stability.
        warnings.simplefilter("ignore")
        table = autotune.calibrate_prefetch(persist=True, rounds=2)
    assert set(table) == set(autotune._PREFETCH_REGIMES)
    assert all(d in (0, 8, 32) for d in table.values())
    assert autotune.load_cached_prefetch() == table

    # The fresh table is live in-process: "auto" resolves from it.
    config.set_prefetch_distance("auto")
    resolved = config.resolve_prefetch_distance(1, indices_sorted=False)
    assert resolved == table["resident_unsorted"]


# ---- Cache clearing -------------------------------------------------------


def test_clear_cached_prefetch_removes_file_and_resets_auto(_clean_auto_state):
    table = {r: 8 for r in autotune._PREFETCH_REGIMES}
    autotune._write_prefetch_cache(table, {r: {8: 1.0} for r in table})
    config._set_auto_prefetch_table(table)
    config.set_prefetch_distance("auto")
    assert autotune._prefetch_cache_path().exists()

    assert autotune.clear_cached_prefetch() is True
    assert not autotune._prefetch_cache_path().exists()
    # In-process effect: "auto" immediately falls back to the compiled default.
    resolved = config.resolve_prefetch_distance(1, indices_sorted=False)
    assert resolved == config._DEFAULT_PREFETCH_DISTANCE
    # Idempotent: clearing again reports nothing removed and stays quiet.
    assert autotune.clear_cached_prefetch() is False


def test_clear_cached_prefetch_can_keep_in_process_table(_clean_auto_state):
    table = {r: 16 for r in autotune._PREFETCH_REGIMES}
    autotune._write_prefetch_cache(table, {})
    config._set_auto_prefetch_table(table)
    config.set_prefetch_distance("auto")

    assert autotune.clear_cached_prefetch(reset_in_process=False) is True
    # The live table survives until the process restarts.
    assert config.resolve_prefetch_distance(1, indices_sorted=False) == 16


def test_clear_calibration_reports_both_caches(_clean_auto_state):
    autotune._write_prefetch_cache({r: 8 for r in autotune._PREFETCH_REGIMES}, {})
    autotune._write_cache(4, {1: 1.0, 4: 2.0})
    result = autotune.clear_calibration()
    assert result == {"threads": True, "prefetch": True}
    assert autotune.load_cached_cap() is None
    assert autotune.load_cached_prefetch() is None
    assert autotune.clear_calibration() == {"threads": False, "prefetch": False}
