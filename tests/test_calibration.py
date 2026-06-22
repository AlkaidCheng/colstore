"""Tests for the calibration targets: gather prefetch distance and mask-density gate.

One section per target; each pins the pick rule, the resolution precedence
(explicit setting > calibrated cache > compiled default), cache fingerprint
and malformed-cache handling, and the read paths that consult the resolved
value. The thread-cap calibration is covered in ``test_cli.py`` and
``test_threading.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import colstore
from colstore import _gather, autotune, config
from colstore.kernels import cpp_available

# ---- Gather prefetch distance --------------------------------------------------
# Tests for the runtime-configurable gather prefetch distance.
#
# The prefetch distance only affects *when* cache lines are requested, never
# which bytes are read or written -- so the load-bearing property is that every
# distance (including 0 = disabled and values past the input length) produces
# byte-identical output, and that the config knob round-trips, validates, and is
# honored by the reader paths.


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
    monkeypatch.setattr(autotune, "_CALIB_WARMUP_ROUNDS", 0)

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


def test_clear_calibration_reports_all_caches(_clean_auto_state):
    autotune._write_prefetch_cache({r: 8 for r in autotune._PREFETCH_REGIMES}, {})
    autotune._write_cache(4, {1: 1.0, 4: 2.0})
    autotune._write_mask_density_cache(0.1, {}, {})
    result = autotune.clear_calibration()
    assert result == {"threads": True, "prefetch": True, "mask-density": True}
    assert autotune.load_cached_cap() is None
    assert autotune.load_cached_prefetch() is None
    assert autotune.load_cached_mask_density() is None
    assert autotune.clear_calibration() == {
        "threads": False,
        "prefetch": False,
        "mask-density": False,
    }


# ---- Mask-density gate ----------------------------------------------------------
# Tests for the mask-density gate calibration (``mask-density`` target).
#
# Crossover rule: the gate is the midpoint between the smallest grid density
# that beats the lowered route by the win margin at itself and every denser
# grid point, and the previous grid point (0 for the first); no qualifying
# density disables the route (gate 1.0). Resolution precedence: an explicit
# ``set_mask_density_gate`` value, then the calibrated per-host gate (cache
# present, fingerprint matching, loaded lazily once), then the compiled
# default. Clearing the cache resets the in-process state. The reader's mask
# routing consults the resolved gate on every read.


@pytest.fixture()
def _isolated_gate(tmp_path, monkeypatch):
    """Isolate the cache dir and reset the gate state around each test."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    config.set_mask_density_gate("auto")
    config._set_auto_mask_density(None)
    yield
    config.set_mask_density_gate("auto")
    config._set_auto_mask_density(None)


# ---- Crossover rule -------------------------------------------------------

GRID = autotune._MASK_DENSITY_GRID


def test_pick_wins_everywhere_gates_below_first_point():
    assert autotune._pick_mask_density_gate({d: 2.0 for d in GRID}) == GRID[0] / 2


def test_pick_clean_crossover_gates_at_midpoint():
    ratios = {d: (1.5 if d >= 0.12 else 0.8) for d in GRID}
    assert autotune._pick_mask_density_gate(ratios) == (0.08 + 0.12) / 2


def test_pick_requires_win_margin_not_bare_parity():
    # 1.04x everywhere is under the 1.05 margin: route disabled.
    assert autotune._pick_mask_density_gate({d: 1.04 for d in GRID}) == 1.0


def test_pick_no_win_disables_route():
    assert autotune._pick_mask_density_gate({d: 0.9 for d in GRID}) == 1.0


def test_pick_monotonicity_blocks_noisy_early_crossover():
    # A near-parity dip at a denser point drags the gate past it: every
    # density at or above the gate must win, not just the first.
    ratios = {d: 2.0 for d in GRID}
    ratios[0.2] = 1.0
    assert autotune._pick_mask_density_gate(ratios) == (0.2 + 0.3) / 2


# ---- Config resolution precedence -----------------------------------------


def test_resolve_defaults_without_calibration(_isolated_gate):
    assert config.resolve_mask_density_gate() == config._MASK_DENSITY_GATE_DEFAULT


def test_explicit_gate_overrides_everything(_isolated_gate):
    config._set_auto_mask_density(0.05)
    config.set_mask_density_gate(0.4)
    assert config.resolve_mask_density_gate() == 0.4
    config.set_mask_density_gate(2.0)  # disable value is legal
    assert config.resolve_mask_density_gate() == 2.0


def test_auto_uses_calibrated_gate(_isolated_gate):
    config._set_auto_mask_density(0.065)
    assert config.resolve_mask_density_gate() == 0.065


def test_set_gate_validation(_isolated_gate):
    for bad in (-0.1, "fast", None, True):
        with pytest.raises(ValueError):
            config.set_mask_density_gate(bad)  # type: ignore[arg-type]
    config.set_mask_density_gate("auto")  # round-trips
    assert config.get_mask_density_gate() == "auto"


def test_auto_lazy_loads_cache_once(_isolated_gate):
    autotune._write_mask_density_cache(0.085, {}, {})
    assert config.resolve_mask_density_gate() == 0.085
    # Cache file removal without reset does not affect the loaded value...
    autotune._mask_density_cache_path().unlink()
    assert config.resolve_mask_density_gate() == 0.085
    # ...while clear with reset_in_process drops it immediately.
    autotune.clear_cached_mask_density()
    assert config.resolve_mask_density_gate() == config._MASK_DENSITY_GATE_DEFAULT


def test_foreign_fingerprint_cache_ignored(_isolated_gate):
    autotune._write_mask_density_cache(0.02, {}, {})
    path = autotune._mask_density_cache_path()
    payload = json.loads(path.read_text())
    payload["fingerprint"] = "some-other-machine"
    path.write_text(json.dumps(payload))
    assert autotune.load_cached_mask_density() is None
    assert config.resolve_mask_density_gate() == config._MASK_DENSITY_GATE_DEFAULT


def test_malformed_cache_ignored(_isolated_gate):
    autotune._write_mask_density_cache(0.02, {}, {})
    path = autotune._mask_density_cache_path()
    for bad_gate in (None, "wide", -0.5, 1.5):
        payload = json.loads(path.read_text())
        payload["gate"] = bad_gate
        path.write_text(json.dumps(payload))
        assert autotune.load_cached_mask_density() is None


# ---- Real calibration run (shrunk constants) -------------------------------


@pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")
@pytest.mark.filterwarnings("ignore:calibration for 'mask-density' is unstable")
def test_calibrate_mask_density_end_to_end(_isolated_gate, monkeypatch):
    monkeypatch.setattr(autotune, "_MASK_CALIB_N_RECORDS", 40)
    monkeypatch.setattr(autotune, "_MASK_CALIB_ROWS_PER_RECORD", 100)
    gate = autotune.calibrate_mask_density(rounds=2, verbose=False)
    assert isinstance(gate, float)
    assert 0.0 <= gate <= 1.0
    # Applied in-process and persisted with this machine's fingerprint.
    assert config.resolve_mask_density_gate() == gate
    assert autotune.load_cached_mask_density() == gate
    payload = json.loads(autotune._mask_density_cache_path().read_text())
    assert payload["fingerprint"] == autotune._hardware_fingerprint()
    assert set(payload["ratios"]) == {str(d) for d in autotune._MASK_DENSITY_GRID}
    # Clear restores the default both on disk and in-process.
    assert autotune.clear_cached_mask_density() is True
    assert config.resolve_mask_density_gate() == config._MASK_DENSITY_GATE_DEFAULT
    assert autotune.clear_cached_mask_density() is False  # idempotent


@pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")
@pytest.mark.filterwarnings("ignore:calibration for 'mask-density' is unstable")
def test_calibrate_no_persist_applies_in_process_only(_isolated_gate, monkeypatch):
    monkeypatch.setattr(autotune, "_MASK_CALIB_N_RECORDS", 40)
    monkeypatch.setattr(autotune, "_MASK_CALIB_ROWS_PER_RECORD", 100)
    gate = autotune.calibrate_mask_density(rounds=2, persist=False)
    assert config.resolve_mask_density_gate() == gate
    assert autotune.load_cached_mask_density() is None
    assert not autotune._mask_density_cache_path().exists()


@pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")
@pytest.mark.filterwarnings("ignore:calibration for 'mask-density' is unstable")
def test_calibrate_restores_gate_setting(_isolated_gate, monkeypatch):
    monkeypatch.setattr(autotune, "_MASK_CALIB_N_RECORDS", 40)
    monkeypatch.setattr(autotune, "_MASK_CALIB_ROWS_PER_RECORD", 100)
    config.set_mask_density_gate(0.33)
    autotune.calibrate_mask_density(rounds=2, persist=False)
    # The sweep toggles the gate internally but must restore the setting.
    assert config.get_mask_density_gate() == 0.33


# ---- Reader integration -----------------------------------------------------


@pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")
def test_reader_consults_resolved_gate(tmp_path, _isolated_gate, monkeypatch):
    rng = np.random.default_rng(71)
    rows_per_record = [400, 700, 300, 600]
    total = sum(rows_per_record)
    full = {"a": rng.standard_normal(total), "b": rng.standard_normal(total)}
    path = tmp_path / "gate.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    mask = rng.random(total) < 0.5

    calls: list[str] = []
    original = _gather.gather_multirecord_mask

    def spy(*args, **kwargs):
        calls.append("mask")
        return original(*args, **kwargs)

    monkeypatch.setattr(_gather, "gather_multirecord_mask", spy)
    dataset = colstore.open(path)
    try:
        config.set_mask_density_gate(2.0)  # explicit: route disabled
        assert np.array_equal(dataset[mask, "a"].array(), full["a"][mask])
        assert calls == []
        config.set_mask_density_gate("auto")
        config._set_auto_mask_density(0.1)  # calibrated: below this mask's density
        assert np.array_equal(dataset[mask, "a"].array(), full["a"][mask])
        assert calls == ["mask"]
    finally:
        dataset.close()


@pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")
def test_frame_edit_consults_resolved_gate(tmp_path, _isolated_gate, monkeypatch):
    # ds[mask].edit() keeps the mask, so a dense-mask in-memory terminal routes to the
    # same mask-native kernel as ds[mask], gated on the same resolved density.
    rng = np.random.default_rng(71)
    rows_per_record = [400, 700, 300, 600]
    total = sum(rows_per_record)
    full = {"a": rng.standard_normal(total), "b": rng.standard_normal(total)}
    path = tmp_path / "edit_gate.cstore"
    offset = 0
    with colstore.create(path) as writer:
        for rows in rows_per_record:
            writer.write({k: v[offset : offset + rows] for k, v in full.items()})
            offset += rows
    mask = rng.random(total) < 0.5

    calls: list[str] = []
    original = _gather.gather_multirecord_mask

    def spy(*args, **kwargs):
        calls.append("mask")
        return original(*args, **kwargs)

    monkeypatch.setattr(_gather, "gather_multirecord_mask", spy)
    store = colstore.open(path)
    try:
        config.set_mask_density_gate(2.0)  # explicit: route disabled -> lowered path
        lowered = store[mask].edit().dict()
        assert np.array_equal(lowered["a"], full["a"][mask])
        assert calls == []
        config.set_mask_density_gate("auto")
        config._set_auto_mask_density(0.1)  # below this mask's ~0.5 density
        native = store[mask].edit().dict()
        for name in ("a", "b"):
            assert np.array_equal(native[name], full[name][mask])
        assert calls == ["mask", "mask"]  # one mask-native gather per column
    finally:
        store.close()
