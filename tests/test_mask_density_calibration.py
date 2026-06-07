"""Tests for the mask-density gate calibration (``mask-density`` target).

Crossover rule: the gate is the midpoint between the smallest grid density
that beats the lowered route by the win margin at itself and every denser
grid point, and the previous grid point (0 for the first); no qualifying
density disables the route (gate 1.0). Resolution precedence: an explicit
``set_mask_density_gate`` value, then the calibrated per-host gate (cache
present, fingerprint matching, loaded lazily once), then the compiled
default. Clearing the cache resets the in-process state. The reader's mask
routing consults the resolved gate on every read.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import colstore
from colstore import _gather, autotune
from colstore import config as config_mod
from colstore.kernels import cpp_available


@pytest.fixture()
def _isolated_gate(tmp_path, monkeypatch):
    """Isolate the cache dir and reset the gate state around each test."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    config_mod.set_mask_density_gate("auto")
    config_mod._set_auto_mask_density(None)
    yield
    config_mod.set_mask_density_gate("auto")
    config_mod._set_auto_mask_density(None)


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
    assert config_mod.resolve_mask_density_gate() == config_mod._MASK_DENSITY_GATE_DEFAULT


def test_explicit_gate_overrides_everything(_isolated_gate):
    config_mod._set_auto_mask_density(0.05)
    config_mod.set_mask_density_gate(0.4)
    assert config_mod.resolve_mask_density_gate() == 0.4
    config_mod.set_mask_density_gate(2.0)  # disable value is legal
    assert config_mod.resolve_mask_density_gate() == 2.0


def test_auto_uses_calibrated_gate(_isolated_gate):
    config_mod._set_auto_mask_density(0.065)
    assert config_mod.resolve_mask_density_gate() == 0.065


def test_set_gate_validation(_isolated_gate):
    for bad in (-0.1, "fast", None, True):
        with pytest.raises(ValueError):
            config_mod.set_mask_density_gate(bad)  # type: ignore[arg-type]
    config_mod.set_mask_density_gate("auto")  # round-trips
    assert config_mod.get_mask_density_gate() == "auto"


def test_auto_lazy_loads_cache_once(_isolated_gate):
    autotune._write_mask_density_cache(0.085, {}, {})
    assert config_mod.resolve_mask_density_gate() == 0.085
    # Cache file removal without reset does not affect the loaded value...
    autotune._mask_density_cache_path().unlink()
    assert config_mod.resolve_mask_density_gate() == 0.085
    # ...while clear with reset_in_process drops it immediately.
    autotune.clear_cached_mask_density()
    assert config_mod.resolve_mask_density_gate() == config_mod._MASK_DENSITY_GATE_DEFAULT


def test_foreign_fingerprint_cache_ignored(_isolated_gate):
    autotune._write_mask_density_cache(0.02, {}, {})
    path = autotune._mask_density_cache_path()
    payload = json.loads(path.read_text())
    payload["fingerprint"] = "some-other-machine"
    path.write_text(json.dumps(payload))
    assert autotune.load_cached_mask_density() is None
    assert config_mod.resolve_mask_density_gate() == config_mod._MASK_DENSITY_GATE_DEFAULT


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
    assert config_mod.resolve_mask_density_gate() == gate
    assert autotune.load_cached_mask_density() == gate
    payload = json.loads(autotune._mask_density_cache_path().read_text())
    assert payload["fingerprint"] == autotune._hardware_fingerprint()
    assert set(payload["ratios"]) == {str(d) for d in autotune._MASK_DENSITY_GRID}
    # Clear restores the default both on disk and in-process.
    assert autotune.clear_cached_mask_density() is True
    assert config_mod.resolve_mask_density_gate() == config_mod._MASK_DENSITY_GATE_DEFAULT
    assert autotune.clear_cached_mask_density() is False  # idempotent


@pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")
@pytest.mark.filterwarnings("ignore:calibration for 'mask-density' is unstable")
def test_calibrate_no_persist_applies_in_process_only(_isolated_gate, monkeypatch):
    monkeypatch.setattr(autotune, "_MASK_CALIB_N_RECORDS", 40)
    monkeypatch.setattr(autotune, "_MASK_CALIB_ROWS_PER_RECORD", 100)
    gate = autotune.calibrate_mask_density(rounds=2, persist=False)
    assert config_mod.resolve_mask_density_gate() == gate
    assert autotune.load_cached_mask_density() is None
    assert not autotune._mask_density_cache_path().exists()


@pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")
@pytest.mark.filterwarnings("ignore:calibration for 'mask-density' is unstable")
def test_calibrate_restores_gate_setting(_isolated_gate, monkeypatch):
    monkeypatch.setattr(autotune, "_MASK_CALIB_N_RECORDS", 40)
    monkeypatch.setattr(autotune, "_MASK_CALIB_ROWS_PER_RECORD", 100)
    config_mod.set_mask_density_gate(0.33)
    autotune.calibrate_mask_density(rounds=2, persist=False)
    # The sweep toggles the gate internally but must restore the setting.
    assert config_mod.get_mask_density_gate() == 0.33


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
        config_mod.set_mask_density_gate(2.0)  # explicit: route disabled
        assert np.array_equal(dataset[mask, "a"].array(), full["a"][mask])
        assert calls == []
        config_mod.set_mask_density_gate("auto")
        config_mod._set_auto_mask_density(0.1)  # calibrated: below this mask's density
        assert np.array_equal(dataset[mask, "a"].array(), full["a"][mask])
        assert calls == ["mask"]
    finally:
        dataset.close()
