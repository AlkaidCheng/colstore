"""Tests for the ``colstore`` CLI console script."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colstore import autotune, cli, config


@pytest.fixture()
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    config._set_auto_prefetch_table(None)
    yield
    config._set_auto_prefetch_table(None)


@pytest.fixture()
def _recorded_runs(monkeypatch):
    """Replace both calibrations with recorders; return the call log."""
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_threads(**kwargs):
        calls.append(("threads", kwargs))
        return 4

    def fake_prefetch(**kwargs):
        calls.append(("prefetch", kwargs))
        return {r: 0 for r in autotune._PREFETCH_REGIMES}

    def fake_mask_density(**kwargs):
        calls.append(("mask-density", kwargs))
        return 0.085

    monkeypatch.setattr(autotune, "calibrate", fake_threads)
    monkeypatch.setattr(autotune, "calibrate_prefetch", fake_prefetch)
    monkeypatch.setattr(autotune, "calibrate_mask_density", fake_mask_density)
    return calls


# ---- calibration run (and the `calibrate` alias) -------------------------


@pytest.mark.parametrize("argv_prefix", [["calibration", "run"], ["calibrate"]])
def test_run_default_runs_all_targets_in_dependency_order(
    _isolated_cache, _recorded_runs, capsys, argv_prefix
):
    assert cli.main(argv_prefix) == 0
    # Registry order: the prefetch sweep runs at the configured cap, so the
    # cap must be calibrated first; the mask-density sweep reads through
    # routes whose timing depends on both, so it runs last.
    assert [name for name, _ in _recorded_runs] == ["threads", "prefetch", "mask-density"]
    assert "-> 4" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (["threads"], ["threads"]),
        (["prefetch"], ["prefetch"]),
        (["mask-density"], ["mask-density"]),
    ],
)
def test_run_target_selection(_isolated_cache, _recorded_runs, targets, expected):
    assert cli.main(["calibration", "run", *targets]) == 0
    assert [name for name, _ in _recorded_runs] == expected


def test_run_subset_never_reorders_execution(_isolated_cache, _recorded_runs):
    # Targets given out of order on the command line still execute in
    # registry (dependency) order.
    assert cli.main(["calibration", "run", "mask-density", "prefetch", "threads"]) == 0
    assert [name for name, _ in _recorded_runs] == ["threads", "prefetch", "mask-density"]


def test_run_rejects_unknown_target(_isolated_cache):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["calibration", "run", "warp-drive"])
    assert excinfo.value.code == 2  # argparse usage error


def test_run_global_rounds_applies_to_all_targets(_isolated_cache, _recorded_runs):
    assert cli.main(["calibration", "run", "--rounds", "11"]) == 0
    assert all(kwargs["rounds"] == 11 for _, kwargs in _recorded_runs)


def test_run_per_target_rounds_overrides_global(_isolated_cache, _recorded_runs):
    argv = [
        "calibration",
        "run",
        "--rounds",
        "11",
        "--prefetch-rounds",
        "13",
        "--mask-density-rounds",
        "7",
    ]
    assert cli.main(argv) == 0
    rounds = {name: kwargs["rounds"] for name, kwargs in _recorded_runs}
    assert rounds == {"threads": 11, "prefetch": 13, "mask-density": 7}


def test_run_default_rounds_come_from_autotune(_isolated_cache, _recorded_runs):
    assert cli.main(["calibration", "run"]) == 0
    assert all(kwargs["rounds"] == autotune._CALIB_ROUNDS for _, kwargs in _recorded_runs)


def test_run_no_persist_forwarded(_isolated_cache, _recorded_runs):
    assert cli.main(["calibration", "run", "--no-persist"]) == 0
    assert all(kwargs["persist"] is False for _, kwargs in _recorded_runs)


# ---- calibration clear ----------------------------------------------------


def test_clear_removes_all_caches_and_is_idempotent(_isolated_cache, capsys):
    autotune._write_cache(4, {1: 1.0, 4: 2.0})
    autotune._write_prefetch_cache({r: 8 for r in autotune._PREFETCH_REGIMES}, {})
    autotune._write_mask_density_cache(0.1, {}, {})

    assert cli.main(["calibration", "clear"]) == 0
    out = capsys.readouterr().out
    assert "threads: removed" in out
    assert "prefetch: removed" in out
    assert "mask-density: removed" in out
    assert autotune.load_cached_cap() is None
    assert autotune.load_cached_prefetch() is None
    assert autotune.load_cached_mask_density() is None

    assert cli.main(["calibration", "clear"]) == 0
    out = capsys.readouterr().out
    assert "threads: no cache present" in out
    assert "prefetch: no cache present" in out
    assert "mask-density: no cache present" in out


def test_clear_target_selection(_isolated_cache, capsys):
    autotune._write_cache(4, {1: 1.0})
    autotune._write_prefetch_cache({r: 8 for r in autotune._PREFETCH_REGIMES}, {})

    assert cli.main(["calibration", "clear", "prefetch"]) == 0
    assert autotune.load_cached_prefetch() is None
    assert autotune.load_cached_cap() == 4  # untouched

    assert cli.main(["calibration", "clear", "threads"]) == 0
    assert autotune.load_cached_cap() is None


# ---- calibration show -----------------------------------------------------


def test_show_reports_absent_then_fingerprint_match(_isolated_cache, capsys):
    assert cli.main(["calibration", "show"]) == 0
    out = capsys.readouterr().out
    assert "(absent)" in out

    autotune._write_cache(4, {1: 1.0})
    assert cli.main(["calibration", "show"]) == 0
    out = capsys.readouterr().out
    assert "MATCHES this machine" in out
    assert "applied value: 4" in out


def test_show_flags_foreign_fingerprint(_isolated_cache, capsys):
    import json

    autotune._write_cache(4, {1: 1.0})
    path = autotune._cache_path()
    payload = json.loads(path.read_text())
    payload["fingerprint"] = "some-other-machine"
    path.write_text(json.dumps(payload))

    assert cli.main(["calibration", "show", "threads"]) == 0
    out = capsys.readouterr().out
    assert "DIFFERENT machine" in out
    assert "applied value: None" in out


# ---- structure ------------------------------------------------------------


def test_registry_covers_all_subcommand_surface():
    # The registry is the single source of truth: every target must expose
    # the full interface the subcommands render from.
    from colstore.cli import calibration

    assert [t.name for t in calibration._TARGETS] == ["threads", "prefetch", "mask-density"]
    for target in calibration._TARGETS:
        assert callable(target.run)
        assert callable(target.clear)
        assert callable(target.cache_path)
        assert callable(target.load)
        assert target.default_rounds() >= 1


def test_vague_names_are_not_top_level_commands():
    # Top-level names are reserved for groups and explicit verbs; generic
    # verbs like `clear`/`show` must stay namespaced under their group so the
    # surface can grow without collisions.
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["clear"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["show"])
    assert excinfo.value.code == 2


# ---- entry point ----------------------------------------------------------


def test_console_script_entry_point():
    exe = shutil.which("colstore")
    if exe is None:
        pytest.skip("colstore console script not on PATH (package not pip-installed)")
    result = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "calibration" in result.stdout
