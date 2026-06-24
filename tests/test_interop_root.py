"""Tests for the ROOT file format (``colstore.interop.root``) and its two kernels.

Both kernels are exercised against real backends where installed (PyROOT for
``kernel="ROOT"``, uproot for ``kernel="uproot"``); each test is parametrized over
the available kernels and skipped when neither is present. Covers round-trip
parity, streaming in batches, cross-kernel interchange, the ``keep_valid_only``
policy, column selection, kernel selection and errors, dataset export, and the
``RootFormat`` (saveas / ingest) wiring.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import colstore
from colstore import interop
from colstore.interop import root as root_mod

_HAS_ROOT = importlib.util.find_spec("ROOT") is not None
_HAS_UPROOT = importlib.util.find_spec("uproot") is not None
_KERNELS = (["ROOT"] if _HAS_ROOT else []) + (["uproot"] if _HAS_UPROOT else [])

pytestmark = pytest.mark.skipif(not _KERNELS, reason="neither PyROOT nor uproot is installed")


@pytest.fixture
def columns():
    return {
        "pt": np.arange(20, dtype=np.float64),
        "eta": np.linspace(-2.5, 2.5, 20).astype(np.float32),
        "n": np.arange(20, dtype=np.int32),
        "ok": (np.arange(20) % 2 == 0),
    }


@pytest.fixture
def store(tmp_path, columns):
    return colstore.store(columns, tmp_path / "s.cstore", show_progress=False)


def _assert_same_columns(reader, expected):
    assert set(reader.columns) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(np.sort(reader.array(name)), np.sort(values))


# ---- round-trip parity, both kernels ---------------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
def test_roundtrip(tmp_path, store, columns, kernel):
    root_path = tmp_path / "out.root"
    result = colstore.to_root(
        store, root_path, kernel=kernel, treename="events", show_progress=False
    )
    assert isinstance(result, Path) and result == root_path and root_path.exists()
    back = colstore.from_root(
        root_path, tmp_path / "back.cstore", kernel=kernel, show_progress=False
    )
    _assert_same_columns(back, columns)
    back.close()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_streaming_batches_roundtrip(tmp_path, store, columns, kernel):
    # a small batch_size forces multiple write/read batches
    root_path = tmp_path / "b.root"
    colstore.to_root(
        store, root_path, kernel=kernel, treename="t", batch_size=7, show_progress=False
    )
    back = colstore.from_root(
        root_path, tmp_path / "b.cstore", kernel=kernel, batch_size=7, show_progress=False
    )
    _assert_same_columns(back, columns)
    back.close()


@pytest.mark.skipif(not (_HAS_ROOT and _HAS_UPROOT), reason="needs both kernels")
def test_cross_kernel_interchange(tmp_path, store, columns):
    # write with each kernel, read back with the other
    for writer, reader in (("ROOT", "uproot"), ("uproot", "ROOT")):
        root_path = tmp_path / f"{writer}.root"
        colstore.to_root(store, root_path, kernel=writer, treename="events", show_progress=False)
        back = colstore.from_root(
            root_path, tmp_path / f"{writer}_{reader}.cstore", kernel=reader, show_progress=False
        )
        _assert_same_columns(back, columns)
        back.close()


# ---- column selection -------------------------------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
def test_column_selection(tmp_path, store, columns, kernel):
    root_path = tmp_path / "sel.root"
    colstore.to_root(store, root_path, kernel=kernel, columns=["pt", "n"], show_progress=False)
    back = colstore.from_root(
        root_path, tmp_path / "sel.cstore", kernel=kernel, columns=["pt"], show_progress=False
    )
    assert back.columns == ["pt"]
    np.testing.assert_array_equal(np.sort(back.array("pt")), columns["pt"])
    back.close()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_missing_export_column_raises(tmp_path, store, kernel):
    with pytest.raises(ValueError, match="not found"):
        colstore.to_root(store, tmp_path / "x.root", kernel=kernel, columns=["nope"])


# ---- keep_valid_only on a file with a jagged branch ------------------------


def _jagged_file(path):
    """Write a ROOT TTree with a scalar ``pt`` and a jagged ``jag`` branch.

    Needs uproot + awkward. uproot also emits an ``njag`` counter branch (a valid
    int32 scalar), so the storable set is ``{pt, njag}`` and ``jag`` is skipped.
    """
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")
    with uproot.recreate(str(path)) as f:
        f.mktree(
            "events",
            {"pt": np.float64, "jag": ak.types.from_datashape("var * float64", highlevel=False)},
        )
        f["events"].extend(
            {
                "pt": np.arange(6, dtype=np.float64),
                "jag": ak.Array([[1.0, 2.0], [3.0], [], [4.0], [5.0, 6.0], [7.0]]),
            }
        )


@pytest.mark.parametrize("kernel", _KERNELS)
def test_keep_valid_only_skips_jagged(tmp_path, kernel):
    src = tmp_path / "jag.root"
    _jagged_file(src)
    with pytest.warns(RuntimeWarning, match="non-fixed-size"):
        back = colstore.from_root(src, tmp_path / "k.cstore", kernel=kernel, show_progress=False)
    assert "jag" not in back.columns  # the jagged branch is skipped
    assert "pt" in back.columns  # the scalar branch is kept
    np.testing.assert_array_equal(np.sort(back.array("pt")), np.arange(6, dtype=np.float64))
    back.close()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_keep_valid_only_false_raises(tmp_path, kernel):
    src = tmp_path / "jag.root"
    _jagged_file(src)
    with pytest.raises(ValueError, match="cannot be stored"):
        colstore.from_root(src, tmp_path / "k.cstore", kernel=kernel, keep_valid_only=False)


@pytest.mark.parametrize("kernel", _KERNELS)
def test_keep_valid_only_skips_requested_invalid(tmp_path, kernel):
    # the unified policy skips an explicitly requested invalid column too (with a warning)
    src = tmp_path / "jag.root"
    _jagged_file(src)
    with pytest.warns(RuntimeWarning, match="non-fixed-size"):
        back = colstore.from_root(
            src, tmp_path / "k.cstore", kernel=kernel, columns=["pt", "jag"], show_progress=False
        )
    assert back.columns == ["pt"]
    back.close()


# ---- dtype fidelity and storability (review regressions) -------------------


@pytest.mark.parametrize("kernel", _KERNELS)
def test_missing_read_column_raises(tmp_path, store, kernel):
    # a typo'd column must raise, not be silently skipped under keep_valid_only=True
    root_path = tmp_path / "m.root"
    colstore.to_root(store, root_path, kernel=kernel, treename="events", show_progress=False)
    with pytest.raises(ValueError, match="not found in the ROOT tree"):
        colstore.from_root(root_path, tmp_path / "m.cstore", kernel=kernel, columns=["pt", "nope"])


def test_root_kernel_rejects_int8(tmp_path):
    # ROOT's Snapshot cannot emit an 8-bit integer; the ROOT kernel must reject it
    # (not segfault/silently drop), while the uproot kernel writes it fine.
    store = colstore.store(
        {"q": np.arange(-3, 3, dtype=np.int8)}, tmp_path / "i.cstore", show_progress=False
    )
    if _HAS_ROOT:
        with pytest.raises(TypeError, match="8-bit integer"):
            colstore.to_root(store, tmp_path / "i.root", kernel="ROOT", show_progress=False)
    if _HAS_UPROOT:
        colstore.to_root(store, tmp_path / "iu.root", kernel="uproot", show_progress=False)
        back = colstore.ingest(
            tmp_path / "iu.root", tmp_path / "iu.cstore", kernel="uproot", show_progress=False
        )
        assert back.dtypes["q"] == np.int8
        np.testing.assert_array_equal(back.array("q"), np.arange(-3, 3, dtype=np.int8))
        back.close()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_reads_cstdint_integer_branches(tmp_path, kernel):
    # uproot's f[name]={dict} form writes branches RDataFrame reports as
    # std::int32_t etc.; storability is judged by dtype, so integers are kept.
    uproot = pytest.importorskip("uproot")
    src = tmp_path / "ints.root"
    # f[name]={dict} writes an RNTuple; pass treename since RNTuple isn't auto-detected
    with uproot.recreate(str(src)) as f:
        f["events"] = {
            "f64": np.arange(6, dtype=np.float64),
            "i64": np.arange(6, dtype=np.int64),
            "u16": np.arange(6, dtype=np.uint16),
        }
    back = colstore.from_root(
        src, tmp_path / "ints.cstore", kernel=kernel, treename="events", show_progress=False
    )
    assert set(back.columns) == {"f64", "i64", "u16"}
    assert back.dtypes["i64"] == np.int64 and back.dtypes["u16"] == np.uint16
    back.close()


# ---- kernel selection and errors -------------------------------------------


def test_unknown_kernel_raises(tmp_path, store):
    with pytest.raises(ValueError, match="unknown kernel"):
        colstore.to_root(store, tmp_path / "x.root", kernel="bogus")


@pytest.mark.parametrize("kernel", _KERNELS)
def test_auto_kernel_roundtrip(tmp_path, store, columns, kernel):
    # 'auto' resolves to an available backend; exercise the default path
    root_path = tmp_path / "auto.root"
    colstore.to_root(store, root_path, show_progress=False)  # kernel='auto'
    back = colstore.from_root(root_path, tmp_path / "auto.cstore", show_progress=False)
    _assert_same_columns(back, columns)
    back.close()


def test_resolve_kernel_names():
    if _HAS_ROOT:
        assert root_mod.resolve_kernel("ROOT").name == "ROOT"
        assert root_mod.resolve_kernel("root").name == "ROOT"  # case-insensitive
    if _HAS_UPROOT:
        assert root_mod.resolve_kernel("uproot").name == "uproot"
    assert root_mod.resolve_kernel("auto").name in {"ROOT", "uproot"}


# ---- RootFormat: saveas / ingest -------------------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
def test_saveas_and_ingest(tmp_path, store, columns, kernel):
    root_path = tmp_path / "f.root"
    store.saveas(root_path, kernel=kernel, treename="events", show_progress=False)
    assert root_path.exists()
    back = colstore.ingest(root_path, tmp_path / "f.cstore", kernel=kernel, show_progress=False)
    _assert_same_columns(back, columns)
    back.close()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_saveas_row_subset(tmp_path, store, columns, kernel):
    root_path = tmp_path / "sub.root"
    store[5:10, ["pt", "n"]].saveas(root_path, kernel=kernel, show_progress=False)
    back = colstore.ingest(root_path, tmp_path / "sub.cstore", kernel=kernel, show_progress=False)
    assert set(back.columns) == {"pt", "n"}
    assert back.n_rows == 5
    np.testing.assert_array_equal(np.sort(back.array("pt")), columns["pt"][5:10])
    back.close()


def test_root_format_registered():
    assert "root" in interop.file_formats()
    assert interop.file_format_for_extension(".root").name == "root"


def test_import_colstore_does_not_load_backends():
    import subprocess
    import sys

    code = (
        "import colstore, sys; " "assert 'ROOT' not in sys.modules and 'uproot' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---- dataset export and name sanitization ----------------------------------


@pytest.mark.parametrize("kernel", _KERNELS)
def test_dataset_export(tmp_path, kernel):
    paths = []
    for i in range(2):
        p = tmp_path / f"part{i}.cstore"
        colstore.store(
            {"x": np.arange(i * 5, i * 5 + 5, dtype=np.int64)}, p, show_progress=False
        ).close()
        paths.append(p)
    ds = colstore.open(paths)
    try:
        root_path = tmp_path / "ds.root"
        colstore.to_root(ds, root_path, kernel=kernel, treename="t", show_progress=False)
        back = colstore.from_root(
            root_path, tmp_path / "ds.cstore", kernel=kernel, show_progress=False
        )
        np.testing.assert_array_equal(np.sort(back.array("x")), np.arange(10))
        back.close()
    finally:
        ds.close()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_branch_name_sanitization(tmp_path, kernel):
    store = colstore.store(
        {"good": np.arange(4, dtype=np.int64), "bad name": np.arange(4, dtype=np.float64)},
        tmp_path / "names.cstore",
        show_progress=False,
    )
    root_path = tmp_path / "names.root"
    with pytest.warns(RuntimeWarning, match="Sanitized"):
        colstore.to_root(store, root_path, kernel=kernel, show_progress=False)
    back = colstore.from_root(
        root_path, tmp_path / "names_back.cstore", kernel=kernel, show_progress=False
    )
    assert "bad_name" in back.columns  # sanitized to a valid branch name
    back.close()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_empty_store_roundtrip(tmp_path, kernel):
    store = colstore.store(
        {"a": np.empty(0, np.int64), "b": np.empty(0, np.float64)},
        tmp_path / "empty.cstore",
        show_progress=False,
    )
    root_path = tmp_path / "empty.root"
    colstore.to_root(store, root_path, kernel=kernel, treename="t", show_progress=False)
    back = colstore.from_root(
        root_path, tmp_path / "empty_back.cstore", kernel=kernel, show_progress=False
    )
    assert back.n_rows == 0
    assert set(back.columns) == {"a", "b"}
    back.close()
