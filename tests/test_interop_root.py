"""Tests for the ROOT file format (``colstore.interop.root``) and its two backends.

Both backends are exercised against real backends where installed (PyROOT for
``backend="ROOT"``, uproot for ``backend="uproot"``); each test is parametrized over
the available backends and skipped when neither is present. Covers round-trip
parity, streaming in batches, cross-backend interchange, the ``keep_valid_only``
policy, column selection, backend selection and errors, dataset export, and the
``RootFormat`` (saveas / convert) wiring.
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
_BACKENDS = (["ROOT"] if _HAS_ROOT else []) + (["uproot"] if _HAS_UPROOT else [])

pytestmark = pytest.mark.skipif(not _BACKENDS, reason="neither PyROOT nor uproot is installed")


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


# ---- round-trip parity, both backends ---------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_roundtrip(tmp_path, store, columns, backend):
    root_path = tmp_path / "out.root"
    result = colstore.to_root(
        store, root_path, backend=backend, treename="events", show_progress=False
    )
    assert isinstance(result, Path) and result == root_path and root_path.exists()
    back = colstore.from_root(
        root_path, tmp_path / "back.cstore", backend=backend, show_progress=False
    )
    _assert_same_columns(back, columns)
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_streaming_batches_roundtrip(tmp_path, store, columns, backend):
    # a small batch_size forces multiple write/read batches
    root_path = tmp_path / "b.root"
    colstore.to_root(
        store, root_path, backend=backend, treename="t", batch_size=7, show_progress=False
    )
    back = colstore.from_root(
        root_path, tmp_path / "b.cstore", backend=backend, batch_size=7, show_progress=False
    )
    _assert_same_columns(back, columns)
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_read_whole_vs_chunked_agree(tmp_path, backend):
    # The whole-file read (a single AsNumpy, implicit-MT for the ROOT backend) and
    # the chunked Range read must reconstruct identical, row-intact data. Sorting by
    # a key column and checking a correlated column catches any cross-column mixup
    # (which per-column sorting in the other tests would miss).
    n = 5000
    data = {"key": np.arange(n, dtype=np.int64), "val": (np.arange(n) * 2).astype(np.float64)}
    src = colstore.store(data, tmp_path / "rw.cstore", show_progress=False)
    rp = tmp_path / "rw.root"
    colstore.to_root(src, rp, treename="events", backend=backend, show_progress=False)

    whole = colstore.from_root(rp, tmp_path / "whole.cstore", backend=backend, show_progress=False)
    chunked = colstore.from_root(
        rp, tmp_path / "chunk.cstore", backend=backend, batch_size=1500, show_progress=False
    )
    for back in (whole, chunked):
        assert back.n_rows == n
        order = np.argsort(np.asarray(back.array("key")))
        np.testing.assert_array_equal(np.asarray(back.array("key"))[order], np.arange(n))
        np.testing.assert_array_equal(
            np.asarray(back.array("val"))[order], (np.arange(n) * 2).astype(np.float64)
        )
    whole.close()
    chunked.close()


# ---- multiple files ---------------------------------------------------------


def _two_files(tmp_path, backend, splits=((0, 600), (600, 1500))):
    """Write two ROOT files sharing one tree over disjoint key ranges; return paths."""
    files = []
    for i, (lo, hi) in enumerate(splits):
        s = colstore.store(
            {
                "key": np.arange(lo, hi, dtype=np.int64),
                "val": (np.arange(lo, hi) * 2).astype(np.float64),
            },
            tmp_path / f"s{i}.cstore",
            show_progress=False,
        )
        p = tmp_path / f"f{i}.root"
        colstore.to_root(s, p, treename="events", backend=backend, show_progress=False)
        files.append(p)
    return files


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("form", ["list", "dict", "list_embedded"])
def test_read_multiple_files(tmp_path, backend, form):
    # Two files with the same tree, read as one combined, row-intact table.
    files = _two_files(tmp_path, backend)
    if form == "list":
        source, treename = [str(f) for f in files], "events"
    elif form == "dict":
        source, treename = {"events": [str(f) for f in files]}, None
    else:  # paths carry their own ":events"
        source, treename = [f"{f}:events" for f in files], None
    back = colstore.from_root(
        source, tmp_path / "out.cstore", backend=backend, treename=treename, show_progress=False
    )
    assert back.n_rows == 1500
    order = np.argsort(np.asarray(back.array("key")))
    np.testing.assert_array_equal(np.asarray(back.array("key"))[order], np.arange(1500))
    np.testing.assert_array_equal(
        np.asarray(back.array("val"))[order], (np.arange(1500) * 2).astype(np.float64)
    )
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_read_file_list_conflicting_trees_raises(tmp_path, store, backend):
    rp = tmp_path / "x.root"
    colstore.to_root(store, rp, treename="events", backend=backend, show_progress=False)
    with pytest.raises(ValueError, match="multiple trees"):
        colstore.from_root(
            [f"{rp}:events", f"{rp}:other"],
            tmp_path / "o.cstore",
            backend=backend,
            show_progress=False,
        )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_read_empty_file_list_raises(tmp_path, backend):
    with pytest.raises(ValueError, match="empty file list"):
        colstore.from_root([], tmp_path / "o.cstore", backend=backend, show_progress=False)


def test_group_files_by_budget():
    assert root_mod._group_files_by_budget([1000, 1000, 1000, 1000], 2500) == [[0, 1], [2, 3]]
    assert root_mod._group_files_by_budget([1000, 1000, 1000, 1000], 1500) == [[0], [1], [2], [3]]
    assert root_mod._group_files_by_budget([3000, 500, 500], 1000) == [
        [0],
        [1, 2],
    ]  # big file alone
    assert root_mod._group_files_by_budget([], 1000) == []


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("batch_size", [500, 100])
def test_read_multifile_over_budget(tmp_path, backend, batch_size):
    # Four files x 200 rows; a budget below the total exercises file-group chunking
    # (batch_size=500: groups of two files; batch_size=100: each 200-row file exceeds
    # the budget, so the ROOT backend falls back to Range within it). Stays row-intact.
    files = []
    for i in range(4):
        lo, hi = i * 200, i * 200 + 200
        s = colstore.store(
            {
                "key": np.arange(lo, hi, dtype=np.int64),
                "val": (np.arange(lo, hi) * 3).astype(np.float64),
            },
            tmp_path / f"s{i}.cstore",
            show_progress=False,
        )
        p = tmp_path / f"f{i}.root"
        colstore.to_root(s, p, treename="events", backend=backend, show_progress=False)
        files.append(str(p))
    back = colstore.from_root(
        files,
        tmp_path / "o.cstore",
        backend=backend,
        treename="events",
        batch_size=batch_size,
        show_progress=False,
    )
    assert back.n_rows == 800
    order = np.argsort(np.asarray(back.array("key")))
    np.testing.assert_array_equal(np.asarray(back.array("key"))[order], np.arange(800))
    np.testing.assert_array_equal(
        np.asarray(back.array("val"))[order], (np.arange(800) * 3).astype(np.float64)
    )
    back.close()


@pytest.mark.skipif(not (_HAS_ROOT and _HAS_UPROOT), reason="needs both backends")
def test_cross_backend_interchange(tmp_path, store, columns):
    # write with each backend, read back with the other
    for writer, reader in (("ROOT", "uproot"), ("uproot", "ROOT")):
        root_path = tmp_path / f"{writer}.root"
        colstore.to_root(store, root_path, backend=writer, treename="events", show_progress=False)
        back = colstore.from_root(
            root_path, tmp_path / f"{writer}_{reader}.cstore", backend=reader, show_progress=False
        )
        _assert_same_columns(back, columns)
        back.close()


# ---- column selection -------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_column_selection(tmp_path, store, columns, backend):
    root_path = tmp_path / "sel.root"
    colstore.to_root(store, root_path, backend=backend, columns=["pt", "n"], show_progress=False)
    back = colstore.from_root(
        root_path, tmp_path / "sel.cstore", backend=backend, columns=["pt"], show_progress=False
    )
    assert back.columns == ["pt"]
    np.testing.assert_array_equal(np.sort(back.array("pt")), columns["pt"])
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_missing_export_column_raises(tmp_path, store, backend):
    with pytest.raises(ValueError, match="not found"):
        colstore.to_root(store, tmp_path / "x.root", backend=backend, columns=["nope"])


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


@pytest.mark.parametrize("backend", _BACKENDS)
def test_keep_valid_only_skips_jagged(tmp_path, backend):
    src = tmp_path / "jag.root"
    _jagged_file(src)
    with pytest.warns(RuntimeWarning, match="non-fixed-size"):
        back = colstore.from_root(src, tmp_path / "k.cstore", backend=backend, show_progress=False)
    assert "jag" not in back.columns  # the jagged branch is skipped
    assert "pt" in back.columns  # the scalar branch is kept
    np.testing.assert_array_equal(np.sort(back.array("pt")), np.arange(6, dtype=np.float64))
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_keep_valid_only_false_raises(tmp_path, backend):
    src = tmp_path / "jag.root"
    _jagged_file(src)
    with pytest.raises(ValueError, match="cannot be stored"):
        colstore.from_root(src, tmp_path / "k.cstore", backend=backend, keep_valid_only=False)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_keep_valid_only_skips_requested_invalid(tmp_path, backend):
    # the unified policy skips an explicitly requested invalid column too (with a warning)
    src = tmp_path / "jag.root"
    _jagged_file(src)
    with pytest.warns(RuntimeWarning, match="non-fixed-size"):
        back = colstore.from_root(
            src, tmp_path / "k.cstore", backend=backend, columns=["pt", "jag"], show_progress=False
        )
    assert back.columns == ["pt"]
    back.close()


# ---- dtype fidelity and storability ----------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_missing_read_column_raises(tmp_path, store, backend):
    # a typo'd column must raise, not be silently skipped under keep_valid_only=True
    root_path = tmp_path / "m.root"
    colstore.to_root(store, root_path, backend=backend, treename="events", show_progress=False)
    with pytest.raises(ValueError, match="not found in the ROOT tree"):
        colstore.from_root(
            root_path, tmp_path / "m.cstore", backend=backend, columns=["pt", "nope"]
        )


def test_root_backend_rejects_int8(tmp_path):
    # ROOT's Snapshot cannot emit an 8-bit integer; the ROOT backend must reject it
    # (not segfault/silently drop), while the uproot backend writes it fine.
    store = colstore.store(
        {"q": np.arange(-3, 3, dtype=np.int8)}, tmp_path / "i.cstore", show_progress=False
    )
    if _HAS_ROOT:
        with pytest.raises(TypeError, match="8-bit integer"):
            colstore.to_root(store, tmp_path / "i.root", backend="ROOT", show_progress=False)
    if _HAS_UPROOT:
        colstore.to_root(store, tmp_path / "iu.root", backend="uproot", show_progress=False)
        back = colstore.convert(
            tmp_path / "iu.root", tmp_path / "iu.cstore", backend="uproot", show_progress=False
        )
        assert back.dtypes["q"] == np.int8
        np.testing.assert_array_equal(back.array("q"), np.arange(-3, 3, dtype=np.int8))
        back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_root_rejects_string_column(tmp_path, backend):
    # ROOT branches have no fixed-width string type; reject cleanly, not crash.
    store = colstore.store(
        {"s": np.array(["a", "bb", "c"]), "n": np.arange(3, dtype=np.int64)},
        tmp_path / "str.cstore",
        show_progress=False,
    )
    with pytest.raises(TypeError, match="string column"):
        colstore.to_root(store, tmp_path / "str.root", backend=backend, show_progress=False)


# The RNTuple this writes triggers a uproot-internal filter_branch DeprecationWarning on read
# (uproot 5.7 raises it even with no column selection); it is uproot's to fix, not ours.
@pytest.mark.filterwarnings("ignore:the filter_branch kwarg:DeprecationWarning")
@pytest.mark.parametrize("backend", _BACKENDS)
def test_reads_cstdint_integer_branches(tmp_path, backend):
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
        src, tmp_path / "ints.cstore", backend=backend, treename="events", show_progress=False
    )
    assert set(back.columns) == {"f64", "i64", "u16"}
    assert back.dtypes["i64"] == np.int64 and back.dtypes["u16"] == np.uint16
    back.close()


# ---- backend selection and errors -------------------------------------------


def test_unknown_backend_raises(tmp_path, store):
    with pytest.raises(ValueError, match="unknown backend"):
        colstore.to_root(store, tmp_path / "x.root", backend="bogus")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_auto_backend_roundtrip(tmp_path, store, columns, backend):
    # 'auto' resolves to an available backend; exercise the default path
    root_path = tmp_path / "auto.root"
    colstore.to_root(store, root_path, show_progress=False)  # backend='auto'
    back = colstore.from_root(root_path, tmp_path / "auto.cstore", show_progress=False)
    _assert_same_columns(back, columns)
    back.close()


def test_resolve_backend_names():
    if _HAS_ROOT:
        assert root_mod.resolve_backend("ROOT").name == "ROOT"
        assert root_mod.resolve_backend("root").name == "ROOT"  # case-insensitive
    if _HAS_UPROOT:
        assert root_mod.resolve_backend("uproot").name == "uproot"
    assert root_mod.resolve_backend("auto").name in {"ROOT", "uproot"}


# ---- RootFormat: saveas / convert -------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_saveas_and_convert(tmp_path, store, columns, backend):
    root_path = tmp_path / "f.root"
    store.saveas(root_path, backend=backend, treename="events", show_progress=False)
    assert root_path.exists()
    back = colstore.convert(root_path, tmp_path / "f.cstore", backend=backend, show_progress=False)
    _assert_same_columns(back, columns)
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_saveas_row_subset(tmp_path, store, columns, backend):
    root_path = tmp_path / "sub.root"
    store[5:10, ["pt", "n"]].saveas(root_path, backend=backend, show_progress=False)
    back = colstore.convert(
        root_path, tmp_path / "sub.cstore", backend=backend, show_progress=False
    )
    assert set(back.columns) == {"pt", "n"}
    assert back.n_rows == 5
    np.testing.assert_array_equal(np.sort(back.array("pt")), columns["pt"][5:10])
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_saveas_row_subset_chunked(tmp_path, store, columns, backend):
    # A row subset is gathered into memory and streamed straight to ROOT; a small
    # batch_size forces several chunks, exercising the in-memory source's slicing.
    root_path = tmp_path / "subchunk.root"
    store[5:15, ["pt", "n"]].saveas(root_path, backend=backend, batch_size=4, show_progress=False)
    back = colstore.convert(
        root_path, tmp_path / "subchunk.cstore", backend=backend, show_progress=False
    )
    assert set(back.columns) == {"pt", "n"}
    assert back.n_rows == 10
    np.testing.assert_array_equal(np.sort(back.array("pt")), columns["pt"][5:15])
    np.testing.assert_array_equal(np.sort(back.array("n")), columns["n"][5:15])
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


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dataset_export(tmp_path, backend):
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
        colstore.to_root(ds, root_path, backend=backend, treename="t", show_progress=False)
        back = colstore.from_root(
            root_path, tmp_path / "ds.cstore", backend=backend, show_progress=False
        )
        np.testing.assert_array_equal(np.sort(back.array("x")), np.arange(10))
        back.close()
    finally:
        ds.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_branch_name_sanitization(tmp_path, backend):
    store = colstore.store(
        {"good": np.arange(4, dtype=np.int64), "bad name": np.arange(4, dtype=np.float64)},
        tmp_path / "names.cstore",
        show_progress=False,
    )
    root_path = tmp_path / "names.root"
    with pytest.warns(RuntimeWarning, match="Sanitized"):
        colstore.to_root(store, root_path, backend=backend, show_progress=False)
    back = colstore.from_root(
        root_path, tmp_path / "names_back.cstore", backend=backend, show_progress=False
    )
    assert "bad_name" in back.columns  # sanitized to a valid branch name
    back.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_empty_store_roundtrip(tmp_path, backend):
    store = colstore.store(
        {"a": np.empty(0, np.int64), "b": np.empty(0, np.float64)},
        tmp_path / "empty.cstore",
        show_progress=False,
    )
    root_path = tmp_path / "empty.root"
    colstore.to_root(store, root_path, backend=backend, treename="t", show_progress=False)
    back = colstore.from_root(
        root_path, tmp_path / "empty_back.cstore", backend=backend, show_progress=False
    )
    assert back.n_rows == 0
    assert set(back.columns) == {"a", "b"}
    back.close()
