"""Tests for colstore.parsers.root.

ROOT is not importable in CI, so these tests drive the parser through fakes:

* the ingest path (from_root) takes a duck-typed fake RNode as its
  source and runs against the real colstore writer/reader, so the column gate,
  batch math, record mapping, and compaction are all exercised for real;
* the export path (to_root) reads a real .cstore file and writes
  through a fake ROOT module (monkeypatched _import_root), so the chunking and
  the snapshot call sequence are checked without a ROOT build.

The irreducibly-ROOT behavior -- that RDF.FromNumpy(...).Snapshot(..., UPDATE,
append) actually appends correctly -- is integration-deferred to a ROOT host.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore.parsers import RootParser, from_root, to_root
from colstore.parsers import root as root_parser

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeResult:
    def __init__(self, value: int) -> None:
        self._value = value

    def GetValue(self) -> int:
        return self._value


class _FakeRNode:
    """Minimal RDataFrame stand-in backed by in-memory numpy columns."""

    def __init__(self, data: dict[str, np.ndarray], types: dict[str, str]) -> None:
        self._data = data
        self._types = types

    def GetColumnNames(self) -> list[str]:
        return list(self._types)

    def GetColumnType(self, name: str) -> str:
        return self._types[name]

    def Count(self) -> _FakeResult:
        n = len(next(iter(self._data.values()))) if self._data else 0
        return _FakeResult(n)

    def Range(self, start: int, stop: int) -> _FakeRNode:
        return _FakeRNode({k: v[start:stop] for k, v in self._data.items()}, self._types)

    def AsNumpy(self, columns: list[str] | None = None) -> dict[str, np.ndarray]:
        names = list(self._data) if columns is None else list(columns)
        return {name: np.array(self._data[name]) for name in names}


def _make_rnode(n_rows: int = 1000, *, with_jagged: bool = False) -> _FakeRNode:
    data = {
        "px": np.linspace(0, 1, n_rows, dtype=np.float64),
        "n": np.arange(n_rows, dtype=np.int32),
    }
    types = {"px": "Double_t", "n": "Int_t"}
    if with_jagged:
        types["hits"] = "ROOT::VecOps::RVec<Float_t>"  # present but non-storable
    return _FakeRNode(data, types)


class _FakeSnapshotOptions:
    def __init__(self) -> None:
        self.fMode = ""
        self.fAppend = False


class _FakeSnapshotRDF:
    def __init__(self, data: dict[str, np.ndarray], sink: list) -> None:
        self._data = data
        self._sink = sink

    def Snapshot(self, treename: str, path: str, columns, options: _FakeSnapshotOptions) -> None:
        self._sink.append(
            {
                "treename": treename,
                "path": path,
                "columns": list(columns),
                "mode": options.fMode,
                "append": options.fAppend,
                "data": {name: np.array(values) for name, values in self._data.items()},
            }
        )


class _FakeRDFNamespace:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    def FromNumpy(self, data: dict[str, np.ndarray]) -> _FakeSnapshotRDF:
        return _FakeSnapshotRDF(data, self._sink)

    def RSnapshotOptions(self) -> _FakeSnapshotOptions:
        return _FakeSnapshotOptions()


class _FakeROOT:
    def __init__(self) -> None:
        self.snapshots: list = []
        self.opened: list = []
        self.RDF = _FakeRDFNamespace(self.snapshots)

    def RDataFrame(self, treename: str, path: str):
        self.opened.append((treename, path))
        return ("FakeRDF", treename, path)


# --------------------------------------------------------------------------- #
# Column gate
# --------------------------------------------------------------------------- #


def test_jagged_columns_skipped_with_warning():
    rnode = _make_rnode(with_jagged=True)
    with pytest.warns(RuntimeWarning, match="hits"):
        selected = root_parser._select_storable_columns(rnode, None)
    assert selected == ["px", "n"]


def test_explicit_non_storable_column_is_error():
    rnode = _make_rnode(with_jagged=True)
    with pytest.raises(ValueError, match="cannot be stored"):
        root_parser._select_storable_columns(rnode, ["px", "hits"])


def test_no_storable_columns_is_error():
    rnode = _FakeRNode({}, {"hits": "ROOT::VecOps::RVec<Float_t>"})
    with pytest.raises(ValueError, match="None of the source columns"):
        root_parser._select_storable_columns(rnode, None)


# --------------------------------------------------------------------------- #
# Ingest: from_root (fake RNode + real colstore)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("batch_size", [None, 256, "4 KiB"])
def test_ingest_roundtrip(tmp_path, batch_size):
    rnode = _make_rnode(1000)
    out = tmp_path / "events.cstore"
    reader = from_root(rnode, out, batch_size=batch_size, show_progress=False)
    assert reader.n_rows == 1000
    assert sorted(reader.columns) == ["n", "px"]
    assert np.array_equal(reader[:, "n"].array(), np.arange(1000, dtype=np.int32))
    assert np.allclose(reader[:, "px"].array(), np.linspace(0, 1, 1000))


def test_ingest_skips_jagged_and_stores_the_rest(tmp_path):
    rnode = _make_rnode(500, with_jagged=True)
    out = tmp_path / "events.cstore"
    with pytest.warns(RuntimeWarning):
        reader = from_root(rnode, out, batch_size=128, show_progress=False)
    assert sorted(reader.columns) == ["n", "px"]
    assert reader.n_rows == 500


def test_ingest_compacts_by_default(tmp_path):
    rnode = _make_rnode(1000)
    compacted = from_root(rnode, tmp_path / "c.cstore", batch_size=100, show_progress=False)
    assert colstore.info(tmp_path / "c.cstore").n_records == 1
    assert compacted.n_rows == 1000

    streamed = from_root(
        rnode, tmp_path / "s.cstore", batch_size=100, compact=False, show_progress=False
    )
    assert colstore.info(tmp_path / "s.cstore").n_records == 10
    assert streamed.n_rows == 1000


def test_ingest_zero_rows_keeps_schema(tmp_path):
    rnode = _FakeRNode(
        {"px": np.array([], dtype=np.float64), "n": np.array([], dtype=np.int32)},
        {"px": "Double_t", "n": "Int_t"},
    )
    out = tmp_path / "empty.cstore"
    reader = from_root(rnode, out, show_progress=False)
    assert reader.n_rows == 0
    assert sorted(reader.columns) == ["n", "px"]


def test_ingest_explicit_columns_subset(tmp_path):
    rnode = _make_rnode(200)
    out = tmp_path / "subset.cstore"
    reader = from_root(rnode, out, columns=["px"], show_progress=False)
    assert reader.columns == ["px"]


# --------------------------------------------------------------------------- #
# Export: to_root (real colstore + fake ROOT)
# --------------------------------------------------------------------------- #


def _store_sample(tmp_path, n_rows: int = 1000):
    data = {
        "px": np.linspace(-1, 1, n_rows, dtype=np.float64),
        "n": np.arange(n_rows, dtype=np.int64),
    }
    path = tmp_path / "src.cstore"
    colstore.store(data, path, show_progress=False)
    return path, data


def test_export_chunks_recreate_then_append(tmp_path, monkeypatch):
    path, data = _store_sample(tmp_path, 1000)
    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)

    result = to_root(
        colstore.open(path),
        tmp_path / "out.root",
        treename="t",
        batch_size=300,
        show_progress=False,
    )

    # 1000 rows / 300 per chunk -> 4 chunks: first RECREATE, rest UPDATE+append.
    modes = [snap["mode"] for snap in fake_root.snapshots]
    appends = [snap["append"] for snap in fake_root.snapshots]
    assert modes == ["RECREATE", "UPDATE", "UPDATE", "UPDATE"]
    assert appends == [False, True, True, True]
    assert all(snap["treename"] == "t" for snap in fake_root.snapshots)

    # The concatenated chunks reproduce the source columns in order.
    stitched_n = np.concatenate([snap["data"]["n"] for snap in fake_root.snapshots])
    stitched_px = np.concatenate([snap["data"]["px"] for snap in fake_root.snapshots])
    assert np.array_equal(stitched_n, data["n"])
    assert np.allclose(stitched_px, data["px"])
    assert result == ("FakeRDF", "t", str(tmp_path / "out.root"))


def test_export_single_pass_when_batch_none(tmp_path, monkeypatch):
    path, data = _store_sample(tmp_path, 50)
    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)

    to_root(path, tmp_path / "out.root", batch_size=None, show_progress=False)
    assert len(fake_root.snapshots) == 1
    assert fake_root.snapshots[0]["mode"] == "RECREATE"
    assert np.array_equal(fake_root.snapshots[0]["data"]["n"], data["n"])


def test_export_zero_rows_writes_one_recreate(tmp_path, monkeypatch):
    path = tmp_path / "empty.cstore"
    with colstore.create(path) as writer:
        writer.write({"px": np.array([], dtype=np.float64)})
    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)

    to_root(path, tmp_path / "out.root", show_progress=False)
    assert len(fake_root.snapshots) == 1
    assert fake_root.snapshots[0]["mode"] == "RECREATE"


# --------------------------------------------------------------------------- #
# Branch-name sanitization (export)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("good_name", "good_name"),
        ("mg_xsec [fb]", "mg_xsec_fb"),
        ("a/b:c", "a_b_c"),
        ("123abc", "_123abc"),
        ("", "branch"),
        ("   ", "branch"),
        ("%$#", "branch"),
    ],
)
def test_sanitize_branch_name(name, expected):
    assert root_parser._sanitize_branch_name(name) == expected


def test_sanitized_name_map_disambiguates_collisions():
    mapping = root_parser._sanitized_name_map(["a b", "a-b", "ok"])
    assert mapping == {"a b": "a_b", "a-b": "a_b_2", "ok": "ok"}


def test_export_sanitizes_illegal_names_and_warns(tmp_path, monkeypatch):
    n = 8
    data = {
        "mg_xsec [fb]": np.linspace(0, 1, n, dtype=np.float32),
        "ok": np.arange(n, dtype=np.int64),
    }
    path = tmp_path / "src.cstore"
    colstore.store(data, path, show_progress=False)

    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)
    with pytest.warns(RuntimeWarning, match="mg_xsec"):
        to_root(path, tmp_path / "out.root", treename="t", show_progress=False)

    written = fake_root.snapshots[0]
    assert written["columns"] == ["mg_xsec_fb", "ok"]
    assert np.allclose(written["data"]["mg_xsec_fb"], data["mg_xsec [fb]"])
    assert np.array_equal(written["data"]["ok"], data["ok"])


def test_export_valid_names_do_not_warn(tmp_path, monkeypatch):
    data = {"px": np.arange(4, dtype=np.float64), "n": np.arange(4, dtype=np.int32)}
    path = tmp_path / "clean.cstore"
    colstore.store(data, path, show_progress=False)

    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        to_root(path, tmp_path / "out.root", show_progress=False)
    assert fake_root.snapshots[0]["columns"] == ["px", "n"]


# --------------------------------------------------------------------------- #
# Column selection (export)
# --------------------------------------------------------------------------- #


def test_export_columns_subset_in_requested_order(tmp_path, monkeypatch):
    data = {
        "a": np.arange(6, dtype=np.int32),
        "b": np.arange(6, dtype=np.float64),
        "c": np.arange(6, dtype=np.int16),
    }
    path = tmp_path / "src.cstore"
    colstore.store(data, path, show_progress=False)

    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)
    to_root(path, tmp_path / "out.root", columns=["b", "a"], show_progress=False)

    written = fake_root.snapshots[0]
    assert written["columns"] == ["b", "a"]
    assert set(written["data"]) == {"a", "b"}


def test_export_columns_unknown_is_error(tmp_path, monkeypatch):
    path = tmp_path / "src.cstore"
    colstore.store({"a": np.arange(3, dtype=np.int32)}, path, show_progress=False)
    monkeypatch.setattr(root_parser, "_import_root", lambda: _FakeROOT())
    with pytest.raises(ValueError, match="not found in the colstore file"):
        to_root(path, tmp_path / "out.root", columns=["a", "nope"], show_progress=False)


def test_export_columns_empty_is_error(tmp_path, monkeypatch):
    path = tmp_path / "src.cstore"
    colstore.store({"a": np.arange(3, dtype=np.int32)}, path, show_progress=False)
    monkeypatch.setattr(root_parser, "_import_root", lambda: _FakeROOT())
    with pytest.raises(ValueError, match="at least one column"):
        to_root(path, tmp_path / "out.root", columns=[], show_progress=False)


def test_export_columns_subset_with_sanitization(tmp_path, monkeypatch):
    data = {
        "mg_xsec [fb]": np.linspace(0, 1, 5, dtype=np.float32),
        "ok": np.arange(5, dtype=np.int64),
    }
    path = tmp_path / "src.cstore"
    colstore.store(data, path, show_progress=False)

    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)
    with pytest.warns(RuntimeWarning, match="mg_xsec"):
        to_root(path, tmp_path / "out.root", columns=["mg_xsec [fb]"], show_progress=False)
    assert fake_root.snapshots[0]["columns"] == ["mg_xsec_fb"]


# --------------------------------------------------------------------------- #
# Tree-name resolution (decision (a): auto-detect the sole tree)
# --------------------------------------------------------------------------- #


class _FakeKey:
    def __init__(self, name: str, class_name: str) -> None:
        self._name = name
        self._class_name = class_name

    def GetName(self) -> str:
        return self._name

    def GetClassName(self) -> str:
        return self._class_name


class _FakeTFile:
    def __init__(self, keys: list[_FakeKey]) -> None:
        self._keys = keys

    def IsZombie(self) -> bool:
        return False

    def GetListOfKeys(self) -> list[_FakeKey]:
        return self._keys

    def Close(self) -> None:
        pass


class _FakeClass:
    def __init__(self, is_tree: bool) -> None:
        self._is_tree = is_tree

    def InheritsFrom(self, name: str) -> bool:
        return self._is_tree and name == "TTree"


def _fake_root_with_keys(keys: list[_FakeKey]):
    fake = _FakeROOT()
    fake.TFile = type("TFile", (), {"Open": staticmethod(lambda _path: _FakeTFile(keys))})
    tree_classes = {"TTree", "TNtuple"}
    fake.TClass = type(
        "TClass",
        (),
        {"GetClass": staticmethod(lambda cls_name: _FakeClass(cls_name in tree_classes))},
    )
    return fake


def test_resolve_tree_explicit_name_skips_probe():
    assert root_parser._resolve_tree_name(_FakeROOT(), "x.root", "myTree") == "myTree"


def test_resolve_tree_single_tree_autodetected():
    fake = _fake_root_with_keys([_FakeKey("Events", "TTree"), _FakeKey("h1", "TH1F")])
    assert root_parser._resolve_tree_name(fake, "x.root", None) == "Events"


def test_resolve_tree_multiple_trees_errors():
    fake = _fake_root_with_keys([_FakeKey("Events", "TTree"), _FakeKey("Runs", "TTree")])
    with pytest.raises(ValueError, match="multiple trees"):
        root_parser._resolve_tree_name(fake, "x.root", None)


def test_resolve_tree_no_tree_errors():
    fake = _fake_root_with_keys([_FakeKey("h1", "TH1F")])
    with pytest.raises(ValueError, match="No TTree"):
        root_parser._resolve_tree_name(fake, "x.root", None)


# --------------------------------------------------------------------------- #
# "file.root:tree" path syntax
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("events.root", ("events.root", None)),
        ("events.root:Events", ("events.root", "Events")),
        ("dir/sub/file.root:tdir/tree", ("dir/sub/file.root", "tdir/tree")),
        ("root://srv//eos/file.root:Events", ("root://srv//eos/file.root", "Events")),
        ("root://srv//eos/file.root", ("root://srv//eos/file.root", None)),
        ("https://user:pw@host/f.root:Events", ("https://user:pw@host/f.root", "Events")),
        ("C:/data/file.root:Events", ("C:/data/file.root", "Events")),
        ("C:\\data\\file.root", ("C:\\data\\file.root", None)),
        ("file.root:", ("file.root", None)),
    ],
)
def test_split_path_and_tree(spec, expected):
    assert root_parser._split_path_and_tree(spec) == expected


def _root_with_rdataframe_recorder():
    fake = _FakeROOT()
    fake.built = []
    original = fake.RDataFrame

    def record(tree, path):
        fake.built.append((tree, path))
        return original(tree, path)

    fake.RDataFrame = record  # type: ignore[method-assign]
    return fake


def test_as_rdataframe_embedded_tree_skips_probe(monkeypatch):
    fake = _root_with_rdataframe_recorder()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake)
    root_parser._as_rdataframe("events.root:Events", None)
    assert fake.built == [("Events", "events.root")]


def test_as_rdataframe_pathlike_never_split(monkeypatch):
    import pathlib

    fake = _root_with_rdataframe_recorder()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake)
    # The colon here is part of the (odd) filename; with a Path it must not split.
    root_parser._as_rdataframe(pathlib.Path("weird:name.root"), "t")
    assert fake.built == [("t", "weird:name.root")]


def test_as_rdataframe_conflicting_tree_names_error(monkeypatch):
    fake = _root_with_rdataframe_recorder()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake)
    with pytest.raises(ValueError, match="Conflicting tree names"):
        root_parser._as_rdataframe("events.root:Events", "Other")


# --------------------------------------------------------------------------- #
# Parser class surface
# --------------------------------------------------------------------------- #


def test_parser_class_delegates(tmp_path, monkeypatch):
    parser = RootParser()
    assert parser.format_name == "root"

    reader = parser.read(_make_rnode(100), tmp_path / "p.cstore", show_progress=False)
    assert reader.n_rows == 100

    fake_root = _FakeROOT()
    monkeypatch.setattr(root_parser, "_import_root", lambda: fake_root)
    result = parser.write(reader, tmp_path / "p.root", treename="t", show_progress=False)
    assert result == ("FakeRDF", "t", str(tmp_path / "p.root"))
