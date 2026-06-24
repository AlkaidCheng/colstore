"""Tests for the colstore.interop format framework: the Format taxonomy,
auto-registration, name/alias collisions, the registry, the Selection handoff, and
the InteropMixin export dispatch. The Arrow data format is tested in test_arrow.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import interop, testing
from colstore.interop import DataFormat, FileFormat, Format, Selection
from colstore.interop import base as interop_base


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot the global registry before each test and restore it after."""
    snapshot = dict(interop_base._REGISTRY)
    yield
    interop_base._REGISTRY.clear()
    interop_base._REGISTRY.update(snapshot)


@pytest.fixture
def dummies():
    """A dummy data format and a dummy file format."""

    class DictFormat(DataFormat):
        name = "dicttest"
        aliases = frozenset({"dictalias"})

        def to_object(self, selection: Selection) -> dict[str, np.ndarray]:
            return {name: selection.gather(name) for name in selection.columns}

    class NullFileFormat(FileFormat):
        name = "filetest"
        extensions = frozenset({".null"})

    return DictFormat, NullFileFormat


def _store(tmp_path, columns):
    return colstore.store(columns, tmp_path / "s.cstore", show_progress=False)


# ---- taxonomy and capabilities ----------------------------------------------


def test_taxonomy_and_kinds(dummies):
    data, file = dummies
    assert issubclass(DataFormat, Format)
    assert issubclass(FileFormat, Format)
    assert data().kind == "data"
    assert file().kind == "file"


def test_capability_flags(dummies):
    data, file = dummies
    assert data().can_export is True  # to_object overridden
    assert data().can_import is False  # from_object not overridden
    assert file().can_export is False
    assert file().can_import is False


# ---- auto-registration, collisions, aliases, override -----------------------


def test_defining_a_format_auto_registers(dummies):
    # No manual register() call: defining the dummy classes registered them.
    assert isinstance(interop.get("dicttest"), dummies[0])
    assert isinstance(interop.get("filetest"), dummies[1])
    assert "dicttest" in interop.data_formats()
    assert "filetest" in interop.file_formats()


def test_alias_resolves_but_is_not_listed(dummies):
    assert interop.get("dictalias") is interop.get("dicttest")  # alias resolves to the format
    assert "dictalias" not in interop.data_formats()  # aliases are not listed


def test_name_collision_raises(dummies):
    with pytest.raises(ValueError, match="already registered"):

        class _Dupe(DataFormat):
            name = "dicttest"  # collides with the dummy data format

            def to_object(self, selection):
                return None


def test_alias_collision_raises(dummies):
    with pytest.raises(ValueError, match="already registered"):

        class _AliasDupe(DataFormat):
            name = "unique_name_here"
            aliases = frozenset({"dicttest"})  # alias collides with an existing name

            def to_object(self, selection):
                return None


def test_override_replaces_a_format():
    class _First(DataFormat):
        name = "ovr"

        def to_object(self, selection):
            return "first"

    class _Second(DataFormat, override=True):  # same name, explicit override
        name = "ovr"

        def to_object(self, selection):
            return "second"

    assert isinstance(interop.get("ovr"), _Second)


def test_format_without_kind_raises():
    with pytest.raises(TypeError, match="DataFormat or FileFormat"):

        class _Bad(Format):  # subclasses Format directly -> no kind
            name = "bad"


# ---- registry ---------------------------------------------------------------


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown format"):
        interop.get("does_not_exist")


def test_kind_accessors_return_frozensets(dummies):
    assert isinstance(interop.data_formats(), frozenset)
    assert isinstance(interop.file_formats(), frozenset)
    assert "dicttest" in interop.data_formats()
    assert "dicttest" not in interop.file_formats()


def test_from_object_rejects_file_format(dummies):
    with pytest.raises(TypeError, match="file"):
        interop.from_object("filetest", object(), "out.cstore")


def test_from_object_unsupported_raises(dummies):
    with pytest.raises(NotImplementedError):
        interop.from_object("dicttest", {}, "out.cstore")


# ---- mixin dispatch ---------------------------------------------------------


def test_to_dispatches_to_data_format(tmp_path, dummies):
    cols = testing.make_columns(50, 2, names=("a", "b"), seed=1)
    with _store(tmp_path, cols) as ds:
        out = ds.to("dicttest")
        assert set(out) == {"a", "b"}
        assert np.array_equal(out["a"], cols["a"])


def test_to_via_alias(tmp_path, dummies):
    cols = testing.make_columns(20, 1, names=("a",), seed=6)
    with _store(tmp_path, cols) as ds:
        assert set(ds.to("dictalias")) == {"a"}


def test_to_file_format_raises(tmp_path, dummies):
    cols = testing.make_columns(10, 1, names=("a",), seed=2)
    with _store(tmp_path, cols) as ds, pytest.raises(TypeError, match="write a file"):
        ds.to("filetest")


def test_to_unknown_format_raises(tmp_path):
    cols = testing.make_columns(10, 1, names=("a",), seed=3)
    with _store(tmp_path, cols) as ds, pytest.raises(KeyError):
        ds.to("nope")


# ---- Selection handoff ------------------------------------------------------


def test_selection_seam(tmp_path):
    cols = testing.make_columns(40, 2, names=("a", "b"), seed=4)
    with _store(tmp_path, cols) as ds:
        sel = ds._interop_selection()
        assert isinstance(sel, Selection)
        assert sel.store is ds
        assert sel.columns == ["a", "b"]
        assert sel.row_indexer is None
        assert sel.single is False
        assert sel.is_whole_column() is True
        assert np.array_equal(sel.gather("a"), cols["a"])
        assert sel.native_dtype("a") == np.dtype("float64")
        col_sel = ds["a"]._interop_selection()
        assert col_sel.single is True
        assert col_sel.columns == ["a"]


def test_selection_column_chunks_are_zero_copy_views(tmp_path):
    cols = testing.make_columns(1000, 1, names=("a",), seed=5)
    with _store(tmp_path, cols) as ds:
        sel = ds._interop_selection()
        chunks = sel.column_chunks("a")
        assert np.array_equal(np.concatenate(chunks), cols["a"])
        assert chunks[0].ctypes.data == ds["a"].array(copy=False).ctypes.data
