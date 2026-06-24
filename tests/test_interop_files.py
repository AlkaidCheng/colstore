"""Tests for the file-format layer: colstore.ingest / saveas, extension dispatch,
and the NPZ format. Framework dispatch is also exercised with a dummy file format,
independent of any optional backend.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import interop
from colstore.interop import FileFormat, Selection
from colstore.interop import base as interop_base

# Built-in file formats register on first dispatch; trigger that at import time so
# every test's registry snapshot (taken by _restore_registry) preserves them -- a
# re-import of the cached module would not re-run __init_subclass__.
interop.file_format_for_extension(".npz")


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot the global registry before each test and restore it after."""
    snapshot = dict(interop_base._REGISTRY)
    yield
    interop_base._REGISTRY.clear()
    interop_base._REGISTRY.update(snapshot)


def _store(tmp_path, columns, name="s"):
    return colstore.store(columns, tmp_path / f"{name}.cstore", show_progress=False)


@pytest.fixture
def sample(tmp_path):
    """A small store mixing numeric, float, fixed-width string, and datetime columns."""
    cols = {
        "i": np.arange(6, dtype=np.int64),
        "f": (np.arange(6) * 1.5).astype(np.float32),
        "s": np.array(["a", "bb", "ccc", "d", "ee", "f"]),  # <U3
        "t": np.arange("2020-01-01", "2020-01-07", dtype="datetime64[D]"),
    }
    return _store(tmp_path, cols), cols


# ---- NPZ round-trip ---------------------------------------------------------


def test_npz_roundtrip_all_dtypes(tmp_path, sample):
    ds, cols = sample
    ds.saveas(tmp_path / "out.npz")
    back = colstore.ingest(tmp_path / "out.npz", tmp_path / "back.cstore")
    assert back.columns == list(cols)
    for name, expected in cols.items():
        np.testing.assert_array_equal(back.array(name), expected)
        assert back.dtypes[name] == expected.dtype


def test_ingest_returns_reader(tmp_path, sample):
    ds, _ = sample
    ds.saveas(tmp_path / "o.npz")
    back = colstore.ingest(tmp_path / "o.npz", tmp_path / "b.cstore")
    assert isinstance(back, colstore.ColStoreReader)


def test_saveas_selection_rows_and_columns(tmp_path, sample):
    ds, cols = sample
    ds[1:4, ["i", "f"]].saveas(tmp_path / "sel.npz")
    back = colstore.ingest(tmp_path / "sel.npz", tmp_path / "sel.cstore")
    assert back.columns == ["i", "f"]
    np.testing.assert_array_equal(back.array("i"), cols["i"][1:4])
    np.testing.assert_array_equal(back.array("f"), cols["f"][1:4])


def test_saveas_single_column(tmp_path, sample):
    ds, cols = sample
    ds["i"].saveas(tmp_path / "one.npz")
    back = colstore.ingest(tmp_path / "one.npz", tmp_path / "one.cstore")
    assert back.columns == ["i"]
    np.testing.assert_array_equal(back.array("i"), cols["i"])


def test_module_saveas_matches_method(tmp_path, sample):
    ds, _ = sample
    colstore.saveas(ds, tmp_path / "m.npz")
    ds.saveas(tmp_path / "d.npz")
    a = colstore.ingest(tmp_path / "m.npz", tmp_path / "m.cstore")
    b = colstore.ingest(tmp_path / "d.npz", tmp_path / "d.cstore")
    np.testing.assert_array_equal(a.recarray(), b.recarray())


def test_compress_option(tmp_path, sample):
    ds, cols = sample
    ds.saveas(tmp_path / "z.npz", compress=True)
    back = colstore.ingest(tmp_path / "z.npz", tmp_path / "z.cstore")
    np.testing.assert_array_equal(back.array("i"), cols["i"])


# ---- format override and extension dispatch ---------------------------------


def test_format_override_on_saveas_and_ingest(tmp_path, sample):
    ds, cols = sample
    # a path whose extension does not name the format; force it with format=
    ds.saveas(tmp_path / "noext", format="npz")
    assert (tmp_path / "noext").exists()  # written exactly at dest
    assert not (tmp_path / "noext.npz").exists()  # no ".npz" appended
    back = colstore.ingest(tmp_path / "noext", tmp_path / "o.cstore", format="npz")
    np.testing.assert_array_equal(back.array("i"), cols["i"])


def test_saveas_writes_exact_dest_even_for_uppercase_ext(tmp_path, sample):
    # dispatch is case-insensitive; the file must still land exactly at dest, not
    # at "UP.NPZ.npz" (numpy's savez would append a lowercase .npz to a path).
    ds, cols = sample
    ds.saveas(tmp_path / "UP.NPZ")
    assert (tmp_path / "UP.NPZ").exists()
    assert not (tmp_path / "UP.NPZ.npz").exists()
    back = colstore.ingest(tmp_path / "UP.NPZ", tmp_path / "u.cstore")
    np.testing.assert_array_equal(back.array("i"), cols["i"])


def test_saveas_overwrites_existing_file(tmp_path, sample):
    ds, cols = sample
    np.savez(tmp_path / "w.npz", junk=np.arange(9))
    ds.saveas(tmp_path / "w.npz")
    back = colstore.ingest(tmp_path / "w.npz", tmp_path / "w.cstore")
    assert back.columns == list(cols)


# ---- registry integrity: extension collisions and empty selections ----------


def test_extension_collision_rejected():
    with pytest.raises(ValueError, match="already claimed"):

        class _DupNpz(FileFormat):
            name = "dupnpz"
            extensions = frozenset({".NPZ"})  # case-folded collision with npz


def test_extension_collision_allows_override():
    class _DupNpz(FileFormat, override=True):
        name = "dupnpz2"
        extensions = frozenset({".npz"})

    assert "dupnpz2" in interop.file_formats()


def test_empty_selection_rejected(tmp_path, sample):
    ds, _ = sample
    with pytest.raises(ValueError, match="no columns"):
        ds.drop(*ds.columns).saveas(tmp_path / "empty.npz")


def test_bytes_path_dispatches():
    assert interop.file_format_for_path(b"data.npz").name == "npz"


def test_ingest_show_progress_does_not_collide(tmp_path, sample):
    ds, _ = sample
    ds.saveas(tmp_path / "p.npz")
    back = colstore.ingest(tmp_path / "p.npz", tmp_path / "p.cstore", show_progress=False)
    assert back.columns == ds.columns


def test_ingest_existing_dest_and_mode_recreate(tmp_path, sample):
    ds, _ = sample
    ds.saveas(tmp_path / "o.npz")
    colstore.ingest(tmp_path / "o.npz", tmp_path / "d.cstore").close()
    with pytest.raises(FileExistsError):
        colstore.ingest(tmp_path / "o.npz", tmp_path / "d.cstore").close()
    back = colstore.ingest(tmp_path / "o.npz", tmp_path / "d.cstore", mode="recreate")
    assert back.columns == ds.columns


def test_file_formats_lists_npz():
    assert "npz" in interop.file_formats()


def test_file_format_for_extension_is_case_insensitive():
    assert interop.file_format_for_extension(".npz").name == "npz"
    assert interop.file_format_for_extension(".NPZ").name == "npz"


def test_unknown_extension_raises(tmp_path, sample):
    ds, _ = sample
    with pytest.raises(KeyError, match="no file format handles extension"):
        colstore.ingest(tmp_path / "x.unknownext", tmp_path / "o.cstore")
    with pytest.raises(KeyError, match="no file format handles extension"):
        ds.saveas(tmp_path / "x.unknownext")


def test_data_format_name_rejected_for_files(tmp_path, sample):
    ds, _ = sample

    class _DictFmt(interop.DataFormat):
        name = "dicttest"

        def to_object(self, selection: Selection) -> dict[str, np.ndarray]:
            return {n: selection.gather(n) for n in selection.columns}

    with pytest.raises(TypeError, match="not a file format"):
        ds.saveas(tmp_path / "x.npz", format="dicttest")
    with pytest.raises(TypeError, match="not a file format"):
        colstore.ingest(tmp_path / "x.npz", tmp_path / "o.cstore", format="dicttest")


# ---- the dtype contract on import -------------------------------------------


def test_ingest_2d_array_rejected(tmp_path):
    np.savez(tmp_path / "bad.npz", a=np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError, match="1D"):
        colstore.ingest(tmp_path / "bad.npz", tmp_path / "o.cstore")


def test_ingest_ragged_columns_rejected(tmp_path):
    np.savez(tmp_path / "ragged.npz", a=np.arange(5), b=np.arange(3))
    with pytest.raises(ValueError):
        colstore.ingest(tmp_path / "ragged.npz", tmp_path / "o.cstore")


# ---- framework dispatch with a dummy file format ----------------------------


def test_dispatch_routes_to_format_by_extension(tmp_path, sample):
    ds, _ = sample
    record: list[tuple] = []

    class _RecFmt(FileFormat):
        name = "rectest"
        extensions = frozenset({".rec"})

        def to_file(self, selection: Selection, dest, **kwargs):
            record.append(("to_file", list(selection.columns), kwargs))

        def from_file(self, source, dest, **kwargs):
            record.append(("from_file", str(source), kwargs))
            return colstore.store({"x": np.arange(3, dtype=np.int64)}, dest, show_progress=False)

    ds[:, ["i", "f"]].saveas(tmp_path / "out.rec", opt=1)
    assert record[-1] == ("to_file", ["i", "f"], {"opt": 1})

    back = colstore.ingest(tmp_path / "in.rec", tmp_path / "o.cstore", key="v")
    assert record[-1] == ("from_file", str(tmp_path / "in.rec"), {"key": "v"})
    assert back.columns == ["x"]


def test_import_colstore_does_not_load_npz_module(tmp_path):
    # The format module loads only on a conversion, not at import.
    import subprocess
    import sys

    code = "import colstore, sys; assert 'colstore.interop.npz' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
