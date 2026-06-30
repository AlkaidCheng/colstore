"""Tests for the Parquet, Feather, JSON, and HDF5 file formats, the per-format
sugar (``ds.to_parquet`` / ``colstore.from_parquet`` ...), the string-coercion /
nested-rejection policy, and HDF5's cross-writer reading.
"""

from __future__ import annotations

import filecmp
import importlib.util

import numpy as np
import pytest

import colstore
from colstore import interop

# "tables" is PyTables, which pandas needs for its HDF5 backend (DataFrame.to_hdf
# / read_hdf); it is a separate, heavyweight optional dependency from pandas.
_HAS = {m: importlib.util.find_spec(m) is not None for m in ("pyarrow", "pandas", "h5py", "tables")}

# (format, extension, required backends, preserves_dtype)
# JSON is text and carries no dtype, so it round-trips values but not the exact
# width (float32 -> float64); the binary formats preserve dtypes.
_FORMATS = [
    ("parquet", "parquet", ("pyarrow",), True),
    ("feather", "feather", ("pyarrow",), True),
    ("json", "json", ("pandas",), False),
    ("hdf5", "h5", ("h5py",), True),
]


@pytest.fixture
def columns():
    return {
        "i": np.arange(8, dtype=np.int64),
        "f": (np.arange(8) * 1.5).astype(np.float32),
        "u": np.arange(8, dtype=np.uint16),
        "ok": (np.arange(8) % 2 == 0),
        "s": np.array(["aa", "bb", "ccc", "d", "e", "ff", "g", "hh"]),  # <U3
    }


@pytest.fixture
def store(tmp_path, columns):
    return colstore.store(columns, tmp_path / "s.cstore", show_progress=False)


def _need(*backends):
    missing = [b for b in backends if not _HAS[b]]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")


# ---- round-trip parity, all four formats -----------------------------------


@pytest.mark.parametrize("fmt, ext, backends, preserves_dtype", _FORMATS)
def test_roundtrip(tmp_path, store, columns, fmt, ext, backends, preserves_dtype):
    _need(*backends)
    path = tmp_path / f"x.{ext}"
    store.saveas(path)
    back = colstore.convert(path, tmp_path / f"b.{ext}.cstore")
    assert set(back.columns) == set(columns)
    for name, values in columns.items():
        if name == "s":
            assert list(back.array("s")) == list(values)
            assert back.dtypes["s"].kind == "U"  # widened to fixed-width unicode
        else:
            np.testing.assert_array_equal(back.array(name), values)
            if preserves_dtype:
                assert back.dtypes[name] == values.dtype
    back.close()


@pytest.mark.parametrize("fmt, ext, backends, preserves_dtype", _FORMATS)
def test_column_projection(tmp_path, store, columns, fmt, ext, backends, preserves_dtype):
    _need(*backends)
    path = tmp_path / f"p.{ext}"
    store.saveas(path)
    back = colstore.convert(path, tmp_path / f"p.{ext}.cstore", columns=["i", "f"])
    assert set(back.columns) == {"i", "f"}
    np.testing.assert_array_equal(back.array("i"), columns["i"])
    back.close()


def test_all_formats_registered():
    assert {"cstore", "parquet", "feather", "json", "hdf5"} <= interop.file_formats()


# ---- native .cstore export -------------------------------------------------


def test_cstore_native_export_roundtrip(tmp_path, store, columns):
    """saveas('out.cstore') writes a new cstore; it needs no backend and keeps strings."""
    out = tmp_path / "copy.cstore"
    store.saveas(out)  # the native format, dispatched by the .cstore extension
    back = colstore.open(out)
    assert set(back.columns) == set(columns)
    for name, values in columns.items():
        if name == "s":
            assert list(back.array("s")) == list(values)  # native handles strings
        else:
            np.testing.assert_array_equal(back.array(name), values)
            assert back.dtypes[name] == values.dtype
    back.close()


def test_cstore_export_respects_selection(tmp_path, store, columns):
    """A row+column selection writes only that subset to the new cstore."""
    out = tmp_path / "sel.cstore"
    store[2:5, ["i", "f"]].saveas(out)
    back = colstore.open(out)
    assert back.columns == ["i", "f"]
    assert back.n_rows == 3
    np.testing.assert_array_equal(back.array("i"), columns["i"][2:5])
    back.close()


def test_convert_cstore_to_cstore_copies(tmp_path, store, columns):
    """convert between two .cstore paths copies the store (both endpoints are native)."""
    p = tmp_path / "x.cstore"
    store.saveas(p)
    out = colstore.convert(p, tmp_path / "y.cstore")
    assert out.n_rows == store.n_rows
    np.testing.assert_array_equal(out.array("i"), columns["i"])


def test_convert_foreign_to_foreign_rejected(tmp_path, store):
    """convert needs one endpoint to be a .cstore; foreign -> foreign is rejected."""
    pytest.importorskip("pyarrow")
    store.saveas(tmp_path / "a.parquet")
    with pytest.raises(ValueError, match="one endpoint"):
        colstore.convert(tmp_path / "a.parquet", tmp_path / "b.feather")


def test_cstore_export_rejects_unknown_kwarg(tmp_path, store):
    """An unknown saveas keyword (e.g. a memory_budget typo) is rejected, not dropped."""
    store.saveas(tmp_path / "ok.cstore", memory_budget=1 << 20)  # the one accepted knob
    with pytest.raises(TypeError, match="memory_budget"):
        store.saveas(tmp_path / "x.cstore", memory_budgett=1024)


def test_cstore_export_whole_store_is_raw_copy(tmp_path):
    """A whole-store saveas('.cstore') raw-copies the source -- nothing is materialized."""
    src = tmp_path / "src.cstore"
    colstore.store(
        {"a": np.arange(40, dtype="f8"), "b": np.arange(40, dtype="i8")}, src, show_progress=False
    )
    out = tmp_path / "copy.cstore"
    colstore.open(src).saveas(out)
    assert filecmp.cmp(src, out, shallow=False)  # byte-identical: the source bytes were copied


def test_cstore_export_of_dataset_matches_concat(tmp_path):
    """dataset.saveas('.cstore') merges the shards exactly like concat() (byte-for-byte)."""
    parts = []
    for i in range(2):
        p = tmp_path / f"p{i}.cstore"
        colstore.store(
            {"a": np.arange(i * 10, i * 10 + 10, dtype="f8"), "b": np.arange(10, dtype="i8")},
            p,
            show_progress=False,
        )
        parts.append(p)
    dset = colstore.open(parts)
    merged = tmp_path / "merged.cstore"
    dset.saveas(merged)
    concatenated = tmp_path / "concat.cstore"
    colstore.concat(parts, out=concatenated)
    assert filecmp.cmp(merged, concatenated, shallow=False)


# ---- per-format sugar ------------------------------------------------------


def test_sugar_methods_and_functions(tmp_path, store, columns):
    pairs = [
        ("to_npz", colstore.from_npz, "npz", ()),
        ("to_parquet", colstore.from_parquet, "parquet", ("pyarrow",)),
        ("to_feather", colstore.from_feather, "feather", ("pyarrow",)),
        ("to_json", colstore.from_json, "json", ("pandas",)),
        ("to_hdf", colstore.from_hdf, "h5", ("h5py",)),
    ]
    for to_method, from_func, ext, backends in pairs:
        if any(not _HAS[b] for b in backends):
            continue
        path = tmp_path / f"sugar.{ext}"
        getattr(store, to_method)(path)
        back = from_func(path, tmp_path / f"sugar.{ext}.cstore")
        np.testing.assert_array_equal(back.array("i"), columns["i"])
        back.close()


# ---- string coercion and nested rejection ----------------------------------


def test_nested_column_rejected(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"a": [1, 2, 3], "lst": [[1, 2], [3], [4, 5]]})
    path = tmp_path / "nested.parquet"
    pq.write_table(table, str(path))
    with pytest.raises(TypeError, match="nested"):
        colstore.convert(path, tmp_path / "nested.cstore")


# ---- HDF5: cross-writer reading + backends + key ---------------------------


def test_hdf5_pandas_written_read_auto(tmp_path, columns):
    _need("h5py", "pandas", "tables")
    import pandas as pd

    path = tmp_path / "pd.h5"
    pd.DataFrame(columns).to_hdf(path, key="frame", mode="w")
    back = colstore.convert(path, tmp_path / "pd.cstore")  # auto-detects pandas store
    assert set(back.columns) == set(columns)
    assert list(back.array("s")) == list(columns["s"])
    back.close()


def test_hdf5_h5py_root_datasets_read(tmp_path):
    _need("h5py")
    import h5py

    path = tmp_path / "root.h5"
    with h5py.File(path, "w") as f:  # datasets at the root, no group
        f.create_dataset("a", data=np.arange(5, dtype=np.int64))
        f.create_dataset("b", data=(np.arange(5) * 2.0))
    back = colstore.convert(path, tmp_path / "root.cstore")
    assert set(back.columns) == {"a", "b"}
    np.testing.assert_array_equal(back.array("a"), np.arange(5))
    back.close()


def test_hdf5_backend_and_key(tmp_path, store, columns):
    _need("h5py", "pandas", "tables")
    # write with the pandas backend under a custom key, read it back
    p1 = tmp_path / "pdkey.h5"
    store.saveas(p1, backend="pandas", key="mytable")
    back = colstore.convert(p1, tmp_path / "pdkey.cstore", key="mytable")
    np.testing.assert_array_equal(back.array("i"), columns["i"])
    back.close()
    # write with the h5py backend under a custom key
    p2 = tmp_path / "h5key.h5"
    store.saveas(p2, backend="h5py", key="grp")
    back2 = colstore.convert(p2, tmp_path / "h5key.cstore", key="grp")
    np.testing.assert_array_equal(back2.array("i"), columns["i"])
    back2.close()


def test_hdf5_unknown_backend_rejected(tmp_path, store):
    _need("h5py")
    with pytest.raises(ValueError, match="unknown hdf5 backend"):
        store.saveas(tmp_path / "x.h5", backend="bogus")


# ---- coercion policy: reject nulls / non-strings ---------------------------


def test_parquet_null_columns_rejected(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    for col in (pa.array(["a", None, "c"]), pa.array([1, None, 3], type=pa.int64())):
        path = tmp_path / "null.parquet"
        pq.write_table(pa.table({"x": col}), str(path))
        with pytest.raises(TypeError, match="null"):
            colstore.from_parquet(path, tmp_path / "null.cstore")


def test_json_null_rejected(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    path = tmp_path / "n.json"
    pd.DataFrame({"s": ["a", None, "c"]}).to_json(path, orient="records")
    with pytest.raises(TypeError, match="null"):
        colstore.from_json(path, tmp_path / "n.cstore")


def test_hdf5_float_nan_is_stored_not_null(tmp_path):
    """A NaN in a native float column is a valid value, not a null -- it round-trips."""
    _need("h5py", "pandas", "tables")
    import pandas as pd

    path = tmp_path / "nan.h5"
    pd.DataFrame({"m": [120.5, np.nan, 88.1], "n": [1, 2, 3]}).to_hdf(path, key="frame", mode="w")
    out = colstore.from_hdf(path, tmp_path / "nan.cstore").dict()
    np.testing.assert_array_equal(np.isnan(out["m"]), [False, True, False])
    assert out["m"][0] == 120.5 and out["m"][2] == 88.1
    assert list(out["n"]) == [1, 2, 3]


def test_json_float_nan_is_stored_not_null(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    path = tmp_path / "nan.json"
    pd.DataFrame({"m": [120.5, np.nan, 88.1]}).to_json(path, orient="records")
    out = colstore.from_json(path, tmp_path / "nan.cstore").dict()
    np.testing.assert_array_equal(np.isnan(out["m"]), [False, True, False])


def test_hdf5_datetime_nat_is_stored_not_null(tmp_path):
    """NaT is datetime64's in-band sentinel -- stored like float NaN, not rejected."""
    _need("h5py", "pandas", "tables")
    import pandas as pd

    path = tmp_path / "nat.h5"
    pd.DataFrame({"t": pd.to_datetime(["2021-01-01", None, "2021-01-03"])}).to_hdf(
        path, key="frame", mode="w"
    )
    out = colstore.from_hdf(path, tmp_path / "nat.cstore").dict()
    assert np.isnat(out["t"]).tolist() == [False, True, False]


def test_all_null_object_column_stored_as_float_nan():
    """A pandas all-null column carries no data, so it stores as float64 NaN, not rejected.

    A column that mixes nulls with real values has no fixed-width form and still raises.
    """
    pytest.importorskip("pandas")
    import pandas as pd

    from colstore.interop._convert import frame_to_columns

    cols = frame_to_columns(
        pd.DataFrame({"id": [1, 2, 3], "sel": pd.Series([np.nan] * 3, dtype=object)})
    )
    assert cols["sel"].dtype == np.float64 and np.isnan(cols["sel"]).all()
    assert list(cols["id"]) == [1, 2, 3]
    with pytest.raises(TypeError, match="null"):
        frame_to_columns(pd.DataFrame({"x": pd.Series(["a", None, "c"], dtype=object)}))


def test_parquet_all_null_column_stored_as_float_nan(tmp_path):
    """An all-null Parquet column (typed or null-typed) stores as float64 NaN, not rejected."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "typed": pa.array([None, None, None], type=pa.int64()),
            "untyped": pa.array([None, None, None]),
            "real": pa.array([1, 2, 3], type=pa.int64()),
        }
    )
    pq.write_table(table, str(tmp_path / "an.parquet"))
    out = colstore.from_parquet(tmp_path / "an.parquet", tmp_path / "an.cstore").dict()
    for name in ("typed", "untyped"):
        assert out[name].dtype == np.float64 and np.isnan(out[name]).all()
    assert list(out["real"]) == [1, 2, 3]


def test_hdf5_all_null_column_ingests_as_float_nan(tmp_path):
    """An all-null column in a pandas HDF5 file ingests without error as float64 NaN."""
    _need("h5py", "pandas", "tables")
    import pandas as pd

    path = tmp_path / "an.h5"
    pd.DataFrame({"sel": pd.Series([np.nan] * 3, dtype=object), "n": [1, 2, 3]}).to_hdf(
        path, key="frame", format="table", mode="w"
    )
    out = colstore.from_hdf(path, tmp_path / "an.cstore").dict()
    assert out["sel"].dtype == np.float64 and np.isnan(out["sel"]).all()
    assert list(out["n"]) == [1, 2, 3]


def test_apply_dtype_overrides_coerces_and_fills():
    """An override casts real values; an all-null column fills its target's zero."""
    from colstore.interop._convert import apply_dtype_overrides

    allnull = np.full(3, np.nan, dtype=np.float64)
    out = apply_dtype_overrides(
        {"flag": allnull.copy(), "kept": allnull.copy(), "real": np.array([0.0, 1.0, 0.0])},
        {"flag": "bool", "kept": "float32", "real": "bool"},
    )
    assert out["flag"].dtype == np.bool_ and out["flag"].tolist() == [False, False, False]
    assert out["kept"].dtype == np.float32 and np.isnan(out["kept"]).all()  # float target keeps NaN
    assert out["real"].dtype == np.bool_ and out["real"].tolist() == [False, True, False]
    # an all-null column to an integer target fills 0
    intout = apply_dtype_overrides({"c": allnull.copy()}, {"c": "int64"})
    assert intout["c"].dtype == np.int64 and intout["c"].tolist() == [0, 0, 0]
    # a name the file lacks is a hard error, not a silent skip
    with pytest.raises(KeyError, match="unknown column"):
        apply_dtype_overrides({"a": allnull.copy()}, {"missing": "bool"})


def test_apply_dtype_overrides_missing_values_are_safe():
    """A NaN (missing) in a float source becomes the target's empty value, never garbage.

    ``float NaN -> int`` is undefined in NumPy, so a NaN among real values must not fall
    through to a bare cast; it fills 0 / False / '' / NaT by target kind, consistently
    whether the column is all-null or partly null.
    """
    import warnings

    from colstore.interop._convert import apply_dtype_overrides

    part = np.array([1.5, np.nan, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an undefined NaN->int cast would warn; it must not
        to_int = apply_dtype_overrides({"c": part.copy()}, {"c": "int64"})["c"]
    assert to_int.tolist() == [1, 0, 3]
    # NaN -> False, not the truthy value a bare cast would give
    to_bool = apply_dtype_overrides({"c": np.array([0.0, np.nan, 2.0])}, {"c": "bool"})["c"]
    assert to_bool.tolist() == [False, False, True]
    # a missing string cell is empty whether the column is all-null or partly null
    to_str = apply_dtype_overrides({"c": part.copy()}, {"c": "U5"})["c"]
    assert to_str.tolist() == ["1.5", "", "3.0"]
    # datetime / timedelta missing values become NaT, not the epoch
    to_dt = apply_dtype_overrides({"c": np.full(3, np.nan)}, {"c": "datetime64[ns]"})["c"]
    assert np.isnat(to_dt).all()


def test_hdf5_dtype_override_unifies_schema(tmp_path):
    """A column that is all-null in one file and real bool in another reads as one schema.

    The reported case: ``dtypes={"sel": "bool"}`` coerces the all-null column to bool
    ``False`` and leaves the real bool column untouched, so a glob open sees one schema.
    """
    _need("h5py", "pandas", "tables")
    import pandas as pd

    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    pd.DataFrame({"id": [1, 2, 3], "sel": pd.Series([np.nan] * 3, dtype=object)}).to_hdf(
        a, key="frame", format="table", mode="w"
    )
    pd.DataFrame({"id": [4, 5, 6], "sel": [True, False, True]}).to_hdf(
        b, key="frame", format="table", mode="w"
    )
    ra = colstore.from_hdf(a, tmp_path / "a.cstore", dtypes={"sel": "bool"})
    rb = colstore.from_hdf(b, tmp_path / "b.cstore", dtypes={"sel": "bool"})
    assert ra.dtypes["sel"] == np.bool_ and ra.array("sel").tolist() == [False, False, False]
    assert rb.array("sel").tolist() == [True, False, True]
    ds = colstore.open(str(tmp_path / "*.cstore"))  # no schema mismatch across the two files
    assert ds.n_rows == 6 and ds.dtypes["sel"] == np.bool_


def test_parquet_dtype_override(tmp_path):
    """The dtype override reaches the Arrow column path too (Parquet / Feather)."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"sel": pa.array([None, None, None], type=pa.float64()), "n": [1, 2, 3]})
    pq.write_table(table, str(tmp_path / "ov.parquet"))
    out = colstore.from_parquet(
        tmp_path / "ov.parquet", tmp_path / "ov.cstore", dtypes={"sel": "bool"}
    )
    assert out.dtypes["sel"] == np.bool_ and out.array("sel").tolist() == [False, False, False]


def test_nested_or_non_string_object_rejected():
    from colstore.interop._convert import storable_column

    # a nested value past the 64-sample window must still be rejected
    with pytest.raises(TypeError, match="not a fixed-width string"):
        storable_column("c", np.array(["x"] * 64 + [{"k": 1}], dtype=object))
    # a numeric object column is not a string column
    with pytest.raises(TypeError, match="not a fixed-width string"):
        storable_column("c", np.array([1, 2, 3], dtype=object))


def test_json_numeric_string_preserved(tmp_path):
    pytest.importorskip("pandas")
    store = colstore.store(
        {"sid": np.array(["1", "2", "30"]), "n": np.arange(3, dtype=np.int64)},
        tmp_path / "s.cstore",
        show_progress=False,
    )
    store.to_json(tmp_path / "j.json")
    back = colstore.from_json(tmp_path / "j.json", tmp_path / "jb.cstore")
    assert back.dtypes["sid"].kind == "U"  # numeric-looking strings stay strings
    assert back.dtypes["n"].kind == "i"
    back.close()


# ---- HDF5 dtype fidelity ---------------------------------------------------


def test_hdf5_datetime_roundtrip(tmp_path):
    _need("h5py")
    times = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
    store = colstore.store({"t": times}, tmp_path / "dt.cstore", show_progress=False)
    store.to_hdf(tmp_path / "dt.h5")
    back = colstore.from_hdf(tmp_path / "dt.h5", tmp_path / "dtb.cstore")
    assert back.dtypes["t"] == np.dtype("datetime64[D]")  # unit preserved
    np.testing.assert_array_equal(back.array("t"), times)
    back.close()


def test_hdf5_bytes_preserved(tmp_path):
    _need("h5py")
    store = colstore.store(
        {"b": np.array([b"x", b"yy", b"z"])}, tmp_path / "b.cstore", show_progress=False
    )
    store.to_hdf(tmp_path / "b.h5")
    back = colstore.from_hdf(tmp_path / "b.h5", tmp_path / "bb.cstore")
    assert back.dtypes["b"].kind == "S"  # fixed bytes stay bytes (not widened to U)
    back.close()


def test_hdf5_unknown_read_backend_rejected(tmp_path, store):
    _need("h5py")
    store.saveas(tmp_path / "x.h5")
    with pytest.raises(ValueError, match="unknown hdf5 backend"):
        colstore.from_hdf(tmp_path / "x.h5", tmp_path / "x.cstore", backend="bogus")


# ---- lazy import -----------------------------------------------------------


def test_import_colstore_loads_no_backend():
    import subprocess
    import sys

    code = (
        "import colstore, sys; "
        "bad=[m for m in ('pyarrow','h5py','pandas') if m in sys.modules]; "
        "assert not bad, bad"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
