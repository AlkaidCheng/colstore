"""Tests for the Apache Arrow export (``colstore.interop.arrow``).

Covers round-trip parity, the zero-copy guarantee (Arrow values buffer aliases
the memmap, no Arrow-pool allocation, survives store close), the ChunkedArray
shape for multi-record and multi-file stores, the dtype mapping, and the Arrow
PyCapsule interface that lets pyarrow consume colstore objects directly.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import colstore
from colstore import interop, testing
from colstore.interop import DataFormat, Format
from colstore.interop import arrow as arrow_interop
from colstore.interop.arrow import ArrowFormat

pa = pytest.importorskip("pyarrow")


def _store(tmp_path, columns, *, records=1):
    """Open a store of ``columns`` (single- or multi-record)."""
    path = tmp_path / "a.cstore"
    if records == 1:
        return colstore.store(columns, path, show_progress=False)
    return testing.write_columns(path, columns, records=records)


# ---- round-trip parity ------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    ["float64", "float32", "float16", "int64", "int32", "int16", "int8", "uint8", "uint32"],
)
def test_numeric_column_roundtrip(tmp_path, dtype):
    col = testing.make_columns(500, 1, names=("x",), dtype=dtype, seed=1)["x"]
    with _store(tmp_path, {"x": col}) as ds:
        arr = ds["x"].arrow()
        assert isinstance(arr, pa.Array)
        assert arr.null_count == 0
        assert np.array_equal(arr.to_numpy(zero_copy_only=True), col)


def test_fixed_bytes_column_roundtrip(tmp_path):
    col = np.array([b"abc", b"defgh", b"z", b""], dtype="S8")
    with _store(tmp_path, {"s": col}) as ds:
        arr = ds["s"].arrow()
        assert arr.type == pa.binary(8)  # fixed_size_binary[8]
        assert np.frombuffer(arr.buffers()[1], dtype="S8").tolist() == col.tolist()


def test_datetime_column_roundtrip(tmp_path):
    col = (np.arange(64, dtype="int64") * 1_000_000_000).astype("datetime64[ns]")
    with _store(tmp_path, {"t": col}) as ds:
        arr = ds["t"].arrow()
        assert arr.type == pa.timestamp("ns")
        assert np.array_equal(arr.to_numpy(zero_copy_only=False).astype("datetime64[ns]"), col)


def test_bool_column_converts_and_roundtrips(tmp_path):
    col = np.array([True, False, True, True, False], dtype=bool)
    with _store(tmp_path, {"b": col}) as ds:
        arr = ds["b"].arrow()
        assert arr.type == pa.bool_()
        assert np.array_equal(arr.to_numpy(zero_copy_only=False), col)


@pytest.mark.parametrize(
    "kind, arrow_type", [("datetime64", pa.timestamp), ("timedelta64", pa.duration)]
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_supported_temporal_units_roundtrip(tmp_path, kind, arrow_type, unit):
    col = (np.arange(40, dtype="int64") * 1000).astype(f"{kind}[{unit}]")
    with _store(tmp_path, {"t": col}) as ds:
        arr = ds["t"].arrow()
        assert arr.type == arrow_type(unit)
        assert np.array_equal(arr.to_numpy(zero_copy_only=True), col)


@pytest.mark.parametrize(
    "dtype",
    [
        "datetime64[D]",
        "datetime64[h]",
        "datetime64[M]",
        "datetime64[Y]",
        "datetime64[ps]",
        "timedelta64[D]",
        "timedelta64[h]",
        "timedelta64[Y]",
    ],
)
def test_unsupported_temporal_unit_raises(tmp_path, dtype):
    # colstore stores any datetime64/timedelta64 unit, but Arrow has only s/ms/us/ns.
    # The export must reject an unrepresentable unit with a clear error rather than
    # crash opaquely or (for datetime64[D]) silently narrow to a 32-bit date.
    col = (np.arange(20, dtype="int64") * 3).astype(dtype)
    with (
        _store(tmp_path, {"t": col}) as ds,
        pytest.raises(TypeError, match="second-to-nanosecond"),
    ):
        ds["t"].arrow()


# ---- zero copy --------------------------------------------------------------


def test_zero_copy_buffer_aliases_memmap(tmp_path):
    col = testing.make_columns(100_000, 1, names=("x",), seed=2)["x"]
    with _store(tmp_path, {"x": col}) as ds:
        view = ds["x"].array(copy=False)
        arr = ds["x"].arrow()
        assert arr.buffers()[1].address == view.ctypes.data


def test_whole_column_slice_is_zero_copy(tmp_path):
    # ds[:, name] resolves to a full slice; it must zero-copy like ds[name],
    # matching array(copy=False). A sub-slice gathers (a copy).
    col = testing.make_columns(50_000, 1, names=("x",), seed=21)["x"]
    with _store(tmp_path, {"x": col}) as ds:
        base = ds["x"].array(copy=False).ctypes.data
        assert ds[:, "x"].arrow().buffers()[1].address == base
        assert ds[5:10, "x"].arrow().buffers()[1].address != base


def test_zero_copy_no_arrow_allocation(tmp_path):
    col = testing.make_columns(100_000, 1, names=("x",), seed=3)["x"]
    with _store(tmp_path, {"x": col}) as ds:
        before = pa.total_allocated_bytes()
        arr = ds["x"].arrow()
        assert pa.total_allocated_bytes() == before
        assert arr.buffers()[0] is None  # no validity bitmap


def test_export_survives_store_close(tmp_path):
    col = testing.make_columns(50_000, 1, names=("x",), seed=4)["x"]
    ds = _store(tmp_path, {"x": col})
    arr = ds["x"].arrow()
    ds.close()
    gc.collect()
    assert np.array_equal(arr.to_numpy(zero_copy_only=True), col)


# ---- chunked: multi-record and multi-file -----------------------------------


def test_multirecord_column_is_chunked(tmp_path):
    col = testing.make_columns(30_000, 1, names=("x",), seed=5)["x"]
    with _store(tmp_path, {"x": col}, records=4) as ds:
        arr = ds["x"].arrow()
        assert isinstance(arr, pa.ChunkedArray)
        assert arr.num_chunks == 4
        assert np.array_equal(arr.combine_chunks().to_numpy(zero_copy_only=True), col)


def test_multifile_column_is_chunked(tmp_path):
    parts = []
    expected = []
    for f in range(3):
        col = testing.make_columns(10_000, 1, names=("x",), seed=10 + f)["x"]
        path = tmp_path / f"part{f}.cstore"
        colstore.store({"x": col}, path, show_progress=False)
        parts.append(path)
        expected.append(col)
    with colstore.open(parts) as ds:
        arr = ds["x"].arrow()
        assert isinstance(arr, pa.ChunkedArray)
        assert arr.num_chunks == 3
        assert np.array_equal(
            arr.combine_chunks().to_numpy(zero_copy_only=True), np.concatenate(expected)
        )


def test_multifile_chunks_are_zero_copy(tmp_path):
    parts = []
    for f in range(2):
        colstore.store(
            {"x": testing.make_columns(8_000, 1, names=("x",), seed=20 + f)["x"]},
            tmp_path / f"p{f}.cstore",
            show_progress=False,
        )
        parts.append(tmp_path / f"p{f}.cstore")
    with colstore.open(parts) as ds:
        arr = ds["x"].arrow()
        for f, child in enumerate(ds._children):
            chunk_addr = arr.chunk(f).buffers()[1].address
            base = child._memmaps["x"].ctypes.data
            assert chunk_addr == base


# ---- table ------------------------------------------------------------------


def test_table_export_parity_and_order(tmp_path):
    cols = testing.make_columns(2_000, 3, names=("a", "b", "c"), dtype=("f8", "i4", "f4"), seed=6)
    with _store(tmp_path, cols) as ds:
        table = ds.arrow()
        assert isinstance(table, pa.Table)
        assert table.column_names == ["a", "b", "c"]
        for name, values in cols.items():
            assert np.array_equal(
                table.column(name).combine_chunks().to_numpy(zero_copy_only=True), values
            )


def test_tableview_subset_export(tmp_path):
    cols = testing.make_columns(2_000, 3, names=("a", "b", "c"), seed=7)
    with _store(tmp_path, cols) as ds:
        table = ds[["c", "a"]].arrow()
        assert table.column_names == ["c", "a"]
        assert np.array_equal(
            table.column("a").combine_chunks().to_numpy(zero_copy_only=True), cols["a"]
        )


# ---- row selection materializes (copy) --------------------------------------


def test_row_selection_materializes_and_matches(tmp_path):
    col = testing.make_columns(1_000, 1, names=("x",), seed=8)["x"]
    with _store(tmp_path, {"x": col}) as ds:
        idx = np.array([5, 100, 7, 999, 0])
        arr = ds[idx, "x"].arrow()
        assert np.array_equal(arr.to_numpy(zero_copy_only=True), col[idx])
        # A gather allocates its own buffer; it does not alias the memmap.
        assert arr.buffers()[1].address != ds["x"].array(copy=False).ctypes.data


# ---- PyCapsule interface (consumers) ----------------------------------------


def test_pyarrow_array_consumes_column(tmp_path):
    col = testing.make_columns(5_000, 1, names=("x",), seed=9)["x"]
    with _store(tmp_path, {"x": col}) as ds:
        imported = pa.array(ds["x"])
        assert imported.buffers()[1].address == ds["x"].array(copy=False).ctypes.data
        assert np.array_equal(imported.to_numpy(zero_copy_only=True), col)


def test_pyarrow_chunked_array_consumes_column(tmp_path):
    col = testing.make_columns(5_000, 1, names=("x",), seed=11)["x"]
    with _store(tmp_path, {"x": col}, records=3) as ds:
        imported = pa.chunked_array(ds["x"])
        assert np.array_equal(imported.combine_chunks().to_numpy(zero_copy_only=True), col)


def test_pyarrow_table_consumes_reader(tmp_path):
    cols = testing.make_columns(3_000, 2, names=("a", "b"), seed=12)
    with _store(tmp_path, cols) as ds:
        table = pa.table(ds)
        assert table.num_rows == 3_000
        assert np.array_equal(
            table.column("a").combine_chunks().to_numpy(zero_copy_only=True), cols["a"]
        )


def test_record_batch_reader_from_stream(tmp_path):
    cols = testing.make_columns(3_000, 2, names=("a", "b"), seed=13)
    with _store(tmp_path, cols) as ds:
        reader = pa.RecordBatchReader.from_stream(ds)
        assert reader.schema.names == ["a", "b"]
        assert reader.read_all().num_rows == 3_000


# ---- empty store, registry, contract ----------------------------------------


def test_empty_store_exports_empty(tmp_path):
    with _store(tmp_path, {"x": np.empty(0, dtype=np.float64)}) as ds:
        arr = ds["x"].arrow()
        assert len(arr) == 0
        assert ds.arrow().num_rows == 0


def test_to_dispatches_by_name(tmp_path):
    # ds.to(name) is the generic verb; ds.arrow() is its shorthand for "arrow".
    cols = testing.make_columns(100, 2, names=("a", "b"), seed=14)
    with _store(tmp_path, cols) as ds:
        assert isinstance(ds.to("arrow"), pa.Table)
        assert isinstance(ds["a"].to("arrow"), pa.Array)
        assert isinstance(ds[["a", "b"]].to("arrow"), pa.Table)


def test_to_unknown_format_raises(tmp_path):
    cols = testing.make_columns(100, 1, names=("a",), seed=15)
    with _store(tmp_path, cols) as ds, pytest.raises(KeyError, match="unknown format"):
        ds.to("nope")


def test_registry_lists_and_resolves_arrow():
    assert interop.data_formats() == frozenset({"arrow"})
    assert "arrow" not in interop.file_formats()
    fmt = interop.get("arrow")
    assert isinstance(fmt, ArrowFormat)
    assert fmt.name == "arrow"
    assert fmt.kind == "data"
    assert fmt.can_export is True
    assert fmt.can_import is False  # Arrow import is not yet implemented


def test_arrow_import_not_yet_supported(tmp_path):
    cols = testing.make_columns(10, 1, names=("a",), seed=16)
    with _store(tmp_path, cols) as ds:
        table = ds.arrow()
    with pytest.raises(NotImplementedError):
        interop.from_object("arrow", table, tmp_path / "out.cstore")


def test_format_contract():
    assert issubclass(ArrowFormat, DataFormat)
    assert issubclass(DataFormat, Format)
    assert ArrowFormat().name == "arrow"


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("float64", pa.float64()),
        ("float16", pa.float16()),
        ("int8", pa.int8()),
        ("uint64", pa.uint64()),
        ("S16", pa.binary(16)),
    ],
)
def test_zero_copy_type_mapping(dtype, expected):
    assert arrow_interop._zero_copy_arrow_type(pa, np.dtype(dtype)) == expected


@pytest.mark.parametrize("dtype", ["bool", "U8", "datetime64[D]"])
def test_non_zero_copy_types_declined(dtype):
    # bool (bit-packed), Unicode (no fixed Arrow type), coarse datetime units.
    assert arrow_interop._zero_copy_arrow_type(pa, np.dtype(dtype)) is None
