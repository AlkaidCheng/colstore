"""Tests for the no-transform merge fast path in the streaming write sink.

A pure passthrough merge (every output column is a bare, native-dtype,
on-disk column with no transform) skips the materializing per-batch write and
copies the sources' column bytes straight into the preallocated body. These
tests cover the contract: the fast path produces a file byte-identical to the
materializing path, across single-record, multi-record, and multi-file sources;
and the predicate declines (falls back) for anything that is not a pure merge --
a transform, a constant/in-memory column, a shared leaf, or a non-native dtype.

The strategy override (``_MERGE_COPY_OVERRIDE``) exercises the mmap copy
explicitly; ``copy_file_range`` is covered by the same assertions wherever the
platform provides it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import colstore
from colstore import format as fmt
from colstore import testing


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_concat(parts: list[Path], out: Path) -> None:
    """Write ``out`` via the materializing streaming path (merge plan disabled)."""
    original = fmt._merge_copy_plan
    fmt._merge_copy_plan = lambda *args, **kwargs: None  # type: ignore[assignment]
    try:
        colstore.concat(parts, out=out).close()
    finally:
        fmt._merge_copy_plan = original


@pytest.fixture(autouse=True)
def _reset_strategy_override():
    """Each test starts and ends with autodetected strategy and no chunking."""
    saved_override = fmt._MERGE_COPY_OVERRIDE
    saved_chunk = fmt._MERGE_COPY_CHUNK_BYTES
    fmt._MERGE_COPY_OVERRIDE = None
    fmt._MERGE_COPY_CHUNK_BYTES = 0
    yield
    fmt._MERGE_COPY_OVERRIDE = saved_override
    fmt._MERGE_COPY_CHUNK_BYTES = saved_chunk


# ---- Byte identity: the merge copy matches the materializing write ---------


@pytest.mark.parametrize("strategy", [None, "mmap", "cfr"])
def test_single_record_merge_is_byte_identical(tmp_path, strategy):
    fmt._MERGE_COPY_OVERRIDE = strategy
    parts = []
    for i in range(4):
        path = tmp_path / f"part_{i}.cstore"
        testing.make_store(path, rows=2000 + 137 * i, cols=3, dtype="float64", seed=i).close()
        parts.append(path)

    reference = tmp_path / "reference.cstore"
    _reference_concat(parts, reference)
    merged = tmp_path / "merged.cstore"
    colstore.concat(parts, out=merged).close()

    assert _digest(merged) == _digest(reference)


def test_chunked_merge_is_byte_identical(tmp_path):
    # Splitting runs into small chunks must not change the output: the sub-runs
    # are disjoint and in order, so the body is filled identically.
    parts = []
    for i in range(3):
        path = tmp_path / f"part_{i}.cstore"
        testing.make_store(path, rows=5000, cols=3, dtype="float64", seed=i).close()
        parts.append(path)

    reference = tmp_path / "reference.cstore"
    _reference_concat(parts, reference)
    fmt._MERGE_COPY_CHUNK_BYTES = 4096  # forces many sub-runs per column
    merged = tmp_path / "merged.cstore"
    colstore.concat(parts, out=merged).close()

    assert _digest(merged) == _digest(reference)


def test_chunk_plan_splits_and_covers():
    # A run larger than the chunk size splits into in-order, disjoint sub-runs
    # whose offsets and sizes exactly tile the original; small runs pass through.
    plan = [("a", 0, 100, 25), ("b", 1000, 200, 10)]
    chunked = fmt._chunk_plan(plan, 10)
    assert chunked == [
        ("a", 0, 100, 10),
        ("a", 10, 110, 10),
        ("a", 20, 120, 5),
        ("b", 1000, 200, 10),
    ]
    assert fmt._chunk_plan(plan, 0) == plan  # disabled: unchanged


def test_multi_record_sources_merge_byte_identical(tmp_path):
    # Sources with several records each (column data split across records) and
    # one file deliberately reused, so the plan stitches repeated runs in order.
    a = tmp_path / "a.cstore"
    testing.make_store(a, rows=900, cols=2, records=4, dtype="float64", seed=1).close()
    b = tmp_path / "b.cstore"
    testing.make_store(b, rows=37, cols=2, records=1, dtype="float64", seed=2).close()
    parts = [a, b, a]

    reference = tmp_path / "reference.cstore"
    _reference_concat(parts, reference)
    merged = tmp_path / "merged.cstore"
    colstore.concat(parts, out=merged).close()

    assert _digest(merged) == _digest(reference)


def test_mixed_dtypes_merge_byte_identical(tmp_path):
    # 1-byte and multi-byte kinds together: itemsize-1 columns carry byteorder
    # "|" and must still copy correctly alongside f8/i4/u2.
    parts = []
    for i in range(3):
        path = tmp_path / f"part_{i}.cstore"
        testing.make_store(
            path, rows=1500, cols=4, dtype=["float64", "int32", "uint16", "int8"], seed=i
        ).close()
        parts.append(path)

    reference = tmp_path / "reference.cstore"
    _reference_concat(parts, reference)
    merged = tmp_path / "merged.cstore"
    colstore.concat(parts, out=merged).close()

    assert _digest(merged) == _digest(reference)


def test_single_source_concat_is_byte_identical(tmp_path):
    only = tmp_path / "only.cstore"
    testing.make_store(only, rows=512, cols=2, dtype="float64", seed=7).close()

    reference = tmp_path / "reference.cstore"
    _reference_concat([only], reference)
    merged = tmp_path / "merged.cstore"
    colstore.concat([only], out=merged).close()

    assert _digest(merged) == _digest(reference)


def test_merged_values_round_trip(tmp_path):
    parts = []
    for i in range(3):
        path = tmp_path / f"part_{i}.cstore"
        testing.make_store(path, rows=1000, cols=2, dtype="float64", seed=i).close()
        parts.append(path)

    merged = colstore.concat(parts, out=tmp_path / "merged.cstore")
    try:
        with colstore.open(parts) as lazy:
            for name in lazy.columns:
                assert np.array_equal(merged[name].array(), lazy[name].array())
    finally:
        merged.close()


# ---- The predicate declines anything that is not a pure merge --------------


def _plan_for_frame(frame, store) -> object:
    """Build the merge plan for ``frame``'s specs as the writer would."""
    names = list(frame._columns)
    on_disk = {name: store.dtypes[name] for name in names if name in store.dtypes}
    # The planner declines before reading a non-passthrough column's dtype, so
    # any added/transformed columns absent from the source schema are filled
    # with a placeholder dtype to complete the mapping passed in.
    for name in names:
        on_disk.setdefault(name, np.dtype("float64"))
    return fmt._merge_copy_plan(frame._columns, names, on_disk, 0, store.n_rows)


def test_transform_declines_merge(tmp_path):
    store = testing.make_store(tmp_path / "s.cstore", rows=100, cols=2, dtype="float64")
    try:
        column = store.columns[0]
        frame = store.edit()
        frame["scaled"] = frame[column] * 2.0
        assert _plan_for_frame(frame, store) is None
    finally:
        store.close()


def test_constant_column_declines_merge(tmp_path):
    store = testing.make_store(tmp_path / "s.cstore", rows=100, cols=2, dtype="float64")
    try:
        frame = store.edit()
        frame["flag"] = 1  # ConstColumn, not a passthrough
        assert _plan_for_frame(frame, store) is None
    finally:
        store.close()


def test_shared_leaf_declines_merge(tmp_path):
    # Two output columns reading the same source leaf: fusible_passthroughs
    # excludes both (to keep the read memoized), so it is not a pure merge.
    store = testing.make_store(tmp_path / "s.cstore", rows=100, cols=2, dtype="float64")
    try:
        column = store.columns[0]
        frame = store.edit()
        frame["copy"] = frame[column]
        assert _plan_for_frame(frame, store) is None
    finally:
        store.close()


def test_non_native_dtype_declines_merge(tmp_path):
    # A big-endian on-disk column cannot be raw-copied; the run accessor raises
    # and the plan declines. _column_disk_runs is the enforcement point.
    store = testing.make_store(tmp_path / "s.cstore", rows=100, cols=1, dtype="float64")
    try:
        column = store.columns[0]
        store._column_dtypes[column] = store._column_dtypes[column].newbyteorder(">")
        with pytest.raises(ValueError):
            store._column_disk_runs(column)
    finally:
        store.close()


def test_copy_file_range_absent_raises_oserror(tmp_path, monkeypatch):
    # On a Linux interpreter whose os module lacks copy_file_range (e.g. a build
    # against an old glibc), the cfr strategy must raise OSError -- which the
    # executor catches to fall back to mmap -- not AttributeError, which would
    # crash the write. Real files and a non-empty run, so without the guard the
    # call would reach os.copy_file_range and raise AttributeError instead.
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 64)
    dst = tmp_path / "dst.bin"
    dst.write_bytes(b"\0" * 64)
    monkeypatch.setattr(fmt.sys, "platform", "linux")
    monkeypatch.delattr(fmt.os, "copy_file_range", raising=False)
    with pytest.raises(OSError):
        fmt._copy_plan_copy_file_range(str(dst), [(src, 0, 0, 64)], 1)


def test_disk_runs_reconstruct_column_bytes(tmp_path):
    # The runs of a multi-record column, concatenated, equal the column's bytes.
    store = testing.make_store(tmp_path / "s.cstore", rows=300, cols=2, records=3, dtype="float64")
    try:
        column = store.columns[0]
        runs = store._column_disk_runs(column)
        data = bytearray()
        for path, offset, nbytes in runs:
            with open(path, "rb") as handle:
                handle.seek(offset)
                data += handle.read(nbytes)
        expected = store[column].array().astype("<f8", copy=False).tobytes()
        assert bytes(data) == expected
    finally:
        store.close()
