"""Tests for fixed-width strings, datetime, dtype rejection, and byte order."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from colstore import ColStore
from colstore import format as fmt
from colstore.kernels import cpp_available, numba_available

_BACKENDS = ["numpy"]
if cpp_available():
    _BACKENDS.append("cpp")
if numba_available():
    _BACKENDS.append("numba")


def test_complex_dtype_rejected(tmp_path):
    with pytest.raises(TypeError, match="unsupported dtype kind"):
        fmt.write_dataset(
            {"z": np.array([1 + 2j], dtype=np.complex128)},
            tmp_path / "z.cstore",
            batch_size=1000,
            show_progress=False,
        )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fixed_width_bytes_roundtrip(tmp_path, backend):
    columns = {"name": np.array([b"alice", b"bob", b"carol"], dtype="S8")}
    store = ColStore.from_dict(columns, tmp_path / "s.cstore", show_progress=False, backend=backend)
    result = store[np.array([2, 0]), "name"].to_array()
    assert result.tolist() == [b"carol", b"alice"]
    store.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fixed_width_unicode_roundtrip(tmp_path, backend):
    columns = {"label": np.array(["alpha", "beta", "gamma"], dtype="U10")}
    store = ColStore.from_dict(columns, tmp_path / "u.cstore", show_progress=False, backend=backend)
    assert store[1:3, "label"].to_array().tolist() == ["beta", "gamma"]
    # Fancy index exercises the kernel-fallback path for unicode.
    assert store[np.array([2, 0]), "label"].to_array().tolist() == ["gamma", "alpha"]
    store.close()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_datetime64_roundtrip(tmp_path, backend):
    frame = pd.DataFrame({"t": pd.to_datetime(["2020-01-01", "2021-06-15"])})
    # cpp/numba backends must not warn here; they silently fall back to NumPy.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = ColStore.from_dataframe(
            frame, tmp_path / "dt.cstore", show_progress=False, backend=backend
        )
        result = store[np.array([1, 0]), "t"].to_array()
    assert np.array_equal(result, frame["t"].to_numpy()[[1, 0]])
    store.close()


def test_big_endian_input_stored_little_endian(tmp_path):
    path = tmp_path / "be.cstore"
    fmt.write_dataset({"v": np.arange(5, dtype=">i4")}, path, batch_size=1000, show_progress=False)
    manifest, _ = fmt.read_header(path)
    assert manifest["columns"][0]["dtype"] == "<i4"
    store = ColStore(path, backend="numpy")
    assert store["v"].to_array().tolist() == [0, 1, 2, 3, 4]
    assert store.dtypes["v"].byteorder in ("=", "<", "|")
    store.close()
