"""Byte-order handling of the multi-record fancy-read fallback paths.

The raw byte-offset gather (``gather_bytes``) copies on-disk little-endian
bytes verbatim, so its destination must be typed with the *disk* dtype and
converted to native order at the return -- otherwise a big-endian host would
misinterpret every value. On little-endian hosts the conversion is a no-op
(the dtypes compare equal); these tests force the non-native branch to
execute on this host and pin that it matches the native route exactly. True
big-endian verification needs BE hardware; the construction -- gather into
disk dtype, convert at return -- is correct by typing.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import reader as reader_mod
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


@pytest.fixture()
def multi_record_store(tmp_path):
    rng = np.random.default_rng(6)
    total = 6_000
    full = {
        "f8": rng.standard_normal(total),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
    }
    path = tmp_path / "m.cstore"
    with colstore.create(path) as writer:
        for offset in range(0, total, 500):
            writer.write({k: v[offset : offset + 500] for k, v in full.items()})
    return path, full, total


def test_forced_non_native_fancy_paths_match(multi_record_store, monkeypatch):
    path, full, total = multi_record_store
    indices = np.random.default_rng(7).integers(0, total, size=400).astype(np.int64)
    monkeypatch.setattr(reader_mod, "_dtype_is_native", lambda dtype: False)
    dataset = colstore.open(path)
    forced_unsorted = {name: dataset[indices, name].array() for name in full}
    forced_sorted = dataset[np.sort(indices), "f8"].array()
    dataset.close()
    monkeypatch.undo()
    for name, values in forced_unsorted.items():
        assert values.dtype == full[name].dtype.newbyteorder("=")
        assert np.array_equal(values, full[name][indices]), name
    assert np.array_equal(forced_sorted, full["f8"][np.sort(indices)])


def test_native_results_unchanged(multi_record_store):
    path, full, total = multi_record_store
    dataset = colstore.open(path)
    indices = np.random.default_rng(8).integers(0, total, size=400).astype(np.int64)
    for name in full:
        assert np.array_equal(dataset[indices, name].array(), full[name][indices]), name
    dataset.close()
