"""Tests for the zero-copy read API (``copy=False``).

Contract: ``array(copy=False)`` / ``dict(copy=False)`` return READ-ONLY
ndarray views sharing memory with the store's open memmaps -- no bytes
copied -- exactly when the store is single-record, the dtype is native byte
order, and the selector is None / int / slice (any step). Every other case
raises ``ValueError`` (never a silent copy): fancy and boolean selectors,
multi-record stores (the error points at ``colstore.compact``), and
non-native dtypes. Views pin the underlying mapping, so they remain valid
after the reader is closed. ``copy=True`` (the default) is byte-for-byte
the previous behavior: owning, writable arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import reader as reader_mod


@pytest.fixture()
def single_record_store(tmp_path):
    rng = np.random.default_rng(51)
    data = {
        "f8": rng.standard_normal(10_000),
        "f4": rng.standard_normal(10_000).astype(np.float32),
        "i2": rng.integers(-(2**14), 2**14, 10_000).astype(np.int16),
    }
    path = tmp_path / "single.cstore"
    with colstore.create(path) as writer:
        writer.write(data)
    dataset = colstore.open(path)
    yield dataset, data
    dataset.close()


def test_full_column_view_shares_memory(single_record_store):
    dataset, data = single_record_store
    for name, values in data.items():
        view = dataset[name].array(copy=False)
        assert np.shares_memory(view, dataset._memmaps[name]), name
        assert not view.flags.writeable
        assert not view.flags.owndata
        assert type(view) is np.ndarray  # re-classed, not a memmap subclass
        assert view.dtype == values.dtype
        assert np.array_equal(view, values), name


@pytest.mark.parametrize(
    "selector",
    [slice(None), slice(100, 5_000), slice(7, 9_000, 13), slice(None, None, -1), slice(50, 10, -3)],
)
def test_slice_views_match_copies(single_record_store, selector):
    dataset, data = single_record_store
    view = dataset[selector, "f8"].array(copy=False)
    owning = dataset[selector, "f8"].array()
    assert np.shares_memory(view, dataset._memmaps["f8"])
    assert np.array_equal(view, owning)
    assert np.array_equal(view, data["f8"][selector])


def test_scalar_selector_view_shape_matches_copy(single_record_store):
    dataset, data = single_record_store
    view = dataset[42, "f8"].array(copy=False)
    assert view.shape == dataset[42, "f8"].array().shape == (1,)
    assert view[0] == data["f8"][42]
    assert np.shares_memory(view, dataset._memmaps["f8"])


def test_table_dict_views(single_record_store):
    dataset, data = single_record_store
    result = dataset[:, ["f8", "i2"]].dict(copy=False)
    assert list(result) == ["f8", "i2"]
    for name, view in result.items():
        assert np.shares_memory(view, dataset._memmaps[name]), name
        assert not view.flags.writeable
        assert np.array_equal(view, data[name]), name


def test_reader_dict_shortcut_views(single_record_store):
    dataset, data = single_record_store
    result = dataset.dict(copy=False)
    assert set(result) == set(data)
    for name, view in result.items():
        assert np.shares_memory(view, dataset._memmaps[name]), name


def test_views_are_immutable(single_record_store):
    dataset, _ = single_record_store
    view = dataset["f8"].array(copy=False)
    with pytest.raises(ValueError, match="read-only"):
        view[0] = 1.0


def test_copy_true_unchanged(single_record_store):
    dataset, data = single_record_store
    owning = dataset["f8"].array()
    assert owning.flags.writeable
    assert not np.shares_memory(owning, dataset._memmaps["f8"])
    owning[0] = 123.0  # mutating the copy must not touch the store
    assert dataset["f8"].array(copy=False)[0] == data["f8"][0]


def test_fancy_and_boolean_selectors_raise(single_record_store):
    dataset, _ = single_record_store
    with pytest.raises(ValueError, match="copy=True"):
        dataset[np.array([1, 5, 2]), "f8"].array(copy=False)
    mask = np.zeros(dataset.n_rows, dtype=bool)
    mask[::7] = True
    with pytest.raises(ValueError, match="copy=True"):
        dataset[mask, "f8"].array(copy=False)


def test_multi_record_store_raises_with_compact_hint(tmp_path):
    rng = np.random.default_rng(52)
    full = rng.standard_normal(2_000)
    path = tmp_path / "multi.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": full[:800]})
        writer.write({"a": full[800:]})
    dataset = colstore.open(path)
    try:
        with pytest.raises(ValueError, match="compact"):
            dataset["a"].array(copy=False)
        with pytest.raises(ValueError, match="compact"):
            dataset.dict(copy=False)
    finally:
        dataset.close()


def test_compacted_store_supports_zero_copy(tmp_path):
    rng = np.random.default_rng(53)
    full = rng.standard_normal(3_000)
    path = tmp_path / "tocompact.cstore"
    with colstore.create(path) as writer:
        for lo in range(0, 3_000, 500):
            writer.write({"a": full[lo : lo + 500]})
    compacted = tmp_path / "compacted.cstore"
    colstore.compact(path, out=compacted, show_progress=False)
    dataset = colstore.open(compacted)
    try:
        view = dataset["a"].array(copy=False)
        assert np.shares_memory(view, dataset._memmaps["a"])
        assert np.array_equal(view, full)
    finally:
        dataset.close()


def test_non_native_dtype_raises(single_record_store, monkeypatch):
    dataset, _ = single_record_store
    monkeypatch.setattr(reader_mod, "_dtype_is_native", lambda dtype: False)
    with pytest.raises(ValueError, match="native byte order"):
        dataset["f8"].array(copy=False)


def test_view_survives_reader_close(tmp_path):
    rng = np.random.default_rng(54)
    data = rng.standard_normal(5_000)
    path = tmp_path / "lifetime.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": data})
    dataset = colstore.open(path)
    view = dataset["a"].array(copy=False)
    dataset.close()
    # The view pins the mapping via .base; reads remain valid after close.
    assert np.array_equal(view, data)
    assert float(view.sum()) == pytest.approx(float(data.sum()))


def test_closed_reader_rejects_new_views(single_record_store, tmp_path):
    rng = np.random.default_rng(55)
    path = tmp_path / "closed.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": rng.standard_normal(100)})
    dataset = colstore.open(path)
    column_view = dataset["a"]  # lazy view created while open
    dataset.close()
    with pytest.raises(ValueError, match="closed"):
        column_view.array(copy=False)
