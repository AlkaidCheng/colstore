"""Reader routing of the multi-record gather family, and its edge cases.

Each section pins one routing contract end to end: which kernel a selector
class reaches (spied call sequences are pinned contracts), that the route
and the fallback it replaces produce identical results across each seam
(forced detection-off, forced non-native, rbase gate, mask-density gate),
and that the contract survives the edge cases that historically broke it
(non-contiguous selectors, misaligned columns, single-record stores).
Direct kernel contracts live in ``test_multirecord_kernels.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from _helpers import kernel_spy, opened, write_records, write_standard_store

import colstore
from colstore import config as config_mod
from colstore import reader as reader_mod
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


# ---- Sortedness detection ---------------------------------------------------
# Helper contract: ``_indices_are_sorted`` must agree with the full check
# ``bool(np.all(indices[1:] >= indices[:-1]))`` on every input -- sampling is
# only ever used to reject sortedness, never to prove it, so an adversarial
# nearly-sorted array whose only descent falls between probe positions must
# still come back False via the full pass.

THRESHOLD = reader_mod._SORTEDNESS_SAMPLE_MIN_SIZE

# Mirrors PARALLEL_THRESHOLD in include/colstore/gather.hpp: the gather
# kernels run serial below this index count and work-proportionally parallel
# at or above it. The bin-reuse routing gate only engages in that parallel
# regime, so the gate tests below draw selectors past this size.
_KERNEL_PARALLEL_THRESHOLD = 1 << 18


def _full_check(indices: np.ndarray) -> bool:
    return bool(np.all(indices[1:] >= indices[:-1]))


def _sample_positions(n: int) -> np.ndarray:
    return (reader_mod._SORTEDNESS_SAMPLE_FRACTIONS * (n - 2)).astype(np.int64)


@pytest.mark.parametrize("n", [0, 1, 2, 100, THRESHOLD - 1, THRESHOLD, THRESHOLD * 4])
def test_helper_agrees_with_full_check_on_battery(n):
    rng = np.random.default_rng(n)
    candidates = [
        np.sort(rng.integers(0, 10**9, n).astype(np.int64)),  # sorted
        rng.integers(0, 10**9, n).astype(np.int64),  # random
        np.full(n, 7, dtype=np.int64),  # all-duplicate (non-decreasing)
        np.arange(n, dtype=np.int64)[::-1].copy(),  # descending
    ]
    for indices in candidates:
        assert reader_mod._indices_are_sorted(indices) == _full_check(indices)


def test_descent_between_probe_positions_is_still_caught():
    # Sorted array with exactly one adjacent descent placed away from every
    # sampled pair: the sampler must pass and the full check must decide.
    n = THRESHOLD * 2
    indices = np.arange(n, dtype=np.int64)
    probed = set(_sample_positions(n).tolist())
    pos = next(p for p in range(n // 3, n) if p not in probed and (p + 1) not in probed)
    indices[pos], indices[pos + 1] = indices[pos + 1], indices[pos]
    assert not _full_check(indices)
    assert reader_mod._indices_are_sorted(indices) is False


def test_descent_at_probe_positions_is_caught_by_sampling():
    n = THRESHOLD * 2
    indices = np.arange(n, dtype=np.int64)
    pos = int(_sample_positions(n)[7])
    indices[pos], indices[pos + 1] = indices[pos + 1], indices[pos]
    assert reader_mod._indices_are_sorted(indices) is False


def test_sorted_above_threshold_returns_true():
    n = THRESHOLD * 8
    indices = np.sort(np.random.default_rng(3).integers(0, 10**12, n).astype(np.int64))
    assert reader_mod._indices_are_sorted(indices) is True


@pytest.fixture()
def sortedness_store(tmp_path):
    rng = np.random.default_rng(21)
    # Irregular record sizes (first record split unevenly): these tests pin
    # the GENERIC kernels' routing, which uniform-record files no longer
    # take (they route to the arithmetic-binning kernels, covered below).
    # Total 160_000 rows so selectors can exceed THRESHOLD.
    rows_per_record = [1_500, 2_500] + [4_000] * 39
    total = sum(rows_per_record)
    full = {
        "f8": rng.standard_normal(total),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
    }
    path = tmp_path / "sortedness.cstore"
    write_records(path, full, rows_per_record)
    return path, full, total


@pytest.mark.parametrize("size", [100, THRESHOLD * 2])
def test_routing_unchanged_for_sorted_and_unsorted(sortedness_store, monkeypatch, size):
    path, full, total = sortedness_store
    rng = np.random.default_rng(size)
    unsorted_idx = rng.integers(0, total, size).astype(np.int64)
    sorted_idx = np.sort(unsorted_idx)

    routes = kernel_spy(monkeypatch, ["gather_multirecord_sorted", "gather_multirecord"])
    with opened(path) as dataset:
        assert np.array_equal(dataset[sorted_idx, "f8"].array(), full["f8"][sorted_idx])
        assert routes == ["gather_multirecord_sorted"]
        routes.clear()
        assert np.array_equal(dataset[unsorted_idx, "f8"].array(), full["f8"][unsorted_idx])
        assert routes == ["gather_multirecord"]


@pytest.mark.parametrize("size", [1_000, THRESHOLD * 2])
def test_bin_reuse_gate_unchanged(sortedness_store, monkeypatch, size):
    path, full, total = sortedness_store
    rng = np.random.default_rng(size + 1)
    unsorted_idx = rng.integers(0, total, size).astype(np.int64)
    sorted_idx = np.sort(unsorted_idx)

    bins_calls = kernel_spy(monkeypatch, ["gather_multirecord_bins"])
    with opened(path) as dataset:
        result = dataset[unsorted_idx, ["f8", "i4"]].dict()
        assert len(bins_calls) == 1  # unsorted multi-column: bin-reuse route
        assert np.array_equal(result["f8"], full["f8"][unsorted_idx])
        assert np.array_equal(result["i4"], full["i4"][unsorted_idx])
        result = dataset[sorted_idx, ["f8", "i4"]].dict()
        assert len(bins_calls) == 1  # sorted multi-column: per-column path, no bins
        assert np.array_equal(result["f8"], full["f8"][sorted_idx])


def test_single_element_selector_keeps_fused_route(sortedness_store, monkeypatch):
    # n == 1 historically routes to the fused (unsorted) kernel via the
    # ``n > 1 and`` guard; the sortedness helper must not move it.
    path, full, _ = sortedness_store
    routes = kernel_spy(monkeypatch, ["gather_multirecord_sorted", "gather_multirecord"])
    with opened(path) as dataset:
        assert dataset[np.array([42], dtype=np.int64), "f8"].array()[0] == full["f8"][42]
        assert routes == ["gather_multirecord"]


# ---- Sorted-read routing ----------------------------------------------------
# Sorted native fancy reads engage the walk kernel; unsorted reads do not;
# results match the boundary-partition pipeline the kernel replaces (which
# survives as the non-native fallback).


@pytest.fixture()
def sorted_mixed_store(tmp_path):
    rng = np.random.default_rng(7)
    total = 50_000
    full = {
        "f8": rng.standard_normal(total),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
        "pad": rng.integers(-9, 9, total).astype(np.int8),
    }
    path = tmp_path / "m.cstore"
    write_records(path, full, [500] * 100)
    return path, full, total


def test_sorted_reads_route_through_walk_kernel(sorted_mixed_store, monkeypatch):
    path, full, total = sorted_mixed_store
    calls = kernel_spy(monkeypatch, ["gather_multirecord_sorted"])
    dataset = colstore.open(path)
    indices = np.sort(np.random.default_rng(9).integers(0, total, size=20_000).astype(np.int64))
    for name in full:
        assert np.array_equal(dataset[indices, name].array(), full[name][indices]), name
    assert len(calls) == len(full)
    dataset.close()


def test_unsorted_reads_do_not_route(sorted_mixed_store, monkeypatch):
    path, full, total = sorted_mixed_store
    calls = kernel_spy(monkeypatch, ["gather_multirecord_sorted"])
    dataset = colstore.open(path)
    indices = np.random.default_rng(10).integers(0, total, size=5_000).astype(np.int64)
    assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
    assert calls == []
    dataset.close()


def test_sorted_matches_partition_pipeline_it_replaces(sorted_mixed_store, monkeypatch):
    # The boundary-partition pipeline survives as the non-native fallback;
    # forcing it must give identical results to the walk kernel route.
    path, full, total = sorted_mixed_store
    indices = np.sort(np.random.default_rng(11).integers(0, total, size=15_000).astype(np.int64))
    dataset = colstore.open(path)
    via_kernel = {name: dataset[indices, name].array() for name in full}
    dataset.close()
    monkeypatch.setattr(reader_mod, "_dtype_is_native", lambda dtype: False)
    dataset = colstore.open(path)
    via_pipeline = {name: dataset[indices, name].array() for name in full}
    dataset.close()
    for name in full:
        assert np.array_equal(via_kernel[name], via_pipeline[name]), name
        assert np.array_equal(via_kernel[name], full[name][indices]), name


# ---- Strided-slice routing --------------------------------------------------
# Multi-record slices with ``step != 1`` and a native dtype engage the
# strided kernel; non-native dtypes keep the ``np.arange`` + fancy fallback;
# results match the fancy path the route replaces on identical selectors.


@pytest.fixture()
def strided_store(tmp_path):
    rows_per_record = [137, 64, 1, 350, 99, 200, 13, 470, 5, 261]
    path, full, total = write_standard_store(tmp_path, rows_per_record, seed=11, name="strided")
    return path, full, total


STEPS = [2, 3, 10, -1, -2, -7, 1000, -1000]


@pytest.mark.parametrize("step", STEPS)
def test_reader_strided_slice_matches_ground_truth(strided_store, step):
    path, full, total = strided_store
    with opened(path) as dataset:
        for name, values in full.items():
            result = dataset[::step, name].array()
            assert result.dtype == values.dtype
            assert np.array_equal(result, values[::step]), (name, step)
        offset_result = dataset[5 : total - 5 : step, "f8"].array()
        assert np.array_equal(offset_result, full["f8"][5 : total - 5 : step])


def test_reader_strided_slice_matches_fancy_path(strided_store):
    # The route this kernel replaces: explicit arange selector through the
    # fancy path. Identical selectors must produce identical results.
    path, _full, total = strided_store
    with opened(path) as dataset:
        for step in (4, -4):
            indices = np.arange(*slice(None, None, step).indices(total), dtype=np.int64)
            assert np.array_equal(
                dataset[::step, "f4"].array(), dataset[indices, "f4"].array()
            ), step


def test_reader_strided_multi_column_dict(strided_store):
    path, full, _ = strided_store
    with opened(path) as dataset:
        result = dataset[::3, ["f8", "i2"]].dict()
        assert np.array_equal(result["f8"], full["f8"][::3])
        assert np.array_equal(result["i2"], full["i2"][::3])


def test_reader_routes_strided_slices_to_kernel(strided_store, monkeypatch):
    path, _, _ = strided_store
    calls = kernel_spy(monkeypatch, ["gather_multirecord_strided"])
    with opened(path) as dataset:
        dataset[::2, "f8"].array()
        assert len(calls) == 1
        dataset[100:1000, "f8"].array()  # unit step: contiguous route, not the kernel
        assert len(calls) == 1
        dataset[::-1, "i2"].array()
        assert len(calls) == 2


def test_reader_non_native_dtype_falls_back(strided_store, monkeypatch):
    # Forcing the native check false must bypass the strided kernel (raw
    # typed loads cannot byteswap) and still return correct, native-order
    # values via the arange + fancy fallback.
    path, full, _ = strided_store
    calls = kernel_spy(monkeypatch, ["gather_multirecord_strided"])
    monkeypatch.setattr(reader_mod, "_dtype_is_native", lambda dtype: False)
    with opened(path) as dataset:
        result = dataset[::5, "f8"].array()
    monkeypatch.undo()
    assert not calls
    assert result.dtype == np.dtype(np.float64).newbyteorder("=")
    assert np.array_equal(result, full["f8"][::5])


# ---- Uniform-record detection and routing -----------------------------------
# A layout qualifies exactly when every record but the last has the same row
# count, the last is no larger, and the body stride is constant; anything
# else returns ``None`` and the generic route is taken.


@pytest.mark.parametrize("tail", [200, 57])
def test_detection_accepts_uniform_layouts(tmp_path, tail):
    path, _, _ = write_standard_store(tmp_path, [200] * 9 + [tail], seed=11)
    with opened(path) as dataset:
        layout = dataset._uniform_record_layout()
        assert layout is not None
        rows, stride, first_body, last_rows = layout
        assert rows == 200 and last_rows == tail
        assert stride > 0 and first_body >= 0


@pytest.mark.parametrize(
    "shape",
    [
        [200] * 5 + [201] + [200] * 4,  # interior record differs
        [200] * 9 + [300],  # last record LARGER than the others
        [100, 200, 200, 200],  # first record differs
    ],
)
def test_detection_rejects_irregular_layouts(tmp_path, shape):
    path, _, _ = write_standard_store(tmp_path, shape, seed=11)
    with opened(path) as dataset:
        assert dataset._uniform_record_layout() is None


def test_uniform_store_routes_single_column_to_uniform_kernel(tmp_path, monkeypatch):
    path, full, total = write_standard_store(tmp_path, [500] * 8, seed=11)
    calls = kernel_spy(monkeypatch, ["gather_multirecord_uniform", "gather_multirecord"])
    indices = np.random.default_rng(12).integers(0, total, 700).astype(np.int64)
    with opened(path) as dataset:
        for name, values in full.items():
            assert np.array_equal(dataset[indices, name].array(), values[indices]), name
        assert calls == ["gather_multirecord_uniform"] * 3


def test_uniform_store_multi_column_uses_uniform_bins_pair(tmp_path, monkeypatch):
    path, full, total = write_standard_store(tmp_path, [500] * 7 + [123], seed=11)
    calls = kernel_spy(
        monkeypatch,
        [
            "gather_multirecord_uniform_bins",
            "gather_multirecord_uniform_withbins",
            "gather_multirecord_bins",
            "gather_multirecord_withbins",
        ],
    )
    indices = np.random.default_rng(13).integers(0, total, 900).astype(np.int64)
    with opened(path) as dataset:
        result = dataset[indices, ["f8", "f4", "i2"]].dict()
        assert calls == [
            "gather_multirecord_uniform_bins",
            "gather_multirecord_uniform_withbins",
            "gather_multirecord_uniform_withbins",
        ]
        for name in ("f8", "f4", "i2"):
            assert np.array_equal(result[name], full[name][indices]), name


def test_uniform_store_sorted_path_unaffected(tmp_path, monkeypatch):
    path, full, total = write_standard_store(tmp_path, [500] * 8, seed=11)
    calls = kernel_spy(monkeypatch, ["gather_multirecord_uniform", "gather_multirecord_sorted"])
    indices = np.sort(np.random.default_rng(15).integers(0, total, 700).astype(np.int64))
    with opened(path) as dataset:
        assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
        assert calls == ["gather_multirecord_sorted"]


def test_forced_generic_route_matches_uniform_route(tmp_path, monkeypatch):
    # The benchmark's baseline seam: detection forced to None must produce
    # identical results through the generic kernels.
    path, _full, total = write_standard_store(tmp_path, [400] * 10, seed=11)
    indices = np.random.default_rng(18).integers(0, total, 1_500).astype(np.int64)
    with opened(path) as dataset:
        via_uniform = dataset[indices, ["f8", "f4"]].dict()
    monkeypatch.setattr(
        reader_mod.ColStoreReader, "_detect_uniform_record_layout", lambda self: None
    )
    with opened(path) as dataset:
        assert dataset._uniform_record_layout() is None
        via_generic = dataset[indices, ["f8", "f4"]].dict()
    for name in ("f8", "f4"):
        assert np.array_equal(via_uniform[name], via_generic[name]), name


# ---- Irregular multi-column routing: bin reuse and record-base --------------
# Unsorted multi-column fancy reads on irregular stores engage the bins
# route (first column binned, the rest served from the bins); above the
# rbase size gate the trailing columns take the record-base variant.

BINS_KERNELS = [
    "gather_multirecord_bins",
    "gather_multirecord_withbins",
    "gather_multirecord_withbins_rbase",
    "gather_multirecord_uniform_bins",
]


@pytest.fixture()
def bin_reuse_store(tmp_path):
    rng = np.random.default_rng(7)
    full = {
        "f8": rng.standard_normal(40_000),
        "f4": rng.standard_normal(40_000).astype(np.float32),
        "i4": rng.integers(-(2**20), 2**20, 40_000).astype(np.int32),
        "i2": rng.integers(-1000, 1000, 40_000).astype(np.int16),
    }
    path = tmp_path / "m.cstore"
    # Irregular record sizes (one record split unevenly): these tests pin the
    # GENERIC bins route, which uniform-record files no longer take.
    boundaries = [0, 500, *range(800, 40_001, 800)]
    rows_per_record = [hi - lo for lo, hi in itertools.pairwise(boundaries)]
    write_records(path, full, rows_per_record)
    return path, full


@pytest.fixture()
def rbase_irregular_store(tmp_path):
    shape = [137, 64, 350, 99, 200, 13, 470, 261, 88, 318]
    return write_standard_store(tmp_path, shape, seed=31, name="rbase")


@pytest.fixture()
def parallel_regime_store(tmp_path):
    # Enough rows to draw a parallel-regime fancy index (>= the kernel
    # threshold). Irregular records pin the generic bins route; two native
    # columns satisfy the bin-reuse precondition.
    rng = np.random.default_rng(118)
    total = 300_000
    full = {
        "f8": rng.standard_normal(total),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
    }
    path = tmp_path / "parallel.cstore"
    boundaries = [0, 500, *range(1_000, total + 1, 1_000)]
    rows_per_record = [hi - lo for lo, hi in itertools.pairwise(boundaries)]
    write_records(path, full, rows_per_record)
    return path, full, total


def _bin_reuse_widths(n: int) -> tuple[int, int]:
    """(sequential, concurrent) parallel widths the gate compares, for ``n``.

    Mirrors the gate in ``_gather_many_bin_reuse`` for a two-column read so a
    test can tell whether the running box can realize a decline at all (a
    decline needs the concurrent column pool to out-field the single
    sequential kernel, which is impossible when the OpenMP width is too low).
    """
    from colstore import _gather

    cap = config_mod.get_gather_thread_cap()
    sequential = _gather.resolve_thread_count(n, cap)
    n_workers = min(config_mod.get_max_workers(), 2)
    per_column_cap = max(1, cap // n_workers)
    concurrent = min(
        _gather.max_threads(), n_workers * _gather.resolve_thread_count(n, per_column_cap)
    )
    return sequential, concurrent


def test_parallel_regime_route_taken_without_concurrent_alternative(
    parallel_regime_store, monkeypatch
):
    # With a single worker the concurrent column pool cannot field more
    # parallel streams than the sequential bins route, so the gate keeps the
    # work-saving route in the parallel regime (and the serial regime never
    # gates at all). Portable: holds whether or not OpenMP can parallelize.
    path, full, total = parallel_regime_store
    n = _KERNEL_PARALLEL_THRESHOLD * 2
    indices = np.random.default_rng(5).integers(0, total, size=n).astype(np.int64)
    bins_calls = kernel_spy(monkeypatch, ["gather_multirecord_bins"])
    original = config_mod.get_max_workers()
    try:
        config_mod.set_max_workers(1)
        with opened(path) as dataset:
            result = dataset[indices, ["f8", "i4"]].dict()
        assert bins_calls == ["gather_multirecord_bins"]  # route taken
        assert np.array_equal(result["f8"], full["f8"][indices])
        assert np.array_equal(result["i4"], full["i4"][indices])
    finally:
        config_mod.set_max_workers(original)


def test_parallel_regime_route_declines_when_pool_fields_more_threads(
    parallel_regime_store, monkeypatch
):
    # Multiple workers let the concurrent column pool field one resolved width
    # per column at once; in the parallel regime that out-fields the single
    # sequential kernel, so the gate declines and the read falls through to
    # the per-column pool -- no bins kernel runs. Skipped when the box's
    # OpenMP width is too low to realize a decline (e.g. a single-core CI box).
    path, full, total = parallel_regime_store
    n = _KERNEL_PARALLEL_THRESHOLD * 2
    original_workers = config_mod.get_max_workers()
    original_cap = config_mod.get_gather_thread_cap()
    try:
        config_mod.set_max_workers(4)
        config_mod.set_gather_thread_cap(8)
        sequential, concurrent = _bin_reuse_widths(n)
        if not (sequential > 1 and concurrent > sequential):
            pytest.skip("OpenMP width too low to realize a bin-reuse decline")
        indices = np.random.default_rng(6).integers(0, total, size=n).astype(np.int64)
        bins_calls = kernel_spy(monkeypatch, ["gather_multirecord_bins"])
        with opened(path) as dataset:
            result = dataset[indices, ["f8", "i4"]].dict()
        assert bins_calls == []  # route declined: fell through to column pool
        assert np.array_equal(result["f8"], full["f8"][indices])
        assert np.array_equal(result["i4"], full["i4"][indices])
    finally:
        config_mod.set_max_workers(original_workers)
        config_mod.set_gather_thread_cap(original_cap)


def test_multicolumn_read_routes_and_matches_per_column(bin_reuse_store, monkeypatch):
    path, full = bin_reuse_store
    calls = kernel_spy(
        monkeypatch, ["gather_multirecord_withbins", "gather_multirecord_withbins_rbase"]
    )
    dataset = colstore.open(path)
    indices = np.random.default_rng(9).integers(0, 40_000, size=15_000).astype(np.int64)

    table = dataset[indices, list(full)].dict()
    assert len(calls) == len(full) - 1  # first column binned, rest reused
    assert list(table) == list(full)  # requested order preserved
    for name, column in full.items():
        assert np.array_equal(table[name], column[indices]), name
        assert np.array_equal(
            table[name], dataset[indices, name].array()
        ), f"per-column mismatch: {name}"
    dataset.close()


def test_duplicates_and_reversed_indices(bin_reuse_store):
    path, full = bin_reuse_store
    dataset = colstore.open(path)
    indices = np.array([39_999, 0, 5, 5, 39_999, 17, 0], dtype=np.int64)
    table = dataset[indices, ["f8", "i4"]].dict()
    assert np.array_equal(table["f8"], full["f8"][indices])
    assert np.array_equal(table["i4"], full["i4"][indices])
    dataset.close()


def test_route_not_taken_for_sorted_single_column_or_slice(bin_reuse_store, monkeypatch):
    path, full = bin_reuse_store
    calls = kernel_spy(
        monkeypatch, ["gather_multirecord_withbins", "gather_multirecord_withbins_rbase"]
    )
    dataset = colstore.open(path)
    indices = np.random.default_rng(2).integers(0, 40_000, size=5_000).astype(np.int64)

    sorted_table = dataset[np.sort(indices), ["f8", "f4"]].dict()
    assert np.array_equal(sorted_table["f8"], full["f8"][np.sort(indices)])
    single = dataset[indices, "f8"].array()
    assert np.array_equal(single, full["f8"][indices])
    sliced = dataset[100:900, ["f8", "f4"]].dict()
    assert np.array_equal(sliced["f8"], full["f8"][100:900])
    assert calls == []  # none of the above may engage the bins kernels
    dataset.close()


def test_large_read_routes_trailing_columns_to_rbase(rbase_irregular_store, monkeypatch):
    path, full, total = rbase_irregular_store
    calls = kernel_spy(monkeypatch, BINS_KERNELS)
    # n >= n_records * gate: 10 records, 5000 indices -> rbase engaged.
    indices = np.random.default_rng(32).integers(0, total, 5_000).astype(np.int64)
    with opened(path) as dataset:
        result = dataset[indices, ["f8", "f4", "i2"]].dict()
        assert calls == [
            "gather_multirecord_bins",
            "gather_multirecord_withbins_rbase",
            "gather_multirecord_withbins_rbase",
        ]
        for name in ("f8", "f4", "i2"):
            assert np.array_equal(result[name], full[name][indices]), name


def test_small_read_keeps_generic_withbins(tmp_path, monkeypatch):
    # 2000 records, 30 indices: below the gate, generic withbins retained.
    path, full, total = write_standard_store(tmp_path, [17, 23] * 1000, seed=33, name="rbase")
    calls = kernel_spy(monkeypatch, BINS_KERNELS)
    indices = np.random.default_rng(34).integers(0, total, 30).astype(np.int64)
    with opened(path) as dataset:
        result = dataset[indices, ["f8", "i2"]].dict()
        assert calls == ["gather_multirecord_bins", "gather_multirecord_withbins"]
        assert np.array_equal(result["f8"], full["f8"][indices])


def test_uniform_store_keeps_uniform_pair(tmp_path, monkeypatch):
    path, full, total = write_standard_store(tmp_path, [300] * 12, seed=35, name="rbase")
    calls = kernel_spy(monkeypatch, BINS_KERNELS)
    indices = np.random.default_rng(36).integers(0, total, 2_000).astype(np.int64)
    with opened(path) as dataset:
        result = dataset[indices, ["f8", "f4"]].dict()
        assert calls == ["gather_multirecord_uniform_bins"]  # withbins variants unspied here
        assert np.array_equal(result["f8"], full["f8"][indices])


def test_rbase_route_matches_forced_generic_route(rbase_irregular_store, monkeypatch):
    path, full, total = rbase_irregular_store
    indices = np.random.default_rng(37).integers(0, total, 8_000).astype(np.int64)
    with opened(path) as dataset:
        via_rbase = dataset[indices, ["f8", "f4", "i2"]].dict()
    monkeypatch.setattr(reader_mod, "_RBASE_MIN_INDICES_PER_RECORD", float("inf"))
    with opened(path) as dataset:
        via_generic = dataset[indices, ["f8", "f4", "i2"]].dict()
    for name in ("f8", "f4", "i2"):
        assert np.array_equal(via_rbase[name], via_generic[name]), name
        assert np.array_equal(via_rbase[name], full[name][indices]), name


# ---- Boolean-mask routing ---------------------------------------------------
# Multi-record reads with native dtypes and mask density at or above the
# gate take the mask kernel; sparse masks, single-record stores, and
# non-native dtypes lower to ``np.flatnonzero`` and the fancy paths.

# Routing tests pin an explicit nonzero gate so both sides of the route
# are exercised regardless of the compiled default (0.0: always native)
# or any calibration cache on the dev machine.
GATE = 0.15

MASK_SPIED = ["gather_segment_mask", "gather_multirecord_sorted", "gather_multirecord_bins"]


@pytest.fixture()
def mask_irregular_store(tmp_path):
    shape = [137, 640, 350, 99, 2000, 13, 470, 2610, 88, 318]
    return write_standard_store(tmp_path, shape, seed=61, name="mask")


def test_dense_mask_routes_to_mask_kernel(mask_irregular_store, monkeypatch):
    path, full, total = mask_irregular_store
    monkeypatch.setattr(config_mod, "_mask_density_gate", GATE)
    calls = kernel_spy(monkeypatch, MASK_SPIED)
    mask = np.random.default_rng(62).random(total) < 0.5
    with opened(path) as dataset:
        assert np.array_equal(dataset[mask, "f8"].array(), full["f8"][mask])
        assert calls == ["gather_segment_mask"]
        calls.clear()
        result = dataset[mask, ["f8", "f4", "i2"]].dict()
        assert calls == ["gather_segment_mask"] * 3
        for name in ("f8", "f4", "i2"):
            assert np.array_equal(result[name], full[name][mask]), name


def test_sparse_mask_lowers_to_indices(mask_irregular_store, monkeypatch):
    path, full, total = mask_irregular_store
    monkeypatch.setattr(config_mod, "_mask_density_gate", GATE)
    calls = kernel_spy(monkeypatch, MASK_SPIED)
    mask = np.zeros(total, dtype=bool)
    mask[:: int(2 / GATE)] = True  # density well below the gate
    assert mask.mean() < GATE
    with opened(path) as dataset:
        assert np.array_equal(dataset[mask, "f8"].array(), full["f8"][mask])
        assert calls == ["gather_multirecord_sorted"]  # flatnonzero is sorted


def test_gate_seam_parity(mask_irregular_store, monkeypatch):
    path, full, total = mask_irregular_store
    mask = np.random.default_rng(63).random(total) < 0.4
    with opened(path) as dataset:
        via_mask = dataset[mask, ["f8", "i2"]].dict()
        monkeypatch.setattr(config_mod, "_mask_density_gate", 2.0)
        via_indices = dataset[mask, ["f8", "i2"]].dict()
    for name in ("f8", "i2"):
        assert np.array_equal(via_mask[name], via_indices[name]), name
        assert np.array_equal(via_mask[name], full[name][mask]), name


def test_mask_zero_copy_still_rejected(tmp_path):
    rng = np.random.default_rng(66)
    path = tmp_path / "zc.cstore"
    with colstore.create(path) as writer:
        writer.write({"a": rng.standard_normal(100)})
    with opened(path) as dataset, pytest.raises(ValueError, match="copy=True"):
        dataset[np.ones(100, dtype=bool), "a"].array(copy=False)


# ---- Backend and byte-order fallbacks ----------------------------------------
# ``backend`` selects the kernel for single-record fancy-index reads;
# multi-record stores require the compiled C++ extension and always use it
# for fancy reads. The raw byte-offset fallback gathers on-disk little-endian
# bytes into the disk dtype and converts to native order at the return; on
# little-endian hosts the conversion is a no-op, so these tests force the
# non-native branch and pin that it matches the native route exactly.


def test_multirecord_reads_work_under_numpy_backend(tmp_path):
    rng = np.random.default_rng(8)
    total = 4_000
    full = {"f8": rng.standard_normal(total), "i4": rng.integers(0, 99, total).astype(np.int32)}
    path = tmp_path / "m.cstore"
    write_records(path, full, [400] * 10)

    dataset = colstore.open(path, backend="numpy")
    assert dataset.backend == "numpy"
    indices = rng.integers(0, total, size=300).astype(np.int64)
    # Single column (unsorted fancy) and the multi-column bin-reuse route
    # both run -- and run through C++ -- regardless of the backend value.
    assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
    table = dataset[indices, ["f8", "i4"]].dict()
    assert np.array_equal(table["i4"], full["i4"][indices])
    dataset.close()


def test_forced_non_native_fancy_paths_match(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    total = 6_000
    full = {
        "f8": rng.standard_normal(total),
        "i4": rng.integers(-(2**20), 2**20, total).astype(np.int32),
    }
    path = tmp_path / "m.cstore"
    write_records(path, full, [500] * 12)
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


# ---- Non-contiguous selectors -------------------------------------------------
# A strided index view (``rows[::2]``, ``rows[::-1]``) used to reach the C++
# kernels unchanged: positive strides read the wrong positions, negative
# strides read out of bounds (observed segfault). The view boundary now
# normalizes with ``np.ascontiguousarray``, the reader's conversion points do
# the same defensively, and every Cython entry validates contiguity (the
# direct-API backstop is pinned in ``test_multirecord_kernels.py``).


@pytest.fixture()
def noncontig_store(tmp_path):
    rng = np.random.default_rng(0)
    total = 5_000
    full = {
        "a": rng.standard_normal(total),
        "b": rng.integers(-(2**20), 2**20, total).astype(np.int32),
        "c": rng.standard_normal(total).astype(np.float32),
    }
    path = tmp_path / "m.cstore"
    write_records(path, full, [250] * 20)
    return path, full, total


def _strided_selectors(total: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    unsorted_base = rng.permutation(total).astype(np.int64)
    return {
        "sorted_step2": np.arange(total, dtype=np.int64)[::2],
        "sorted_step7": np.arange(total, dtype=np.int64)[::7],
        "reversed": np.arange(total, dtype=np.int64)[::-1],  # was a segfault
        "unsorted_step2": unsorted_base[::2],
        "unsorted_negative_step": unsorted_base[::-3],
        "int32_step2": np.arange(total, dtype=np.int32)[::2],
    }


def test_multi_record_strided_selectors(noncontig_store):
    path, full, total = noncontig_store
    dataset = colstore.open(path)
    for name, selector in _strided_selectors(total).items():
        expected = selector.astype(np.int64)
        for column in full:
            got = dataset[selector, column].array()
            assert np.array_equal(got, full[column][expected]), (name, column)
    dataset.close()


def test_multi_record_strided_multicolumn_bin_reuse(noncontig_store):
    path, full, total = noncontig_store
    dataset = colstore.open(path)
    selector = np.random.default_rng(2).permutation(total).astype(np.int64)[::2]
    table = dataset[selector, list(full)].dict()
    for column in full:
        assert np.array_equal(table[column], full[column][selector]), column
    dataset.close()


def test_single_record_strided_selectors(tmp_path):
    rng = np.random.default_rng(3)
    total = 3_000
    full = {"a": rng.standard_normal(total)}
    path = tmp_path / "s.cstore"
    colstore.store(full, path, show_progress=False)
    dataset = colstore.open(path)
    for name, selector in _strided_selectors(total).items():
        expected = selector.astype(np.int64)
        got = dataset[selector, "a"].array()
        assert np.array_equal(got, full["a"][expected]), name
    dataset.close()


def test_strided_selector_reproducer_verbatim(tmp_path):
    # Minimal reproducer of the original negative-stride segfault, verbatim.
    path = tmp_path / "x.cstore"
    a = np.arange(20, dtype=np.int64)
    with colstore.create(path) as writer:
        writer.write({"a": a[:10]})
        writer.write({"a": a[10:]})
    dataset = colstore.open(path)
    idx = np.arange(20, dtype=np.int64)[::2]
    assert not idx.flags.c_contiguous
    assert np.array_equal(dataset[idx, "a"].array(), a[idx])
    dataset.close()


def test_boolean_mask_selectors_unaffected(noncontig_store):
    path, full, total = noncontig_store
    dataset = colstore.open(path)
    mask = np.zeros(total, dtype=bool)
    mask[::3] = True
    got = dataset[mask, "a"].array()
    assert np.array_equal(got, full["a"][mask])
    dataset.close()


# ---- Misaligned columns (consolidated across routes) --------------------------
# Record bodies are packed with no inter-column padding, so an odd-length
# ``int8`` column puts every later column at odd byte addresses. The C++
# kernels must load such sources with alignment-safe ``memcpy`` loads:
# dereferencing a misaligned typed pointer is undefined behavior even where
# x86 tolerates it. These tests pin the *semantics* on every read path that
# reaches the kernels (one uniform-shape store for the uniform kernels, one
# irregular-shape store for the generic/bins/rbase kernels); the UB itself
# was verified by compiling the kernels under ``-fsanitize=alignment``.


def _write_misaligned_store(tmp_path, rows_per_record, columns, seed, name):
    rng = np.random.default_rng(seed)
    total = sum(rows_per_record)
    full = {"pad": rng.integers(-100, 100, total).astype(np.int8)}
    if "f8" in columns:
        full["f8"] = rng.standard_normal(total)
    if "f4" in columns:
        full["f4"] = rng.standard_normal(total).astype(np.float32)
    if "i4" in columns:
        full["i4"] = rng.integers(-(2**20), 2**20, total).astype(np.int32)
    path = tmp_path / f"{name}.cstore"
    write_records(path, full, rows_per_record)
    return path, full, total


@pytest.fixture()
def misaligned_uniform_store(tmp_path):
    # 12 records x 7 rows: uniform shape, routed through the uniform kernels.
    return _write_misaligned_store(tmp_path, [7] * 12, ("f8", "f4", "i4"), seed=3, name="mis_u")


@pytest.fixture()
def misaligned_irregular_store(tmp_path):
    # Irregular record sizes: record bases vary non-affinely; generic routes.
    shape = [7, 13, 7, 9, 11, 7, 15, 7]
    return _write_misaligned_store(tmp_path, shape, ("f8", "f4"), seed=38, name="mis_i")


def test_misaligned_columns_are_actually_misaligned(misaligned_uniform_store):
    path, _, _ = misaligned_uniform_store
    dataset = colstore.open(path)
    offset = int(dataset._record_starts_bytes[0]) + int(dataset._column_prefix_bytes["f8"]) * int(
        dataset._n_rows_per_record[0]
    )
    assert offset % 8 != 0, "fixture no longer exercises misalignment"
    dataset.close()


@pytest.mark.parametrize(
    "store_fixture, uniform",
    [("misaligned_uniform_store", True), ("misaligned_irregular_store", False)],
)
def test_misaligned_store_reads_match_on_all_routes(request, store_fixture, uniform):
    path, full, total = request.getfixturevalue(store_fixture)
    value_columns = [name for name in full if name != "pad"]
    rng = np.random.default_rng(1)
    with opened(path) as dataset:
        # Route discrimination: the two shapes must keep their own routes.
        assert (dataset._uniform_record_layout() is not None) == uniform
        # Unsorted fancy, single column (uniform kernel vs generic fused).
        indices = rng.integers(0, total, size=300).astype(np.int64)
        for name in full:
            assert np.array_equal(dataset[indices, name].array(), full[name][indices]), name
        # Multi-column unsorted fancy (uniform bins pair vs bins/rbase).
        table = dataset[indices, value_columns].dict()
        for name, values in table.items():
            assert np.array_equal(values, full[name][indices]), name
        # Sorted fancy (walk kernel) and contiguous slice.
        sorted_indices = np.sort(rng.integers(0, total, size=200).astype(np.int64))
        assert np.array_equal(dataset[sorted_indices, "f8"].array(), full["f8"][sorted_indices])
        assert np.array_equal(dataset[5:60, "f8"].array(), full["f8"][5:60])
        # Strided slices, both directions.
        for step in (2, 3, -1, -5):
            for name in value_columns:
                assert np.array_equal(dataset[::step, name].array(), full[name][::step]), (
                    name,
                    step,
                )
        # Boolean mask (mask kernel on dense masks).
        mask = rng.random(total) < 0.5
        assert np.array_equal(dataset[mask, "f8"].array(), full["f8"][mask])


def test_misaligned_single_record_store(tmp_path):
    rng = np.random.default_rng(5)
    full = {"pad": rng.integers(0, 9, 1001).astype(np.int8), "f8": rng.standard_normal(1001)}
    path = tmp_path / "sr.cstore"
    colstore.store(full, path, show_progress=False)
    dataset = colstore.open(path)
    indices = rng.integers(0, 1001, size=257).astype(np.int64)
    assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
    dataset.close()


# ---- Single-record stores and thread caps (cross-cutting) ---------------------


def test_single_record_store_never_routes_to_multirecord_kernels(tmp_path, monkeypatch):
    # Single-record stores keep their existing paths (plain gather for fancy
    # reads -- the backend contract -- numpy strided copy for slices, and
    # flatnonzero lowering for masks); no multi-record kernel may run.
    rng = np.random.default_rng(11)
    full = {"a": rng.standard_normal(10_000), "b": rng.standard_normal(10_000)}
    path = tmp_path / "single.cstore"
    colstore.store(full, path, show_progress=False)
    calls = kernel_spy(
        monkeypatch,
        [
            "gather_multirecord_sorted",
            "gather_multirecord_bins",
            "gather_multirecord_withbins",
            "gather_multirecord_withbins_rbase",
            "gather_multirecord_uniform",
            "gather_multirecord_uniform_bins",
            "gather_multirecord_uniform_withbins",
            "gather_segment_mask",
            "gather_multirecord_strided",
        ],
    )
    dataset = colstore.open(path)
    indices = rng.integers(0, 10_000, size=3_000).astype(np.int64)
    table = dataset[indices, ["a", "b"]].dict()
    assert np.array_equal(table["a"], full["a"][indices])
    mask = rng.random(10_000) < 0.6
    assert np.array_equal(dataset[mask, "a"].array(), full["a"][mask])
    for step in (2, -3):
        assert np.array_equal(dataset[::step, "a"].array(), full["a"][::step])
    assert calls == []
    dataset.close()


@pytest.mark.parametrize("cap", [1, 8])
def test_reads_respect_thread_cap_config(bin_reuse_store, cap):
    path, full = bin_reuse_store
    original = config_mod.get_gather_thread_cap()
    try:
        config_mod.set_gather_thread_cap(cap)
        dataset = colstore.open(path)
        indices = np.random.default_rng(4).integers(0, 40_000, size=8_000).astype(np.int64)
        table = dataset[indices, ["f8", "f4", "i4"]].dict()
        for name in ("f8", "f4", "i4"):
            assert np.array_equal(table[name], full[name][indices])
        sorted_indices = np.sort(indices)
        assert np.array_equal(dataset[sorted_indices, "f8"].array(), full["f8"][sorted_indices])
        dataset.close()
    finally:
        config_mod.set_gather_thread_cap(original)
