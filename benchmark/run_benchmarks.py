"""Comprehensive colstore benchmark: one run, one JSON summary for plotting.

Exercises the package's headline read/write paths as labelled scenarios and
emits a flat list of measurement records (see ``_common.Result``) plus a
machine fingerprint. Each scenario runs a correctness gate before timing and,
where it replaced a slower path, times that baseline too so the JSON carries
an honest speedup. Baselines are forced through the documented seams (numpy
backend, uniform-detection off, mask-density gate high) -- the same seams the
focused ``check_*.py`` scripts use.

This is the *showcase / plotting* benchmark; ``perf_suite.py`` remains the
regression-comparison tool (``--compare`` against a saved baseline). The two
share the fingerprint shape, and both build on ``_common``.

    PYTHONPATH=src python benchmark/run_benchmarks.py --json summary.json
    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmark/run_benchmarks.py \\
        --json summary.json --repeat 20

Plot ideas the JSON supports (no plotting code here yet): speedup vs record
count (fancy/sorted), throughput vs operation (write/read), mask-native
vs lowered across density, zero-copy vs copy.
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import _common as C
import numpy as np
from _common import Result, check_equal, time_stats

import colstore
from colstore import config, testing
from colstore import reader as reader_mod

_WARMUP = 1  # warmup passes discarded before timing; set from --warmup in main


@contextlib.contextmanager
def _temp_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    """Temporarily set ``obj.name = value`` (a benchmark seam), then restore."""
    sentinel = object()
    original = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if original is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, original)


def _measure(
    results: list[Result],
    scenario: str,
    variant: str,
    params: dict[str, Any],
    thunk: Callable[[], Any],
    *,
    repeat: int,
    rows: int | None = None,
) -> Result:
    res = Result.from_stats(
        scenario, variant, params, time_stats(thunk, repeat=repeat, warmup=_WARMUP), rows=rows
    )
    results.append(res)
    return res


# ---------------------------------------------------------------------------
# Scenarios. Each appends Result records; correctness gates raise on mismatch.
# ---------------------------------------------------------------------------


def scenario_write(results: list[Result], rng, total: int, *, repeat: int, gate: bool, bench: bool):
    cols = testing.make_columns(
        total, 4, dtype=("f8", "f4", "i4", "i2"), names=("f8", "f4", "i4", "i2"), rng=rng
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.cstore"
        if gate:
            colstore.store(cols, path, mode="recreate", show_progress=False).close()
            with colstore.open(path) as ds:
                check_equal(ds[:, "f8"].array(), cols["f8"], "write/readback")
        if not bench:
            return

        def write_once() -> None:
            colstore.store(cols, path, mode="recreate", show_progress=False).close()

        params = {"rows": total, "columns": len(cols)}
        _measure(
            results, "write_throughput", "store", params, write_once, repeat=repeat, rows=total
        )


def scenario_single_record_fancy(
    results: list[Result], rng, total: int, k: int, *, repeat: int, gate: bool, bench: bool
):
    cols = testing.make_columns(total, 1, dtype="f8", names=("f8",), rng=rng)
    indices = rng.integers(0, total, size=k).astype(np.int64)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s.cstore"
        colstore.store(cols, path, mode="recreate", show_progress=False).close()
        cpp = colstore.open(path)
        numpy_ds = colstore.open(path, backend="numpy")
        try:
            if gate:
                check_equal(cpp[indices, "f8"].array(), cols["f8"][indices], "single/cpp")
                check_equal(numpy_ds[indices, "f8"].array(), cols["f8"][indices], "single/numpy")
            if not bench:
                return
            params = {"rows": total, "k": k, "dtype": "<f8"}
            slow = _measure(
                results,
                "single_record_fancy",
                "numpy_backend",
                params,
                lambda: numpy_ds[indices, "f8"].array(),
                repeat=repeat,
                rows=k,
            )
            fast = _measure(
                results,
                "single_record_fancy",
                "cpp",
                params,
                lambda: cpp[indices, "f8"].array(),
                repeat=repeat,
                rows=k,
            )
            C.set_speedup(fast, slow)
        finally:
            cpp.close()
            numpy_ds.close()


def scenario_multirecord_fancy(
    results: list[Result], rng, total: int, k: int, n_records_list, *, repeat: int, gate, bench
):
    full = testing.make_columns(total, 1, dtype="f8", rng=rng)["c0"]
    indices = rng.integers(0, total, size=k).astype(np.int64)
    sorted_idx = np.sort(indices)
    for n_records in n_records_list:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.cstore"
            testing.write_columns(
                path, {"f8": full}, records=testing.uniform_record_rows(total, n_records)
            ).close()
            ds = colstore.open(path)
            try:
                if gate:
                    check_equal(ds[indices, "f8"].array(), full[indices], "mr/unsorted")
                    check_equal(ds[sorted_idx, "f8"].array(), full[sorted_idx], "mr/sorted")
                if not bench:
                    continue
                params = {"rows": total, "k": k, "n_records": n_records, "dtype": "<f8"}
                unsorted = _measure(
                    results,
                    "multirecord_fancy",
                    "unsorted",
                    params,
                    lambda ds=ds: ds[indices, "f8"].array(),
                    repeat=repeat,
                    rows=k,
                )
                srt = _measure(
                    results,
                    "multirecord_fancy",
                    "sorted",
                    params,
                    lambda ds=ds: ds[sorted_idx, "f8"].array(),
                    repeat=repeat,
                    rows=k,
                )
                C.set_speedup(srt, unsorted)  # sorted walk vs unsorted fused
            finally:
                ds.close()


def scenario_uniform_vs_generic(
    results: list[Result], rng, total: int, k: int, n_records: int, *, repeat, gate, bench
):
    cols = testing.make_columns(
        total, 4, dtype=("f8", "f4", "i4", "i2"), names=("f8", "f4", "i4", "i2"), rng=rng
    )
    indices = rng.integers(0, total, size=k).astype(np.int64)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "u.cstore"
        testing.write_columns(
            path, cols, records=[total // n_records] * n_records
        ).close()  # exactly uniform
        ds = colstore.open(path)
        try:
            if gate:
                check_equal(ds[indices, "f8"].array(), cols["f8"][indices], "uniform")
                with _temp_attr(
                    reader_mod.ColStoreReader, "_detect_uniform_record_layout", lambda self: None
                ):
                    check_equal(ds[indices, "f8"].array(), cols["f8"][indices], "uniform/generic")
            if not bench:
                return
            params = {"rows": total, "k": k, "n_records": n_records}
            with _temp_attr(
                reader_mod.ColStoreReader, "_detect_uniform_record_layout", lambda self: None
            ):
                generic = _measure(
                    results,
                    "uniform_record",
                    "generic",
                    params,
                    lambda ds=ds: ds[indices, "f8"].array(),
                    repeat=repeat,
                    rows=k,
                )
            uniform = _measure(
                results,
                "uniform_record",
                "uniform_kernel",
                params,
                lambda ds=ds: ds[indices, "f8"].array(),
                repeat=repeat,
                rows=k,
            )
            C.set_speedup(uniform, generic)
        finally:
            ds.close()


def scenario_multicolumn_bin_reuse(
    results: list[Result], rng, total: int, k: int, n_records: int, *, repeat, gate, bench
):
    cols = testing.make_columns(
        total, 4, dtype=("f8", "f4", "i4", "i2"), names=("f8", "f4", "i4", "i2"), rng=rng
    )
    names = list(cols)
    indices = rng.integers(0, total, size=k).astype(np.int64)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mc.cstore"
        # Irregular sizes so the generic bin-reuse route is exercised.
        rows = testing.uniform_record_rows(total, n_records)
        rows[0] += 1
        rows[-1] -= 1
        testing.write_columns(path, cols, records=rows).close()
        ds = colstore.open(path)
        try:
            if gate:
                table = ds[indices, names].dict()
                for name in names:
                    check_equal(table[name], cols[name][indices], f"binreuse/{name}")
            if not bench:
                return
            params = {"rows": total, "k": k, "n_records": n_records, "columns": len(names)}
            per_col = _measure(
                results,
                "multicolumn",
                "per_column",
                params,
                lambda: {nm: ds[indices, nm].array() for nm in names},
                repeat=repeat,
                rows=k,
            )
            reuse = _measure(
                results,
                "multicolumn",
                "bin_reuse",
                params,
                lambda: ds[indices, names].dict(),
                repeat=repeat,
                rows=k,
            )
            C.set_speedup(reuse, per_col)
        finally:
            ds.close()
def scenario_strided(
    results: list[Result], rng, total: int, n_records: int, step: int, *, repeat, gate, bench
):
    full = testing.make_columns(total, 1, dtype="f8", rng=rng)["c0"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "st.cstore"
        testing.write_columns(
            path, {"f8": full}, records=testing.uniform_record_rows(total, n_records)
        ).close()
        ds = colstore.open(path)
        try:
            arange_idx = np.arange(*slice(None, None, step).indices(total), dtype=np.int64)
            if gate:
                check_equal(ds[::step, "f8"].array(), full[::step], "strided/kernel")
                check_equal(ds[arange_idx, "f8"].array(), full[::step], "strided/fancy")
            if not bench:
                return
            n_out = len(range(*slice(None, None, step).indices(total)))
            params = {"rows": total, "n_records": n_records, "step": step}
            fancy = _measure(
                results,
                "strided_slice",
                "arange_fancy",
                params,
                lambda: ds[arange_idx, "f8"].array(),
                repeat=repeat,
                rows=n_out,
            )
            kern = _measure(
                results,
                "strided_slice",
                "strided_kernel",
                params,
                lambda: ds[::step, "f8"].array(),
                repeat=repeat,
                rows=n_out,
            )
            C.set_speedup(kern, fancy)
        finally:
            ds.close()


def scenario_mask_native(
    results: list[Result], rng, total: int, n_records: int, densities, *, repeat, gate, bench
):
    full = testing.make_columns(total, 1, dtype="f8", rng=rng)["c0"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mk.cstore"
        testing.write_columns(
            path, {"f8": full}, records=testing.uniform_record_rows(total, n_records)
        ).close()
        ds = colstore.open(path)
        try:
            for density in densities:
                mask = rng.random(total) < density
                if gate:
                    with _temp_attr(config, "_mask_density_gate", 0.0):
                        check_equal(ds[mask, "f8"].array(), full[mask], f"mask/native@{density}")
                    with _temp_attr(config, "_mask_density_gate", 2.0):
                        check_equal(ds[mask, "f8"].array(), full[mask], f"mask/lowered@{density}")
                if not bench:
                    continue
                n_sel = int(mask.sum())
                params = {"rows": total, "n_records": n_records, "density": density}
                with _temp_attr(config, "_mask_density_gate", 2.0):
                    lowered = _measure(
                        results,
                        "mask_native",
                        "lowered",
                        params,
                        lambda mask=mask: ds[mask, "f8"].array(),
                        repeat=repeat,
                        rows=n_sel,
                    )
                with _temp_attr(config, "_mask_density_gate", 0.0):
                    native = _measure(
                        results,
                        "mask_native",
                        "native",
                        params,
                        lambda mask=mask: ds[mask, "f8"].array(),
                        repeat=repeat,
                        rows=n_sel,
                    )
                C.set_speedup(native, lowered)
        finally:
            ds.close()


def scenario_range_copy(
    results: list[Result], rng, total: int, n_records: int, *, repeat, gate, bench
):
    full = testing.make_columns(total, 1, dtype="f8", rng=rng)["c0"]
    lo, hi = total // 4, total // 4 + total // 2
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rc.cstore"
        testing.write_columns(
            path, {"f8": full}, records=testing.uniform_record_rows(total, n_records)
        ).close()
        ds = colstore.open(path)
        try:
            arange_idx = np.arange(lo, hi, dtype=np.int64)
            if gate:
                check_equal(ds[lo:hi, "f8"].array(), full[lo:hi], "range/copy")
                check_equal(ds[arange_idx, "f8"].array(), full[lo:hi], "range/fancy")
            if not bench:
                return
            params = {"rows": total, "n_records": n_records, "span": hi - lo}
            fancy = _measure(
                results,
                "range_copy",
                "arange_fancy",
                params,
                lambda: ds[arange_idx, "f8"].array(),
                repeat=repeat,
                rows=hi - lo,
            )
            copy = _measure(
                results,
                "range_copy",
                "range_kernel",
                params,
                lambda: ds[lo:hi, "f8"].array(),
                repeat=repeat,
                rows=hi - lo,
            )
            C.set_speedup(copy, fancy)
        finally:
            ds.close()


def scenario_zero_copy(results: list[Result], rng, total: int, *, repeat, gate, bench):
    cols = testing.make_columns(total, 1, dtype="f8", names=("f8",), rng=rng)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "z.cstore"
        colstore.store(cols, path, mode="recreate", show_progress=False).close()  # single record
        ds = colstore.open(path)
        try:
            if gate:
                check_equal(ds["f8"].array(copy=False), cols["f8"], "zerocopy/view")
            if not bench:
                return
            params = {"rows": total}
            copy = _measure(
                results,
                "zero_copy",
                "copy",
                params,
                lambda: ds["f8"].array(copy=True),
                repeat=repeat,
                rows=total,
            )
            view = _measure(
                results,
                "zero_copy",
                "view",
                params,
                lambda: ds["f8"].array(copy=False),
                repeat=repeat,
                rows=total,
            )
            C.set_speedup(view, copy)
        finally:
            ds.close()


def scenario_frame(results: list[Result], rng, total: int, *, repeat, gate, bench):
    import pandas as pd

    cols = testing.make_columns(
        total, 4, dtype=("f8", "f4", "i4", "i2"), names=("f8", "f4", "i4", "i2"), rng=rng
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "f.cstore"
        colstore.store(cols, path, mode="recreate", show_progress=False).close()
        ds = colstore.open(path)
        try:
            if gate:
                frame = ds[:, list(cols)].frame()
                check_equal(frame["f8"].to_numpy(), cols["f8"], "frame/values")
            if not bench:
                return
            params = {"rows": total, "columns": len(cols)}
            baseline = _measure(
                results,
                "frame_construction",
                "pandas_default",
                params,
                lambda: pd.DataFrame({k: ds[:, k].array() for k in cols}),
                repeat=repeat,
                rows=total,
            )
            fast = _measure(
                results,
                "frame_construction",
                "no_consolidate",
                params,
                lambda: ds[:, list(cols)].frame(),
                repeat=repeat,
                rows=total,
            )
            C.set_speedup(fast, baseline)
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _print_table(results: list[Result]) -> None:
    print(f"\n{'scenario':<20} {'variant':<16} {'median_ms':>10} {'rows/s':>13} {'speedup':>8}")
    print("-" * 70)
    for r in results:
        tput = f"{r.throughput_rows_per_s:,.0f}" if r.throughput_rows_per_s else ""
        spd = f"{r.speedup:.2f}x" if r.speedup else ""
        print(f"{r.scenario:<20} {r.variant:<16} {r.median_ms:>10.3f} {tput:>13} {spd:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    C.add_common_args(parser, repeat=10, scale=True, json=True)
    args = parser.parse_args()
    global _WARMUP
    _WARMUP = args.warmup
    scale = args.scale
    repeat = args.repeat
    gate = not args.skip_correctness
    bench = not args.skip_bench
    rng = np.random.default_rng(20240607)

    def n(x: int) -> int:
        return max(1, int(x * scale))

    results: list[Result] = []
    scenario_write(results, rng, n(2_000_000), repeat=repeat, gate=gate, bench=bench)
    scenario_single_record_fancy(
        results, rng, n(2_000_000), n(500_000), repeat=repeat, gate=gate, bench=bench
    )
    scenario_multirecord_fancy(
        results, rng, n(2_000_000), n(500_000), [100, 1000], repeat=repeat, gate=gate, bench=bench
    )
    scenario_uniform_vs_generic(
        results, rng, n(2_000_000), n(500_000), 1000, repeat=repeat, gate=gate, bench=bench
    )
    scenario_multicolumn_bin_reuse(
        results, rng, n(2_000_000), n(500_000), 1000, repeat=repeat, gate=gate, bench=bench
    )
    scenario_strided(results, rng, n(2_000_000), 1000, 8, repeat=repeat, gate=gate, bench=bench)
    scenario_mask_native(
        results, rng, n(2_000_000), 1000, [0.5, 0.1, 0.02], repeat=repeat, gate=gate, bench=bench
    )
    scenario_range_copy(results, rng, n(2_000_000), 1000, repeat=repeat, gate=gate, bench=bench)
    scenario_zero_copy(results, rng, n(2_000_000), repeat=repeat, gate=gate, bench=bench)
    scenario_frame(results, rng, n(1_000_000), repeat=repeat, gate=gate, bench=bench)

    if bench:
        _print_table(results)
    if args.json is not None:
        C.write_summary(args.json, results, meta={"scale": scale, "repeat": repeat})
        print(f"\nwrote {len(results)} records to {args.json}")


if __name__ == "__main__":
    main()
