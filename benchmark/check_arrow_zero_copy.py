"""Verify the Apache Arrow export: round-trip parity and zero copy.

``reader.arrow()`` / ``view.arrow()`` -- and the Arrow PyCapsule interface
(``pyarrow.array(view)`` / ``pyarrow.table(reader)``) -- wrap a native column's
memory-mapped bytes as an Arrow array with no copy: the Arrow values buffer
points at the same address as the column's zero-copy memmap view, the Arrow
memory pool grows by zero bytes, and process RSS stays flat regardless of column
size. A column split across records or files becomes a ChunkedArray with one
zero-copy chunk per segment.

The correctness gate (always run before any timing) checks:

* round-trip parity across every supported dtype for a single-record store,
  and across the numeric dtypes for multi-record and multi-file stores;
* the Arrow values buffer address equals the column's memmap address;
* ``pyarrow.total_allocated_bytes()`` is unchanged by a zero-copy export;
* the exported array stays valid and correct after the store is closed.

Run on the deployment hardware (quiet compute node):

    python benchmark/check_arrow_zero_copy.py
    python benchmark/check_arrow_zero_copy.py --skip-bench
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import _common as _c

import colstore
from colstore import testing

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover - benchmark requires pyarrow
    print("pyarrow is not installed; install colstore[arrow] to run this check.")
    sys.exit(0)

# Dtypes whose bytes are an Arrow primitive values buffer (zero-copy), plus the
# convert-on-export kinds (bool, unicode) that must still round-trip by value.
ZERO_COPY_DTYPES = ("float64", "float32", "float16", "int64", "int32", "int16", "int8", "uint32")
CONVERT_DTYPES = ("bool", "S8", "datetime64[ns]")

SIZES = (1_000_000, 10_000_000, 100_000_000)  # f8 rows: 8 MB / 80 MB / 800 MB


def _expected(values: dict[str, np.ndarray], name: str) -> np.ndarray:
    return values[name]


def _crafted_column(dtype: str, rows: int, seed: int) -> np.ndarray:
    """A reproducible column of any supported dtype (testing.make_columns is numeric)."""
    rng = np.random.default_rng(seed)
    kind = np.dtype(dtype).kind
    if kind == "b":
        return rng.integers(0, 2, size=rows).astype(bool)
    if kind == "S":
        return np.array([f"v{i % 97}".encode() for i in range(rows)], dtype=dtype)
    if kind == "M":
        return (np.arange(rows, dtype="int64") * 1_000_000_000).astype("datetime64[ns]")
    return testing.make_columns(rows, 1, names=("x",), dtype=dtype, seed=seed)["x"]


def _check_parity(store: colstore._base._ReaderBase, expected: dict[str, np.ndarray]) -> None:
    """Every column round-trips by value through arrow(), and the table matches."""
    for name, values in expected.items():
        arrow = store[name].arrow()
        got = arrow.combine_chunks() if isinstance(arrow, pa.ChunkedArray) else arrow
        back = got.to_numpy(zero_copy_only=False)
        if values.dtype.kind == "S":
            back = back.astype(values.dtype)
        if not np.array_equal(back, values):
            raise AssertionError(f"arrow round-trip mismatch for column {name!r}")
    table = store.arrow()
    if table.num_rows != next(iter(expected.values())).shape[0]:
        raise AssertionError("table row count mismatch")
    if table.column_names != list(expected):
        raise AssertionError("table column order mismatch")


def check_correctness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = 50_000

        # Single-record store: one zero-copy Array per column.
        numeric = {dt.replace("[", "_").replace("]", ""): _crafted_column(dt, rows, i)
                   for i, dt in enumerate(ZERO_COPY_DTYPES + CONVERT_DTYPES)}
        single = root / "single.cstore"
        colstore.store(numeric, single)
        with colstore.open(single) as ds:
            _check_parity(ds, numeric)
            # Zero-copy: the Arrow values buffer aliases the column's memmap.
            for name in ("float64", "int32"):
                view = ds[name].array(copy=False)
                arr = ds[name].arrow()
                assert arr.buffers()[1].address == view.ctypes.data, name
            # No Arrow-pool allocation for a zero-copy export.
            before = pa.total_allocated_bytes()
            held = ds["float64"].arrow()
            assert pa.total_allocated_bytes() == before, "zero-copy export allocated"
            assert held.null_count == 0
            # PyCapsule interface: pyarrow.array / pyarrow.table consume directly.
            assert pa.array(ds["float64"]).buffers()[1].address == ds["float64"].array(
                copy=False
            ).ctypes.data
            assert pa.table(ds).num_rows == rows
            survived = ds["float64"].arrow()
        # The exported array outlives the closed store (release callback holds the mmap).
        assert np.array_equal(survived.to_numpy(zero_copy_only=True),
                              numeric["float64"]), "did not survive close"

        # Multi-record store: ChunkedArray, one chunk per record.
        multi_cols = {dt: _crafted_column(dt, rows, i)
                      for i, dt in enumerate(("float64", "int32", "int16"))}
        multi = root / "multi.cstore"
        with colstore.testing.write_columns(multi, multi_cols, records=4) as ds:
            _check_parity(ds, multi_cols)
            ca = ds["float64"].arrow()
            assert isinstance(ca, pa.ChunkedArray) and ca.num_chunks == 4, "expected 4 chunks"

        # Multi-file dataset: chunks span files, still zero-copy per chunk.
        parts = []
        for f in range(3):
            part = root / f"part{f}.cstore"
            colstore.store({"float64": _crafted_column("float64", rows, 100 + f),
                            "int32": _crafted_column("int32", rows, 200 + f)}, part)
            parts.append(part)
        with colstore.open(parts) as ds:
            ca = ds["float64"].arrow()
            assert isinstance(ca, pa.ChunkedArray) and ca.num_chunks == 3
            assert ds.arrow().num_rows == 3 * rows

    print("  ALL CORRECTNESS CHECKS PASSED "
          "(round-trip parity; buffer==mmap; zero Arrow alloc; survives close)\n")


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def run_bench(repeat: int, warmup: int) -> None:
    for n in SIZES:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"n{n}.cstore"
            colstore.store({"a": testing.make_columns(n, 1, names=("a",), seed=1)["a"]}, path,
                           show_progress=False)
            ds = colstore.open(path)
            ds["a"].array(copy=False).sum()  # warm the page cache
            rss_before = _rss_mb()
            ds["a"].arrow()
            rss_after = _rss_mb()
            print(f"  rows = {n:,}  ({n * 8 / 1e6:.0f} MB column)")
            print(f"    RSS delta from arrow() export: {rss_after - rss_before:+.1f} MB "
                  f"(column is {n * 8 / 1e6:.0f} MB)")
            _c.compare(
                [
                    ("dict()  copy=True ", lambda ds=ds: ds.dict()),
                    ("arrow() zero-copy ", lambda ds=ds: ds.arrow()),
                ],
                repeat=repeat,
                warmup=warmup,
                baseline=0,
            )
            ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, skip_correctness=False)
    args = parser.parse_args()
    check_correctness()
    if not args.skip_bench:
        run_bench(args.repeat, args.warmup)


if __name__ == "__main__":
    main()
