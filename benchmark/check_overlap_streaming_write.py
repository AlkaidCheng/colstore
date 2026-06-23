"""A/B the I/O-compute-overlapped streaming write against the inline (serial) one.

The streaming write (``write_dataset_streaming`` -> ``_fill_streaming_pwrite``)
reads+transforms a batch, then ``pwrite``s it, then moves to the next batch. The
compute is memory-bandwidth-bound and the write is filesystem-bound, so they can
run at once: the change writes each batch on a single background thread while the
main thread prepares the next, hiding the smaller phase under the larger.
``format._STREAMING_OVERLAP_WRITE`` toggles it for this A/B.

Three shapes spanning the compute/write balance: a light transform (compute and
write comparable, so overlap approaches the filesystem write ceiling), a deep
transform (compute-bound, so overlap hides the write), and a filtered write (a
sorted fancy gather comparable to the write, the largest overlap headroom).

The correctness gate asserts the overlapped output matches the direct transform
and the inline route byte-for-byte before any timing. Run on the deployment node
with ``--tmpdir`` (or ``TMPDIR``) on the parallel filesystem; local numbers are
indicative only.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import _common as _c
import numpy as np

import colstore
import colstore.format as _fmt
from colstore import col, config, testing
from colstore.format import write_dataset_streaming

_FILTER_FRACTION = 0.5
_FILTER_SEED = 12345
_CORRECTNESS_ROWS = 200_000
_CORRECTNESS_RECORDS = 50


@dataclass(frozen=True)
class Shape:
    key: str
    label: str
    build: Callable[[colstore.ColStoreReader, int], colstore.ColStoreFrame]
    expected: Callable[[dict[str, np.ndarray], int], dict[str, np.ndarray]]
    out_rows: Callable[[int], int]


def _filter_index(n_rows: int) -> np.ndarray:
    rng = np.random.default_rng(_FILTER_SEED)
    keep = max(1, int(n_rows * _FILTER_FRACTION))
    return np.sort(rng.choice(n_rows, keep, replace=False)).astype(np.int64)


def _build_transform(store: colstore.ColStoreReader, n_rows: int) -> colstore.ColStoreFrame:
    cf = store.edit()
    for name in store.columns:
        cf[name] = col(name) + 2.0
    return cf


def _expected_transform(source: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    return {name: data + 2.0 for name, data in source.items()}


def _build_deep(store: colstore.ColStoreReader, n_rows: int) -> colstore.ColStoreFrame:
    cf = store.edit()
    for name in store.columns:
        c = col(name)
        cf[name] = np.sqrt(np.abs(c)) * c + c
    return cf


def _expected_deep(source: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    return {name: np.sqrt(np.abs(data)) * data + data for name, data in source.items()}


def _build_filter(store: colstore.ColStoreReader, n_rows: int) -> colstore.ColStoreFrame:
    return store[_filter_index(n_rows)].edit()


def _expected_filter(source: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    idx = _filter_index(n_rows)
    return {name: data[idx] for name, data in source.items()}


_SHAPES = (
    Shape(
        "transform",
        "transform (+2.0 per column)",
        _build_transform,
        _expected_transform,
        lambda n: n,
    ),
    Shape("deep", "deep transform (sqrt*c + c)", _build_deep, _expected_deep, lambda n: n),
    Shape(
        "filter",
        "filter 50% (gather + write)",
        _build_filter,
        _expected_filter,
        lambda n: max(1, n // 2),
    ),
)


def _build_source(
    path: Path, rows: int, cols: int, records: int, dtype: str
) -> dict[str, np.ndarray]:
    """(Re)create the source store at ``path`` and return its in-memory columns."""
    path.unlink(missing_ok=True)  # --tmpdir persists between runs, unlike a TemporaryDirectory
    source = testing.make_columns(rows, cols, dtype=dtype, seed=0)
    testing.write_columns(path, source, records=min(records, rows)).close()
    return source


def _write_frame(frame: colstore.ColStoreFrame, out: Path) -> None:
    """The streaming fill itself (the call ``ColStoreFrame.write`` makes, minus the
    reader it opens on the result -- excluded so the timing reflects the fill)."""
    selection = frame._resolve_index_selection()
    if selection is None:
        write_dataset_streaming(frame._columns, frame._n_rows, out)
    else:
        write_dataset_streaming(frame._columns, len(selection), out, rows=selection)


def _write_with(frame: colstore.ColStoreFrame, out: Path, *, overlap: bool) -> None:
    _fmt._STREAMING_OVERLAP_WRITE = overlap
    out.unlink(missing_ok=True)
    _write_frame(frame, out)


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    rows = min(_CORRECTNESS_ROWS, _c.scaled_rows(args.rows, args))
    records = min(args.records, _CORRECTNESS_RECORDS, rows)
    source = _build_source(directory / "correctness.cstore", rows, args.cols, records, args.dtype)
    store = colstore.open(directory / "correctness.cstore")
    try:
        for shape in _SHAPES:
            expected = shape.expected(source, rows)
            inline_out = directory / f"correctness_{shape.key}_inline.cstore"
            overlap_out = directory / f"correctness_{shape.key}_overlap.cstore"
            _write_with(shape.build(store, rows), inline_out, overlap=False)
            _write_with(shape.build(store, rows), overlap_out, overlap=True)
            reader_i = colstore.open(inline_out)
            reader_o = colstore.open(overlap_out)
            try:
                got_i, got_o = reader_i.dict(), reader_o.dict()
                for name in got_o:
                    _c.check_equal(got_o[name], expected[name], f"{shape.key}:{name} vs reference")
                    _c.check_equal(
                        got_o[name], got_i[name], f"{shape.key}:{name} overlap vs inline"
                    )
            finally:
                reader_i.close()
                reader_o.close()
                inline_out.unlink(missing_ok=True)
                overlap_out.unlink(missing_ok=True)
    finally:
        store.close()
    print("  CORRECTNESS OK (overlap == inline == direct transform for every shape)\n")


def run_bench(directory: Path, args: argparse.Namespace) -> None:
    rows = _c.scaled_rows(args.rows, args)
    src_path = directory / f"src_r{rows}_c{args.cols}.cstore"
    _build_source(src_path, rows, args.cols, args.records, args.dtype)
    store = colstore.open(src_path)
    try:
        print("Environment:")
        print(f"  rows={rows:,}  cols={args.cols}  records={args.records}  dtype={args.dtype}")
        print(
            f"  memory budget={config.get_default_memory_budget() / 2**20:.0f} MiB"
            f"  gather thread cap={config.get_gather_thread_cap()}"
        )
        print(f"  repeat={args.repeat}  warmup={args.warmup}  store dir={directory}\n")
        out = directory / "out.cstore"
        for shape in _SHAPES:
            frame = shape.build(store, rows)
            n_out = shape.out_rows(rows)
            print(f"=== {shape.label}  ({n_out:,} rows) ===")
            _c.compare(
                [
                    ("inline  ", lambda f=frame: _write_with(f, out, overlap=False)),
                    ("overlap ", lambda f=frame: _write_with(f, out, overlap=True)),
                ],
                repeat=args.repeat,
                warmup=args.warmup,
                baseline=0,
                setups=[lambda: out.unlink(missing_ok=True)] * 2,
                throughput_rows=n_out,
            )
            print()
        out.unlink(missing_ok=True)
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=5,
        warmup=2,
        rows=10_000_000,
        cols=12,
        dtype="float64",
        threads=True,
        scale=True,
        tmpdir=True,
    )
    parser.add_argument("--records", type=int, default=1000, help="records in the source store")
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    previous_override = _fmt._STREAMING_FILL_OVERRIDE
    previous_overlap = _fmt._STREAMING_OVERLAP_WRITE
    _fmt._STREAMING_FILL_OVERRIDE = "pwrite"
    try:
        if args.tmpdir is not None:
            directory = Path(args.tmpdir)
            directory.mkdir(parents=True, exist_ok=True)
            if not args.skip_correctness:
                check_correctness(directory, args)
            if not args.skip_bench:
                run_bench(directory, args)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                if not args.skip_correctness:
                    check_correctness(directory, args)
                if not args.skip_bench:
                    run_bench(directory, args)
    finally:
        _fmt._STREAMING_FILL_OVERRIDE = previous_override
        _fmt._STREAMING_OVERLAP_WRITE = previous_overlap


if __name__ == "__main__":
    main()
