"""Prototype benchmark: file-level parallelism for ``colstore.convert``.

Measures whether converting N independent files concurrently -- a ``ThreadPoolExecutor``
or ``ProcessPoolExecutor`` over ``colstore.convert``, one file per task -- speeds up a
batch convert, and by how much. This is the measurement behind a possible ``max_workers``
option; nothing in the library is changed, the parallel wrappers live here so the win can
be validated on the target node before any implementation.

For each format it interleaves (via the suite's A/B harness) a sequential loop against
thread and process pools at the requested worker counts, and reports per variant: wall,
cpu, cpu/wall (``>1`` proves real parallelism), peak threads, and speedup over the
sequential loop. ``--cold`` evicts the input page cache before each timed run.

What the earlier feasibility study found (10-core laptop, warm cache), to check on the node:

* ``convert`` is a two-stage pipeline (read/decode -> write). Parquet decode releases the
  GIL and is already multi-core (pyarrow); the ``.cstore`` write is I/O-bound. HDF5 read
  serializes on h5py's process-global lock, so **threads do not help hdf5 -- only processes
  do**. The string-coercion glue is pure-Python (GIL-bound), which also favors processes.
* Threads gave ~1.3x for parquet import, ~1.0x for hdf5. A warm/reused process pool gave
  ~1.5-1.7x (incl. hdf5) but a cold one-shot pool was net-negative at small file sizes
  (spawn + re-import ~ the whole job). Warm page cache understates cold/Lustre I/O, where
  GIL-releasing readers overlap I/O across workers -- expected to raise the win on Perlmutter.

Caveats: ``cpu/wall`` is measured in the parent only, so it is meaningful for the
sequential/thread variants but reads ~0 for process variants (child CPU is not counted) --
use ``wall`` / ``speedup`` for those. ``--mp-context spawn`` (default) is safe; ``fork`` is
faster on Linux but can deadlock a child that inherits pyarrow/OpenMP threads.

    PYTHONPATH=src python benchmark/check_convert_parallel.py \\
        --formats parquet,hdf5 --files 16 --rows 2000000 --workers 4,8,16 \\
        --repeat 5 --cold --json convert_parallel.json
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import _common as _c
import numpy as np

import colstore
from colstore import config

_ALL_FORMATS = ("parquet", "feather", "hdf5", "npz")
_CSTORE = ".cstore"


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


# ---- Worker-side helpers (module level so ProcessPoolExecutor can pickle them) ----


def _apply_cap(cap: int | None) -> None:
    """Cap each worker's internal parallelism so N workers x internal pool != cores**2."""
    if cap is None:
        return
    try:
        import pyarrow

        pyarrow.set_cpu_count(cap)
        pyarrow.set_io_thread_count(cap)
    except Exception:  # pragma: no cover - pyarrow optional
        pass
    config.set_gather_thread_cap(cap)


def _init_worker(cap: int | None) -> None:
    _apply_cap(cap)


def _convert_file(pair: tuple[str, str]) -> None:
    """Convert one file and close the reader (never ship an open reader back to the parent)."""
    src, dst = pair
    result = colstore.convert(src, dst, overwrite=True)
    close = getattr(result, "close", None)
    if callable(close):
        close()


# ---- Input builders --------------------------------------------------------


def _columns(rows: int, string_cols: int) -> dict[str, np.ndarray]:
    """A synthetic column mapping: numeric columns plus ``string_cols`` fixed-width strings."""
    cols: dict[str, np.ndarray] = {}
    for i in range(max(1, 6 - string_cols)):
        cols[f"n{i}"] = (np.arange(rows, dtype=np.float64) + i) * 1.5
    pattern = np.array(["aa", "bb", "ccc", "dddd"], dtype="U4")
    for j in range(string_cols):
        cols[f"s{j}"] = np.tile(pattern, rows // len(pattern) + 1)[:rows]
    return cols


def _write_foreign(
    fmt: str, path: Path, cols: dict[str, np.ndarray], compression: str | None
) -> None:
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table(cols), str(path), compression=compression or "none")
    elif fmt == "feather":
        import pyarrow as pa
        import pyarrow.feather as feather

        feather.write_feather(pa.table(cols), str(path))
    elif fmt == "hdf5":
        import h5py

        with h5py.File(path, "w", track_order=True) as handle:
            group = handle.create_group("data", track_order=True)
            for name, array in cols.items():
                if array.dtype.kind == "U":
                    group.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype())
                else:
                    group.create_dataset(name, data=array)
    elif fmt == "npz":
        np.savez(str(path), **cols)
    else:  # pragma: no cover
        raise ValueError(fmt)


def _extension(fmt: str) -> str:
    return {"parquet": ".parquet", "feather": ".feather", "hdf5": ".h5", "npz": ".npz"}[fmt]


def build_inputs(
    directory: Path,
    fmt: str,
    direction: str,
    files: int,
    rows: int,
    string_cols: int,
    compression: str | None,
) -> tuple[list[Path], list[Path], list[tuple[str, str]]]:
    """Create ``files`` inputs and return (inputs, outputs, (src, dst) pairs) for the direction."""
    cols = _columns(rows, string_cols)
    inputs: list[Path] = []
    outputs: list[Path] = []
    for index in range(files):
        if direction == "import":
            src = directory / f"in_{index:03d}{_extension(fmt)}"
            _write_foreign(fmt, src, cols, compression)
            dst = directory / f"out_{index:03d}{_CSTORE}"
        else:  # export
            src = directory / f"in_{index:03d}{_CSTORE}"
            colstore.store(cols, src, show_progress=False).close()
            dst = directory / f"out_{index:03d}{_extension(fmt)}"
        inputs.append(src)
        outputs.append(dst)
    pairs = [(str(s), str(d)) for s, d in zip(inputs, outputs, strict=True)]
    return inputs, outputs, pairs


# ---- Convert runners (the prototype under test) ----------------------------


def run_sequential(pairs: list[tuple[str, str]]) -> None:
    for pair in pairs:
        _convert_file(pair)


def run_threads(pairs: list[tuple[str, str]], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_convert_file, pairs))


def run_process(pairs: list[tuple[str, str]], workers: int, cap: int | None, mp: str) -> None:
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context(mp),
        initializer=_init_worker,
        initargs=(cap,),
    ) as pool:
        list(pool.map(_convert_file, pairs))


# ---- Correctness gate ------------------------------------------------------


def check_parallel_matches_sequential(
    pairs: list[tuple[str, str]], direction: str, cap: int | None, mp: str
) -> None:
    """Convert once each way to distinct outputs and assert byte-for-byte-equal columns."""
    if direction == "export":
        return  # foreign outputs differ per writer; the round-trip is covered by the test suite

    def tag(
        dst: str, mode: str
    ) -> str:  # keep the .cstore extension so convert infers the direction
        path = Path(dst)
        return str(path.with_name(f"{mode}_{path.name}"))

    seq = [(s, tag(d, "seq")) for s, d in pairs]
    thr = [(s, tag(d, "thr")) for s, d in pairs]
    proc = [(s, tag(d, "proc")) for s, d in pairs]
    run_sequential(seq)
    run_threads(thr, min(4, len(pairs)))
    run_process(proc, min(4, len(pairs)), cap, mp)
    for (_, a), (_, b), (_, c) in zip(seq, thr, proc, strict=True):
        readers = [colstore.open(a), colstore.open(b), colstore.open(c)]
        try:
            da, db, dc = (readers[0].dict(), readers[1].dict(), readers[2].dict())
            assert da.keys() == db.keys() == dc.keys(), "column set differs across modes"
            for key in da:
                _c.check_equal(db[key], da[key], f"thread vs sequential [{key}]")
                _c.check_equal(dc[key], da[key], f"process vs sequential [{key}]")
        finally:
            for reader in readers:
                reader.close()
    print("  correctness: parallel outputs match the sequential loop (columns + values).")


# ---- Main ------------------------------------------------------------------


def _bench_one(fmt: str, args: argparse.Namespace, results: list[_c.Result]) -> None:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        rows = _c.scaled_rows(args.rows, args)
        inputs, _outputs, pairs = build_inputs(
            directory, fmt, args.direction, args.files, rows, args.string_cols, args.compression
        )
        total_mb = sum(p.stat().st_size for p in inputs) / 1e6
        banner(
            f"{fmt.upper()} {args.direction}: {args.files} files x {rows:,} rows "
            f"(~{total_mb:.0f} MB in){' [cold]' if args.cold else ''}"
        )
        if not getattr(args, "skip_correctness", False):
            check_parallel_matches_sequential(
                pairs, args.direction, args.cap_internal, args.mp_context
            )
        if args.skip_bench:
            return

        specs: list[tuple[str, object]] = [("sequential", lambda: run_sequential(pairs))]
        for worker in args.workers:
            specs.append((f"thread-{worker}", lambda w=worker: run_threads(pairs, w)))
        for worker in args.workers:
            specs.append(
                (
                    f"process-{worker}",
                    lambda w=worker: run_process(pairs, w, args.cap_internal, args.mp_context),
                )
            )

        setups = [lambda: _c.drop_pagecache(inputs)] * len(specs) if args.cold else None
        profiles = _c.compare(
            specs, repeat=args.repeat, warmup=args.warmup, baseline=0, setups=setups
        )
        base = profiles[0].wall_ms
        for spec, prof in zip(specs, profiles, strict=True):
            result = _c.Result(
                scenario=f"convert_parallel:{fmt}:{args.direction}",
                variant=spec[0],
                params={
                    "files": args.files,
                    "rows": rows,
                    "workers": args.workers,
                    "cold": args.cold,
                    "cap_internal": args.cap_internal,
                    "string_cols": args.string_cols,
                    "mp_context": args.mp_context,
                },
                median_ms=prof.wall_ms,
                min_ms=prof.wall_ms,
                p95_ms=prof.wall_ms,
                repeat=args.repeat,
                rows=rows * args.files,
                speedup_vs="sequential",
                speedup=(base / prof.wall_ms) if prof.wall_ms else None,
            )
            results.append(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _c.add_common_args(parser, repeat=5, warmup=2, rows=2_000_000, scale=True, json=True)
    parser.add_argument(
        "--formats", default="parquet,hdf5", help="comma-separated: parquet,feather,hdf5,npz"
    )
    parser.add_argument("--direction", choices=("import", "export"), default="import")
    parser.add_argument(
        "--files", type=int, default=16, help="number of independent files to convert"
    )
    parser.add_argument("--workers", default="4,8", help="comma-separated pool sizes to sweep")
    parser.add_argument(
        "--string-cols", type=int, default=0, help="string columns (GIL-bound coercion)"
    )
    parser.add_argument(
        "--compression", default=None, help="parquet compression (e.g. zstd); default none"
    )
    parser.add_argument(
        "--cold", action="store_true", help="evict the input page cache before each timed run"
    )
    parser.add_argument(
        "--cap-internal", type=int, default=None, help="cap pyarrow/gather threads per worker"
    )
    parser.add_argument("--mp-context", choices=("spawn", "fork", "forkserver"), default="spawn")
    args = parser.parse_args()
    args.workers = [int(w) for w in str(args.workers).split(",") if w]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    if args.cold and not sys.platform.startswith("linux"):
        # drop_pagecache uses posix_fadvise via libc.so.6 -- Linux only (the target node).
        print("  note: --cold page-cache eviction needs Linux; running warm on this platform.")
        args.cold = False

    print("Environment:")
    print(f"  cpu_count               = {_c.machine_fingerprint()['cpu_count_logical']}")
    print(f"  gather_thread_cap       = {config.get_gather_thread_cap()}")
    print(f"  mp_context / cap        = {args.mp_context} / {args.cap_internal}")

    results: list[_c.Result] = []
    for fmt in formats:
        if fmt not in _ALL_FORMATS:
            raise SystemExit(f"unknown format {fmt!r}; choose from {_ALL_FORMATS}")
        _bench_one(fmt, args, results)

    if args.json:
        serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        _c.write_summary(
            args.json,
            results,
            meta={"benchmark": "check_convert_parallel", "args": serializable},
        )
        print(f"\nwrote {args.json} ({len(results)} rows)")


if __name__ == "__main__":
    main()
