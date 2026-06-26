"""Fair cross-format read/write/memory comparison for the access colstore targets.

Stores one synthetic table (random-normal ``float64``, chosen because it does not
compress, so every format sits on equal on-disk footing) as a ``.cstore`` and as
each competitor format, then measures, end-to-end from disk with a warm cache and
best-of-N timing:

  * random gather of ``K`` sorted row ids over all columns,
  * boolean-mask filter,
  * single-column scan (sum),
  * write time and on-disk size, and
  * peak resident memory, in a second subprocess-isolated pass.

Competitors are pandas, polars, Arrow/Feather, Parquet (pyarrow), NumPy ``.npy``,
and HDF5 (h5py); each is optional -- a format whose library is not installed is
skipped and reported. The numbers feed the standalone "colstore vs the field"
comparison page; this script asserts no verdict, it only measures::

    PYTHONPATH=src python benchmark/compare_formats.py --rows 5000000 --json compare.json

Peak memory is read in a fresh process per format (RSS is process-wide), so the
script re-invokes itself with ``--peak-read``; that flag is internal.
"""

from __future__ import annotations

import argparse
import importlib.util
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _common as _c
import numpy as np

import colstore

# Localized-read probe: a small contiguous slice from the store's middle, to
# contrast mmap page-residency against load-everything peak memory.
LOCAL_START_FRAC = 0.4
LOCAL_ROWS = 50_000


# ---- workload ---------------------------------------------------------------


@dataclass
class Workload:
    """The shared table and the two selectors every format is measured against."""

    cols: dict[str, np.ndarray]
    names: list[str]
    struct: np.ndarray
    idx: np.ndarray  # sorted gather row ids
    mask: np.ndarray  # boolean row mask


def _names(ncols: int) -> list[str]:
    return [f"c{i}" for i in range(ncols)]


def _sorted_indices(rows: int, k: int, seed: int) -> np.ndarray:
    # Sorted is fair to all formats: a random order would punish chunked readers
    # (Parquet/HDF5) far more than the mmap formats.
    return np.sort(np.random.default_rng(seed).integers(0, rows, k).astype(np.int64))


def make_workload(rows: int, ncols: int, gather_k: int, mask_frac: float, seed: int) -> Workload:
    """Build the random-normal table plus the gather index and the boolean mask."""
    rng = np.random.default_rng(seed)
    names = _names(ncols)
    cols = {n: rng.standard_normal(rows) for n in names}
    struct = np.empty(rows, dtype=np.dtype([(n, np.float64) for n in names]))
    for n in names:
        struct[n] = cols[n]
    return Workload(
        cols=cols,
        names=names,
        struct=struct,
        idx=_sorted_indices(rows, gather_k, seed),
        mask=rng.random(rows) < mask_frac,
    )


# ---- per-format adapters ----------------------------------------------------


@dataclass
class FormatSpec:
    """One format's write and read closures, plus the libraries it needs.

    ``write`` takes the whole :class:`Workload`; the read closures take only the
    selector they need, so the memory subprocess can drive them without building
    the in-memory table. ``local`` is the contiguous-slice read used for the
    localized-memory probe, and is ``None`` for formats that load everything.
    """

    name: str
    requires: tuple[str, ...]
    filename: str
    write: Callable[[Path, Workload], None]
    gather: Callable[[Path, np.ndarray, list[str]], Any]
    scan: Callable[[Path, list[str]], Any]
    mask: Callable[[Path, np.ndarray, list[str]], Any]
    local: Callable[[Path, list[str], int, int], Any] | None = None

    @property
    def available(self) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in self.requires)


def _colstore_spec() -> FormatSpec:
    def write(p: Path, w: Workload) -> None:
        colstore.store(w.cols, p, show_progress=False).close()

    def gather(p: Path, idx: np.ndarray, names: list[str]) -> Any:
        r = colstore.open(p)
        try:
            return r[idx].dict()
        finally:
            r.close()

    def scan(p: Path, names: list[str]) -> Any:
        r = colstore.open(p)
        try:
            return r[names[0]].array(copy=False).sum()
        finally:
            r.close()

    def mask(p: Path, m: np.ndarray, names: list[str]) -> Any:
        r = colstore.open(p)
        try:
            return r[m].dict()
        finally:
            r.close()

    def local(p: Path, names: list[str], start: int, count: int) -> Any:
        r = colstore.open(p)
        try:
            return r[start : start + count].dict()
        finally:
            r.close()

    return FormatSpec("colstore", (), "t.cstore", write, gather, scan, mask, local)


def _parquet_spec() -> FormatSpec:
    def write(p: Path, w: Workload) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table(w.cols), p)

    def gather(p: Path, idx: np.ndarray, names: list[str]) -> Any:
        import pyarrow.parquet as pq

        return pq.read_table(p).take(idx)

    def scan(p: Path, names: list[str]) -> Any:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        return pc.sum(pq.read_table(p, columns=[names[0]]).column(0))

    def mask(p: Path, m: np.ndarray, names: list[str]) -> Any:
        import pyarrow as pa
        import pyarrow.parquet as pq

        return pq.read_table(p).filter(pa.array(m))

    return FormatSpec("parquet", ("pyarrow",), "t.parquet", write, gather, scan, mask)


def _feather_spec() -> FormatSpec:
    def write(p: Path, w: Workload) -> None:
        import pyarrow as pa
        import pyarrow.feather as feather

        feather.write_feather(pa.table(w.cols), p, compression="uncompressed")

    def _read(p: Path) -> Any:
        import pyarrow as pa

        with pa.memory_map(str(p)) as src:
            return pa.ipc.open_file(src).read_all()

    def gather(p: Path, idx: np.ndarray, names: list[str]) -> Any:
        return _read(p).take(idx)

    def scan(p: Path, names: list[str]) -> Any:
        import pyarrow.compute as pc

        return pc.sum(_read(p).column(0))

    def mask(p: Path, m: np.ndarray, names: list[str]) -> Any:
        import pyarrow as pa

        return _read(p).filter(pa.array(m))

    return FormatSpec("arrow_feather", ("pyarrow",), "t.arrow", write, gather, scan, mask)


def _pandas_spec() -> FormatSpec:
    def write(p: Path, w: Workload) -> None:
        import pandas as pd

        pd.DataFrame(w.cols).to_parquet(p)

    def gather(p: Path, idx: np.ndarray, names: list[str]) -> Any:
        import pandas as pd

        return pd.read_parquet(p).iloc[idx]

    def scan(p: Path, names: list[str]) -> Any:
        import pandas as pd

        return pd.read_parquet(p, columns=[names[0]])[names[0]].sum()

    def mask(p: Path, m: np.ndarray, names: list[str]) -> Any:
        import pandas as pd

        return pd.read_parquet(p)[m]

    def local(p: Path, names: list[str], start: int, count: int) -> Any:
        import pandas as pd

        return pd.read_parquet(p).iloc[start : start + count]

    return FormatSpec("pandas", ("pandas",), "t_pd.parquet", write, gather, scan, mask, local)


def _polars_spec() -> FormatSpec:
    def write(p: Path, w: Workload) -> None:
        import polars as pl

        pl.DataFrame(w.cols).write_parquet(p)

    def gather(p: Path, idx: np.ndarray, names: list[str]) -> Any:
        import polars as pl

        return pl.read_parquet(p)[idx]

    def scan(p: Path, names: list[str]) -> Any:
        import polars as pl

        return pl.read_parquet(p, columns=[names[0]])[names[0]].sum()

    def mask(p: Path, m: np.ndarray, names: list[str]) -> Any:
        import polars as pl

        return pl.read_parquet(p).filter(pl.Series(m))

    return FormatSpec("polars", ("polars",), "t_pl.parquet", write, gather, scan, mask)


def _npy_spec() -> FormatSpec:
    def write(p: Path, w: Workload) -> None:
        np.save(p, w.struct)

    def gather(p: Path, idx: np.ndarray, names: list[str]) -> Any:
        m = np.load(p, mmap_mode="r")
        return {n: m[n][idx] for n in names}

    def scan(p: Path, names: list[str]) -> Any:
        return np.load(p, mmap_mode="r")[names[0]].sum()

    def mask(p: Path, mk: np.ndarray, names: list[str]) -> Any:
        m = np.load(p, mmap_mode="r")
        return {n: m[n][mk] for n in names}

    def local(p: Path, names: list[str], start: int, count: int) -> Any:
        m = np.load(p, mmap_mode="r")
        return {n: m[n][start : start + count] for n in names}

    # The filename already carries the .npy suffix np.save would otherwise append.
    return FormatSpec("npy", (), "t.npy", write, gather, scan, mask, local)


def _hdf5_spec() -> FormatSpec:
    def write(p: Path, w: Workload) -> None:
        import h5py

        with h5py.File(p, "w") as f:
            for n in w.names:
                f.create_dataset(n, data=w.cols[n], chunks=(1 << 16,))

    def gather(p: Path, idx: np.ndarray, names: list[str]) -> Any:
        import h5py

        # h5py fancy-index on a sorted array is slow; read then index in numpy.
        with h5py.File(p, "r") as f:
            return {n: f[n][:][idx] for n in names}

    def scan(p: Path, names: list[str]) -> Any:
        import h5py

        with h5py.File(p, "r") as f:
            return f[names[0]][:].sum()

    def mask(p: Path, mk: np.ndarray, names: list[str]) -> Any:
        import h5py

        with h5py.File(p, "r") as f:
            return {n: f[n][:][mk] for n in names}

    return FormatSpec("hdf5", ("h5py",), "t.h5", write, gather, scan, mask)


def all_specs() -> list[FormatSpec]:
    """Every format adapter, colstore first."""
    return [
        _colstore_spec(),
        _parquet_spec(),
        _feather_spec(),
        _pandas_spec(),
        _polars_spec(),
        _npy_spec(),
        _hdf5_spec(),
    ]


# ---- measurement ------------------------------------------------------------


def _file_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*")) / 1e6
    return path.stat().st_size / 1e6


def peak_rss_mb() -> float:
    """Peak resident set size in MB (``ru_maxrss`` is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def _time_reads(
    spec: FormatSpec, path: Path, w: Workload, *, repeat: int, warmup: int
) -> dict[str, _c.TimeStats]:
    """Best-of-N timings for the three read shapes against one written file."""
    return {
        "random_gather": _c.time_stats(
            lambda: spec.gather(path, w.idx, w.names), repeat=repeat, warmup=warmup
        ),
        "column_scan": _c.time_stats(
            lambda: spec.scan(path, w.names), repeat=repeat, warmup=warmup
        ),
        "mask_filter": _c.time_stats(
            lambda: spec.mask(path, w.mask, w.names), repeat=repeat, warmup=warmup
        ),
    }


def run_speed(
    specs: list[FormatSpec],
    workdir: Path,
    w: Workload,
    *,
    repeat: int,
    warmup: int,
    params: dict[str, Any],
) -> list[_c.Result]:
    """Time write, on-disk size, gather, scan, and mask for each available format."""
    rows = len(w.struct)
    print(
        f"\nDataset: {rows:,} rows x {len(w.names)} float64 cols "
        f"({w.struct.nbytes / 1e6:.0f} MB raw). Gather K={len(w.idx):,} (sorted). "
        f"Mask {w.mask.mean():.0%}. Best-of-{repeat}.\n"
    )
    header = (
        f"{'format':16} {'write s':>8} {'file MB':>8} "
        f"{'gather ms':>10} {'scan ms':>9} {'mask ms':>9}"
    )
    print(header)
    print("-" * len(header))

    records: list[_c.Result] = []
    for spec in specs:
        if not spec.available:
            print(f"{spec.name:16}  skipped (missing {', '.join(spec.requires)})")
            continue
        path = workdir / spec.filename
        start = time.perf_counter()
        spec.write(path, w)
        write_s = time.perf_counter() - start
        size_mb = _file_mb(path)

        timed = _time_reads(spec, path, w, repeat=repeat, warmup=warmup)
        records.extend(
            _c.Result.from_stats(scenario, spec.name, params, stats, rows=rows)
            for scenario, stats in timed.items()
        )
        records.append(
            _c.Result(
                scenario="write",
                variant=spec.name,
                params={**params, "file_mb": round(size_mb, 1)},
                median_ms=write_s * 1000.0,
                min_ms=write_s * 1000.0,
                p95_ms=write_s * 1000.0,
                repeat=1,
                rows=rows,
            )
        )
        print(
            f"{spec.name:16} {write_s:8.2f} {size_mb:8.1f} "
            f"{timed['random_gather'].min_ms:10.1f} {timed['column_scan'].min_ms:9.2f} "
            f"{timed['mask_filter'].min_ms:9.1f}"
        )
    return records


def run_memory(specs: list[FormatSpec], args: argparse.Namespace) -> None:
    """Peak RSS per format for a spanning gather and a localized slice, each isolated."""
    available = [s for s in specs if s.available]
    with tempfile.TemporaryDirectory(prefix="colstore_cmp_mem") as tmp:
        d = Path(tmp)
        w = make_workload(args.rows, args.cols, args.indices, args.mask_frac, args.seed)
        for spec in available:
            spec.write(d / spec.filename, w)
        del w  # free the writer-side table before spawning the read probes

        def probe(name: str, kind: str) -> str:
            out = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--peak-read",
                    name,
                    "--peak-kind",
                    kind,
                    "--data-dir",
                    str(d),
                    "--rows",
                    str(args.rows),
                    "--cols",
                    str(args.cols),
                    "--indices",
                    str(args.indices),
                    "--seed",
                    str(args.seed),
                ],
                capture_output=True,
                text=True,
            )
            for line in out.stdout.splitlines():
                if line.startswith("PEAK_RSS_MB "):
                    return line.split()[1]
            return f"ERR ({out.stderr.strip()[-60:]})"

        print(f"\nPeak RSS — spanning gather (K={args.indices:,}), isolated process, MB:")
        for spec in available:
            print(f"  {spec.name:16} {probe(spec.name, 'gather')}")
        print(f"\nPeak RSS — localized {LOCAL_ROWS:,}-row slice, isolated process, MB:")
        for spec in available:
            if spec.local is not None:
                print(f"  {spec.name:16} {probe(spec.name, 'local')}")


def _peak_read(args: argparse.Namespace) -> None:
    """Subprocess entry: open one pre-written format, do one read, print peak RSS."""
    spec = {s.name: s for s in all_specs()}[args.peak_read]
    path = args.data_dir / spec.filename
    names = _names(args.cols)
    if args.peak_kind == "gather":
        spec.gather(path, _sorted_indices(args.rows, args.indices, args.seed), names)
    else:
        assert spec.local is not None
        spec.local(path, names, int(args.rows * LOCAL_START_FRAC), LOCAL_ROWS)
    print(f"PEAK_RSS_MB {peak_rss_mb():.0f}")


# ---- entry point ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(
        parser,
        repeat=5,
        warmup=1,
        rows=5_000_000,
        cols=8,
        indices=1_000_000,
        json=True,
        skip_correctness=False,
    )
    parser.add_argument("--mask-frac", type=float, default=0.3, help="boolean mask selectivity")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--no-memory", action="store_true", help="skip the peak-memory pass")
    # Internal: a fresh-process peak-RSS probe re-invoked by run_memory.
    parser.add_argument("--peak-read", help=argparse.SUPPRESS)
    parser.add_argument("--peak-kind", choices=("gather", "local"), help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.peak_read:
        _peak_read(args)
        return

    specs = all_specs()
    params = {
        "rows": args.rows,
        "cols": args.cols,
        "gather_k": args.indices,
        "mask_frac": args.mask_frac,
    }
    if args.skip_bench:
        present = [s.name for s in specs if s.available]
        print("formats available:", ", ".join(present))
        return

    records: list[_c.Result] = []
    with tempfile.TemporaryDirectory(prefix="colstore_cmp") as tmp:
        w = make_workload(args.rows, args.cols, args.indices, args.mask_frac, args.seed)
        records = run_speed(
            specs, Path(tmp), w, repeat=args.repeat, warmup=args.warmup, params=params
        )
    if not args.no_memory:
        run_memory(specs, args)

    if args.json:
        _c.write_summary(args.json, records, meta=params)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
