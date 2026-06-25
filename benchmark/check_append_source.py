"""A/B the strategies for appending a colstore source as a shard.

Appending an existing ``.cstore`` source can be done three ways:

* **materialize** -- read every column into memory, then write it back out (the
  original append path; peak memory scales with the whole source);
* **stream** -- evaluate the columns one row range at a time into the new shard
  (bounded memory; a no-transform source takes the merge-copy fast path);
* **copy** -- a single-file source already has the optimal layout, so copy the
  file byte-for-byte (``shutil.copyfile`` -> ``copy_file_range`` / ``sendfile``).

This measures all three on one source file: a correctness gate that each produces
a read-identical shard, an interleaved wall-time A/B, and a subprocess-isolated
peak-RSS reading (the point of the change -- materialize holds the whole file in
RAM, the other two do not).

    PYTHONPATH=src python benchmark/check_append_source.py --tmpdir $SCRATCH/ --rows 10000000
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import _common as _c
import numpy as np

_VARIANTS = ("materialize", "stream", "copy")


def _build_source(path: Path, rows: int) -> None:
    cols = {
        "key": np.arange(rows, dtype=np.int64),
        "val": (np.arange(rows) * 1.5).astype(np.float64),
        "flag": (np.arange(rows) % 7).astype(np.int32),
    }
    _c.colstore.store(cols, path, mode="recreate", show_progress=False).close()


def _materialize(src: Path, out: Path) -> None:
    """Original path: read the whole source into memory, then write it back."""
    reader = _c.colstore.open(src)
    try:
        cols = {name: reader[name].array() for name in reader.columns}
    finally:
        reader.close()
    _c.colstore.store(cols, out, mode="recreate", show_progress=False).close()


def _stream(src: Path, out: Path) -> None:
    """Bounded-memory streaming write (the multi-file append path)."""
    reader = _c.colstore.open(src)
    try:
        reader.edit().write(out).close()
    finally:
        reader.close()


def _copy(src: Path, out: Path) -> None:
    """Whole-file byte copy (the single-file append fast path)."""
    shutil.copyfile(src, out)


_STRATEGY = {"materialize": _materialize, "stream": _stream, "copy": _copy}


def _private_rss_bytes() -> int | None:
    """Current *private* (anonymous) resident bytes on Linux, else ``None``.

    ``resident - shared`` from ``/proc/self/statm`` excludes the file-backed,
    reclaimable source mmap and counts only private memory -- the materialized
    arrays -- which is the real demand on the memory budget.
    """
    try:
        with open("/proc/self/statm") as handle:
            fields = handle.read().split()
        private_pages = int(fields[1]) - int(fields[2])
    except (OSError, ValueError, IndexError):
        return None
    return private_pages * os.sysconf("SC_PAGE_SIZE")


class _PeakRSS:
    """Sample *current private* RSS in a background thread; report the peak rise.

    Uses current private RSS rather than lifetime ``ru_maxrss``, which on a
    many-core node is swamped by the one-time import/threading-init peak (identical
    for every strategy) and would also count the shared source mmap. The figure is
    the private memory the operation adds. Where ``/proc`` is unavailable (e.g.
    macOS) it falls back to the lifetime peak, less precise.
    """

    def __init__(self, interval: float = 0.001) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._baseline = _private_rss_bytes()
        self._peak = self._baseline or 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _PeakRSS:
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _private_rss_bytes()
            if rss is not None and rss > self._peak:
                self._peak = rss
            time.sleep(self._interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        rss = _private_rss_bytes()
        if rss is not None and rss > self._peak:
            self._peak = rss

    @property
    def peak_rise_bytes(self) -> int:
        if self._baseline is None:  # no /proc: lifetime peak (units per platform)
            lifetime = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return lifetime * 1024 if sys.platform == "linux" else lifetime
        return self._peak - self._baseline


def _rss_warmup(src: Path) -> None:
    """Spin up the gather thread pool and steady state before the RSS baseline,
    so the measured rise is the operation's data, not one-time init."""
    reader = _c.colstore.open(src)
    try:
        reader[:1024, reader.columns[0]].array()
    finally:
        reader.close()


def _measure_rss(variant: str, src: Path, out: Path) -> int:
    """Peak RSS rise of writing ``out`` via ``variant``, isolated in a fresh process."""
    out.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, __file__, "--rss-variant", variant, "--src", str(src), "--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK_RSS_RISE_BYTES="):
            return int(line.split("=", 1)[1])
    raise RuntimeError(f"no RSS from the {variant} subprocess:\n{proc.stdout}\n{proc.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=2, rows=10_000_000, tmpdir=True, json=True)
    parser.add_argument("--rss-variant", choices=_VARIANTS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--src", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess RSS mode: warm up, then report the RSS the one strategy adds.
    if args.rss_variant is not None:
        _rss_warmup(args.src)
        with _PeakRSS() as peak:
            _STRATEGY[args.rss_variant](args.src, args.out)
        print(f"PEAK_RSS_RISE_BYTES={peak.peak_rise_bytes}")
        return 0

    work = Path(args.tmpdir) if args.tmpdir is not None else Path(".")
    work.mkdir(parents=True, exist_ok=True)
    src = work / "append_source.cstore"
    outs = {v: work / f"append_out_{v}.cstore" for v in _VARIANTS}
    rows = args.rows

    _build_source(src, rows)
    src_mb = src.stat().st_size / 1e6

    # Correctness gate: every strategy reproduces the source, in order.
    oracle = _c.colstore.open(src)
    key_ref, val_ref = oracle.array("key"), oracle.array("val")
    oracle.close()
    for variant, out in outs.items():
        out.unlink(missing_ok=True)
        _STRATEGY[variant](src, out)
        reader = _c.colstore.open(out)
        _c.check_equal(reader.array("key"), key_ref, f"{variant}: key")
        _c.check_equal(reader.array("val"), val_ref, f"{variant}: val")
        reader.close()
    print(f"# correctness gate passed: all strategies == source ({rows} rows, {src_mb:.0f} MB)")
    if args.skip_bench:
        return 0

    print(f"\n=== wall time: append a {src_mb:.0f} MB source ({rows} rows) ===")
    specs = [(v, (lambda v=v: _STRATEGY[v](src, outs[v]))) for v in _VARIANTS]
    setups = [(lambda v=v: outs[v].unlink(missing_ok=True)) for v in _VARIANTS]
    results = _c.compare(specs, repeat=args.repeat, warmup=args.warmup, baseline=0, setups=setups)
    for variant, res in zip(_VARIANTS, results, strict=True):
        print(f"  {variant:>12}  {src_mb / (res.wall_ms / 1e3):8.0f} MB/s")

    print("\n=== peak RSS added by the operation (subprocess-isolated, baselined) ===")
    rss = {v: _measure_rss(v, src, outs[v]) for v in _VARIANTS}
    for variant in _VARIANTS:
        print(f"  {variant:>12}  +{rss[variant] / 1e6:8.1f} MB resident")

    if args.json is not None:
        summary = [
            _c.Result(
                scenario="append_source",
                variant=variant,
                params={"peak_rss_rise_mb": round(rss[variant] / 1e6, 1)},
                median_ms=res.wall_ms,
                min_ms=res.wall_ms,
                p95_ms=res.wall_ms,
                repeat=args.repeat,
                rows=rows,
            )
            for variant, res in zip(_VARIANTS, results, strict=True)
        ]
        _c.write_summary(args.json, summary, meta={"benchmark": "check_append_source"})
        print(f"\n# wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
