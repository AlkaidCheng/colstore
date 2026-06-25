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
import resource
import shutil
import subprocess
import sys
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


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss * 1024 if sys.platform == "linux" else rss  # Linux KiB, macOS bytes


def _measure_rss(variant: str, src: Path, out: Path) -> int:
    """Peak RSS of writing ``out`` via ``variant``, isolated in a fresh process."""
    out.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, __file__, "--rss-variant", variant, "--src", str(src), "--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK_RSS_BYTES="):
            return int(line.split("=", 1)[1])
    raise RuntimeError(f"no RSS from the {variant} subprocess:\n{proc.stdout}\n{proc.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=5, warmup=2, rows=10_000_000, tmpdir=True, json=True)
    parser.add_argument("--rss-variant", choices=_VARIANTS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--src", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess RSS mode: run one strategy, report this process's peak RSS.
    if args.rss_variant is not None:
        _STRATEGY[args.rss_variant](args.src, args.out)
        print(f"PEAK_RSS_BYTES={_peak_rss_bytes()}")
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

    print("\n=== peak RSS: subprocess-isolated, one strategy per process ===")
    rss = {v: _measure_rss(v, src, outs[v]) for v in _VARIANTS}
    base = rss["copy"]  # copy holds no column data; the rest is interpreter baseline
    for variant in _VARIANTS:
        delta = (rss[variant] - base) / 1e6
        print(f"  {variant:>12}  peak={rss[variant] / 1e6:8.0f} MB   (+{delta:7.0f} MB vs copy)")

    if args.json is not None:
        summary = [
            _c.Result(
                scenario="append_source",
                variant=variant,
                params={"peak_rss_mb": round(rss[variant] / 1e6, 1)},
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
