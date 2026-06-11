"""Verify the vectored writer: correctness and throughput vs the sequential path.

The streaming writer emits each record (32-byte header + column bodies +
alignment padding) with a single writev() where the platform provides it;
previously every record cost one buffered write per piece, and each numpy
``tofile()`` also forced a flush of the buffered layer -- one syscall plus a
flush per column. For many-small-record streams that machinery dominates the
write cost; for large records the page-cache memcpy dominates and the two
paths converge.

Run on the deployment hardware:

    python benchmark/check_writer_throughput.py
    python benchmark/check_writer_throughput.py --skip-bench   # correctness only

Expected shape: several-fold gain for tiny records, converging to ~1.0x for
records of hundreds of KB and above. The sequential column reports the
no-writev fallback path, which remains in use on platforms without
``os.writev``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import _common as _c
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import colstore
from colstore import writer as writer_mod

REGIMES = (
    (20_000, 50),
    (5_000, 200),
    (500, 20_000),
    (20, 1_000_000),
    (10, 2_000_000),
    (2, 5_000_000),
    (1, 10_000_000),
)


def check_correctness(n_columns: int) -> None:
    rng = np.random.default_rng(3)
    records = [
        {
            "a": rng.standard_normal(n),
            "b": rng.integers(0, 100, n).astype(np.int32),
            "c": rng.standard_normal(n).astype(np.float32),
        }
        for n in (50, 0, 1, 1000, 7)
    ]
    digests = {}
    for label, vectored in (("vectored", True), ("sequential", False)):
        original = writer_mod._HAS_WRITEV
        writer_mod._HAS_WRITEV = original and vectored
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "c.cstore"
                with colstore.create(path) as writer:
                    for record in records:
                        writer.write(record)
                with colstore.update(path) as writer:
                    writer.write(records[0])
                digests[label] = hashlib.sha256(path.read_bytes()).hexdigest()
                dataset = colstore.open(path)
                expected = np.concatenate([r["a"] for r in records] + [records[0]["a"]])
                assert np.array_equal(dataset[:, "a"].array(), expected), label
                dataset.close()
        finally:
            writer_mod._HAS_WRITEV = original
    assert digests["vectored"] == digests["sequential"], "paths diverged on disk"
    print("  ALL CORRECTNESS CHECKS PASSED (paths byte-identical)\n")


def _make_write_spec(vectored, pool, n_records, base_dir):
    """A (setup, write) pair: setup mints a fresh store dir outside timing;
    write streams ``n_records`` records into it with writev forced on/off."""
    state: dict[str, Path] = {}

    def setup() -> None:
        state["path"] = Path(tempfile.mkdtemp(dir=base_dir)) / "b.cstore"

    def write() -> None:
        original = writer_mod._HAS_WRITEV
        writer_mod._HAS_WRITEV = original and vectored
        try:
            with colstore.create(state["path"]) as writer:
                for i in range(n_records):
                    writer.write(pool[i % len(pool)])
        finally:
            writer_mod._HAS_WRITEV = original

    return setup, write


def run_bench(repeat: int, warmup: int, n_columns: int) -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as base:
        base_dir = Path(base)
        for n_records, rows in REGIMES:
            pool = [
                {f"c{j}": rng.standard_normal(rows) for j in range(n_columns)}
                for _ in range(min(n_records, 50))
            ]
            seq_setup, seq_write = _make_write_spec(False, pool, n_records, base_dir)
            vec_setup, vec_write = _make_write_spec(True, pool, n_records, base_dir)
            total_mb = n_records * rows * n_columns * 8 / 1e6
            print(f"R={n_records:<7} rows/rec={rows:<10} ({total_mb:.0f} MB total)")
            _seq_res, vec_res = _c.compare(
                [("sequential", seq_write), ("vectored", vec_write)],
                repeat=repeat,
                warmup=warmup,
                baseline=0,
                setups=[seq_setup, vec_setup],
            )
            print(f"    vectored throughput: {total_mb / (vec_res.wall_ms / 1000.0):.0f} MB/s\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _c.add_common_args(parser, repeat=3, cols=4, skip_correctness=False)
    args = parser.parse_args()

    if not writer_mod._HAS_WRITEV:
        print("os.writev unavailable on this platform; only the fallback path exists.")
        return
    check_correctness(args.cols)
    if not args.skip_bench:
        run_bench(args.repeat, args.warmup, args.cols)


if __name__ == "__main__":
    main()
