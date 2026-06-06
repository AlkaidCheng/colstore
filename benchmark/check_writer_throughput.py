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
import time
from pathlib import Path

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
N_COLUMNS = 4


def check_correctness() -> None:
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


def _bench_one(n_records: int, rows_per_record: int, repeat: int, vectored: bool) -> float:
    rng = np.random.default_rng(0)
    pool = [
        {f"c{j}": rng.standard_normal(rows_per_record) for j in range(N_COLUMNS)}
        for _ in range(min(n_records, 50))
    ]
    original = writer_mod._HAS_WRITEV
    writer_mod._HAS_WRITEV = original and vectored
    try:
        best = float("inf")
        for _ in range(repeat):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "b.cstore"
                start = time.perf_counter()
                with colstore.create(path) as writer:
                    for i in range(n_records):
                        writer.write(pool[i % len(pool)])
                best = min(best, time.perf_counter() - start)
        return best
    finally:
        writer_mod._HAS_WRITEV = original


def run_bench(repeat: int) -> None:
    print(f"{'regime':<30}{'sequential':>12}{'vectored':>12}{'speedup':>9}")
    for n_records, rows in REGIMES:
        t_seq = _bench_one(n_records, rows, repeat, vectored=False)
        t_vec = _bench_one(n_records, rows, repeat, vectored=True)
        total_mb = n_records * rows * N_COLUMNS * 8 / 1e6
        print(
            f"R={n_records:<7} rows/rec={rows:<10}"
            f"{t_seq * 1e3:10.1f}ms{t_vec * 1e3:10.1f}ms{t_seq / t_vec:8.2f}x"
            f"   ({total_mb / t_vec:6.0f} MB/s vectored)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    if not writer_mod._HAS_WRITEV:
        print("os.writev unavailable on this platform; only the fallback path exists.")
        return
    check_correctness()
    if not args.skip_bench:
        run_bench(args.repeat)


if __name__ == "__main__":
    main()
