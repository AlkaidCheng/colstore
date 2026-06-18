"""Benchmark: no-transform merge fast path vs. the materializing write.

``concat(parts, out=...)`` of same-schema files with no transform is a pure
merge: the destination body is exactly the sources' column bytes concatenated.
This compares three ways to fill that body, interleaved A/B so page-cache and
scheduler state stay comparable:

  streaming     : the materializing per-batch write (the prior path) -- read
                  each column region into NumPy, assign into the output memmap
  merge / mmap  : the merge fast path, mmap memcpy of the source byte ranges
  merge / cfr   : the merge fast path via ``copy_file_range`` (Linux only;
                  on reflink/networked filesystems the kernel can share extents
                  or copy server-side, avoiding the client byte transfer)

The ``cfr`` row appears only where ``os.copy_file_range`` exists, so the same
script reports streaming vs. mmap on a local machine and adds the
``copy_file_range`` comparison on a multi-node Linux target. A correctness gate
asserts all active strategies produce a byte-identical file before any timing.
No numbers are baked in -- run it on the target machine.

    PYTHONPATH=src python benchmark/check_concat_merge.py
    PYTHONPATH=src python benchmark/check_concat_merge.py --files 16 --rows 2000000
    PYTHONPATH=src python benchmark/check_concat_merge.py --records 8   # multi-record sources

To sweep run-split sizes (the question of whether finer parallel granularity
helps the copy), pass several ``--chunk-mb`` values; ``0`` is no split. Splitting
can only help when the run count (files x columns) is below the copy's thread
budget, so test it on a compute node with a high ``--thread`` cap and few large
files -- otherwise the runs already fill the threads and splitting is a no-op:

    PYTHONPATH=src python benchmark/check_concat_merge.py --files 2 --rows 16000000 \\
        --thread 64 --chunk-mb 0 4 16 64
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import _common as _c

import colstore
from colstore import format as fmt
from colstore import testing


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def build_parts(
    directory: str, n_files: int, rows_per_file: int, cols: int, dtype: str, records: int
) -> list[Path]:
    """Write ``n_files`` same-schema stores (distinct seeds); return their paths."""
    paths = []
    for i in range(n_files):
        path = Path(directory) / f"part_{i:03d}.cstore"
        testing.make_store(
            path, rows=rows_per_file, cols=cols, records=records, dtype=dtype, seed=i
        ).close()
        paths.append(path)
    return paths


def make_writer(parts: list[Path], out: Path, strategy: str | None, streaming: bool, chunk: int):
    """A no-arg callable that writes ``out`` with the chosen fill strategy.

    ``streaming=True`` forces the materializing path (the merge plan disabled);
    otherwise the merge fast path runs with ``strategy`` ("mmap"/"cfr"/None) and
    runs split at ``chunk`` bytes (0 = no chunking). Module state is set per call
    and restored, so interleaving stays correct.
    """

    def run() -> None:
        saved_plan = fmt._merge_copy_plan
        saved_override = fmt._MERGE_COPY_OVERRIDE
        saved_chunk = fmt._MERGE_COPY_CHUNK_BYTES
        if streaming:
            fmt._merge_copy_plan = lambda *args, **kwargs: None  # type: ignore[assignment]
        else:
            fmt._MERGE_COPY_OVERRIDE = strategy
            fmt._MERGE_COPY_CHUNK_BYTES = chunk
        try:
            colstore.concat(parts, out=out).close()
        finally:
            fmt._merge_copy_plan = saved_plan
            fmt._MERGE_COPY_OVERRIDE = saved_override
            fmt._MERGE_COPY_CHUNK_BYTES = saved_chunk

    return run


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=8, help="number of files to merge")
    parser.add_argument(
        "--records", type=int, default=1, help="records per source file (>1 = multi-record)"
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        nargs="+",
        default=[0],
        help="run-split sizes (MiB) to sweep for the merge/mmap copy; 0 = no split",
    )
    _c.add_common_args(
        parser,
        repeat=7,
        warmup=2,
        rows=1_000_000,
        cols=4,
        dtype="float64",
        threads=True,
        scale=True,
    )
    args = parser.parse_args()
    _c.apply_runtime_config(args)

    rows_per_file = _c.scaled_rows(args.rows, args)
    total_rows = rows_per_file * args.files
    have_cfr = hasattr(os, "copy_file_range")

    print("Environment:")
    print(f"  platform={os.uname().sysname}  copy_file_range={have_cfr}")
    print(
        f"  files={args.files}  rows/file={rows_per_file:,}  total={total_rows:,}  "
        f"cols={args.cols}  records/file={args.records}  dtype={args.dtype}"
    )

    with tempfile.TemporaryDirectory() as directory:
        parts = build_parts(
            directory, args.files, rows_per_file, args.cols, args.dtype, args.records
        )

        # Build the variant list: streaming baseline, then one merge/mmap variant
        # per swept chunk size, and cfr where present. (label, strategy,
        # streaming, chunk_bytes).
        variants: list[tuple[str, str | None, bool, int]] = [
            ("streaming (materialize)", None, True, 0)
        ]
        for mb in args.chunk_mb:
            label = "merge / mmap" if mb == 0 else f"merge / mmap (chunk={mb}MB)"
            variants.append((label, "mmap", False, mb * 1024 * 1024))
        if have_cfr:
            variants.append(("merge / copy_file_range", "cfr", False, 0))

        outputs = {
            label: Path(directory) / f"out_{i}.cstore"
            for i, (label, _, _, _) in enumerate(variants)
        }

        # ---- Correctness gate: every strategy is byte-identical -------------
        if not getattr(args, "skip_correctness", False):
            digests = {}
            for label, strategy, streaming, chunk in variants:
                make_writer(parts, outputs[label], strategy, streaming, chunk)()
                digests[label] = _digest(outputs[label])
            reference = digests["streaming (materialize)"]
            for label, value in digests.items():
                status = "ok" if value == reference else "MISMATCH"
                print(f"  correctness {label:32s} {status}")
                if value != reference:
                    raise SystemExit(f"byte-identity check failed for {label}")

        if args.skip_bench:
            return

        banner(
            f"CONCAT WRITE: {args.files} files -> 1 file "
            f"({total_rows:,} rows, {args.cols} cols, {args.records} rec/file)"
        )
        specs = [
            (label, make_writer(parts, outputs[label], strategy, streaming, chunk))
            for label, strategy, streaming, chunk in variants
        ]
        _c.compare(
            specs,
            repeat=args.repeat,
            warmup=args.warmup,
            baseline=0,
            throughput_rows=total_rows,
        )


if __name__ == "__main__":
    main()
