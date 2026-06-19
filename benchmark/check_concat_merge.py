"""Benchmark: no-transform merge fast path vs. the materializing write.

``concat(parts, out=...)`` of same-schema files with no transform is a pure
merge: the destination body is exactly the sources' column bytes concatenated.
This compares two write paths -- the materializing per-batch write (used by every
transform/general write) and the no-transform merge fast path -- each with an
mmap fill and a pwrite fill, interleaved A/B so page-cache and scheduler state
stay comparable:

  streaming / mmap   : materialize each batch, store it through a per-column mmap
                       (the pre-pwrite default, and the baseline)
  streaming / pwrite : materialize each batch, write it with large ``os.pwrite``
  merge / mmap       : merge fast path, mmap memcpy of the source byte ranges
  merge / pwrite     : merge fast path, large sequential ``os.pwrite`` of them
  merge / cfr        : merge fast path via ``copy_file_range`` (Linux only)

mmap dirties the destination one page at a time, which a parallel filesystem
serves poorly; pwrite issues large contiguous writes it serves well.
The ``cfr`` row appears only where ``os.copy_file_range`` exists. A correctness
gate asserts all active variants produce a byte-identical file before any timing.
No numbers are baked in -- run it on the target machine, and set ``TMPDIR`` to the
real target filesystem (an mmap store to a parallel filesystem behaves nothing
like one to a node-local disk):

    PYTHONPATH=src python benchmark/check_concat_merge.py
    PYTHONPATH=src python benchmark/check_concat_merge.py --files 16 --rows 2000000
    TMPDIR=/scratch/... PYTHONPATH=src python benchmark/check_concat_merge.py --files 2 --rows 16000000
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


def make_writer(parts: list[Path], out: Path, strategy: str | None, streaming: bool):
    """A no-arg callable that writes ``out`` with the chosen method.

    ``streaming=True`` forces the materializing path (the merge plan disabled)
    with its fill method set to ``strategy`` ("mmap"/"pwrite"); otherwise the
    merge fast path runs with ``strategy`` ("mmap"/"pwrite"/"cfr"). Module state
    is set per call and restored, so interleaving stays correct.
    """

    def run() -> None:
        saved_plan = fmt._merge_copy_plan
        saved_merge = fmt._MERGE_COPY_OVERRIDE
        saved_stream = fmt._STREAMING_FILL_OVERRIDE
        if streaming:
            fmt._merge_copy_plan = lambda *args, **kwargs: None  # type: ignore[assignment]
            fmt._STREAMING_FILL_OVERRIDE = strategy
        else:
            fmt._MERGE_COPY_OVERRIDE = strategy
        try:
            colstore.concat(parts, out=out).close()
        finally:
            fmt._merge_copy_plan = saved_plan
            fmt._MERGE_COPY_OVERRIDE = saved_merge
            fmt._STREAMING_FILL_OVERRIDE = saved_stream

    return run


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=8, help="number of files to merge")
    parser.add_argument(
        "--records", type=int, default=1, help="records per source file (>1 = multi-record)"
    )
    _c.add_common_args(
        parser, repeat=7, warmup=2, rows=1_000_000, cols=4, dtype="float64", scale=True
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

        # Build the variant list. Both the materializing (streaming) write and
        # the merge fast path are compared with their mmap fill against their
        # pwrite fill: pwrite issues large contiguous writes instead of mmap's
        # page-granular dirtying -- the question for a parallel filesystem. The
        # mmap streaming write is the baseline (the pre-pwrite default).
        variants: list[tuple[str, str | None, bool]] = [
            ("streaming / mmap", "mmap", True),
            ("streaming / pwrite", "pwrite", True),
            ("merge / mmap", "mmap", False),
            ("merge / pwrite", "pwrite", False),
        ]
        if have_cfr:
            variants.append(("merge / copy_file_range", "cfr", False))

        outputs = {
            label: Path(directory) / f"out_{i}.cstore" for i, (label, _, _) in enumerate(variants)
        }

        # ---- Correctness gate: every strategy is byte-identical -------------
        if not getattr(args, "skip_correctness", False):
            digests = {}
            for label, strategy, streaming in variants:
                make_writer(parts, outputs[label], strategy, streaming)()
                digests[label] = _digest(outputs[label])
            reference = digests[variants[0][0]]
            for label, value in digests.items():
                status = "ok" if value == reference else "MISMATCH"
                print(f"  correctness {label:28s} {status}")
                if value != reference:
                    raise SystemExit(f"byte-identity check failed for {label}")

        if args.skip_bench:
            return

        banner(
            f"CONCAT WRITE: {args.files} files -> 1 file "
            f"({total_rows:,} rows, {args.cols} cols, {args.records} rec/file)"
        )
        specs = [
            (label, make_writer(parts, outputs[label], strategy, streaming))
            for label, strategy, streaming in variants
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
