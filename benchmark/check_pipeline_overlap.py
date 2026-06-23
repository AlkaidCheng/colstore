"""End-to-end streaming-write pipeline profile: where the time goes, and the
I/O-compute overlap ceiling.

The I/O-compute overlap this profiler sizes shipped as optimization Round 6 (a
background writer); the parallel-per-column-compute lever was measured and
rejected. The profiler stays useful as a serial-baseline phase decomposition: it
forces the inline (non-overlapped) write so each batch's compute and write are
sequential, then reports the per-phase split and the overlap/parallel-compute
ceilings the levers were judged against.

The streaming write (``frame.write`` -> ``write_dataset_streaming`` ->
``_fill_streaming_pwrite``), with the overlap disabled, runs strictly serial per
batch: it reads+transforms every column of a batch into memory, then ``pwrite``s
them, and only then reads the next batch. Both phases release the GIL (the
gather/ufunc kernels and ``os.pwrite``), so a double-buffer pipeline overlaps one
batch's write with the next batch's read+transform -- the ceiling this measures.

It decomposes the *real* write path, with no source change, into:

* per-batch **compute** (read+transform), the gaps between the single per-batch
  ``_run_copy_jobs`` write call;
* per-batch **write**, the ``_run_copy_jobs`` call itself (``pwrite``, which on a
  parallel filesystem fills the page cache; the bytes reach the OSTs at flush);
* the **durability flush**, the final whole-file ``os.fsync`` after all batches --
  a hard serial barrier no overlap can remove;
* **setup** (preallocation + its fsync) and other fixed teardown.

and reports the **overlap ceiling** -- the upper-bound wall a perfect read/write
double-buffer would save, ``sum_b min(compute_{b+1}, write_b)`` -- as a fraction
of the wall. A shape whose ceiling is small is write-bound, compute-bound, or
flush-bound and gains nothing from overlap; a large ceiling is the case the
pipeline change would help.

Three workload shapes bracket the compute/write ratio: a filtered passthrough
write (a gather read plus the write, no transform), a light transform (one add
per column), and a deep transform (sqrt + multiply-add per column).

This profiles ``write_dataset_streaming`` -- the streaming fill itself. The public
``frame.write`` additionally opens a reader on the finished file (an O(file_size)
mmap fault), excluded here so the residual reflects the write, not the re-open.

Run on the deployment node with ``--tmpdir`` (or ``TMPDIR``) pointing at the
parallel filesystem; local numbers are indicative only.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import _common as _c
import numpy as np

import colstore
import colstore.format as _fmt
import colstore.frame as _frm
from colstore import col, config, testing
from colstore.format import write_dataset_streaming

# A filtered write keeps this fraction of the source rows (a realistic cut),
# sorted (a cut preserves row order) so the read leg is a sorted multi-record
# gather rather than a random scatter.
_FILTER_FRACTION = 0.5
_FILTER_SEED = 12345

# Correctness runs on a small store so a full in-memory reference stays cheap;
# the timed profile runs on the full store with no reference held.
_CORRECTNESS_ROWS = 200_000
_CORRECTNESS_RECORDS = 50


# --------------------------------------------------------------------------- #
# Faithful, zero-source-change phase recorder.
# --------------------------------------------------------------------------- #


class _PipelineRecorder:
    """Decompose one ``write_dataset_streaming`` call into its real phases.

    ``_fill_streaming_pwrite`` builds every column's bytes for a batch (compute),
    then calls ``_run_copy_jobs`` exactly once to ``pwrite`` that batch's disjoint
    regions (write); ``write_dataset_streaming`` ``os.fsync``s once after
    preallocation and once after all batches (the durability flush). Installed in
    place of ``colstore.format._run_copy_jobs`` and ``os.fsync``, this timestamps
    each such call -- only while ``armed`` -- so the phases can be reconstructed in
    post. Disarmed it is a transparent pass-through, so correctness writes and any
    merge-copy in the same process are not recorded.
    """

    def __init__(self) -> None:
        self.armed = False
        self._orig_run_copy_jobs = _fmt._run_copy_jobs
        self._orig_fsync = os.fsync
        self._orig_read = _frm.NativeColumn._read
        self._orig_fill = _frm.NativeColumn._fill_into
        self.t_start = 0.0
        self.read_s = 0.0  # gather time within the compute phases (read, not transform)
        self.writes: list[tuple[float, float]] = []  # (entry, done) perf_counter
        self.fsyncs: list[tuple[float, float]] = []

    def install(self) -> None:
        rec, orig_read, orig_fill = self, self._orig_read, self._orig_fill

        def read_leaf(native: object, rows: object) -> object:
            if not rec.armed:
                return orig_read(native, rows)
            t = time.perf_counter()
            try:
                return orig_read(native, rows)
            finally:
                rec.read_s += time.perf_counter() - t

        def fill_leaf(native: object, out: object, start: int, stop: int) -> object:
            if not rec.armed:
                return orig_fill(native, out, start, stop)
            t = time.perf_counter()
            try:
                return orig_fill(native, out, start, stop)
            finally:
                rec.read_s += time.perf_counter() - t

        _frm.NativeColumn._read = read_leaf  # type: ignore[assignment,method-assign]
        _frm.NativeColumn._fill_into = fill_leaf  # type: ignore[assignment,method-assign]
        _fmt._run_copy_jobs = self._run_copy_jobs  # type: ignore[assignment]
        os.fsync = self._fsync  # type: ignore[assignment]

    def restore(self) -> None:
        _frm.NativeColumn._read = self._orig_read  # type: ignore[method-assign]
        _frm.NativeColumn._fill_into = self._orig_fill  # type: ignore[method-assign]
        _fmt._run_copy_jobs = self._orig_run_copy_jobs  # type: ignore[assignment]
        os.fsync = self._orig_fsync

    def arm(self) -> float:
        self.writes = []
        self.fsyncs = []
        self.read_s = 0.0
        self.t_start = time.perf_counter()
        self.armed = True
        return self.t_start

    def disarm(self) -> None:
        self.armed = False

    def _run_copy_jobs(self, jobs: list[Callable[[], None]], workers: int) -> None:
        if not self.armed:
            self._orig_run_copy_jobs(jobs, workers)
            return
        entry = time.perf_counter()
        self._orig_run_copy_jobs(jobs, workers)
        self.writes.append((entry, time.perf_counter()))

    def _fsync(self, fd: int) -> None:
        if not self.armed:
            self._orig_fsync(fd)
            return
        entry = time.perf_counter()
        self._orig_fsync(fd)
        self.fsyncs.append((entry, time.perf_counter()))


_recorder = _PipelineRecorder()


@dataclass(frozen=True)
class PhaseRun:
    """The least-perturbed (min-wall) decomposed run of one shape (all ms)."""

    wall_ms: float
    compute_ms: list[float]
    write_ms: list[float]
    setup_flush_ms: float
    durability_flush_ms: float
    read_ms: float  # gather share of compute; transform = compute_total - read_ms

    @staticmethod
    def analyze(
        t_start: float,
        t_end: float,
        read_ms: float,
        writes: list[tuple[float, float]],
        fsyncs: list[tuple[float, float]],
    ) -> PhaseRun:
        assert writes, "no streaming batches recorded (did the merge-copy fast path fire?)"
        first_write = writes[0][0]
        last_write = writes[-1][1]
        compute, write = [], []
        for b, (entry, done) in enumerate(writes):
            prev_end = t_start if b == 0 else writes[b - 1][1]
            fsync_in_gap = sum(fd - fe for fe, fd in fsyncs if prev_end <= fe < entry)
            compute.append((entry - prev_end - fsync_in_gap) * 1000.0)
            write.append((done - entry) * 1000.0)
        setup_flush = sum(fd - fe for fe, fd in fsyncs if fe < first_write)
        durability_flush = sum(fd - fe for fe, fd in fsyncs if fe >= last_write)
        return PhaseRun(
            (t_end - t_start) * 1000.0,
            compute,
            write,
            setup_flush * 1000.0,
            durability_flush * 1000.0,
            read_ms,
        )

    @property
    def n_batches(self) -> int:
        return len(self.write_ms)

    @property
    def compute_total(self) -> float:
        return float(sum(self.compute_ms))

    @property
    def transform_ms(self) -> float:
        """Compute minus gather: the per-column transform (NumPy ufuncs) plus the
        serial loop's per-column Python orchestration -- what evaluating the batch's
        columns on a thread pool could reclaim (the gather is already parallel)."""
        return max(0.0, self.compute_total - self.read_ms)

    @property
    def write_total(self) -> float:
        return float(sum(self.write_ms))

    @property
    def flush_total(self) -> float:
        return self.setup_flush_ms + self.durability_flush_ms

    @property
    def other_ms(self) -> float:
        """Fixed cost outside any measured phase (truncate, mkstemp, rename, header)."""
        return max(0.0, self.wall_ms - self.compute_total - self.write_total - self.flush_total)

    def pipelined_ms(self) -> float:
        """Double-buffer wall ceiling: a background writer overlaps batch b's
        ``pwrite`` with batch b+1's compute.

        ``compute_0 + sum_{b=0}^{B-2} max(write_b, compute_{b+1}) + write_{B-1}``,
        plus the fixed residual (flush + setup + other), which overlap cannot
        touch. One batch cannot overlap, so the ceiling is the serial wall there.
        """
        c, w = self.compute_ms, self.write_ms
        fixed = self.wall_ms - self.compute_total - self.write_total
        total = c[0] + w[-1] + fixed
        for b in range(len(w) - 1):
            total += max(w[b], c[b + 1])
        return total

    def overlap_savings_ms(self) -> float:
        """Wall the ideal pipeline removes: ``sum_b min(compute_{b+1}, write_b)``."""
        return max(0.0, self.wall_ms - self.pipelined_ms())


# --------------------------------------------------------------------------- #
# Workload shapes.
# --------------------------------------------------------------------------- #


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


def _build_filter(store: colstore.ColStoreReader, n_rows: int) -> colstore.ColStoreFrame:
    # rows=<index array>: a gather read leg, passthrough columns, no transform.
    return store[_filter_index(n_rows)].edit()


def _expected_filter(source: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    idx = _filter_index(n_rows)
    return {name: data[idx] for name, data in source.items()}


def _build_transform(store: colstore.ColStoreReader, n_rows: int) -> colstore.ColStoreFrame:
    # rows=None, every column a ufunc -> streaming, one add per column (light compute).
    cf = store.edit()
    for name in store.columns:
        cf[name] = col(name) + 2.0
    return cf


def _expected_transform(source: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    return {name: data + 2.0 for name, data in source.items()}


def _build_deep(store: colstore.ColStoreReader, n_rows: int) -> colstore.ColStoreFrame:
    # rows=None, a sqrt + multiply-add per column (heavy compute relative to the write).
    cf = store.edit()
    for name in store.columns:
        c = col(name)
        cf[name] = np.sqrt(np.abs(c)) * c + c
    return cf


def _expected_deep(source: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    return {name: np.sqrt(np.abs(data)) * data + data for name, data in source.items()}


_SHAPES = (
    Shape(
        "filter",
        "filter 50% (gather read + write)",
        _build_filter,
        _expected_filter,
        lambda n: max(1, n // 2),
    ),
    Shape(
        "transform",
        "transform (+2.0 per column)",
        _build_transform,
        _expected_transform,
        lambda n: n,
    ),
    Shape("deep", "deep transform (sqrt*c + c)", _build_deep, _expected_deep, lambda n: n),
)


# --------------------------------------------------------------------------- #
# Store building.
# --------------------------------------------------------------------------- #


def _build_store(
    directory: Path, rows: int, cols: int, records: int, dtype: str, seed: int
) -> Path:
    path = directory / f"src_r{rows}_c{cols}.cstore"
    columns = testing.make_columns(rows, cols, dtype=dtype, seed=seed)
    testing.write_columns(path, columns, records=min(records, rows)).close()
    return path


def _bytes_per_row(store: colstore.ColStoreReader) -> int:
    return sum(int(np.dtype(dt).itemsize) for dt in store.dtypes.values())


def _write_frame(frame: colstore.ColStoreFrame, out: Path) -> None:
    """The exact call ``ColStoreFrame.write`` makes, minus opening a reader on the
    result -- so the timed region is the streaming fill, not the re-open."""
    selection = frame._resolve_index_selection()
    if selection is None:
        write_dataset_streaming(frame._columns, frame._n_rows, out)
    else:
        write_dataset_streaming(frame._columns, len(selection), out, rows=selection)


# --------------------------------------------------------------------------- #
# Correctness gate (runs, and prints, before any timing).
# --------------------------------------------------------------------------- #


def check_correctness(directory: Path, args: argparse.Namespace) -> None:
    rows = min(_CORRECTNESS_ROWS, _c.scaled_rows(args.rows, args))
    records = min(args.records, _CORRECTNESS_RECORDS, rows)
    source = testing.make_columns(rows, args.cols, dtype=args.dtype, seed=0)
    path = directory / "correctness.cstore"
    testing.write_columns(path, source, records=records).close()
    store = colstore.open(path)
    try:
        for shape in _SHAPES:
            out = directory / f"correctness_{shape.key}.cstore"
            out.unlink(missing_ok=True)
            _write_frame(shape.build(store, rows), out)
            written = colstore.open(out)
            try:
                got = written.dict()
                expected = shape.expected(source, rows)
                for name in store.columns:
                    _c.check_equal(got[name], expected[name], f"{shape.key}:{name}")
            finally:
                written.close()
                out.unlink(missing_ok=True)
    finally:
        store.close()
    print("  CORRECTNESS OK (write output matches the direct transform for every shape)\n")


# --------------------------------------------------------------------------- #
# Phase profile.
# --------------------------------------------------------------------------- #


def _profile_shape(
    store: colstore.ColStoreReader,
    shape: Shape,
    n_rows: int,
    out: Path,
    *,
    repeat: int,
    warmup: int,
) -> PhaseRun:
    """Best-of (min-wall) decomposed run of one shape's streaming write."""
    best: PhaseRun | None = None
    for i in range(warmup + repeat):
        frame = shape.build(store, n_rows)  # built outside the timed region
        out.unlink(missing_ok=True)
        t0 = _recorder.arm()
        try:
            _write_frame(frame, out)
        finally:
            _recorder.disarm()
        t_end = time.perf_counter()
        run = PhaseRun.analyze(
            t0, t_end, _recorder.read_s * 1000.0, _recorder.writes, _recorder.fsyncs
        )
        if i >= warmup and (best is None or run.wall_ms < best.wall_ms):
            best = run
    assert best is not None
    return best


def run_bench(directory: Path, args: argparse.Namespace) -> None:
    rows = _c.scaled_rows(args.rows, args)
    path = _build_store(directory, rows, args.cols, args.records, args.dtype, seed=0)
    store = colstore.open(path)
    try:
        bpr = _bytes_per_row(store)
        budget = config.get_default_memory_budget()
        print("Environment:")
        print(f"  rows={rows:,}  cols={args.cols}  records={args.records}  dtype={args.dtype}")
        print(
            f"  row bytes={bpr}  memory budget={budget / 2**20:.0f} MiB"
            f"  gather thread cap={config.get_gather_thread_cap()}"
        )
        print(f"  repeat={args.repeat}  warmup={args.warmup}  store dir={directory}\n")

        out = directory / "out.cstore"
        runs: list[tuple[Shape, PhaseRun, int, float]] = []
        for shape in _SHAPES:
            run = _profile_shape(store, shape, rows, out, repeat=args.repeat, warmup=args.warmup)
            n_out = shape.out_rows(rows)
            out_gib = n_out * bpr / 2**30
            runs.append((shape, run, n_out, out_gib))
            _print_shape(shape, run, n_out, out_gib)
        out.unlink(missing_ok=True)
        _print_summary(runs)
    finally:
        store.close()


def _pct(part: float, whole: float) -> str:
    return f"{part / whole:5.1%}" if whole else "  n/a"


def _print_shape(shape: Shape, run: PhaseRun, n_out: int, out_gib: float) -> None:
    wall = run.wall_ms

    def line(label: str, ms: float) -> None:
        print(f"    {label:<25} = {ms:8.1f} ms ({_pct(ms, wall)})")

    print(f"=== {shape.label} ===")
    print(f"  batches={run.n_batches}  written={out_gib:.2f} GiB ({n_out:,} rows)")
    print(f"  wall={wall:8.1f} ms")
    line("compute total", run.compute_total)
    line("  - read   (gather)", run.read_ms)
    line("  - transform/per-col", run.transform_ms)
    line("write   (pwrite -> cache)", run.write_total)
    line("durability flush (fsync)", run.durability_flush_ms)
    line("setup flush + other", run.setup_flush_ms + run.other_ms)
    print(
        f"  lever 1 -- I/O overlap ceiling:  savings={run.overlap_savings_ms():7.1f} ms "
        f"({_pct(run.overlap_savings_ms(), wall)} of wall)"
    )
    print(
        f"  lever 2 -- parallel column loop: reclaimable={run.transform_ms:7.1f} ms "
        f"({_pct(run.transform_ms, wall)} of wall)"
    )
    if wall > 0:
        print(
            f"  throughput: {n_out / (wall / 1000.0) / 1e6:6.1f} M rows/s"
            f"   {out_gib / (wall / 1000.0):5.2f} GiB/s written"
        )
    print()


def _print_summary(runs: list[tuple[Shape, PhaseRun, int, float]]) -> None:
    print("=" * 90)
    print(
        f"{'shape':<34}{'read%':>8}{'xform%':>8}{'write%':>8}{'flush%':>8}"
        f"{'L1 overlap%':>14}{'L2 xform%':>12}"
    )
    print("-" * 90)
    for shape, run, _n_out, _gib in runs:
        wall = run.wall_ms
        print(
            f"{shape.label:<34}"
            f"{run.read_ms / wall:>7.0%}{run.transform_ms / wall:>8.0%}"
            f"{run.write_total / wall:>8.0%}{run.flush_total / wall:>8.0%}"
            f"{run.overlap_savings_ms() / wall:>13.0%}{run.transform_ms / wall:>12.0%}"
        )
    print("=" * 90)
    print(
        "\nTwo candidate levers, each as a share of wall:\n"
        "  L1 overlap%  = ceiling a perfect read/write double-buffer removes\n"
        "                 (sum_b min(compute_{b+1}, write_b)); bounded by per-batch\n"
        "                 pwrite, and zero if the write is flush-bound or one-phase.\n"
        "  L2 xform%    = compute minus gather (per-column ufunc transform + the serial\n"
        "                 loop's per-column overhead); the per-batch column loop is\n"
        "                 serial, so evaluating columns on a GIL-releasing thread pool\n"
        "                 could reclaim most of it (bounded by cores/bandwidth).\n"
        "Both are ceilings; realized wins are lower. read% (gather) is already\n"
        "OpenMP-parallel and bandwidth-bound -- little headroom there. Decide direction\n"
        "from the node numbers, where the durability flush (Lustre) may dominate."
    )


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


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

    # Profile the serial per-batch phases: force the pwrite path, and disable the
    # background-writer overlap (shipped in Round 6) so each batch's compute and write
    # run sequentially and the single per-batch _run_copy_jobs call cleanly separates
    # them. With overlap on, the write runs concurrently with the next batch's compute
    # and the phase decomposition no longer holds.
    previous_override = _fmt._STREAMING_FILL_OVERRIDE
    previous_overlap = _fmt._STREAMING_OVERLAP_WRITE
    _fmt._STREAMING_FILL_OVERRIDE = "pwrite"
    _fmt._STREAMING_OVERLAP_WRITE = False
    _recorder.install()
    try:
        if args.tmpdir is not None:
            directory = Path(args.tmpdir)
            directory.mkdir(parents=True, exist_ok=True)
            if not args.skip_correctness:
                check_correctness(directory, args)
            if not args.skip_bench:
                run_bench(directory, args)
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                if not args.skip_correctness:
                    check_correctness(directory, args)
                if not args.skip_bench:
                    run_bench(directory, args)
    finally:
        _recorder.restore()
        _fmt._STREAMING_FILL_OVERRIDE = previous_override
        _fmt._STREAMING_OVERLAP_WRITE = previous_overlap


if __name__ == "__main__":
    main()
