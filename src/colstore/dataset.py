"""The colstore data handle: one or more ``.cstore`` files as a single table.

:class:`ColStoreDataset` is the general-purpose handle for working with colstore
data. It is a thin coordinator that *holds* per-file
:class:`~colstore.reader.ColStoreReader` children and presents them as one
logical table: every read is decomposed against the children's cumulative row
offsets, dispatched to the relevant child, and stitched back together. The
dataset owns no storage kernels of its own, so it inherits whatever the
single-file reader already does well, and it satisfies the shared read interface
(:class:`~colstore._base._ReaderBase`) -- so indexing, the lazy views, and the
whole-store materializers all work against it unchanged.

A dataset is built from any mix of file paths (which it opens and *owns*) and
already-open readers or datasets (which it *borrows* and leaves open). It may be
empty and grown later with :meth:`append` or ``|=``. That generality is
deliberate: the dataset is the surface the planned machine-learning features --
index and shuffled sampling, train/val/test splits, batch iteration -- are built
on, because those are operations over the global row-index space and the read
seam, both of which the dataset already owns.

This module implements every read selector across multiple files. Contiguous
selectors -- ``None`` (whole table), a scalar row, and a ``step >= 1`` slice --
decompose against the children's cumulative row offsets. Fancy integer arrays
and boolean masks decompose too: a fancy gather groups the requested rows by
child, gathers each child's rows with one local call, and scatters the results
back into the requested order; a boolean mask is split at the file boundaries
and stitched in file order. A negative-step slice is order-bearing, so it routes
through the same gather path. A single-file dataset short-circuits straight to
its child, so it matches the bare reader on the hot path.

Cross-file zero-copy reads remain unavailable and raise :class:`ValueError` --
the permanent contract, since the files are not contiguous in memory: a whole
multi-file view, a slice that spans files, a fancy/boolean selection, and a
reversed selection all require a copying gather.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from . import config
from ._base import _indices_are_sorted, _ReaderBase
from ._paths import expand_glob
from ._types import Source
from .reader import ColStoreReader
from .shards import list_shards

# Segment ids are recorded as int32 bins for cross-column reuse; above this many
# segments the dict path keeps an independent pass per column instead.
_INT32_MAX = (1 << 31) - 1

# One column's native multi-file gather table: cumulative segment start rows and
# each segment's absolute byte base (see _native_segment_table).
SegmentTable: TypeAlias = tuple[NDArray[np.int64], NDArray[np.int64]]


def _uniform_segment_rows(starts: NDArray[np.int64]) -> int | None:
    """``rows_per_segment`` if ``starts`` describes a uniform grid, else ``None``.

    A uniform grid has every segment of equal row count except possibly the
    global-last (which is no larger) -- the layout under which the per-index
    binary search over the segment table collapses to ``s = idx / rows_per_segment``.
    Cheap O(n_segments) vectorized pass; returns ``None`` for fewer than two
    segments, where there is no search to amortize.
    """
    n_segments = starts.shape[0] - 1
    if n_segments < 2:
        return None
    seg_rows = np.diff(starts)
    rows = int(seg_rows[0])
    if rows <= 0:
        return None
    if not bool(np.all(seg_rows[:-1] == rows)):
        return None
    if int(seg_rows[-1]) > rows:
        return None
    return rows


# One file's contribution to a contiguous read: the half-open output row range
# ``[out_lo, out_hi)`` it fills, its child reader, and the per-file selector
# (``None``, a slice, or a boolean sub-mask) -- see _fill_contiguous_columns.
Region: TypeAlias = tuple[int, int, ColStoreReader, Any]

_VIEW_FANCY_MESSAGE = (
    "A zero-copy read is available only for contiguous selectors; fancy and "
    "boolean selection require a copying gather. Use copy=True."
)
_VIEW_REVERSED_MESSAGE = (
    "A zero-copy view of a reversed (negative-step) selection is unavailable; it "
    "requires a copying gather. Use copy=True, or select within a single file."
)
_VIEW_WHOLE_MESSAGE = (
    "A zero-copy view of a whole multi-file dataset is unavailable: the files "
    "are not contiguous in memory. Select within a single file, or use copy=True."
)
_VIEW_SPAN_MESSAGE = (
    "A zero-copy slice that spans multiple files is unavailable: the files are "
    "not contiguous in memory. Select within a single file, or use copy=True."
)


class ColStoreDataset(_ReaderBase):
    """One or more same-schema ``.cstore`` files presented as one logical table.

    Construct from any mix of paths and open readers/datasets, or empty::

        ColStoreDataset()                 # empty; grow later with append()/|=
        ColStoreDataset(path)             # open and own one file
        ColStoreDataset([path])           # same
        ColStoreDataset(reader)           # borrow one open reader
        ColStoreDataset([p, reader, ds])  # paths owned, readers/datasets borrowed
        ColStoreDataset("run_*.cstore")   # glob: own every match, numeric order
        ColStoreDataset("trades/")        # a directory: its .cstore shards, in order

    Paths are opened and *owned* (closed on :meth:`close`); readers and datasets
    are *borrowed* (left open). Datasets are flattened to their children. A path
    *string* may be a glob (``*``, ``?``, ``[``; ``**`` recursive), expanded to
    its matches in numeric order; a pattern matching no files raises
    :class:`FileNotFoundError`. A path naming a *directory* expands to its
    ``.cstore`` shards in numeric order -- the managed dataset :func:`append` /
    :func:`appender` write to -- so a directory and a list of files compose the
    same way (an empty directory is an empty dataset). The public read surface
    matches :class:`~colstore.reader.ColStoreReader`.

    The native multi-file gather's per-column segment table (:meth:`_native_segment_table`)
    depends only on the children and the column, not on the rows a read requests,
    so it is **memoized per column** in ``_segment_table_cache`` and reused across
    reads. Rebuilding it dominates the per-read cost of the small, repeated fancy
    reads that index sampling over many files issues -- the table-build is
    ``O(n_files)`` while the kernel touches only the sampled rows -- so the memo is
    what keeps that workload fast. The cache is cleared whenever the children
    change (:meth:`_rebuild_offsets`), so it never serves a stale table.

    Whether that table is a *uniform grid* -- every segment the same row count
    except possibly the global-last -- is likewise a property of the children, not
    the read, so it too is memoized (``_uniform_grid_rows``, reset on the same
    child change). On a uniform grid the unsorted gather divides instead of
    searching (:meth:`_uniform_segment_grid`).
    """

    def __init__(
        self,
        sources: Source | Sequence[Source] | None = None,
        **reader_kwargs: Any,
    ) -> None:
        self._children: list[ColStoreReader] = []
        self._owned: list[bool] = []
        self._column_dtypes: dict[str, np.dtype[Any]] = {}
        self._offsets: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self._n_rows = 0
        self._segment_table_cache: dict[str, SegmentTable | None] = {}
        self._uniform_grid_rows: int | None = None
        self._uniform_grid_known = False
        self._closed = False
        if sources is not None:
            self.append(sources, **reader_kwargs)

    # ---- Construction helpers ------------------------------------------

    @staticmethod
    def _as_source_list(sources: Any) -> list[Any]:
        """Normalize a source argument to a flat list of individual sources."""
        if isinstance(sources, (str, os.PathLike, ColStoreReader, ColStoreDataset)):
            return [sources]
        if isinstance(sources, (list, tuple)):
            return list(sources)
        raise TypeError(
            f"ColStoreDataset accepts a path, a reader, a dataset, or a list/tuple "
            f"of them; got {type(sources).__name__}."
        )

    def _coerce_to_children(
        self, sources: Any, reader_kwargs: dict[str, Any]
    ) -> list[tuple[ColStoreReader, bool]]:
        """Resolve sources to ``(reader, owned)`` pairs, opening any paths.

        Readers opened here are closed again if any later open fails, so a
        failure opens no leaked files.
        """
        pairs: list[tuple[ColStoreReader, bool]] = []
        opened: list[ColStoreReader] = []
        try:
            for item in self._as_source_list(sources):
                if isinstance(item, (str, os.PathLike)):
                    # A directory is a managed shard dataset (its ``.cstore`` files,
                    # in order); anything else is a literal path or a glob pattern.
                    paths = list_shards(item) if os.path.isdir(item) else expand_glob(item)
                    for path in paths:
                        reader = ColStoreReader(path, **reader_kwargs)
                        opened.append(reader)
                        pairs.append((reader, True))
                elif isinstance(item, ColStoreDataset):
                    pairs.extend((child, False) for child in item._children)
                elif isinstance(item, ColStoreReader):
                    pairs.append((item, False))
                else:
                    raise TypeError(
                        f"ColStoreDataset sources must be a path, a reader, or a "
                        f"dataset; got {type(item).__name__}."
                    )
        except BaseException:
            for reader in opened:
                reader.close()
            raise
        return pairs

    def _validate_against(
        self, reader: ColStoreReader, reference: dict[str, np.dtype[Any]]
    ) -> None:
        """Require ``reader`` to match the reference schema (names, order, dtype)."""
        reference_names = list(reference)
        names = list(reader._column_dtypes)
        if names != reference_names:
            raise ValueError(
                f"Schema mismatch: {reader.path} has columns {names}, but the "
                f"dataset schema is {reference_names} (same names and order required)."
            )
        for name in reference_names:
            child_dtype = reader._column_dtypes[name].str
            reference_dtype = reference[name].str
            if child_dtype != reference_dtype:
                raise ValueError(
                    f"Schema mismatch: {reader.path} column {name!r} has dtype "
                    f"'{child_dtype}', but the dataset schema has '{reference_dtype}'."
                )

    def _rebuild_offsets(self) -> None:
        self._segment_table_cache.clear()
        self._uniform_grid_known = False
        self._offsets = np.zeros(len(self._children) + 1, dtype=np.int64)
        if self._children:
            self._offsets[1:] = np.cumsum([child.n_rows for child in self._children])
        self._n_rows = int(self._offsets[-1])

    def append(self, source: Source | Sequence[Source], **reader_kwargs: Any) -> ColStoreDataset:
        """Grow the dataset in place; return ``self`` so calls can be chained.

        ``source`` is anything the constructor accepts: a path -- a file, a glob,
        or a directory of shards -- (opened and owned), a reader or dataset
        (borrowed), or a list/tuple mixing them. The first child establishes the
        schema; later children must match it.
        A schema mismatch leaves the dataset unchanged and closes anything this
        call opened.
        """
        self._check_open()
        pairs = self._coerce_to_children(source, reader_kwargs)
        try:
            reference: dict[str, np.dtype[Any]] | None = (
                dict(self._column_dtypes) if self._column_dtypes else None
            )
            for reader, _owned in pairs:
                if reference is None:
                    reference = dict(reader._column_dtypes)
                else:
                    self._validate_against(reader, reference)
        except BaseException:
            for reader, owned in pairs:
                if owned:
                    reader.close()
            raise
        if not self._column_dtypes and pairs:
            self._column_dtypes = dict(pairs[0][0]._column_dtypes)
        for reader, owned in pairs:
            self._children.append(reader)
            self._owned.append(owned)
        self._rebuild_offsets()
        return self

    # ---- Properties ----------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Total number of rows across all files (0 when empty)."""
        return self._n_rows

    @property
    def paths(self) -> tuple[Path, ...]:
        """The child files' paths, in order (empty for borrowed-only/empty)."""
        return tuple(child.path for child in self._children)

    @property
    def backend(self) -> str:
        """Gather backend, taken from the first file, else the configured default."""
        if self._children:
            return self._children[0].backend
        return config.get_default_backend()

    @property
    def max_workers(self) -> int:
        """Multi-column thread-pool size, from the first file, else the default."""
        if self._children:
            return self._children[0].max_workers
        return config.get_max_workers()

    @property
    def needs_compaction(self) -> bool:
        """True while the data is spread across more than one file."""
        return len(self._children) > 1

    def __repr__(self) -> str:
        preview = self.columns[:5]
        suffix = "..." if len(self._column_dtypes) > len(preview) else ""
        return (
            f"ColStoreDataset(files={len(self._children)}, "
            f"shape={self.shape}, columns={preview}{suffix})"
        )

    # ---- Row-locating helpers ------------------------------------------

    def _locate(self, global_index: int) -> tuple[int, int]:
        """Map a global row index to ``(file_index, local_index)``."""
        file_index = int(np.searchsorted(self._offsets[1:], global_index, side="right"))
        return file_index, global_index - int(self._offsets[file_index])

    def _slice_parts(self, row_slice: slice) -> list[tuple[int, slice]]:
        """Decompose a global ``step >= 1`` slice into per-file local slices.

        Each local slice preserves the global step *phase*, so concatenating the
        per-file reads in file order reproduces the global strided selection.
        Negative-step slices are order-bearing and routed through the gather
        path by the callers, so this method only ever sees ``step >= 1``.
        """
        start, stop, step = row_slice.indices(self._n_rows)
        parts: list[tuple[int, slice]] = []
        for file_index in range(len(self._children)):
            lo = int(self._offsets[file_index])
            hi = int(self._offsets[file_index + 1])
            if lo >= hi:
                continue  # a zero-row file contributes nothing
            last_excl = min(hi, stop)
            first = start if start >= lo else start + ((lo - start + step - 1) // step) * step
            if first >= last_excl:
                continue
            parts.append((file_index, slice(first - lo, last_excl - lo, step)))
        return parts

    # ---- Parallel fill across files ------------------------------------
    #
    # Every multi-file read preallocates one output and fills disjoint regions of
    # it concurrently, so the fills need no locking. A contiguous read (whole
    # table or a forward slice) of native, single-record files copies through the
    # parallel-copy kernel: each file's bytes become a run, and one kernel call
    # splits the runs across the gather thread budget -- so a few large files no
    # longer copy single-threaded one at a time, the way the single-file reader
    # already splits a large contiguous copy. A region the kernel cannot take (a
    # strided view, a multi-record or non-native file, a boolean mask) fills on a
    # thread pool instead, one job per region at thread_cap=1 so the total thread
    # count stays in the single-file envelope. The fancy gather still scatters
    # through _fill_one (see _fancy_one / _fancy_many).

    def _read_budget(self) -> int:
        return max(1, config.get_gather_thread_cap())

    def _run_fill_jobs(self, jobs: list[Callable[[], None]]) -> None:
        if len(jobs) <= 1:
            for job in jobs:
                job()
            return
        workers = min(len(jobs), self._read_budget())
        if workers <= 1:
            for job in jobs:
                job()
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for future in [executor.submit(job) for job in jobs]:
                future.result()

    def _portion_source(self, child: ColStoreReader, column_name: str, sub: Any) -> NDArray[Any]:
        """A readable array for one file's portion of a column.

        A zero-copy view when the child can give one (native single-record,
        contiguous selector), so the value is copied into the output exactly
        once; otherwise a single-threaded gather (file-level parallelism owns the
        threads, so the per-file read does not also spin up its own).
        """
        try:
            return child._view_one(column_name, sub)
        except ValueError:
            return child._gather_one(column_name, sub, thread_cap=1)

    def _fill_one(
        self, out: NDArray[Any], lo: int, hi: int, child: ColStoreReader, column_name: str, sub: Any
    ) -> Callable[[], None]:
        def job() -> None:
            out[lo:hi] = self._portion_source(child, column_name, sub)

        return job

    def _offset_regions(self) -> list[Region]:
        """Whole-read regions: each non-empty file owns its global row range."""
        regions: list[Region] = []
        for file_index, child in enumerate(self._children):
            lo, hi = int(self._offsets[file_index]), int(self._offsets[file_index + 1])
            if lo < hi:
                regions.append((lo, hi, child, None))
        return regions

    def _contiguous_regions(self, parts: list[tuple[int, Any]], lengths: list[int]) -> list[Region]:
        """Slice/mask regions: each file's portion mapped to its output write offset."""
        regions: list[Region] = []
        write = 0
        for (file_index, sub), length in zip(parts, lengths, strict=True):
            regions.append((write, write + length, self._children[file_index], sub))
            write += length
        return regions

    def _fill_contiguous_columns(
        self, out: dict[str, NDArray[Any]], column_names: list[str], regions: list[Region]
    ) -> None:
        """Fill each column's disjoint output regions from the files' bytes.

        Each native, single-record, contiguously-viewable file portion becomes a
        run of the parallel-copy kernel, which splits the column's bytes across
        the gather thread budget in one pass. A portion the kernel cannot take --
        a strided view, or a multi-record/non-native file that needs a gather --
        fills on the side, single-threaded. With the kernel unavailable every
        portion takes that side path, so the read still completes.
        """
        from . import kernels

        use_kernel = kernels.cpp_available()
        cap = config.get_gather_thread_cap()
        for name in column_names:
            target = out[name]
            itemsize = target.dtype.itemsize
            src: list[int] = []
            dst: list[int] = []
            lengths: list[int] = []
            held: list[NDArray[Any]] = []
            jobs: list[Callable[[], None]] = []
            for out_lo, out_hi, child, sub in regions:
                try:
                    view = child._view_one(name, sub)
                except ValueError:
                    jobs.append(self._gather_fill_job(target, out_lo, out_hi, child, name, sub))
                    continue
                if use_kernel and view.flags["C_CONTIGUOUS"]:
                    src.append(view.ctypes.data)
                    dst.append(out_lo * itemsize)
                    lengths.append((out_hi - out_lo) * itemsize)
                    held.append(view)
                else:
                    jobs.append(self._copy_view_job(target, out_lo, out_hi, view))
            if src:
                from . import _gather as _cpp_module  # type: ignore[attr-defined]

                _cpp_module.parallel_copy_runs(
                    target,
                    np.array(src, dtype=np.int64),
                    np.array(dst, dtype=np.int64),
                    np.array(lengths, dtype=np.int64),
                    cap,
                )
                held.clear()  # release the source views now the kernel has read them
            self._run_fill_jobs(jobs)

    @staticmethod
    def _copy_view_job(
        out: NDArray[Any], out_lo: int, out_hi: int, view: NDArray[Any]
    ) -> Callable[[], None]:
        def job() -> None:
            out[out_lo:out_hi] = view

        return job

    def _gather_fill_job(
        self,
        out: NDArray[Any],
        out_lo: int,
        out_hi: int,
        child: ColStoreReader,
        name: str,
        sub: Any,
    ) -> Callable[[], None]:
        def job() -> None:
            out[out_lo:out_hi] = child._gather_one(name, sub, thread_cap=1)

        return job

    def _mask_parts(self, mask: NDArray[np.bool_]) -> list[tuple[int, NDArray[np.bool_]]]:
        """Split a global boolean mask into per-file sub-masks, skipping empties."""
        parts: list[tuple[int, NDArray[np.bool_]]] = []
        for file_index in range(len(self._children)):
            lo = int(self._offsets[file_index])
            hi = int(self._offsets[file_index + 1])
            sub = mask[lo:hi]
            if sub.any():
                parts.append((file_index, sub))
        return parts

    def _contiguous_lengths(self, parts: list[tuple[int, Any]], is_mask: bool) -> list[int]:
        """Per-part output lengths for a slice (range size) or mask (popcount)."""
        if is_mask:
            return [int(sub.sum()) for _, sub in parts]
        return [
            len(range(*sub.indices(self._children[file_index].n_rows))) for file_index, sub in parts
        ]

    def _sorted_blocks(
        self, indices: NDArray[np.int64]
    ) -> tuple[NDArray[np.intp], list[tuple[int, int, int, NDArray[np.int64]]]]:
        """Group global indices by owning file with a single sort.

        Sorting the indices ascending turns each file's rows into one contiguous
        run (the files are offset-ordered), whose bounds in the sorted array fall
        out of searchsorting the file boundaries -- replacing a per-file scan of
        the whole index array with one sort. Returns the argsort permutation (to
        un-sort the gathered values back into the requested order) and, per
        non-empty file, the sorted half-open range ``[lo, hi)`` and the
        file-local, ascending row indices to gather.
        """
        order = np.argsort(indices)
        sorted_indices = indices[order]
        boundaries = np.searchsorted(sorted_indices, self._offsets)
        blocks: list[tuple[int, int, int, NDArray[np.int64]]] = []
        for file_index in range(len(self._children)):
            lo, hi = int(boundaries[file_index]), int(boundaries[file_index + 1])
            if lo < hi:
                local = sorted_indices[lo:hi] - int(self._offsets[file_index])
                blocks.append((file_index, lo, hi, local))
        return order, blocks

    def _native_segment_table(self, column_name: str) -> SegmentTable | None:
        """Global segment table across all files for the native multi-file gather.

        Returns the column's ``(start_rows, segment_base)`` stitch, or ``None``
        -- the caller then takes the portable sort-once path -- when the kernel
        is unavailable, the element size is unsupported, or any file cannot
        supply native segments (e.g. a non-native on-disk dtype).

        The stitch is row-independent, so it is memoized per column (see the
        class docstring). The cheap availability and element-size guards stay
        live -- re-evaluated on every call -- so the memo holds only the
        structure-dependent stitch and a build under one ``cpp_available()``
        regime is never replayed under another.
        """
        from . import kernels

        if not kernels.cpp_available():
            return None
        itemsize = self._column_dtypes[column_name].itemsize
        if itemsize not in (1, 2, 4, 8):
            return None
        cache = self._segment_table_cache
        if column_name not in cache:
            cache[column_name] = self._stitch_native_segment_table(column_name, itemsize)
        return cache[column_name]

    def _stitch_native_segment_table(self, column_name: str, itemsize: int) -> SegmentTable | None:
        """Build one column's global segment table from the children's local ones.

        Stitches each file's local table
        (:meth:`ColStoreReader._column_segment_table`) into one global table by
        folding the file's row offset into the segment start rows and bases, so
        a global index reads at ``segment_base[s] + idx * itemsize``. Returns
        ``None`` when a file cannot supply native segments or every file is empty.
        """
        start_parts: list[NDArray[np.int64]] = []
        base_parts: list[NDArray[np.int64]] = []
        for file_index, child in enumerate(self._children):
            if child.n_rows == 0:
                continue
            try:
                local_starts, local_base = child._column_segment_table(column_name)
            except (ValueError, AttributeError):
                return None
            offset = int(self._offsets[file_index])
            start_parts.append(local_starts[:-1] + offset)
            base_parts.append(local_base - offset * itemsize)
        if not base_parts:
            return None
        starts = np.concatenate([*start_parts, np.array([self._n_rows], dtype=np.int64)])
        return starts.astype(np.int64, copy=False), np.concatenate(base_parts)

    def _column_disk_runs(self, column_name: str) -> list[tuple[Path, int, int]]:
        """On-disk byte runs for one column across all files, in global row order.

        Concatenates each child's :meth:`ColStoreReader._column_disk_runs` in
        file order, which is the global row order, so the runs reproduce the
        column's image as the dataset would gather it. Propagates the child's
        ``ValueError`` for a non-native dtype, so the merge-copy caller falls
        back to the materializing write.
        """
        self._check_open()
        self._require_columns([column_name])
        runs: list[tuple[Path, int, int]] = []
        for child in self._children:
            if child.n_rows:
                runs.extend(child._column_disk_runs(column_name))
        return runs

    def _column_chunks(self, column_name: str) -> list[NDArray[Any]]:
        """Zero-copy native views of one column across all files, in row order.

        Concatenates each child's :meth:`ColStoreReader._column_chunks`, so each
        chunk aliases its own file's mapping and keeps it alive. Propagates a
        child's ``ValueError`` for a non-native dtype, so the Arrow caller falls
        back to a materializing gather.
        """
        self._check_open()
        self._require_columns([column_name])
        chunks: list[NDArray[Any]] = []
        for child in self._children:
            if child.n_rows:
                chunks.extend(child._column_chunks(column_name))
        return chunks

    def _uniform_segment_grid(self, starts: NDArray[np.int64]) -> int | None:
        """The common segment row count if the segment table is a uniform grid,
        else ``None``.

        ``starts`` is the global row partition, identical for every column, so the
        grid test is column-independent and memoized once across the dataset's
        lifetime (reset when files are added). See :func:`_uniform_segment_rows`.
        """
        if not self._uniform_grid_known:
            self._uniform_grid_rows = _uniform_segment_rows(starts)
            self._uniform_grid_known = True
        return self._uniform_grid_rows

    def _native_gather(
        self,
        out: NDArray[Any],
        indices: NDArray[np.int64],
        table: SegmentTable,
        indices_sorted: bool,
    ) -> None:
        """Fill ``out`` with one native pass: a cursor walk if the indices are
        non-decreasing, the division-binning kernel on a uniform grid, otherwise
        the searching kernel."""
        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        starts, segment_base = table
        cap = config.get_gather_thread_cap()
        if indices_sorted:
            _cpp_module.gather_segment_sorted(indices, out, starts, segment_base, cap, -1)
            return
        rows_per_segment = self._uniform_segment_grid(starts)
        if rows_per_segment is not None:
            _cpp_module.gather_segment_uniform(
                indices, out, rows_per_segment, segment_base, cap, -1
            )
        else:
            _cpp_module.gather_segment(indices, out, starts, segment_base, cap, -1)

    def _native_gather_many(
        self,
        out: dict[str, NDArray[Any]],
        column_names: list[str],
        indices: NDArray[np.int64],
        tables: list[SegmentTable | None],
    ) -> None:
        """Native dict gather.

        Sorted indices take the per-column cursor walk -- a walk has no search to
        amortize, and its within-segment access is sequential. Unsorted reads of
        two or more columns compute the (column-independent) segment once for the
        first column and replay it per column with ``gather_segment_withbins``, the
        same amortization the single-file multi-column path gives across records.
        That first column divides (``gather_segment_uniform_bins``) on a uniform
        grid and otherwise searches (``gather_segment_bins``). A single column, or a
        segment count past the int32 bin range, takes an independent pass per
        column.
        """
        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        indices_sorted = _indices_are_sorted(indices)
        first = tables[0]
        assert first is not None  # the caller passes only all-native tables
        first_starts, first_base = first
        if indices_sorted or len(column_names) == 1 or first_base.shape[0] > _INT32_MAX:
            for name, table in zip(column_names, tables, strict=True):
                assert table is not None
                self._native_gather(out[name], indices, table, indices_sorted)
            return
        cap = config.get_gather_thread_cap()
        bins = np.empty(len(indices), dtype=np.int32)
        rows_per_segment = self._uniform_segment_grid(first_starts)
        if rows_per_segment is not None:
            _cpp_module.gather_segment_uniform_bins(
                indices, out[column_names[0]], bins, rows_per_segment, first_base, cap, -1
            )
        else:
            _cpp_module.gather_segment_bins(
                indices, out[column_names[0]], bins, first_starts, first_base, cap, -1
            )
        for name, table in zip(column_names[1:], tables[1:], strict=True):
            assert table is not None
            _, segment_base = table
            _cpp_module.gather_segment_withbins(indices, out[name], bins, segment_base, cap, -1)

    def _fancy_one(
        self,
        column_name: str,
        indices: NDArray[np.int64],
        out: NDArray[Any] | None = None,
        indices_sorted: bool | None = None,
    ) -> NDArray[Any]:
        """Gather arbitrary global rows across files, preserving requested order.

        Uses the native fused multi-file kernel (one pass over the indices, each
        binned to its file/record segment) when available. The portable fallback
        sorts the indices once to group them into one contiguous block per file,
        fills a sorted buffer concurrently (disjoint regions, no locking), then
        un-sorts into the requested order with one scatter. ``indices_sorted`` is
        the selector's precomputed order (a multi-column read supplies it once);
        ``None`` resolves it here.
        """
        dst = (
            out
            if out is not None
            else np.empty(len(indices), dtype=self._native_dtype(column_name))
        )
        if indices.size == 0:
            return dst
        table = self._native_segment_table(column_name)
        if table is not None:
            sorted_selector = (
                indices_sorted if indices_sorted is not None else _indices_are_sorted(indices)
            )
            self._native_gather(dst, indices, table, sorted_selector)
            return dst
        order, blocks = self._sorted_blocks(indices)
        buffer = np.empty(len(indices), dtype=self._native_dtype(column_name))
        jobs = [
            self._fill_one(buffer, lo, hi, self._children[file_index], column_name, local)
            for file_index, lo, hi, local in blocks
        ]
        self._run_fill_jobs(jobs)
        dst[order] = buffer
        return dst

    def _fancy_many(
        self, column_names: list[str], indices: NDArray[np.int64]
    ) -> dict[str, NDArray[Any]]:
        """Multi-column :meth:`_fancy_one`.

        Takes the native path when every requested column can use it (mixing the
        two paths within one read is avoided). The portable fallback shares one
        sort across all columns and reuses a single per-column buffer, so peak
        memory stays at one column's worth rather than scaling with the count.
        """
        out = {
            name: np.empty(len(indices), dtype=self._native_dtype(name)) for name in column_names
        }
        if indices.size == 0:
            return out
        tables = [self._native_segment_table(name) for name in column_names]
        if all(table is not None for table in tables):
            self._native_gather_many(out, column_names, indices, tables)
            return out
        order, blocks = self._sorted_blocks(indices)
        for name in column_names:
            buffer = np.empty(len(indices), dtype=self._native_dtype(name))
            jobs = [
                self._fill_one(buffer, lo, hi, self._children[file_index], name, local)
                for file_index, lo, hi, local in blocks
            ]
            self._run_fill_jobs(jobs)
            out[name][order] = buffer
        return out

    def _mask_native(
        self, column_names: list[str], mask: NDArray[np.bool_]
    ) -> dict[str, NDArray[Any]] | None:
        """Gather a boolean mask with the native segment mask kernel, or ``None``.

        Returns ``None`` to decline -- the caller then takes the per-file path --
        when the selection is too sparse for the kernel's full-mask scan to pay
        off (below :func:`config.get_multifile_mask_density_gate`), or when any
        column cannot supply a native segment table (the same gate the fancy path
        uses: extension unavailable, unsupported itemsize, or a non-native on-disk
        dtype). The kernel scans the mask once for lock-free output offsets and
        once to gather, so the output is byte-identical to numpy mask indexing.
        """
        if self._n_rows == 0:
            return None
        selected = int(np.count_nonzero(mask))
        if selected < self._n_rows * config.get_multifile_mask_density_gate():
            return None
        tables = [self._native_segment_table(name) for name in column_names]
        if any(table is None for table in tables):
            return None

        from . import _gather as _cpp_module  # type: ignore[attr-defined]

        contiguous_mask = np.ascontiguousarray(mask)
        out: dict[str, NDArray[Any]] = {}
        for name, table in zip(column_names, tables, strict=True):
            assert table is not None  # filtered above
            segment_starts_rows, segment_base = table
            column = np.empty(selected, dtype=self._native_dtype(name))
            _cpp_module.gather_segment_mask(
                contiguous_mask, column, segment_starts_rows, segment_base
            )
            out[name] = column
        return out

    def _keeps_boolean_mask(self, selected: int, n_rows: int) -> bool:
        """Whether a base boolean mask should reach the gather as a mask, not indices.

        True for a multi-file dataset whose selection is dense enough to take the
        native mask kernel (the same density gate :meth:`_mask_native` applies); the
        gather then makes the final native-vs-per-file choice. A sparse mask is lowered
        to indices by the caller, where the per-file path serves it well. A single-file
        dataset short-circuits to its child, so it does not keep the mask here.
        """
        if len(self._children) <= 1:
            return False
        return selected >= n_rows * config.get_multifile_mask_density_gate()

    # ---- Copying seam --------------------------------------------------

    def _classify_rows(self, row_indexer: Any) -> tuple[str, Any]:
        """Map a row selector to ``(kind, payload)`` for the gather dispatch.

        The classification is identical for single- and multi-column reads --
        only the terminal materialization differs -- so it lives here once.
        ``kind`` is ``"whole"`` (payload ``None``), ``"scalar"`` (payload
        ``(file_index, local)``), ``"fancy"`` (an ``int64`` index array, also
        covering a negative-step slice), or ``"contiguous"`` (payload
        ``(parts, is_mask)`` for a forward slice or a boolean mask).
        """
        if row_indexer is None:
            return "whole", None
        if isinstance(row_indexer, (int, np.integer)):
            return "scalar", self._locate(int(row_indexer))
        if isinstance(row_indexer, slice):
            start, stop, step = row_indexer.indices(self._n_rows)
            if step < 0:
                return "fancy", np.arange(start, stop, step, dtype=np.int64)
            return "contiguous", (self._slice_parts(row_indexer), False)
        if isinstance(row_indexer, np.ndarray):
            if row_indexer.dtype == np.bool_:
                return "contiguous", (self._mask_parts(row_indexer), True)
            return "fancy", row_indexer.astype(np.int64, copy=False)
        raise TypeError(f"Unsupported row indexer of type {type(row_indexer).__name__}.")

    def _gather_one(
        self,
        column_name: str,
        row_indexer: Any,
        thread_cap: int | None = None,
        out: NDArray[Any] | None = None,
        indices_sorted: bool | None = None,
    ) -> NDArray[Any]:
        self._check_open()
        self._require_columns([column_name])
        children = self._children
        if len(children) == 1:
            return children[0]._gather_one(
                column_name, row_indexer, thread_cap, out=out, indices_sorted=indices_sorted
            )
        if isinstance(row_indexer, np.ndarray) and row_indexer.dtype == np.bool_:
            native = self._mask_native([column_name], row_indexer)
            if native is not None:
                result = native[column_name]
                if out is None:
                    return result
                out[:] = result
                return out
        kind, payload = self._classify_rows(row_indexer)
        if kind == "scalar":
            file_index, local = payload
            child: ColStoreReader = children[file_index]
            return child._gather_one(column_name, local, thread_cap, out=out)
        if kind == "fancy":
            return self._fancy_one(column_name, payload, out=out, indices_sorted=indices_sorted)
        if kind == "whole":
            whole = (
                out
                if out is not None
                else np.empty(self._n_rows, dtype=self._native_dtype(column_name))
            )
            self._fill_contiguous_columns(
                {column_name: whole}, [column_name], self._offset_regions()
            )
            return whole
        parts, is_mask = payload
        return self._gather_one_contiguous(column_name, parts, is_mask, out=out)

    def _fill_contiguous(
        self,
        out: NDArray[Any],
        column_name: str,
        parts: list[tuple[int, Any]],
        lengths: list[int],
    ) -> None:
        """Fill ``out`` with one column's per-file portions in file order."""
        self._fill_contiguous_columns(
            {column_name: out}, [column_name], self._contiguous_regions(parts, lengths)
        )

    def _gather_slice_into(
        self, out: NDArray[Any], column_name: str, start: int, stop: int
    ) -> None:
        """Fill ``out`` with a forward slice's rows directly, one file at a time.

        The forward-slice case of :meth:`_gather_one`, but writing each file's
        portion straight into the caller's array (e.g. a region of a memory-mapped
        output) instead of allocating and returning a fresh one -- one copy from
        source to destination rather than two.
        """
        self._check_open()
        self._require_columns([column_name])
        parts = self._slice_parts(slice(start, stop))
        lengths = self._contiguous_lengths(parts, False)
        self._fill_contiguous(out, column_name, parts, lengths)

    def _gather_one_contiguous(
        self,
        column_name: str,
        parts: list[tuple[int, Any]],
        is_mask: bool,
        out: NDArray[Any] | None = None,
    ) -> NDArray[Any]:
        lengths = self._contiguous_lengths(parts, is_mask)
        dst = (
            out
            if out is not None
            else np.empty(sum(lengths), dtype=self._native_dtype(column_name))
        )
        self._fill_contiguous(dst, column_name, parts, lengths)
        return dst

    def _gather_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        self._check_open()
        self._require_columns(column_names)
        children = self._children
        if len(children) == 1:
            return children[0]._gather_many(column_names, row_indexer)
        if isinstance(row_indexer, np.ndarray) and row_indexer.dtype == np.bool_:
            native = self._mask_native(column_names, row_indexer)
            if native is not None:
                return native
        kind, payload = self._classify_rows(row_indexer)
        if kind == "scalar":
            file_index, local = payload
            child: ColStoreReader = children[file_index]
            return child._gather_many(column_names, local)
        if kind == "fancy":
            return self._fancy_many(column_names, payload)
        if kind == "whole":
            out = {
                name: np.empty(self._n_rows, dtype=self._native_dtype(name))
                for name in column_names
            }
            self._fill_contiguous_columns(out, column_names, self._offset_regions())
            return out
        parts, is_mask = payload
        return self._gather_many_contiguous(column_names, parts, is_mask)

    def _gather_many_contiguous(
        self, column_names: list[str], parts: list[tuple[int, Any]], is_mask: bool
    ) -> dict[str, NDArray[Any]]:
        lengths = self._contiguous_lengths(parts, is_mask)
        total = sum(lengths)
        out = {name: np.empty(total, dtype=self._native_dtype(name)) for name in column_names}
        self._fill_contiguous_columns(out, column_names, self._contiguous_regions(parts, lengths))
        return out

    # ---- Zero-copy seam ------------------------------------------------

    def _view_one(self, column_name: str, row_indexer: Any) -> NDArray[Any]:
        self._check_open()
        self._require_columns([column_name])
        if len(self._children) == 1:
            return self._children[0]._view_one(column_name, row_indexer)
        if isinstance(row_indexer, np.ndarray):
            raise ValueError(_VIEW_FANCY_MESSAGE)
        if row_indexer is None:
            raise ValueError(_VIEW_WHOLE_MESSAGE)
        if isinstance(row_indexer, (int, np.integer)):
            file_index, local = self._locate(int(row_indexer))
            return self._children[file_index]._view_one(column_name, local)
        if isinstance(row_indexer, slice):
            if row_indexer.indices(self._n_rows)[2] < 0:
                raise ValueError(_VIEW_REVERSED_MESSAGE)
            parts = self._slice_parts(row_indexer)
            if not parts:
                return self._children[0]._view_one(column_name, slice(0, 0))
            if len(parts) == 1:
                file_index, sub = parts[0]
                return self._children[file_index]._view_one(column_name, sub)
            raise ValueError(_VIEW_SPAN_MESSAGE)
        raise TypeError(f"Unsupported row indexer of type {type(row_indexer).__name__}.")

    def _view_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        self._check_open()
        self._require_columns(column_names)
        if len(self._children) == 1:
            return self._children[0]._view_many(column_names, row_indexer)
        return {name: self._view_one(name, row_indexer) for name in column_names}

    # ---- Combination ---------------------------------------------------

    def __or__(self, other: object) -> ColStoreDataset:
        """Combine with another reader/dataset into a new dataset (flattens)."""
        return _combine_readers(self, other)

    def __ror__(self, other: object) -> ColStoreDataset:
        return _combine_readers(other, self)

    def __ior__(self, other: object) -> ColStoreDataset:
        """In-place ``ds |= other``: append a borrowed reader/dataset."""
        if not isinstance(other, (ColStoreReader, ColStoreDataset)):
            raise TypeError(
                f"Unsupported operand for |=: {type(other).__name__}; expected "
                f"ColStoreReader or ColStoreDataset. Use append() for paths."
            )
        return self.append(other)

    # ---- Lifecycle -----------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("ColStoreDataset is closed.")

    def close(self) -> None:
        """Close the dataset, closing only the children it opened (owns)."""
        if self._closed:
            return
        for child, owned in zip(self._children, self._owned, strict=True):
            if owned:
                child.close()
        self._closed = True

    def __enter__(self) -> ColStoreDataset:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _combine_readers(left: object, right: object) -> ColStoreDataset:
    """Build a dataset spanning two readers/datasets, flattening any datasets.

    Backs the ``|`` operator. The result *borrows* the child readers -- the
    operands stay open and are not closed by the result -- and validates the
    combined schema (raising :class:`ValueError` on mismatch).
    """
    operands: list[ColStoreReader | ColStoreDataset] = []
    for operand in (left, right):
        if not isinstance(operand, (ColStoreReader, ColStoreDataset)):
            raise TypeError(
                f"Unsupported operand for |: {type(operand).__name__}; expected "
                f"ColStoreReader or ColStoreDataset. Combine paths with "
                f"colstore.open([...]) or ColStoreDataset.append() instead."
            )
        operands.append(operand)
    return ColStoreDataset(operands)
