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
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from . import config
from ._base import _ReaderBase
from .reader import ColStoreReader, _make_dataframe_no_consolidate

if TYPE_CHECKING:
    import pandas as pd

# A single source: a path to open (owned), or an already-open reader/dataset
# (borrowed). The constructor and append() accept one of these or a sequence.
_SourceLike: TypeAlias = "str | os.PathLike[str] | ColStoreReader | ColStoreDataset"

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

    Paths are opened and *owned* (closed on :meth:`close`); readers and datasets
    are *borrowed* (left open). Datasets are flattened to their children. The
    public read surface matches :class:`~colstore.reader.ColStoreReader`.
    """

    def __init__(
        self,
        sources: _SourceLike | Sequence[_SourceLike] | None = None,
        **reader_kwargs: Any,
    ) -> None:
        self._children: list[ColStoreReader] = []
        self._owned: list[bool] = []
        self._column_dtypes: dict[str, np.dtype[Any]] = {}
        self._offsets: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self._n_rows = 0
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
                    reader = ColStoreReader(item, **reader_kwargs)
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
        self._offsets = np.zeros(len(self._children) + 1, dtype=np.int64)
        if self._children:
            self._offsets[1:] = np.cumsum([child.n_rows for child in self._children])
        self._n_rows = int(self._offsets[-1])

    def append(
        self, source: _SourceLike | Sequence[_SourceLike], **reader_kwargs: Any
    ) -> ColStoreDataset:
        """Grow the dataset in place; return ``self`` so calls can be chained.

        ``source`` is anything the constructor accepts: a path (opened and
        owned), a reader or dataset (borrowed), or a list/tuple mixing them.
        The first child establishes the schema; later children must match it.
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
    def path(self) -> tuple[Path, ...]:
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

    def _require_columns(self, column_names: list[str]) -> None:
        unknown = [name for name in column_names if name not in self._column_dtypes]
        if unknown:
            raise KeyError(f"Unknown column(s): {unknown}. Available columns: {self.columns}")

    # ---- Parallel fill across files ------------------------------------
    #
    # Every multi-file read preallocates one output and fills each file's region
    # concurrently: the per-file reads are independent and write disjoint regions
    # (a contiguous slice, or disjoint scattered positions), so they need no
    # locking. The pool is sized to the gather thread budget and each per-file
    # read runs single-threaded, so the total thread count stays in the same
    # envelope as a single-file read -- the parallelism just moves up to the file
    # level, where each file's copy is too small to trip the single-file
    # parallel-copy size gate on its own.

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

    def _fill_many(
        self,
        out: dict[str, NDArray[Any]],
        lo: int,
        hi: int,
        child: ColStoreReader,
        column_names: list[str],
        sub: Any,
    ) -> Callable[[], None]:
        def job() -> None:
            for name in column_names:
                out[name][lo:hi] = self._portion_source(child, name, sub)

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

    def _fancy_one(self, column_name: str, indices: NDArray[np.int64]) -> NDArray[Any]:
        """Gather arbitrary global rows in parallel, preserving requested order.

        Sort the indices once to group them into one contiguous block per file,
        fill each file's block of a sorted buffer concurrently (disjoint regions,
        so no locking), then un-sort the buffer into the requested order with one
        scatter.
        """
        out = np.empty(len(indices), dtype=self.dtypes[column_name])
        if indices.size == 0:
            return out
        order, blocks = self._sorted_blocks(indices)
        buffer = np.empty(len(indices), dtype=self.dtypes[column_name])
        jobs = [
            self._fill_one(buffer, lo, hi, self._children[file_index], column_name, local)
            for file_index, lo, hi, local in blocks
        ]
        self._run_fill_jobs(jobs)
        out[order] = buffer
        return out

    def _fancy_many(
        self, column_names: list[str], indices: NDArray[np.int64]
    ) -> dict[str, NDArray[Any]]:
        """Multi-column :meth:`_fancy_one` sharing one sort across all columns.

        The grouping is computed once; each column fills and un-sorts through a
        single reused buffer, so peak memory stays at one column's worth rather
        than scaling with the column count.
        """
        out = {name: np.empty(len(indices), dtype=self.dtypes[name]) for name in column_names}
        if indices.size == 0:
            return out
        order, blocks = self._sorted_blocks(indices)
        for name in column_names:
            buffer = np.empty(len(indices), dtype=self.dtypes[name])
            jobs = [
                self._fill_one(buffer, lo, hi, self._children[file_index], name, local)
                for file_index, lo, hi, local in blocks
            ]
            self._run_fill_jobs(jobs)
            out[name][order] = buffer
        return out

    # ---- Copying seam --------------------------------------------------

    def _gather_one(
        self, column_name: str, row_indexer: Any, thread_cap: int | None = None
    ) -> NDArray[Any]:
        self._check_open()
        self._require_columns([column_name])
        children = self._children
        if len(children) == 1:
            return children[0]._gather_one(column_name, row_indexer, thread_cap)
        dtype = self.dtypes[column_name]
        if row_indexer is None:
            out = np.empty(self._n_rows, dtype=dtype)
            jobs: list[Callable[[], None]] = []
            for file_index, child in enumerate(children):
                lo, hi = int(self._offsets[file_index]), int(self._offsets[file_index + 1])
                if lo < hi:
                    jobs.append(self._fill_one(out, lo, hi, child, column_name, None))
            self._run_fill_jobs(jobs)
            return out
        if isinstance(row_indexer, (int, np.integer)):
            file_index, local = self._locate(int(row_indexer))
            return children[file_index]._gather_one(column_name, local, thread_cap)
        if isinstance(row_indexer, slice):
            start, stop, step = row_indexer.indices(self._n_rows)
            if step < 0:
                return self._fancy_one(column_name, np.arange(start, stop, step, dtype=np.int64))
            return self._gather_one_contiguous(
                column_name, dtype, self._slice_parts(row_indexer), False
            )
        if isinstance(row_indexer, np.ndarray):
            if row_indexer.dtype == np.bool_:
                return self._gather_one_contiguous(
                    column_name, dtype, self._mask_parts(row_indexer), True
                )
            return self._fancy_one(column_name, row_indexer.astype(np.int64, copy=False))
        raise TypeError(f"Unsupported row indexer of type {type(row_indexer).__name__}.")

    def _fill_contiguous(
        self,
        out: NDArray[Any],
        column_name: str,
        parts: list[tuple[int, Any]],
        lengths: list[int],
    ) -> None:
        """Fill ``out`` left to right, each file's portion of one column in turn.

        The portions cover disjoint, in-order regions of ``out``, so the per-file
        fills run concurrently without locking.
        """
        jobs: list[Callable[[], None]] = []
        write = 0
        for (file_index, sub), length in zip(parts, lengths, strict=True):
            jobs.append(
                self._fill_one(
                    out, write, write + length, self._children[file_index], column_name, sub
                )
            )
            write += length
        self._run_fill_jobs(jobs)

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
        self, column_name: str, dtype: np.dtype[Any], parts: list[tuple[int, Any]], is_mask: bool
    ) -> NDArray[Any]:
        lengths = self._contiguous_lengths(parts, is_mask)
        out = np.empty(sum(lengths), dtype=dtype)
        self._fill_contiguous(out, column_name, parts, lengths)
        return out

    def _gather_many(self, column_names: list[str], row_indexer: Any) -> dict[str, NDArray[Any]]:
        self._check_open()
        self._require_columns(column_names)
        children = self._children
        if len(children) == 1:
            return children[0]._gather_many(column_names, row_indexer)
        if row_indexer is None:
            out = {name: np.empty(self._n_rows, dtype=self.dtypes[name]) for name in column_names}
            jobs: list[Callable[[], None]] = []
            for file_index, child in enumerate(children):
                lo, hi = int(self._offsets[file_index]), int(self._offsets[file_index + 1])
                if lo < hi:
                    jobs.append(self._fill_many(out, lo, hi, child, column_names, None))
            self._run_fill_jobs(jobs)
            return out
        if isinstance(row_indexer, (int, np.integer)):
            file_index, local = self._locate(int(row_indexer))
            return children[file_index]._gather_many(column_names, local)
        if isinstance(row_indexer, slice):
            start, stop, step = row_indexer.indices(self._n_rows)
            if step < 0:
                return self._fancy_many(column_names, np.arange(start, stop, step, dtype=np.int64))
            return self._gather_many_contiguous(column_names, self._slice_parts(row_indexer), False)
        if isinstance(row_indexer, np.ndarray):
            if row_indexer.dtype == np.bool_:
                return self._gather_many_contiguous(
                    column_names, self._mask_parts(row_indexer), True
                )
            return self._fancy_many(column_names, row_indexer.astype(np.int64, copy=False))
        raise TypeError(f"Unsupported row indexer of type {type(row_indexer).__name__}.")

    def _gather_many_contiguous(
        self, column_names: list[str], parts: list[tuple[int, Any]], is_mask: bool
    ) -> dict[str, NDArray[Any]]:
        lengths = self._contiguous_lengths(parts, is_mask)
        total = sum(lengths)
        out = {name: np.empty(total, dtype=self.dtypes[name]) for name in column_names}
        jobs: list[Callable[[], None]] = []
        write = 0
        for (file_index, sub), length in zip(parts, lengths, strict=True):
            jobs.append(
                self._fill_many(
                    out, write, write + length, self._children[file_index], column_names, sub
                )
            )
            write += length
        self._run_fill_jobs(jobs)
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

    # ---- Whole-store materializers -------------------------------------

    def dict(self, copy: bool = True) -> dict[str, NDArray[Any]]:
        """Materialize every column as a mapping of name to ndarray."""
        self._check_open()
        column_names = list(self._column_dtypes)
        if not copy:
            return self._view_many(column_names, None)
        return self._gather_many(column_names, None)

    def recarray(self) -> NDArray[Any]:
        """Materialize the whole dataset as a structured ndarray."""
        column_data = self.dict()
        record_dtype = np.dtype([(name, self.dtypes[name]) for name in self._column_dtypes])
        record_array = np.empty(self._n_rows, dtype=record_dtype)
        for name in self._column_dtypes:
            record_array[name] = column_data[name]
        return record_array

    def frame(self) -> pd.DataFrame:
        """Materialize the whole dataset as a pandas DataFrame."""
        return _make_dataframe_no_consolidate(self.dict())

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
