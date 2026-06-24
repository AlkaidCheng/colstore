"""The uproot backend for the ROOT file format (pure-Python backend).

Reads stream through ``tree.iterate`` and writes stream through ``mktree`` +
``extend``, both honoring a ``batch_size`` memory budget. uproot is imported
lazily, only when this backend runs, so it is never pulled in by ``import
colstore`` or by selecting the ROOT backend.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np

from .._base import _ReaderBase
from ..progress import progress_bar
from ._streaming import ColumnBatch
from .root import (
    _SCALAR_KINDS,
    RootBackend,
    _batch_dict,
    _relabel,
    _resolve_chunk_rows,
    _sanitized_name_map,
    _split_path_and_tree,
    _warn_branch_renames,
    filter_storable,
    require_existing,
)


def _import_uproot() -> Any:
    try:
        import uproot
    except ImportError as exc:  # pragma: no cover - exercised only without uproot
        raise ImportError(
            "The uproot backend requires uproot, which is not installed "
            "(install it with 'pip install uproot')."
        ) from exc
    return uproot


def _resolve_source(source: Any, treename: str | None) -> tuple[Any, str | None]:
    """Split a uproot source into ``(files, tree_name)``; ``tree_name`` may be None."""
    if isinstance(source, dict):
        if len(source) != 1:
            raise ValueError(
                f"A {{treename: files}} mapping must name exactly one tree; got {len(source)}."
            )
        ((tree, files),) = source.items()
        if treename is not None and treename != tree:
            raise ValueError(f"Conflicting tree names: treename={treename!r} but mapping {tree!r}.")
        return files, tree
    if isinstance(source, str):
        path, embedded = _split_path_and_tree(source)
        if embedded is not None and treename is not None and treename != embedded:
            raise ValueError(
                f"Conflicting tree names: treename={treename!r} but the path names {embedded!r}."
            )
        return path, embedded or treename
    if isinstance(source, os.PathLike):
        return os.fspath(source), treename
    raise TypeError(
        f"the uproot backend reads a path or a {{tree: files}} mapping, not "
        f"{type(source).__name__}; use backend='ROOT' for an RDataFrame."
    )


def _first_file(files: Any) -> Any:
    return files[0] if isinstance(files, (list, tuple)) else files


def _sole_tree(opened: Any) -> str:
    """The name of the file's only TTree, or raise if there are none or several."""
    trees = list(dict.fromkeys(opened.keys(filter_classname="TTree", cycle=False)))
    if not trees:
        raise ValueError("No TTree found in the file; pass treename=... to select one.")
    if len(trees) > 1:
        raise ValueError(f"Multiple trees ({', '.join(trees)}); pass treename=... to choose one.")
    return str(trees[0])


def _is_storable(tree: Any, name: str) -> bool:
    """Whether a branch reads as a fixed-size scalar (1-D, numeric/bool) column."""
    try:
        sample = tree[name].array(entry_stop=1, library="np")
    except Exception:
        return False
    dtype = getattr(sample, "dtype", None)
    return getattr(sample, "ndim", 0) == 1 and dtype is not None and dtype.kind in _SCALAR_KINDS


def _iterate_spec(files: Any, tree: str) -> Any:
    if isinstance(files, (list, tuple)):
        return [f"{os.fspath(f)}:{tree}" for f in files]
    return {os.fspath(files): tree}


def _total_entries(uproot: Any, files: Any, tree: str) -> int:
    paths = files if isinstance(files, (list, tuple)) else [files]
    total = 0
    for path in paths:
        with uproot.open(os.fspath(path)) as opened:
            total += int(opened[tree].num_entries)
    return total


class UprootBackend(RootBackend):
    """The pure-Python uproot backend."""

    name: ClassVar[str] = "uproot"

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("uproot") is not None

    def read_batches(
        self,
        source: Any,
        *,
        treename: str | None,
        columns: list[str] | None,
        keep_valid_only: bool,
        batch_size: int | str | None,
    ) -> tuple[Iterator[ColumnBatch], int]:
        uproot = _import_uproot()
        files, tree = _resolve_source(source, treename)
        with uproot.open(os.fspath(_first_file(files))) as opened:
            tree_name = tree or _sole_tree(opened)
            rep = opened[tree_name]
            keys = rep.keys()
            branch_names = [str(key) for key in keys]
            names = columns if columns is not None else branch_names
            if columns is not None:
                require_existing(names, branch_names)
            selected = filter_storable(names, lambda n: _is_storable(rep, n), keep_valid_only)
            empty = {name: rep[name].array(entry_stop=0, library="np") for name in selected}
        total = _total_entries(uproot, files, tree_name)

        def batches() -> Iterator[ColumnBatch]:
            if total == 0:
                yield empty
                return
            step = batch_size if batch_size is not None else total
            for chunk in uproot.iterate(
                _iterate_spec(files, tree_name),
                expressions=selected,
                step_size=step,
                library="np",
            ):
                yield {name: np.ascontiguousarray(chunk[name]) for name in selected}

        return batches(), total

    def write(
        self,
        reader: _ReaderBase,
        *,
        columns: list[str],
        dest: Any,
        treename: str,
        batch_size: int | str | None,
        show_progress: bool,
        **options: Any,
    ) -> None:
        uproot = _import_uproot()
        name_map = _sanitized_name_map(columns)
        _warn_branch_renames(name_map)
        dtypes = reader.dtypes
        row_nbytes = sum(dtypes[name].itemsize for name in columns)
        total_rows = reader.n_rows
        rows_per_chunk = _resolve_chunk_rows(batch_size, row_nbytes)
        with (
            uproot.recreate(os.fspath(dest)) as out_file,
            progress_bar(
                total=total_rows * row_nbytes,
                desc=f"{os.fspath(dest)} <- colstore",
                unit="B",
                unit_scale=True,
                enabled=show_progress,
            ) as bar,
        ):
            out_file.mktree(treename, {name_map[name]: dtypes[name] for name in columns})
            if total_rows == 0:
                return
            step = rows_per_chunk or total_rows
            for start in range(0, total_rows, step):
                end = min(start + step, total_rows)
                out_file[treename].extend(
                    _relabel(_batch_dict(reader[start:end, columns]), name_map)
                )
                bar.update((end - start) * row_nbytes)
