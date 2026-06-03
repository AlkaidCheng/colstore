"""Module-level convenience functions: open, create, recreate, update, store.

These thin wrappers around :class:`ColStoreReader` and :class:`ColStoreWriter` give the
package a uroot-style entry-point surface where each function does one
obvious thing.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import format as fmt
from .reader import ColStoreReader
from .writer import ColStoreWriter


def open(path: str | os.PathLike[str], **kwargs: Any) -> ColStoreReader:
    """Open an existing ``.cstore`` file for read.

    Equivalent to ``ColStoreReader(path, **kwargs)``. The file must exist and be
    a valid colstore file; otherwise :class:`FileNotFoundError` or
    :class:`FormatError` propagates.
    """
    return ColStoreReader(path, **kwargs)


def create(path: str | os.PathLike[str]) -> ColStoreWriter:
    """Open a new file for streaming writes; fail if it already exists.

    Use this when you want to be sure you are not overwriting anything.
    """
    return ColStoreWriter(path, mode="create")


def recreate(path: str | os.PathLike[str]) -> ColStoreWriter:
    """Open a file for streaming writes, truncating any existing content.

    Use this when you intentionally want to overwrite. To fail on
    overwrite instead, use :func:`create`.
    """
    return ColStoreWriter(path, mode="recreate")


def update(path: str | os.PathLike[str]) -> ColStoreWriter:
    """Open an existing file for append.

    The schema is loaded from the existing manifest; every :meth:`write`
    must match it exactly. Orphan bytes from a crashed prior writer (if
    any) are truncated on open. Raises :class:`FileNotFoundError` if the
    file does not exist.
    """
    return ColStoreWriter(path, mode="update")


def store(
    data: Any,
    path: str | os.PathLike[str],
    *,
    mode: str = "create",
    show_progress: bool = True,
    **open_kwargs: Any,
) -> ColStoreReader:
    """One-shot: write a single-record file and return an opened reader.

    Accepted ``data`` types:

    * ``dict[str, numpy.ndarray]`` -- column-major mapping.
    * Structured ``numpy.ndarray`` (``dtype.names`` non-None) -- one
      column per field.
    * pandas ``DataFrame`` -- one column per series.

    ``mode`` is ``"create"`` (default; fail if file exists) or
    ``"recreate"`` (truncate if exists). For multi-record streaming
    writes, use :func:`create` / :func:`recreate` / :func:`update`
    directly.

    Returns the opened :class:`ColStoreReader` for immediate use.
    """
    if mode not in ("create", "recreate"):
        raise ValueError(f"Invalid mode {mode!r} for store(); expected 'create' or 'recreate'.")

    columns = _coerce_to_columns(data)

    # write_dataset is the single-record writer that includes a progress bar;
    # ColStoreWriter is for multi-record streams. For one-shot writes, write_dataset
    # is slightly cheaper (no counter rewrite at the end -- the counters are
    # right the first time) and surfaces a progress bar.
    if mode == "create" and os.path.exists(path):
        raise FileExistsError(f"{path} already exists; use mode='recreate' to overwrite.")
    fmt.write_dataset(columns, path, batch_size=100_000, show_progress=show_progress)
    return ColStoreReader(path, **open_kwargs)


def _coerce_to_columns(
    data: Any,
) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Dispatch on the input type and return a uniform ``dict[name, ndarray]``."""
    if isinstance(data, dict):
        return {str(name): np.ascontiguousarray(array) for name, array in data.items()}
    if isinstance(data, np.ndarray):
        if data.dtype.names is None:
            raise TypeError(
                "store() received a plain ndarray; pass {name: array} as a dict "
                "(or a structured ndarray with named fields)."
            )
        return {name: np.ascontiguousarray(data[name]) for name in data.dtype.names}
    if _is_pandas_dataframe(data):
        return _dataframe_to_columns(data)
    raise TypeError(
        f"store() does not know how to handle {type(data).__name__}. "
        f"Expected dict[str, ndarray], structured ndarray, or pandas DataFrame."
    )


def _is_pandas_dataframe(data: Any) -> bool:
    """Duck-typed pandas check so we don't import pandas at module load."""
    return (
        hasattr(data, "columns")
        and hasattr(data, "to_numpy")
        and type(data).__name__ == "DataFrame"
    )


def _dataframe_to_columns(frame: Any) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    """Convert a pandas DataFrame to a column-name -> ndarray dict.

    Object-dtype columns are rejected up front with a clearer message than
    the writer's generic "unsupported dtype" error.
    """
    columns: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    for column_name in frame.columns:
        series = frame[column_name]
        array = series.to_numpy()
        if array.dtype.kind == "O":
            raise TypeError(
                f"Column {column_name!r} (pandas dtype {series.dtype}) converts to "
                f"an object array and cannot be stored. Cast it to a fixed-size NumPy "
                f"dtype (e.g. float64, int64, or a fixed-width string like 'S16') first."
            )
        columns[str(column_name)] = array
    return columns
