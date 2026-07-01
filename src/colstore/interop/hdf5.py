"""HDF5 file format (``colstore.interop.hdf5``), reading across writers.

An HDF5 file can be written several ways, so import auto-detects the layout and
reads it gracefully:

* a **pandas / PyTables** store (``DataFrame.to_hdf``) -- a group carrying a
  ``pandas_type`` attribute -- is read with ``pandas.read_hdf``;
* a **plain h5py** file -- one dataset per column, at the root or under a group
  -- is read with h5py (variable-length strings via ``asstr``).

Export chooses the writer with ``backend=`` (``"h5py"`` default, or ``"pandas"``)
and the dataset/group with ``key=`` (default ``"data"``). h5py needs no pandas/
PyTables; the pandas backend needs them.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np

from .._sizes import resolve_batch_rows
from .._types import Columns
from ._convert import columns_to_frame, frame_to_columns, storable_column
from ._stream_import import StreamPlan
from .base import FileFormat, Selection

#: Default group/key for an HDF5 dataset written by colstore.
_DEFAULT_KEY = "data"


class Hdf5Format(FileFormat):
    """HDF5, read across h5py / pandas writers; write with a chosen ``backend``."""

    name: ClassVar[str] = "hdf5"
    aliases: ClassVar[frozenset[str]] = frozenset({"h5"})
    extensions: ClassVar[frozenset[str]] = frozenset({".h5", ".hdf5"})

    def to_file(
        self,
        selection: Selection,
        dest: Any,
        *,
        backend: str = "h5py",
        key: str = _DEFAULT_KEY,
        **kwargs: Any,
    ) -> None:
        """Write the selection to an HDF5 file under ``key`` with the chosen ``backend``."""
        data = selection.gather_all()
        if backend == "h5py":
            _write_h5py(os.fspath(dest), key, data)
        elif backend == "pandas":
            columns_to_frame(data).to_hdf(os.fspath(dest), key=key, mode="w", **kwargs)
        else:
            raise ValueError(f"unknown hdf5 backend {backend!r}; expected 'h5py' or 'pandas'.")

    def read_columns(
        self,
        source: Any,
        *,
        columns: list[str] | None = None,
        backend: str = "auto",
        key: str | None = None,
        **kwargs: Any,
    ) -> Columns:
        """Read the whole HDF5 file into a column mapping; auto-detects the writer."""
        data = _read_hdf5(os.fspath(source), backend, key)
        if columns is not None:
            missing = [name for name in columns if name not in data]
            if missing:
                raise ValueError(f"Column(s) not found in the HDF5 file: {', '.join(missing)}.")
            data = {name: data[name] for name in columns}
        return data

    def stream_import(
        self,
        source: Any,
        *,
        columns: list[str] | None = None,
        batch_size: int | str | None,
        backend: str = "auto",
        key: str | None = None,
        **kwargs: Any,
    ) -> StreamPlan | str:
        """Stream an h5py file (dataset slicing) or a pandas *table* store (chunked read).

        The pandas *fixed* format (the ``DataFrame.to_hdf`` default) and any
        variable-length-string / object column return a reason so the file is read whole.
        """
        return _hdf5_stream_plan(os.fspath(source), backend, key, columns, batch_size)

    def stream_export(
        self,
        selection: Selection,
        dest: Any,
        *,
        batch_size: int | str | None,
        backend: str = "h5py",
        key: str = _DEFAULT_KEY,
        **kwargs: Any,
    ) -> str | None:
        """Stream the selection to an HDF5 file, one batch at a time (h5py backend only).

        The h5py backend appends each batch to resizable datasets in bounded memory. The
        pandas backend writes a whole-file fixed-format store (its default), which has no
        appendable layout, so it declines and is written whole instead.
        """
        if backend == "pandas":
            return "the pandas backend writes a whole-file fixed-format store"
        if backend != "h5py":
            raise ValueError(f"unknown hdf5 backend {backend!r}; expected 'h5py' or 'pandas'.")
        _stream_h5py(os.fspath(dest), key, selection, batch_size)
        return None


def _h5_dataset_dtype(array: np.ndarray[Any, Any]) -> Any:
    """The h5py dtype a column's dataset is created with.

    h5py has no UTF-32 or datetime type, so a ``U`` column stores as a variable-length
    string and a datetime / timedelta column as its ``int64`` view; everything else keeps
    its own dtype (numeric, bool, fixed bytes ``S``).
    """
    import h5py

    if array.dtype.kind == "U":
        return h5py.string_dtype()
    if array.dtype.kind in "Mm":
        return np.dtype("i8")
    return array.dtype


def _h5_encode(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A column's values in the form its dataset stores (see :func:`_h5_dataset_dtype`)."""
    if array.dtype.kind == "U":
        return array.astype(object)
    if array.dtype.kind in "Mm":
        return array.view("i8")
    return array


def _h5_stamp_dtype(dataset: Any, array: np.ndarray[Any, Any]) -> None:
    """Record a datetime / timedelta column's real dtype so the reader can restore it."""
    if array.dtype.kind in "Mm":
        dataset.attrs["np_dtype"] = str(array.dtype)


def _order_tracking() -> dict[str, bool]:
    """``{"track_order": True}`` on h5py that supports it, else ``{}``.

    ``track_order`` makes a group store its links (columns) in creation order so the column
    order round-trips, instead of HDF5's default alphabetical link storage. It was added in
    h5py 2.9; on an older h5py the argument is unknown (it would raise), so it is omitted and
    columns read back alphabetically -- a graceful degradation rather than a failed write.
    """
    import h5py

    version = tuple(int(part) for part in h5py.__version__.split(".")[:2] if part.isdigit())
    return {"track_order": True} if version >= (2, 9) else {}


def _write_group(handle: Any, key: str) -> Any:
    """The group columns are written into: a named group (order-tracked), or the file root."""
    if key in (None, "", "/"):
        return handle
    return handle.create_group(key, **_order_tracking())


def _write_h5py(path: str, key: str, data: dict[str, np.ndarray[Any, Any]]) -> None:
    import h5py

    with h5py.File(path, "w", **_order_tracking()) as handle:
        group = _write_group(handle, key)
        for name, array in data.items():
            dataset = group.create_dataset(
                name, data=_h5_encode(array), dtype=_h5_dataset_dtype(array)
            )
            _h5_stamp_dtype(dataset, array)


def _stream_h5py(path: str, key: str, selection: Selection, batch_size: int | str | None) -> None:
    """Stream the selection into one resizable h5py dataset per column, appending each batch."""
    import h5py

    with h5py.File(path, "w", **_order_tracking()) as handle:
        group = _write_group(handle, key)
        datasets: dict[str, Any] = {}
        offset = 0
        # copy=False is safe here: each dataset assignment copies the batch into the file
        # synchronously, before the next batch is drawn into the reused buffer.
        for batch in selection.iter_batches(batch_size, copy=False):
            length = len(next(iter(batch.values())))
            for name, array in batch.items():
                if name not in datasets:
                    datasets[name] = group.create_dataset(
                        name, shape=(0,), maxshape=(None,), dtype=_h5_dataset_dtype(array)
                    )
                    _h5_stamp_dtype(datasets[name], array)
                dataset = datasets[name]
                dataset.resize(offset + length, axis=0)
                if length:
                    dataset[offset : offset + length] = _h5_encode(array)
            offset += length


def _read_hdf5(path: str, backend: str, key: str | None) -> dict[str, np.ndarray[Any, Any]]:
    import h5py

    if backend not in ("auto", "h5py", "pandas"):
        raise ValueError(f"unknown hdf5 backend {backend!r}; expected 'auto', 'h5py', or 'pandas'.")
    if key in ("", "/"):  # the writer's "root" sentinels; read them as auto/root
        key = None
    with h5py.File(path, "r") as handle:
        pandas_key = _pandas_key(handle, key)
        if pandas_key is not None and backend in ("auto", "pandas"):
            return _read_pandas(path, pandas_key)
        if backend == "pandas":
            raise ValueError(f"hdf5 backend='pandas' but no pandas table found in {path!r}.")
        return _h5_columns(_resolve_group(handle, key))


def _pandas_key(handle: Any, key: str | None) -> str | None:
    """The key of a pandas HDFStore group (``pandas_type`` attr), or ``None``."""
    import h5py

    def is_pandas(name: str) -> bool:
        obj = handle.get(name)
        # pandas/PyTables stamps both attributes; require both so a stray
        # `pandas_type` attribute from another tool does not hijack the route.
        return (
            isinstance(obj, h5py.Group)
            and "pandas_type" in obj.attrs
            and "pandas_version" in obj.attrs
        )

    if key is not None:
        return key if is_pandas(key) else None
    pandas_keys = [name for name in handle if is_pandas(name)]
    if len(pandas_keys) > 1:
        raise ValueError(f"multiple pandas tables {pandas_keys}; pass key= to choose one.")
    return pandas_keys[0] if pandas_keys else None


def _read_pandas(path: str, key: str) -> dict[str, np.ndarray[Any, Any]]:
    import pandas as pd

    return frame_to_columns(pd.read_hdf(path, key=key))


def _resolve_group(handle: Any, key: str | None) -> Any:
    """The h5py group holding the column datasets: ``key``, the root, or the sole subgroup."""
    import h5py

    if key is not None:
        obj = handle[key]
        if not isinstance(obj, h5py.Group):
            raise TypeError(f"hdf5 key {key!r} is a dataset, not a group of columns.")
        return obj
    if any(isinstance(handle[name], h5py.Dataset) for name in handle):
        return handle  # one dataset per column at the root
    groups = [name for name in handle if isinstance(handle[name], h5py.Group)]
    if len(groups) == 1:
        return handle[groups[0]]
    raise ValueError(f"HDF5 file has {len(groups)} groups {groups}; pass key= to choose one.")


def _h5_columns(group: Any) -> dict[str, np.ndarray[Any, Any]]:
    import h5py

    columns: dict[str, np.ndarray[Any, Any]] = {}
    for name in group:
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            continue
        string_info = h5py.check_string_dtype(dataset.dtype)
        if string_info is not None and string_info.length is None:
            columns[name] = dataset.asstr()[()].astype("U")  # variable-length -> unicode
        elif "np_dtype" in dataset.attrs:  # an int64 view of a datetime/timedelta column
            columns[name] = dataset[()].view(np.dtype(dataset.attrs["np_dtype"]))
        else:
            columns[name] = storable_column(name, dataset[()])  # numeric / bool / fixed bytes (S)
    if not columns:
        raise ValueError("no readable datasets found in the HDF5 group.")
    return columns


# ---- Bounded-memory streaming import ---------------------------------------


def _hdf5_stream_plan(
    path: str,
    backend: str,
    key: str | None,
    columns: list[str] | None,
    batch_size: int | str | None,
) -> StreamPlan | str:
    """A :class:`StreamPlan` for an HDF5 file, or a reason string to read it whole.

    Returns a reason when the layout cannot stream stably -- a pandas fixed-format store,
    or any variable-length-string / object column.
    """
    import h5py

    norm_key = None if key in ("", "/") else key
    with h5py.File(path, "r") as handle:
        pandas_key = _pandas_key(handle, norm_key)
        if pandas_key is not None and backend in ("auto", "pandas"):
            if "table" not in str(handle[pandas_key].attrs.get("pandas_type", "")):
                return "pandas fixed-format HDF5 has no chunked reader"
        elif backend == "pandas":
            return "no pandas table found"  # the whole-file path raises the clear error
        else:
            pandas_key = None
            names, n_rows, bytes_per_row, reason = _h5py_stream_plan(
                _resolve_group(handle, norm_key), columns
            )
            if reason:
                return reason
    if pandas_key is not None:
        reason = _pandas_object_reason(path, pandas_key, columns)
        if reason:
            return reason
        rows = resolve_batch_rows(batch_size) if isinstance(batch_size, int) else None
        if rows is None:  # a byte budget needs a per-row size; size it from the first chunk
            rows = _pandas_rows_for_budget(path, pandas_key, columns, batch_size)
        return StreamPlan(_pandas_stream_batches(path, pandas_key, columns, rows))
    rows = resolve_batch_rows(batch_size, bytes_per_row=bytes_per_row)
    return StreamPlan(_h5py_stream_batches(path, norm_key, names, rows or n_rows), n_rows)


def _h5py_stream_plan(group: Any, columns: list[str] | None) -> tuple[list[str], int, int, str]:
    """Validate an h5py group for streaming; return (names, n_rows, bytes_per_row, reason)."""
    import h5py

    names = (
        columns
        if columns is not None
        else [name for name in group if isinstance(group[name], h5py.Dataset)]
    )
    n_rows: int | None = None
    bytes_per_row = 0
    for name in names:
        if name not in group or not isinstance(group[name], h5py.Dataset):
            return [], 0, 0, f"column {name!r} is not a dataset"
        dataset = group[name]
        string_info = h5py.check_string_dtype(dataset.dtype)
        if string_info is not None and string_info.length is None:
            return [], 0, 0, f"column {name!r} is a variable-length string"
        if n_rows is not None and len(dataset) != n_rows:
            return [], 0, 0, f"column {name!r} has a different length"  # ragged group
        n_rows = len(dataset)
        bytes_per_row += dataset.dtype.itemsize
    return list(names), n_rows or 0, max(1, bytes_per_row), ""


def _h5py_stream_batches(
    path: str, key: str | None, names: list[str], rows: int
) -> Iterator[Columns]:
    """Yield column dicts by slicing each h5py dataset over row ranges (file stays open)."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = _resolve_group(handle, key)
        datasets = {name: group[name] for name in names}
        n_rows = len(next(iter(datasets.values()))) if datasets else 0
        starts = range(0, n_rows, rows) if n_rows else range(1)  # one empty batch when empty
        for start in starts:
            stop = min(start + rows, n_rows)
            yield {name: _h5_slice(name, ds, start, stop) for name, ds in datasets.items()}


def _h5_slice(name: str, dataset: Any, start: int, stop: int) -> np.ndarray[Any, Any]:
    """One row range of an h5py dataset, as a fixed-width column (see :func:`_h5_columns`)."""
    if "np_dtype" in dataset.attrs:  # an int64 view of a datetime/timedelta column
        view: np.ndarray[Any, Any] = np.asarray(dataset[start:stop]).view(
            np.dtype(dataset.attrs["np_dtype"])
        )
        return view
    return storable_column(name, dataset[start:stop])  # numeric / bool / fixed bytes (S)


def _pandas_object_reason(path: str, key: str, columns: list[str] | None) -> str:
    """Empty if the selected pandas columns are non-object, else why streaming is unsafe."""
    import pandas as pd

    head = pd.read_hdf(path, key=key, stop=0)
    names = columns if columns is not None else list(head.columns)
    for name in names:
        if name not in head.columns:
            return f"column {name!r} not found in the pandas table"
        if head[name].dtype == object or pd.api.types.is_extension_array_dtype(head[name].dtype):
            return f"column {name!r} is an object or extension column"
    return ""


def _pandas_rows_for_budget(
    path: str, key: str, columns: list[str] | None, batch_size: int | str | None
) -> int:
    """Resolve a ``batch_size`` byte budget to rows from the pandas table's per-row size."""
    import pandas as pd

    head = pd.read_hdf(path, key=key, stop=1)
    if columns is not None:
        head = head[columns]
    bytes_per_row = max(1, sum(dtype.itemsize for dtype in head.dtypes))
    return resolve_batch_rows(batch_size, bytes_per_row=bytes_per_row) or len(head)


def _pandas_stream_batches(
    path: str, key: str, columns: list[str] | None, rows: int
) -> Iterator[Columns]:
    """Yield column dicts from a pandas table read in ``rows``-sized chunks."""
    import pandas as pd

    # closing() releases the PyTables file handle even if the consumer raises
    # mid-stream; a bare iterator only auto-closes on full consumption.
    with contextlib.closing(pd.read_hdf(path, key=key, chunksize=rows)) as chunks:
        for chunk in chunks:
            if columns is not None:
                chunk = chunk[columns]
            yield frame_to_columns(chunk)
