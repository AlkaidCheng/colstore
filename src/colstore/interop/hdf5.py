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

import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ._convert import columns_to_frame, frame_to_columns, storable_column, store_columns
from .base import FileFormat, Selection

if TYPE_CHECKING:
    from ..reader import ColStoreReader

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

    def from_file(
        self,
        source: Any,
        dest: Any,
        *,
        backend: str = "auto",
        key: str | None = None,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> ColStoreReader:
        """Read an HDF5 file into a ``.cstore`` and open it; auto-detects the writer."""
        data = _read_hdf5(os.fspath(source), backend, key)
        if columns is not None:
            missing = [name for name in columns if name not in data]
            if missing:
                raise ValueError(f"Column(s) not found in the HDF5 file: {', '.join(missing)}.")
            data = {name: data[name] for name in columns}
        return store_columns(data, dest, **kwargs)


def _write_h5py(path: str, key: str, data: dict[str, np.ndarray[Any, Any]]) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        group = handle.create_group(key) if key not in (None, "", "/") else handle
        for name, array in data.items():
            if array.dtype.kind == "U":
                # h5py has no UTF-32 type; store as a variable-length string.
                group.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype())
            elif array.dtype.kind in "Mm":
                # h5py has no datetime type; store the int64 view + the dtype to restore it.
                dataset = group.create_dataset(name, data=array.view("i8"))
                dataset.attrs["np_dtype"] = str(array.dtype)
            else:
                group.create_dataset(name, data=array)  # numeric, bool, fixed bytes (S)


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
