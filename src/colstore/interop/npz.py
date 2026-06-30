"""NPZ file format: exchange colstore data with NumPy ``.npz`` archives.

An ``.npz`` archive is a zip of named arrays, which maps directly onto colstore's
columns -- one array per column. Export writes each selected column as a named
array; import reads every array in the archive as a column (all must share a
length and a fixed-width dtype, as for any colstore write).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .._types import Columns
from .base import FileFormat, Selection


class NpzFormat(FileFormat):
    """NumPy ``.npz`` archive (``colstore.interop.npz``), one array per column."""

    name: ClassVar[str] = "npz"
    extensions: ClassVar[frozenset[str]] = frozenset({".npz"})

    def to_file(
        self, selection: Selection, dest: Any, *, compress: bool = False, **kwargs: Any
    ) -> None:
        """Write the selected columns to ``dest`` as named arrays.

        ``compress=True`` uses ``numpy.savez_compressed``; the default is the
        uncompressed ``numpy.savez``.
        """
        # dict[str, Any] (not NDArray): np.savez's stub has a bool keyword, so a
        # **mapping of arrays only type-checks when the values widen to Any.
        columns: dict[str, Any] = selection.gather_all()
        # Write to an open handle, not the path: np.savez appends ".npz" to a path
        # argument (case-sensitively), so the file would otherwise not land exactly
        # at `dest`. A file object is written verbatim.
        with open(dest, "wb") as handle:
            if compress:
                np.savez_compressed(handle, **columns)
            else:
                np.savez(handle, **columns)

    def read_columns(
        self, source: Any, *, columns: list[str] | None = None, **kwargs: Any
    ) -> Columns:
        """Read every array in ``source`` as a column.

        ``np.load`` materializes whole arrays, so :meth:`stream_import` is not overridden --
        a ``batch_size`` request reads the file whole with a warning.
        """
        with np.load(source) as archive:
            data = {name: archive[name] for name in archive.files}
        if columns is not None:
            data = {name: data[name] for name in columns}
        return data
