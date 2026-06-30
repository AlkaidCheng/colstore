"""NPZ file format: exchange colstore data with NumPy ``.npz`` archives.

An ``.npz`` archive is a zip of named arrays, which maps directly onto colstore's
columns -- one array per column. Export writes each selected column as a named
array; import reads every array in the archive as a column (all must share a
length and a fixed-width dtype, as for any colstore write).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ._convert import store_columns
from ._stream_import import warn_whole_file
from .base import FileFormat, Selection

if TYPE_CHECKING:
    from ..reader import ColStoreReader


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

    def from_file(
        self,
        source: Any,
        dest: Any,
        *,
        batch_size: int | str | None = None,
        compact: bool = True,
        **kwargs: Any,
    ) -> ColStoreReader:
        """Read every array in ``source`` as a column and write a ``.cstore`` at ``dest``.

        Extra keyword arguments pass through to :func:`colstore.store` (e.g.
        ``mode="recreate"`` to overwrite an existing ``dest``). ``batch_size`` is accepted
        for a uniform :func:`colstore.convert` surface but ``np.load`` materializes whole
        arrays, so the file is read whole with a warning.
        """
        if batch_size is not None:
            warn_whole_file(source, "np.load materializes whole arrays")
        with np.load(source) as archive:
            columns = {name: archive[name] for name in archive.files}
        return store_columns(columns, dest, **kwargs)
