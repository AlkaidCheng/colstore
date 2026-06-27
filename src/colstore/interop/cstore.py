"""colstore's own ``.cstore`` file format -- export a selection to a new file.

This is the one target ``saveas`` had no handler for: the native format. With it,
``ds[rows, cols].saveas('out.cstore')`` writes the selected columns and rows to a new
``.cstore``. The write goes through colstore's own editing-frame writer rather than a
materialize-then-encode step, so it pulls in no optional backend and never holds the
whole selection in memory: a whole store -- or a multi-file dataset -- is raw-copied /
merged like :func:`colstore.concat`, and a row/column selection streams in bounded
memory.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import FileFormat, Selection


class CStoreFormat(FileFormat):
    """The native ``.cstore`` format, as a ``saveas`` export target.

    Export (:meth:`to_file`) writes the selection to a new ``.cstore`` through the
    editing frame, so it streams in bounded memory and raw-copies unchanged columns
    (a whole store, or a multi-file dataset, is merged like :func:`colstore.concat`).
    Import is not offered: a ``.cstore`` is already native -- open it with
    :func:`colstore.open`, or :func:`colstore.compact` to rewrite it.
    """

    name: ClassVar[str] = "cstore"
    extensions: ClassVar[frozenset[str]] = frozenset({".cstore"})

    def to_file(
        self, selection: Selection, dest: Any, *, memory_budget: int | None = None, **kwargs: Any
    ) -> None:
        """Write the selected columns and rows to a new ``.cstore`` at ``dest``.

        Streams through the editing frame, raw-copying unchanged columns rather than
        materializing the selection; ``memory_budget`` (bytes; ``None`` uses the
        configured default) bounds peak memory of any streamed batches.
        """
        if kwargs:
            # The native format has no backend to forward extra options to, so an
            # unknown keyword (e.g. a ``memory_budget`` typo) is rejected rather than
            # silently dropped, which would quietly defeat the memory bound.
            raise TypeError(
                "saveas to a .cstore accepts only 'memory_budget'; got unexpected "
                f"keyword argument(s): {', '.join(map(repr, sorted(kwargs)))}."
            )
        from ..frame import ColStoreFrame
        from ..view import edit_row_selection

        rows = edit_row_selection(selection.row_indexer, selection.store.n_rows)
        frame = ColStoreFrame(selection.store, selection.columns, rows)
        frame.write(dest, memory_budget=memory_budget)

    def from_file(self, source: Any, dest: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "a .cstore file is already native; open it with colstore.open() (or "
            "compact() to rewrite it), not ingest()."
        )
