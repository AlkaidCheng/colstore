"""Tabular preview rendering for ``head()`` / ``tail()`` and the rich reprs.

A :class:`Preview` renders as an ASCII table via ``__repr__`` (terminals) and as
an HTML table via ``_repr_html_`` (notebooks), and delegates indexing and array
attributes to the underlying numpy data so it doubles as the values.
``render_html`` / ``render_text`` build the two table forms from numpy values;
``render_lazy_card`` is the HTML shown for a still-lazy view, which must not be
evaluated just to display it.
"""

from __future__ import annotations

import html
import math
from typing import Any

import numpy as np

from . import config

_MAX_COLS = 20  # columns past this are elided with a trailing "..." column
_MAX_COLWIDTH = 32  # cell text wider than this is clipped in the ASCII table

_STYLE = (
    "<style>.cstore-tbl{border-collapse:collapse;font-family:ui-monospace,SFMono-Regular,"
    "Menlo,monospace;font-size:12px}.cstore-tbl td,.cstore-tbl th{border:1px solid #d0d7de;"
    "padding:2px 8px;text-align:right;white-space:nowrap}.cstore-tbl thead th{background:#f6f8fa;"
    "font-weight:600}.cstore-tbl tbody th{background:#f6f8fa;color:#57606a;font-weight:400}</style>"
)


def _decode_bytes(value: Any) -> str:
    try:
        text: str = value.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return repr(value)
    return text


def _format_floats(values: np.ndarray, precision: int) -> list[str]:
    """Fixed ``precision`` decimals, then trim zeros that trail every value in the column.

    A column of integer-valued floats renders as ``1.0`` / ``2.5`` (one decimal); a column
    needing more keeps up to ``precision`` places. ``NaN`` / ``inf`` pass through.
    """
    fmt = f"%.{precision}f"
    out = [fmt % f for f in values.tolist()]
    if not np.isfinite(values).all():
        for idx in np.flatnonzero(~np.isfinite(values)):
            f = float(values[idx])
            out[idx] = "NaN" if math.isnan(f) else ("-inf" if f < 0 else "inf")
    decimals = [i for i, s in enumerate(out) if "." in s]
    while decimals and all(out[i].endswith("0") and out[i][-2] != "." for i in decimals):
        for i in decimals:
            out[i] = out[i][:-1]
    return out


def _format_datetimes(values: np.ndarray) -> list[str]:
    """ISO datetimes, space-separated; the time part is dropped when all are midnight."""
    strs = [str(v) for v in values]
    date_only = all(s == "NaT" or s.endswith("T00:00:00") or "T" not in s for s in strs)
    out = []
    for s in strs:
        if s == "NaT":
            out.append("NaT")
        elif date_only:
            out.append(s.split("T", 1)[0] if "T" in s else s)
        else:
            out.append(s.replace("T", " "))
    return out


def _format_column(values: np.ndarray, precision: int) -> list[str]:
    """Format one column's values to display strings, dispatched on dtype kind."""
    kind = values.dtype.kind
    if kind == "f":
        return _format_floats(values, precision)
    if kind == "M":
        return _format_datetimes(values)
    if kind in ("S", "a"):
        return [_decode_bytes(v) for v in values.tolist()]
    return [str(v) for v in values.tolist()]


def _format_columns(data: np.ndarray, columns: list[str], precision: int) -> list[list[str]]:
    """Per-column display strings; ``data`` is a recarray or a single 1-D array."""
    if data.dtype.names is not None:
        return [_format_column(data[c], precision) for c in columns]
    return [_format_column(data, precision)]


def _shown_columns(columns: list[str]) -> tuple[list[str], bool]:
    """The columns to render and whether the rest were elided."""
    if len(columns) <= _MAX_COLS:
        return columns, False
    return columns[:_MAX_COLS], True


def _caption_meta(total_rows: int | None, total_cols: int, shown: int) -> str:
    if total_rows is None:
        return f"{total_cols} columns &middot; showing {shown}"
    return f"{total_rows:,} rows &times; {total_cols} columns &middot; showing {shown}"


def render_html(
    label: str,
    total_rows: int | None,
    total_cols: int,
    columns: list[str],
    index: list[int],
    cells: list[list[str]],
    safe: list[bool],
) -> str:
    """Render a preview as an HTML table (the notebook repr).

    ``safe[j]`` marks a column whose formatted values cannot contain an HTML
    metacharacter (numeric / datetime), so its cells skip escaping.
    """
    shown_cols, truncated = _shown_columns(columns)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in shown_cols)
    if truncated:
        head += "<th>&hellip;</th>"
    body = []
    for pos, row in zip(index, cells, strict=False):
        tds = "".join(
            f"<td>{v}</td>" if safe[j] else f"<td>{html.escape(v)}</td>"
            for j, v in enumerate(row[: len(shown_cols)])
        )
        if truncated:
            tds += "<td>&hellip;</td>"
        body.append(f"<tr><th>{pos}</th>{tds}</tr>")
    meta = _caption_meta(total_rows, total_cols, len(index))
    label_html = f"<b>{html.escape(label)}</b> &middot; " if label else ""
    caption = (
        '<div style="font-family:ui-monospace,monospace;font-size:90%;color:#57606a;'
        f'margin-bottom:4px">{label_html}{meta}</div>'
    )
    table = (
        f'<table class="cstore-tbl"><thead><tr><th></th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )
    return f'<div class="cstore-preview">{_STYLE}{caption}{table}</div>'


def render_text(
    label: str,
    total_rows: int | None,
    total_cols: int,
    columns: list[str],
    index: list[int],
    cells: list[list[str]],
) -> str:
    """Render a preview as a fixed-width ASCII table (terminal repr)."""
    shown_cols, truncated = _shown_columns(columns)
    header = ["", *shown_cols] + (["..."] if truncated else [])
    matrix = [header]
    for pos, row in zip(index, cells, strict=False):
        vals = list(row[: len(shown_cols)])
        if truncated:
            vals.append("...")
        matrix.append([str(pos), *vals])
    ncol = len(header)
    widths = [0] * ncol
    for r in matrix:
        for j in range(ncol):
            w = len(r[j])
            if w > widths[j]:
                widths[j] = w
    widths = [min(w, _MAX_COLWIDTH) for w in widths]
    rows = ["  ".join(r[j][: widths[j]].rjust(widths[j]) for j in range(ncol)) for r in matrix]
    shown = len(index)
    name = f"{label}: " if label else ""
    if shown == 0:
        return f"[{name}empty, {total_cols} columns]"
    if total_rows is None:
        footer = f"[{name}showing {shown} rows x {total_cols} columns]"
    else:
        footer = f"[{name}{total_rows} rows x {total_cols} columns, showing {shown}]"
    return "\n".join(rows) + "\n" + footer


def render_lazy_card(label: str, columns: list[str]) -> str:
    """An HTML card describing a lazy (unevaluated) view -- reads no data.

    Shown for a view still carrying a ``col()`` / ``query`` predicate: a data
    preview would have to read the predicate columns, which a repr must not
    trigger. The user opts into reading with ``.head()`` / ``.evaluate()``.
    """
    shown = ", ".join(html.escape(c) for c in columns[:12])
    if len(columns) > 12:
        shown += ", ..."
    count = f"{len(columns)} column{'' if len(columns) == 1 else 's'}"
    return (
        '<div style="font-family:ui-monospace,monospace;font-size:90%;color:#57606a;'
        'border:1px solid #d0d7de;border-radius:6px;padding:8px 10px;display:inline-block">'
        f"<b>{html.escape(label)}</b> &middot; lazy selection &middot; {count}"
        f'<br><span style="color:#8b949e">[{shown}]</span><br>'
        "call <code>.head()</code> or <code>.evaluate()</code> to read rows</div>"
    )


def render_table_text(headers: list[str], rows: list[list[str]]) -> str:
    """Render a table of preformatted string cells as fixed-width text (terminal repr).

    The first column is left-aligned (a row label), the rest right-aligned (numbers).
    """
    table = [headers, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    return "\n".join(
        "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)
        )
        for row in table
    )


def render_table_html(caption: str, headers: list[str], rows: list[list[str]]) -> str:
    """Render a table of preformatted string cells as the house-style HTML table.

    Matches the data-preview look (the ``cstore-tbl`` style); the first column of
    each row is a row header, like the preview's positional index.
    """
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        f"<tr><th>{html.escape(row[0])}</th>"
        + "".join(f"<td>{html.escape(c)}</td>" for c in row[1:])
        + "</tr>"
        for row in rows
    )
    cap = (
        '<div style="font-family:ui-monospace,monospace;font-size:90%;color:#57606a;'
        f'margin-bottom:4px">{html.escape(caption)}</div>'
    )
    table = (
        f'<table class="cstore-tbl"><thead><tr>{head}</tr></thead>' f"<tbody>{body}</tbody></table>"
    )
    return f'<div class="cstore-preview">{_STYLE}{cap}{table}</div>'


class Preview:
    """A materialized head/tail peek that renders as a table in both reprs.

    Wraps the underlying recarray (a 1-D array for a single column) and adds the
    ASCII (``__repr__``) and HTML (``_repr_html_``) table forms; indexing,
    ``len``, and unknown attributes (e.g. ``shape``, ``dtype``, ``tolist``)
    delegate to that array, so the object also serves as the raw values.
    """

    __slots__ = ("_columns", "_data", "_index", "_label", "_total_rows")

    def __init__(
        self,
        data: np.ndarray,
        columns: list[str],
        index: list[int],
        total_rows: int | None = None,
        label: str = "",
    ) -> None:
        self._data = data
        self._columns = columns
        self._index = index
        self._total_rows = total_rows
        self._label = label

    def _str_cells(self) -> list[list[str]]:
        """Row-major display strings, formatted per column at the configured precision."""
        cols = _format_columns(self._data, self._columns, config.get_preview_precision())
        return [list(row) for row in zip(*cols, strict=False)] if cols else []

    def _safe_columns(self) -> list[bool]:
        """Per column, whether its formatted values are free of HTML metacharacters."""
        dtype = self._data.dtype
        if dtype.names is not None:
            return [dtype[c].kind not in "USOa" for c in self._columns]
        return [dtype.kind not in "USOa"]

    @property
    def values(self) -> np.ndarray:
        """The underlying numpy array (recarray for a table, 1-D for a column)."""
        return self._data

    def __repr__(self) -> str:
        return render_text(
            self._label,
            self._total_rows,
            len(self._columns),
            self._columns,
            self._index,
            self._str_cells(),
        )

    def _repr_html_(self) -> str:
        return render_html(
            self._label,
            self._total_rows,
            len(self._columns),
            self._columns,
            self._index,
            self._str_cells(),
            self._safe_columns(),
        )

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        # Delegate value attributes (shape, dtype, tolist, ...) to the data; the
        # leading-underscore guard keeps slots/dunders from recursing here.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._data, name)
