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
import shutil
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


def _caption_div(inner_html: str) -> str:
    """Wrap caption content in the house-style muted-monospace caption div."""
    return (
        '<div style="font-family:ui-monospace,monospace;font-size:90%;color:#57606a;'
        f'margin-bottom:4px">{inner_html}</div>'
    )


def _preview_div(caption: str, table: str) -> str:
    """Wrap a caption and table in the outer ``cstore-preview`` container with the shared style."""
    return f'<div class="cstore-preview">{_STYLE}{caption}{table}</div>'


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
    caption = _caption_div(f"{label_html}{meta}")
    table = (
        f'<table class="cstore-tbl"><thead><tr><th></th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )
    return _preview_div(caption, table)


def _fit_columns(widths: list[int]) -> int:
    """How many leading data columns fit the terminal width (always >= 1).

    ``widths[0]`` is the index column (always shown); the rest are data columns,
    added left to right until the row would exceed the terminal width, reserving
    room for a trailing ``...`` column -- the horizontal fit pandas uses.
    """
    avail = shutil.get_terminal_size((80, 24)).columns
    used = widths[0]
    kept = 0
    for w in widths[1:]:
        if kept >= 1 and used + 2 + w + 5 > avail:  # +2 separator, +5 reserves "  ..."
            break
        used += 2 + w
        kept += 1
    return kept


def render_text(
    label: str,
    total_rows: int | None,
    total_cols: int,
    columns: list[str],
    index: list[int],
    cells: list[list[str]],
) -> str:
    """Render a preview as a fixed-width ASCII table that fits the terminal width."""
    matrix = [["", *columns]]
    for pos, row in zip(index, cells, strict=False):
        matrix.append([str(pos), *row])
    ncol = len(matrix[0])
    widths = [min(max(len(r[j]) for r in matrix), _MAX_COLWIDTH) for j in range(ncol)]
    kept = _fit_columns(widths)
    truncated = kept < ncol - 1
    shown_idx = range(kept + 1)  # the index column plus the data columns that fit
    rows = []
    for r in matrix:
        line = [r[j][: widths[j]].rjust(widths[j]) for j in shown_idx]
        if truncated:
            line.append("...")
        rows.append("  ".join(line))
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


def render_lazy_card_text(label: str, columns: list[str]) -> str:
    """The text twin of :func:`render_lazy_card`: a short note that reads no data."""
    shown, truncated = _shown_columns(columns)
    cols = ", ".join(shown) + (", ..." if truncated else "")
    n = len(columns)
    return (
        f"{label}: lazy selection, {n} column{'' if n == 1 else 's'} [{cols}]\n"
        "call .head() or .evaluate() to read rows"
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
    cap = _caption_div(html.escape(caption))
    table = (
        f'<table class="cstore-tbl"><thead><tr>{head}</tr></thead>' f"<tbody>{body}</tbody></table>"
    )
    return _preview_div(cap, table)


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
