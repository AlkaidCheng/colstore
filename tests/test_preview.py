"""Tests for head()/tail() previews, the preview config, and _repr_html_."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import colstore
from colstore import ColStoreDataset, col, config
from colstore._render import Preview


@pytest.fixture(autouse=True)
def _reset_preview_config():
    rows = config.get_preview_rows()
    limit = config.get_preview_memory_limit()
    precision = config.get_preview_precision()
    yield
    config.set_preview_rows(rows)
    config.set_preview_memory_limit(limit)
    config.set_preview_precision(precision)


@pytest.fixture()
def pstore(tmp_path):
    data = {"pt": np.arange(20.0), "q": np.arange(20, dtype=np.int32)}
    path = tmp_path / "p.cstore"
    colstore.store(data, path, show_progress=False).close()
    ds = colstore.open(path)
    yield ds
    ds.close()


def test_head_tail_reader(pstore):
    assert pstore.head(3)["pt"].tolist() == [0.0, 1.0, 2.0]
    assert pstore.tail(3)["pt"].tolist() == [17.0, 18.0, 19.0]


def test_head_default_uses_config_rows(pstore):
    assert pstore.head().shape[0] == 10  # the default
    config.set_preview_rows(4)
    assert pstore.head().shape[0] == 4
    assert pstore.head(7).shape[0] == 7  # an explicit n overrides the default


def test_head_clamps_to_n_rows(pstore):
    assert pstore.head(1000).shape[0] == 20
    assert pstore.tail(1000).shape[0] == 20


def test_head_tail_on_slice_view(pstore):
    assert pstore[5:15].head(3)["pt"].tolist() == [5.0, 6.0, 7.0]
    assert pstore[5:15].tail(3)["pt"].tolist() == [12.0, 13.0, 14.0]


def test_head_tail_on_filtered_view(pstore):
    hot = pstore[col("pt") > 10]  # pt 11..19, in order
    assert hot.head(3)["pt"].tolist() == [11.0, 12.0, 13.0]
    assert hot.tail(3)["pt"].tolist() == [17.0, 18.0, 19.0]


def test_head_tail_on_column_view(pstore):
    assert pstore["pt"].head(3).tolist() == [0.0, 1.0, 2.0]
    assert pstore["pt"].tail(3).tolist() == [17.0, 18.0, 19.0]
    assert pstore[col("pt") > 10, "pt"].head(2).tolist() == [11.0, 12.0]


def test_preview_memory_warning(pstore):
    config.set_preview_memory_limit(1)  # 1 byte -> any non-empty preview warns
    with pytest.warns(UserWarning, match="materialize"):
        pstore.head(5)


def test_preview_memory_limit_zero_disables_warning(pstore):
    config.set_preview_memory_limit(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pstore.head(1_000_000)
        assert not any("materialize" in str(w.message) for w in caught)


def test_preview_config_validation():
    with pytest.raises(ValueError, match=">= 0"):
        config.set_preview_rows(-1)
    with pytest.raises(ValueError, match=">= 0"):
        config.set_preview_memory_limit(-1)


def test_repr_html_reader(pstore):
    rendered = pstore._repr_html_()
    assert rendered is not None
    assert rendered.startswith("<div")
    assert "20 rows" in rendered  # the shape caption
    assert "<table" in rendered  # the preview table


def test_repr_html_lazy_view_shows_card_without_evaluating(pstore, monkeypatch):
    # Displaying a lazy (unevaluated predicate) view must NOT read its predicate
    # columns; it shows a metadata card, not a data preview.
    def _no_read(self, name):
        raise AssertionError("a lazy repr must not read predicate columns")

    monkeypatch.setattr(type(pstore), "_read_query_column", _no_read)
    rendered = pstore[col("pt") > 10]._repr_html_()
    assert rendered is not None
    assert "lazy" in rendered
    assert "<table" not in rendered  # no data preview


def test_repr_html_lazy_column_view_shows_card(pstore):
    rendered = pstore[col("pt") > 10, "pt"]._repr_html_()
    assert rendered is not None and "lazy" in rendered and "<table" not in rendered


def test_repr_html_evaluated_view_shows_preview(pstore):
    # Once evaluated, the selection is a concrete mask -> a real data preview.
    rendered = pstore[col("pt") > 10].evaluate()._repr_html_()
    assert rendered is not None and "<table" in rendered


def test_repr_html_concrete_views_show_preview(pstore):
    # A slice view and a whole-column view carry no predicate -> they preview.
    assert "<table" in (pstore[5:15]._repr_html_() or "")
    assert "<table" in (pstore["pt"]._repr_html_() or "")


def test_repr_html_falls_back_to_none(pstore, monkeypatch):
    # If building the preview fails for any reason, _repr_html_ returns None so
    # Jupyter falls back to the text repr.
    def _boom(self, *args, **kwargs):
        raise RuntimeError("cannot build preview")

    monkeypatch.setattr(type(pstore), "_build_recarray", _boom)
    assert pstore._repr_html_() is None


def test_head_returns_dual_repr_preview(pstore):
    out = pstore.head(3)
    assert isinstance(out, Preview)
    # Indexing and array attributes delegate to the underlying numpy data.
    assert out["pt"].tolist() == [0.0, 1.0, 2.0]
    assert out.shape == (3,)
    assert out.values.dtype.names == ("pt", "q")
    # Two repr modes: HTML for notebooks, plain-text ASCII for terminals.
    assert "<table" in out._repr_html_()
    text = repr(out)
    assert "<" not in text  # no HTML in the terminal repr
    assert "pt" in text and "0.0" in text


def test_ascii_repr_is_a_plain_text_table(pstore):
    text = repr(pstore.head(3))
    lines = text.splitlines()
    assert "pt" in lines[0] and "q" in lines[0]  # header row
    assert lines[1].split() == ["0", "0.0", "0"]  # row 0: index, pt, q
    assert lines[3].split() == ["2", "2.0", "2"]  # row 2
    assert "rows x" in lines[-1]  # footer caption


def test_ascii_repr_decodes_bytes(tmp_path):
    colstore.store(
        {"region": np.array(["SR", "CR"], dtype="S2")}, tmp_path / "b.cstore", show_progress=False
    ).close()
    with colstore.open(tmp_path / "b.cstore") as ds:
        text = repr(ds.head())
        assert "SR" in text and "b'SR'" not in text  # decoded, not the bytes repr


def test_repr_html_uses_builtin_table_markup(pstore):
    rendered = pstore._repr_html_()
    assert 'class="cstore-tbl"' in rendered
    assert "<table" in rendered and "<td>0.0</td>" in rendered


def test_repr_html_decodes_bytes_columns(tmp_path):
    colstore.store(
        {"region": np.array(["SR", "CR"], dtype="S2")}, tmp_path / "b.cstore", show_progress=False
    ).close()
    with colstore.open(tmp_path / "b.cstore") as ds:
        rendered = ds._repr_html_()
        assert "<td>SR</td>" in rendered  # decoded, not b'SR'
        assert "b&#x27;SR&#x27;" not in rendered


def test_head_tail_on_multifile_dataset(tmp_path):
    for i in range(2):
        colstore.store(
            {"pt": np.arange(5.0) + i * 100}, tmp_path / f"d{i}.cstore", show_progress=False
        ).close()
    with colstore.open([str(tmp_path / "d0.cstore"), str(tmp_path / "d1.cstore")]) as ds:
        assert isinstance(ds, ColStoreDataset)
        assert ds.head(2)["pt"].tolist() == [0.0, 1.0]
        assert ds.tail(2)["pt"].tolist() == [103.0, 104.0]
        assert (ds._repr_html_() or "").startswith("<div")


# ---- float precision and special dtypes ------------------------------------


def _store_open(tmp_path, name, data):
    colstore.store(data, tmp_path / name, show_progress=False).close()
    return colstore.open(tmp_path / name)


def test_float_precision_default_is_six(tmp_path):
    assert config.get_preview_precision() == 6
    with _store_open(tmp_path, "f.cstore", {"x": np.array([1 / 3, 1.0])}) as ds:
        text = repr(ds.head())
        assert "0.333333" in text  # six decimals
        assert "1.000000" in text  # padded to the column's six places (0.333333 needs them)


def test_set_preview_precision(tmp_path):
    with _store_open(tmp_path, "f.cstore", {"x": np.array([1 / 3, 2 / 3])}) as ds:
        config.set_preview_precision(2)
        text = repr(ds.head())
        assert "0.33" in text and "0.333" not in text


def test_float_column_trims_common_trailing_zeros(tmp_path):
    with _store_open(tmp_path, "f.cstore", {"x": np.array([1.0, 2.5, 3.0])}) as ds:
        text = repr(ds.head())
        assert "1.0" in text and "2.5" in text  # one decimal, not six
        assert "1.000000" not in text


def test_preview_precision_validation():
    with pytest.raises(ValueError, match=">= 0"):
        config.set_preview_precision(-1)


def test_datetime_columns_render(tmp_path):
    data = {
        "d": np.array(["2020-01-01", "2021-06-15"], dtype="datetime64[s]"),
        "t": np.array(["2020-01-01T09:30:00", "2020-01-01T17:45:30"], dtype="datetime64[s]"),
    }
    with _store_open(tmp_path, "dt.cstore", data) as ds:
        text = repr(ds.head())
        assert "00:00:00" not in text  # the all-midnight column drops the time
        assert "2020-01-01 09:30:00" in text  # the timed column is space-separated
        assert "<td>2020-01-01</td>" in ds.head()._repr_html_()


def test_datetime_nat(tmp_path):
    data = {"t": np.array(["2020-01-01T09:30:00", "NaT"], dtype="datetime64[s]")}
    with _store_open(tmp_path, "nat.cstore", data) as ds:
        assert "NaT" in repr(ds.head())


def test_repr_html_escapes_string_columns(tmp_path):
    # String/bytes columns with HTML metacharacters are still escaped; numeric
    # columns (which cannot contain them) skip escaping.
    data = {"s": np.array(["a<b", "x&y"], dtype="S8"), "n": np.array([1.0, 2.0])}
    with _store_open(tmp_path, "h.cstore", data) as ds:
        rendered = ds.head()._repr_html_()
        assert "<td>a&lt;b</td>" in rendered
        assert "<td>x&amp;y</td>" in rendered


# ---- view repr: a formatted table (Reader/Dataset keep the compact repr) -----


def test_view_repr_is_a_table(pstore):
    r = repr(pstore[:5])
    assert "pt" in r and "q" in r  # the data, not the programmatic form
    assert "showing" in r and "columns" in r  # the table footer
    assert "TableView(columns=" not in r
    rc = repr(pstore["pt"])  # a single column reprs as a one-column table
    assert "pt" in rc and "showing" in rc and "ColumnView(column=" not in rc


def test_lazy_view_repr_is_a_card_without_reading(pstore):
    r = repr(pstore[col("pt") > 5])
    assert "lazy selection" in r and ".head()" in r and ".evaluate()" in r
    assert "showing" not in r  # no data table -- the predicate was not evaluated


def test_reader_repr_unchanged(pstore):
    r = repr(pstore)
    assert r.startswith("ColStoreReader(") and "shape=" in r and "columns=" in r


def test_view_repr_fits_terminal_width(tmp_path, monkeypatch):
    data = {f"c{i}": np.arange(4, dtype=np.float64) for i in range(30)}
    ds = colstore.store(data, tmp_path / "w.cstore", show_progress=False)
    try:
        monkeypatch.setenv("COLUMNS", "40")
        narrow = repr(ds[:4])
        assert "..." in narrow and "c29" not in narrow  # trailing columns elided
        monkeypatch.setenv("COLUMNS", "10000")
        wide = repr(ds[:4])
        assert "c29" in wide and "..." not in wide  # all columns fit a wide terminal
    finally:
        ds.close()
