"""Tests for the ``colstore convert`` CLI command (:mod:`colstore.cli.convert`).

The command is driven through :func:`colstore.cli.main` (the installed entry point) so the
argument parsing, exit codes, and printed plan are exercised exactly as a shell would.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

import colstore
from colstore.cli import main


def _store(path, **columns):
    if not columns:
        columns = {"id": np.arange(5, dtype=np.int64), "x": (np.arange(5) * 1.5)}
    colstore.store(columns, path, show_progress=False).close()
    return path


def _run(argv):
    return main(["convert", *[str(a) for a in argv]])


# ---- Direction and naming --------------------------------------------------


def test_export_cstore_to_parquet(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    src = _store(tmp_path / "s.cstore")
    code = _run([src, "-o", tmp_path / "s.parquet"])
    assert code == 0
    assert "s.cstore -> " in capsys.readouterr().out
    assert pq.read_table(str(tmp_path / "s.parquet")).num_rows == 5


def test_import_auto_named_output(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"id": np.arange(4, dtype=np.int64)}), str(tmp_path / "d.parquet"))
    assert _run([tmp_path / "d.parquet"]) == 0  # -> d.cstore beside it
    assert colstore.open(tmp_path / "d.cstore").array("id").tolist() == [0, 1, 2, 3]


def test_merge_many_into_one(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    for i in range(3):
        _store(tmp_path / f"f{i}.cstore", id=np.arange(5, dtype=np.int64) + i * 10)
    code = _run(
        [
            tmp_path / "f0.cstore",
            tmp_path / "f1.cstore",
            tmp_path / "f2.cstore",
            "-o",
            tmp_path / "all.parquet",
        ]
    )
    assert code == 0
    assert "(merge) 3 files" in capsys.readouterr().out
    assert pq.read_table(str(tmp_path / "all.parquet")).num_rows == 15


def test_template_one_to_one(tmp_path):
    for i in range(2):
        _store(tmp_path / f"g{i}.cstore")
    code = _run(
        [tmp_path / "g0.cstore", tmp_path / "g1.cstore", "-o", str(tmp_path / "out_{stem}.npz")]
    )
    assert code == 0
    assert (tmp_path / "out_g0.npz").exists() and (tmp_path / "out_g1.npz").exists()


def test_rename_and_output_dir(tmp_path):
    _store(tmp_path / "raw.cstore")
    dest_dir = tmp_path / "out"
    dest_dir.mkdir()
    code = _run(
        [
            tmp_path / "raw.cstore",
            "--format",
            "npz",
            "--rename",
            "raw=clean",
            "--output-dir",
            dest_dir,
        ]
    )
    assert code == 0
    assert (dest_dir / "clean.npz").exists()


# ---- Options ---------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path, capsys):
    _store(tmp_path / "s.cstore")
    code = _run([tmp_path / "s.cstore", "-o", tmp_path / "s.npz", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Would convert" in out and "nothing was written" in out
    assert not (tmp_path / "s.npz").exists()


def test_columns_projection_on_export(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    _store(
        tmp_path / "s.cstore",
        id=np.arange(5, dtype=np.int64),
        x=np.arange(5) * 1.0,
        y=np.arange(5) * 2.0,
    )
    assert _run([tmp_path / "s.cstore", "-o", tmp_path / "s.parquet", "--columns", "id,x"]) == 0
    assert pq.read_table(str(tmp_path / "s.parquet")).column_names == ["id", "x"]


def test_columns_projection_on_import(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"a": np.arange(4, dtype=np.int64), "b": np.arange(4, dtype=np.int64)}),
        str(tmp_path / "d.parquet"),
    )
    assert _run([tmp_path / "d.parquet", "-o", tmp_path / "d.cstore", "--columns", "a"]) == 0
    assert colstore.open(tmp_path / "d.cstore").columns == ["a"]


def test_dtype_override_on_import(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"flag": np.arange(4, dtype=np.int64) % 2}), str(tmp_path / "d.parquet")
    )
    assert _run([tmp_path / "d.parquet", "-o", tmp_path / "d.cstore", "--dtype", "flag=bool"]) == 0
    assert colstore.open(tmp_path / "d.cstore").dtypes["flag"] == np.bool_


def test_batch_size_int_and_byte_budget(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    _store(tmp_path / "s.cstore", id=np.arange(2000, dtype=np.int64))
    assert (
        _run([tmp_path / "s.cstore", "-o", tmp_path / "rows.parquet", "--batch-size", "500"]) == 0
    )
    assert (
        _run([tmp_path / "s.cstore", "-o", tmp_path / "bytes.parquet", "--batch-size", "16 KiB"])
        == 0
    )
    assert pq.ParquetFile(str(tmp_path / "rows.parquet")).num_row_groups > 1


def test_overwrite(tmp_path):
    _store(tmp_path / "s.cstore")
    assert _run([tmp_path / "s.cstore", "-o", tmp_path / "s.npz"]) == 0
    assert _run([tmp_path / "s.cstore", "-o", tmp_path / "s.npz"]) == 1  # exists, no overwrite
    assert _run([tmp_path / "s.cstore", "-o", tmp_path / "s.npz", "--overwrite"]) == 0


def test_on_mismatch_drop(tmp_path):
    _store(tmp_path / "a.cstore", id=np.arange(3, dtype=np.int64), x=np.arange(3) * 1.0)
    _store(tmp_path / "b.cstore", id=np.arange(3, dtype=np.int64))  # missing x
    code = _run(
        [
            tmp_path / "a.cstore",
            tmp_path / "b.cstore",
            "-o",
            tmp_path / "m.cstore",
            "--on-mismatch",
            "drop",
        ]
    )
    assert code == 0
    assert colstore.open(tmp_path / "m.cstore").columns == ["id"]


# ---- Errors and exit codes -------------------------------------------------


def test_no_cstore_endpoint_errors(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"a": [1, 2]}), str(tmp_path / "d.parquet"))
    code = _run([tmp_path / "d.parquet", "-o", tmp_path / "e.parquet"])
    captured = capsys.readouterr()
    assert code == 1
    assert "one endpoint to be a .cstore" in captured.err
    assert captured.out == ""  # nothing announced before the failure


def test_no_files_matched_errors(tmp_path, capsys):
    code = _run([str(tmp_path / "missing_*.cstore"), "-o", tmp_path / "x.npz"])
    assert code == 1
    assert "no files matched" in capsys.readouterr().err


def test_malformed_dtype_is_usage_error(tmp_path):
    _store(tmp_path / "s.cstore")
    with pytest.raises(SystemExit) as exc:  # argparse exits 2 on a bad argument
        _run([tmp_path / "s.cstore", "-o", tmp_path / "s.npz", "--dtype", "bogus"])
    assert exc.value.code == 2


def test_colliding_outputs_rejected(tmp_path, capsys):
    # Two inputs with the same stem in different directories would collide on one output.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _store(tmp_path / "a" / "run.cstore")
    _store(tmp_path / "b" / "run.cstore")
    out = tmp_path / "out"
    out.mkdir()
    code = _run(
        [
            tmp_path / "a" / "run.cstore",
            tmp_path / "b" / "run.cstore",
            "--format",
            "npz",
            "--output-dir",
            out,
        ]
    )
    assert code == 1
    assert "same output" in capsys.readouterr().err
    assert list(out.iterdir()) == []  # nothing written -- the collision is caught before any write


@pytest.mark.parametrize(
    "argv, needle",
    [
        (["--format", "nosuchformat"], "unknown format"),  # KeyError from the registry
        (["-o", "out_{bad}.parquet"], "unknown field"),  # KeyError from the template
        (["-o", "out_{0}.parquet"], "positional field"),  # IndexError from the template
    ],
)
def test_user_errors_are_clean_not_tracebacks(tmp_path, capsys, argv, needle):
    _store(tmp_path / "s.cstore")
    code = _run([tmp_path / "s.cstore", *argv])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("convert:") and needle in captured.err
    assert "Traceback" not in captured.err


def test_huge_batch_size_on_import_is_clean(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"id": np.arange(5, dtype=np.int64)}), str(tmp_path / "d.parquet"))
    code = _run([tmp_path / "d.parquet", "-o", tmp_path / "d.cstore", "--batch-size", "9" * 21])
    captured = capsys.readouterr()
    assert code == 1 and captured.err.startswith("convert:") and "Traceback" not in captured.err


def test_merge_columns_excludes_unstorable_column(tmp_path):
    # A column excluded by --columns must not block a merge, matching the single-file behavior.
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    for name in ("p1", "p2"):
        pq.write_table(
            pa.table({"id": np.arange(3, dtype=np.int64), "bad": pa.array([[1], [2], [3]])}),
            str(tmp_path / f"{name}.parquet"),
        )
    code = _run(
        [
            tmp_path / "p1.parquet",
            tmp_path / "p2.parquet",
            "-o",
            tmp_path / "m.cstore",
            "--columns",
            "id",
        ]
    )
    assert code == 0
    merged = colstore.open(tmp_path / "m.cstore")
    assert merged.columns == ["id"] and merged.n_rows == 6


def test_merge_columns_over_mismatched_excluded_column(tmp_path):
    # id/x are shared; a and b differ but are excluded, so strict merge still succeeds.
    _store(
        tmp_path / "c1.cstore",
        id=np.arange(3, dtype=np.int64),
        x=np.arange(3) * 1.0,
        a=np.arange(3),
    )
    _store(
        tmp_path / "c2.cstore",
        id=np.arange(3, dtype=np.int64),
        x=np.arange(3) * 1.0,
        b=np.arange(3),
    )
    code = _run(
        [
            tmp_path / "c1.cstore",
            tmp_path / "c2.cstore",
            "-o",
            tmp_path / "m.cstore",
            "--columns",
            "id,x",
        ]
    )
    assert code == 0
    assert colstore.open(tmp_path / "m.cstore").columns == ["id", "x"]


def test_mixed_format_inputs_merge(tmp_path):
    # A merge infers each input's format from its own extension.
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    _store(tmp_path / "a.cstore", id=np.arange(3, dtype=np.int64))
    pq.write_table(pa.table({"id": np.arange(3, 6, dtype=np.int64)}), str(tmp_path / "b.parquet"))
    np.savez(tmp_path / "c.npz", id=np.arange(6, 9, dtype=np.int64))
    code = _run(
        [
            tmp_path / "a.cstore",
            tmp_path / "b.parquet",
            tmp_path / "c.npz",
            "-o",
            tmp_path / "m.cstore",
        ]
    )
    assert code == 0
    assert colstore.open(tmp_path / "m.cstore").array("id").tolist() == list(range(9))


def test_repeated_input_in_merge_warns_but_succeeds(tmp_path):
    _store(tmp_path / "a.cstore", id=np.arange(3, dtype=np.int64))
    _store(tmp_path / "b.cstore", id=np.arange(3, 6, dtype=np.int64))
    with pytest.warns(RuntimeWarning, match="same input more than once"):
        code = _run(
            [
                tmp_path / "a.cstore",
                tmp_path / "a.cstore",
                tmp_path / "b.cstore",
                "-o",
                tmp_path / "m.cstore",
            ]
        )
    assert code == 0
    assert colstore.open(tmp_path / "m.cstore").n_rows == 9  # a's rows appear twice, plus b's


def test_repeated_input_one_to_one_collides(tmp_path):
    # Auto-named, both copies resolve to the same output -> the collision guard rejects it.
    _store(tmp_path / "a.cstore")
    code = _run([tmp_path / "a.cstore", tmp_path / "a.cstore", "--format", "npz"])
    assert code == 1


def test_help_shows_option_defaults(capsys):
    with pytest.raises(SystemExit) as exc:  # --help exits 0
        _run(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "(default: strict)" in out  # --on-mismatch
    assert "(default: False)" in out  # a flag such as --overwrite


def test_key_value_helper():
    from colstore.cli._common import key_value

    assert key_value("flag=bool") == ("flag", "bool")
    assert key_value("k=") == ("k", "")  # an empty value is allowed
    for bad in ("noequals", "=value"):
        with pytest.raises(argparse.ArgumentTypeError):
            key_value(bad)
