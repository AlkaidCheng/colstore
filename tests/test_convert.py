"""Tests for the batch ``colstore.convert`` orchestration: directions, multi-file
output naming (auto / merge / template / rename), and the overwrite policy. NPZ is
used as the foreign format so no optional backend is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import colstore


def _npz(tmp_path, name, ids):
    path = tmp_path / f"{name}.npz"
    np.savez(path, id=np.asarray(ids, dtype=np.int64), v=(np.asarray(ids) * 1.5))
    return path


# ---- direction inference ----------------------------------------------------


def test_convert_single_foreign_to_cstore_returns_reader(tmp_path):
    src = _npz(tmp_path, "a", [0, 1, 2])
    out = colstore.convert(src, tmp_path / "a.cstore")
    assert isinstance(out, colstore.ColStoreReader)
    assert out.array("id").tolist() == [0, 1, 2]


def test_convert_cstore_to_foreign_returns_path(tmp_path):
    colstore.store({"id": np.arange(3, dtype=np.int64)}, tmp_path / "s.cstore", show_progress=False)
    out = colstore.convert(tmp_path / "s.cstore", tmp_path / "s.npz")
    assert isinstance(out, Path) and out.exists()


def test_convert_foreign_to_foreign_rejected(tmp_path):
    src = _npz(tmp_path, "a", [0, 1, 2])
    with pytest.raises(ValueError, match="one endpoint"):
        colstore.convert(src, tmp_path / "a.json")


# ---- output naming: auto / template / rename --------------------------------


def test_convert_auto_names_by_swapping_extension(tmp_path):
    _npz(tmp_path, "data", [1, 2])
    colstore.convert(tmp_path / "data.npz")
    assert (tmp_path / "data.cstore").exists()


def test_convert_glob_one_to_one_returns_list(tmp_path):
    for name in ("a", "b", "c"):
        _npz(tmp_path, name, [1, 2])
    results = colstore.convert(str(tmp_path / "*.npz"))
    assert isinstance(results, list) and len(results) == 3
    for name in ("a", "b", "c"):
        assert (tmp_path / f"{name}.cstore").exists()


def test_convert_template_names(tmp_path):
    for name in ("a", "b"):
        _npz(tmp_path, name, [1, 2])
    colstore.convert(str(tmp_path / "*.npz"), str(tmp_path / "run_{index}.cstore"))
    assert (tmp_path / "run_0.cstore").exists() and (tmp_path / "run_1.cstore").exists()


def test_convert_rename_callable(tmp_path):
    _npz(tmp_path, "a", [1, 2])
    colstore.convert(str(tmp_path / "*.npz"), rename=lambda stem: stem.upper() + "_x")
    assert (tmp_path / "A_x.cstore").exists()  # in the source's directory


def test_convert_rename_mapping_keeps_unlisted(tmp_path):
    _npz(tmp_path, "a", [1, 2])
    _npz(tmp_path, "b", [3, 4])
    colstore.convert([tmp_path / "a.npz", tmp_path / "b.npz"], rename={"a": "alpha"})
    assert (tmp_path / "alpha.cstore").exists()  # mapped
    assert (tmp_path / "b.cstore").exists()  # unlisted stem keeps its name


def test_convert_output_dir(tmp_path):
    _npz(tmp_path, "a", [1, 2])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    colstore.convert(tmp_path / "a.npz", output_dir=out_dir)
    assert (out_dir / "a.cstore").exists()


# ---- merge ------------------------------------------------------------------


def test_convert_merges_foreign_glob_into_one_cstore(tmp_path):
    _npz(tmp_path, "a", [0, 1, 2])
    _npz(tmp_path, "b", [3, 4, 5])
    out = colstore.convert(str(tmp_path / "*.npz"), tmp_path / "all.cstore")
    assert isinstance(out, colstore.ColStoreReader)
    assert out.n_rows == 6
    assert out.array("id").tolist() == [0, 1, 2, 3, 4, 5]


def test_convert_merges_cstore_files(tmp_path):
    colstore.store({"id": np.arange(3, dtype=np.int64)}, tmp_path / "a.cstore", show_progress=False)
    colstore.store(
        {"id": np.arange(3, 6, dtype=np.int64)}, tmp_path / "b.cstore", show_progress=False
    )
    out = colstore.convert([tmp_path / "a.cstore", tmp_path / "b.cstore"], tmp_path / "m.cstore")
    assert out.n_rows == 6 and out.array("id").tolist() == [0, 1, 2, 3, 4, 5]


def test_convert_merge_honors_on_mismatch_drop(tmp_path):
    np.savez(tmp_path / "a.npz", id=np.arange(3, dtype=np.int64), sel=np.array([True, False, True]))
    np.savez(tmp_path / "b.npz", id=np.arange(3, 6, dtype=np.int64), sel=np.full(3, np.nan))
    with pytest.warns(RuntimeWarning, match="sel"):
        out = colstore.convert(str(tmp_path / "*.npz"), tmp_path / "all.cstore", on_mismatch="drop")
    assert out.columns == ["id"] and out.n_rows == 6


# ---- target format and overwrite --------------------------------------------


def test_convert_cstore_to_foreign_one_to_one_with_format(tmp_path):
    for name in ("a", "b"):
        colstore.store(
            {"id": np.arange(2, dtype=np.int64)}, tmp_path / f"{name}.cstore", show_progress=False
        )
    results = colstore.convert(str(tmp_path / "*.cstore"), format="npz")
    assert isinstance(results, list) and len(results) == 2
    assert (tmp_path / "a.npz").exists() and (tmp_path / "b.npz").exists()


def test_convert_cstore_source_without_target_raises(tmp_path):
    colstore.store({"id": np.arange(2, dtype=np.int64)}, tmp_path / "a.cstore", show_progress=False)
    with pytest.raises(ValueError, match="needs a target format"):
        colstore.convert(tmp_path / "a.cstore")


def test_convert_overwrite_policy(tmp_path):
    src = _npz(tmp_path, "a", [1, 2])
    colstore.convert(src, tmp_path / "a.cstore").close()
    with pytest.raises(FileExistsError):
        colstore.convert(src, tmp_path / "a.cstore")
    out = colstore.convert(src, tmp_path / "a.cstore", overwrite=True)
    assert out.n_rows == 2


def test_convert_empty_glob_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files matched"):
        colstore.convert(str(tmp_path / "nope_*.npz"))


def test_convert_colliding_outputs_raise_before_writing(tmp_path):
    # Two same-stem inputs from different directories resolve to one templated output.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    colstore.store(
        {"v": np.array([1, 1])}, tmp_path / "a" / "run.cstore", show_progress=False
    ).close()
    colstore.store(
        {"v": np.array([2, 2])}, tmp_path / "b" / "run.cstore", show_progress=False
    ).close()
    with pytest.raises(ValueError, match="same output"):
        colstore.convert(
            [tmp_path / "a" / "run.cstore", tmp_path / "b" / "run.cstore"],
            str(tmp_path / "{stem}.npz"),
        )
    assert not (tmp_path / "run.npz").exists()  # nothing written before the guard fires


def test_convert_columns_projects_symmetrically(tmp_path):
    colstore.store(
        {"id": np.arange(4, dtype=np.int64), "x": np.arange(4) * 1.0, "y": np.arange(4) * 2.0},
        tmp_path / "s.cstore",
        show_progress=False,
    ).close()
    # export projection
    colstore.convert(tmp_path / "s.cstore", tmp_path / "s.npz", columns=["id", "x"])
    back = colstore.convert(tmp_path / "s.npz", tmp_path / "back.cstore")
    assert back.columns == ["id", "x"]
    # merge projection drops an excluded column that differs across inputs
    colstore.store(
        {"id": np.arange(2, dtype=np.int64), "k": np.arange(2)},
        tmp_path / "m1.cstore",
        show_progress=False,
    ).close()
    colstore.store(
        {"id": np.arange(2, dtype=np.int64), "j": np.arange(2)},
        tmp_path / "m2.cstore",
        show_progress=False,
    ).close()
    merged = colstore.convert(
        [tmp_path / "m1.cstore", tmp_path / "m2.cstore"], tmp_path / "m.cstore", columns=["id"]
    )
    assert merged.columns == ["id"] and merged.n_rows == 4


def test_convert_repeated_input_in_merge_warns(tmp_path):
    _npz(tmp_path, "a", [0, 1, 2])
    _npz(tmp_path, "b", [3, 4, 5])
    with pytest.warns(RuntimeWarning, match="same input"):
        out = colstore.convert(
            [tmp_path / "a.npz", tmp_path / "a.npz", tmp_path / "b.npz"], tmp_path / "m.cstore"
        )
    assert out.n_rows == 9  # a's rows included twice, plus b's
