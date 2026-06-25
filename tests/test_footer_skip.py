"""Tests for Stage B: read-side record skipping driven by the statistics footer.

A ``col(name) <op> scalar`` filter on a multi-record file written with
``statistics=True`` consults each record's ``[min, max]`` and reads only the
records that can contain a matching row. The result must be identical to reading
in full; these tests check that bit-for-bit and that the skip engages only where
it is sound.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore import col


@pytest.fixture
def records():
    rng = np.random.default_rng(0)
    out = []
    for i in range(6):
        lo, n = i * 100, 100
        out.append(
            {
                "x": np.arange(lo, lo + n, dtype=np.int64),  # clustered, disjoint per record
                "r": rng.integers(0, 600, n, dtype=np.int64),  # random, spans the full range
                "f": np.arange(lo, lo + n, dtype=np.float64) + (np.nan if i == 2 else 0.0),
                "s": np.array([f"k{(lo + j) % 7}" for j in range(n)]),
            }
        )
    return out


@pytest.fixture
def pair(tmp_path, records):
    """A statistics file (skip active) and an identical no-statistics file."""

    def build(name, stats):
        path = tmp_path / name
        with colstore.create(path, statistics=stats) as writer:
            for record in records:
                writer.write(record)
        return colstore.open(path)

    stats, plain = build("stats.cstore", True), build("plain.cstore", False)
    yield stats, plain
    stats.close()
    plain.close()


@pytest.mark.parametrize(
    "name, predicate, prunes",
    [
        ("x>250", col("x") > 250, True),
        ("x>=300", col("x") >= 300, True),
        ("x<150", col("x") < 150, True),
        ("x<=99", col("x") <= 99, True),
        ("x==255", col("x") == 255, True),
        ("250<x", 250 < col("x"), True),  # reflected form  # noqa: SIM300
        ("x>1000", col("x") > 1000, True),  # prunes every record -> empty, no read
        ("f>250", col("f") > 250.0, True),  # the NaN record stays non-prunable
        ("x!=255", col("x") != 255, False),  # != does not prune
        ("x>-1", col("x") > -1, False),  # nothing prunes
        ("r>500", col("r") > 500, False),  # random data: every record survives
        ("s==k3", col("s") == "k3", False),  # string column: not prunable
        ("and", (col("x") > 250) & (col("x") < 400), False),  # not a single comparison
    ],
)
def test_skip_matches_full_read(pair, name, predicate, prunes):
    stats, plain = pair
    assert (stats._try_skip_query_mask(predicate) is not None) is prunes
    for column in ("x", "r", "f", "s"):
        skipped = stats[predicate, column].array()
        full = plain[predicate, column].array()
        if column == "f":
            np.testing.assert_array_equal(np.isnan(skipped), np.isnan(full))
            mask = ~np.isnan(full)
            np.testing.assert_array_equal(skipped[mask], full[mask])
        else:
            np.testing.assert_array_equal(skipped, full)


def test_skip_reads_only_surviving_records(tmp_path, records):
    # A predicate that prunes every record returns an empty result without reading.
    with colstore.create(tmp_path / "s.cstore", statistics=True) as writer:
        for record in records:
            writer.write(record)
    reader = colstore.open(tmp_path / "s.cstore")
    mask = reader._try_skip_query_mask(col("x") > 10_000)
    assert mask is not None and not mask.any()  # all pruned, nothing matches
    assert reader[col("x") > 10_000, "x"].array().size == 0
    reader.close()


def test_no_skip_without_statistics(tmp_path, records):
    with colstore.create(tmp_path / "n.cstore") as writer:  # statistics=False
        for record in records:
            writer.write(record)
    reader = colstore.open(tmp_path / "n.cstore")
    assert reader._record_stats is None
    assert reader._try_skip_query_mask(col("x") > 250) is None  # falls back to full read
    np.testing.assert_array_equal(
        reader[col("x") > 250, "x"].array(), np.arange(251, 600, dtype=np.int64)
    )
    reader.close()


def test_single_record_file_does_not_skip(tmp_path):
    # Single-record files have one block; the skip path is multi-record only.
    reader = colstore.store(
        {"x": np.arange(100, dtype=np.int64)}, tmp_path / "one.cstore", statistics=True
    )
    assert reader._try_skip_query_mask(col("x") > 50) is None
    np.testing.assert_array_equal(
        reader[col("x") > 50, "x"].array(), np.arange(51, 100, dtype=np.int64)
    )
    reader.close()
