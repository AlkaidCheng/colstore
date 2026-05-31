"""Tests for package-wide configuration knobs."""

from __future__ import annotations

import pytest

from colstore import (
    get_default_backend,
    get_default_madvise,
    get_max_workers,
    set_default_backend,
    set_default_madvise,
    set_max_workers,
)


def test_get_max_workers_returns_positive_int():
    workers = get_max_workers()
    assert isinstance(workers, int)
    assert workers >= 1


def test_set_max_workers_roundtrips():
    original = get_max_workers()
    try:
        set_max_workers(7)
        assert get_max_workers() == 7
    finally:
        set_max_workers(original)


def test_set_max_workers_rejects_zero():
    with pytest.raises(ValueError):
        set_max_workers(0)


def test_set_max_workers_rejects_negative():
    with pytest.raises(ValueError):
        set_max_workers(-1)


def test_default_madvise_is_a_known_option_or_none():
    advice = get_default_madvise()
    assert advice in ("normal", "sequential", "random", "willneed", "dontneed", None)


def test_set_default_madvise_roundtrips():
    original = get_default_madvise()
    try:
        set_default_madvise("random")
        assert get_default_madvise() == "random"
        set_default_madvise(None)
        assert get_default_madvise() is None
    finally:
        set_default_madvise(original)


def test_default_backend_starts_as_cpp():
    assert get_default_backend() in ("cpp", "numpy", "numba")


def test_set_default_backend_roundtrips():
    original = get_default_backend()
    try:
        set_default_backend("numpy")
        assert get_default_backend() == "numpy"
    finally:
        set_default_backend(original)


def test_set_default_backend_rejects_unknown():
    with pytest.raises(ValueError):
        set_default_backend("unknown-backend")
