"""Tests for package-wide configuration knobs."""

from __future__ import annotations

import pytest

from colstore import (
    config,
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


def test_convert_auto_workers_roundtrips_and_validates():
    from colstore import get_convert_auto_workers, set_convert_auto_workers

    original = get_convert_auto_workers()
    assert isinstance(original, int) and original >= 1
    try:
        set_convert_auto_workers(16)
        assert get_convert_auto_workers() == 16
        assert config.get_convert_auto_workers() == 16
        with pytest.raises(ValueError):
            set_convert_auto_workers(0)
    finally:
        set_convert_auto_workers(original)


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
    assert get_default_backend() in ("cpp", "numpy")


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


def test_write_method_starts_as_auto():
    assert config.get_write_method() == "auto"


def test_set_write_method_roundtrips():
    original = config.get_write_method()
    try:
        config.set_write_method("mmap")
        assert config.get_write_method() == "mmap"
        config.set_write_method("pwrite")
        assert config.get_write_method() == "pwrite"
    finally:
        config.set_write_method(original)


def test_set_write_method_rejects_unknown():
    with pytest.raises(ValueError):
        config.set_write_method("bogus")
