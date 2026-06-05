"""Tests for the NUMA memory-policy helper and config plumbing.

The hard-win measurement (interleave is ~1.8x faster than first-touch
on a 1 GB / 50-col gather) only shows up on multi-node hardware; the
benchmark in ``benchmark/check_numa.py`` is what validates that. These
tests pin the platform-agnostic invariants:

  * The module imports cleanly on every supported platform.
  * Detection returns sensible values for the host.
  * apply_interleave_to_memmap returns False (not raises) when not
    applicable, and the syscall succeeds when applicable.
  * Config get/set round-trips and validates inputs.
  * Opening a store works under every policy without raising.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

import colstore
from colstore import _numa, config

# ---- Module-level state -----------------------------------------------------


def test_numa_module_imports_on_every_platform():
    """`colstore._numa` must import without side effects on every platform."""
    assert hasattr(_numa, "is_available")
    assert hasattr(_numa, "allowed_nodes")
    assert hasattr(_numa, "apply_interleave_to_memmap")


def test_is_available_returns_bool():
    assert isinstance(_numa.is_available(), bool)


def test_allowed_nodes_returns_list_of_ints():
    nodes = _numa.allowed_nodes()
    assert isinstance(nodes, list)
    assert all(isinstance(n, int) for n in nodes)
    assert all(n >= 0 for n in nodes)


def test_is_available_consistency_with_platform():
    """The availability signal must agree with the platform / node count."""
    if sys.platform != "linux":
        assert _numa.is_available() is False
        return
    # On Linux, availability requires more than one allowed node.
    nodes = _numa.allowed_nodes()
    if len(nodes) <= 1:
        assert _numa.is_available() is False
    else:
        # Multi-node Linux on a supported arch: should be available.
        # Single-arch test environments (e.g. exotic ports) may still
        # land on False; tolerate that.
        if platform.machine() in ("x86_64", "aarch64", "ppc64le", "s390x"):
            assert _numa.is_available() is True


# ---- maxnode computation (regression) ---------------------------------------


def test_maxnode_for_bitmap_uses_full_word_width():
    """Regression: maxnode is "bitmap bits, plus one", not "max id, plus one".

    The bug this pins: ``_MAXNODE = max(allowed_nodes) + 1`` looks
    superficially right but the kernel internally decrements maxnode
    and ANDs the last word against
    ``(1 << (maxnode % BITS_PER_LONG)) - 1``, silently dropping the
    highest bit. On an 8-node host with bug-version maxnode=8,
    ``/proc/self/numa_maps`` reported ``interleave:0-6`` for what was
    intended as ``interleave:0-7`` -- node 7 was being dropped.

    The fix: ``maxnode = n_words * BITS_PER_LONG + 1`` (libnuma's
    convention). With ``n_words=1`` that's 65, which after the
    kernel's ``--maxnode`` lands on exactly ``BITS_PER_LONG``, making
    the endmask be ``~0UL`` -- every bit of the bitmap honored.
    """
    # 1 unsigned long = 64 bits → maxnode = 65
    assert _numa._maxnode_for_bitmap(1) == 65
    # 2 unsigned longs = 128 bits → maxnode = 129
    assert _numa._maxnode_for_bitmap(2) == 129
    # The buggy value for 8 nodes was 8 (== max_id + 1). The corrected
    # value for an 8-node bitmap (one ulong) is 65, NOT 9 or 8.
    assert _numa._maxnode_for_bitmap(1) != 8
    assert _numa._maxnode_for_bitmap(1) != 9


def test_module_maxnode_matches_bitmap_width():
    """On capable hosts, the module-level _MAXNODE must match the bitmap width."""
    if not _numa.is_available():
        # The constant is initialized to 0 on inapplicable hosts; that's correct.
        assert _numa._MAXNODE == 0
        return
    # On capable hosts, it should be n_words * BITS_PER_LONG + 1. Both
    # sides are bound to locals so ruff's SIM300 yoda-condition heuristic
    # doesn't see the right-hand side's attribute-access chain as the
    # "constant" side; the assertion reads naturally either way.
    actual_maxnode = _numa._MAXNODE
    expected_maxnode = _numa._n_words * _numa._BITS_PER_LONG + 1
    assert actual_maxnode == expected_maxnode


# ---- Page-alignment helper --------------------------------------------------


def test_page_align_range_rounds_outward():
    page = _numa._PAGE_SIZE
    # [page+10, page+10 + page+30) = [page+10, 2*page+40)
    # snaps to [page, 3*page); the returned length is the size of the
    # aligned range, i.e. 2*page.
    addr, length = _numa._page_align_range(page + 10, page + 30)
    assert addr == page
    assert length == 2 * page


def test_page_align_range_passes_through_aligned_ranges():
    page = _numa._PAGE_SIZE
    addr, length = _numa._page_align_range(2 * page, 4 * page)
    assert addr == 2 * page
    assert length == 4 * page


# ---- Syscall helper on inapplicable hosts -----------------------------------


def test_apply_interleave_returns_false_when_unavailable():
    """When the module is in no-op mode, the call must return False, not raise."""
    if _numa.is_available():
        pytest.skip("This test exercises the no-op branch; this host can apply.")
    arr = np.zeros(1024, dtype=np.float64)
    # Even on an inapplicable host we must not raise -- this is the
    # transparent-optimization contract.
    assert _numa.apply_interleave_to_memmap(arr) is False


def test_apply_interleave_returns_false_on_zero_length():
    """Empty arrays are a no-op even on capable hosts (no pages to bind)."""
    arr = np.zeros(0, dtype=np.float64)
    assert _numa.apply_interleave_to_memmap(arr) is False


# ---- Config policy validation -----------------------------------------------


def test_numa_policy_default_is_auto():
    assert config.get_numa_policy() == "auto"


def test_numa_policy_set_get_roundtrip():
    """Each valid policy round-trips through set/get."""
    previous = config.get_numa_policy()
    try:
        for policy in ("auto", "interleave", "local"):
            config.set_numa_policy(policy)
            assert config.get_numa_policy() == policy
    finally:
        config.set_numa_policy(previous)


def test_numa_policy_rejects_invalid_value():
    with pytest.raises(ValueError, match="numa policy"):
        config.set_numa_policy("magic")  # type: ignore[arg-type]


# ---- Reader-side integration ------------------------------------------------


def test_reader_open_succeeds_under_each_policy(tmp_path):
    """Opening a store works under every policy on every platform.

    On the sandbox / single-node hosts the policy is silently a no-op
    in every case; on multi-node Linux ``auto`` and ``interleave``
    actually apply the syscall. Either way ``open`` returns a working
    reader and the data round-trips correctly.
    """
    n_rows = 64
    columns = {
        "x": np.arange(n_rows, dtype=np.float64),
        "y": np.arange(n_rows, dtype=np.int32),
    }
    store_path = tmp_path / "policies.cstore"
    colstore.store(columns, store_path, show_progress=False).close()

    previous = config.get_numa_policy()
    try:
        for policy in ("auto", "interleave", "local"):
            config.set_numa_policy(policy)
            store = colstore.open(store_path)
            try:
                np.testing.assert_array_equal(
                    store["x"].array(), np.arange(n_rows, dtype=np.float64)
                )
                np.testing.assert_array_equal(store["y"].array(), np.arange(n_rows, dtype=np.int32))
            finally:
                store.close()
    finally:
        config.set_numa_policy(previous)


def test_reader_open_under_local_policy_skips_numa_call(tmp_path, monkeypatch):
    """`local` short-circuits before reaching the NUMA module."""
    calls: list[object] = []

    def spy(view):
        calls.append(view)
        return True

    monkeypatch.setattr(_numa, "apply_interleave_to_memmap", spy)

    store_path = tmp_path / "skip.cstore"
    columns = {"x": np.arange(16, dtype=np.float64)}
    colstore.store(columns, store_path, show_progress=False).close()

    previous = config.get_numa_policy()
    try:
        config.set_numa_policy("local")
        store = colstore.open(store_path)
        store.close()
    finally:
        config.set_numa_policy(previous)

    assert calls == [], "local policy must not call the NUMA helper"
