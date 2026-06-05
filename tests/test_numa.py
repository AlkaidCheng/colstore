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

import contextlib
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
    convention). After the kernel's ``--maxnode`` this lands on exactly
    ``BITS_PER_LONG``, making the endmask be ``~0UL`` -- every bit of
    the bitmap honored.

    The assertion is in terms of ``_BITS_PER_LONG`` so it stays correct
    on both LP64 (Linux/macOS, 64-bit ulong) and LLP64 (Windows, 32-bit
    ulong). The helper is dead code on platforms without NUMA syscalls,
    but the formula's invariant is platform-independent and worth
    pinning anyway.
    """
    bits = _numa._BITS_PER_LONG
    # An n-word bitmap gets maxnode = n * BITS_PER_LONG + 1.
    assert _numa._maxnode_for_bitmap(1) == bits + 1
    assert _numa._maxnode_for_bitmap(2) == 2 * bits + 1
    # The buggy value for an 8-node host was 8 (== max_id + 1). The
    # correct value for any nonzero-word bitmap is significantly larger,
    # regardless of word width.
    assert _numa._maxnode_for_bitmap(1) > 9


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


# ---- Writer-side integration ------------------------------------------------


def test_interleave_thread_policy_yields_bool_on_every_platform():
    """The context manager returns a boolean indicating actual application.

    On a single-node sandbox the manager yields False; on multi-node
    Linux it yields True after the syscall succeeds. Either way the
    body runs and the context unwinds cleanly.
    """
    with _numa.interleave_thread_policy() as applied:
        assert isinstance(applied, bool)
        # On unavailable hosts we MUST report False, not silently apply.
        if not _numa.is_available():
            assert applied is False


def test_interleave_thread_policy_is_reentrant():
    """Nesting the scope must work even if the outer policy was applied.

    Inner scopes capture the policy that's active on entry (which may
    be MPOL_INTERLEAVE from an outer scope) and restore it on exit.
    Without the capture-and-restore design, nesting would silently
    drop us to MPOL_DEFAULT after the inner exits.
    """
    with _numa.interleave_thread_policy() as outer, _numa.interleave_thread_policy() as inner:
        assert outer == inner  # both no-op or both applied


def test_writer_under_auto_enters_interleave_scope(tmp_path, monkeypatch):
    """`auto` policy wraps writer body in interleave_thread_policy.

    Patches the shared ``_numa.writer_policy_scope`` dispatcher --
    both ``ColStoreWriter.write`` and ``format.write_dataset`` go
    through it, so this catches a regression in either path.
    """
    entered: list[bool] = []

    @contextlib.contextmanager
    def spy():
        entered.append(True)
        yield True

    # Patch BOTH the canonical location (used by ColStoreWriter via
    # `_numa.writer_policy_scope`) and the re-export through `format`
    # (used by `colstore.store` -> `format.write_dataset`). The two
    # callers reach the dispatcher through different binding paths.
    monkeypatch.setattr(_numa, "writer_policy_scope", spy)
    from colstore import format as fmt_mod

    monkeypatch.setattr(fmt_mod._numa, "writer_policy_scope", spy)

    previous = config.get_numa_policy()
    try:
        config.set_numa_policy("auto")
        store_path = tmp_path / "writer_auto.cstore"
        colstore.store(
            {"x": np.arange(16, dtype=np.float64)}, store_path, show_progress=False
        ).close()
    finally:
        config.set_numa_policy(previous)

    assert entered, "auto policy must enter the writer-side interleave scope"


def test_writer_under_local_skips_interleave_call(tmp_path, monkeypatch):
    """`local` policy must NOT enter the interleave scope on the writer.

    The dispatcher returns a ``nullcontext`` for "local" or non-
    applicable hosts; ``interleave_thread_policy`` itself must not be
    invoked. Patches ``interleave_thread_policy`` to detect any call.
    """
    invoked: list[bool] = []

    @contextlib.contextmanager
    def spy():
        invoked.append(True)
        yield True

    monkeypatch.setattr(_numa, "interleave_thread_policy", spy)

    previous = config.get_numa_policy()
    try:
        config.set_numa_policy("local")
        store_path = tmp_path / "writer_local.cstore"
        colstore.store(
            {"x": np.arange(16, dtype=np.float64)}, store_path, show_progress=False
        ).close()
    finally:
        config.set_numa_policy(previous)

    assert invoked == [], "local policy must not invoke interleave_thread_policy"


def test_writer_writes_correct_data_under_each_policy(tmp_path):
    """The actual file contents are independent of NUMA policy.

    The optimization changes WHERE pages live, not WHAT they contain.
    Pin that the round-trip is byte-equivalent under every policy.
    """
    expected_x = np.arange(1024, dtype=np.float64)
    expected_y = np.arange(1024, dtype=np.int32)

    previous = config.get_numa_policy()
    try:
        for policy in ("auto", "interleave", "local"):
            config.set_numa_policy(policy)
            store_path = tmp_path / f"under_{policy}.cstore"
            colstore.store(
                {"x": expected_x, "y": expected_y}, store_path, show_progress=False
            ).close()
            store = colstore.open(store_path)
            try:
                np.testing.assert_array_equal(store["x"].array(), expected_x)
                np.testing.assert_array_equal(store["y"].array(), expected_y)
            finally:
                store.close()
    finally:
        config.set_numa_policy(previous)
