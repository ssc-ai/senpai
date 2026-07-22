"""Unit tests for per-collect process memory reclamation.

Covers ``senpai.engine.utils.memory.reclaim_process_memory``, which collects Python
reference cycles and then asks glibc to trim its arenas (``malloc_trim``) so a run's
large transient allocations are returned to the OS rather than retained.

The tests use mocks only -- no real large allocations, no astrometry, no catalog.
"""

from __future__ import annotations

import pytest

from senpai.engine.utils import memory


def test_reclaim_process_memory_runs_without_error() -> None:
    """The real reclaim path (glibc/Linux) completes silently."""
    memory.reclaim_process_memory()


def test_reclaim_invokes_gc_collect_and_malloc_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reclaim runs a full gc collection and then calls ``malloc_trim(0)``.

    Args:
        monkeypatch: Pytest fixture used to stub ``gc.collect`` and the libc handle.
    """
    gc_calls: list[int] = []
    trim_calls: list[int] = []

    monkeypatch.setattr(memory.gc, "collect", lambda *a, **k: gc_calls.append(1))

    class _FakeLibc:
        """Minimal stand-in for the C library exposing ``malloc_trim``."""

        def malloc_trim(self, pad: int) -> int:
            """Record the requested trim padding.

            Args:
                pad: The padding argument forwarded to ``malloc_trim``.

            Returns:
                A truthy status code, mirroring glibc's return contract.
            """
            trim_calls.append(pad)
            return 1

    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _name: _FakeLibc())

    memory.reclaim_process_memory()

    assert gc_calls == [1]
    assert trim_calls == [0]  # malloc_trim(0) -- release everything trimmable


def test_reclaim_survives_missing_malloc_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform without a loadable libc (non-glibc) makes the trim a swallowed no-op.

    Args:
        monkeypatch: Pytest fixture used to stub ``gc.collect`` and force a libc load failure.
    """
    monkeypatch.setattr(memory.gc, "collect", lambda *a, **k: None)

    def _no_libc(_name: str) -> None:
        raise OSError("libc not found")

    monkeypatch.setattr(memory.ctypes, "CDLL", _no_libc)

    memory.reclaim_process_memory()  # must not raise
