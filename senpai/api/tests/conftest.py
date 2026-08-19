"""Fixtures for the SENPAI API test suite.

The project does not depend on httpx, so Starlette's TestClient is unavailable.
Instead we test the app at two levels, both fully hermetic and offline:

* ``create_app`` construction and route registration, with every
  host-touching startup hook (Astrometry.net validation, the
  ProcessPoolExecutor) monkeypatched away.
* The route coroutines invoked directly with a lightweight fake ``Request``
  and pydantic-parsed bodies, with the heavy processing layer mocked.

The config singleton is process-wide (reading ``settings`` raises if uninitialized)
and shared with other test modules, so we initialize it once per session.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace

import pytest

import senpai.api.main as main_mod
from senpai.core.config import AppConfig, initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE


@pytest.fixture(scope="session", autouse=True)
def _init_config() -> AppConfig:
    """Initialize the process-wide config singleton from local.yaml."""
    return initialize_config(LOCAL_APP_CONFIG_OVERRIDE)


class _InlineExecutor:
    """A drop-in ProcessPoolExecutor stand-in that never spawns processes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept and ignore the pool arguments, since nothing is spawned."""
        pass

    def submit(self, fn: object, *args: object, **kwargs: object) -> Future:
        """Run the callable immediately and return an already-completed future."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        return fut

    def shutdown(self, *args: object, **kwargs: object) -> None:
        """Nothing to shut down."""
        pass


@pytest.fixture
def patched_app_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Neutralize all host-touching startup work in senpai.api.main."""
    monkeypatch.setattr(main_mod, "test_astrometry_install", lambda *a, **k: True)
    monkeypatch.setattr(main_mod, "examine_indices", lambda *a, **k: True)
    monkeypatch.setattr(
        main_mod.concurrent.futures,
        "ProcessPoolExecutor",
        _InlineExecutor,
    )
    return main_mod


def make_request(path: str = "/senpai/", base_url: str = "http://testserver/") -> Callable[..., object]:
    """Build a minimal stand-in for a Starlette Request.

    The SENPAI route handlers only read ``request.base_url`` and
    ``request.url.path``, so a SimpleNamespace suffices.
    """
    return SimpleNamespace(base_url=base_url, url=SimpleNamespace(path=path))
