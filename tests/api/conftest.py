"""Fixtures for the SENPAI API test suite.

The project does not depend on httpx, so Starlette's TestClient is unavailable.
Instead we test the app at two levels, both fully hermetic and offline:

* ``create_app`` construction and route registration, with every
  host-touching startup hook (Astrometry.net validation, the
  ProcessPoolExecutor) monkeypatched away.
* The route coroutines invoked directly with a lightweight fake ``Request``
  and pydantic-parsed bodies, with the heavy processing layer mocked.

The config singleton is process-wide (get_config() raises if uninitialized)
and shared with other test modules, so we initialize it once per session.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest

from senpai.core.config import AppConfig, get_config, initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE


@pytest.fixture(scope="session", autouse=True)
def _init_config() -> AppConfig:
    """Initialize the process-wide config singleton from local.yaml.

    Returns:
        The initialized application config singleton.
    """
    initialize_config(LOCAL_APP_CONFIG_OVERRIDE)
    return get_config()


class _InlineExecutor:
    """A drop-in ProcessPoolExecutor stand-in that never spawns processes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept and ignore any ProcessPoolExecutor constructor arguments.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).
        """

    def submit(self, fn: Callable[..., object], *args: object, **kwargs: object) -> concurrent.futures.Future:
        """Run ``fn`` inline and wrap its outcome in a completed Future.

        Args:
            fn: The callable to execute synchronously.
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            A Future already resolved with ``fn``'s result or exception.
        """
        fut: concurrent.futures.Future = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        return fut

    def shutdown(self, *args: object, **kwargs: object) -> None:
        """Accept and ignore any executor shutdown arguments.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).
        """


@pytest.fixture
def patched_app_env(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Neutralize all host-touching startup work in senpai.api.main.

    Args:
        monkeypatch: The pytest monkeypatch fixture used to swap out the
            host-touching hooks.

    Returns:
        The patched ``senpai.api.main`` module.
    """
    import senpai.api.main as main_mod

    monkeypatch.setattr(main_mod, "test_astrometry_install", lambda *a, **k: True)
    monkeypatch.setattr(main_mod, "examine_indices", lambda *a, **k: True)
    monkeypatch.setattr(
        main_mod.concurrent.futures,
        "ProcessPoolExecutor",
        _InlineExecutor,
    )
    return main_mod


def make_request(path: str = "/senpai/", base_url: str = "http://testserver/") -> SimpleNamespace:
    """Build a minimal stand-in for a Starlette Request.

    The SENPAI route handlers only read ``request.base_url`` and
    ``request.url.path``, so a SimpleNamespace suffices.

    Args:
        path: The URL path exposed as ``request.url.path``.
        base_url: The base URL exposed as ``request.base_url``.

    Returns:
        A SimpleNamespace mimicking the attributes the handlers read.
    """
    return SimpleNamespace(base_url=base_url, url=SimpleNamespace(path=path))
