"""Guards on where shipped code is allowed to look for files.

A wheel installs `senpai/` and nothing else -- no `resources/`, no `tests/`.
Anything shipped code reads at import or startup must therefore live inside the
package, and anything it writes must live outside the install tree. Issue #6:
`create_app()` raised on every non-editable install because the OpenAPI example
loaded a repo test fixture and the default config resolved into site-packages.

The end-to-end version of this check (build a wheel, install it, call
`create_app()`) runs in CI's `build` job; these tests catch the same class of
regression without a build.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from senpai.core import constants


def _is_relative_to(path: Path, other: Path) -> bool:
    return other in path.resolve().parents or path.resolve() == other.resolve()


PACKAGE_DIR = constants.PACKAGE_DIR


def test_package_dir_is_the_senpai_package() -> None:
    assert PACKAGE_DIR.name == "senpai"
    assert (PACKAGE_DIR / "core" / "constants.py").is_file()


@pytest.mark.parametrize(
    "name",
    ["RESOURCES_DIR", "ASSETS_DIR", "DATA_DIR", "CONFIG_DIR", "APP_DIR", "APP_CONFIG_PATH"],
)
def test_shipped_paths_stay_inside_the_package(name) -> None:
    # These resolve into site-packages/ when anchored at the repo root, where
    # nothing exists. Anchor them at the package or they break when installed.
    assert _is_relative_to(getattr(constants, name), PACKAGE_DIR), f"{name} escapes the installed package"


@pytest.mark.parametrize(
    "name",
    [
        "LOCAL_APP_CONFIG_OVERRIDE",
        "LOCAL_APP_LOCAL_ASTROMETRY_CONFIG_OVERRIDE",
    ],
)
def test_default_configs_are_shipped(name) -> None:
    # The API and CLI fall back to these with no --config; they must be in the
    # wheel, not merely in a checkout.
    assert getattr(constants, name).is_file()


@pytest.mark.parametrize("name", ["CACHE_DIR", "LOG_DIR", "LOG_PATH"])
def test_writable_paths_stay_outside_the_package(name) -> None:
    # An installed package may sit on a read-only filesystem, and writing into
    # site-packages is wrong even where it is permitted.
    assert not _is_relative_to(getattr(constants, name), PACKAGE_DIR), f"{name} writes into the installed package"


def test_cache_dir_honors_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SENPAI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SENPAI_LOG_DIR", str(tmp_path / "logs"))
    expected_cache = tmp_path / "cache"
    expected_log = tmp_path / "logs" / "app.log"
    reloaded = importlib.reload(constants)
    try:
        assert expected_cache == reloaded.CACHE_DIR
        assert expected_log == reloaded.LOG_PATH
    finally:
        monkeypatch.undo()
        importlib.reload(constants)


def test_openapi_example_needs_no_files_on_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    # The /solve/sources example is built at import time by FastAPI's
    # ``Body(examples=...)``. Reading anything from disk there is what broke
    # installs; a literal cannot.
    def explode(*args, **kwargs):
        raise AssertionError("OpenAPI example payload must not read from disk")

    import numpy as np

    monkeypatch.setattr(np, "loadtxt", explode)
    monkeypatch.setattr(Path, "open", explode)

    examples = importlib.reload(importlib.import_module("senpai.api.models.examples"))
    example = examples.StarListImageExample().value

    assert len(example.detections) > 0
    assert example.image_metadata.width == 1024
