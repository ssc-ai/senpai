"""Tests for solve_field against a real Astrometry.net install (local + Docker)."""

import os
from pathlib import Path

import pytest

from senpai.astrometry import solve_field
from senpai.core.config import initialize_config
from senpai.core.constants import (
    CI_PIPELINE_CONFIG_PATH,
    LOCAL_APP_CONFIG_OVERRIDE,
    LOCAL_APP_LOCAL_ASTROMETRY_CONFIG_OVERRIDE,
)
from senpai.engine.models.starfield import StarListImage

# These tests need a working Astrometry.net (local or Docker) + index files.
pytestmark = pytest.mark.requires_astrometry


def _get_test_config_path() -> Path:
    """Get the appropriate config path based on environment.

    In CI/pipeline environments, use the pipeline config.
    Otherwise, use the local config.
    """
    # Check for CI environment variable (set by the CI environment)
    if os.getenv("CI") or os.getenv("SENPAI_TEST_CONFIG"):
        return CI_PIPELINE_CONFIG_PATH

    # Default to local config
    return LOCAL_APP_CONFIG_OVERRIDE


def test_astrometric_fit_local(xyls_data: StarListImage) -> None:
    """solve_field returns a result using the local Astrometry.net config."""
    config_path = _get_test_config_path()
    initialize_config(config_path=config_path)
    wcs_field = solve_field(xyls_data)

    assert wcs_field is not None


def test_astrometric_fit_docker(xyls_data: StarListImage) -> None:
    """solve_field returns a result using the Docker Astrometry.net config."""
    initialize_config(config_path=LOCAL_APP_LOCAL_ASTROMETRY_CONFIG_OVERRIDE)
    wcs_field = solve_field(xyls_data)

    assert wcs_field is not None
