"""Tests for senpai.core.config — YAML loading, the process-wide singleton,
and validation/defaults/frozen-ness of the AppConfig pydantic models.

The config singleton is process-wide and other test modules initialize it. These
tests never permanently clear it: where uninitialized state is required, the
``_config_instance`` module global is patched via monkeypatch (auto-restored).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from senpai.core import config as cfg_mod
from senpai.core.config import (
    AppConfig,
    DetectionConfig,
    PhotometryConfig,
    RuntimeConfig,
    StarCatalogConfig,
    ValidationConfig,
    get_config,
    get_or_initialize_config,
    initialize_config,
    load_yaml,
)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "resources" / "config"

# AppConfig's astrometry/plotting sub-configs have required fields (no usable
# default_factory), so a bare AppConfig(version=...) fails. This is the minimal
# valid payload for constructing an AppConfig directly in tests.
_MIN_ASTROMETRY = {
    "indices_series": "5200_LITE",
    "indices_path": "/nonexistent/idx",
    "max_sources": 500,
    "min_sources_for_attempt": 4,
    "min_width_degrees": 0.1,
    "max_width_degrees": 10.0,
    "cpulimit_seconds": 30,
    "docker_image": None,
}
_MIN_PLOTTING = {"debug": False, "review": False}


def _min_app(**overrides) -> dict:
    data = {
        "version": "1.0.0",
        "astrometry": dict(_MIN_ASTROMETRY),
        "plotting": dict(_MIN_PLOTTING),
        "star_catalog": {"type": "gaia"},
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# load_yaml
# --------------------------------------------------------------------------- #
def test_load_yaml_valid_returns_app_section(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": {"version": "9.9.9", "debug": True}}))
    data = load_yaml(p)
    assert data == {"version": "9.9.9", "debug": True}


def test_load_yaml_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_yaml(tmp_path / "does_not_exist.yaml") == {}


def test_load_yaml_no_app_key_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"not_app": {"version": "1"}}))
    assert load_yaml(p) == {}


def test_load_yaml_malformed_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("app: [unterminated\n  : :")
    assert load_yaml(p) == {}


# --------------------------------------------------------------------------- #
# Singleton behaviour (get_config / initialize_config / get_or_initialize_config)
# --------------------------------------------------------------------------- #
def test_get_config_raises_when_uninitialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        get_config()


def test_initialize_config_sets_singleton(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": _min_app(version="1.2.3")}))
    returned = initialize_config(p)
    assert returned.version == "1.2.3"
    # get_config now returns the same instance
    assert get_config() is returned


def test_get_or_initialize_uses_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = AppConfig(**_min_app(version="sentinel"))
    monkeypatch.setattr(cfg_mod, "_config_instance", sentinel)
    assert get_or_initialize_config() is sentinel


def test_get_or_initialize_loads_path_when_uninitialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": _min_app(version="from-path")}))
    loaded = get_or_initialize_config(p)
    assert loaded.version == "from-path"


# --------------------------------------------------------------------------- #
# StarCatalogConfig validator
# --------------------------------------------------------------------------- #
def test_sstrc7_requires_path() -> None:
    with pytest.raises(ValidationError, match="path is required"):
        StarCatalogConfig(type="sstrc7")


def test_sstrc7_with_path_ok() -> None:
    c = StarCatalogConfig(type="sstrc7", path="/some/path")
    assert c.path == "/some/path"


@pytest.mark.parametrize("online", ["sdss", "gaia"])
def test_online_catalogs_no_path_required(online: str) -> None:
    c = StarCatalogConfig(type=online)
    assert c.path is None
    # default faint limit applied
    assert c.faint_limit == 18.0


def test_star_catalog_faint_limit_override() -> None:
    c = StarCatalogConfig(type="gaia", faint_limit=None)
    assert c.faint_limit is None


# --------------------------------------------------------------------------- #
# Sub-config defaults
# --------------------------------------------------------------------------- #
def test_photometry_defaults() -> None:
    p = PhotometryConfig()
    assert p.aperture_radius_factor == 2.0
    assert p.bg_inner_factor == 3.0
    assert p.bg_outer_factor == 5.0
    assert p.zp_min_snr == 20.0
    assert p.color_index_bands == ("Gaia_BP", "Gaia_RP")
    assert "Johnson_V" in p.preferred_filters


def test_validation_defaults() -> None:
    v = ValidationConfig()
    assert v.box_size == 11
    assert v.n_random_trials == 8
    assert v.min_correlation_ratio == 0.98
    assert v.max_validation_stars == 50


def test_detection_defaults() -> None:
    d = DetectionConfig()
    assert d.detect is False
    assert d.detect_streaks is True
    assert d.snr_threshold == 3.0
    assert d.streak_angle_tolerance_deg == 15.0


# --------------------------------------------------------------------------- #
# Frozen-ness / mutability
# --------------------------------------------------------------------------- #
def test_appconfig_is_frozen() -> None:
    c = AppConfig(**_min_app())
    with pytest.raises(ValidationError):
        c.version = "2.0.0"


def test_runtime_config_is_mutable() -> None:
    r = RuntimeConfig()
    r.run_id = "changed"
    r.output_dir = "/nonexistent/out"
    assert r.run_id == "changed"
    assert r.output_dir == "/nonexistent/out"


def test_appconfig_defaults_populate_subconfigs() -> None:
    c = AppConfig(**_min_app())
    assert isinstance(c.photometry, PhotometryConfig)
    assert isinstance(c.detection, DetectionConfig)
    assert isinstance(c.runtime, RuntimeConfig)
    assert c.debug is False


# --------------------------------------------------------------------------- #
# Every shipped YAML loads into a valid AppConfig
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "yaml_path",
    sorted(CONFIG_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_all_shipped_configs_load(yaml_path: Path) -> None:
    data = load_yaml(yaml_path)
    config = AppConfig(**data)
    assert config.version  # version is required and present in every shipped config
