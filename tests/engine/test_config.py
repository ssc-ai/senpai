"""Tests for senpai.core.config.

YAML loading, the process-wide singleton, and validation/defaults/frozen-ness of
the AppConfig pydantic models.

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

CONFIG_DIR = Path(__file__).resolve().parents[2] / "resources" / "config"

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


def _min_app(**overrides: object) -> dict:
    """Return the minimal valid ``app`` payload for constructing an AppConfig.

    Args:
        **overrides: Top-level keys to merge over the minimal payload.

    Returns:
        A dict suitable for ``AppConfig(**...)`` or nesting under an ``app`` key.
    """
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
    """load_yaml returns the ``app`` section of a valid YAML file."""
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": {"version": "9.9.9", "debug": True}}))
    data = load_yaml(p)
    assert data == {"version": "9.9.9", "debug": True}


def test_load_yaml_missing_file_returns_empty(tmp_path: Path) -> None:
    """load_yaml returns an empty dict when the file does not exist."""
    assert load_yaml(tmp_path / "does_not_exist.yaml") == {}


def test_load_yaml_no_app_key_returns_empty(tmp_path: Path) -> None:
    """load_yaml returns an empty dict when there is no ``app`` key."""
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"not_app": {"version": "1"}}))
    assert load_yaml(p) == {}


def test_load_yaml_malformed_returns_empty(tmp_path: Path) -> None:
    """load_yaml returns an empty dict on malformed YAML rather than raising."""
    p = tmp_path / "bad.yaml"
    p.write_text("app: [unterminated\n  : :")
    assert load_yaml(p) == {}


# --------------------------------------------------------------------------- #
# Singleton behaviour (get_config / initialize_config / get_or_initialize_config)
# --------------------------------------------------------------------------- #
def test_get_config_raises_when_uninitialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_config raises RuntimeError before the singleton is initialized."""
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        get_config()


def test_initialize_config_sets_singleton(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """initialize_config loads the file and installs it as the singleton."""
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": _min_app(version="1.2.3")}))
    returned = initialize_config(p)
    assert returned.version == "1.2.3"
    # get_config now returns the same instance
    assert get_config() is returned


def test_get_or_initialize_uses_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_or_initialize_config returns the existing singleton unchanged."""
    sentinel = AppConfig(**_min_app(version="sentinel"))
    monkeypatch.setattr(cfg_mod, "_config_instance", sentinel)
    assert get_or_initialize_config() is sentinel


def test_get_or_initialize_loads_path_when_uninitialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_or_initialize_config loads the given path when uninitialized."""
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": _min_app(version="from-path")}))
    loaded = get_or_initialize_config(p)
    assert loaded.version == "from-path"


# --------------------------------------------------------------------------- #
# StarCatalogConfig validator
# --------------------------------------------------------------------------- #
def test_sstrc7_requires_path() -> None:
    """The sstrc7 catalog type requires a path and rejects its absence."""
    with pytest.raises(ValidationError, match="path is required"):
        StarCatalogConfig(type="sstrc7")


def test_sstrc7_with_path_ok() -> None:
    """The sstrc7 catalog type validates when a path is supplied."""
    c = StarCatalogConfig(type="sstrc7", path="/some/path")
    assert c.path == "/some/path"


@pytest.mark.parametrize("online", ["sdss", "gaia"])
def test_online_catalogs_no_path_required(online: str) -> None:
    """Online catalog types need no path and get the default faint limit."""
    c = StarCatalogConfig(type=online)
    assert c.path is None
    # default faint limit applied
    assert c.faint_limit == 18.0


def test_star_catalog_faint_limit_override() -> None:
    """An explicit ``faint_limit=None`` overrides the default."""
    c = StarCatalogConfig(type="gaia", faint_limit=None)
    assert c.faint_limit is None


# --------------------------------------------------------------------------- #
# Sub-config defaults
# --------------------------------------------------------------------------- #
def test_photometry_defaults() -> None:
    """PhotometryConfig exposes the expected default factors and bands."""
    p = PhotometryConfig()
    assert p.aperture_radius_factor == 2.0
    assert p.bg_inner_factor == 3.0
    assert p.bg_outer_factor == 5.0
    assert p.zp_min_snr == 20.0
    assert p.color_index_bands == ("Gaia_BP", "Gaia_RP")
    assert "Johnson_V" in p.preferred_filters


def test_validation_defaults() -> None:
    """ValidationConfig exposes the expected default thresholds."""
    v = ValidationConfig()
    assert v.box_size == 11
    assert v.n_random_trials == 8
    assert v.min_correlation_ratio == 0.98
    assert v.max_validation_stars == 50


def test_detection_defaults() -> None:
    """DetectionConfig exposes the expected detection defaults."""
    d = DetectionConfig()
    assert d.detect is False
    assert d.detect_streaks is True
    assert d.snr_threshold == 3.0
    assert d.streak_angle_tolerance_deg == 15.0


# --------------------------------------------------------------------------- #
# Frozen-ness / mutability
# --------------------------------------------------------------------------- #
def test_appconfig_is_frozen() -> None:
    """AppConfig is frozen: assigning to a field raises ValidationError."""
    c = AppConfig(**_min_app())
    with pytest.raises(ValidationError):
        c.version = "2.0.0"


def test_runtime_config_is_mutable() -> None:
    """RuntimeConfig fields can be reassigned at runtime."""
    r = RuntimeConfig()
    r.run_id = "changed"
    r.output_dir = "/nonexistent/out"
    assert r.run_id == "changed"
    assert r.output_dir == "/nonexistent/out"


def test_appconfig_defaults_populate_subconfigs() -> None:
    """AppConfig auto-populates its sub-configs from defaults."""
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
    """Every shipped YAML config loads into a valid AppConfig."""
    data = load_yaml(yaml_path)
    config = AppConfig(**data)
    assert config.version  # version is required and present in every shipped config


# --------------------------------------------------------------------------- #
# Merged settings framework (pydantic-settings): env overrides, flat YAML,
# the senpai.settings facade, and the fields added by the engine port.
# --------------------------------------------------------------------------- #
def test_env_vars_override_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested env vars override YAML values while other YAML values persist."""
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    monkeypatch.setenv("LOGGING__LEVEL", "CRITICAL")
    monkeypatch.setenv("ASTROMETRY__CPULIMIT_SECONDS", "123")
    monkeypatch.setenv("DETECTION__SNR_THRESHOLD", "7.5")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": _min_app(version="env-test")}))
    config = initialize_config(p)
    assert config.logging.level == "CRITICAL"
    assert config.astrometry.cpulimit_seconds == 123
    assert config.detection.snr_threshold == 7.5
    # non-overridden YAML values still land
    assert config.version == "env-test"


def test_flat_yaml_loads_without_app_nesting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat YAML file (no ``app`` nesting) still loads into an AppConfig."""
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    p = tmp_path / "flat.yaml"
    p.write_text(yaml.safe_dump(_min_app(version="flat-style")))
    config = initialize_config(p)
    assert config.version == "flat-style"
    assert config.astrometry.max_sources == 500


def test_direct_construction_ignores_stale_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct AppConfig construction does not re-read a previously loaded YAML."""
    # After initialize_config(p), constructing AppConfig(**kwargs) directly must
    # not silently re-read p (stale-yaml leakage into unit-test constructions).
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": _min_app(debug=True)}))
    assert initialize_config(p).debug is True
    assert AppConfig(**_min_app()).debug is False


def test_get_or_initialize_respects_senpai_config_path_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_or_initialize_config honours the SENPAI_CONFIG_PATH env var."""
    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"app": _min_app(version="from-env-path")}))
    monkeypatch.setenv("SENPAI_CONFIG_PATH", str(p))
    assert get_or_initialize_config().version == "from-env-path"


def test_new_astrometry_fields_default() -> None:
    """The ported astrometry fields carry their expected defaults."""
    c = AppConfig(**_min_app())
    assert c.astrometry.search_radius_degrees == 5.0
    assert c.astrometry.source_extractor == "sextractor"
    assert c.astrometry.sip_order == 3
    assert c.astrometry.min_logodds_threshold == 21.0
    assert c.astrometry.error_on_plate_solve_failure is False


def test_new_detection_fields_default() -> None:
    """The ported detection fields carry their expected defaults."""
    d = DetectionConfig()
    assert d.require_wcs_refinement is True
    assert d.centroid_guard_mode == "fwhm"
    assert d.centroid_guard_value == 0.4


def test_observations_section_defaults() -> None:
    """The observations sub-config carries its expected defaults."""
    c = AppConfig(**_min_app())
    assert c.observations.centroid_localization_std_pix is None
    assert c.observations.uncertainty_warn_threshold_deg == 1.0
    assert c.observations.time_offsets_s == {}


def test_streak_max_fwhm_default() -> None:
    """The streak sub-config carries its default max-FWHM extraction limit."""
    c = AppConfig(**_min_app())
    assert c.streak.max_fwhm_for_streak_extraction == 10.0


def test_plotting_output_dir_default() -> None:
    """The plotting sub-config defaults its output directory to the cwd."""
    c = AppConfig(**_min_app())
    assert c.plotting.output_dir == "."


def test_settings_facade_proxy_uninitialized_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings facade raises ConfigError when the config is uninitialized."""
    from senpai.exceptions import ConfigError
    from senpai.settings import settings

    monkeypatch.setattr(cfg_mod, "_config_instance", None)
    with pytest.raises(ConfigError, match="not initialized"):
        _ = settings.version


def test_settings_facade_proxy_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings facade forwards attribute access to the live config."""
    from senpai.settings import settings

    sentinel = AppConfig(**_min_app(version="proxy-sentinel"))
    monkeypatch.setattr(cfg_mod, "_config_instance", sentinel)
    assert settings.version == "proxy-sentinel"
    assert settings.detection.detect is False


def test_settings_facade_shares_initialize(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings facade re-exports the same init/get functions."""
    import senpai.settings as settings_mod

    assert settings_mod.initialize_config is initialize_config
    assert settings_mod.get_config is get_config
