"""`astrometry.verify_solution` reaches astroeasy, so a deployment can turn the solver's verify/re-tweak
step on or off from its config alone.

astroeasy 1.3 runs the verify step on every host (before, it ran only where the host's default
astrometry.net config happened to list the index files). The step re-fits the solution; a deployment
that wants the solve's own, unverified solution sets `verify_solution: false`.
"""

from __future__ import annotations

from senpai.core.config import AstrometryConfig

BLOCK = {
    # A deployed config's required fields, with no verify_solution key.
    "indices_series": "5200_LITE",
    "indices_path": "/data/indices",
    "max_sources": 100,
    "min_sources_for_attempt": 10,
    "min_width_degrees": 1.0,
    "max_width_degrees": 3.0,
    "cpulimit_seconds": 60,
    "docker_image": None,
}


def _astroeasy_config_for(monkeypatch, astrometry: AstrometryConfig):
    import senpai.astrometry as adapter

    class _Stub:
        pass

    _Stub.astrometry = astrometry
    monkeypatch.setattr(adapter, "get_or_initialize_config", lambda: _Stub)
    return adapter._build_astroeasy_config()


def test_existing_configs_keep_the_default():
    """A config written before the key existed loads, and verifies as astroeasy does by default."""
    assert AstrometryConfig(**BLOCK).verify_solution is True


def test_off_reaches_astroeasy(monkeypatch):
    ae = _astroeasy_config_for(monkeypatch, AstrometryConfig(**BLOCK, verify_solution=False))
    assert ae.verify_solution is False


def test_on_reaches_astroeasy(monkeypatch):
    ae = _astroeasy_config_for(monkeypatch, AstrometryConfig(**BLOCK, verify_solution=True))
    assert ae.verify_solution is True
