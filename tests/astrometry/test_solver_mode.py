"""Backward-compatibility tests for the additive solver_mode config.

The contract (astroeasy docs/catalog-native-solving-roadmap.md §0.2): existing
configs — which never mention solver_mode or fast_solve — must parse and behave
exactly as before, and non-dotnet modes must be a clean opt-in.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from senpai.core.config import AstrometryConfig

LEGACY_ASTROMETRY_BLOCK = {
    # A pre-solver_mode config block: required fields only, as deployed configs have.
    "indices_series": "5200_LITE",
    "indices_path": "/data/indices",
    "max_sources": 100,
    "min_sources_for_attempt": 10,
    "min_width_degrees": 1.0,
    "max_width_degrees": 3.0,
    "cpulimit_seconds": 60,
    "docker_image": "astrometry-cli",
}


class TestSolverModeCompat:
    def test_legacy_config_defaults_to_dotnet(self):
        cfg = AstrometryConfig(**LEGACY_ASTROMETRY_BLOCK)
        assert cfg.solver_mode == "dotnet"
        assert cfg.fast_solve.mirror_dir is None
        assert cfg.fast_solve.tetra3_db_path is None
        assert cfg.fast_solve.sensor_profile is None

    @pytest.mark.parametrize("mode", ["dotnet", "tetra3", "chain"])
    def test_valid_modes_accepted(self, mode):
        cfg = AstrometryConfig(**LEGACY_ASTROMETRY_BLOCK, solver_mode=mode)
        assert cfg.solver_mode == mode

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            AstrometryConfig(**LEGACY_ASTROMETRY_BLOCK, solver_mode="cascade")

    def test_fast_solve_block_parses(self):
        cfg = AstrometryConfig(
            **LEGACY_ASTROMETRY_BLOCK,
            solver_mode="chain",
            fast_solve={"mirror_dir": "/data/gaia_mirror", "tetra3_db_path": "/data/db.npz"},
        )
        assert cfg.fast_solve.mirror_dir == "/data/gaia_mirror"
        assert cfg.fast_solve.tetra3_db_path == "/data/db.npz"


@pytest.fixture
def stub_config(monkeypatch):
    """Point the astrometry adapter at a config we control."""
    import senpai.astrometry as adapter

    class _Stub:
        astrometry = AstrometryConfig(**LEGACY_ASTROMETRY_BLOCK)

    monkeypatch.setattr(adapter, "get_or_initialize_config", lambda: _Stub)
    return _Stub


class TestSolveFieldModeDispatch:
    def test_dotnet_mode_reaches_original_path(self, stub_config, xyls_data):
        """With the default mode, the dispatch is transparent: the original
        too-few-sources early return still triggers (no exception)."""
        from senpai.astrometry import solve_field

        block = dict(LEGACY_ASTROMETRY_BLOCK, min_sources_for_attempt=10**9)
        stub_config.astrometry = AstrometryConfig(**block)
        starfield = solve_field(xyls_data)
        assert starfield.wcs is None

    def test_tetra3_mode_without_mirror_fails_gracefully(self, stub_config, xyls_data):
        """tetra3 mode with no catalog configured: unfit StarField, no raise."""
        from senpai.astrometry import solve_field

        stub_config.astrometry = AstrometryConfig(**LEGACY_ASTROMETRY_BLOCK, solver_mode="tetra3")
        stub_config.star_catalog = None
        starfield = solve_field(xyls_data)
        assert not starfield.fit
        assert starfield.wcs is None


class TestCascadeEndToEnd:
    """Full senpai-adapter -> astroeasy-cascade path on a synthetic sky."""

    RA0, DEC0, W, H, SCALE = 150.0, 30.0, 2048, 2048, 2.0  # arcsec/px

    @pytest.fixture
    def synthetic(self, tmp_path):
        import json
        import math

        import numpy as np
        from astroeasy.catalog.mirror import MIRROR_DTYPE, load_mirror_index
        from astropy.wcs import WCS

        from senpai.engine.models.metadata import ImageMetadata
        from senpai.engine.models.starfield import StarInImage, StarListImage

        rng = np.random.default_rng(7)
        truth = WCS(naxis=2)
        truth.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        truth.wcs.crval = [self.RA0, self.DEC0]
        truth.wcs.crpix = [self.W / 2, self.H / 2]
        s, t = self.SCALE / 3600.0, math.radians(15.0)
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        truth.wcs.cd = rot @ np.array([[-s, 0.0], [0.0, s]])

        n = 500
        cosd = math.cos(math.radians(self.DEC0))
        ra = self.RA0 + rng.uniform(-1.2, 1.2, n) / cosd
        dec = self.DEC0 + rng.uniform(-1.2, 1.2, n)
        g = rng.uniform(8.0, 15.0, n)

        arr = np.zeros(n, dtype=MIRROR_DTYPE)
        arr["source_id"] = np.arange(1, n + 1)
        arr["ra"], arr["dec"], arr["g"] = ra, dec, g
        mirror = tmp_path / "mirror"
        mirror.mkdir()
        arr.tofile(mirror / "tile.bin")
        (mirror / "index.json").write_text(json.dumps({"tiles": {"t": {
            "file": "tile.bin", "ra_min": float(ra.min()), "ra_max": float(ra.max()),
            "dec_min": float(dec.min()), "dec_max": float(dec.max())}}}))
        load_mirror_index.cache_clear()

        px, py = truth.all_world2pix(ra, dec, 0)
        ok = (px > 10) & (px < self.W - 10) & (py > 10) & (py < self.H - 10) & (g < 14.0)
        sources = StarListImage(
            detections=[
                StarInImage(x=float(x + rng.normal(0, 0.3)), y=float(y + rng.normal(0, 0.3)),
                            counts=float(10 ** (-0.4 * (m - 20.0))))
                for x, y, m in zip(px[ok], py[ok], g[ok], strict=True)
            ],
            image_metadata=ImageMetadata(
                image_id="synthetic", width=self.W, height=self.H,
                boresight_ra=self.RA0 + 0.05, boresight_dec=self.DEC0 - 0.04,
            ),
        )
        return sources, str(mirror)

    def test_tetra3_mode_solves_via_t0(self, stub_config, synthetic):
        """Boresight + scale bounds, no tetra3 DB: the cascade's T0
        constrained tier solves it natively — no astrometry.net, no Docker."""
        from senpai.astrometry import solve_field

        sources, mirror = synthetic
        block = dict(
            LEGACY_ASTROMETRY_BLOCK,
            min_width_degrees=1.0, max_width_degrees=1.3,  # truth fov ~1.14 deg
        )
        stub_config.astrometry = AstrometryConfig(
            **block, solver_mode="tetra3",
            fast_solve={"mirror_dir": mirror},
        )
        starfield = solve_field(sources)
        assert starfield.fit
        assert starfield.wcs is not None
        assert starfield.astrometric_fit_stars  # gate matches -> catalog stars
        # WCS lands within arcseconds of the truth center.
        from astropy.wcs import WCS as AWCS

        hdr = {k: v for k, v in starfield.wcs.model_dump().items()
               if v is not None and k not in ("NAXIS1", "NAXIS2")}
        w = AWCS(hdr)
        ra_c, dec_c = w.all_pix2world(self.W / 2, self.H / 2, 0)
        import math
        err_asec = math.hypot(
            (float(ra_c) - self.RA0) * math.cos(math.radians(self.DEC0)),
            float(dec_c) - self.DEC0,
        ) * 3600
        assert err_asec < 5.0
