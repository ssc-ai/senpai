"""Tests for the reduced ``astrometry.pipeline_mode`` values ('detect_solve', 'detect').

These modes trim the sidereal pipeline down to point-source detection (plus, for
'detect_solve', the plate solve) so a frame yields a detection-stage FWHM in
seconds instead of tens of seconds — the autofocus-sweep case, where WCS
refinement, the catalog query, catalog FWHM, and photometry are all wasted work.

Everything here is hermetic: the solver, catalog, refinement, and FWHM-measurement
calls are monkeypatched, so no Astrometry.net install, index files, or catalog
access is required.  The assertions that a stage was *not* called are the point of
the tests — that is what the modes buy.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from astropy.io import fits
from pydantic import ValidationError

from senpai.core import config as cfg_mod
from senpai.core.config import AstrometryConfig, get_or_initialize_config
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import CollectionMetadata, ImageMetadata
from senpai.engine.models.senpai import SenpaiRun, SiderealFrame
from senpai.engine.models.starfield import (
    StarField,
    StarInImage,
    StarListImage,
    StarListSpace,
)
from senpai.engine.processing import collect as collect_mod
from senpai.engine.processing import sidereal as sidereal_mod

CONFIG_DIR = Path(__file__).resolve().parents[2] / "resources" / "config"

DETECTION_FWHM = 4.75


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _use_pipeline_mode(monkeypatch, mode: str):
    """Point the process-wide config singleton at a copy with ``pipeline_mode=mode``.

    AppConfig is frozen, so this copies rather than mutates; monkeypatch restores
    the original singleton afterwards (it is shared with other test modules).
    """
    base = get_or_initialize_config(CONFIG_DIR / "local.yaml")
    astrometry = base.astrometry.model_copy(update={"pipeline_mode": mode})
    monkeypatch.setattr(cfg_mod, "_config_instance", base.model_copy(update={"astrometry": astrometry}))


def _fake_image(width: int = 64, height: int = 64) -> ProcessedFitsImage:
    data = np.zeros((height, width), dtype=np.float32)
    return ProcessedFitsImage(
        data=data,
        header=fits.Header(),
        data_type=data.dtype,
        metadata=ImageMetadata(width=width, height=height),
    )


def _fake_sources(width: int = 64, height: int = 64) -> StarListImage:
    return StarListImage(
        detections=[
            StarInImage(x=10.0, y=10.0, counts=1000.0),
            StarInImage(x=30.0, y=40.0, counts=800.0),
        ],
        image_metadata=ImageMetadata(width=width, height=height),
        sat_level=50000.0,
    )


def _fake_wcs() -> WCSModel:
    """A minimal, valid tangent-plane WCS — 1 arcsec/pixel."""
    return WCSModel(
        WCSAXES=2,
        NAXIS1=64,
        NAXIS2=64,
        CRPIX1=32.0,
        CRPIX2=32.0,
        PC1_1=1.0,
        PC1_2=0.0,
        PC2_1=0.0,
        PC2_2=1.0,
        CDELT1=-1.0 / 3600.0,
        CDELT2=1.0 / 3600.0,
        CUNIT1="deg",
        CUNIT2="deg",
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
        CRVAL1=180.0,
        CRVAL2=25.0,
    )


@pytest.fixture
def stubbed_stages(monkeypatch):
    """Stub every stage of the sidereal pipeline and record which ones ran."""
    calls: dict[str, int] = {
        "extract": 0,
        "solve": 0,
        "refine": 0,
        "catalog": 0,
        "catalog_fwhm": 0,
    }

    def fake_extract(fits_image, max_detections=None):
        calls["extract"] += 1
        return _fake_sources(), DETECTION_FWHM

    def fake_solve(sources, wcs=None):
        calls["solve"] += 1
        return StarField(
            wcs=_fake_wcs(),
            fit=True,
            detections=sources.detections,
            image_metadata=sources.image_metadata,
        )

    def fake_refine(sidereal_frame):
        calls["refine"] += 1
        return sidereal_frame

    def fake_catalog(wcs, max_stars=None, **kwargs):
        calls["catalog"] += 1
        return StarListSpace(stars=[], image_metadata=ImageMetadata(width=64, height=64))

    def fake_catalog_fwhm(*args, **kwargs):
        calls["catalog_fwhm"] += 1
        from senpai.engine.models.metadata import FWHMMetadata

        return FWHMMetadata(
            n_measurements=25,
            median_fwhm=9.0,  # deliberately != DETECTION_FWHM
            mean_fwhm=9.0,
            std_fwhm=0.5,
            min_fwhm=8.0,
            max_fwhm=10.0,
            fwhm_vs_position=[],
            fwhm_vs_magnitude=[],
            fwhm_vs_counts=[],
        )

    monkeypatch.setattr(sidereal_mod, "extract_point_sources", fake_extract)
    monkeypatch.setattr(sidereal_mod, "solve_field", fake_solve)
    monkeypatch.setattr(sidereal_mod, "refine_sidereal_frame", fake_refine)
    monkeypatch.setattr(sidereal_mod, "query_catalog", fake_catalog)
    monkeypatch.setattr(sidereal_mod, "measure_fwhm_from_catalog_stars", fake_catalog_fwhm)
    monkeypatch.setattr(sidereal_mod, "extract_boresight_from_header", lambda header: (None, None))
    return calls


# --------------------------------------------------------------------------
# Config field
# --------------------------------------------------------------------------


def _min_astrometry(**overrides) -> dict:
    data = {
        "indices_series": "5200_LITE",
        "indices_path": "/nonexistent/idx",
        "max_sources": 500,
        "min_sources_for_attempt": 4,
        "min_width_degrees": 0.1,
        "max_width_degrees": 10.0,
        "cpulimit_seconds": 30,
        "docker_image": None,
    }
    data.update(overrides)
    return data


def test_pipeline_mode_defaults_to_full():
    assert AstrometryConfig(**_min_astrometry()).pipeline_mode == "full"


@pytest.mark.parametrize("mode", ["full", "detect_solve", "detect"])
def test_pipeline_mode_accepts_known_modes(mode):
    assert AstrometryConfig(**_min_astrometry(pipeline_mode=mode)).pipeline_mode == mode


def test_pipeline_mode_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        AstrometryConfig(**_min_astrometry(pipeline_mode="detect_only"))


def test_shipped_configs_default_to_full():
    """No shipped config silently opts into a reduced mode."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg_data = cfg_mod.load_yaml(path)
        assert cfg_data.get("astrometry", {}).get("pipeline_mode", "full") == "full", (
            f"{path.name} enables a reduced pipeline_mode"
        )


# --------------------------------------------------------------------------
# process_astrometry_fits_sidereal
# --------------------------------------------------------------------------


def test_detect_mode_skips_solve_and_everything_after(monkeypatch, stubbed_stages):
    _use_pipeline_mode(monkeypatch, "detect")

    starfield = sidereal_mod.process_astrometry_fits_sidereal(_fake_image())

    assert stubbed_stages["extract"] == 1
    assert stubbed_stages["solve"] == 0
    assert stubbed_stages["refine"] == 0
    assert stubbed_stages["catalog"] == 0
    assert stubbed_stages["catalog_fwhm"] == 0

    assert starfield.fit is False
    assert starfield.wcs is None
    assert starfield.catalog_stars is None
    assert starfield.detections  # detection stage still ran
    assert starfield.detection_metadata.pixel_fwhm == DETECTION_FWHM
    assert starfield.fwhm_stats.median_fwhm == DETECTION_FWHM
    assert starfield.fwhm_stats.n_measurements == 1


def test_detect_solve_mode_keeps_wcs_and_plate_scale(monkeypatch, stubbed_stages):
    _use_pipeline_mode(monkeypatch, "detect_solve")

    starfield = sidereal_mod.process_astrometry_fits_sidereal(_fake_image())

    assert stubbed_stages["extract"] == 1
    assert stubbed_stages["solve"] == 1
    assert stubbed_stages["refine"] == 0
    assert stubbed_stages["catalog"] == 0
    assert stubbed_stages["catalog_fwhm"] == 0

    # fit stays False so the collect pipeline's .fit-gated passes stay skipped,
    # but the WCS (and with it the plate scale) survives for consumers.
    assert starfield.fit is False
    assert starfield.wcs is not None
    assert starfield.wcs_metadata is not None
    assert starfield.wcs_metadata.x_ifov_arcsec == pytest.approx(1.0, rel=1e-3)

    assert starfield.detection_metadata.pixel_fwhm == DETECTION_FWHM
    assert starfield.fwhm_stats.median_fwhm == DETECTION_FWHM


def test_full_mode_still_runs_the_whole_pipeline(monkeypatch, stubbed_stages):
    """Regression guard: the default mode is untouched by this change."""
    _use_pipeline_mode(monkeypatch, "full")

    starfield = sidereal_mod.process_astrometry_fits_sidereal(_fake_image())

    assert stubbed_stages["solve"] == 1
    assert stubbed_stages["refine"] == 1
    assert stubbed_stages["catalog"] == 1
    assert stubbed_stages["catalog_fwhm"] == 1

    assert starfield.fit is True
    assert starfield.wcs is not None
    # Catalog-measured FWHM wins in full mode, not the detection-stage value.
    assert starfield.fwhm_stats.median_fwhm == 9.0
    assert starfield.detection_metadata.pixel_fwhm == 9.0


# --------------------------------------------------------------------------
# process_senpai_collect
# --------------------------------------------------------------------------


@pytest.fixture
def stubbed_collect(monkeypatch):
    """Neutralize preprocessing and frame organization for collect-level tests."""
    monkeypatch.setattr(
        "senpai.engine.utils.preprocessing.preprocess_image",
        lambda frame, config, store_intermediates=False: frame,
    )

    def fake_organize(cls, frames, id="", force_track_mode=None):  # noqa: A002 - matches the real signature
        return SenpaiRun(
            id=id,
            num_frames=len(frames),
            collect_metadata=CollectionMetadata(),
            sidereal_frames=[SiderealFrame(frame=frame, index=i) for i, frame in enumerate(frames)],
        )

    monkeypatch.setattr(SenpaiRun, "organize_senpai_frames", classmethod(fake_organize))


def test_collect_detect_mode_completes_without_a_wcs(monkeypatch, stubbed_stages, stubbed_collect):
    """fit=False must not land the run in the 'No valid WCS solution found' state."""
    _use_pipeline_mode(monkeypatch, "detect")

    # Anything past the sidereal loop needs a WCS chain and must not be reached.
    def _unreachable(*args, **kwargs):  # pragma: no cover - the assertion is the point
        raise AssertionError("post-solve stage ran in detect mode")

    monkeypatch.setattr(collect_mod, "solve_shift", _unreachable)
    monkeypatch.setattr(collect_mod, "refine_sidereal_frame", _unreachable)

    senpai_run = collect_mod.process_senpai_collect([_fake_image(), _fake_image()], id="focus-sweep")

    assert senpai_run.completed is True
    assert senpai_run.error_message is None
    assert senpai_run.compute_seconds is not None

    # Every frame processed — the loop does not stop at the first one, since a
    # focus sweep needs a FWHM per frame.
    assert stubbed_stages["extract"] == 2
    assert stubbed_stages["solve"] == 0
    assert len(senpai_run.sidereal_frames) == 2

    for frame in senpai_run.sidereal_frames:
        assert frame.starfield is not None
        assert frame.starfield.fit is False
        assert frame.starfield.detection_metadata.pixel_fwhm == DETECTION_FWHM
        assert frame.starfield.fwhm_stats.median_fwhm == DETECTION_FWHM
        assert frame.seeing is not None
        assert frame.seeing.pixel_fwhm == DETECTION_FWHM
        # No catalog work, no photometry.
        assert not frame.starfield.catalog_stars
        assert frame.photometry_summary is None


def test_collect_detect_solve_mode_completes_with_a_wcs(monkeypatch, stubbed_stages, stubbed_collect):
    _use_pipeline_mode(monkeypatch, "detect_solve")

    senpai_run = collect_mod.process_senpai_collect([_fake_image()], id="focus-sweep")

    assert senpai_run.completed is True
    assert senpai_run.error_message is None
    assert stubbed_stages["solve"] == 1
    assert stubbed_stages["catalog"] == 0

    starfield = senpai_run.sidereal_frames[0].starfield
    assert starfield.fit is False
    assert starfield.wcs is not None
    assert starfield.fwhm_stats.median_fwhm == DETECTION_FWHM


def test_collect_full_mode_unsolved_frame_still_errors(monkeypatch, stubbed_stages, stubbed_collect):
    """Regression guard: a genuine solve failure in full mode is still an error."""
    _use_pipeline_mode(monkeypatch, "full")

    def failed_solve(sources, wcs=None):
        stubbed_stages["solve"] += 1
        return StarField(
            wcs=None,
            detections=sources.detections,
            image_metadata=sources.image_metadata,
        )

    monkeypatch.setattr(sidereal_mod, "solve_field", failed_solve)

    senpai_run = collect_mod.process_senpai_collect([_fake_image()], id="unsolved")

    assert senpai_run.completed is False
    assert senpai_run.error_message == "No valid WCS solution found"
    # ...while the retained starfield still carries the detection-stage FWHM.
    starfield = senpai_run.sidereal_frames[0].starfield
    assert starfield is not None
    assert starfield.fwhm_stats.median_fwhm == DETECTION_FWHM


# --------------------------------------------------------------------------
# Per-call override: `pipeline_mode=` argument beats the configured mode
#
# This is what lets ONE process interleave reduced-mode batches (an autofocus
# focus sweep) with full science batches. The config stays 'full'; only the
# sweep call asks for less. Nothing global is mutated, so ordering and
# concurrency between batches cannot leak one batch's mode into another.
# --------------------------------------------------------------------------


def test_sidereal_override_beats_configured_full(monkeypatch, stubbed_stages):
    """config=full, call=detect -> no solve for that call."""
    _use_pipeline_mode(monkeypatch, "full")

    starfield = sidereal_mod.process_astrometry_fits_sidereal(_fake_image(), pipeline_mode="detect")

    assert stubbed_stages["solve"] == 0
    assert starfield.fit is False
    assert starfield.detection_metadata.pixel_fwhm == DETECTION_FWHM


def test_sidereal_override_beats_configured_detect(monkeypatch, stubbed_stages):
    """The override works in both directions: config=detect, call=full -> full pipeline."""
    _use_pipeline_mode(monkeypatch, "detect")

    sidereal_mod.process_astrometry_fits_sidereal(_fake_image(), pipeline_mode="full")

    assert stubbed_stages["solve"] == 1
    assert stubbed_stages["catalog"] == 1


def test_omitting_the_override_uses_configured_mode(monkeypatch, stubbed_stages):
    """Byte-identical to the pre-override behaviour for every existing caller."""
    _use_pipeline_mode(monkeypatch, "detect")

    sidereal_mod.process_astrometry_fits_sidereal(_fake_image())

    assert stubbed_stages["solve"] == 0


def test_collect_override_runs_a_sweep_while_config_stays_full(monkeypatch, stubbed_stages, stubbed_collect):
    """The autofocus case end-to-end: science config untouched, sweep batch reduced."""
    _use_pipeline_mode(monkeypatch, "full")

    def _unreachable(*args, **kwargs):  # pragma: no cover - the assertion is the point
        raise AssertionError("post-solve stage ran for an overridden detect batch")

    monkeypatch.setattr(collect_mod, "solve_shift", _unreachable)
    monkeypatch.setattr(collect_mod, "refine_sidereal_frame", _unreachable)

    senpai_run = collect_mod.process_senpai_collect(
        [_fake_image(), _fake_image()], id="focus-sweep", pipeline_mode="detect"
    )

    assert senpai_run.completed is True
    assert senpai_run.error_message is None
    assert stubbed_stages["solve"] == 0
    assert len(senpai_run.sidereal_frames) == 2
    for frame in senpai_run.sidereal_frames:
        assert frame.starfield.fwhm_stats.median_fwhm == DETECTION_FWHM


def test_override_does_not_leak_into_config(monkeypatch, stubbed_stages, stubbed_collect):
    """A reduced-mode batch must not degrade the science batch that follows it."""
    _use_pipeline_mode(monkeypatch, "full")

    collect_mod.process_senpai_collect([_fake_image()], id="focus-sweep", pipeline_mode="detect")
    assert stubbed_stages["solve"] == 0
    assert get_or_initialize_config().astrometry.pipeline_mode == "full"

    # Same process, next batch, no override: the full pipeline must be back.
    collect_mod.process_senpai_collect([_fake_image()], id="science")
    assert stubbed_stages["solve"] == 1
    assert stubbed_stages["catalog"] == 1
