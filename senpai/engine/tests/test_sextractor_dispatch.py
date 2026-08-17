"""The SExtractor sidereal dispatch must not hand the solve an empty source list.

Under ``astrometry.source_extractor = "sextractor"`` the sidereal path replaces the point
detector's source list with SExtractor's. `_detect_sources_sextractor` returns an empty table
when background estimation misbehaves, so an unconditional replacement would send a frame to
the plate solve with zero sources where the default path had some -- strictly worse than not
opting in at all.

Our 134-collect benchmark never produces that condition (every zero-source line in its logs
comes from the rate-track detector on satellite-free frames), so it cannot validate the guard;
these tests are the only thing that does. Hermetic: extraction is stubbed and
``pipeline_mode='detect'`` skips the solve, so no Astrometry.net install or catalog is needed.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from astropy.io import fits

from senpai.core import config as cfg_mod
from senpai.core.config import get_or_initialize_config
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.starfield import StarInImage, StarListImage
from senpai.engine.processing import sidereal as sidereal_mod

CONFIG_DIR = Path(__file__).resolve().parents[2] / "resources" / "config"

POINT_DETECTOR_SOURCES = [
    StarInImage(x=10.0, y=10.0, counts=1000.0),
    StarInImage(x=30.0, y=40.0, counts=800.0),
]


@pytest.fixture
def sextractor_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the config singleton at a copy in sextractor mode, detect-only.

    AppConfig is frozen, so this copies rather than mutates; monkeypatch restores the
    original singleton, which is shared with other test modules.
    """
    base = get_or_initialize_config(CONFIG_DIR / "local.yaml")
    astrometry = base.astrometry.model_copy(update={"source_extractor": "sextractor", "pipeline_mode": "detect"})
    monkeypatch.setattr(cfg_mod, "_config_instance", base.model_copy(update={"astrometry": astrometry}))


@pytest.fixture
def stubbed_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the point detector and the daofind seed; leave the SExtractor call to the test."""

    def fake_point_sources(
        fits_image: ProcessedFitsImage, max_detections: int | None = None
    ) -> tuple[StarListImage, float]:
        return (
            StarListImage(
                detections=list(POINT_DETECTOR_SOURCES),
                image_metadata=ImageMetadata(width=64, height=64),
                sat_level=50000.0,
            ),
            4.75,
        )

    monkeypatch.setattr(sidereal_mod, "extract_point_sources", fake_point_sources)


def _image(width: int = 64, height: int = 64) -> ProcessedFitsImage:
    """Build a frame with faint structure, so preprocessing has something to work on."""
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, size=(height, width)).astype(np.float32)
    return ProcessedFitsImage(
        data=data,
        header=fits.Header(),
        data_type=data.dtype,
        metadata=ImageMetadata(width=width, height=height),
    )


def test_empty_sextractor_result_keeps_the_point_detector_sources(
    sextractor_mode: None,
    stubbed_extraction: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty SExtractor extraction must not wipe the source list."""
    monkeypatch.setattr(
        "senpai.engine.detection.point.sextractor.extract_sextractor_sources",
        lambda image, max_detections=100: StarListImage(
            detections=[], image_metadata=ImageMetadata(width=64, height=64)
        ),
    )

    with caplog.at_level("WARNING"):
        starfield = sidereal_mod.process_astrometry_fits_sidereal(_image())

    assert [(s.x, s.y) for s in starfield.detections] == [(10.0, 10.0), (30.0, 40.0)]
    assert "SExtractor found no sources" in caplog.text


def test_non_empty_sextractor_result_replaces_the_source_list(
    sextractor_mode: None, stubbed_extraction: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal case still replaces: the whole point of the dispatch."""
    monkeypatch.setattr(
        "senpai.engine.detection.point.sextractor.extract_sextractor_sources",
        lambda image, max_detections=100: StarListImage(
            detections=[StarInImage(x=55.0, y=5.0, counts=2500.0)],
            image_metadata=ImageMetadata(width=64, height=64),
        ),
    )

    starfield = sidereal_mod.process_astrometry_fits_sidereal(_image())

    assert [(s.x, s.y) for s in starfield.detections] == [(55.0, 5.0)]
