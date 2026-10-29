"""Stage-1 candidate budget, refiner length window, cutout size and seed sanity checks.

Found on sdasim frames of a Canon 60D behind a 135 mm lens (2 px stars, 6.57"/px) rendered by
oblio's simulator with SSTRC7 stars, 2026-10-28:

- 30 s, 23.6 px trails: every one of the 30 candidates stage 1 examines brightest-first was a
  saturated trail, so it found no streak for 12 of 13 FWHM seeds.
- 10 s, 7.8 px trails: 27 of the 30 went to saturated trails; the one cutout left had a
  neighbouring star at its edge, and the outermost half-peak crossings spanned to it (50-54 px)
  whenever the seed made the cutout large enough (seed 4.6-5.5 px).
- Sparse frames (few unsaturated stars): stage 1 measured faint blobs at scattered angles
  (2.7 px at 102 deg for 23.6 px at 144 deg) and the refiner turned that into 47-157 px.

``tests/data/sim/sdasim_135mm_30s_saturated.fits.gz`` is a 1024 px crop of the 30 s frame
(uint16); its header carries the truth (SYN_STRK, TRAILPA). The full frames are too large for
the repository: set SENPAI_SDASIM_FRAMES to their directory to also run the full-frame sweep.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR, TEST_DATA_DIR
from senpai.engine.detection.streak import rate_extraction
from senpai.engine.detection.streak.extraction import (
    extract_streak_dims_robust,
    refine_robust_streak,
)
from senpai.engine.detection.streak.rate_extraction import (
    extract_rate_streak_measurement,
    prepare_rate_frame,
)
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.senpai import RateTrackFrame
from senpai.engine.models.streak_measurement import StreakMeasurement

SATURATED_CROP = TEST_DATA_DIR / "sim" / "sdasim_135mm_30s_saturated.fits.gz"


@pytest.fixture(scope="module", autouse=True)
def _config():
    initialize_config(CONFIG_DIR / "burr.yaml")
    get_config().plotting.debug = False


def _rate_frame(data: np.ndarray) -> RateTrackFrame:
    d = np.asarray(data, dtype=np.float32)
    d = d - np.float32(np.median(d))
    h, w = d.shape
    img = ProcessedFitsImage(
        data=d, header=fits.Header(), data_type=d.dtype, metadata=ImageMetadata(width=w, height=h)
    )
    return RateTrackFrame(frame=img, index=0)


def _axial_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _trail(length: int, fwhm: float, rot: float, size: int) -> np.ndarray:
    """A flat trail with a Gaussian cross-section, rotated, peak 1, in a size x size stamp."""
    from scipy.ndimage import rotate

    sigma = fwhm / 2.355
    c = size // 2
    cross = np.exp(-0.5 * ((np.arange(size) - c) / sigma) ** 2)[:, None]
    bar = np.zeros((size, size))
    bar[:, c - length // 2 : c - length // 2 + length] = cross
    bar = rotate(bar, -rot, reshape=False, order=1)
    return bar / bar.max()


# ---------------------------------------------------------------- candidate budget


@pytest.mark.parametrize("seed", [None, 2.0, 4.0, 6.0])
def test_sdasim_saturated_frame_measures_the_trail(seed):
    """Saturated trails no longer use up the 30-candidate budget (sdasim crop)."""
    data = fits.getdata(SATURATED_CROP)
    hdr = fits.getheader(SATURATED_CROP)
    m, _psf, _f = extract_rate_streak_measurement(_rate_frame(data), n_streaks=10, initial_fwhm=seed)
    assert m is not None, "no streak model: the budget was used up by saturated trails"
    assert m.length == pytest.approx(float(hdr["SYN_STRK"]), abs=4.0)
    assert _axial_diff(float(m.rotation), float(hdr["TRAILPA"])) < 8.0


def test_clipped_trails_do_not_exhaust_the_budget():
    """60 bright clipped trails ahead of 20 unsaturated ones: stage 1 still finds the trail."""
    rng = np.random.default_rng(3)
    size, length, rot = 1800, 24, 40.0
    stamp = _trail(length, 3.0, rot, 61)
    img = rng.normal(0.0, 0.01, (size, size))
    for i in range(80):
        y, x = (int(v) for v in rng.integers(60, size - 120, 2))
        star = np.minimum(stamp * 20.0, 1.0) if i < 60 else stamp * 0.4  # flat-topped vs unsaturated
        img[y : y + 61, x : x + 61] = np.maximum(img[y : y + 61, x : x + 61], star)
    m, _psf, _f = extract_streak_dims_robust(
        prepare_rate_frame(_rate_frame(img)), n_streaks=10, length=None, rotation=None, fwhm=3.0
    )
    assert m is not None
    assert _axial_diff(float(m.rotation), rot) < 8.0


# ---------------------------------------------------------------- refiner window


def test_refined_length_ignores_a_neighbour_on_the_trail_row():
    """The half-peak run that holds the centre, not the outermost crossings across the cutout."""
    psf = np.zeros((100, 100))
    psf[30:70, 30:70] = _trail(10, 3.0, 0.0, 40)  # 10 px trail, centred
    psf[30:70, 0:20] = 0.7 * _trail(6, 3.0, 0.0, 40)[:, 10:30]  # neighbour at the left edge
    m, _w = refine_robust_streak(psf, StreakMeasurement(rotation=0.0, length=6.0, fwhm=3.0))
    assert m.length < 20.0  # the trail (~10 px + PSF), not the ~50 px span to the neighbour


def test_fragmented_trail_still_bridged():
    """A mid-trail dip below half peak is still bridged (DAO-01 behaviour kept)."""
    psf = _trail(40, 8.0, 0.0, 160)
    c = 80
    psf[:, c - 3 : c + 3] *= 0.3
    m, _w = refine_robust_streak(psf, StreakMeasurement(rotation=0.0, length=40.0, fwhm=8.0))
    assert m.length == pytest.approx(40, abs=3)


# ---------------------------------------------------------------- seed sanity


def test_aligned_candidates_are_trusted_over_a_wrong_seed_angle():
    """The seed's grid angle can be >20 deg off with a wide kernel; consistent trails win."""
    rng = np.random.default_rng(5)
    size, length, rot = 1200, 24, 144.0
    stamp = _trail(length, 2.5, rot, 61)
    img = rng.normal(0.0, 0.01, (size, size))
    for _ in range(40):
        y, x = (int(v) for v in rng.integers(40, size - 100, 2))
        img[y : y + 61, x : x + 61] += stamp * rng.uniform(0.3, 0.8)
    m, _psf, _f = extract_streak_dims_robust(
        prepare_rate_frame(_rate_frame(img)), n_streaks=10, length=27.0, rotation=165.0, fwhm=2.5
    )
    assert m is not None
    assert _axial_diff(float(m.rotation), rot) < 8.0
    assert m.prior_length == 27.0


def test_refined_length_inconsistent_with_seed_is_rejected(monkeypatch):
    """A refined length far beyond the matched-filter seed gives no streak, not a wrong one."""
    seed = StreakMeasurement(rotation=144.0, length=5.0, fwhm=3.0, prior_length=27.0)
    monkeypatch.setattr(rate_extraction, "extract_streak_dims_robust", lambda *a, **k: (seed, np.ones((9, 9)), 3.0))
    monkeypatch.setattr(
        rate_extraction,
        "refine_robust_streak",
        lambda psf, m, frame_index=None: (StreakMeasurement(rotation=144.0, length=104.0, fwhm=3.0), 3.0),
    )
    m, _psf, _f = extract_rate_streak_measurement(_rate_frame(np.zeros((64, 64))))
    assert m is None
    monkeypatch.setattr(
        rate_extraction,
        "refine_robust_streak",
        lambda psf, m, frame_index=None: (StreakMeasurement(rotation=144.0, length=7.0, fwhm=3.0), 3.0),
    )
    m, _psf, _f = extract_rate_streak_measurement(_rate_frame(np.zeros((64, 64))))
    assert m is not None and m.length == 7.0  # short trails: the 12 px grid floor is not a reason


def test_prior_length_is_not_serialized():
    m = StreakMeasurement(rotation=10.0, length=5.0, fwhm=2.0, prior_length=27.0)
    assert "prior_length" not in m.model_dump()
    assert "prior_length" not in m.model_dump_json()


# ---------------------------------------------------------------- full frames (optional)

_FRAMES = os.environ.get("SENPAI_SDASIM_FRAMES")


@pytest.mark.slow
@pytest.mark.skipif(not _FRAMES, reason="set SENPAI_SDASIM_FRAMES to the sdasim frame directory")
@pytest.mark.parametrize("seed", [None, 1.5, 3.0, 4.6, 5.0, 5.5, 8.0])
def test_full_frame_seed_sweep(seed):
    """10 s frame, 7.8 px trails: was 50-54 px for seeds 4.6-5.5 (the neighbour at the edge)."""
    path = Path(_FRAMES) / "135mm" / "drift5_10s.fits"
    data = fits.getdata(path)
    true_l = float(fits.getheader(path)["SYN_STRK"])
    m, _psf, _f = extract_rate_streak_measurement(_rate_frame(data), n_streaks=10, initial_fwhm=seed)
    assert m is not None and m.length == pytest.approx(true_l, abs=2.5)
