"""Tests for streak length/width refinement (refine_robust_streak).

The length is the full-width at half of a *stable* (hot-pixel-robust) max,
measured from the streak collapsed to 1D — so it survives the streak
fragmenting into blobs at the 0.5 level (tracking jitter / optics), a single
hot pixel, and an off seed. These guard the DAO-01 fix where the old
profile-extent adoption over-measured (42 px for a ~40 px FWHM trail) and the
2D connected-component core under-measured fragmented trails.
"""

from __future__ import annotations

import numpy as np
import pytest

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.streak.extraction import refine_robust_streak
from senpai.engine.models.streak_measurement import StreakMeasurement


@pytest.fixture(scope="module", autouse=True)
def _config():
    initialize_config(CONFIG_DIR / "burr.yaml")
    get_config().plotting.debug = False  # skip the debug plot in the refiner


def _streak(length: int, fwhm: float = 8.0, *, dip: bool = False,
            hot: bool = False, rot: float = 0.0) -> np.ndarray:
    """A synthetic horizontal streak: flat trail with a Gaussian cross-section."""
    n = max(160, length + 50)
    cy = n // 2
    sigma = fwhm / 2.355
    x0 = (n - length) // 2
    cross = np.exp(-0.5 * ((np.arange(n) - cy) / sigma) ** 2)[:, None]
    bar = np.zeros((n, n))
    bar[:, x0:x0 + length] = cross
    if dip:  # knock the middle below 0.5 → fragments at the half level
        bar[:, x0 + length // 2 - 3:x0 + length // 2 + 3] *= 0.3
    if hot:  # lone hot pixels off the trail
        bar[cy + 1, x0 - 25] = 10.0
        bar[cy - 3, x0 + length + 20] = 8.0
    if rot:
        from scipy.ndimage import rotate
        bar = rotate(bar, -rot, reshape=False, order=1)
    return bar


# Off seed on purpose — the refiner must recover length from the image.
_SEED = StreakMeasurement(rotation=0.0, length=25.0, fwhm=6.0)


@pytest.mark.parametrize("length", [40, 60, 80])
def test_measures_clean_streak_length(length):
    m, _ = refine_robust_streak(_streak(length), _SEED)
    assert m.length == pytest.approx(length, abs=3)
    assert m.fwhm == pytest.approx(8.0, abs=2)


def test_fragmented_streak_not_truncated():
    # A 0.5-level gap mid-trail must not shorten the measured length.
    m, _ = refine_robust_streak(_streak(40, dip=True), _SEED)
    assert m.length == pytest.approx(40, abs=3)


def test_hot_pixels_do_not_inflate_or_break():
    m, _ = refine_robust_streak(_streak(40, hot=True), _SEED)
    assert m.length == pytest.approx(40, abs=3)
    assert m.fwhm == pytest.approx(8.0, abs=2)


def test_fragmented_plus_hot():
    m, _ = refine_robust_streak(_streak(60, dip=True, hot=True), _SEED)
    assert m.length == pytest.approx(60, abs=4)


def test_rotated_streak():
    seed = StreakMeasurement(rotation=30.0, length=25.0, fwhm=6.0)
    m, _ = refine_robust_streak(_streak(50, rot=30.0), seed)
    assert m.length == pytest.approx(50, abs=4)


def _streak_field(length: int, rot: float, n: int = 30, size: int = 900) -> np.ndarray:
    """A field of identical streaks for seed-estimation tests."""
    rng = np.random.default_rng(0)
    one = _streak(length, rot=rot)
    s = one.shape[0]
    img = np.zeros((size, size))
    for _ in range(n):
        y, x = int(rng.integers(0, size - s)), int(rng.integers(0, size - s))
        img[y:y + s, x:x + s] += one * rng.uniform(0.3, 1.0)
    img += rng.normal(0, 0.02, img.shape)
    return img


class TestSeedEstimate:
    def test_recovers_ballpark_seed_not_frame_fraction(self):
        from senpai.engine.detection.streak.extraction import _estimate_streak_seed
        # Old default would be size*0.05 = 45 here regardless of the streak;
        # the estimator must track the actual streak instead.
        for length, rot in [(40, 0.0), (60, 60.0), (30, 120.0)]:
            est_len, est_rot = _estimate_streak_seed(_streak_field(length, rot), crop=900)
            # Ballpark only — it just sizes the cutout; the refiner does the rest.
            # The key win is it tracks the streak instead of the old frame-fraction
            # default (which would over-size the PSF ~10x on a large frame).
            assert 0.4 * length <= est_len <= 3.0 * length
            # Angle within one search step (15 deg), modulo 180.
            d = abs(est_rot - rot) % 180.0
            assert min(d, 180.0 - d) <= 16.0
