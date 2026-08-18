"""Tests for the lightweight box-based shift validation path.

``validation.py`` was slimmed to a single strategy: for a proposed (dx, dy)
shift between two frames it measures box statistics around each catalog star in
the source frame and at the shifted position in the target frame, computes a
correlation of the per-star net fluxes, and accepts the shift only if it
correlates much better than a set of random alternative shifts.

The tests build a synthetic pair of frames where frame B is frame A's stars
translated by a known integer (dx, dy). The true shift should validate with a
high correlation; a wrong shift should fail. Lower-level helpers
(``extract_box_statistics``, ``quick_correlation_from_boxes``) are exercised
directly as well.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import initialize_config, settings
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.streak.validation import (
    extract_box_statistics,
    quick_correlation_from_boxes,
    validate_proposed_shift,
    validate_shift_lightweight,
)
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.senpai import RateTrackFrame
from senpai.engine.models.starfield import StarField, StarInSpace


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialise the process-wide config, which the validation helpers read."""
    initialize_config(CONFIG_DIR / "burr.yaml")
    # Validation has debug-plot paths gated on config.plotting.debug; keep off.
    settings.plotting.debug = False


@pytest.fixture(autouse=True)
def _seed() -> None:
    # Validation draws random alternative shifts via the global numpy RNG.
    """Build a seeded generator, so a failure is reproducible rather than intermittent."""
    np.random.seed(1234)


# --------------------------------------------------------------------------- #
# Synthetic-frame helpers
# --------------------------------------------------------------------------- #
IMG = 512
BG = 100.0
PEAK = 4000.0
FWHM = 3.0


def _add_star(img: np.ndarray, x: float, y: float, peak: float, fwhm: float = FWHM) -> None:
    """Add one Gaussian star to a frame in place."""
    sigma = fwhm / 2.355
    half = int(np.ceil(4 * sigma))
    xi, yi = round(x), round(y)
    y0, y1 = max(0, yi - half), min(img.shape[0], yi + half + 1)
    x0, x1 = max(0, xi - half), min(img.shape[1], xi + half + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += peak * np.exp(-0.5 * (((xx - x) ** 2 + (yy - y) ** 2) / sigma**2))


def _star_field(n_stars: int = 30, seed: int = 0, margin: int = 60) -> StarField:
    """Return (positions, peaks) for n_stars with a spread of brightnesses."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(margin, IMG - margin, n_stars)
    ys = rng.uniform(margin, IMG - margin, n_stars)
    # A wide spread of peaks so the flux ordering is informative for correlation.
    peaks = rng.uniform(0.2 * PEAK, PEAK, n_stars)
    return list(zip(xs, ys, peaks, strict=True))


def _render(
    positions: list[tuple[float, float]], dx: float = 0.0, dy: float = 0.0, noise: float = 3.0, seed: int = 7
) -> np.ndarray:
    """Render a frame; each source is placed at (x - dx, y - dy)."""
    rng = np.random.default_rng(seed)
    img = np.full((IMG, IMG), BG, dtype=np.float64)
    img += rng.normal(0.0, noise, img.shape)
    for x, y, peak in positions:
        _add_star(img, x - dx, y - dy, peak)
    return img.astype(np.float32)


def _frame(data: np.ndarray, index: int = 0) -> RateTrackFrame:
    """Build a frame and its starfield for a given set of star positions."""
    img = ProcessedFitsImage(
        data=data,
        header=fits.Header(),
        data_type=np.dtype("uint16"),
        metadata=ImageMetadata(width=IMG, height=IMG),
    )
    return RateTrackFrame(frame=img, index=index, timestamp=datetime(2024, 1, 1))


def _catalog(positions: list[tuple[float, float]]) -> list[StarInSpace]:
    """Catalog stars at the SOURCE-frame positions (x, y), with magnitudes."""
    stars = []
    for i, (x, y, peak) in enumerate(positions):
        # Brighter peak -> smaller magnitude.
        mag = 20.0 - 2.5 * np.log10(peak)
        stars.append(StarInSpace(ra=10.0 + i * 0.001, dec=20.0, x=float(x), y=float(y), magnitude=float(mag)))
    return stars


# --------------------------------------------------------------------------- #
# extract_box_statistics
# --------------------------------------------------------------------------- #
class TestExtractBoxStatistics:
    """Extracting the small-box statistics that shift validation compares."""

    def test_returns_valid_stats_for_interior_point(self) -> None:
        """An interior position yields usable statistics."""
        img = np.full((50, 50), 10.0)
        img[25, 25] = 1000.0
        stats = extract_box_statistics(img, 25, 25, box_size=11)
        assert stats["valid"] is True
        assert stats["max"] == 1000.0
        assert stats["median"] == 10.0
        assert stats["sum"] == pytest.approx(10.0 * (121 - 1) + 1000.0)

    def test_invalid_when_box_out_of_bounds(self) -> None:
        """A box falling outside the frame is reported invalid rather than clipped.

        A clipped box averages fewer pixels and would compare unlike quantities.
        """
        img = np.full((50, 50), 10.0)
        stats = extract_box_statistics(img, 2, 2, box_size=11)
        assert stats["valid"] is False
        assert stats["max"] == 0.0

    def test_invalid_near_far_edge(self) -> None:
        """A box overrunning the far edge is invalid too, not just the near one."""
        img = np.full((50, 50), 10.0)
        stats = extract_box_statistics(img, 48, 48, box_size=11)
        assert stats["valid"] is False

    def test_box_centered_on_rounded_position(self) -> None:
        """The box centres on the rounded pixel, so a fractional position does not shift it."""
        img = np.zeros((50, 50))
        img[30, 20] = 500.0
        # x=20.3, y=29.6 rounds to (20, 30); box should capture the peak.
        stats = extract_box_statistics(img, 20.3, 29.6, box_size=5)
        assert stats["max"] == 500.0


# --------------------------------------------------------------------------- #
# quick_correlation_from_boxes
# --------------------------------------------------------------------------- #
class TestQuickCorrelation:
    """The cheap box-based correlation used to score a proposed shift."""

    def test_true_shift_correlates_strongly(self) -> None:
        """The correct shift scores high."""
        positions = _star_field(n_stars=30, seed=1)
        dx, dy = 20, -15
        source = _render(positions, dx=0, dy=0, seed=2)
        target = _render(positions, dx=dx, dy=dy, seed=3)
        cat = _catalog(positions)

        corr, n, _ = quick_correlation_from_boxes(target, source, dx, dy, cat, box_size=11, max_stars=50)
        assert n >= 4
        assert corr > 0.6

    def test_wrong_shift_correlates_poorly(self) -> None:
        """A wrong shift scores low, so the score discriminates rather than always passing."""
        positions = _star_field(n_stars=30, seed=1)
        dx, dy = 20, -15
        source = _render(positions, dx=0, dy=0, seed=2)
        target = _render(positions, dx=dx, dy=dy, seed=3)
        cat = _catalog(positions)

        # Offset the shift far enough that target boxes land on empty background.
        corr_true, _, _ = quick_correlation_from_boxes(target, source, dx, dy, cat, box_size=11)
        corr_wrong, _, _ = quick_correlation_from_boxes(target, source, dx + 40, dy + 40, cat, box_size=11)
        assert corr_true > corr_wrong

    def test_too_few_valid_stars_returns_zero(self) -> None:
        """Too few usable stars scores zero rather than a confident value from two points."""
        positions = _star_field(n_stars=2, seed=5)
        source = _render(positions, seed=2)
        target = _render(positions, dx=5, dy=5, seed=3)
        cat = _catalog(positions)
        corr, n, pairs = quick_correlation_from_boxes(target, source, 5, 5, cat, box_size=11)
        # Fewer than 3 valid -> correlation 0 and empty pair list.
        assert n < 3
        assert corr == 0.0
        assert pairs == []

    def test_respects_max_stars_limit(self) -> None:
        """The correlation stops at the star cap, bounding its cost per candidate."""
        positions = _star_field(n_stars=40, seed=8)
        source = _render(positions, seed=2)
        target = _render(positions, dx=10, dy=10, seed=3)
        cat = _catalog(positions)
        _, n, _ = quick_correlation_from_boxes(target, source, 10, 10, cat, box_size=11, max_stars=12)
        assert n <= 12


# --------------------------------------------------------------------------- #
# validate_shift_lightweight / validate_proposed_shift
# --------------------------------------------------------------------------- #
class TestValidateShift:
    """Accepting or rejecting a proposed frame-to-frame shift."""

    def _pair(self, dx: float, dy: float, n_stars: int = 30, seed: int = 1, noise: float = 3.0) -> tuple:
        """Build a source and target frame separated by a known shift."""
        positions = _star_field(n_stars=n_stars, seed=seed)
        source_img = _render(positions, dx=0, dy=0, noise=noise, seed=2)
        target_img = _render(positions, dx=dx, dy=dy, noise=noise, seed=3)
        return _frame(source_img, 0), _frame(target_img, 1), _catalog(positions)

    def test_true_shift_validates(self) -> None:
        """The true shift is accepted."""
        dx, dy = 20, -15
        source, target, cat = self._pair(dx, dy)
        valid, corr, streak, correction = validate_shift_lightweight(target, source, dx, dy, cat)
        assert bool(valid)
        assert corr > 0.6
        assert streak is None
        assert correction == (0.0, 0.0)

    def test_wrong_shift_fails(self) -> None:
        """A wrong shift is rejected.

        This is the whole point of validation: a cross-correlation always returns a peak, and accepting
        it unchecked propagates a wrong WCS through the rest of the collect.
        """
        dx, dy = 20, -15
        source, target, cat = self._pair(dx, dy)
        # Propose a badly wrong shift; the random trials around the (now empty)
        # proposed position should beat it / it should not clear the threshold.
        valid, _corr, _, _ = validate_shift_lightweight(target, source, dx + 35, dy - 35, cat)
        assert not bool(valid)

    def test_insufficient_stars_rejected(self) -> None:
        # Only 3 stars in frame -> < 4 valid stars -> immediate reject.
        """Too few stars to judge with is a rejection, not a pass by default."""
        dx, dy = 10, 10
        source, target, cat = self._pair(dx, dy, n_stars=3, seed=9)
        valid, corr, _streak, correction = validate_shift_lightweight(target, source, dx, dy, cat)
        assert not bool(valid)
        assert corr == 0.0
        assert correction == (0.0, 0.0)

    def test_validate_proposed_shift_delegates(self) -> None:
        """The public entry point delegates to the same validation."""
        dx, dy = 18, 12
        source, target, cat = self._pair(dx, dy)
        v1 = validate_proposed_shift(target, source, dx, dy, cat)
        v2 = validate_shift_lightweight(target, source, dx, dy, cat)
        # Same RNG seed (autouse fixture re-seeds each test) -> identical result.
        assert v1[0] == v2[0]

    def test_zero_shift_with_aligned_frames_validates(self) -> None:
        # Identical frames, zero proposed shift -> perfect alignment.
        """A zero shift between aligned frames validates, so the check is not biased against it."""
        positions = _star_field(n_stars=30, seed=4)
        img = _render(positions, dx=0, dy=0, seed=2)
        source, target = _frame(img.copy(), 0), _frame(img.copy(), 1)
        cat = _catalog(positions)
        valid, corr, _, _ = validate_shift_lightweight(target, source, 0, 0, cat)
        assert bool(valid)
        assert corr > 0.6

    def test_fwhm_exclusion_path_validates_true_shift(self) -> None:
        # A large shift plus fwhm_exclusion triggers the perpendicular-sampling
        # branch; the true shift must still validate.
        """The FWHM-based exclusion path still accepts a true shift."""
        dx, dy = 30, 0
        source, target, cat = self._pair(dx, dy)
        valid, _corr, _, _ = validate_shift_lightweight(target, source, dx, dy, cat, fwhm_exclusion=6.0)
        assert bool(valid)

    def test_returns_four_tuple(self) -> None:
        """Validation returns its full result tuple, which callers unpack."""
        dx, dy = 12, 8
        source, target, cat = self._pair(dx, dy)
        result = validate_shift_lightweight(target, source, dx, dy, cat)
        assert isinstance(result, tuple)
        assert len(result) == 4
