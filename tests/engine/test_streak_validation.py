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

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.streak.validation_extra import (
    extract_box_statistics,
    quick_correlation_from_boxes,
    validate_proposed_shift,
    validate_shift_lightweight,
)
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.senpai import RateTrackFrame
from senpai.engine.models.starfield import StarInSpace


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialise the process-wide config singleton and disable debug plots."""
    initialize_config(CONFIG_DIR / "burr.yaml")
    # Validation has debug-plot paths gated on config.plotting.debug; keep off.
    get_config().plotting.debug = False


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed the global numpy RNG so random alternative shifts are deterministic."""
    # Validation draws random alternative shifts via the global numpy RNG.
    np.random.seed(1234)


# --------------------------------------------------------------------------- #
# Synthetic-frame helpers
# --------------------------------------------------------------------------- #
IMG = 512
BG = 100.0
PEAK = 4000.0
FWHM = 3.0


def _add_star(img: np.ndarray, x: float, y: float, peak: float, fwhm: float = FWHM) -> None:
    """Add a Gaussian star of the given peak and FWHM to ``img`` in place.

    Args:
        img: 2D image to modify in place.
        x: Sub-pixel x position of the star center.
        y: Sub-pixel y position of the star center.
        peak: Peak amplitude of the Gaussian.
        fwhm: Full width at half maximum of the star in pixels.
    """
    sigma = fwhm / 2.355
    half = int(np.ceil(4 * sigma))
    xi, yi = round(x), round(y)
    y0, y1 = max(0, yi - half), min(img.shape[0], yi + half + 1)
    x0, x1 = max(0, xi - half), min(img.shape[1], xi + half + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += peak * np.exp(-0.5 * (((xx - x) ** 2 + (yy - y) ** 2) / sigma**2))


def _star_field(
    n_stars: int = 30, seed: int = 0, margin: int = 60
) -> list[tuple[float, float, float]]:
    """Return (positions, peaks) for n_stars with a spread of brightnesses.

    Args:
        n_stars: Number of stars to generate.
        seed: Seed for the deterministic RNG.
        margin: Minimum distance from the frame edge in pixels.

    Returns:
        A list of ``(x, y, peak)`` tuples, one per star.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(margin, IMG - margin, n_stars)
    ys = rng.uniform(margin, IMG - margin, n_stars)
    # A wide spread of peaks so the flux ordering is informative for correlation.
    peaks = rng.uniform(0.2 * PEAK, PEAK, n_stars)
    return list(zip(xs, ys, peaks, strict=True))


def _render(
    positions: list[tuple[float, float, float]],
    dx: float = 0.0,
    dy: float = 0.0,
    noise: float = 3.0,
    seed: int = 7,
) -> np.ndarray:
    """Render a frame; each source is placed at (x - dx, y - dy).

    Args:
        positions: ``(x, y, peak)`` source tuples in source-frame coordinates.
        dx: Shift in x applied to every source before rendering.
        dy: Shift in y applied to every source before rendering.
        noise: Standard deviation of the additive Gaussian read noise.
        seed: Seed for the deterministic RNG.

    Returns:
        The rendered frame as a ``float32`` array.
    """
    rng = np.random.default_rng(seed)
    img = np.full((IMG, IMG), BG, dtype=np.float64)
    img += rng.normal(0.0, noise, img.shape)
    for x, y, peak in positions:
        _add_star(img, x - dx, y - dy, peak)
    return img.astype(np.float32)


def _frame(data: np.ndarray, index: int = 0) -> RateTrackFrame:
    """Wrap frame data in a :class:`RateTrackFrame` with minimal metadata.

    Args:
        data: The frame pixel data.
        index: The frame's index within its collection.

    Returns:
        The constructed :class:`RateTrackFrame`.
    """
    img = ProcessedFitsImage(
        data=data,
        header=fits.Header(),
        data_type=np.dtype("uint16"),
        metadata=ImageMetadata(width=IMG, height=IMG),
    )
    return RateTrackFrame(frame=img, index=index, timestamp=datetime(2024, 1, 1))


def _catalog(positions: list[tuple[float, float, float]]) -> list[StarInSpace]:
    """Catalog stars at the SOURCE-frame positions (x, y), with magnitudes.

    Args:
        positions: ``(x, y, peak)`` source tuples; peak sets the magnitude.

    Returns:
        A list of :class:`StarInSpace` catalog entries.
    """
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
    """Tests for ``extract_box_statistics`` box-statistics extraction."""

    def test_returns_valid_stats_for_interior_point(self) -> None:
        """An interior point returns valid max/median/sum box statistics."""
        img = np.full((50, 50), 10.0)
        img[25, 25] = 1000.0
        stats = extract_box_statistics(img, 25, 25, box_size=11)
        assert stats["valid"] is True
        assert stats["max"] == 1000.0
        assert stats["median"] == 10.0
        assert stats["sum"] == pytest.approx(10.0 * (121 - 1) + 1000.0)

    def test_invalid_when_box_out_of_bounds(self) -> None:
        """A box running past the near edge is flagged invalid."""
        img = np.full((50, 50), 10.0)
        stats = extract_box_statistics(img, 2, 2, box_size=11)
        assert stats["valid"] is False
        assert stats["max"] == 0.0

    def test_invalid_near_far_edge(self) -> None:
        """A box running past the far edge is flagged invalid."""
        img = np.full((50, 50), 10.0)
        stats = extract_box_statistics(img, 48, 48, box_size=11)
        assert stats["valid"] is False

    def test_box_centered_on_rounded_position(self) -> None:
        """The box centers on the rounded position and captures the peak."""
        img = np.zeros((50, 50))
        img[30, 20] = 500.0
        # x=20.3, y=29.6 rounds to (20, 30); box should capture the peak.
        stats = extract_box_statistics(img, 20.3, 29.6, box_size=5)
        assert stats["max"] == 500.0


# --------------------------------------------------------------------------- #
# quick_correlation_from_boxes
# --------------------------------------------------------------------------- #
class TestQuickCorrelation:
    """Tests for ``quick_correlation_from_boxes`` net-flux correlation."""

    def test_true_shift_correlates_strongly(self) -> None:
        """The true shift produces a strong per-star flux correlation."""
        positions = _star_field(n_stars=30, seed=1)
        dx, dy = 20, -15
        source = _render(positions, dx=0, dy=0, seed=2)
        target = _render(positions, dx=dx, dy=dy, seed=3)
        cat = _catalog(positions)

        corr, n, _ = quick_correlation_from_boxes(target, source, dx, dy, cat, box_size=11, max_stars=50)
        assert n >= 4
        assert corr > 0.6

    def test_wrong_shift_correlates_poorly(self) -> None:
        """A wrong shift correlates worse than the true shift."""
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
        """Fewer than three valid stars yields zero correlation and no pairs."""
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
        """The correlation uses no more stars than the max_stars limit."""
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
    """Tests for ``validate_shift_lightweight`` / ``validate_proposed_shift``."""

    def _pair(
        self, dx: int, dy: int, n_stars: int = 30, seed: int = 1, noise: float = 3.0
    ) -> tuple[RateTrackFrame, RateTrackFrame, list[StarInSpace]]:
        """Build a source/target frame pair offset by a known integer shift.

        Args:
            dx: True shift in x between the frames, in pixels.
            dy: True shift in y between the frames, in pixels.
            n_stars: Number of stars to plant in the field.
            seed: Seed for the deterministic star field.
            noise: Standard deviation of the additive read noise.

        Returns:
            A tuple ``(source_frame, target_frame, catalog)``.
        """
        positions = _star_field(n_stars=n_stars, seed=seed)
        source_img = _render(positions, dx=0, dy=0, noise=noise, seed=2)
        target_img = _render(positions, dx=dx, dy=dy, noise=noise, seed=3)
        return _frame(source_img, 0), _frame(target_img, 1), _catalog(positions)

    def test_true_shift_validates(self) -> None:
        """The true shift validates with a high correlation and no correction."""
        dx, dy = 20, -15
        source, target, cat = self._pair(dx, dy)
        valid, corr, streak, correction = validate_shift_lightweight(target, source, dx, dy, cat)
        assert bool(valid)
        assert corr > 0.6
        assert streak is None
        assert correction == (0.0, 0.0)

    def test_wrong_shift_fails(self) -> None:
        """A badly wrong proposed shift fails validation."""
        dx, dy = 20, -15
        source, target, cat = self._pair(dx, dy)
        # Propose a badly wrong shift; the random trials around the (now empty)
        # proposed position should beat it / it should not clear the threshold.
        valid, _corr, _, _ = validate_shift_lightweight(target, source, dx + 35, dy - 35, cat)
        assert not bool(valid)

    def test_insufficient_stars_rejected(self) -> None:
        """Too few valid stars rejects the shift immediately."""
        # Only 3 stars in frame -> < 4 valid stars -> immediate reject.
        dx, dy = 10, 10
        source, target, cat = self._pair(dx, dy, n_stars=3, seed=9)
        valid, corr, _streak, correction = validate_shift_lightweight(target, source, dx, dy, cat)
        assert not bool(valid)
        assert corr == 0.0
        assert correction == (0.0, 0.0)

    def test_validate_proposed_shift_delegates(self) -> None:
        """``validate_proposed_shift`` delegates to the lightweight validator."""
        dx, dy = 18, 12
        source, target, cat = self._pair(dx, dy)
        v1 = validate_proposed_shift(target, source, dx, dy, cat)
        v2 = validate_shift_lightweight(target, source, dx, dy, cat)
        # Same RNG seed (autouse fixture re-seeds each test) -> identical result.
        assert v1[0] == v2[0]

    def test_zero_shift_with_aligned_frames_validates(self) -> None:
        """Identical frames validate a zero proposed shift."""
        # Identical frames, zero proposed shift -> perfect alignment.
        positions = _star_field(n_stars=30, seed=4)
        img = _render(positions, dx=0, dy=0, seed=2)
        source, target = _frame(img.copy(), 0), _frame(img.copy(), 1)
        cat = _catalog(positions)
        valid, corr, _, _ = validate_shift_lightweight(target, source, 0, 0, cat)
        assert bool(valid)
        assert corr > 0.6

    def test_fwhm_exclusion_path_validates_true_shift(self) -> None:
        """The fwhm-exclusion sampling branch still validates the true shift."""
        # A large shift plus fwhm_exclusion triggers the perpendicular-sampling
        # branch; the true shift must still validate.
        dx, dy = 30, 0
        source, target, cat = self._pair(dx, dy)
        valid, _corr, _, _ = validate_shift_lightweight(target, source, dx, dy, cat, fwhm_exclusion=6.0)
        assert bool(valid)

    def test_returns_four_tuple(self) -> None:
        """The validator returns a four-element result tuple."""
        dx, dy = 12, 8
        source, target, cat = self._pair(dx, dy)
        result = validate_shift_lightweight(target, source, dx, dy, cat)
        assert isinstance(result, tuple)
        assert len(result) == 4
