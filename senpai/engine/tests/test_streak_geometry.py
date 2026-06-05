"""Pure-geometry / measurement tests for the streak extraction and masking
modules.

These cover the small, deterministic helpers that do not require a full
pipeline run:

- ``extract_streak_from_metadata`` — turns track-rate + exposure + WCS PC
  matrix into a (length, rotation) StreakMeasurement.
- ``streak_fwhm_from_cutout`` / ``streak_length_from_cutout`` /
  ``refine_streak_len`` — 1D profile measurements on synthetic streak PSFs.
- ``mask_streak_region`` / ``is_valid_psf`` — bookkeeping helpers used by the
  robust extractor (not exercised by the existing ``test_masking.py``).
- ``mask_all_but_border`` / ``mask_border`` — trivial border helpers in
  ``masking.py`` that have no other coverage.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.streak.extraction import (
    extract_streak_from_metadata,
    is_valid_psf,
    mask_streak_region,
    refine_streak_len,
    streak_fwhm_from_cutout,
    streak_length_from_cutout,
)
from senpai.engine.detection.streak.masking import (
    map_cluster,
    mask_all_but_border,
    mask_border,
)
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.metadata import FrameMetadata


@pytest.fixture(scope="module", autouse=True)
def _config():
    initialize_config(CONFIG_DIR / "burr.yaml")
    get_config().plotting.debug = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _identity_wcs(plate_scale_deg: float = 1.0 / 3600.0) -> WCSModel:
    """A WCS whose PC matrix maps RA/Dec axes straight onto x/y pixel axes."""
    return WCSModel(
        WCSAXES=2,
        NAXIS1=512,
        NAXIS2=512,
        CRPIX1=256.0,
        CRPIX2=256.0,
        PC1_1=1.0,
        PC1_2=0.0,
        PC2_1=0.0,
        PC2_2=1.0,
        CDELT1=plate_scale_deg,
        CDELT2=plate_scale_deg,
        CUNIT1="deg",
        CUNIT2="deg",
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
        CRVAL1=10.0,
        CRVAL2=20.0,
    )


def _frame_metadata(ra_rate: float, dec_rate: float, exposure: float) -> FrameMetadata:
    return FrameMetadata(
        exposure_time_seconds=exposure,
        observation_time=datetime(2024, 1, 1, 0, 0, 0),
        track_rate_ra_arcsec_per_second=ra_rate,
        track_rate_dec_arcsec_per_second=dec_rate,
    )


def _horizontal_streak(length: int, fwhm: float = 8.0, n: int = 160) -> np.ndarray:
    """A flat horizontal trail with a Gaussian cross-section, peak = 1.0."""
    cy = n // 2
    sigma = fwhm / 2.355
    x0 = (n - length) // 2
    cross = np.exp(-0.5 * ((np.arange(n) - cy) / sigma) ** 2)[:, None]
    img = np.zeros((n, n))
    img[:, x0:x0 + length] = cross
    return img


# --------------------------------------------------------------------------- #
# extract_streak_from_metadata
# --------------------------------------------------------------------------- #
class TestExtractStreakFromMetadata:
    def test_length_is_rate_times_exposure_over_plate_scale(self):
        # 2 arcsec/s in RA only, 10 s exposure, 1 arcsec/pixel -> 20 px streak.
        meta = _frame_metadata(ra_rate=2.0, dec_rate=0.0, exposure=10.0)
        m = extract_streak_from_metadata(meta, plate_scale_arcsec=1.0, wcs_model=_identity_wcs())
        assert m is not None
        assert m.length == pytest.approx(20.0, abs=1e-6)

    def test_combines_ra_and_dec_rates_in_quadrature(self):
        # 3,4 -> hypotenuse 5 arcsec/s * 4 s / 0.5 arcsec/px = 40 px.
        meta = _frame_metadata(ra_rate=3.0, dec_rate=4.0, exposure=4.0)
        m = extract_streak_from_metadata(meta, plate_scale_arcsec=0.5, wcs_model=_identity_wcs())
        assert m.length == pytest.approx(40.0, abs=1e-6)

    def test_rotation_for_pure_ra_rate_is_zero(self):
        meta = _frame_metadata(ra_rate=5.0, dec_rate=0.0, exposure=2.0)
        m = extract_streak_from_metadata(meta, plate_scale_arcsec=1.0, wcs_model=_identity_wcs())
        # dx=5, dy=0 -> arctan2(0, 5) = 0 deg.
        assert m.rotation == pytest.approx(0.0, abs=1e-6)

    def test_rotation_for_pure_dec_rate_is_ninety(self):
        meta = _frame_metadata(ra_rate=0.0, dec_rate=5.0, exposure=2.0)
        m = extract_streak_from_metadata(meta, plate_scale_arcsec=1.0, wcs_model=_identity_wcs())
        # dx=0, dy=5 -> arctan2(5, 0) = 90 deg.
        assert m.rotation == pytest.approx(90.0, abs=1e-6)

    def test_rotation_normalized_into_zero_to_180(self):
        # Negative dec rate would give -45 deg; must be folded into [0, 180).
        meta = _frame_metadata(ra_rate=5.0, dec_rate=-5.0, exposure=2.0)
        m = extract_streak_from_metadata(meta, plate_scale_arcsec=1.0, wcs_model=_identity_wcs())
        assert 0.0 <= m.rotation < 180.0
        assert m.rotation == pytest.approx(135.0, abs=1e-6)

    def test_returns_none_without_track_rates(self):
        meta = _frame_metadata(ra_rate=None, dec_rate=None, exposure=10.0)
        assert extract_streak_from_metadata(meta, 1.0, _identity_wcs()) is None

    def test_returns_none_without_exposure(self):
        # Both rates set but exposure 0 -> falsy -> None.
        meta = _frame_metadata(ra_rate=2.0, dec_rate=0.0, exposure=0.0)
        assert extract_streak_from_metadata(meta, 1.0, _identity_wcs()) is None

    def test_fwhm_is_none(self):
        meta = _frame_metadata(ra_rate=2.0, dec_rate=0.0, exposure=10.0)
        m = extract_streak_from_metadata(meta, 1.0, _identity_wcs())
        assert m.fwhm is None


# --------------------------------------------------------------------------- #
# streak_fwhm_from_cutout
# --------------------------------------------------------------------------- #
class TestStreakFWHMFromCutout:
    def test_measures_cross_section_fwhm(self):
        psf = _horizontal_streak(length=60, fwhm=8.0)
        fwhm = streak_fwhm_from_cutout(psf, rotation=0.0)
        assert fwhm == pytest.approx(8.0, abs=2.0)

    def test_wider_psf_gives_larger_fwhm(self):
        narrow = streak_fwhm_from_cutout(_horizontal_streak(60, fwhm=5.0), 0.0)
        wide = streak_fwhm_from_cutout(_horizontal_streak(60, fwhm=12.0), 0.0)
        assert wide > narrow

    def test_returns_float(self):
        fwhm = streak_fwhm_from_cutout(_horizontal_streak(40, fwhm=6.0), 0.0)
        assert isinstance(fwhm, float)


# --------------------------------------------------------------------------- #
# streak_length_from_cutout
# --------------------------------------------------------------------------- #
class TestStreakLengthFromCutout:
    def test_measures_length_of_clean_streak(self):
        psf = _horizontal_streak(length=50, fwhm=8.0)
        length = streak_length_from_cutout(psf, plot=False)
        assert length == pytest.approx(50.0, abs=8.0)

    def test_longer_streak_measures_longer(self):
        short = streak_length_from_cutout(_horizontal_streak(30, fwhm=8.0), plot=False)
        long = streak_length_from_cutout(_horizontal_streak(80, fwhm=8.0), plot=False)
        assert long > short


# --------------------------------------------------------------------------- #
# refine_streak_len
# --------------------------------------------------------------------------- #
class TestRefineStreakLen:
    def test_recovers_length_of_horizontal_streak(self):
        psf = _horizontal_streak(length=50, fwhm=8.0)
        length = refine_streak_len(psf, pixel_fwhm=8.0, rotation=0.0)
        assert length == pytest.approx(50.0, abs=8.0)

    def test_longer_streak_refines_longer(self):
        short = refine_streak_len(_horizontal_streak(30, fwhm=8.0), 8.0, 0.0)
        long = refine_streak_len(_horizontal_streak(70, fwhm=8.0), 8.0, 0.0)
        assert long > short

    def test_measures_when_fwhm_not_provided(self):
        psf = _horizontal_streak(length=50, fwhm=8.0)
        length = refine_streak_len(psf, pixel_fwhm=None, rotation=0.0)
        assert length == pytest.approx(50.0, abs=10.0)


# --------------------------------------------------------------------------- #
# mask_streak_region
# --------------------------------------------------------------------------- #
class TestMaskStreakRegion:
    def test_marks_processed_region_and_fills_data(self):
        working = np.full((80, 80), 100.0)
        working[40, 40] = 5000.0
        processed = np.zeros((80, 80), dtype=bool)
        # Small box kernel as the "detection kernel".
        kernel = np.zeros((9, 9))
        kernel[3:6, 3:6] = 1.0

        out_mask, out_data = mask_streak_region(processed, working, 40, 40, kernel)

        assert out_mask[40, 40]
        assert bool(np.any(out_mask))
        # The bright peak was filled toward the background median.
        assert out_data[40, 40] < 5000.0

    def test_leaves_far_region_untouched(self):
        working = np.full((80, 80), 100.0)
        processed = np.zeros((80, 80), dtype=bool)
        kernel = np.zeros((9, 9))
        kernel[3:6, 3:6] = 1.0

        out_mask, _ = mask_streak_region(processed, working, 40, 40, kernel)
        # A corner far from (40,40) stays unmasked.
        assert not out_mask[0, 0]

    def test_handles_point_near_border(self):
        working = np.full((50, 50), 100.0)
        processed = np.zeros((50, 50), dtype=bool)
        kernel = np.zeros((9, 9))
        kernel[3:6, 3:6] = 1.0

        # Should clamp indices and not raise for a near-edge point.
        out_mask, _ = mask_streak_region(processed, working, 1, 1, kernel)
        assert out_mask.shape == (50, 50)


# --------------------------------------------------------------------------- #
# is_valid_psf
# --------------------------------------------------------------------------- #
class TestIsValidPSF:
    def test_valid_when_no_overlap(self):
        processed = np.zeros((200, 200), dtype=bool)
        cutout = np.zeros((40, 40))
        assert bool(is_valid_psf(cutout, processed, 100, 100, cutout_size=20))

    def test_invalid_when_heavy_overlap(self):
        processed = np.zeros((200, 200), dtype=bool)
        # Mark the whole region around (100, 100) as already processed.
        processed[80:120, 80:120] = True
        cutout = np.zeros((40, 40))
        assert not bool(is_valid_psf(cutout, processed, 100, 100, cutout_size=20))

    def test_small_overlap_still_valid(self):
        processed = np.zeros((200, 200), dtype=bool)
        # Mark only a tiny corner (<10% of the 40x40 cutout region).
        processed[81:83, 81:83] = True
        cutout = np.zeros((40, 40))
        assert bool(is_valid_psf(cutout, processed, 100, 100, cutout_size=20))


# --------------------------------------------------------------------------- #
# mask_all_but_border / mask_border  (untested helpers in masking.py)
# --------------------------------------------------------------------------- #
class TestBorderMasks:
    def test_mask_all_but_border_zeros_interior(self):
        img = np.ones((10, 10))
        out = mask_all_but_border(img, n_pixels=2)
        # Interior zeroed, border preserved.
        assert np.all(out[2:-2, 2:-2] == 0.0)
        assert np.all(out[:2, :] == 1.0)
        assert np.all(out[-2:, :] == 1.0)

    def test_mask_all_but_border_does_not_mutate_input(self):
        img = np.ones((10, 10))
        mask_all_but_border(img, n_pixels=1)
        assert np.all(img == 1.0)

    def test_mask_border_zeros_edges_keeps_interior(self):
        img = np.ones((10, 10))
        out = mask_border(img, n_pixels=2)
        assert np.all(out[2:-2, 2:-2] == 1.0)
        assert np.all(out[0:2, :] == 0.0)
        assert np.all(out[:, 0:2] == 0.0)
        assert np.all(out[-2:, :] == 0.0)
        assert np.all(out[:, -2:] == 0.0)

    def test_mask_border_does_not_mutate_input(self):
        img = np.ones((8, 8))
        mask_border(img, n_pixels=1)
        assert np.all(img == 1.0)


# --------------------------------------------------------------------------- #
# map_cluster (flood-fill) basic behavior — complements test_masking.py which
# only covers the bounded variant via remove_streak_at_point.
# --------------------------------------------------------------------------- #
class TestMapCluster:
    def test_fills_connected_blob_above_threshold(self):
        img = np.zeros((30, 30))
        img[10:15, 10:15] = 100.0
        mask = map_cluster(img, (12, 12), flux_threshold=50.0)
        assert mask[12, 12]
        assert int(np.sum(mask)) == 25  # the full 5x5 blob

    def test_does_not_cross_below_threshold_gap(self):
        img = np.zeros((30, 30))
        img[10:13, 10:13] = 100.0
        img[10:13, 20:23] = 100.0  # disconnected second blob
        mask = map_cluster(img, (11, 11), flux_threshold=50.0)
        # Only the first blob is filled, not the second.
        assert mask[11, 11]
        assert not mask[11, 21]

    def test_out_of_bounds_start_returns_empty(self):
        img = np.zeros((10, 10))
        mask = map_cluster(img, (20, 20), flux_threshold=1.0)
        assert not np.any(mask)
