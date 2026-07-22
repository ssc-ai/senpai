"""Behavioral tests for master dark/flat matching and application.

Covers:
- senpai.engine.utils.darks: apply_dark_subtraction (exposure scaling, hot-pixel
  cleaning, shape checks), find_best_dark_for_exposure, _group_frames_by_headers.
- senpai.engine.utils.flats: apply_flat_field (division, division-by-zero guard).
- preprocessing._find_master_calibration / _find_best_dark_calibration: header
  matching, rejection, and exposure-ratio gating driven by CalibrationsConfig.

Synthetic master darks/flats are written as FITS files into tmp_path with
crafted headers; matching/rejection is asserted by which file is selected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.images import ProcessedFitsImage, ProcessingStep
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.utils.darks import (
    _group_frames_by_headers,
    apply_dark_subtraction,
    find_best_dark_for_exposure,
)
from senpai.engine.utils.flats import apply_flat_field
from senpai.engine.utils.preprocessing import (
    _find_best_dark_calibration,
    _find_master_calibration,
)


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialise the process-wide config singleton with debug plotting off."""
    initialize_config(CONFIG_DIR / "burr.yaml")
    get_config().plotting.debug = False


def _write_fits(path: Path, data: np.ndarray, **header_kw: float | int | str) -> Path:
    """Write ``data`` to a FITS file at ``path`` with the given header cards.

    Args:
        path: Destination FITS path.
        data: Image data (cast to float32).
        **header_kw: FITS header cards to set.

    Returns:
        The path written to.
    """
    hdr = fits.Header()
    for key, value in header_kw.items():
        hdr[key] = value
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(path, overwrite=True)
    return path


def _make_image(data: np.ndarray, **header_kw: float | int | str) -> ProcessedFitsImage:
    """Build a :class:`ProcessedFitsImage` from ``data`` with header cards.

    Args:
        data: Image data (cast to float64).
        **header_kw: FITS header cards to set on the science frame.

    Returns:
        The constructed :class:`ProcessedFitsImage`.
    """
    h, w = data.shape
    hdr = fits.Header()
    hdr["BITPIX"] = -64
    for key, value in header_kw.items():
        hdr[key] = value
    return ProcessedFitsImage(
        data=data.astype(np.float64),
        header=hdr,
        data_type=data.dtype,
        metadata=ImageMetadata(width=w, height=h),
        file_path="science.fits",
        correction_frames={},
    )


# --- apply_dark_subtraction ---------------------------------------------------


class TestApplyDarkSubtraction:
    """Dark-frame subtraction with exposure scaling and hot-pixel cleaning."""

    def test_subtracts_matching_exposure(self) -> None:
        """A same-exposure dark is subtracted without scaling."""
        dark = np.full((20, 20), 100.0)
        image = _make_image(np.full((20, 20), 500.0), EXPTIME=10.0)
        out = apply_dark_subtraction(image, dark, dark_exposure_time=10.0)
        # No scaling -> straight subtraction.
        assert np.allclose(out.data, 400.0)

    def test_scales_dark_for_longer_exposure(self) -> None:
        """A dark shorter than the science frame is scaled up before subtraction."""
        dark = np.full((20, 20), 100.0)
        # image exposure is 2x the dark -> dark scaled by 2 -> subtract 200.
        image = _make_image(np.full((20, 20), 1000.0), EXPTIME=20.0)
        out = apply_dark_subtraction(image, dark, dark_exposure_time=10.0)
        assert np.allclose(out.data, 800.0)
        meta = next(m for m in out.processing_history if m.step_type == ProcessingStep.DARK_SUBTRACT)
        assert meta.parameters["exposure_time_scaling"] == pytest.approx(2.0)

    def test_scales_dark_for_shorter_exposure(self) -> None:
        """A dark longer than the science frame is scaled down before subtraction."""
        dark = np.full((20, 20), 100.0)
        image = _make_image(np.full((20, 20), 1000.0), EXPTIME=5.0)
        out = apply_dark_subtraction(image, dark, dark_exposure_time=10.0)
        # dark scaled by 0.5 -> subtract 50.
        assert np.allclose(out.data, 950.0)

    def test_hot_pixels_cleaned_before_subtraction(self) -> None:
        """Hot pixels in the dark are replaced by the median before subtraction."""
        dark = np.full((30, 30), 100.0)
        dark[5, 5] = 50000.0  # single hot pixel far above median+5*std
        image = _make_image(np.full((30, 30), 500.0), EXPTIME=10.0)
        out = apply_dark_subtraction(image, dark, dark_exposure_time=10.0)
        # The hot pixel is replaced by the dark median (100) before subtraction,
        # so that pixel is 500-100=400, not a huge negative number.
        assert out.data[5, 5] == pytest.approx(400.0)
        meta = next(m for m in out.processing_history if m.step_type == ProcessingStep.DARK_SUBTRACT)
        assert meta.parameters["hot_pixels_cleaned"] >= 1

    def test_reads_dark_exposure_from_file_header(self, tmp_path: Path) -> None:
        """When given a dark file, its EXPTIME header drives the scaling."""
        dpath = _write_fits(tmp_path / "dark.fits", np.full((16, 16), 80.0), EXPTIME=4.0)
        image = _make_image(np.full((16, 16), 800.0), EXPTIME=8.0)
        out = apply_dark_subtraction(image, dpath)
        # dark scaled by 8/4 = 2 -> subtract 160.
        assert np.allclose(out.data, 640.0)

    def test_shape_mismatch_raises(self) -> None:
        """A dark whose shape differs from the image raises ValueError."""
        dark = np.full((10, 10), 100.0)
        image = _make_image(np.full((20, 20), 500.0), EXPTIME=10.0)
        with pytest.raises(ValueError):
            apply_dark_subtraction(image, dark, dark_exposure_time=10.0)

    def test_numpy_array_input_no_scaling(self) -> None:
        """A raw ndarray image is subtracted without scaling and returned as ndarray."""
        dark = np.full((8, 8), 30.0)
        image = np.full((8, 8), 200.0)
        out = apply_dark_subtraction(image, dark)
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, 170.0)


# --- apply_flat_field ---------------------------------------------------------


class TestApplyFlatField:
    """Flat-field division with a division-by-zero guard."""

    def test_divides_by_flat(self) -> None:
        """The image is divided by the flat and the step is recorded."""
        flat = np.full((20, 20), 2.0)
        image = _make_image(np.full((20, 20), 1000.0))
        out = apply_flat_field(image, flat)
        assert np.allclose(out.data, 500.0)
        assert any(m.step_type == ProcessingStep.FLAT_DIVIDE for m in out.processing_history)

    def test_low_flat_values_guarded(self) -> None:
        """Flat pixels below the guard are treated as 1.0 to avoid blow-up."""
        # Pixels below 0.1 in the flat are treated as 1.0 to avoid blow-up.
        flat = np.ones((10, 10))
        flat[0, 0] = 0.0
        image = _make_image(np.full((10, 10), 300.0))
        out = apply_flat_field(image, flat)
        assert out.data[0, 0] == pytest.approx(300.0)
        assert np.isfinite(out.data).all()

    def test_normalized_flat_corrects_vignette(self) -> None:
        """A normalized flat (<1 at edges) brightens vignetted regions."""
        # A normalized flat (<1 at edges) brightens vignetted regions.
        flat = np.ones((20, 20))
        flat[:, :5] = 0.5  # left columns receive half the light
        image = _make_image(np.full((20, 20), 400.0))
        out = apply_flat_field(image, flat)
        assert np.allclose(out.data[:, :5], 800.0)
        assert np.allclose(out.data[:, 5:], 400.0)

    def test_loads_flat_from_file(self, tmp_path: Path) -> None:
        """A flat given by file path is loaded and divided out."""
        fpath = _write_fits(tmp_path / "flat.fits", np.full((12, 12), 4.0))
        image = _make_image(np.full((12, 12), 1000.0))
        out = apply_flat_field(image, str(fpath))
        assert np.allclose(out.data, 250.0)

    def test_shape_mismatch_raises(self) -> None:
        """A flat whose shape differs from the image raises ValueError."""
        flat = np.ones((10, 10))
        image = _make_image(np.full((20, 20), 500.0))
        with pytest.raises(ValueError):
            apply_flat_field(image, flat)


# --- find_best_dark_for_exposure ---------------------------------------------


class TestFindBestDarkForExposure:
    """Selection of the closest-exposure dark from a directory."""

    def _populate(self, tmp_path: Path) -> None:
        """Write 5s/10s/30s master darks into ``tmp_path``.

        Args:
            tmp_path: Directory to populate with dark FITS files.
        """
        data = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark_5s.fits", data, EXPTIME=5.0, BINNING="1x1")
        _write_fits(tmp_path / "dark_10s.fits", data, EXPTIME=10.0, BINNING="1x1")
        _write_fits(tmp_path / "dark_30s.fits", data, EXPTIME=30.0, BINNING="1x1")

    def test_picks_closest_exposure(self, tmp_path: Path) -> None:
        """The dark nearest the target exposure is selected."""
        self._populate(tmp_path)
        result = find_best_dark_for_exposure(tmp_path, target_exptime=12.0, matching_headers=[])
        assert result is not None
        _, exptime = result
        assert exptime == 10.0

    def test_rejects_when_ratio_too_high(self, tmp_path: Path) -> None:
        """A dark whose exposure ratio exceeds the maximum is rejected."""
        # Only a 5s dark; target 100s -> ratio 20 > max 3 -> no match.
        _write_fits(tmp_path / "dark_5s.fits", np.full((8, 8), 100.0), EXPTIME=5.0)
        result = find_best_dark_for_exposure(
            tmp_path, target_exptime=100.0, matching_headers=[], max_exptime_ratio=3.0
        )
        assert result is None

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        """A nonexistent dark directory returns None."""
        assert find_best_dark_for_exposure(tmp_path / "nope", target_exptime=10.0) is None


# --- _group_frames_by_headers -------------------------------------------------


class TestGroupFramesByHeaders:
    """Grouping of frames by selected header values."""

    def test_groups_by_binning_and_exptime(self, tmp_path: Path) -> None:
        """Frames are grouped by the (binning, exptime) header tuple."""
        d = np.zeros((4, 4))
        f1 = _write_fits(tmp_path / "a.fits", d, BINNING="1x1", EXPTIME=10.0)
        f2 = _write_fits(tmp_path / "b.fits", d, BINNING="1x1", EXPTIME=10.0)
        f3 = _write_fits(tmp_path / "c.fits", d, BINNING="2x2", EXPTIME=10.0)
        groups = _group_frames_by_headers([f1, f2, f3], ["BINNING", "EXPTIME"])
        assert len(groups) == 2
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2]

    def test_exptime_rounded_for_grouping(self, tmp_path: Path) -> None:
        """Near-equal exposure times round together into one group."""
        d = np.zeros((4, 4))
        f1 = _write_fits(tmp_path / "a.fits", d, EXPTIME=10.001)
        f2 = _write_fits(tmp_path / "b.fits", d, EXPTIME=10.004)
        groups = _group_frames_by_headers([f1, f2], ["EXPTIME"])
        # Both round to 10.0 -> single group.
        assert len(groups) == 1

    def test_empty_headers_single_group(self, tmp_path: Path) -> None:
        """With no grouping headers, all frames land in one group."""
        f1 = _write_fits(tmp_path / "a.fits", np.zeros((4, 4)))
        groups = _group_frames_by_headers([f1], [])
        assert len(groups) == 1


# --- _find_master_calibration (flat-style exact matching) --------------------


class TestFindMasterCalibration:
    """Exact header matching for master calibration frames (flat-style)."""

    def test_matches_on_binning_and_filter(self, tmp_path: Path) -> None:
        """A master is matched by binning and filter to the science frame."""
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat_V.fits", d, XBINNING=1, FILTER="V")
        _write_fits(tmp_path / "flat_R.fits", d, XBINNING=1, FILTER="R")
        image = _make_image(d, XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is not None
        assert match.name == "flat_V.fits"

    def test_filter_case_insensitive(self, tmp_path: Path) -> None:
        """Filter matching is case-insensitive."""
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat.fits", d, XBINNING=1, FILTER="v")
        image = _make_image(d, XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is not None

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        """A master whose headers differ from the science frame is not matched."""
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat.fits", d, XBINNING=2, FILTER="V")
        image = _make_image(d, XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is None

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        """A nonexistent calibration directory returns None."""
        image = _make_image(np.ones((8, 8)), XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path / "absent"), ["XBINNING"], "flat")
        assert match is None

    def test_missing_required_header_in_science_returns_none(self, tmp_path: Path) -> None:
        """A science frame missing a required header cannot match, returning None."""
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat.fits", d, XBINNING=1, FILTER="V")
        image = _make_image(d, XBINNING=1)  # no FILTER on science frame
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is None


# --- _find_best_dark_calibration (header match + exposure ratio gating) ------


class TestFindBestDarkCalibration:
    """Dark selection combining header matching and exposure-ratio gating."""

    def test_selects_closest_exposure_among_matching(self, tmp_path: Path) -> None:
        """Among header-matched darks, the closest exposure is selected."""
        d = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark_9s.fits", d, XBINNING=1, EXPTIME=9.0)
        _write_fits(tmp_path / "dark_20s.fits", d, XBINNING=1, EXPTIME=20.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is not None
        assert match.name == "dark_9s.fits"

    def test_rejects_dark_with_wrong_binning(self, tmp_path: Path) -> None:
        """A dark whose binning differs from the science frame is rejected."""
        d = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark.fits", d, XBINNING=2, EXPTIME=10.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is None

    def test_exposure_ratio_within_limit_accepted(self, tmp_path: Path) -> None:
        """A dark at exactly the maximum exposure ratio is accepted."""
        d = np.full((8, 8), 100.0)
        # ratio 30/10 = 3.0 == max -> accepted (<=).
        _write_fits(tmp_path / "dark.fits", d, XBINNING=1, EXPTIME=30.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is not None

    def test_exposure_ratio_beyond_limit_rejected(self, tmp_path: Path) -> None:
        """A dark beyond the maximum exposure ratio is rejected."""
        d = np.full((8, 8), 100.0)
        # ratio 40/10 = 4.0 > max 3.0 -> rejected.
        _write_fits(tmp_path / "dark.fits", d, XBINNING=1, EXPTIME=40.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is None

    def test_falls_back_to_exact_match_without_image_exptime(self, tmp_path: Path) -> None:
        """Without a science EXPTIME, selection falls back to exact header matching."""
        d = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark.fits", d, XBINNING=1, EXPTIME=10.0)
        image = _make_image(d, XBINNING=1)  # science frame has no EXPTIME
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        # Falls back to exact-header matching; XBINNING matches -> found.
        assert match is not None


# --- end-to-end: best dark selection then subtraction ------------------------


def test_dark_selection_then_apply_scales_correctly(tmp_path: Path) -> None:
    """The exposure-ratio-selected dark is scaled correctly when applied.

    The exposure-ratio-selected dark, when applied, is scaled by the
    image/dark exposure ratio.
    """
    dark_data = np.full((16, 16), 50.0)
    _write_fits(tmp_path / "dark_5s.fits", dark_data, XBINNING=1, EXPTIME=5.0)
    image = _make_image(np.full((16, 16), 600.0), XBINNING=1, EXPTIME=10.0)

    best = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
    assert best is not None and best.name == "dark_5s.fits"

    out = apply_dark_subtraction(image, best)
    # dark (50) scaled by 10/5 = 2 -> subtract 100 -> 500.
    assert np.allclose(out.data, 500.0)
