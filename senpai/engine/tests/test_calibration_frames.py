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

from senpai.core.config import initialize_config, settings
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
    """Initialise the process-wide config, which the calibration helpers read."""
    initialize_config(CONFIG_DIR / "burr.yaml")
    settings.plotting.debug = False


def _write_fits(path: Path, data: np.ndarray, **header_kw: object) -> Path:
    """Write a FITS file with chosen header keywords, for the calibration search to find."""
    hdr = fits.Header()
    for key, value in header_kw.items():
        hdr[key] = value
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(path, overwrite=True)
    return path


def _make_image(data: np.ndarray, **header_kw: object) -> ProcessedFitsImage:
    """Wrap an array as a ProcessedFitsImage with chosen header keywords."""
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
    """Subtracting a master dark, including the exposure scaling."""

    def test_subtracts_matching_exposure(self) -> None:
        """A dark of the same exposure is subtracted as-is."""
        dark = np.full((20, 20), 100.0)
        image = _make_image(np.full((20, 20), 500.0), EXPTIME=10.0)
        out = apply_dark_subtraction(image, dark, dark_exposure_time=10.0)
        # No scaling -> straight subtraction.
        assert np.allclose(out.data, 400.0)

    def test_scales_dark_for_longer_exposure(self) -> None:
        """A shorter dark is scaled up for a longer exposure.

        Dark current accumulates with time, so the correction has to be scaled rather than reused.
        """
        dark = np.full((20, 20), 100.0)
        # image exposure is 2x the dark -> dark scaled by 2 -> subtract 200.
        image = _make_image(np.full((20, 20), 1000.0), EXPTIME=20.0)
        out = apply_dark_subtraction(image, dark, dark_exposure_time=10.0)
        assert np.allclose(out.data, 800.0)
        meta = next(m for m in out.processing_history if m.step_type == ProcessingStep.DARK_SUBTRACT)
        assert meta.parameters["exposure_time_scaling"] == pytest.approx(2.0)

    def test_scales_dark_for_shorter_exposure(self) -> None:
        """A longer dark is scaled down for a shorter exposure."""
        dark = np.full((20, 20), 100.0)
        image = _make_image(np.full((20, 20), 1000.0), EXPTIME=5.0)
        out = apply_dark_subtraction(image, dark, dark_exposure_time=10.0)
        # dark scaled by 0.5 -> subtract 50.
        assert np.allclose(out.data, 950.0)

    def test_hot_pixels_cleaned_before_subtraction(self) -> None:
        """Hot pixels are cleaned from the dark before it is used.

        Scaling a hot pixel by the exposure ratio would inject a brighter artefact than the one it was
        meant to remove.
        """
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
        """The dark's own exposure is read from its header rather than assumed."""
        dpath = _write_fits(tmp_path / "dark.fits", np.full((16, 16), 80.0), EXPTIME=4.0)
        image = _make_image(np.full((16, 16), 800.0), EXPTIME=8.0)
        out = apply_dark_subtraction(image, dpath)
        # dark scaled by 8/4 = 2 -> subtract 160.
        assert np.allclose(out.data, 640.0)

    def test_shape_mismatch_raises(self) -> None:
        """A dark of the wrong shape raises rather than being broadcast or cropped."""
        dark = np.full((10, 10), 100.0)
        image = _make_image(np.full((20, 20), 500.0), EXPTIME=10.0)
        with pytest.raises(ValueError):
            apply_dark_subtraction(image, dark, dark_exposure_time=10.0)

    def test_numpy_array_input_no_scaling(self) -> None:
        """A bare array carries no exposure, so it is subtracted without scaling."""
        dark = np.full((8, 8), 30.0)
        image = np.full((8, 8), 200.0)
        out = apply_dark_subtraction(image, dark)
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, 170.0)


# --- apply_flat_field ---------------------------------------------------------


class TestApplyFlatField:
    """Dividing by a master flat."""

    def test_divides_by_flat(self) -> None:
        """The frame is divided by the flat."""
        flat = np.full((20, 20), 2.0)
        image = _make_image(np.full((20, 20), 1000.0))
        out = apply_flat_field(image, flat)
        assert np.allclose(out.data, 500.0)
        assert any(m.step_type == ProcessingStep.FLAT_DIVIDE for m in out.processing_history)

    def test_low_flat_values_guarded(self) -> None:
        # Pixels below 0.1 in the flat are treated as 1.0 to avoid blow-up.
        """Near-zero flat values are guarded rather than dividing through.

        Dividing by a near-zero pixel turns a vignetted corner into a bright artefact.
        """
        flat = np.ones((10, 10))
        flat[0, 0] = 0.0
        image = _make_image(np.full((10, 10), 300.0))
        out = apply_flat_field(image, flat)
        assert out.data[0, 0] == pytest.approx(300.0)
        assert np.isfinite(out.data).all()

    def test_normalized_flat_corrects_vignette(self) -> None:
        # A normalized flat (<1 at edges) brightens vignetted regions.
        """A normalised flat removes a vignette without changing the overall level."""
        flat = np.ones((20, 20))
        flat[:, :5] = 0.5  # left columns receive half the light
        image = _make_image(np.full((20, 20), 400.0))
        out = apply_flat_field(image, flat)
        assert np.allclose(out.data[:, :5], 800.0)
        assert np.allclose(out.data[:, 5:], 400.0)

    def test_loads_flat_from_file(self, tmp_path: Path) -> None:
        """A flat given as a path is loaded from disk."""
        fpath = _write_fits(tmp_path / "flat.fits", np.full((12, 12), 4.0))
        image = _make_image(np.full((12, 12), 1000.0))
        out = apply_flat_field(image, str(fpath))
        assert np.allclose(out.data, 250.0)

    def test_shape_mismatch_raises(self) -> None:
        """A flat of the wrong shape raises."""
        flat = np.ones((10, 10))
        image = _make_image(np.full((20, 20), 500.0))
        with pytest.raises(ValueError):
            apply_flat_field(image, flat)


# --- find_best_dark_for_exposure ---------------------------------------------


class TestFindBestDarkForExposure:
    """Choosing the closest usable dark for a given exposure."""

    def _populate(self, tmp_path: Path) -> None:
        """Write a set of darks at 5, 10 and 30 seconds for the search to choose among."""
        data = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark_5s.fits", data, EXPTIME=5.0, BINNING="1x1")
        _write_fits(tmp_path / "dark_10s.fits", data, EXPTIME=10.0, BINNING="1x1")
        _write_fits(tmp_path / "dark_30s.fits", data, EXPTIME=30.0, BINNING="1x1")

    def test_picks_closest_exposure(self, tmp_path: Path) -> None:
        """The dark closest in exposure is chosen."""
        self._populate(tmp_path)
        result = find_best_dark_for_exposure(tmp_path, target_exptime=12.0, matching_headers=[])
        assert result is not None
        _, exptime = result
        assert exptime == 10.0

    def test_rejects_when_ratio_too_high(self, tmp_path: Path) -> None:
        # Only a 5s dark; target 100s -> ratio 20 > max 3 -> no match.
        """A dark too far off in exposure is rejected rather than scaled aggressively.

        Scaling by a large ratio amplifies the dark's own noise into the science frame.
        """
        _write_fits(tmp_path / "dark_5s.fits", np.full((8, 8), 100.0), EXPTIME=5.0)
        result = find_best_dark_for_exposure(tmp_path, target_exptime=100.0, matching_headers=[], max_exptime_ratio=3.0)
        assert result is None

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        """A missing dark directory yields None rather than raising."""
        assert find_best_dark_for_exposure(tmp_path / "nope", target_exptime=10.0) is None


# --- _group_frames_by_headers -------------------------------------------------


class TestGroupFramesByHeaders:
    """Grouping calibration frames by the headers that must match."""

    def test_groups_by_binning_and_exptime(self, tmp_path: Path) -> None:
        """Frames are grouped by the headers that have to match, binning and exposure."""
        d = np.zeros((4, 4))
        f1 = _write_fits(tmp_path / "a.fits", d, BINNING="1x1", EXPTIME=10.0)
        f2 = _write_fits(tmp_path / "b.fits", d, BINNING="1x1", EXPTIME=10.0)
        f3 = _write_fits(tmp_path / "c.fits", d, BINNING="2x2", EXPTIME=10.0)
        groups = _group_frames_by_headers([f1, f2, f3], ["BINNING", "EXPTIME"])
        assert len(groups) == 2
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2]

    def test_exptime_rounded_for_grouping(self, tmp_path: Path) -> None:
        """Exposure is rounded for grouping, so float noise does not split one set into many."""
        d = np.zeros((4, 4))
        f1 = _write_fits(tmp_path / "a.fits", d, EXPTIME=10.001)
        f2 = _write_fits(tmp_path / "b.fits", d, EXPTIME=10.004)
        groups = _group_frames_by_headers([f1, f2], ["EXPTIME"])
        # Both round to 10.0 -> single group.
        assert len(groups) == 1

    def test_empty_headers_single_group(self, tmp_path: Path) -> None:
        """With no grouping headers, every frame lands in one group."""
        f1 = _write_fits(tmp_path / "a.fits", np.zeros((4, 4)))
        groups = _group_frames_by_headers([f1], [])
        assert len(groups) == 1


# --- _find_master_calibration (flat-style exact matching) --------------------


class TestFindMasterCalibration:
    """Finding a master calibration frame whose headers match the science frame."""

    def test_matches_on_binning_and_filter(self, tmp_path: Path) -> None:
        """A master is matched on both binning and filter."""
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat_V.fits", d, XBINNING=1, FILTER="V")
        _write_fits(tmp_path / "flat_R.fits", d, XBINNING=1, FILTER="R")
        image = _make_image(d, XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is not None
        assert match.name == "flat_V.fits"

    def test_filter_case_insensitive(self, tmp_path: Path) -> None:
        """Filter matching ignores case, since sensors are inconsistent about it."""
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat.fits", d, XBINNING=1, FILTER="v")
        image = _make_image(d, XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is not None

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        """No matching master yields None rather than the nearest wrong one."""
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat.fits", d, XBINNING=2, FILTER="V")
        image = _make_image(d, XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is None

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        """A missing calibration directory yields None rather than raising."""
        image = _make_image(np.ones((8, 8)), XBINNING=1, FILTER="V")
        match = _find_master_calibration(image, str(tmp_path / "absent"), ["XBINNING"], "flat")
        assert match is None

    def test_missing_required_header_in_science_returns_none(self, tmp_path: Path) -> None:
        """A science frame missing a header needed for matching yields no master.

        Matching on a keyword the frame does not carry would pair it with an arbitrary master.
        """
        d = np.ones((8, 8))
        _write_fits(tmp_path / "flat.fits", d, XBINNING=1, FILTER="V")
        image = _make_image(d, XBINNING=1)  # no FILTER on science frame
        match = _find_master_calibration(image, str(tmp_path), ["XBINNING", "FILTER"], "flat")
        assert match is None


# --- _find_best_dark_calibration (header match + exposure ratio gating) ------


class TestFindBestDarkCalibration:
    """Choosing a dark that matches on binning and is close enough in exposure."""

    def test_selects_closest_exposure_among_matching(self, tmp_path: Path) -> None:
        """Among darks that match on binning, the closest exposure wins."""
        d = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark_9s.fits", d, XBINNING=1, EXPTIME=9.0)
        _write_fits(tmp_path / "dark_20s.fits", d, XBINNING=1, EXPTIME=20.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is not None
        assert match.name == "dark_9s.fits"

    def test_rejects_dark_with_wrong_binning(self, tmp_path: Path) -> None:
        """A dark with the wrong binning is rejected however close its exposure."""
        d = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark.fits", d, XBINNING=2, EXPTIME=10.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is None

    def test_exposure_ratio_within_limit_accepted(self, tmp_path: Path) -> None:
        """An exposure ratio inside the limit is accepted."""
        d = np.full((8, 8), 100.0)
        # ratio 30/10 = 3.0 == max -> accepted (<=).
        _write_fits(tmp_path / "dark.fits", d, XBINNING=1, EXPTIME=30.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is not None

    def test_exposure_ratio_beyond_limit_rejected(self, tmp_path: Path) -> None:
        """An exposure ratio beyond the limit is rejected."""
        d = np.full((8, 8), 100.0)
        # ratio 40/10 = 4.0 > max 3.0 -> rejected.
        _write_fits(tmp_path / "dark.fits", d, XBINNING=1, EXPTIME=40.0)
        image = _make_image(d, XBINNING=1, EXPTIME=10.0)
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        assert match is None

    def test_falls_back_to_exact_match_without_image_exptime(self, tmp_path: Path) -> None:
        """With no science exposure to compare, only an exact header match is accepted."""
        d = np.full((8, 8), 100.0)
        _write_fits(tmp_path / "dark.fits", d, XBINNING=1, EXPTIME=10.0)
        image = _make_image(d, XBINNING=1)  # science frame has no EXPTIME
        match = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
        # Falls back to exact-header matching; XBINNING matches -> found.
        assert match is not None


# --- end-to-end: best dark selection then subtraction ------------------------


def test_dark_selection_then_apply_scales_correctly(tmp_path: Path) -> None:
    """The exposure-ratio-selected dark, when applied, is scaled by the image/dark exposure ratio."""
    dark_data = np.full((16, 16), 50.0)
    _write_fits(tmp_path / "dark_5s.fits", dark_data, XBINNING=1, EXPTIME=5.0)
    image = _make_image(np.full((16, 16), 600.0), XBINNING=1, EXPTIME=10.0)

    best = _find_best_dark_calibration(image, str(tmp_path), ["XBINNING"], max_exposure_ratio=3.0)
    assert best is not None and best.name == "dark_5s.fits"

    out = apply_dark_subtraction(image, best)
    # dark (50) scaled by 10/5 = 2 -> subtract 100 -> 500.
    assert np.allclose(out.data, 500.0)
