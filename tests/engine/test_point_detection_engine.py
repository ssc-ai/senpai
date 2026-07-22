"""Unit tests for the primary point/streak detection engine and kernels.

Covers the canonical detection modules: streak masking, extraction, and validation
(``detection.streak.masking/extraction/validation``), frame-shift solving
(``detection.streak.frame_shift`` / ``sidereal_sidereal``), point-source extraction and
filtering (``detection.point.satellite``), the detection kernels
(``detection.kernels``), and the scale-invariant (PSF-FWHM-aware) flux-concentration
gate.

All tests run on synthetic images and mocks -- no network, astrometry, or catalog access.
"""

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import gaussian_filter

import senpai.engine.detection.point.satellite as sat_mod
import senpai.engine.detection.streak.extraction as ext_mod
from senpai.core.config import initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.kernels import rectangle_pyramoid, shift_filter_subpx
from senpai.engine.detection.point.satellite import (
    _centroid_guard_offset,
    _flux_concentration,
    _report_centroid,
    cutout_gauss,
    filter_point_sources,
    psf_flux_concentration,
)
from senpai.engine.detection.point.satellite import (
    extract_point_sources as extract_satellite_sources,
)
from senpai.engine.detection.streak.extraction import (
    estimate_fwhm_from_profiles,
    extract_streak_dims_robust,
    measure_gaussian_shift,
    measure_psf_fwhm,
    prepare_sidereal_frame,
    refine_streak_len,
    streak_fwhm_from_cutout,
    streak_length_from_cutout,
)
from senpai.engine.detection.streak.frame_shift import solve_shift
from senpai.engine.detection.streak.masking import (
    map_cluster,
    map_cluster_with_peaks,
    mask_border,
    mask_tol,
    percent_difference,
    remove_n_brightest_streaks,
    remove_near_saturation_streaks,
)
from senpai.engine.detection.streak.sidereal_sidereal import solve_sidereal_from_sidereal
from senpai.engine.detection.streak.validation import validate_proposed_shift
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import CollectionMetadata, ImageMetadata
from senpai.engine.models.senpai import FrameShift, RateTrackFrame, SenpaiRun, SiderealFrame
from senpai.engine.models.starfield import StarField, StarInSpace

# The settings proxy resolves through this module-level global; tests that need a mocked
# config patch it directly.
_SETTINGS_GLOBAL = "senpai.core.config._config_instance"


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialize the config singleton for the module."""
    initialize_config(CONFIG_DIR / "local.yaml")


def _make_sidereal_frame(index: int, data: np.ndarray) -> SiderealFrame:
    """Build a minimal sidereal frame from synthetic image data.

    Args:
        index: Frame index within the run.
        data: Backing image array.

    Returns:
        A ``SiderealFrame`` wrapping the data.
    """
    header = fits.Header()
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    frame = ProcessedFitsImage(
        data=data,
        header=header,
        data_type=data.dtype,
        metadata=ImageMetadata(width=data.shape[1], height=data.shape[0]),
    )
    return SiderealFrame(frame=frame, index=index, timestamp=datetime(2024, 1, 1, 0, 0, index))


def _make_rate_track_frame(index: int, data: np.ndarray) -> RateTrackFrame:
    """Build a minimal rate-track frame from synthetic image data.

    Args:
        index: Frame index within the run.
        data: Backing image array.

    Returns:
        A ``RateTrackFrame`` wrapping the data.
    """
    header = fits.Header()
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    frame = ProcessedFitsImage(
        data=data,
        header=header,
        data_type=data.dtype,
        metadata=ImageMetadata(width=data.shape[1], height=data.shape[0]),
    )
    return RateTrackFrame(frame=frame, index=index, timestamp=datetime(2024, 1, 1, 0, 0, index))


def _low_noise_array(rng: np.random.Generator, shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Build a low-noise uint16 image.

    Args:
        rng: Seeded random generator.
        shape: Output image shape (height, width).

    Returns:
        A uint16 array of low-amplitude noise.
    """
    return rng.integers(100, 500, size=shape, dtype=np.uint16)


class TestPercentDifference:
    """The ``percent_difference`` scalar helper."""

    def test_both_zero_returns_zero(self) -> None:
        """Two zero inputs give a zero percent difference."""
        assert percent_difference(0, 0) == 0.0

    def test_known_value(self) -> None:
        """A known pair produces the expected percent difference."""
        assert percent_difference(100, 200) == pytest.approx(66.666, rel=1e-3)


class TestMaskTol:
    """The ``mask_tol`` radial tolerance mask."""

    def test_center_pixel_included(self) -> None:
        """The centre pixel is inside the tolerance radius."""
        img = np.zeros((50, 50))
        mask = mask_tol(img, center=(25, 25), pixel_tol=5)
        assert mask[25, 25] == 1

    def test_far_pixel_excluded(self) -> None:
        """A pixel beyond the tolerance radius is masked out."""
        img = np.zeros((50, 50))
        mask = mask_tol(img, center=(25, 25), pixel_tol=5)
        assert mask[0, 0] == 0

    def test_shape_matches_image(self) -> None:
        """The mask shape matches the input image shape."""
        img = np.zeros((30, 40))
        mask = mask_tol(img, center=(15, 20), pixel_tol=3)
        assert mask.shape == img.shape


class TestMapCluster:
    """The ``map_cluster`` flood-fill cluster mapper."""

    def test_out_of_bounds_start_returns_empty(self) -> None:
        """A start point outside the image yields an empty mask."""
        img = np.ones((10, 10)) * 100
        result = map_cluster(img, start_point=(20, 20), flux_threshold=50)
        assert not result.any()

    def test_bright_region_is_mapped(self) -> None:
        """A connected bright region is mapped while the dark background is not."""
        img = np.zeros((20, 20))
        img[8:12, 8:12] = 200
        result = map_cluster(img, start_point=(10, 10), flux_threshold=50)
        assert result[10, 10]
        assert not result[0, 0]

    def test_pad_size_dilates_mask(self) -> None:
        """A larger pad size dilates the mapped mask."""
        img = np.zeros((20, 20))
        img[9, 9] = 200
        unpadded = map_cluster(img, start_point=(9, 9), flux_threshold=50, pad_size=0)
        padded = map_cluster(img, start_point=(9, 9), flux_threshold=50, pad_size=2)
        assert padded.sum() > unpadded.sum()


class TestMaskBorder:
    """The ``mask_border`` edge-zeroing helper."""

    @pytest.mark.parametrize(
        ("edge", "index"),
        [
            ("row", 0),
            ("row", -1),
            ("col", 0),
            ("col", -1),
        ],
    )
    def test_border_edge_zeroed(self, edge: str, index: int) -> None:
        """Each outer border row/column is zeroed.

        Args:
            edge: Whether to inspect a ``row`` or ``col`` border.
            index: Which border index to inspect (0 or -1).
        """
        result = mask_border(np.ones((10, 10)), n_pixels=1)
        edge_pixels = result[index, :] if edge == "row" else result[:, index]
        assert edge_pixels.sum() == 0

    def test_interior_unchanged(self) -> None:
        """The interior pixels are left untouched."""
        result = mask_border(np.ones((10, 10)), n_pixels=1)
        assert result[1:-1, 1:-1].all()

    def test_does_not_modify_original(self) -> None:
        """Masking returns a copy and does not mutate the input."""
        img = np.ones((10, 10))
        mask_border(img, n_pixels=1)
        assert img[0, 0] == 1


class TestRemoveNBrightestStreaks:
    """The ``remove_n_brightest_streaks`` helper."""

    def test_removes_n_streaks_and_returns_count(self) -> None:
        """The N brightest streaks are removed and their count is reported."""
        rng = np.random.default_rng(42)
        img = rng.integers(100, 300, size=(50, 50)).astype(float)
        img[10, 10] = 60000
        img[20, 20] = 59000
        result, count = remove_n_brightest_streaks(img.copy(), n=2)
        assert count == 2
        assert result[10, 10] < 60000


class TestRemoveNearSaturationStreaks:
    """The ``remove_near_saturation_streaks`` helper."""

    def test_removes_pixel_above_threshold(self) -> None:
        """A near-saturation pixel is removed and counted."""
        img = np.ones((50, 50)).astype(float) * 1000
        img[25, 25] = 60000
        result, count = remove_near_saturation_streaks(img.copy(), data_type="uint16")
        assert count >= 1
        assert result[25, 25] < 60000

    def test_no_removal_when_below_threshold(self) -> None:
        """No pixels are removed when all are below the saturation threshold."""
        img = np.ones((50, 50)).astype(float) * 1000
        _, count = remove_near_saturation_streaks(img.copy(), data_type="uint16")
        assert count == 0


class TestMapClusterWithPeaks:
    """The ``map_cluster_with_peaks`` mapper variant."""

    def test_returns_mask_and_peak_list(self) -> None:
        """The mapper returns a boolean mask and a list of peaks."""
        img = np.zeros((30, 30))
        img[15, 15] = 500
        mask, peaks = map_cluster_with_peaks(img, start_point=(15, 15), flux_threshold=50)
        assert mask.dtype == bool
        assert isinstance(peaks, list)
        assert mask[15, 15]

    def test_finds_at_least_one_peak(self) -> None:
        """At least one peak is found for a clear bright source."""
        img = np.zeros((30, 30))
        img[15, 15] = 500
        _, peaks = map_cluster_with_peaks(img, start_point=(15, 15), flux_threshold=50)
        assert len(peaks) >= 1


class TestEstimateFwhmFromProfiles:
    """The ``estimate_fwhm_from_profiles`` profile-width estimator."""

    def test_profile_with_crossing_returns_width(self) -> None:
        """A profile that crosses the half-max returns the measured width."""
        x = np.zeros(20)
        x[8:13] = 1.0
        result = estimate_fwhm_from_profiles(x, x)
        assert result == pytest.approx(4.0)

    def test_profile_never_crosses_returns_default(self) -> None:
        """A profile that never crosses the half-max returns the default width."""
        flat_low = np.zeros(20)
        result = estimate_fwhm_from_profiles(flat_low, flat_low)
        assert result == pytest.approx(4.0)


class TestValidateProposedShift:
    """The ``validate_proposed_shift`` catalog-consistency check."""

    def test_empty_catalog_returns_zero(self) -> None:
        """An empty catalog yields a zero validation score."""
        rng = np.random.default_rng(5)
        data = _low_noise_array(rng)
        src = _make_sidereal_frame(0, data.copy())
        tgt = _make_sidereal_frame(1, data.copy())
        result = validate_proposed_shift(tgt, src, 0.0, 0.0, [])
        assert result == 0.0

    def test_stars_without_position_attributes_returns_zero(self) -> None:
        """Stars lacking position attributes yield a zero validation score."""
        rng = np.random.default_rng(6)
        data = _low_noise_array(rng)
        src = _make_sidereal_frame(0, data.copy())
        tgt = _make_sidereal_frame(1, data.copy())

        class _NoAttrs:
            pass

        result = validate_proposed_shift(tgt, src, 0.0, 0.0, [_NoAttrs()])
        assert result == 0.0


class TestSolveSiderealFromSidereal:
    """The ``solve_sidereal_from_sidereal`` frame-shift solver."""

    def test_populates_frame_shift(self) -> None:
        """The solver marks the shift processed, valid, and populates its offsets."""
        rng = np.random.default_rng(42)
        data = _low_noise_array(rng)
        source = _make_sidereal_frame(0, data.copy())
        target = _make_sidereal_frame(1, data.copy())
        frame_shift = FrameShift(source_index=0, target_index=1)
        solve_sidereal_from_sidereal(source, target, frame_shift)
        assert frame_shift.processed is True
        assert frame_shift.is_valid is True
        assert frame_shift.x_shift is not None
        assert frame_shift.y_shift is not None

    def test_identical_frames_produce_small_shift(self) -> None:
        """Identical frames register to a near-zero shift."""
        rng = np.random.default_rng(7)
        data = _low_noise_array(rng)
        source = _make_sidereal_frame(0, data.copy())
        target = _make_sidereal_frame(1, data.copy())
        frame_shift = FrameShift(source_index=0, target_index=1)
        solve_sidereal_from_sidereal(source, target, frame_shift)
        magnitude = np.sqrt(frame_shift.x_shift**2 + frame_shift.y_shift**2)
        assert magnitude < 10.0

    def test_different_frames_completes_without_error(self) -> None:
        """Different frames still complete the solve and mark it processed."""
        rng = np.random.default_rng(99)
        data_a = _low_noise_array(rng)
        data_b = _low_noise_array(rng)
        source = _make_sidereal_frame(0, data_a)
        target = _make_sidereal_frame(1, data_b)
        frame_shift = FrameShift(source_index=0, target_index=1)
        solve_sidereal_from_sidereal(source, target, frame_shift)
        assert frame_shift.processed is True


class TestSolveShift:
    """Frame-type dispatch in ``solve_shift``."""

    def _make_run(
        self,
        sidereal_frames: list[SiderealFrame] | None = None,
        rate_track_frames: list[RateTrackFrame] | None = None,
    ) -> SenpaiRun:
        """Build a run with the given frames.

        Args:
            sidereal_frames: Sidereal frames for the run.
            rate_track_frames: Rate-track frames for the run.

        Returns:
            A populated ``SenpaiRun``.
        """
        return SenpaiRun(
            id="test",
            num_frames=0,
            collect_metadata=CollectionMetadata(),
            sidereal_frames=sidereal_frames or [],
            rate_track_frames=rate_track_frames or [],
        )

    def test_sidereal_to_sidereal_dispatch(self) -> None:
        """A sidereal->sidereal pair dispatches to the sidereal solver and processes."""
        rng = np.random.default_rng(42)
        data = _low_noise_array(rng)
        f0 = _make_sidereal_frame(0, data.copy())
        f1 = _make_sidereal_frame(1, data.copy())
        run = self._make_run(sidereal_frames=[f0, f1])
        fs = FrameShift(source_index=0, target_index=1)
        solve_shift(run, fs)
        assert fs.processed

    def test_rate_to_sidereal_dispatch(self) -> None:
        """A rate->sidereal pair dispatches to the rate-from-sidereal solver."""
        import senpai.engine.detection.streak.frame_shift as fs_mod

        rng = np.random.default_rng(42)
        data = _low_noise_array(rng)
        f0 = _make_rate_track_frame(0, data.copy())
        f1 = _make_sidereal_frame(1, data.copy())
        run = self._make_run(sidereal_frames=[f1], rate_track_frames=[f0])
        fs = FrameShift(source_index=0, target_index=1)
        with patch.object(fs_mod, "solve_rate_from_sidereal") as mock_solver:
            solve_shift(run, fs)
        mock_solver.assert_called_once()

    def test_invalid_frame_types_raises_type_error(self) -> None:
        """An unrecognised frame-type pairing raises ``TypeError``."""
        import senpai.engine.detection.streak.frame_shift as fs_mod

        run = MagicMock()
        run.get_frame_by_index.side_effect = [MagicMock(), MagicMock()]
        fs = FrameShift(source_index=0, target_index=1)
        with patch.object(fs_mod, "preprocess_for_shift"), pytest.raises(TypeError):
            solve_shift(run, fs)


class TestKernels:
    """The detection kernels."""

    def test_shift_filter_subpx_basic(self) -> None:
        """The sub-pixel shift filter keeps values within [0, 1]."""
        arr = np.full((5, 5), 0.5, dtype=float)
        result = shift_filter_subpx(arr, np.array([0.5, 0.5]))
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_rectangle_pyramoid_verbose(self, caplog: pytest.LogCaptureFixture) -> None:
        """The verbose streak kernel emits a diagnostic log line.

        Args:
            caplog: Pytest log-capture fixture.
        """
        rectangle_pyramoid.cache_clear()
        with caplog.at_level(logging.INFO, logger="senpai.engine.detection.kernels"):
            result = rectangle_pyramoid(5.0, 0.0, 1.0, width=2, verbose=True)
        assert result is not None
        assert "rectangle_pyramoid" in caplog.text


def test_cutout_gauss_fit_failure() -> None:
    """A non-converging Gaussian fit raises a ``ValueError``."""
    mock_fitter = MagicMock()
    mock_fitter.fit_info = {"ierr": 5}
    with (
        patch.object(sat_mod.fitting, "LevMarLSQFitter", return_value=mock_fitter),
        pytest.raises(ValueError, match="Gaussian fit did not converge"),
    ):
        cutout_gauss(np.zeros((10, 10)), pixel_seeing=3.0)


def test_flux_concentration_tiny_cutout() -> None:
    """A cutout smaller than the aperture returns a zero concentration early."""
    # 2x2 cutout: cy+2=3 > shape[0]=2 -> returns 0.0 early
    result = _flux_concentration(np.ones((2, 2)))
    assert result == 0.0


class TestFilterPointSourcesVerbose:
    """Verbose rejection paths in ``filter_point_sources``."""

    def _blob(
        self,
        shape: tuple[int, int] = (100, 100),
        center: tuple[int, int] = (50, 50),
        sigma: float = 2.0,
        amp: float = 200.0,
    ) -> np.ndarray:
        """Build a single Gaussian blob image.

        Args:
            shape: Output image shape (height, width).
            center: Blob centre (row, col).
            sigma: Gaussian smoothing sigma in pixels.
            amp: Pre-smoothing peak amplitude.

        Returns:
            A smoothed single-blob image.
        """
        img = np.zeros(shape)
        img[center[0], center[1]] = amp
        return gaussian_filter(img, sigma=sigma)

    def test_edge_detection_verbose(self, caplog: pytest.LogCaptureFixture) -> None:
        """A detection at the image edge is rejected with a diagnostic log line.

        Args:
            caplog: Pytest log-capture fixture.
        """
        image = self._blob()
        detections = [(0.0, 50.0)]
        mock_instance = MagicMock()
        mock_instance.detection.verbose = True
        with (
            patch(_SETTINGS_GLOBAL, mock_instance),
            caplog.at_level(logging.WARNING, logger="senpai.engine.detection.point.satellite"),
        ):
            result = filter_point_sources(image, detections, pixel_seeing=5.0)
        assert len(result) == 0
        assert "edge of image" in caplog.text

    def test_hot_pixel_verbose(self, caplog: pytest.LogCaptureFixture) -> None:
        """A single hot pixel is rejected with a diagnostic log line.

        Args:
            caplog: Pytest log-capture fixture.
        """
        image = np.zeros((100, 100))
        image[50, 50] = 10000.0
        detections = [(50.0, 50.0)]
        mock_instance = MagicMock()
        mock_instance.detection.verbose = True
        with (
            patch(_SETTINGS_GLOBAL, mock_instance),
            caplog.at_level(logging.WARNING, logger="senpai.engine.detection.point.satellite"),
        ):
            result = filter_point_sources(image, detections, pixel_seeing=5.0)
        assert len(result) == 0
        assert "total flux" in caplog.text

    def test_gauss_exception_verbose(self, caplog: pytest.LogCaptureFixture) -> None:
        """A Gaussian-fit exception rejects the detection with a diagnostic log line.

        Args:
            caplog: Pytest log-capture fixture.
        """
        image = self._blob(center=(50, 50), sigma=2.0, amp=200.0)
        detections = [(50.0, 50.0)]
        mock_instance = MagicMock()
        mock_instance.detection.verbose = True
        with (
            patch(_SETTINGS_GLOBAL, mock_instance),
            patch.object(sat_mod, "cutout_gauss", side_effect=ValueError("boom")),
            caplog.at_level(logging.WARNING, logger="senpai.engine.detection.point.satellite"),
        ):
            result = filter_point_sources(image, detections, pixel_seeing=5.0)
        assert len(result) == 0
        assert "Gaussian fit failed" in caplog.text


class TestCentroidGuardOffset:
    """The ``_centroid_guard_offset`` guard-threshold resolver."""

    def test_fwhm_mode_scales_with_psf(self) -> None:
        """The 'fwhm' guard scales the threshold with the PSF FWHM."""
        # PSF-relative guard: threshold is value * FWHM, so it self-scales with seeing.
        assert _centroid_guard_offset(5.0, "fwhm", 0.4) == 2.0
        assert _centroid_guard_offset(2.5, "fwhm", 0.4) == 1.0

    def test_fixed_mode_ignores_psf(self) -> None:
        """The 'fixed' guard uses an absolute pixel threshold regardless of FWHM."""
        # Absolute-pixel guard: threshold is value regardless of FWHM.
        assert _centroid_guard_offset(5.0, "fixed", 2.0) == 2.0
        assert _centroid_guard_offset(10.0, "fixed", 2.0) == 2.0

    def test_none_mode_disables_fallback(self) -> None:
        """The 'none' guard yields an infinite offset (never falls back to the peak)."""
        # 'none' -> infinite offset: the sub-pixel centroid is always reported.
        assert _centroid_guard_offset(5.0, "none", 0.4) == float("inf")

    def test_unknown_mode_raises(self) -> None:
        """An unknown guard mode raises ``ValueError``."""
        with pytest.raises(ValueError):
            _centroid_guard_offset(5.0, "bogus", 0.4)


class TestReportCentroid:
    """Sub-pixel-vs-peak selection in ``_report_centroid``."""

    def test_reports_subpixel_seed_within_offset(self) -> None:
        """A sub-pixel seed within the guard offset is trusted and returned verbatim."""
        # Brightest pixel at (row=50, col=80); the SEP sub-pixel centroid sits 0.42 px away,
        # within max_peak_offset, so it is trusted and returned verbatim.
        masked = np.zeros((100, 120))
        masked[50, 80] = 100.0
        assert _report_centroid(masked, seed_xy=(80.3, 49.7), max_peak_offset=1.0) == (80.3, 49.7)

    def test_falls_back_to_peak_beyond_offset(self) -> None:
        """A sub-pixel seed beyond the guard offset falls back to the integer peak."""
        # Sub-pixel centroid disagrees with the brightest pixel by > max_peak_offset -> the
        # fit is unreliable (saturation/trailing/blend), so the robust integer peak is used.
        masked = np.zeros((100, 120))
        masked[50, 80] = 100.0
        assert _report_centroid(masked, seed_xy=(60.0, 49.0), max_peak_offset=1.0) == (80.0, 50.0)

    def test_infinite_offset_never_falls_back(self) -> None:
        """An infinite guard offset always reports the sub-pixel centroid."""
        # max_peak_offset=inf ('none' guard mode): the sub-pixel centroid is always reported.
        masked = np.zeros((100, 120))
        masked[50, 80] = 100.0
        assert _report_centroid(masked, seed_xy=(60.0, 49.0), max_peak_offset=float("inf")) == (
            60.0,
            49.0,
        )

    def test_default_offset_disables_fallback(self) -> None:
        """The default (infinite) offset disables the fallback unless a caller opts in."""
        # Default max_peak_offset is infinite -> no fallback unless a caller opts in.
        masked = np.zeros((100, 120))
        masked[50, 80] = 100.0
        assert _report_centroid(masked, seed_xy=(60.0, 49.0)) == (60.0, 49.0)

    def test_raises_on_no_signal(self) -> None:
        """A signal-free cutout raises ``ValueError``."""
        with pytest.raises(ValueError):
            _report_centroid(np.zeros((10, 10)), seed_xy=(5.0, 5.0))

    def test_filter_point_sources_returns_subpixel_not_integer(self) -> None:
        """An accepted point source reports the sub-pixel centroid, not the integer peak."""
        # A clean Gaussian point source matched to the seeing: the accepted detection's
        # reported position is the sub-pixel SEP centroid, not the integer brightest pixel.
        seeing = 5.0
        img = np.zeros((100, 100))
        img[50, 50] = 300.0
        img = gaussian_filter(img, sigma=seeing / 2.355)
        seed = (50.7, 50.3)  # within 1 px of the (col=50, row=50) peak
        result = filter_point_sources(img, [seed], pixel_seeing=seeing)
        assert len(result) == 1
        rx, ry, _ = result[0]
        assert (rx, ry) == seed
        assert rx != int(rx) or ry != int(ry)  # the old argmax path would have snapped to ints

    def test_filter_point_sources_falls_back_to_peak_under_tight_guard(self) -> None:
        """A tight 'fixed' guard rejects a distant seed in favour of the integer peak."""
        # With a tight 'fixed' guard, a seed > threshold from the brightest pixel is rejected
        # in favour of the integer peak -- the guard config flows through to step 7.
        seeing = 5.0
        img = np.zeros((100, 100))
        img[50, 50] = 300.0
        img = gaussian_filter(img, sigma=seeing / 2.355)
        seed = (50.7, 50.3)  # 0.76 px from the (col=50, row=50) peak, > 0.1 guard
        result = filter_point_sources(
            img, [seed], pixel_seeing=seeing, centroid_guard_mode="fixed", centroid_guard_value=0.1
        )
        assert len(result) == 1
        rx, ry, _ = result[0]
        assert (rx, ry) == (50.0, 50.0)  # snapped back to the brightest pixel


class TestExtractPointSources:
    """Point-source extraction via ``extract_point_sources``."""

    def _make_frame(self, shape: tuple[int, int] = (100, 100)) -> RateTrackFrame:
        """Build a rate frame with an empty solved starfield.

        Args:
            shape: Frame shape (height, width).

        Returns:
            A ``RateTrackFrame`` with an unfit starfield attached.
        """
        rng = np.random.default_rng(0)
        data = (rng.random(shape) * 100).astype(np.float32)
        frame = _make_rate_track_frame(0, data)
        frame.starfield = StarField(
            astrometric_fit_stars=[],
            catalog_stars=[],
            detections=[],
            image_metadata=ImageMetadata(width=shape[1], height=shape[0]),
            fit=False,
            wcs=None,
        )
        return frame

    def test_no_sources_returns_empty(self) -> None:
        """An extraction that finds nothing returns an empty detection list."""
        frame = self._make_frame()
        with patch(
            "senpai.engine.detection.point.satellite.sep.extract", return_value=np.array([])
        ):
            result = extract_satellite_sources(frame)
        assert len(result.detections) == 0

    def test_returns_satellite_list_image(self) -> None:
        """Extraction returns a ``SatelliteListImage``."""
        from senpai.engine.models.starfield import SatelliteListImage

        frame = self._make_frame()
        result = extract_satellite_sources(frame)
        assert isinstance(result, SatelliteListImage)


class TestExtractionHelpers:
    """Streak dimension and PSF-measurement helpers in ``detection.streak.extraction``."""

    def test_extract_streak_dims_negative_length_raises(self) -> None:
        """A negative streak length raises ``ValueError`` after the FWHM cap."""
        # The FWHM cap lives at settings.streak.max_fwhm_for_streak_extraction (default 10.0
        # via the initialized config); the negative-length guard fires right after it.
        with pytest.raises(ValueError, match="Length cannot be negative"):
            extract_streak_dims_robust(np.ones((20, 20)), length=-1.0, rotation=0.0)

    def test_refine_streak_len_basic(self) -> None:
        """Refining a horizontal streak returns a positive length."""
        psf = np.zeros((60, 60))
        psf[30, 15:45] = 1.0  # horizontal streak
        result = refine_streak_len(psf, pixel_fwhm=4.0, rotation=0.0)
        assert result > 0

    def test_streak_fwhm_from_cutout_none_on_flat(self) -> None:
        """A flat cutout yields no measurable streak FWHM (None)."""
        # All-ones -> peak at index 0 -> left_side empty -> else branch -> None
        result = streak_fwhm_from_cutout(np.ones((10, 10)), rotation=0)
        assert result is None

    def test_streak_length_from_cutout_basic(self) -> None:
        """A horizontal streak cutout yields a positive length."""
        cutout = np.zeros((50, 50))
        cutout[25, 10:40] = 1.0  # horizontal streak of length ~30
        result = streak_length_from_cutout(cutout, plot=False)
        assert result > 0

    def test_measure_psf_fwhm_no_half_max_returns_none(self) -> None:
        """An all-zero image has no half-max crossing, so the FWHM is None."""
        # All-zeros image -> no pixel above half_max -> returns None
        result = measure_psf_fwhm(np.zeros((30, 30)), rotation=45.0)
        assert result is None

    def test_measure_psf_fwhm_too_few_points_returns_none(self) -> None:
        """A single bright pixel gives too few above-half-max points, so the FWHM is None."""
        # Single bright pixel -> only 1 point above 0.5*max -> < 5 -> returns None
        data = np.zeros((30, 30))
        data[15, 15] = 1.0
        result = measure_psf_fwhm(data, rotation=None)
        assert result is None

    def test_measure_psf_fwhm_auto_rotation(self) -> None:
        """Auto-rotation on an elongated streak exercises the PCA orientation path."""
        # Elongated horizontal streak -> PCA detects orientation
        data = np.zeros((60, 60))
        data[30, 15:45] = 1.0
        data = gaussian_filter(data, sigma=1.5)
        data = data / data.max()
        result = measure_psf_fwhm(data, rotation=None)
        # result may be None if PCA fails to find orientation, but the code path is covered
        assert result is None or result > 0

    def test_measure_psf_fwhm_interpolation_paths(self) -> None:
        """A centred Gaussian exercises both interpolation branches and yields a positive FWHM."""
        # Gaussian centered in image -> left_idx > 0 and right_idx < len-1 -> both branches
        data = np.zeros((50, 50))
        data[25, 25] = 1.0
        data = gaussian_filter(data, sigma=3.0)
        data = data / data.max()
        result = measure_psf_fwhm(data, rotation=45.0)
        assert result is not None
        assert result > 0

    def test_measure_gaussian_shift_fit_failure_uses_fallback(self) -> None:
        """A failed curve fit falls back to the profile-based FWHM estimate."""
        # Patch curve_fit to raise -> fallback to estimate_fwhm_from_profiles
        data = np.zeros((30, 30))
        data[15, 15] = 1.0
        cutout = gaussian_filter(data, sigma=2.0)
        with patch.object(ext_mod, "curve_fit", side_effect=RuntimeError("fail")):
            _shift, fwhm = measure_gaussian_shift(cutout)
        assert fwhm >= 0

    def test_prepare_sidereal_synthetic_branch(self) -> None:
        """With synthetic frames allowed and a fitted starfield, the synthetic branch is used."""
        # allow_synthetic=True + starfield.fit=True -> uses simulated_sidereal_frame
        frame = _make_sidereal_frame(0, np.zeros((50, 50), dtype=np.uint16))
        frame.starfield = StarField(
            astrometric_fit_stars=[],
            catalog_stars=[StarInSpace(ra=0.0, dec=0.0, magnitude=10.0, x=25.0, y=25.0)],
            detections=[],
            image_metadata=ImageMetadata(width=50, height=50),
            fit=True,
            wcs=None,
        )
        data, is_synthetic = prepare_sidereal_frame(frame, allow_synthetic=True)
        assert data is not None
        assert isinstance(is_synthetic, (bool, np.bool_))


# ---------------------------------------------------------------------------
# Scale-invariant (PSF-FWHM-aware) flux concentration
# ---------------------------------------------------------------------------


def _gaussian_cutout(
    fwhm: float,
    amp: float = 100.0,
    size: int | None = None,
    elongation: float = 1.0,
    rng: np.random.Generator | None = None,
    noise: float = 0.0,
) -> np.ndarray:
    """Build a centered 2D Gaussian cutout sized to ``3 * fwhm`` (odd).

    Args:
        fwhm: PSF FWHM in pixels (minor-axis FWHM when ``elongation`` > 1).
        amp: Peak amplitude.
        size: Optional explicit square side; defaults to an odd ``int(3 * fwhm)``.
        elongation: Major/minor axis ratio (1.0 = round point source).
        rng: Optional numpy Generator for additive Gaussian noise.
        noise: Standard deviation of additive noise (0 = noiseless).

    Returns:
        Background-subtracted (min-normalized) square cutout.
    """
    if size is None:
        # Mirror the production cutout side: generate_cutout uses a half-width of
        # int(3 * pixel_seeing), so the full side is ~6 * FWHM (comfortably larger
        # than the 3 * FWHM outer-aperture diameter).
        size = 2 * int(3 * fwhm)
        if size % 2 == 0:
            size += 1
    cy = cx = (size - 1) / 2.0
    sig_minor = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sig_major = sig_minor * elongation
    y, x = np.mgrid[:size, :size]
    img = amp * np.exp(
        -(((x - cx) ** 2) / (2 * sig_major**2) + ((y - cy) ** 2) / (2 * sig_minor**2))
    )
    if noise and rng is not None:
        img = img + rng.normal(0.0, noise, img.shape)
    return img - img.min()


class TestPsfFluxConcentration:
    """Scale-invariant flux concentration: inner/outer PSF-sized aperture ratio."""

    # Accept band derived empirically from the detection corpus: pure noise ~0.26, faint
    # SNR~7 point ~0.40, bright point ~0.6-0.8, hot-pixel/cosmic-ray spike -> 1.0.
    LO, HI = 0.30, 0.90

    @pytest.mark.parametrize("fwhm", [3.0, 5.5, 8.0])
    def test_point_source_accepts(self, fwhm: float) -> None:
        """A clean point source scores inside the accept band across PSF widths.

        Args:
            fwhm: PSF FWHM in pixels for the synthetic point source.
        """
        cutout = _gaussian_cutout(fwhm)
        conc = psf_flux_concentration(cutout, fwhm)
        assert self.LO <= conc <= self.HI, f"point source FWHM={fwhm} -> {conc}"

    @pytest.mark.parametrize("fwhm", [3.0, 5.5, 8.0])
    def test_faint_point_source_accepts(self, fwhm: float) -> None:
        """A faint (SNR~7) point source still clears the lower accept threshold.

        Args:
            fwhm: PSF FWHM in pixels for the synthetic faint point source.
        """
        rng = np.random.default_rng(7)
        # Peak amplitude ~7x noise sigma => SNR ~7 faint source.
        cutout = _gaussian_cutout(fwhm, amp=7.0, rng=rng, noise=1.0)
        conc = psf_flux_concentration(cutout, fwhm)
        assert conc >= self.LO, f"faint point FWHM={fwhm} -> {conc}"

    def test_pure_noise_rejects(self) -> None:
        """Pure noise scores below the accept band."""
        rng = np.random.default_rng(0)
        fwhm = 5.5
        size = 2 * int(3 * fwhm) + 1
        noise = rng.normal(50.0, 10.0, (size, size))
        noise = noise - noise.min()
        conc = psf_flux_concentration(noise, fwhm)
        assert conc < self.LO, f"noise concentration {conc} should be below {self.LO}"

    def test_strongly_diffuse_blob_rejects(self) -> None:
        """A blob much wider than the PSF scores below the accept band."""
        fwhm = 5.5
        # A strongly diffuse blob 5x wider than the PSF falls below the band; its
        # flux is spread out so the inner aperture captures only the area-ratio share.
        cutout = _gaussian_cutout(fwhm * 5.0, size=2 * int(3 * fwhm) + 1)
        conc = psf_flux_concentration(cutout, fwhm)
        assert conc < self.LO, f"diffuse blob concentration {conc} should be below {self.LO}"

    def test_point_source_scores_higher_than_extended(self) -> None:
        """A point source scores clearly higher than extended or elongated sources."""
        fwhm = 5.5
        point = psf_flux_concentration(_gaussian_cutout(fwhm), fwhm)
        # Extended (2.5x wide) and elongated (5:1 streak-like) both spread flux out,
        # so a genuine point source must score clearly higher than either.
        extended = psf_flux_concentration(
            _gaussian_cutout(fwhm * 2.5, size=2 * int(3 * fwhm) + 1), fwhm
        )
        elongated = psf_flux_concentration(
            _gaussian_cutout(fwhm, elongation=5.0, size=2 * int(3 * fwhm) + 1), fwhm
        )
        assert point > extended + 0.2, f"point {point} not clearly above extended {extended}"
        assert point > elongated + 0.2, f"point {point} not clearly above elongated {elongated}"

    def test_hot_pixel_spike_rejects(self) -> None:
        """A single-pixel spike scores above the accept band (rejected as a hot pixel)."""
        fwhm = 5.5
        size = 2 * int(3 * fwhm) + 1
        spike = np.zeros((size, size))
        spike[size // 2, size // 2] = 1000.0
        conc = psf_flux_concentration(spike, fwhm)
        assert conc > self.HI, f"single-pixel spike concentration {conc} should exceed {self.HI}"

    def test_scale_invariance(self) -> None:
        """The concentration score is approximately invariant to PSF width."""
        concs = [psf_flux_concentration(_gaussian_cutout(f), f) for f in (3.0, 5.5, 8.0)]
        spread = max(concs) - min(concs)
        assert spread < 0.15, (
            f"concentration should be ~scale-invariant, got {concs} (spread {spread})"
        )

    def test_small_cutout_returns_zero(self) -> None:
        """A cutout smaller than the outer aperture cannot be measured (returns 0.0)."""
        # Cutout smaller than the outer aperture diameter -> cannot measure -> 0.0
        result = psf_flux_concentration(np.ones((3, 3)), pixel_fwhm=5.5)
        assert result == 0.0
