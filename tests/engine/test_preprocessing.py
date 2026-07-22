"""Behavioral tests for senpai.engine.utils.preprocessing.

Covers row/column median removal, 2D background subtraction, the two image
scaling methods (block_median + blur_decimate), the FWHM-driven scaling
decision, and the config-driven preprocess_image pipeline.

All data is synthetic with a seeded RNG and crafted offset/gradient patterns so
the expected output is known a priori.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.images import ProcessedFitsImage, ProcessingStep
from senpai.engine.models.metadata import FWHMMetadata, ImageMetadata
from senpai.engine.utils.preprocessing import (
    estimate_gain_from_sky,
    measure_background,
    preprocess_image,
    remove_background,
    remove_column_and_row_medians,
    scale_image_block_median,
    scale_image_blur_decimate,
    scale_image_to_target_fwhm,
)

RNG = np.random.default_rng(20260603)


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialize the process-wide AppConfig singleton from a bundled YAML.

    The global config is process-wide, so it is seeded from a bundled YAML file
    to ensure get_config() does not raise for the modules under test.
    """
    initialize_config(CONFIG_DIR / "burr.yaml")
    get_config().plotting.debug = False


def _make_image(
    data: np.ndarray,
    header: fits.Header | None = None,
    correction_frames: dict[ProcessingStep, np.ndarray] | None = None,
) -> ProcessedFitsImage:
    """Wrap a raw pixel array in a minimal ProcessedFitsImage for testing.

    Args:
        data: 2D pixel array; promoted to float64 on the returned image.
        header: Optional FITS header; a minimal float64 header is synthesized
            when omitted.
        correction_frames: Optional per-step correction arrays to seed on the
            image (callers that exercise store_intermediates must pass a dict).

    Returns:
        A ProcessedFitsImage backed by the supplied data and metadata.
    """
    h, w = data.shape
    if header is None:
        header = fits.Header()
        header["BITPIX"] = -64
    meta = ImageMetadata(width=w, height=h)
    return ProcessedFitsImage(
        data=data.astype(np.float64),
        header=header,
        data_type=data.dtype,
        metadata=meta,
        file_path="synthetic.fits",
        correction_frames=correction_frames,
    )


def _fwhm_stats(median_fwhm: float) -> FWHMMetadata:
    """Build FWHMMetadata whose oversampling flags follow the median FWHM.

    Args:
        median_fwhm: Median FWHM (pixels) used to populate every summary field
            and to derive is_oversampled / recommended_scale_factor.

    Returns:
        A fully populated FWHMMetadata instance.
    """
    return FWHMMetadata(
        n_measurements=10,
        median_fwhm=median_fwhm,
        mean_fwhm=median_fwhm,
        std_fwhm=0.1,
        min_fwhm=median_fwhm,
        max_fwhm=median_fwhm,
        fwhm_vs_position=[],
        fwhm_vs_magnitude=[],
        fwhm_vs_counts=[],
        is_oversampled=median_fwhm > 4.0,
        recommended_scale_factor=(median_fwhm / 3.0 if median_fwhm > 4.0 else None),
    )


# --- row/column median removal -----------------------------------------------


class TestRemoveColumnAndRowMedians:
    """Tests for remove_column_and_row_medians (per-axis median subtraction)."""

    def test_removes_per_column_offsets(self) -> None:
        """A pure per-column offset pattern is flattened to zero."""
        # Each column has a distinct constant offset; after column-median
        # removal every column median should be ~0.
        base = np.zeros((40, 40))
        col_offsets = np.arange(40, dtype=float) * 3.0
        data = base + col_offsets[np.newaxis, :]
        img = _make_image(data)

        out = remove_column_and_row_medians(img)

        # Column medians removed first, then row medians. For a pure per-column
        # offset pattern the row medians are already 0, so the result is flat.
        assert np.allclose(out.data, 0.0, atol=1e-9)

    def test_removes_per_row_offsets(self) -> None:
        """A pure per-row offset pattern is flattened to zero."""
        row_offsets = np.arange(30, dtype=float) * 5.0
        data = np.zeros((30, 25)) + row_offsets[:, np.newaxis]
        img = _make_image(data)

        out = remove_column_and_row_medians(img)

        assert np.allclose(out.data, 0.0, atol=1e-9)

    def test_removes_combined_row_and_column_pattern(self) -> None:
        """A separable additive row+column pattern is fully removed."""
        rows = np.arange(32, dtype=float)[:, np.newaxis] * 2.0
        cols = np.arange(28, dtype=float)[np.newaxis, :] * 4.0
        data = 100.0 + rows + cols
        img = _make_image(data)

        out = remove_column_and_row_medians(img)

        # A separable additive pattern is fully removed by col-then-row median.
        assert np.allclose(out.data, 0.0, atol=1e-9)

    def test_records_processing_history(self) -> None:
        """Both column- and row-median steps are logged in processing history."""
        img = _make_image(RNG.normal(100, 5, (20, 20)))
        out = remove_column_and_row_medians(img)
        steps = [m.step_type for m in out.processing_history]
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT in steps
        assert ProcessingStep.ROW_MEDIAN_SUBTRACT in steps

    def test_promotes_to_float_allowing_negatives(self) -> None:
        """Integer input is promoted to float64 so subtraction can go negative."""
        data = np.full((16, 16), 10, dtype=np.int16)
        data[0, 0] = 0  # one low pixel so a column/row median > min exists
        img = _make_image(data, header=fits.Header())
        out = remove_column_and_row_medians(img)
        # float64: the detection/WCS chain was validated end-to-end at double
        # precision (see preprocessing.remove_column_and_row_medians).
        assert out.data.dtype == np.float64
        # Median-subtracted data must contain negatives somewhere.
        assert out.data.min() < 0

    def test_store_intermediates_keeps_correction_frames(self) -> None:
        """store_intermediates stores the per-axis median frames with right shapes."""
        data = RNG.normal(50, 3, (24, 24))
        # NOTE: remove_column_and_row_medians indexes correction_frames without
        # initializing it from None (unlike apply_dark_subtraction), so a caller
        # must supply a dict. See module docstring / final report.
        img = _make_image(data, correction_frames={})
        out = remove_column_and_row_medians(img, store_intermediates=True)
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT in out.correction_frames
        assert ProcessingStep.ROW_MEDIAN_SUBTRACT in out.correction_frames
        assert out.correction_frames[ProcessingStep.COLUMN_MEDIAN_SUBTRACT].shape == (1, 24)
        assert out.correction_frames[ProcessingStep.ROW_MEDIAN_SUBTRACT].shape == (24, 1)


# --- background subtraction ---------------------------------------------------


class TestBackground:
    """Tests for measure_background and remove_background on smooth gradients."""

    def _gradient(self, h: int = 128, w: int = 128, amp: float = 500.0) -> np.ndarray:
        """Build a smooth bilinear gradient field.

        Args:
            h: Image height in pixels.
            w: Image width in pixels.
            amp: Peak amplitude added along each axis.

        Returns:
            An (h, w) float array ramping in both x and y.
        """
        yy, xx = np.mgrid[0:h, 0:w]
        return amp * (xx / w) + amp * (yy / h)

    def test_measure_background_tracks_smooth_gradient(self) -> None:
        """The estimated background mesh tracks a smooth gradient in the interior."""
        grad = self._gradient()
        bg = measure_background(grad, box_size=16, filter_size=3)
        assert bg.shape == grad.shape
        # Interior of the mesh should track the gradient well (edge mesh boxes
        # extrapolate, so compare away from the borders).
        interior = slice(20, 108)
        assert np.allclose(bg[interior, interior], grad[interior, interior], atol=20.0)

    def test_remove_background_flattens_gradient(self) -> None:
        """Background subtraction collapses the gradient spread and floors at zero."""
        grad = self._gradient() + 1000.0
        original_spread = grad.std()
        img = _make_image(grad)
        out = remove_background(img, box_size=16, filter_size=3)
        # Subtraction should collapse most of the ~140-count gradient spread.
        assert out.data.std() < 0.25 * original_spread
        # remove_background floors the image at zero.
        assert out.data.min() == pytest.approx(0.0, abs=1e-6)

    def test_remove_background_preserves_point_source(self) -> None:
        """A bright compact source survives background removal."""
        grad = self._gradient(amp=300.0) + 1000.0
        # Inject a bright compact source; it must survive background removal.
        grad[64, 64] += 5000.0
        img = _make_image(grad.copy())
        out = remove_background(img, box_size=16, filter_size=3)
        peak = out.data[64, 64]
        local_med = np.median(out.data[58:70, 58:70])
        assert peak - local_med > 3000.0

    def test_remove_background_records_history(self) -> None:
        """The background-subtract step is logged in processing history."""
        img = _make_image(self._gradient() + 200.0)
        out = remove_background(img, box_size=16)
        assert any(m.step_type == ProcessingStep.BACKGROUND_SUBTRACT for m in out.processing_history)

    def test_store_intermediates_keeps_background_frame(self) -> None:
        """store_intermediates stores the background frame at full image shape."""
        # correction_frames must be a dict (None is not auto-initialized here).
        img = _make_image(self._gradient() + 200.0, correction_frames={})
        out = remove_background(img, box_size=16, store_intermediates=True)
        assert ProcessingStep.BACKGROUND_SUBTRACT in out.correction_frames
        assert out.correction_frames[ProcessingStep.BACKGROUND_SUBTRACT].shape == img.data.shape

    def test_store_intermediates_initializes_none_frames(self) -> None:
        """remove_background initializes correction_frames from None before storing."""
        # remove_background initializes correction_frames from None (same
        # guard as apply_dark_subtraction) before storing the background.
        img = _make_image(self._gradient() + 200.0, correction_frames=None)
        out = remove_background(img, box_size=16, store_intermediates=True)
        assert out.correction_frames is not None
        assert ProcessingStep.BACKGROUND_SUBTRACT in out.correction_frames


# --- image scaling: block_median ---------------------------------------------


class TestScaleBlockMedian:
    """Tests for scale_image_block_median (median-pooled downsampling)."""

    def test_downsamples_dimensions(self) -> None:
        """Block-median scaling downsamples data and metadata dimensions."""
        data = RNG.normal(100, 5, (100, 80))
        img = _make_image(data)
        out = scale_image_block_median(img, scale_factor=2.0)
        assert out.data.shape == (50, 40)
        assert out.metadata.width == 40
        assert out.metadata.height == 50

    def test_block_median_rejects_hot_pixel(self) -> None:
        """The per-block median rejects a single hot pixel in a flat field."""
        # A flat field with a single hot pixel inside a 2x2 block: the median
        # of the block ignores the outlier, so the scaled pixel stays at the
        # flat level (this is the documented hot-pixel-removal property).
        data = np.full((4, 4), 10.0)
        data[0, 0] = 100000.0
        img = _make_image(data)
        out = scale_image_block_median(img, scale_factor=2.0)
        assert out.data[0, 0] == pytest.approx(10.0)

    def test_records_scale_factor_metadata(self) -> None:
        """The processing step records the block_median method and scale factor."""
        img = _make_image(RNG.normal(100, 5, (40, 40)))
        out = scale_image_block_median(img, scale_factor=2.0)
        last = out.processing_history[-1]
        assert last.parameters["method"] == "block_median"
        assert last.parameters["scale_factor"] == 2.0

    def test_handles_non_divisible_dimensions(self) -> None:
        """Non-divisible dimensions are padded and yield the floored target shape."""
        # 101x83 is not divisible by the block size; padding must let it run
        # and still produce the floor(dim/scale) target shape.
        img = _make_image(RNG.normal(100, 5, (101, 83)))
        out = scale_image_block_median(img, scale_factor=2.0)
        assert out.data.shape == (50, 41)


# --- image scaling: blur_decimate --------------------------------------------


class TestScaleBlurDecimate:
    """Tests for scale_image_blur_decimate (Gaussian-blur then decimate)."""

    def test_downsamples_dimensions(self) -> None:
        """Blur-decimate scaling downsamples the image to the target shape."""
        img = _make_image(RNG.normal(100, 5, (120, 90)))
        out = scale_image_blur_decimate(img, scale_factor=3.0)
        assert out.data.shape == (40, 30)

    def test_preserves_mean_level(self) -> None:
        """Blur-decimate approximately conserves the overall flux level."""
        # Blur + decimate should approximately conserve the overall flux level
        # of a smooth field (no clipping, no flooring).
        data = np.full((60, 60), 250.0) + RNG.normal(0, 1, (60, 60))
        img = _make_image(data)
        out = scale_image_blur_decimate(img, scale_factor=2.0)
        assert out.data.mean() == pytest.approx(250.0, abs=2.0)

    def test_records_sigma_metadata(self) -> None:
        """The processing step records the blur_decimate method and blur sigma."""
        img = _make_image(RNG.normal(100, 5, (40, 40)))
        out = scale_image_blur_decimate(img, scale_factor=4.0)
        last = out.processing_history[-1]
        assert last.parameters["method"] == "blur_decimate"
        assert last.parameters["sigma"] == pytest.approx(2.0)


# --- FWHM-driven scaling decision --------------------------------------------


class TestScaleToTargetFWHM:
    """Tests for scale_image_to_target_fwhm (FWHM-driven scaling decision)."""

    def test_no_scaling_when_under_threshold(self) -> None:
        """No scaling is applied when the FWHM is below the oversample threshold."""
        img = _make_image(RNG.normal(100, 5, (50, 50)))
        out, factor = scale_image_to_target_fwhm(
            img, _fwhm_stats(3.0), target_fwhm=3.0, oversample_threshold=4.0
        )
        assert factor == 1.0
        assert out.data.shape == (50, 50)

    def test_scales_when_oversampled(self) -> None:
        """An oversampled image is scaled by median_fwhm / target_fwhm."""
        img = _make_image(RNG.normal(100, 5, (100, 100)))
        out, factor = scale_image_to_target_fwhm(
            img, _fwhm_stats(9.0), target_fwhm=3.0, method="block_median", oversample_threshold=4.0
        )
        # 9.0 / 3.0 = 3.0
        assert factor == pytest.approx(3.0)
        assert out.data.shape[0] < 100

    def test_blur_decimate_method_path(self) -> None:
        """The blur_decimate method path scales by the expected factor."""
        img = _make_image(RNG.normal(100, 5, (100, 100)))
        out, factor = scale_image_to_target_fwhm(
            img, _fwhm_stats(6.0), target_fwhm=3.0, method="blur_decimate", oversample_threshold=4.0
        )
        assert factor == pytest.approx(2.0)
        assert out.data.shape == (50, 50)

    def test_unknown_method_raises(self) -> None:
        """An unknown scaling method raises ValueError."""
        img = _make_image(RNG.normal(100, 5, (50, 50)))
        with pytest.raises(ValueError):
            scale_image_to_target_fwhm(
                img, _fwhm_stats(9.0), target_fwhm=3.0, method="nope", oversample_threshold=4.0
            )


# --- full preprocess_image pipeline ------------------------------------------


class TestPreprocessImage:
    """Tests for the config-driven preprocess_image pipeline."""

    def _config_image(self) -> ProcessedFitsImage:
        """Build a synthetic image with a gradient plus a fixed-pattern offset.

        Returns:
            A ProcessedFitsImage carrying a bilinear gradient, a per-column
            fixed-pattern offset, and a minimal header with EXPTIME set.
        """
        yy, xx = np.mgrid[0:96, 0:96]
        grad = 200.0 * (xx / 96.0) + 200.0 * (yy / 96.0) + 500.0
        # add a per-column fixed-pattern offset
        grad = grad + (np.arange(96, dtype=float)[np.newaxis, :] % 7) * 20.0
        header = fits.Header()
        header["BITPIX"] = -64
        header["EXPTIME"] = 5.0
        return _make_image(grad, header)

    def test_applies_configured_steps(self) -> None:
        """Enabled calibration steps each run and are logged in history."""
        cfg = get_config()
        cfg.calibrations.auto_remove_row_median = True
        cfg.calibrations.auto_remove_column_median = True
        cfg.calibrations.auto_subtract_background = True
        cfg.calibrations.auto_apply_darks = False
        cfg.calibrations.auto_apply_flats = False

        img = self._config_image()
        out = preprocess_image(img, config=cfg)
        steps = {m.step_type for m in out.processing_history}
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT in steps
        assert ProcessingStep.ROW_MEDIAN_SUBTRACT in steps
        assert ProcessingStep.BACKGROUND_SUBTRACT in steps

    def test_idempotent_does_not_double_apply(self) -> None:
        """An already-applied step is skipped on a second preprocess pass."""
        cfg = get_config()
        cfg.calibrations.auto_remove_row_median = True
        cfg.calibrations.auto_remove_column_median = True
        cfg.calibrations.auto_subtract_background = False
        cfg.calibrations.auto_apply_darks = False
        cfg.calibrations.auto_apply_flats = False

        img = self._config_image()
        once = preprocess_image(img, config=cfg)
        n_col = sum(m.step_type == ProcessingStep.COLUMN_MEDIAN_SUBTRACT for m in once.processing_history)
        twice = preprocess_image(once, config=cfg)
        n_col_after = sum(m.step_type == ProcessingStep.COLUMN_MEDIAN_SUBTRACT for m in twice.processing_history)
        # Already-applied step is skipped on the second pass.
        assert n_col == 1
        assert n_col_after == 1

    def test_disabled_steps_are_skipped(self) -> None:
        """Disabled calibration steps leave no trace in processing history."""
        cfg = get_config()
        cfg.calibrations.auto_remove_row_median = False
        cfg.calibrations.auto_remove_column_median = False
        cfg.calibrations.auto_subtract_background = False
        cfg.calibrations.auto_apply_darks = False
        cfg.calibrations.auto_apply_flats = False

        img = self._config_image()
        out = preprocess_image(img, config=cfg)
        steps = {m.step_type for m in out.processing_history}
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT not in steps
        assert ProcessingStep.BACKGROUND_SUBTRACT not in steps


# --- gain estimate from sky shot noise (photon transfer) ---------------------


def _poisson_sky(
    sky_adu: float,
    gain: float,
    shape: tuple[int, int] = (1500, 1500),
    seed: int = 0,
) -> np.ndarray:
    """Flat-fielded sky frame whose shot noise encodes a known gain (e-/ADU).

    sky electrons ~ Poisson(sky_adu * gain); ADU = electrons / gain, so
    var(ADU) = sky_adu / gain and the true gain is recoverable as
    sky_adu / var(ADU).

    Args:
        sky_adu: Mean sky level in ADU.
        gain: True detector gain in electrons per ADU to encode in the noise.
        shape: Output frame shape (height, width).
        seed: Seed for the frame's Poisson RNG (independent per frame).

    Returns:
        An ADU frame whose per-pixel variance encodes the requested gain.
    """
    rng = np.random.default_rng(seed)
    electrons = rng.poisson(sky_adu * gain, size=shape).astype(np.float64)
    return electrons / gain


class TestEstimateGainFromSky:
    """Tests for estimate_gain_from_sky (photon-transfer gain recovery)."""

    def test_recovers_known_gain(self) -> None:
        """The estimator recovers the true gain across a range of values."""
        for gain in (0.5, 1.0, 2.5):
            adu = _poisson_sky(1200.0, gain)
            est = estimate_gain_from_sky(adu, float(np.median(adu)))
            assert est == pytest.approx(gain, rel=0.10)

    def test_robust_to_gradient_and_stars(self) -> None:
        """Gain recovery is robust to a smooth gradient and bright stars."""
        gain = 1.6
        adu = _poisson_sky(1200.0, gain)
        # Smooth gradient (adjacent-column differencing should cancel it) and
        # bright stars (MAD should reject them).
        _yy, xx = np.mgrid[0:adu.shape[0], 0:adu.shape[1]]
        adu = adu + 0.003 * 1200.0 * (xx / adu.shape[1])
        adu[::150, ::150] += 40000.0
        est = estimate_gain_from_sky(adu, float(np.median(adu)))
        assert est == pytest.approx(gain, rel=0.12)

    def test_degenerate_inputs_return_none(self) -> None:
        """Degenerate inputs (no signal, no noise, no median) return None."""
        assert estimate_gain_from_sky(np.zeros((50, 50)), 0.0) is None
        assert estimate_gain_from_sky(np.ones((20, 20)), 5.0) is None  # no noise
        assert estimate_gain_from_sky(np.zeros((50, 50)), None) is None

    def test_metadata_carries_gain(self) -> None:
        """remove_column_and_row_medians records the measured gain alongside sky."""
        img = _make_image(_poisson_sky(1200.0, 2.0))
        out = remove_column_and_row_medians(img)
        col_step = next(m for m in out.processing_history
                        if m.step_type == ProcessingStep.COLUMN_MEDIAN_SUBTRACT)
        g = col_step.parameters["gain_e_per_adu"]
        assert g == pytest.approx(2.0, rel=0.10)
