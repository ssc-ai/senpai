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

from senpai.core.config import AppConfig, initialize_config, load_yaml, settings
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
    """Initialise the process-wide config from a bundled YAML.

    Reading ``settings`` raises until something has been loaded, and every module under test
    here reads it.
    """
    initialize_config(CONFIG_DIR / "burr.yaml")
    settings.plotting.debug = False


def _make_image(
    data: np.ndarray,
    header: fits.Header | None = None,
    correction_frames: dict | None = None,
) -> ProcessedFitsImage:
    """Wrap an array as a ProcessedFitsImage the preprocessing steps can consume."""
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
    """FWHM metadata carrying a chosen median, for the scaling decisions that read it."""
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
    """Removing per-column and per-row fixed-pattern offsets."""

    def test_removes_per_column_offsets(self) -> None:
        # Each column has a distinct constant offset; after column-median
        # removal every column median should be ~0.
        """A per-column offset pattern is removed."""
        base = np.zeros((40, 40))
        col_offsets = np.arange(40, dtype=float) * 3.0
        data = base + col_offsets[np.newaxis, :]
        img = _make_image(data)

        out = remove_column_and_row_medians(img)

        # Column medians removed first, then row medians. For a pure per-column
        # offset pattern the row medians are already 0, so the result is flat.
        assert np.allclose(out.data, 0.0, atol=1e-9)

    def test_removes_per_row_offsets(self) -> None:
        """A per-row offset pattern is removed."""
        row_offsets = np.arange(30, dtype=float) * 5.0
        data = np.zeros((30, 25)) + row_offsets[:, np.newaxis]
        img = _make_image(data)

        out = remove_column_and_row_medians(img)

        assert np.allclose(out.data, 0.0, atol=1e-9)

    def test_removes_combined_row_and_column_pattern(self) -> None:
        """Row and column patterns present together are both removed."""
        rows = np.arange(32, dtype=float)[:, np.newaxis] * 2.0
        cols = np.arange(28, dtype=float)[np.newaxis, :] * 4.0
        data = 100.0 + rows + cols
        img = _make_image(data)

        out = remove_column_and_row_medians(img)

        # A separable additive pattern is fully removed by col-then-row median.
        assert np.allclose(out.data, 0.0, atol=1e-9)

    def test_records_processing_history(self) -> None:
        """The step records itself in the frame's processing history.

        That record is what stops a re-entered pipeline applying it twice.
        """
        img = _make_image(RNG.normal(100, 5, (20, 20)))
        out = remove_column_and_row_medians(img)
        steps = [m.step_type for m in out.processing_history]
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT in steps
        assert ProcessingStep.ROW_MEDIAN_SUBTRACT in steps

    def test_promotes_to_float_allowing_negatives(self) -> None:
        """The frame is promoted to float so the subtraction can go negative.

        On an integer frame the residual would clip at zero and the pattern would only half come out.
        """
        data = np.full((16, 16), 10, dtype=np.int16)
        data[0, 0] = 0  # one low pixel so a column/row median > min exists
        img = _make_image(data, header=fits.Header())
        out = remove_column_and_row_medians(img)
        # float32 by design: the whole downstream pipeline inherits this
        # dtype and float64 doubles its cost for ADU-scale data.
        assert out.data.dtype == np.float32
        # Median-subtracted data must contain negatives somewhere.
        assert out.data.min() < 0

    def test_dtype_override_promotes_to_float64(self) -> None:
        """An explicit dtype overrides the default promotion."""
        data = np.full((16, 16), 10, dtype=np.int16)
        data[0, 0] = 0
        img = _make_image(data, header=fits.Header())
        out = remove_column_and_row_medians(img, dtype=np.float64)
        # Opt-in float64 for callers whose downstream math is sensitive to
        # float32's ADU-scale rounding (rate-track registration).
        assert out.data.dtype == np.float64
        assert out.data.min() < 0

    def test_store_intermediates_keeps_correction_frames(self) -> None:
        """With intermediates requested, the correction actually applied is retained."""
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
    """Measuring and subtracting a smooth 2D background."""

    def _gradient(self, h: int = 128, w: int = 128, amp: float = 500.0) -> np.ndarray:
        """Build a smooth linear gradient, standing in for a sky background."""
        yy, xx = np.mgrid[0:h, 0:w]
        return amp * (xx / w) + amp * (yy / h)

    def test_measure_background_tracks_smooth_gradient(self) -> None:
        """The measured background follows a smooth gradient rather than flattening it to a constant."""
        grad = self._gradient()
        bg = measure_background(grad, box_size=16, filter_size=3)
        assert bg.shape == grad.shape
        # Interior of the mesh should track the gradient well (edge mesh boxes
        # extrapolate, so compare away from the borders).
        interior = slice(20, 108)
        assert np.allclose(bg[interior, interior], grad[interior, interior], atol=20.0)

    def test_remove_background_flattens_gradient(self) -> None:
        """Subtracting the measured background leaves a flat frame."""
        grad = self._gradient() + 1000.0
        original_spread = grad.std()
        img = _make_image(grad)
        out = remove_background(img, box_size=16, filter_size=3)
        # Subtraction should collapse most of the ~140-count gradient spread.
        assert out.data.std() < 0.25 * original_spread
        # remove_background floors the image at zero.
        assert out.data.min() == pytest.approx(0.0, abs=1e-6)

    def test_remove_background_preserves_point_source(self) -> None:
        """Background subtraction leaves a point source intact.

        A mesh fine enough to follow the sky must not follow the stars, or it subtracts the signal
        the detector is looking for.
        """
        grad = self._gradient(amp=300.0) + 1000.0
        # Inject a bright compact source; it must survive background removal.
        grad[64, 64] += 5000.0
        img = _make_image(grad.copy())
        out = remove_background(img, box_size=16, filter_size=3)
        peak = out.data[64, 64]
        local_med = np.median(out.data[58:70, 58:70])
        assert peak - local_med > 3000.0

    def test_remove_background_records_history(self) -> None:
        """The step records itself in the processing history."""
        img = _make_image(self._gradient() + 200.0)
        out = remove_background(img, box_size=16)
        assert any(m.step_type == ProcessingStep.BACKGROUND_SUBTRACT for m in out.processing_history)

    def test_store_intermediates_keeps_background_frame(self) -> None:
        # correction_frames must be a dict (None is not auto-initialized here).
        """With intermediates requested, the background model itself is retained."""
        img = _make_image(self._gradient() + 200.0, correction_frames={})
        out = remove_background(img, box_size=16, store_intermediates=True)
        assert ProcessingStep.BACKGROUND_SUBTRACT in out.correction_frames
        assert out.correction_frames[ProcessingStep.BACKGROUND_SUBTRACT].shape == img.data.shape

    def test_store_intermediates_initializes_none_frames(self) -> None:
        # remove_background initializes correction_frames from None (same
        # guard as apply_dark_subtraction) before storing the background.
        """A frame with no correction-frame dict gets one created rather than raising."""
        img = _make_image(self._gradient() + 200.0, correction_frames=None)
        out = remove_background(img, box_size=16, store_intermediates=True)
        assert out.correction_frames is not None
        assert ProcessingStep.BACKGROUND_SUBTRACT in out.correction_frames


# --- image scaling: block_median ---------------------------------------------


class TestScaleBlockMedian:
    """Downsampling by block median."""

    def test_downsamples_dimensions(self) -> None:
        """Block-median scaling reduces the frame dimensions by the factor."""
        data = RNG.normal(100, 5, (100, 80))
        img = _make_image(data)
        out = scale_image_block_median(img, scale_factor=2.0)
        assert out.data.shape == (50, 40)
        assert out.metadata.width == 40
        assert out.metadata.height == 50

    def test_block_median_rejects_hot_pixel(self) -> None:
        # A flat field with a single hot pixel inside a 2x2 block: the median
        # of the block ignores the outlier, so the scaled pixel stays at the
        # flat level (this is the documented hot-pixel-removal property).
        """A hot pixel does not survive a block median.

        This is the reason for choosing median over mean: a mean would spread the hot pixel across
        its whole block instead of discarding it.
        """
        data = np.full((4, 4), 10.0)
        data[0, 0] = 100000.0
        img = _make_image(data)
        out = scale_image_block_median(img, scale_factor=2.0)
        assert out.data[0, 0] == pytest.approx(10.0)

    def test_records_scale_factor_metadata(self) -> None:
        """The applied scale factor is recorded on the frame."""
        img = _make_image(RNG.normal(100, 5, (40, 40)))
        out = scale_image_block_median(img, scale_factor=2.0)
        last = out.processing_history[-1]
        assert last.parameters["method"] == "block_median"
        assert last.parameters["scale_factor"] == 2.0

    def test_handles_non_divisible_dimensions(self) -> None:
        # 101x83 is not divisible by the block size; padding must let it run
        # and still produce the floor(dim/scale) target shape.
        """Dimensions that do not divide evenly are handled by dropping the partial blocks."""
        img = _make_image(RNG.normal(100, 5, (101, 83)))
        out = scale_image_block_median(img, scale_factor=2.0)
        assert out.data.shape == (50, 41)


# --- image scaling: blur_decimate --------------------------------------------


class TestScaleBlurDecimate:
    """Downsampling by blur-then-decimate."""

    def test_downsamples_dimensions(self) -> None:
        """Blur-then-decimate scaling reduces the frame dimensions by the factor."""
        img = _make_image(RNG.normal(100, 5, (120, 90)))
        out = scale_image_blur_decimate(img, scale_factor=3.0)
        assert out.data.shape == (40, 30)

    def test_preserves_mean_level(self) -> None:
        # Blur + decimate should approximately conserve the overall flux level
        # of a smooth field (no clipping, no flooring).
        """Blur-then-decimate preserves the mean level, so photometry is not offset by scaling."""
        data = np.full((60, 60), 250.0) + RNG.normal(0, 1, (60, 60))
        img = _make_image(data)
        out = scale_image_blur_decimate(img, scale_factor=2.0)
        assert out.data.mean() == pytest.approx(250.0, abs=2.0)

    def test_records_sigma_metadata(self) -> None:
        """The blur sigma used is recorded on the frame."""
        img = _make_image(RNG.normal(100, 5, (40, 40)))
        out = scale_image_blur_decimate(img, scale_factor=4.0)
        last = out.processing_history[-1]
        assert last.parameters["method"] == "blur_decimate"
        assert last.parameters["sigma"] == pytest.approx(2.0)


# --- FWHM-driven scaling decision --------------------------------------------


class TestScaleToTargetFWHM:
    """Choosing whether and how to rescale a frame toward a target PSF width."""

    def test_no_scaling_when_under_threshold(self) -> None:
        """A frame that is not oversampled is left alone."""
        img = _make_image(RNG.normal(100, 5, (50, 50)))
        out, factor = scale_image_to_target_fwhm(img, _fwhm_stats(3.0), target_fwhm=3.0, oversample_threshold=4.0)
        assert factor == 1.0
        assert out.data.shape == (50, 50)

    def test_scales_when_oversampled(self) -> None:
        """An oversampled frame is scaled toward the target width."""
        img = _make_image(RNG.normal(100, 5, (100, 100)))
        out, factor = scale_image_to_target_fwhm(
            img, _fwhm_stats(9.0), target_fwhm=3.0, method="block_median", oversample_threshold=4.0
        )
        # 9.0 / 3.0 = 3.0
        assert factor == pytest.approx(3.0)
        assert out.data.shape[0] < 100

    def test_blur_decimate_method_path(self) -> None:
        """The blur-decimate method is selectable."""
        img = _make_image(RNG.normal(100, 5, (100, 100)))
        out, factor = scale_image_to_target_fwhm(
            img, _fwhm_stats(6.0), target_fwhm=3.0, method="blur_decimate", oversample_threshold=4.0
        )
        assert factor == pytest.approx(2.0)
        assert out.data.shape == (50, 50)

    def test_unknown_method_raises(self) -> None:
        """An unknown scaling method raises rather than silently skipping the scaling."""
        img = _make_image(RNG.normal(100, 5, (50, 50)))
        with pytest.raises(ValueError):
            scale_image_to_target_fwhm(img, _fwhm_stats(9.0), target_fwhm=3.0, method="nope", oversample_threshold=4.0)


# --- full preprocess_image pipeline ------------------------------------------


class TestPreprocessImage:
    """The full preprocessing pipeline, and which steps it runs."""

    def _config_image(self) -> ProcessedFitsImage:
        """Build a frame with both a smooth gradient and a per-column fixed-pattern offset.

        Carrying both means the row/column-median and background steps each have something to
        remove, so a test can tell which ones ran.
        """
        yy, xx = np.mgrid[0:96, 0:96]
        grad = 200.0 * (xx / 96.0) + 200.0 * (yy / 96.0) + 500.0
        # add a per-column fixed-pattern offset
        grad = grad + (np.arange(96, dtype=float)[np.newaxis, :] % 7) * 20.0
        header = fits.Header()
        header["BITPIX"] = -64
        header["EXPTIME"] = 5.0
        return _make_image(grad, header)

    def test_explicit_config_overrides_the_singleton(self) -> None:
        """A config passed in is used instead of the process-wide settings.

        The COCO export relies on this: it rebuilds a frame under the config the night run
        used, which is deliberately not the config the export process loaded.
        """
        settings.calibrations.auto_remove_row_median = True
        settings.calibrations.auto_remove_column_median = True
        settings.calibrations.auto_subtract_background = False
        settings.calibrations.auto_apply_darks = False
        settings.calibrations.auto_apply_flats = False

        injected = AppConfig(**load_yaml(CONFIG_DIR / "local.yaml"))
        injected.calibrations.auto_remove_row_median = False
        injected.calibrations.auto_remove_column_median = False
        injected.calibrations.auto_subtract_background = False
        injected.calibrations.auto_apply_darks = False
        injected.calibrations.auto_apply_flats = False

        out = preprocess_image(self._config_image(), config=injected)
        steps = {m.step_type for m in out.processing_history}
        assert ProcessingStep.ROW_MEDIAN_SUBTRACT not in steps
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT not in steps

    def test_applies_configured_steps(self) -> None:
        """Every step enabled in config runs."""
        settings.calibrations.auto_remove_row_median = True
        settings.calibrations.auto_remove_column_median = True
        settings.calibrations.auto_subtract_background = True
        settings.calibrations.auto_apply_darks = False
        settings.calibrations.auto_apply_flats = False

        img = self._config_image()
        out = preprocess_image(img)
        steps = {m.step_type for m in out.processing_history}
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT in steps
        assert ProcessingStep.ROW_MEDIAN_SUBTRACT in steps
        assert ProcessingStep.BACKGROUND_SUBTRACT in steps

    def test_idempotent_does_not_double_apply(self) -> None:
        """Running the pipeline twice does not apply any step twice.

        Subtracting the same dark or median a second time leaves a negative pedestal that every
        later estimate inherits.
        """
        settings.calibrations.auto_remove_row_median = True
        settings.calibrations.auto_remove_column_median = True
        settings.calibrations.auto_subtract_background = False
        settings.calibrations.auto_apply_darks = False
        settings.calibrations.auto_apply_flats = False

        img = self._config_image()
        once = preprocess_image(img)
        n_col = sum(m.step_type == ProcessingStep.COLUMN_MEDIAN_SUBTRACT for m in once.processing_history)
        twice = preprocess_image(once)
        n_col_after = sum(m.step_type == ProcessingStep.COLUMN_MEDIAN_SUBTRACT for m in twice.processing_history)
        # Already-applied step is skipped on the second pass.
        assert n_col == 1
        assert n_col_after == 1

    def test_disabled_steps_are_skipped(self) -> None:
        """A step disabled in config does not run."""
        settings.calibrations.auto_remove_row_median = False
        settings.calibrations.auto_remove_column_median = False
        settings.calibrations.auto_subtract_background = False
        settings.calibrations.auto_apply_darks = False
        settings.calibrations.auto_apply_flats = False

        img = self._config_image()
        out = preprocess_image(img)
        steps = {m.step_type for m in out.processing_history}
        assert ProcessingStep.COLUMN_MEDIAN_SUBTRACT not in steps
        assert ProcessingStep.BACKGROUND_SUBTRACT not in steps


# --- gain estimate from sky shot noise (photon transfer) ---------------------


def _poisson_sky(sky_adu: float, gain: float, shape: tuple[int, int] = (1500, 1500), seed: int = 0) -> np.ndarray:
    """Build a flat-fielded sky frame whose shot noise encodes a known gain in e-/ADU.

    sky electrons ~ Poisson(sky_adu * gain); ADU = electrons / gain, so
    var(ADU) = sky_adu / gain and the true gain is recoverable as
    sky_adu / var(ADU).
    """
    rng = np.random.default_rng(seed)
    electrons = rng.poisson(sky_adu * gain, size=shape).astype(np.float64)
    return electrons / gain


class TestEstimateGainFromSky:
    """Recovering detector gain from the shot noise of a flat sky."""

    def test_recovers_known_gain(self) -> None:
        """The estimator recovers a known gain from Poisson sky noise."""
        for gain in (0.5, 1.0, 2.5):
            adu = _poisson_sky(1200.0, gain)
            est = estimate_gain_from_sky(adu, float(np.median(adu)))
            assert est == pytest.approx(gain, rel=0.10)

    def test_robust_to_gradient_and_stars(self) -> None:
        """The estimate survives a gradient and stars in the frame.

        Neither is background, and both would bias a naive variance.
        """
        gain = 1.6
        adu = _poisson_sky(1200.0, gain)
        # Smooth gradient (adjacent-column differencing should cancel it) and
        # bright stars (MAD should reject them).
        yy, xx = np.mgrid[0 : adu.shape[0], 0 : adu.shape[1]]
        adu = adu + 0.003 * 1200.0 * (xx / adu.shape[1])
        adu[::150, ::150] += 40000.0
        est = estimate_gain_from_sky(adu, float(np.median(adu)))
        assert est == pytest.approx(gain, rel=0.12)

    def test_degenerate_inputs_return_none(self) -> None:
        """Degenerate input yields None rather than a meaningless gain."""
        assert estimate_gain_from_sky(np.zeros((50, 50)), 0.0) is None
        assert estimate_gain_from_sky(np.ones((20, 20)), 5.0) is None  # no noise
        assert estimate_gain_from_sky(np.zeros((50, 50)), None) is None

    def test_metadata_carries_gain(self) -> None:
        """remove_column_and_row_medians records the measured gain alongside sky."""
        img = _make_image(_poisson_sky(1200.0, 2.0))
        out = remove_column_and_row_medians(img)
        col_step = next(m for m in out.processing_history if m.step_type == ProcessingStep.COLUMN_MEDIAN_SUBTRACT)
        g = col_step.parameters["gain_e_per_adu"]
        assert g == pytest.approx(2.0, rel=0.10)
