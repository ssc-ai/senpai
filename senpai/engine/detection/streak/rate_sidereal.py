"""Solve the pixel shift between a sidereal frame and a rate-tracked frame.

Cross correlates a sidereal frame against a rate-tracked frame, extracts the streak
geometry, iteratively refines the shift with Bayesian flux validation, and records
the resulting shift and streak metadata.
"""

import logging
from pathlib import Path

import numpy as np

from senpai.engine.detection.kernels import rectangle_pyramoid
from senpai.engine.detection.streak.extraction import (
    cross_corr,
    extract_streak_dims_robust,
    mask_streak_region,
    measure_psf_shift,
    prepare_rate_frame,
    prepare_sidereal_frame,
    streak_parameters_from_xcorr,
)
from senpai.engine.detection.streak.masking import (
    remove_border_crossing_streaks,
    remove_n_brightest_streaks,
    remove_near_saturation_streaks,
)
from senpai.engine.detection.streak.rate_rate import bayesian_optimize_proposed_shift
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.senpai import FrameShift, RateTrackFrame, SiderealFrame
from senpai.engine.plotting.images import plot_single_frame
from senpai.exceptions import WcsPropagationError
from senpai.settings import settings

logger = logging.getLogger(__name__)


def solve_rate_from_sidereal(
    sidereal_frame: SiderealFrame, rate_frame: RateTrackFrame, frame_shift: FrameShift
) -> None:
    """Measure the pixel registration shift between a sidereal frame and a rate-tracked frame.

    Cross correlates the prepared frames, estimates the streak rotation and length from
    both the correlation peak and the rate frame, then iteratively refines the shift
    using Bayesian flux validation (masking the current peak between trials). The best
    shift (the WCS anchor) is written to the frame shift, and the streak geometry is stored
    on the rate frame. All updates are performed in place. The object's pixel track rate is
    NOT derived here -- see :func:`rate_rate._initial_track_rate`.

    Args:
        sidereal_frame: The reference sidereal frame.
        rate_frame: The rate-tracked frame to align with the sidereal frame.
        frame_shift: The frame shift to populate with the measured shift in place.

    Returns:
        None.

    Raises:
        WcsPropagationError: if the frame timing is degenerate (overlapping exposures) or the
            cross-correlation has no measurable streak -- no WCS can be propagated either way.
    """
    rate_exposure_time = float(rate_frame.frame.header.get("EXPTIME", 1))
    sidereal_exposure_time = float(sidereal_frame.frame.header.get("EXPTIME", 1))
    inter_frame_gap_seconds = abs(
        (sidereal_frame.timestamp - rate_frame.timestamp).total_seconds()
    ) - 0.5 * (rate_exposure_time + sidereal_exposure_time)

    # This sidereal->rate shift is the anchor that carries the solved sidereal WCS into the
    # collect. A non-positive gap+0.5*rate_exposure means the sidereal and rate exposures overlap
    # in time (degenerate/corrupt frame timing); the registration cannot be trusted, so fail fast
    # with a meaningful error rather than propagating a garbage anchor.
    elapsed_seconds = inter_frame_gap_seconds + 0.5 * rate_exposure_time
    if elapsed_seconds <= 0:
        raise WcsPropagationError(
            f"Cannot register sidereal->rate anchor shift {frame_shift.source_index}->"
            f"{frame_shift.target_index}: time from the rate-frame exposure midpoint to the "
            f"sidereal frame ({elapsed_seconds:.3f} s) is not positive, implying overlapping "
            "exposures (degenerate frame timing); no WCS can be propagated for this collect."
        )

    pixel_fwhm = sidereal_frame.starfield.detection_metadata.pixel_fwhm

    sidereal_data, is_synthetic = prepare_sidereal_frame(sidereal_frame, allow_synthetic=False)
    rate_data = prepare_rate_frame(rate_frame)

    # whopping bright streaks can mess with correlation
    if not is_synthetic:
        rate_data, removed_streaks = remove_near_saturation_streaks(
            rate_data, rate_frame.frame.data_type
        )
        sidereal_data, removed_streaks = remove_n_brightest_streaks(sidereal_data, removed_streaks)

    rate_data = remove_border_crossing_streaks(rate_data)
    sidereal_data = remove_border_crossing_streaks(sidereal_data)
    rate_data, removed_streaks = remove_near_saturation_streaks(
        rate_data, rate_frame.frame.data_type
    )

    logger.info("Cross correlating rate and sidereal frames")

    # fast fourier-based cross correlation
    cross_correlated_image = cross_corr(rate_data, sidereal_data)

    valid = False
    max_trials = 3
    trials = 0
    best_shift = None
    best_correlation = None

    while not valid and trials < max_trials:
        trials += 1

        if settings.plotting.debug:
            plot_single_frame(
                cross_correlated_image,
                output_file=Path(settings.plotting.output_dir)
                / f"sidereal_to_rate_cc_{sidereal_frame.index}-{rate_frame.index}-{trials}.png",
                scale=True,
            )
        # extract rotation and streak length from cc, assuming alignment feature is brightest
        streak_parameters = streak_parameters_from_xcorr(
            cross_correlated_image,
            plate_scale_arcsec=sidereal_frame.starfield.wcs_metadata.x_ifov_arcsec,
            seeing_fwhm_pixels=pixel_fwhm,
        )
        if streak_parameters is None:
            # The cross-correlation has no extended cluster, so the frame-to-frame streak (and
            # therefore this anchor shift) cannot be measured and the WCS cannot be propagated.
            raise WcsPropagationError(
                f"Cannot register sidereal->rate anchor shift {frame_shift.source_index}->"
                f"{frame_shift.target_index}: the sidereal/rate cross-correlation contains no "
                "measurable streak, so a WCS solution cannot be propagated for this collect."
            )
        rotation_estimate_1, length_estimate_1, subcc = streak_parameters

        rotation_estimate_2, length_estimate_2, psf, fwhm = extract_streak_dims_robust(
            rate_data,
            n_streaks=7,
            rotation=rotation_estimate_1,
            length=length_estimate_1,
            fwhm=pixel_fwhm,
        )

        if settings.plotting.debug:
            plot_single_frame(
                psf,
                scale=False,
                output_file=Path(settings.plotting.output_dir)
                / f"{rate_frame.index}_streak_psf.png",
            )

        pixel_shift_rate_to_sidereal_xy = measure_psf_shift(
            subcc, length_estimate_2, rotation_estimate_2, pixel_fwhm
        )[::-1]

        # Validate this proposal
        search_radius_pixels = 10.0
        pixel_shift_rate_to_sidereal_xy[0], pixel_shift_rate_to_sidereal_xy[1], correlation = (
            bayesian_optimize_proposed_shift(
                target=rate_frame,
                source=sidereal_frame,
                shift_x_low=pixel_shift_rate_to_sidereal_xy[0] - search_radius_pixels,
                shift_x_high=pixel_shift_rate_to_sidereal_xy[0] + search_radius_pixels,
                shift_y_low=pixel_shift_rate_to_sidereal_xy[1] - search_radius_pixels,
                shift_y_high=pixel_shift_rate_to_sidereal_xy[1] + search_radius_pixels,
                catalog_stars=sidereal_frame.starfield.catalog_stars,
                n_calls=10,
            )
        )

        # Check if this is better than what we had previously
        if not best_correlation or correlation > best_correlation:
            best_correlation = correlation
            best_shift = pixel_shift_rate_to_sidereal_xy

        valid = correlation > 0.9
        if not valid:
            y_max, x_max = np.unravel_index(
                np.argmax(cross_correlated_image), cross_correlated_image.shape
            )
            mask = np.zeros_like(cross_correlated_image, dtype=bool)
            mask_kernel = rectangle_pyramoid(
                length_estimate_2 * 1.2,
                np.sin(np.deg2rad(rotation_estimate_2)),
                np.cos(np.deg2rad(rotation_estimate_2)),
                int(fwhm * 2.2),
                halo_fwhm=4,
            )

            mask, cross_correlated_image = mask_streak_region(
                mask, cross_correlated_image, y_max, x_max, mask_kernel
            )

    # Log the quality of this fit
    if best_correlation > 0.9:
        # Use the best shift we managed to find
        logger.info(
            f"Fit frame shift from frame {sidereal_frame.index} to frame {rate_frame.index} with {best_correlation} correlation"
        )
        pixel_shift_rate_to_sidereal_xy = best_shift
    else:
        # Use the best shift we managed to find
        logger.warning(
            f"Poor fit ({best_correlation} correlation) for frame shift from frame {sidereal_frame.index} to frame {rate_frame.index}"
        )
        pixel_shift_rate_to_sidereal_xy = best_shift

    pixel_shift_rate_to_sidereal = np.linalg.norm(pixel_shift_rate_to_sidereal_xy)
    logger.info(f"Sidereal->rate registration shift: {pixel_shift_rate_to_sidereal:.1f} pixels")

    # Record the streak geometry measured off the rate frame. The pixel track rate is NOT derived
    # here from the offset/elapsed: that offset is a star-field registration shift spanning a
    # sidereal->rate slew-mode change, so dividing it by an exposure time under-estimates the
    # object's rate. The rate->rate solver instead seeds the first window from the mount track-rate
    # header (or this streak length); see rate_rate._initial_track_rate.
    rate_frame.streak = StreakMetadata(
        pixel_length=length_estimate_2,
        sine_angle=np.sin(np.deg2rad(rotation_estimate_2)),
        cosine_angle=np.cos(np.deg2rad(rotation_estimate_2)),
        fwhm=fwhm,
    )

    frame_shift.x_shift = pixel_shift_rate_to_sidereal_xy[0]
    frame_shift.y_shift = pixel_shift_rate_to_sidereal_xy[1]
    frame_shift.is_valid = True
    frame_shift.processed = True
    frame_shift.correlation = best_correlation
    return
