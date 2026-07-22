"""Solve the pixel shift between two rate-tracked frames via cross correlation.

Balances and cross correlates a pair of rate-tracked frames, iteratively refines the
shift with Bayesian flux validation, and records the resulting shift, streak geometry,
and pixel track rate. Also exposes the shared Bayesian shift-optimization helper used
by the rate/sidereal solvers.
"""

import logging
from pathlib import Path

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from senpai.engine.detection.streak.extraction import (
    cross_corr,
    extract_streak_dims_robust,
    prepare_rate_frame,
)
from senpai.engine.detection.streak.masking import (
    percent_difference,
    remove_border_crossing_streaks,
    remove_brightest_streak,
    remove_near_saturation_streaks,
)
from senpai.engine.detection.streak.validation import validate_proposed_shift
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.senpai import FrameShift, RateTrackFrame, SiderealFrame
from senpai.engine.models.starfield import StarInSpace
from senpai.engine.plotting.images import plot_single_frame
from senpai.settings import settings

logger = logging.getLogger(__name__)


def strip_unbalanced_streaks(rate1_img: np.ndarray, rate2_img: np.ndarray) -> None:
    """Iteratively remove the brightest streak from whichever frame is brighter.

    While the percent difference between the two frames' near-peak (99.99th percentile)
    values exceeds the threshold, the brightest streak in the brighter frame is filled in,
    rebalancing the two frames so a subsequent cross correlation is not dominated by a
    single saturated streak. A bounded number of attempts is made.

    Args:
        rate1_img: The first rate-tracked frame's image data.
        rate2_img: The second rate-tracked frame's image data.

    Returns:
        None.
    """
    max_r1 = np.percentile(rate1_img, 99.99)
    max_r2 = np.percentile(rate2_img, 99.99)
    pdthresh = 20.0
    pd = percent_difference(max_r1, max_r2)

    max_attempts = 5
    attempts = 0
    while pd > pdthresh and attempts < max_attempts:
        attempts += 1
        logger.debug(f"percent difference {pd:.1f}% is greater than threshold of {pdthresh:.1f}%")

        if max_r1 > max_r2:
            logger.debug("removing near-saturation streak in rate1")
            fill_min = np.median(rate1_img) + 0.5 * np.std(rate1_img)
            rate1_img = remove_brightest_streak(rate1_img, fill_min)

        else:
            logger.debug("removing near-saturation streak in rate2")
            fill_min = np.median(rate2_img) + 0.5 * np.std(rate2_img)
            rate2_img = remove_brightest_streak(rate2_img, fill_min)

        max_r1 = np.percentile(rate1_img, 99.99)
        max_r2 = np.percentile(rate2_img, 99.99)
        pd = percent_difference(max_r1, max_r2)

    if pd <= pdthresh:
        logger.debug(f"percent difference {pd:.1f}% is below threshold of {pdthresh:.1f}%")
    else:
        logger.warning(f"percent difference {pd:.1f}% is above threshold of {pdthresh:.1f}%")


def windowed_correlation_peaks(
    cross_correlated_image: np.ndarray, expected_shift: float | None, n_peaks: int = 1
) -> list[np.ndarray]:
    """Locate up to ``n_peaks`` DISTINCT correlation peaks within the expected-shift window.

    Masks the central self-correlation spike and restricts the search to a window scaled to the
    expected pixel shift. Peaks are returned brightest-first; after each peak its neighborhood is
    suppressed so the next ``argmax`` finds a genuinely different feature. This lets the caller
    validate several candidates: a bright SPURIOUS peak (e.g. a long star-streak artifact) can
    outshine the true frame-to-frame shift, so taking only the brightest -- or re-finding the same
    cluster after masking a single pixel -- can lock onto the wrong peak (the 963e1bc5 / obj-28190
    failure). The mask/window are clamped positive so the search region is never empty (a
    non-positive seed otherwise produced an empty slice and crashed ``np.argmax``).

    Args:
        cross_correlated_image: The frame-to-frame cross-correlation image.
        expected_shift: Expected pixel-shift magnitude used to size the window, or None when no
            prior estimate is available.
        n_peaks: Maximum number of distinct peaks to return.

    Returns:
        A list of up to ``n_peaks`` (row, column) peak locations in the full image's coordinates,
        brightest first.
    """
    center = np.array(cross_correlated_image.shape) / 2
    if expected_shift and expected_shift > 0:
        mask_center = max(1, int(0.2 * expected_shift))
        scale = max(1, int(1.2 * expected_shift))
    else:
        mask_center = 3
        scale = 150

    subcc = cross_correlated_image.copy()
    subcc[
        int(center[0] - mask_center) : int(center[0] + mask_center),
        int(center[1] - mask_center) : int(center[1] + mask_center),
    ] = np.min(subcc)

    y_min = max(0, int(center[0] - scale))
    y_max = min(subcc.shape[0], int(center[0] + scale))
    x_min = max(0, int(center[1] - scale))
    x_max = min(subcc.shape[1], int(center[1] + scale))
    subcc = subcc[y_min:y_max, x_min:x_max]

    # Suppress a peak's whole neighborhood (a streak cluster) between picks, so candidates are
    # distinct features rather than adjacent pixels of one peak.
    suppress_radius = max(mask_center, 5)
    floor = float(np.min(subcc))
    work = subcc.copy()
    rows, cols = np.indices(work.shape)
    peaks: list[np.ndarray] = []
    for _ in range(max(1, n_peaks)):
        peak = np.unravel_index(np.argmax(work), work.shape)
        peaks.append(np.array([peak[0] + y_min, peak[1] + x_min]))
        work[(rows - peak[0]) ** 2 + (cols - peak[1]) ** 2 <= suppress_radius**2] = floor
    return peaks


def track_rate_from_header(
    header: object, plate_scale_arcsec: float, dec_deg: float
) -> float | None:
    """Derive the object's pixel track rate from the mount track-rate headers, if available.

    ``TELTKRA``/``TELTKDEC`` are the mount's commanded RA/Dec track rates (arcsec/s); the mount
    physically slewed at this rate, so it is the object's apparent rate. Returns ``None`` when
    either header is absent, unparseable, the plate scale is non-positive, or the derived rate is
    non-positive, so the caller can fall back.

    Args:
        header: FITS header (mapping) of the rate-tracked frame.
        plate_scale_arcsec: Plate scale in arcsec/pixel (from the solved WCS).
        dec_deg: Field declination in degrees, used to scale the RA rate by ``cos(dec)``.

    Returns:
        The pixel track rate in pixels per second, or ``None`` if it cannot be derived.
    """
    ra_rate = header.get("TELTKRA")
    dec_rate = header.get("TELTKDEC")
    if ra_rate is None or dec_rate is None or plate_scale_arcsec <= 0:
        return None
    try:
        great_circle_arcsec_s = float(
            np.hypot(float(ra_rate) * np.cos(np.deg2rad(dec_deg)), float(dec_rate))
        )
    except (TypeError, ValueError):
        return None
    rate = great_circle_arcsec_s / plate_scale_arcsec
    return rate if rate > 0 else None


def _initial_track_rate(rate_frame: RateTrackFrame) -> float | None:
    """Estimate the object's pixel rate to seed the first rate->rate search window.

    The first rate->rate pair has no measured rate yet. Rather than back the rate out of the
    sidereal->rate registration offset (a star-field shift across a slew-mode change, which
    under-estimates the object rate), measure it directly: prefer the mount track-rate header,
    then fall back to the rate frame's measured streak length over the exposure.

    Args:
        rate_frame: The (already solved) rate frame anchoring this pair.

    Returns:
        The object's pixel track rate in pixels per second, or ``None`` if neither source is
        available (then the search window falls back to its default size).
    """
    starfield = rate_frame.starfield
    if starfield is not None and starfield.wcs is not None and starfield.wcs_metadata is not None:
        astropy_wcs = starfield.wcs.to_astropy_wcs()
        if astropy_wcs is not None:
            rate = track_rate_from_header(
                rate_frame.frame.header,
                starfield.wcs_metadata.x_ifov_arcsec,
                float(astropy_wcs.wcs.crval[1]),
            )
            if rate is not None:
                return rate
    if rate_frame.streak is not None and rate_frame.streak.pixel_length:
        exposure = float(rate_frame.frame.header.get("EXPTIME", 1))
        if exposure > 0:
            return rate_frame.streak.pixel_length / exposure
    return None


def solve_rate_from_rate(
    rate_frame_a: RateTrackFrame, rate_frame_b: RateTrackFrame, frame_shift: FrameShift
) -> None:
    """Measure the pixel shift between two rate-tracked frames by cross correlation.

    Prepares and balances both frames, cross correlates them, and searches the
    correlation surface (optionally seeded by the expected shift from the frames' track
    rates) for the best peak, refining each candidate with Bayesian flux validation. The
    best shift is written to the frame shift, and, if the second frame has no streak
    metadata yet, its streak geometry and pixel track rate are measured and stored. All
    updates are performed in place.

    Args:
        rate_frame_a: The reference rate-tracked frame.
        rate_frame_b: The rate-tracked frame to align with the first.
        frame_shift: The frame shift to populate with the measured shift in place.

    Returns:
        None.
    """
    # The reference (source) frame must already have a WCS solution. If an upstream shift was
    # rejected, this frame was never solved, so there is nothing to register against. This is a
    # recoverable, routed-around condition (a symptom of an upstream failure), so warn and mark
    # the shift invalid rather than raising or dereferencing a missing starfield.
    if rate_frame_a.starfield is None:
        logger.warning(
            "Skipping rate->rate shift %d->%d: source frame %d has no WCS solution "
            "(an upstream shift was rejected); routing around it.",
            rate_frame_a.index,
            rate_frame_b.index,
            rate_frame_a.index,
        )
        frame_shift.is_valid = False
        frame_shift.processed = True
        return

    # Return the modified object
    rate_a_exposure_time = float(rate_frame_a.frame.header.get("EXPTIME", 1))
    rate_b_exposure_time = float(rate_frame_b.frame.header.get("EXPTIME", 1))
    inter_frame_gap_seconds = abs(
        (rate_frame_a.timestamp - rate_frame_b.timestamp).total_seconds()
    ) - 0.5 * (rate_a_exposure_time + rate_b_exposure_time)

    # Elapsed time between the two frames' exposure midpoints; the denominator for the object's pixel
    # track rate. Two rate frames sharing a timestamp (duplicate/degenerate DATE-OBS) make this zero,
    # which would divide the rate to infinity and crash streak sizing (int(inf)). Route around the
    # pair -- as with a missing source WCS above -- rather than killing the whole collect.
    elapsed_seconds = inter_frame_gap_seconds + 0.5 * (rate_a_exposure_time + rate_b_exposure_time)
    if elapsed_seconds <= 0:
        logger.warning(
            "Skipping rate->rate shift %d->%d: non-positive elapsed time (%.3f s) between frame "
            "exposure midpoints (duplicate/degenerate frame timing); routing around it.",
            rate_frame_a.index,
            rate_frame_b.index,
            elapsed_seconds,
        )
        frame_shift.is_valid = False
        frame_shift.processed = True
        return

    # Get the average pixel track rate if available from both frames. Each rate->rate pair stores
    # its measured rate on its target frame, so by the second pair onward these are populated.
    rates = []
    if rate_frame_a.pixel_track_rate_per_second is not None:
        rates.append(rate_frame_a.pixel_track_rate_per_second)
    if rate_frame_b.pixel_track_rate_per_second is not None:
        rates.append(rate_frame_b.pixel_track_rate_per_second)

    if not rates:
        # First pair: no rate measured yet. Seed directly from the mount track rate (or streak
        # length) rather than from the sidereal->rate offset, which under-estimates the object
        # rate (it spans a sidereal->rate slew-mode change) and can leave the window too tight.
        initial_rate = _initial_track_rate(rate_frame_a)
        if initial_rate is not None:
            rates.append(initial_rate)

    pixel_track_rate_per_second = np.mean(rates) if rates else None

    fwhms = []
    if rate_frame_a.streak is not None:
        fwhms.append(rate_frame_a.streak.fwhm)
    if rate_frame_b.streak is not None:
        fwhms.append(rate_frame_b.streak.fwhm)

    streak_fwhm = np.mean(fwhms) if fwhms else 4.0

    rate_a_data = prepare_rate_frame(rate_frame_a)
    rate_b_data = prepare_rate_frame(rate_frame_b)

    # whopping bright streaks can mess with correlation
    rate_a_data, _ = remove_near_saturation_streaks(rate_a_data, rate_frame_a.frame.data_type)
    rate_b_data, _ = remove_near_saturation_streaks(rate_b_data, rate_frame_b.frame.data_type)

    rate_a_data = remove_border_crossing_streaks(rate_a_data)
    rate_b_data = remove_border_crossing_streaks(rate_b_data)

    strip_unbalanced_streaks(rate_a_data, rate_b_data)

    rate_a_data = rate_a_data / np.std(rate_a_data)
    rate_a_data -= np.mean(rate_a_data)
    rate_b_data = rate_b_data / np.std(rate_b_data)
    rate_b_data -= np.mean(rate_b_data)

    # fast fourier-based cross correlation
    cross_correlated_image = cross_corr(rate_a_data, rate_b_data)

    expected_shift = None
    if pixel_track_rate_per_second is not None:
        # expected pixel shift...
        expected_shift = pixel_track_rate_per_second * elapsed_seconds

    # Examine several DISTINCT candidate peaks, not just the brightest. The brightest in-window
    # peak can be a spurious star-streak artifact that fails the flux-correlation gate while the
    # true (fainter) peak validates; validate candidates in brightness order and take the first
    # that passes. If none passes, fall back to the brightest rather than a distinct peak with a
    # marginally higher (still-sub-gate) correlation -- across the MDP set, alternative selectors
    # (best-correlation, closest-to-seed) each recovered one collect but regressed more, because
    # which signal points at the true peak depends on per-collect seed quality. Brightest-with-
    # fallback is the empirical best net.
    max_trials = 3
    search_radius_pixels = 5.0
    original_center = np.array(cross_correlated_image.shape) / 2
    candidate_peaks = windowed_correlation_peaks(
        cross_correlated_image, expected_shift, n_peaks=max_trials
    )

    best_shift = None
    best_correlation = None
    fallback_shift = None
    fallback_correlation = None
    for trial_index, cc_max in enumerate(candidate_peaks):
        # Shift vector from center to peak, in (x, y) ordering.
        shift_rate_to_rate_xy = (original_center - cc_max)[::-1]

        # Validate this proposal with Bayesian flux correlation.
        shift_rate_to_rate_xy[0], shift_rate_to_rate_xy[1], correlation = (
            bayesian_optimize_proposed_shift(
                target=rate_frame_b,
                source=rate_frame_a,
                shift_x_low=shift_rate_to_rate_xy[0] - search_radius_pixels,
                shift_x_high=shift_rate_to_rate_xy[0] + search_radius_pixels,
                shift_y_low=shift_rate_to_rate_xy[1] - search_radius_pixels,
                shift_y_high=shift_rate_to_rate_xy[1] + search_radius_pixels,
                catalog_stars=rate_frame_a.starfield.catalog_stars,
                n_calls=10,
            )
        )

        # The brightest in-window peak (first candidate) is the fallback when nothing clears the gate.
        if trial_index == 0:
            fallback_shift = shift_rate_to_rate_xy
            fallback_correlation = correlation

        if correlation > 0.9:
            best_shift = shift_rate_to_rate_xy
            best_correlation = correlation
            break

    if best_shift is None:
        best_shift = fallback_shift
        best_correlation = fallback_correlation

    # Log the quality of this fit
    if best_correlation > 0.9:
        # Use the best shift we managed to find
        logger.info(
            f"Fit frame shift from frame {rate_frame_a.index} to frame {rate_frame_b.index} with {best_correlation} correlation"
        )
        shift_rate_to_rate_xy = best_shift
    else:
        # Use the best shift we managed to find
        logger.warning(
            f"Poor fit ({best_correlation} correlation) for frame shift from frame {rate_frame_a.index} to frame {rate_frame_b.index}"
        )
        shift_rate_to_rate_xy = best_shift

    # Calculate the magnitude of the shift - apply the -1 adjustment here for consistency
    # This ensures the magnitude calculation matches the adjusted shift values
    adjusted_shift = np.array([shift_rate_to_rate_xy[0] - 1, shift_rate_to_rate_xy[1] - 1])
    pixel_shift = np.linalg.norm(adjusted_shift)

    # Calculate rate based on the time between frames
    estimated_pixel_track_rate_per_second = pixel_shift / elapsed_seconds

    logger.info(
        f"Pixel shift rate to rate: {pixel_shift:.1f} pixels, {estimated_pixel_track_rate_per_second:.1f} pixels/s"
    )

    frame_shift.x_shift = shift_rate_to_rate_xy[0] - 1
    frame_shift.y_shift = shift_rate_to_rate_xy[1] - 1
    frame_shift.processed = True
    frame_shift.is_valid = True
    frame_shift.correlation = best_correlation

    streak_length_expected_from_shift = (
        estimated_pixel_track_rate_per_second * rate_a_exposure_time
    )

    streak_orientation_expected_from_shift = np.rad2deg(
        np.arctan2(shift_rate_to_rate_xy[1], shift_rate_to_rate_xy[0])
    )

    if not rate_frame_b.streak:
        # refine length on image
        rotation_estimate, length_estimate, psf, fwhm = extract_streak_dims_robust(
            rate_b_data,
            n_streaks=5,
            rotation=streak_orientation_expected_from_shift,
            length=streak_length_expected_from_shift,
            fwhm=streak_fwhm,
        )

        if settings.plotting.debug:
            plot_single_frame(
                psf,
                scale=False,
                output_file=Path(settings.plotting.output_dir)
                / f"{rate_frame_b.index}_streak_psf.png",
            )

        if percent_difference(length_estimate, streak_length_expected_from_shift) > 10:
            logger.warning(
                f"The estimated streak length ({length_estimate:.1f} pixels) is more than 10% different than the expected streak length ({streak_length_expected_from_shift:.1f} pixels)"
            )
        rate_frame_b.streak = StreakMetadata(
            pixel_length=length_estimate,
            sine_angle=np.sin(np.deg2rad(rotation_estimate)),
            cosine_angle=np.cos(np.deg2rad(rotation_estimate)),
            fwhm=fwhm,
        )
        rate_frame_b.pixel_track_rate_per_second = (
            np.linalg.norm(shift_rate_to_rate_xy) / elapsed_seconds
        )


def bayesian_optimize_proposed_shift(
    target: RateTrackFrame | SiderealFrame,
    source: RateTrackFrame | SiderealFrame,
    shift_x_low: float,
    shift_x_high: float,
    shift_y_low: float,
    shift_y_high: float,
    catalog_stars: list[StarInSpace],
    n_calls: int = 10,
) -> tuple[float, float, float]:
    """Search for the shift that maximizes flux correlation via Bayesian optimization.

    Runs Gaussian-process minimization over the (x, y) shift search box, maximizing the
    flux correlation returned by ``validate_proposed_shift`` (by minimizing its negation).

    Args:
        target: The frame being shifted to align with the source.
        source: The reference frame.
        shift_x_low: Lower bound of the x-shift search range (pixels).
        shift_x_high: Upper bound of the x-shift search range (pixels).
        shift_y_low: Lower bound of the y-shift search range (pixels).
        shift_y_high: Upper bound of the y-shift search range (pixels).
        catalog_stars: Catalog stars from the source frame used for flux validation.
        n_calls: Total number of objective evaluations allowed.

    Returns:
        A tuple of ``(best_shift_x, best_shift_y, best_correlation)`` for the highest-
        correlation shift found.
    """
    # Define the search space for the parameters.
    space = [
        Real(shift_x_low, shift_x_high, name="shift_x"),
        Real(shift_y_low, shift_y_high, name="shift_y"),
    ]

    # Run the Bayesian optimization.
    # `n_calls` specifies the total number of function calls allowed.
    result = gp_minimize(
        lambda params: (
            -1.0
            * validate_proposed_shift(
                target=target,
                source=source,
                shift_x=params[0],
                shift_y=params[1],
                catalog_stars=catalog_stars,
            )
        ),
        space,
        n_calls=n_calls,  # Budget for the number of function calls
        random_state=0,  # Make the results reproducable
    )

    best_shift_x = result.x[0]
    best_shift_y = result.x[1]
    best_correlation = -1.0 * result.fun
    return best_shift_x, best_shift_y, best_correlation
