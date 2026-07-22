"""Streak extraction and PSF-measurement utilities for the streak detectors.

Collects the frame-preparation, cross-correlation, streak-geometry, and PSF-FWHM
helpers shared by the rate/sidereal streak shift solvers.
"""

import logging

import numpy as np
from numpy.fft import fft2, ifft2
from scipy import ndimage
from scipy.ndimage import median_filter, rotate
from scipy.optimize import curve_fit
from scipy.signal import convolve

from senpai.engine.detection.kernels import rectangle_pyramoid
from senpai.engine.detection.streak.masking import (
    map_cluster,
)
from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
from senpai.engine.utils.simulation import simulated_sidereal_frame
from senpai.settings import settings

logger = logging.getLogger(__name__)


def prepare_rate_frame(rate_frame: RateTrackFrame, padding: float = 0.1) -> np.ndarray:
    """Prepare a rate-tracked frame for cross correlation.

    Crops a fractional border off the frame, median filters it to suppress hot pixels,
    and shifts it so the minimum value is zero.

    Args:
        rate_frame: The rate-tracked frame whose image data is prepared.
        padding: Fraction of each dimension to crop from every edge.

    Returns:
        The cropped, median-filtered, minimum-subtracted image data as float32.
    """
    p10w = int(rate_frame.frame.data.shape[0] * padding)
    p10h = int(rate_frame.frame.data.shape[1] * padding)

    rate_data = rate_frame.frame.data.copy()[p10w:-p10w, p10h:-p10h].astype(np.float32)

    rate_data = median_filter(rate_data, size=2)
    rate_data -= np.min(rate_data)
    return rate_data


def prepare_sidereal_frame(
    sidereal_frame: SiderealFrame, padding: float = 0.1, allow_synthetic: bool = False
) -> np.ndarray:
    """Prepare a sidereal frame (real or synthetic) for cross correlation.

    When ``allow_synthetic`` is set and the frame has a fitted starfield, a narrow
    synthetic sidereal frame is generated from the starfield; otherwise the real frame
    data is cropped and median filtered. In both cases the result is minimum subtracted.

    Args:
        sidereal_frame: The sidereal frame whose image data is prepared.
        padding: Fraction of each dimension to crop from every edge.
        allow_synthetic: Whether to generate a synthetic frame from a fitted starfield.

    Returns:
        A tuple of the prepared float32 image data and a boolean flag indicating whether
        a (non-empty) synthetic frame was produced.
    """
    p10w = int(sidereal_frame.frame.data.shape[0] * padding)
    p10h = int(sidereal_frame.frame.data.shape[1] * padding)

    if allow_synthetic and sidereal_frame.starfield and sidereal_frame.starfield.fit:
        # generate a very narrow pseudo sidereal frame
        sidereal_data = simulated_sidereal_frame(sidereal_frame.starfield)[
            p10w:-p10w, p10h:-p10h
        ].astype(np.float32)
        is_synthetic = np.max(sidereal_data) != 0
    else:
        sidereal_data = sidereal_frame.frame.data.copy()[p10w:-p10w, p10h:-p10h].astype(np.float32)
        sidereal_data = median_filter(sidereal_data, size=2)
        is_synthetic = False

    sidereal_data -= np.min(sidereal_data)
    return sidereal_data, is_synthetic


def refine_streak_len(
    psf: np.array,
    pixel_fwhm: float | None,
    rotation: float,
    half_max_value: float = 0.55,
) -> float:
    """Refine streak length in cases of noisy PSF by looking along streak angle.

    Args:
        psf (np.array): PSF
        pixel_fwhm (float): estimate of pixel FWHM
        rotation (float): streak rotation
        half_max_value (float, optional): value of PSF (0-1) at which to cut for length. default 0.55

    Returns:
        float: refined estimate of streak length
    """
    rotated_psf = rotate(psf, angle=rotation, mode="constant", cval=np.min(psf))
    rotated_psf /= np.max(rotated_psf)

    if pixel_fwhm is None:
        pixel_fwhm = streak_fwhm_from_cutout(rotated_psf, 0)

    start_point = np.unravel_index(np.argmax(rotated_psf), rotated_psf.shape)
    cutout = rotated_psf[int(start_point[0] - pixel_fwhm) : int(start_point[0] + pixel_fwhm), :]

    valids = np.where(np.max(cutout, 0) > half_max_value)

    # calculate length and return
    return np.max(valids) - np.min(valids)


def extract_streak_dims_robust(
    data: np.ndarray,
    n_streaks: int = 10,
    length: float | None = None,
    rotation: float | None = None,
    fwhm: float | None = None,
) -> tuple[float | None, float | None, np.ndarray | None, float | None]:
    """Extract streak dimensions using a robust matched-filter approach.

    Combines matched filtering, morphological (connected-component and covariance) shape
    analysis, and statistical validation to estimate the streak rotation, length, stacked
    PSF, and measured FWHM. Falls back to the initial estimates when no robust candidate
    is found.

    Args:
        data: Input image data
        n_streaks: Maximum number of streaks to extract
        length: Initial estimate of streak length in pixels
        rotation: Initial estimate of streak rotation in degrees
        fwhm: Estimated FWHM of the PSF in pixels (if None, will be measured)

    Returns:
        A tuple of ``(rotation, length, psf, measured_fwhm)``. When no valid candidate is
        found, ``psf`` is None and the input estimates are returned unchanged.

    Raises:
        ValueError: If ``length`` is negative.
    """
    # Get maximum FWHM from config
    max_fwhm = settings.streak.max_fwhm_for_streak_extraction

    # Argument validation
    if length < 0:
        raise ValueError("Length cannot be negative.")

    # If FWHM is not provided, make an initial estimate
    if fwhm is None:
        # Start with a reasonable default
        fwhm = 4.0
        logger.info(f"No FWHM provided, using initial estimate of {fwhm:.1f} pixels")
    elif fwhm > max_fwhm:
        logger.warning(
            f"FWHM ({fwhm}) is larger than can safely be processed (max of {max_fwhm}). Reducing fwhm to {max_fwhm}."
        )
        fwhm = max_fwhm

    logger.info(
        f"Extracting streak parameters using robust method (initial est: len={length:.1f}, rot={rotation:.1f}°, fwhm={fwhm:.1f})"
    )

    # Step 1: Create a clean working copy of the data
    working_data = data.copy()

    # Background statistics for thresholding
    bg_median = np.median(working_data)
    bg_std = np.std(working_data[working_data < np.percentile(working_data, 80)])

    # Step 2: Create matched filter kernel based on initial estimates
    kernel = rectangle_pyramoid(
        length,
        np.sin(np.deg2rad(rotation)),
        np.cos(np.deg2rad(rotation)),
        int(fwhm * 2),
        halo_fwhm=4,
    )

    mask_kernel = rectangle_pyramoid(
        length * 1.2,
        np.sin(np.deg2rad(rotation)),
        np.cos(np.deg2rad(rotation)),
        int(fwhm * 2.2),
        halo_fwhm=4,
    )

    # Step 3: Apply matched filter
    filtered_data = convolve(working_data, kernel, mode="same")

    # Step 4: Clean up borders to avoid edge artifacts
    border_width = max(10, int(length * 0.5))
    filtered_data[:border_width, :] = bg_median
    filtered_data[-border_width:, :] = bg_median
    filtered_data[:, :border_width] = bg_median
    filtered_data[:, -border_width:] = bg_median

    # Step 5: Extract streak candidates
    streak_candidates = []
    streak_metrics = []

    # Size of cutout region (make it generously larger than expected streak)
    cutout_size = int(max(length * 2, fwhm * 10))

    # Keep track of already processed regions
    processed_mask = np.zeros_like(filtered_data, dtype=bool)

    for _ in range(min(30, n_streaks * 3)):  # Try more candidates than needed
        if np.all(processed_mask[border_width:-border_width, border_width:-border_width]):
            break  # Stop if we've processed all valid regions

        # Find the brightest unprocessed point
        temp_data = filtered_data.copy()
        temp_data[processed_mask] = bg_median

        y_max, x_max = np.unravel_index(np.argmax(temp_data), temp_data.shape)

        # Skip if too close to edge for full cutout
        if (
            y_max < cutout_size
            or y_max >= working_data.shape[0] - cutout_size
            or x_max < cutout_size
            or x_max >= working_data.shape[1] - cutout_size
        ):
            # Mark this region as processed and mask the working data
            processed_mask, working_data = mask_streak_region(
                processed_mask, working_data, y_max, x_max, mask_kernel
            )
            continue

        # Extract cutout from original data
        cutout = working_data[
            y_max - cutout_size : y_max + cutout_size, x_max - cutout_size : x_max + cutout_size
        ].copy()

        # Check if the cutout overlaps with previously masked regions
        if not is_valid_psf(cutout, processed_mask, y_max, x_max, cutout_size):
            # Mark this region as processed and mask the working data
            processed_mask, working_data = mask_streak_region(
                processed_mask, working_data, y_max, x_max, mask_kernel
            )
            continue

        # Calculate SNR and other metrics
        local_bg = np.median(cutout)
        local_noise = np.std(cutout[cutout < local_bg + 2 * bg_std])
        peak_value = np.max(cutout)
        snr = (peak_value - local_bg) / local_noise if local_noise > 0 else 0

        # Skip low SNR detections
        if snr < 3.0:
            # Mark this region as processed and mask the working data
            processed_mask, working_data = mask_streak_region(
                processed_mask, working_data, y_max, x_max, mask_kernel
            )
            continue

        # Normalize cutout for analysis
        norm_cutout = cutout.copy()
        norm_cutout -= local_bg
        norm_cutout = np.clip(norm_cutout, 0, None)  # Remove negative values
        norm_cutout /= np.max(norm_cutout) if np.max(norm_cutout) > 0 else 1.0

        # Analyze shape using connected components
        binary_cutout = norm_cutout > 0.3  # Threshold at 30% of peak

        # Skip if no pixels above threshold
        if not np.any(binary_cutout):
            # Mark this region as processed and mask the working data
            processed_mask, working_data = mask_streak_region(
                processed_mask, working_data, y_max, x_max, mask_kernel
            )
            continue

        # Find connected component containing peak
        labeled, _num_features = ndimage.label(binary_cutout)
        peak_y, peak_x = np.unravel_index(np.argmax(norm_cutout), norm_cutout.shape)
        peak_label = labeled[peak_y, peak_x]

        if peak_label == 0:  # No label at peak (shouldn't happen)
            # Mark this region as processed and mask the working data
            processed_mask, working_data = mask_streak_region(
                processed_mask, working_data, y_max, x_max, mask_kernel
            )
            continue

        # Extract just the connected component containing the peak
        component_mask = labeled == peak_label

        # Calculate shape metrics
        y_indices, x_indices = np.where(component_mask)

        # Skip if too few pixels
        if len(y_indices) < 5:
            # Mark this region as processed and mask the working data
            processed_mask, working_data = mask_streak_region(
                processed_mask, working_data, y_max, x_max, mask_kernel
            )
            continue

        # Calculate covariance matrix for shape analysis
        points = np.column_stack([y_indices, x_indices])
        points_centered = points - np.mean(points, axis=0)
        cov = np.cov(points_centered, rowvar=False)

        # Get eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Calculate aspect ratio
            aspect_ratio = np.sqrt(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 10.0

            # Calculate orientation
            streak_angle = np.degrees(np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0]))
            streak_angle = streak_angle % 180

            # Calculate length using eigenvalues
            # The length is approximately 4 * sqrt(largest eigenvalue)
            # This corresponds to ~95% of the mass of a Gaussian distribution
            component_length = 4 * np.sqrt(eigenvalues[0])

            # Calculate width similarly
            component_width = 4 * np.sqrt(eigenvalues[1])

            # Calculate angle difference from expected
            angle_diff = min(
                abs(streak_angle - rotation),
                abs(streak_angle - rotation + 180),
                abs(streak_angle - rotation - 180),
            )

            # Calculate length difference from expected
            length_ratio = component_length / length if length > 0 else 1.0

            # Skip if aspect ratio is too low (not streak-like)
            if aspect_ratio < 1.5:
                # Mark this region as processed and mask the working data
                processed_mask, working_data = mask_streak_region(
                    processed_mask, working_data, y_max, x_max, mask_kernel
                )
                continue

            # Calculate overall quality score
            # Higher for: high SNR, high aspect ratio, angle close to expected, length close to expected
            quality_score = (
                snr
                * aspect_ratio
                * (1.0 / (1.0 + angle_diff / 10))
                * (1.0 / (1.0 + abs(length_ratio - 1.0)))
            )

            # Store candidate and metrics
            streak_candidates.append(norm_cutout)
            streak_metrics.append(
                {
                    "snr": snr,
                    "aspect_ratio": aspect_ratio,
                    "angle": streak_angle,
                    "length": component_length,
                    "width": component_width,
                    "quality_score": quality_score,
                }
            )

            logger.debug(
                f"Found streak candidate: SNR={snr:.1f}, AR={aspect_ratio:.1f}, "
                f"angle={streak_angle:.1f}°, length={component_length:.1f}, "
                f"score={quality_score:.1f}"
            )

        except np.linalg.LinAlgError:
            # Skip if eigenvalue decomposition fails
            pass

        # After processing, mark this region as processed and mask the working data
        processed_mask, working_data = mask_streak_region(
            processed_mask, working_data, y_max, x_max, mask_kernel
        )

    # Step 6: Select best candidates and create PSF
    if not streak_candidates:
        logger.warning("No valid streak candidates found")
        return rotation, length, None, fwhm

    # Sort by quality score
    sorted_indices = np.argsort([m["quality_score"] for m in streak_metrics])[::-1]

    # Take top n candidates (or all if fewer)
    top_n = min(n_streaks, len(streak_candidates))
    selected_indices = sorted_indices[:top_n]

    selected_streaks = [streak_candidates[i] for i in selected_indices]
    selected_metrics = [streak_metrics[i] for i in selected_indices]

    logger.info(f"Selected {top_n} best streak candidates out of {len(streak_candidates)}")

    # Step 7: Align streaks before stacking
    aligned_streaks = []
    for i, streak in enumerate(selected_streaks):
        # Rotate to align with horizontal axis
        angle_to_horizontal = selected_metrics[i]["angle"] - 90
        aligned = rotate(streak, angle_to_horizontal, reshape=False, mode="constant", cval=0)
        aligned_streaks.append(aligned)

    # Step 8: Create PSF by stacking aligned streaks
    psf = np.median(np.stack(aligned_streaks), axis=0)

    # Rotate back to original orientation
    psf = rotate(
        psf,
        90 - np.median([m["angle"] for m in selected_metrics]),
        reshape=False,
        mode="constant",
        cval=0,
    )

    # Normalize PSF
    psf -= np.min(psf)
    psf /= np.max(psf) if np.max(psf) > 0 else 1.0

    # Step 9: Calculate final streak parameters
    # Use median of individual measurements for robustness
    raw_length = np.median([m["length"] for m in selected_metrics])

    # Ensure the corrected length is not negative or too small
    final_length = max(raw_length, fwhm * 0.5)

    final_angle = np.median([m["angle"] for m in selected_metrics])

    # Log both raw and corrected measurements
    logger.info(
        f"Raw streak length: {raw_length:.1f}, corrected to: {final_length:.1f} after PSF subtraction"
    )

    # Sanity check on length
    if final_length < fwhm:
        logger.warning(f"Corrected length ({final_length:.1f}) is smaller than FWHM ({fwhm:.1f})")
        # Fall back to original estimate if available and reasonable
        if length is not None and length > final_length:
            logger.info(f"Using original length estimate: {length:.1f}")
            final_length = length

    # Sanity check on angle
    angle_diffs = [
        min(
            abs(m["angle"] - rotation),
            abs(m["angle"] - rotation + 180),
            abs(m["angle"] - rotation - 180),
        )
        for m in selected_metrics
    ]
    if np.median(angle_diffs) > 20:  # If median angle differs by more than 20 degrees
        logger.warning(
            f"Measured angle ({final_angle:.1f}°) differs significantly from expected ({rotation:.1f}°)"
        )
        # Fall back to original estimate
        if rotation is not None:
            logger.info(f"Using original angle estimate: {rotation:.1f}°")
            final_angle = rotation
            logger.info(f"Using original length estimate: {length:.1f}")
            final_length = length
            # Skip further length corrections since we're using the original length
            logger.info(
                f"Final streak parameters: length={final_length:.1f}, angle={final_angle:.1f}°"
            )
            return final_angle, final_length, psf, fwhm

    logger.info(f"Final streak parameters: length={final_length:.1f}, angle={final_angle:.1f}°")

    # After creating the PSF, measure its FWHM perpendicular to the streak direction
    measured_fwhm = measure_psf_fwhm(psf, final_angle)

    if measured_fwhm is not None and measured_fwhm > 0:
        if measured_fwhm > max_fwhm:
            # A degraded streak can yield an implausibly large measured FWHM, which then bloats
            # downstream aperture photometry. Cap it like the initial estimate above.
            logger.warning(
                f"Measured PSF FWHM ({measured_fwhm:.1f}) exceeds the maximum ({max_fwhm}); "
                "capping it."
            )
            measured_fwhm = max_fwhm
        logger.info(f"Measured PSF FWHM: {measured_fwhm:.1f} pixels")
        # Use the measured FWHM for length correction
        fwhm_for_correction = measured_fwhm
    else:
        logger.warning("Could not measure PSF FWHM, using initial estimate")
        fwhm_for_correction = fwhm

    # Calculate raw and corrected lengths
    raw_length = final_length

    # Correct for PSF blurring by subtracting FWHM
    # Option 1: Subtract more than one FWHM (e.g., 1.5 times)
    corrected_length = raw_length - (1.5 * fwhm_for_correction)

    # Ensure the corrected length is not negative or too small
    final_length = max(corrected_length, fwhm_for_correction * 0.5)

    logger.info(
        f"Raw streak length: {raw_length:.1f}, corrected to: {final_length:.1f} after PSF subtraction"
    )

    # Sanity checks and fallbacks as before...

    # Return the measured FWHM along with other parameters
    return final_angle, final_length, psf, measured_fwhm


def streak_fwhm_from_cutout(cutout_frame: np.ndarray, rotation: float) -> float:
    """Measure the Full Width at Half Maximum (FWHM) of a streak PSF.

    Args:
        cutout_frame: 2D array containing the PSF
        rotation: Angle to rotate the cutout (degrees) to align streak vertically

    Returns:
        float: FWHM in pixels, or None if measurement fails
    """
    if rotation != 0:
        rotated_cutout = rotate(cutout_frame, angle=rotation, mode="constant", cval=np.nan)
    else:
        rotated_cutout = cutout_frame

    # Compress along horizontal axis using mean to get vertical profile
    vertical_profile = np.nanmean(rotated_cutout, axis=1)

    # Remove any NaN values
    vertical_profile = vertical_profile[~np.isnan(vertical_profile)]

    # Find peak value and location
    peak_value = np.max(vertical_profile)
    peak_idx = np.argmax(vertical_profile)

    # Calculate half maximum value
    half_max = peak_value / 2

    # Find points where profile crosses half maximum
    above_half = vertical_profile >= half_max

    # Use more robust method to find FWHM
    # Look for crossings on both sides of the peak separately
    left_side = above_half[:peak_idx]
    right_side = above_half[peak_idx:]

    if len(left_side) > 0 and len(right_side) > 0:
        # Find left crossing (last True before peak)
        left_idx = len(left_side) - 1 - np.argmax(not left_side[::-1]) if not np.all(left_side) else 0

        # Find right crossing (first False after peak)
        right_idx = peak_idx + np.argmax(not right_side) if not np.all(right_side) else len(vertical_profile) - 1

        # Calculate FWHM in pixels
        fwhm = right_idx - left_idx
        return float(fwhm)
    else:
        # If can't find clear crossings, return None
        return None


def streak_length_from_cutout(cutout_frame: np.ndarray, plot: bool = True) -> float:
    """Estimate a streak's length from a cutout via flood-fill clustering.

    Normalizes the cutout, flood fills the cluster around its brightest pixel at the
    half-maximum level, and measures the diagonal extent of the resulting cluster.

    Args:
        cutout_frame: 2D array containing the streak cutout.
        plot: Unused; retained for backward compatibility.

    Returns:
        The estimated streak length in pixels.
    """
    subcc = cutout_frame.copy()
    subcc = subcc.copy() - np.median(subcc)
    subcc /= np.max(subcc)

    fill_min = 0.50  # FWHM

    start_point = np.unravel_index(np.argmax(subcc), subcc.shape)
    mapped = map_cluster(subcc, start_point, fill_min)

    xlen = min(np.where(mapped)[1]) - max(np.where(mapped)[1])
    ylen = min(np.where(mapped)[0]) - max(np.where(mapped)[0])
    estimated_length = np.sqrt(xlen**2 + ylen**2)

    return estimated_length


def streak_parameters_from_xcorr(
    cutout_frame: np.ndarray,
    plate_scale_arcsec: float,
    seeing_fwhm_pixels: float,
    expected_max_star_distance_arcsec: float = 200,
) -> tuple[float, float, np.ndarray] | None:
    """Estimate streak rotation and length from a cross-correlation image.

    Zeroes the central correlation spike, windows the correlation around the center to the
    expected maximum star travel, flood fills the brightest cluster to estimate its length
    (corrected for seeing), and derives the principal-axis rotation via PCA.

    Args:
        cutout_frame: The cross-correlation image to analyze.
        plate_scale_arcsec: Plate scale in arcsec/pixel, used to size the search window.
        seeing_fwhm_pixels: Seeing FWHM in pixels, subtracted from the raw length estimate.
        expected_max_star_distance_arcsec: Maximum expected star travel in arcsec, used to
            bound the correlation search window.

    Returns:
        A tuple of ``(rotation_deg, estimated_length, subcc)`` where ``subcc`` is the
        windowed, normalized correlation sub-image, or ``None`` if the correlation has no
        extended cluster (no measurable streak). Callers route around a ``None`` rather than
        catching an exception.
    """
    cutout_frame = (cutout_frame.copy() - np.min(cutout_frame)) / np.max(cutout_frame)

    center = np.array(cutout_frame.shape) / 2

    subcc = cutout_frame.copy()

    # this stuff helps if a frame is a correlation of two frames
    subcc[int(center[0]), int(center[1])] = 0

    # max range we expect a star to travel, pixels
    pixel_scale = 150

    if plate_scale_arcsec is not None:
        # define pixel cutout range if we know plate_scale
        pixel_scale = int(2 * expected_max_star_distance_arcsec / plate_scale_arcsec)

    # Ensure indices stay within frame boundaries
    x_min = max(0, int(center[0]) - pixel_scale + 1)
    x_max = min(subcc.shape[0], int(center[0]) + pixel_scale)
    y_min = max(0, int(center[1]) - pixel_scale + 1)
    y_max = min(subcc.shape[1], int(center[1]) + pixel_scale)

    subcc = subcc[x_min:x_max, y_min:y_max]

    subcc = subcc.copy() - np.median(subcc)
    subcc /= np.max(subcc)

    subc = np.array(subcc.shape) / 2

    start_point = np.unravel_index(np.argmax(subcc), subcc.shape)

    # Make sure we have enough points
    fill_min = 0.50  # FWHM
    trials = 0
    max_trials = 10
    while True:
        trials += 1
        mapped = map_cluster(subcc, start_point, fill_min)
        Xy = np.stack(np.where(mapped)).T
        if len(Xy) > 1:
            break
        else:
            fill_min *= 0.9

        if trials >= max_trials:
            # No extended cluster: the correlation has no measurable streak. Signal this with a
            # sentinel so the caller can route around it (or raise a typed error) rather than
            # catching a bare ValueError as control flow.
            return None

    xlen = min(np.where(mapped)[1]) - max(np.where(mapped)[1])
    ylen = min(np.where(mapped)[0]) - max(np.where(mapped)[0])
    estimated_length = np.sqrt(xlen**2 + ylen**2)

    # Negative lengths are nonsensical, so guard against that before performing this subtraction
    if seeing_fwhm_pixels < estimated_length:
        estimated_length -= seeing_fwhm_pixels

    # Calculate the principal axis of the cluster using PCA-like approach
    # First, center the points
    centered_points = Xy - subc

    # Calculate covariance matrix
    cov_matrix = np.cov(centered_points.T)

    # Get eigenvectors and eigenvalues. The covariance matrix is symmetric so the
    # eigensystem is real; numpy 2 can still return a complex dtype with ~0
    # imaginary parts, so take the real part (a no-op on real returns).
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # The eigenvector with the largest eigenvalue gives the principal direction
    principal_direction = eigenvectors[:, np.argmax(eigenvalues)]

    # Calculate angle in degrees, handling the orientation correctly
    rotation_deg = np.rad2deg(np.arctan2(principal_direction[0], principal_direction[1]))

    # Normalize to 0-180 degrees
    rotation_deg = rotation_deg % 180

    rotframe = rotate(subcc, angle=rotation_deg)
    start_point = np.unravel_index(np.argmax(rotframe), rotframe.shape)
    mapped = map_cluster(rotframe, start_point, fill_min)

    return rotation_deg, estimated_length, subcc


def measure_gaussian_shift(centered_cutout: np.ndarray) -> tuple[np.ndarray, float]:
    """Measure the shift of a PSF from the center and its FWHM by fitting a Gaussian.

    Args:
        centered_cutout: 2D array containing a centered PSF

    Returns:
        tuple: (shift_vector, fwhm) where shift_vector is the offset from center and fwhm is the
               full width at half maximum of the fitted Gaussian in pixels
    """
    # Find the peak location
    psf_center = np.unravel_index(np.argmax(centered_cutout), centered_cutout.shape)
    shift = psf_center - np.array(centered_cutout.shape) / 2

    # Extract profiles through the peak
    y_profile = centered_cutout[psf_center[0], :]
    x_profile = centered_cutout[:, psf_center[1]]

    # Normalize profiles
    y_profile = y_profile / np.max(y_profile)
    x_profile = x_profile / np.max(x_profile)

    # Create coordinate arrays
    y_coords = np.arange(len(y_profile))
    x_coords = np.arange(len(x_profile))

    # Define 1D Gaussian function
    def gaussian(
        x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float
    ) -> np.ndarray:
        """Evaluate a 1D Gaussian with an additive offset.

        Args:
            x: Coordinate(s) at which to evaluate the Gaussian.
            amplitude: Peak amplitude of the Gaussian.
            center: Center position of the Gaussian.
            sigma: Standard deviation of the Gaussian.
            offset: Constant additive offset.

        Returns:
            The Gaussian evaluated at ``x``.
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + offset

    # Initial parameter guesses
    p0_y = [1.0, np.argmax(y_profile), 3.0, 0.0]
    p0_x = [1.0, np.argmax(x_profile), 3.0, 0.0]

    try:
        # Fit Gaussians to both profiles
        popt_y, _ = curve_fit(gaussian, y_coords, y_profile, p0=p0_y)
        popt_x, _ = curve_fit(gaussian, x_coords, x_profile, p0=p0_x)

        # Extract sigma values
        sigma_y = abs(popt_y[2])
        sigma_x = abs(popt_x[2])

        # Calculate FWHM (FWHM = 2.355 * sigma for a Gaussian)
        fwhm_y = 2.355 * sigma_y
        fwhm_x = 2.355 * sigma_x

        # Use the average FWHM
        fwhm = (fwhm_x + fwhm_y) / 2

    except (RuntimeError, ValueError):
        # If fitting fails, estimate FWHM using half-maximum points
        fwhm = estimate_fwhm_from_profiles(x_profile, y_profile)

    return shift, fwhm


def estimate_fwhm_from_profiles(x_profile: np.ndarray, y_profile: np.ndarray) -> float:
    """Estimate FWHM from profiles when curve fitting fails.

    Finds the half-maximum crossing points in each normalized profile and averages the
    two widths, falling back to a default width for profiles with no points above the
    half-maximum.

    Args:
        x_profile: Normalized profile along the x direction.
        y_profile: Normalized profile along the y direction.

    Returns:
        The estimated FWHM in pixels, averaged across the two profiles.
    """
    # Find half-maximum points in both profiles
    half_max = 0.5

    # Process x profile
    above_half_x = x_profile >= half_max
    if np.any(above_half_x):
        left_x = np.argmax(above_half_x)
        right_x = len(above_half_x) - np.argmax(above_half_x[::-1]) - 1
        fwhm_x = right_x - left_x
    else:
        fwhm_x = 4.0  # Default value

    # Process y profile
    above_half_y = y_profile >= half_max
    if np.any(above_half_y):
        left_y = np.argmax(above_half_y)
        right_y = len(above_half_y) - np.argmax(above_half_y[::-1]) - 1
        fwhm_y = right_y - left_y
    else:
        fwhm_y = 4.0  # Default value

    return (fwhm_x + fwhm_y) / 2


def measure_psf_shift(
    centered_cutout: np.ndarray, length: float, rotation: float, pixel_fwhm: float
) -> np.ndarray:
    """Measure a PSF's offset from center by matched-filtering a streak kernel.

    Convolves the cutout with a rectangle-pyramid streak kernel of the given geometry and
    returns the offset of the convolution peak from the cutout center.

    Args:
        centered_cutout: 2D array containing the (approximately centered) PSF.
        length: Streak length in pixels used to build the matched-filter kernel.
        rotation: Streak rotation in degrees used to build the matched-filter kernel.
        pixel_fwhm: PSF FWHM in pixels used to size the kernel width.

    Returns:
        The (row, column) offset of the convolution peak from the cutout center.
    """
    kernel = rectangle_pyramoid(
        length,
        np.sin(np.deg2rad(rotation)),
        np.cos(np.deg2rad(rotation)),
        int(pixel_fwhm + 1.5),
        halo_fwhm=4,
    )

    conv = convolve(centered_cutout, kernel, mode="same")

    psf_center = np.unravel_index(np.argmax(conv), conv.shape)

    return psf_center - np.array(centered_cutout.shape) / 2


def cross_corr(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Cross correlate two images using FFT.

    Args:
        img1 (np.ndarray): First image to cross correlate
        img2 (np.ndarray): Second image to cross correlate

    Returns:
        np.ndarray: Cross correlated image
    """
    ccf = np.roll(
        ifft2(fft2(img1).conj() * fft2(img2)).real,
        np.array([img1.shape[0] - 1, img1.shape[1] - 1]) // 2,
        axis=(0, 1),
    )

    return ccf


def measure_psf_fwhm(data: np.ndarray, rotation: float | None = None) -> float:
    """Measure the FWHM of the PSF perpendicular to the streak direction with sub-pixel precision.

    Args:
        data: 2D array containing a normalized PSF or source
        rotation: Angle of the streak in degrees (if None, will try to determine)

    Returns:
        float: FWHM in pixels
    """
    # If rotation is not provided, try to determine it
    if rotation is None:
        # Use PCA to find principal axes
        y_indices, x_indices = np.where(data > 0.5 * np.max(data))
        if len(y_indices) < 5:  # Not enough points
            return None

        points = np.column_stack([y_indices, x_indices])
        points_centered = points - np.mean(points, axis=0)

        try:
            cov = np.cov(points_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Calculate orientation (perpendicular to streak)
            streak_angle = np.degrees(np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0]))
            # The width direction is perpendicular to the streak
            width_angle = (streak_angle + 90) % 180
        except np.linalg.LinAlgError:
            return None
    else:
        # Width direction is perpendicular to streak
        width_angle = (rotation + 90) % 180

    # Rotate the data to align the width direction with horizontal axis
    rotated_data = rotate(data, width_angle, reshape=False, mode="constant", cval=0)

    # Find the peak
    peak_y, _peak_x = np.unravel_index(np.argmax(rotated_data), rotated_data.shape)

    # Extract a profile through the peak perpendicular to the streak
    profile = rotated_data[peak_y, :]

    # Normalize the profile. A non-positive peak means there is no usable signal in this
    # profile, so there is no measurable FWHM.
    peak = np.max(profile)
    if peak <= 0:
        return None
    profile = profile / peak

    # Find the half-maximum value
    half_max = 0.5

    # Find indices where profile crosses half-maximum
    above_half = profile >= half_max
    if not np.any(above_half):
        return None

    # Find approximate crossing points
    left_idx = np.argmax(above_half)  # First crossing
    right_idx = len(above_half) - np.argmax(above_half[::-1]) - 1  # Last crossing

    # Refine left crossing with linear interpolation
    if left_idx > 0:
        y1 = profile[left_idx - 1]
        y2 = profile[left_idx]
        # Linear interpolation: x = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
        left_precise = (left_idx - 1) + (half_max - y1) / (y2 - y1) if y2 > y1 else float(left_idx)
    else:
        left_precise = float(left_idx)

    # Refine right crossing with linear interpolation
    if right_idx < len(profile) - 1:
        y1 = profile[right_idx]
        y2 = profile[right_idx + 1]
        # Linear interpolation: x = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
        right_precise = right_idx + (half_max - y1) / (y2 - y1) if y1 > y2 else float(right_idx)
    else:
        right_precise = float(right_idx)

    # Calculate FWHM with sub-pixel precision
    fwhm = right_precise - left_precise

    # If FWHM is too small, it might be noise - set a minimum
    fwhm = max(fwhm, 2.0)

    return fwhm


def mask_streak_region(
    processed_mask: np.ndarray,
    working_data: np.ndarray,
    y_max: int,
    x_max: int,
    kernel: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask a streak region in the processed mask and working data.

    Convolves the kernel with itself to determine the effective area of influence, then
    marks that region in the processed mask and replaces the corresponding working-data
    pixels with the background median.

    Args:
        processed_mask: The existing mask to update.
        working_data: The image data to mask.
        y_max: The row coordinate of the detected streak.
        x_max: The column coordinate of the detected streak.
        kernel: The kernel used for detection.
        threshold: The fraction of max value to use for thresholding.

    Returns:
        A tuple of the updated processed mask and updated working data.
    """
    # Convolve the kernel with itself to get the effective area
    from scipy import signal

    effective_kernel = signal.convolve2d(kernel, kernel, mode="full")

    # Normalize and threshold to create a binary mask
    effective_kernel = effective_kernel / np.max(effective_kernel)
    effective_kernel = effective_kernel > threshold

    ek_height, ek_width = effective_kernel.shape
    y_start = max(0, y_max - ek_height // 2)
    y_end = min(processed_mask.shape[0], y_max + ek_height // 2)
    x_start = max(0, x_max - ek_width // 2)
    x_end = min(processed_mask.shape[1], x_max + ek_width // 2)

    # Calculate kernel indices
    k_y_start = max(0, ek_height // 2 - y_max)
    k_y_end = min(ek_height, k_y_start + (y_end - y_start))
    k_x_start = max(0, ek_width // 2 - x_max)
    k_x_end = min(ek_width, k_x_start + (x_end - x_start))

    kernel_part = effective_kernel[k_y_start:k_y_end, k_x_start:k_x_end]

    # Apply the mask to the processed_mask
    mask_height, mask_width = kernel_part.shape
    if mask_height > 0 and mask_width > 0:  # Ensure we have a valid region
        # Update the processed mask
        processed_mask[y_start:y_end, x_start:x_end] |= kernel_part

        # Also mask the working data with background value
        # Create a mask for the working data
        mask_region = np.zeros_like(working_data, dtype=bool)
        mask_region[y_start:y_end, x_start:x_end] = kernel_part

        # Replace masked pixels with background value
        bg_value = np.median(working_data)
        working_data[mask_region] = bg_value

    return processed_mask, working_data


def is_valid_psf(
    cutout: np.ndarray,
    processed_mask: np.ndarray,
    y_max: int,
    x_max: int,
    cutout_size: int,
) -> bool:
    """Check if a PSF cutout is valid by ensuring it doesn't overlap with masked regions.

    Args:
        cutout: The extracted PSF cutout.
        processed_mask: The mask of processed regions.
        y_max: The row coordinate of the detected streak.
        x_max: The column coordinate of the detected streak.
        cutout_size: Size of the cutout.

    Returns:
        True if the cutout's overlap with masked regions is below threshold, else False.
    """
    # Extract the corresponding region from the processed mask
    y_start = max(0, y_max - cutout_size)
    y_end = min(processed_mask.shape[0], y_max + cutout_size)
    x_start = max(0, x_max - cutout_size)
    x_end = min(processed_mask.shape[1], x_max + cutout_size)

    mask_cutout = processed_mask[y_start:y_end, x_start:x_end]

    # Check if the cutout overlaps with previously masked regions
    # Allow some overlap (e.g., less than 10% of pixels)
    overlap_fraction = np.sum(mask_cutout) / mask_cutout.size

    # Return True if overlap is below threshold
    return overlap_fraction < 0.1  # Adjust threshold as needed
