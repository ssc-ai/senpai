"""WCS refinement for the SENPAI detection pipeline.

Refines a frame's WCS against catalog and astrometric stars for both rate-track
and sidereal frames: a kernel-convolution pass that first applies a global pixel
shift from astrometric stars and then refits against catalog stars, plus the
sidereal equivalent, and the helper that backfills aperture-photometry counts
onto stars that lack them.
"""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from scipy.signal import convolve

from senpai.core.config import settings
from senpai.engine.detection.jacobian import get_local_streak_kernel
from senpai.engine.detection.kernels import sidereal_kernel
from senpai.engine.models.astrometry import WCSMetadata, WCSModel, WCSStatus
from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
from senpai.engine.models.starfield import StarInImage, StarInSpace, StarListImage
from senpai.engine.photometry.utils import (
    calculate_star_snrs_with_aperture_photometry,
    estimate_limiting_magnitude_from_photometry,
)
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.plotting.wcs_diagnostics import (
    plot_variable_kernel_grid,
    plot_variable_kernel_star_diagnostic,
)
from senpai.engine.utils.wcs_helpers import (
    extract_counts_with_rectangular_aperture,
    find_local_maxima,
    match_stars_to_detections,
)
from senpai.engine.utils.wcs_ops import (
    catalog_stars_from_wcs,
    compute_wcs_distortion_metrics,
    existing_stars_from_wcs,
    shift_wcs,
)
from senpai.exceptions import WcsPropagationError

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_star_counts",
    "get_global_shift_from_astrometric_stars",
    "refine_sidereal_frame",
    "refine_sidereal_with_catalog_stars",
    "refine_wcs_by_kernel_convolution",
    "refine_wcs_with_catalog_stars",
]


def refine_wcs_by_kernel_convolution(frame: RateTrackFrame) -> bool:
    """Refine the WCS by convolving the image with a streak kernel.

    Args:
        frame (RateTrackFrame): The frame for which to refine the WCS.

    Returns:
        bool: True if catalog-star refinement succeeded, False if only the global
            shift was applied.

    Raises:
        ValueError: If the frame's WCS status is not PIXEL_SHIFTED_WCS.

    """
    if frame.starfield.wcs_status != WCSStatus.PIXEL_SHIFTED_WCS:
        logger.error(
            "WCS status is not PIXEL_SHIFTED_WCS, skipping kernel convolution [call senpai.engine.utils.wcs_ops.shift_wcs_by_pixel_shift first]"
        )
        raise ValueError("WCS status is not PIXEL_SHIFTED_WCS, skipping kernel convolution")

    # Enable per-star variable kernels when WCS distortion varies across the field (upstream, config-gated).
    use_variable_kernels = False
    try:
        vk = settings.streak.variable_kernel
        if vk.enable and frame.starfield.wcs is not None:
            metrics = compute_wcs_distortion_metrics(frame.starfield.wcs, frame.frame.data.shape)
            if metrics:
                frame.starfield.distortion_metrics = metrics
                if (
                    metrics.get("max_angle_variation_deg", 0.0) >= vk.angle_thresh_deg
                    or metrics.get("max_length_variation_fraction", 0.0) >= vk.length_thresh_fraction
                ):
                    use_variable_kernels = True
    except Exception as e:
        logger.warning("Variable-kernel decision failed for frame %d: %s; using single kernel.", frame.index, e)
    if frame.streak is not None:
        frame.streak.use_variable_kernel = use_variable_kernels

    # Get the kernel
    kernel = frame.streak.to_pyramoid()
    convolved_image = convolve(frame.frame.data, kernel, mode="same")

    # First pass: Get global shift using astrometric fit stars
    global_shift_x, global_shift_y = get_global_shift_from_astrometric_stars(frame, convolved_image)

    # Apply the global shift to the WCS
    original_wcs_model = frame.starfield.wcs
    updated_wcs_model = shift_wcs(original_wcs_model, -global_shift_x, -global_shift_y)

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            streak=frame.streak,
            output_file=Path(settings.plotting.output_dir) / f"{frame.index}_kernel_0_init.png",
        )

    # Update the WCS with the global shift
    frame.starfield.wcs = updated_wcs_model
    frame.starfield.wcs_metadata = WCSMetadata.from_wcsmodel(updated_wcs_model)

    # Update star positions based on the new WCS
    frame.starfield.astrometric_fit_stars = existing_stars_from_wcs(
        updated_wcs_model, frame.starfield.astrometric_fit_stars
    )
    catalog_stars = catalog_stars_from_wcs(updated_wcs_model, 14.0)
    frame.starfield.catalog_stars = catalog_stars.stars

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            streak=frame.streak,
            output_file=Path(settings.plotting.output_dir) / f"{frame.index}_kernel_1_global.png",
        )

    # Second pass: Refine WCS using catalog stars
    refined_wcs = refine_wcs_with_catalog_stars(frame, convolved_image)

    if refined_wcs is not None:
        # Update with the refined WCS if successful
        frame.starfield.wcs = refined_wcs
        frame.starfield.wcs_metadata = WCSMetadata.from_wcsmodel(refined_wcs)

        # Update star positions based on the new WCS
        frame.starfield.astrometric_fit_stars = existing_stars_from_wcs(
            refined_wcs, frame.starfield.astrometric_fit_stars
        )

        # Update catalog stars with the new WCS
        catalog_stars = catalog_stars_from_wcs(refined_wcs, limiting_magnitude=frame.starfield.limiting_magnitude)
        frame.starfield.catalog_stars = catalog_stars.stars

        if settings.plotting.debug:  # pragma: no cover
            plot_single_frame(
                frame.frame.data,
                starfield=frame.starfield,
                streak=frame.streak,
                output_file=Path(settings.plotting.output_dir) / f"{frame.index}_kernel_3_refined.png",
            )

    # Update WCS status
    frame.starfield.wcs_status = WCSStatus.KERNEL_REFINED_WCS

    # Ensure all stars have counts
    ensure_star_counts(frame)
    return refined_wcs is not None


def get_global_shift_from_astrometric_stars(frame: RateTrackFrame, convolved_image: np.ndarray) -> tuple[float, float]:
    """Get global shift using astrometric fit stars.

    Args:
        frame (RateTrackFrame): The frame containing the stars.
        convolved_image (np.ndarray): The convolved image.

    Returns:
        tuple[float, float]: The median shifts in x and y.

    """
    logger.info("First pass: Getting global shift from astrometric fit stars")

    # Use astrometric_fit_stars directly from the starfield
    astrometric_stars = frame.starfield.astrometric_fit_stars

    if not astrometric_stars:
        logger.warning("No astrometric fit stars found, using catalog stars for global shift")
        astrometric_stars = frame.starfield.catalog_stars if frame.starfield.catalog_stars else []

    # Find local maxima in the convolved image
    detected_points = find_local_maxima(convolved_image, min_distance=30, max_detections=50)
    logger.info(f"Found {len(detected_points)} local maxima in the convolved image")

    # Get the stars in the frame as StarInImage objects
    stars_in_image = []
    for star in astrometric_stars:
        if star.x is not None and star.y is not None:
            stars_in_image.append(StarInImage(x=star.x, y=star.y, counts=None))

    # Match stars to detections - using max_distance instead of max_match_distance
    matched_pairs, _unmatched_stars, _unmatched_detections = match_stars_to_detections(
        stars_in_image, detected_points, max_distance=10
    )

    # Track shifts for each matched star
    x_shifts = []
    y_shifts = []

    # Create a list to store detected stars with their new positions
    detected_stars = []

    # Calculate shifts for matched stars
    for star_idx, detection_idx in matched_pairs:
        y, x = detected_points[detection_idx]

        # Calculate shift from original position
        original_x = stars_in_image[star_idx].x
        original_y = stars_in_image[star_idx].y

        # Record the shifts
        x_shift = x - original_x
        y_shift = y - original_y
        x_shifts.append(x_shift)
        y_shifts.append(y_shift)

        # Create StarInImage for this detection (without counts for now)
        star_in_image = StarInImage(x=float(x), y=float(y), counts=None)
        detected_stars.append(star_in_image)

    # Use calculate_star_snrs_with_aperture_photometry to efficiently get counts for all stars at once
    if detected_stars:
        # Create temporary StarInSpace objects with the detected positions
        temp_space_stars = []
        for star in detected_stars:
            # Create a minimal StarInSpace with just the position information
            temp_space_star = StarInSpace(
                ra=0.0,  # Dummy value, not used for photometry
                dec=0.0,  # Dummy value, not used for photometry
                x=star.x,
                y=star.y,
                magnitude=None,
                catalog=None,
                catalog_id=None,
            )
            temp_space_stars.append(temp_space_star)

        # Get SNR and counts for all stars at once
        star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, temp_space_stars, plot=False)

        # Update the detected stars with their counts
        for i, (_temp_star, _snr, counts) in enumerate(star_snr_results):
            detected_stars[i].counts = counts

        # Add to detections if not already present
        for star in detected_stars:
            if star not in frame.starfield.detections:
                frame.starfield.detections.append(star)

    # Calculate median shifts (more robust than mean)
    if x_shifts and y_shifts:
        median_x_shift = float(np.median(x_shifts))
        median_y_shift = float(np.median(y_shifts))
    else:
        median_x_shift = 0.0
        median_y_shift = 0.0

    logger.info(f"Global shift: x={median_x_shift:.2f}, y={median_y_shift:.2f} from {len(x_shifts)} matched stars")

    return median_x_shift, median_y_shift


def refine_sidereal_frame(frame: SiderealFrame) -> None:
    """Refine WCS for sidereal frames using catalog stars from brightest to dimmest.

    Args:
        frame (SiderealFrame): The sidereal frame containing the stars.

    """
    # Same contract as the anchor registration: the refinement kernel is sized from the
    # frame's measured PSF scale. Guarding here also covers the plotting-branch reads further
    # down this call chain, which would otherwise be the second place to fail on None.
    starfield = frame.starfield
    if starfield is None or starfield.detection_metadata is None:
        missing = "starfield" if starfield is None else "detection metadata"
        raise WcsPropagationError(
            f"Cannot refine the WCS of sidereal frame {frame.index}: no {missing} is present, so "
            "there is no measured PSF scale to size the refinement kernel."
        )

    convolved_image = convolve(
        frame.frame.data,
        sidereal_kernel(starfield.detection_metadata.pixel_fwhm),
        mode="same",
    )

    wcs_model = refine_sidereal_with_catalog_stars(frame, convolved_image)

    frame.starfield.wcs = wcs_model
    frame.starfield.wcs_metadata = WCSMetadata.from_wcsmodel(wcs_model)

    catalog_stars = catalog_stars_from_wcs(wcs_model)
    frame.starfield.catalog_stars = catalog_stars.stars

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=Path(settings.plotting.output_dir) / f"{frame.index}_side_kernel_3_refit.png",
        )


def refine_sidereal_with_catalog_stars(frame: SiderealFrame, convolved_image: np.ndarray) -> WCSModel:
    """Refine WCS for sidereal frames using catalog stars from brightest to dimmest.

    Similar to refine_wcs_with_catalog_stars but adapted for sidereal frames where
    stars are point sources rather than streaks.

    Args:
        frame (SiderealFrame): The sidereal frame containing the stars.
        convolved_image (np.ndarray): The convolved image (with a 2D Gaussian kernel).

    Returns:
        WCSModel: The refined WCS model, or None if refinement failed.

    """
    # First pass: Get global shift using astrometric fit stars
    global_shift_x, global_shift_y = get_global_shift_from_astrometric_stars(frame, convolved_image)

    # Apply the global shift to the WCS
    original_wcs_model = frame.starfield.wcs
    updated_wcs_model = shift_wcs(original_wcs_model, -global_shift_x, -global_shift_y)

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=Path(settings.plotting.output_dir) / f"{frame.index}_side_kernel_0_init.png",
        )

    # Update the WCS with the global shift
    frame.starfield.wcs = updated_wcs_model
    frame.starfield.wcs_metadata = WCSMetadata.from_wcsmodel(updated_wcs_model)

    # Update star positions based on the new WCS
    frame.starfield.astrometric_fit_stars = existing_stars_from_wcs(
        updated_wcs_model, frame.starfield.astrometric_fit_stars
    )
    catalog_stars = catalog_stars_from_wcs(updated_wcs_model, 14.0)
    frame.starfield.catalog_stars = catalog_stars.stars

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=Path(settings.plotting.output_dir) / f"{frame.index}_side_kernel_1_global.png",
        )

    logger.info("Refining WCS for sidereal frame with catalog stars")

    # Get catalog stars and sort by magnitude (brightest first)
    catalog_stars = frame.starfield.catalog_stars
    catalog_stars.sort(key=lambda star: star.magnitude if star.magnitude is not None else float("inf"))

    # Calculate proper SNRs using aperture photometry
    star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, catalog_stars, plot=False)

    # Store SNR with each star for later use
    for star, snr, counts in star_snr_results:
        star.snr = snr
        star.counts = counts  # Store counts while we're at it

    # Filter stars by SNR
    min_snr = 8.0  # Minimum SNR threshold
    filtered_catalog_stars = [star for star, snr, _ in star_snr_results if snr >= min_snr]

    # Estimate limiting magnitude
    limiting_magnitude = estimate_limiting_magnitude_from_photometry(frame, star_snr_results, min_snr)

    # Store the limiting magnitude in the starfield
    if hasattr(frame.starfield, "limiting_magnitude") and limiting_magnitude is not None:
        frame.starfield.limiting_magnitude = limiting_magnitude

    # Filter out stars that are too dim (beyond the limiting magnitude)
    if limiting_magnitude is not None:
        # Add a small margin to the limiting magnitude (0.5 mag)
        margin = 0.5
        filtered_catalog_stars = [
            star
            for star in filtered_catalog_stars
            if star.magnitude is None or star.magnitude <= limiting_magnitude - margin
        ]
        logger.info(
            f"Filtered out {len(filtered_catalog_stars) - len(filtered_catalog_stars)} stars beyond limiting magnitude {limiting_magnitude - margin:.2f}"
        )

    logger.info(
        f"Filtered catalog from {len(catalog_stars)} to {len(filtered_catalog_stars)} stars above SNR threshold"
    )

    # Minimum separation between stars to use for WCS refinement
    min_separation = 15  # For sidereal frames, we can use a smaller separation than for streaks

    # Get image dimensions
    height, width = frame.frame.data.shape

    # Store (detection, star, measured_x, measured_y) tuples during filtering
    filtered_star_data = []

    # Process filtered catalog stars from brightest to dimmest
    for star in filtered_catalog_stars:
        # Skip stars that are too close to already processed stars
        too_close = False
        for processed_data in filtered_star_data:
            processed_detection = processed_data[0]  # Get the detection from the tuple
            dist = np.sqrt((star.x - processed_detection.x) ** 2 + (star.y - processed_detection.y) ** 2)
            if dist < min_separation:
                too_close = True
                break

        if too_close:
            continue

        # Get current position
        x, y = star.x, star.y

        # Find the local maximum near this position
        search_radius = 10  # pixels
        x_min, x_max = max(0, int(x - search_radius)), min(width, int(x + search_radius + 1))
        y_min, y_max = max(0, int(y - search_radius)), min(height, int(y + search_radius + 1))

        # Extract local region
        local_region = convolved_image[y_min:y_max, x_min:x_max]

        if local_region.size == 0:
            continue

        # Find maximum in local region
        max_idx = np.argmax(local_region)
        local_y, local_x = np.unravel_index(max_idx, local_region.shape)

        # Convert to global coordinates
        measured_x = x_min + local_x
        measured_y = y_min + local_y

        # For sidereal frames, use circular aperture photometry instead of rectangular

        # Use a circular aperture with radius based on typical PSF size
        aperture_radius = 3.0  # Typical radius for point sources, adjust as needed
        aperture = CircularAperture((measured_x, measured_y), r=aperture_radius)

        # Background annulus
        bg_aperture = CircularAnnulus((measured_x, measured_y), r_in=aperture_radius * 1.5, r_out=aperture_radius * 2.5)

        # Perform photometry
        phot_table = aperture_photometry(frame.frame.data, aperture)
        bg_phot_table = aperture_photometry(frame.frame.data, bg_aperture)

        # Calculate background-subtracted counts
        aperture_sum = float(phot_table["aperture_sum"][0])
        bg_sum = float(bg_phot_table["aperture_sum"][0])
        bg_per_pixel = bg_sum / bg_aperture.area
        counts = aperture_sum - (bg_per_pixel * aperture.area)

        # Create a detection for this position
        detection = StarInImage(x=float(measured_x), y=float(measured_y), counts=counts)

        # Store the detection along with the star and measured position
        filtered_star_data.append((detection, star, measured_x, measured_y))

        logger.debug(
            f"Added star with magnitude {star.magnitude:.2f}, SNR {getattr(star, 'snr', 'N/A'):.1f} at ({measured_x:.1f}, {measured_y:.1f})"
        )

    logger.info(f"Found {len(filtered_star_data)} well-separated, high-SNR stars for WCS refinement")

    # If no stars passed the SNR / separation filters, return the WCS that was
    # already updated by the global-shift pass.  Returning updated_wcs_model
    # (not None) keeps refine_sidereal_frame's unconditional assignments safe.
    if len(filtered_star_data) == 0:
        logger.warning("No catalog stars passed SNR filter; returning WCS after global shift only")
        return updated_wcs_model

    # Update the detections list with just the detection objects
    frame.starfield.detections = [data[0] for data in filtered_star_data]

    # Now we can directly use the filtered_star_data for WCS fitting
    world_coords = []  # (ra, dec) pairs
    pixel_coords = []  # (x, y) pairs

    for _detection, star, measured_x, measured_y in filtered_star_data:
        world_coords.append((star.ra, star.dec))
        pixel_coords.append((measured_x, measured_y))

    logger.info(f"Using {len(world_coords)} well-separated star positions for WCS fitting")

    # Use astropy WCS fitting to refine the WCS

    # Convert world_coords to SkyCoord
    ra_values = [wc[0] for wc in world_coords]
    dec_values = [wc[1] for wc in world_coords]
    sky_coords = SkyCoord(ra_values, dec_values, unit=u.deg)

    # Convert pixel_coords to the format expected by fit_wcs_from_points
    x_values = np.array([pc[0] for pc in pixel_coords])
    y_values = np.array([pc[1] for pc in pixel_coords])

    # Make sure coordinates are in FITS convention (starting from 1,1)
    # If your coordinates are 0-indexed, add 1 to convert to FITS convention
    if x_values.min() < 1 or y_values.min() < 1:
        x_values = x_values + 1
        y_values = y_values + 1

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starlist=StarListImage(
                detections=frame.starfield.detections,
                image_metadata=frame.starfield.image_metadata,
            ),
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=Path(settings.plotting.output_dir) / f"{frame.index}_side_kernel_2_refit.png",
        )

    # Upstream catalog SIP refit (config-gated); None = Bayesian-engine default (astroeasy does SIP).
    sip_degree = settings.astrometry.sip_refit_order if settings.astrometry.sip_refit_enabled else None

    # Fit new WCS. Degenerate star geometry (e.g. matched pixel positions collapsing to ~one
    # point) makes fit_wcs_from_points raise "Initial guess is outside of provided bounds". That
    # is a recoverable refinement failure, not a fatal one: fall back to the global-shift WCS
    # (as the no-stars branch above does) rather than letting the ValueError fail the collect.
    # Mirrors refine_wcs_with_catalog_stars (the rate-frame sibling), which already catches this.
    try:
        refined_astropy_wcs = fit_wcs_from_points(
            (x_values, y_values), sky_coords, proj_point="center", sip_degree=sip_degree
        )
    except ValueError as e:
        logger.warning(
            "WCS could not be refined for sidereal frame due to an error in "
            "fit_wcs_from_points: %s. Using WCS after global shift only.",
            e,
        )
        return updated_wcs_model

    wcs_model = WCSModel.from_astropy_wcs(refined_astropy_wcs, image_shape=frame.frame.data.shape)

    logger.info("Successfully refined WCS for sidereal frame using catalog stars")
    return wcs_model


def refine_wcs_with_catalog_stars(frame: RateTrackFrame, convolved_image: np.ndarray) -> WCSModel:
    """Refine WCS using catalog stars from brightest to dimmest.

    Args:
        frame (RateTrackFrame): The frame containing the stars.
        convolved_image (np.ndarray): The convolved image.

    Returns:
        WCSModel: The refined WCS model, or None if refinement failed.

    """
    logger.info("Second pass: Refining WCS with catalog stars")

    # Get catalog stars and sort by magnitude (brightest first)
    catalog_stars = frame.starfield.catalog_stars
    catalog_stars.sort(key=lambda star: star.magnitude if star.magnitude is not None else float("inf"))

    # Calculate proper SNRs using aperture photometry
    star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, catalog_stars, plot=False)

    # Store SNR with each star for later use
    for star, snr, counts in star_snr_results:
        star.snr = snr
        star.counts = counts  # Store counts while we're at it

    # Filter stars by SNR
    min_snr = 8.0  # Minimum SNR threshold
    filtered_catalog_stars = [star for star, snr, _ in star_snr_results if snr >= min_snr]

    # Estimate limiting magnitude
    limiting_magnitude = estimate_limiting_magnitude_from_photometry(frame, star_snr_results, min_snr)

    # Store the limiting magnitude in the starfield
    if hasattr(frame.starfield, "limiting_magnitude") and limiting_magnitude is not None:
        frame.starfield.limiting_magnitude = limiting_magnitude

    # Filter out stars that are too dim (beyond the limiting magnitude)
    if limiting_magnitude is not None:
        # Add a small margin to the limiting magnitude (0.5 mag)
        margin = 0.5
        filtered_catalog_stars = [
            star
            for star in filtered_catalog_stars
            if star.magnitude is None or star.magnitude <= limiting_magnitude - margin
        ]
        logger.info(
            f"Filtered out {len(filtered_catalog_stars) - len(filtered_catalog_stars)} stars beyond limiting magnitude {limiting_magnitude - margin:.2f}"
        )

    logger.info(
        f"Filtered catalog from {len(catalog_stars)} to {len(filtered_catalog_stars)} stars above SNR threshold"
    )

    # Minimum separation between stars to use for WCS refinement
    min_separation = max(frame.streak.pixel_length, 15)  # At least 15 pixels

    # Get image dimensions
    height, width = frame.frame.data.shape

    # Variable-kernel setup: per-star distortion-aware local kernels (upstream feature, config-gated).
    use_variable_kernel = bool(getattr(frame.streak, "use_variable_kernel", False))
    vk_cfg = settings.streak.variable_kernel
    astropy_wcs_for_kernels = None
    diag_star_counter = 0
    if use_variable_kernel and frame.starfield.wcs is not None:
        astropy_wcs_for_kernels = frame.starfield.wcs.to_astropy_wcs()
        if astropy_wcs_for_kernels is not None:
            astropy_wcs_for_kernels.array_shape = (height, width)
            if settings.plotting.debug:
                try:
                    plot_variable_kernel_grid(
                        frame,
                        astropy_wcs_for_kernels,
                        nx=vk_cfg.diagnostics_grid_nx,
                        ny=vk_cfg.diagnostics_grid_ny,
                    )
                except Exception as e:
                    logger.warning("Variable-kernel grid diagnostic failed for frame %d: %s", frame.index, e)
    if use_variable_kernel and astropy_wcs_for_kernels is None:
        use_variable_kernel = False

    # Instead of just storing detections, store (detection, star, measured_x, measured_y) tuples
    # during the filtering process
    filtered_star_data = []

    # Process filtered catalog stars from brightest to dimmest
    for star in filtered_catalog_stars:
        # Skip stars that are too close to already processed stars
        too_close = False
        for processed_data in filtered_star_data:
            processed_detection = processed_data[0]  # Get the detection from the tuple
            dist = np.sqrt((star.x - processed_detection.x) ** 2 + (star.y - processed_detection.y) ** 2)
            if dist < min_separation:
                too_close = True
                break

        if too_close:
            continue

        # Get current position
        x, y = star.x, star.y
        measured_x = measured_y = None

        if use_variable_kernel and astropy_wcs_for_kernels is not None and frame.streak is not None:
            # Per-star distortion-aware local kernel, correlated in a cutout (upstream variable-kernel path).
            streak = frame.streak
            half_size = int(max(streak.pixel_length + 2 * streak.fwhm, streak.fwhm * 4) / 2) + 4
            x0, y0 = round(x), round(y)
            cx_min, cx_max = max(0, x0 - half_size), min(width, x0 + half_size + 1)
            cy_min, cy_max = max(0, y0 - half_size), min(height, y0 + half_size + 1)
            if cx_max > cx_min and cy_max > cy_min:
                image_cutout = frame.frame.data[cy_min:cy_max, cx_min:cx_max]
                try:
                    local_kernel = get_local_streak_kernel(
                        astropy_wcs_for_kernels,
                        streak,
                        x=float(x),
                        y=float(y),
                        scale_width=True,
                        upsample=100,
                        halo_fwhm=None,
                        halo_level=1e-3,
                        verbose=False,
                    )
                except Exception as e:
                    logger.warning(
                        "Local streak kernel failed at (%.1f, %.1f) frame %d: %s; using global.",
                        x,
                        y,
                        frame.index,
                        e,
                    )
                    local_kernel = None
                if local_kernel is not None:
                    correlation = convolve(image_cutout, local_kernel, mode="same")
                    if correlation.size > 0:
                        local_y, local_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
                        measured_x = cx_min + local_x
                        measured_y = cy_min + local_y
                        if settings.plotting.debug and diag_star_counter < vk_cfg.diagnostics_max_stars:
                            try:
                                plot_variable_kernel_star_diagnostic(
                                    frame,
                                    image_cutout,
                                    local_kernel,
                                    correlation,
                                    cx_min,
                                    cy_min,
                                    measured_x,
                                    measured_y,
                                    diag_star_counter,
                                )
                                diag_star_counter += 1
                            except Exception as e:
                                logger.warning("Variable-kernel star diagnostic failed frame %d: %s", frame.index, e)

        if measured_x is None:
            # Single global kernel: local maximum in the convolved image near the star.
            search_radius = 10  # pixels
            x_min, x_max = max(0, int(x - search_radius)), min(width, int(x + search_radius + 1))
            y_min, y_max = max(0, int(y - search_radius)), min(height, int(y + search_radius + 1))
            local_region = convolved_image[y_min:y_max, x_min:x_max]
            if local_region.size == 0:
                continue
            max_idx = np.argmax(local_region)
            local_y, local_x = np.unravel_index(max_idx, local_region.shape)
            measured_x = x_min + local_x
            measured_y = y_min + local_y

        # Extract counts using rectangular aperture
        counts, _ = extract_counts_with_rectangular_aperture(
            frame.frame.data, float(measured_x), float(measured_y), frame.streak
        )

        # Create a detection for this position
        detection = StarInImage(x=float(measured_x), y=float(measured_y), counts=counts)

        # Store the detection along with the star and measured position
        filtered_star_data.append((detection, star, measured_x, measured_y))

        logger.debug(
            f"Added star with magnitude {star.magnitude:.2f}, SNR {getattr(star, 'snr', 'N/A'):.1f} at ({measured_x:.1f}, {measured_y:.1f})"
        )

    # We can stop now if there are no stars
    if len(filtered_star_data) == 0:
        return None

    logger.info(f"Found {len(filtered_star_data)} well-separated, high-SNR stars for WCS refinement")

    # Update the detections list with just the detection objects
    frame.starfield.detections = [data[0] for data in filtered_star_data]

    # Now we can directly use the filtered_star_data for WCS fitting
    world_coords = []  # (ra, dec) pairs
    pixel_coords = []  # (x, y) pairs

    for _detection, star, measured_x, measured_y in filtered_star_data:
        world_coords.append((star.ra, star.dec))
        pixel_coords.append((measured_x, measured_y))

    logger.info(f"Using {len(world_coords)} well-separated star positions for WCS fitting")

    # Use astropy WCS fitting to refine the WCS

    # Convert world_coords to SkyCoord
    ra_values = [wc[0] for wc in world_coords]
    dec_values = [wc[1] for wc in world_coords]
    sky_coords = SkyCoord(ra_values, dec_values, unit=u.deg)

    # Convert pixel_coords to the format expected by fit_wcs_from_points
    x_values = np.array([pc[0] for pc in pixel_coords])
    y_values = np.array([pc[1] for pc in pixel_coords])

    # Make sure coordinates are in FITS convention (starting from 1,1)
    # If your coordinates are 0-indexed, add 1 to convert to FITS convention
    if x_values.min() < 1 or y_values.min() < 1:
        x_values = x_values + 1
        y_values = y_values + 1

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starlist=StarListImage(
                detections=frame.starfield.detections,
                image_metadata=frame.starfield.image_metadata,
            ),
            streak=frame.streak,
            output_file=Path(settings.plotting.output_dir) / f"{frame.index}_kernel_2_torefit.png",
        )

    # Upstream catalog SIP refit (config-gated) — see the sidereal sibling.
    sip_degree = settings.astrometry.sip_refit_order if settings.astrometry.sip_refit_enabled else None

    # Fit new WCS
    try:
        refined_astropy_wcs = fit_wcs_from_points(
            (x_values, y_values),
            sky_coords,
            proj_point="center",
            projection=frame.starfield.wcs.to_astropy_wcs(),
            sip_degree=sip_degree,
        )
    except ValueError as e:
        logger.warning(f"WCS could not be refined with catalog stars due to an error in fit_wcs_from_points: {e}.")
        return None

    wcs_model = WCSModel.from_astropy_wcs(refined_astropy_wcs, image_shape=frame.frame.data.shape)

    logger.info("Successfully refined WCS using catalog stars")
    return wcs_model


def ensure_star_counts(frame: RateTrackFrame) -> None:
    """Ensure all stars in the starfield have counts by extracting them if needed.

    Uses batch processing for efficiency.

    Args:
        frame (RateTrackFrame): The frame containing stars to check.

    """
    # Collect all stars that need counts
    stars_needing_counts = []

    # Check catalog stars
    for star in frame.starfield.catalog_stars:
        if hasattr(star, "counts") and star.counts is None:
            stars_needing_counts.append(star)

    # Check astrometric fit stars
    for star in frame.starfield.astrometric_fit_stars:
        if hasattr(star, "counts") and star.counts is None:
            stars_needing_counts.append(star)

    if not stars_needing_counts:
        return  # No stars need counts

    # Use the batch processing function to calculate counts for all stars at once
    star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, stars_needing_counts, plot=False)

    # Update the counts for each star
    for star, _, counts in star_snr_results:
        star.counts = counts

    logger.debug(f"Extracted counts for {len(star_snr_results)} stars in batch")
