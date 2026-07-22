"""WCS propagation and refinement utilities for the SENPAI detection pipeline.

Provides helpers to shift a WCS by a pixel offset, refine it against catalog and
astrometric stars (for both rate-track and sidereal frames), perform aperture
photometry, and estimate the per-frame limiting magnitude.
"""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

# ``NoConvergence`` is astropy's own exception (raised by ``all_world2pix`` when the SIP inverse
# diverges). It is recoverable here -- we fall back to best-effort positions and warn -- so we catch
# the astropy class directly rather than wrapping it in a SenpaiError.
from astropy.wcs.wcs import NoConvergence
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    RectangularAnnulus,
    RectangularAperture,
    aperture_photometry,
)
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve

from senpai.catalog.runner import query_catalog
from senpai.engine.detection.kernels import sidereal_kernel
from senpai.engine.models.astrometry import WCSMetadata, WCSModel, WCSStatus
from senpai.engine.models.metadata import StreakMetadata
from senpai.engine.models.senpai import FrameShift, RateTrackFrame, SenpaiRun, SiderealFrame
from senpai.engine.models.starfield import (
    StarField,
    StarInImage,
    StarInSpace,
    StarListImage,
    StarListSpace,
)
from senpai.engine.plotting.images import plot_photometry_frame, plot_single_frame
from senpai.settings import settings

logger = logging.getLogger(__name__)


def shift_wcs(wcs_model: WCSModel, shift_x: float, shift_y: float) -> WCSModel:
    """Return a copy of a WCS model with its reference pixel shifted.

    Args:
        wcs_model (WCSModel): Source WCS model to copy.
        shift_x (float): Pixel shift to subtract from CRPIX1.
        shift_y (float): Pixel shift to subtract from CRPIX2.

    Returns:
        WCSModel: A new WCS model with updated CRPIX values (unchanged if the
            model has no CRPIX1/CRPIX2 attributes).
    """
    # Create a new WCSModel by copying the source model and updating CRPIX values
    # Use model_dump() and model_validate() to create a copy with updated values
    wcs_data = wcs_model.model_dump()

    # Update the CRPIX values with the shifts
    if hasattr(wcs_model, "CRPIX1") and hasattr(wcs_model, "CRPIX2"):
        wcs_data["CRPIX1"] = wcs_model.CRPIX1 - shift_x
        wcs_data["CRPIX2"] = wcs_model.CRPIX2 - shift_y

    # Create new WCS model from the updated data
    return WCSModel.model_validate(wcs_data)


def catalog_stars_from_wcs(
    wcs_model: WCSModel,
    limiting_magnitude: float | None = None,
    max_stars: int | None = None,
) -> StarListSpace:
    """Query the star catalog for the field described by a WCS model.

    Args:
        wcs_model (WCSModel): WCS model defining the field of view.
        limiting_magnitude (float | None): Faint magnitude limit for the query,
            or None to use the catalog default.
        max_stars (int | None): Maximum number of stars to return, or None for
            no cap.

    Returns:
        StarListSpace: Catalog stars projected into the image frame.
    """
    return query_catalog(wcs_model, faint_lim=limiting_magnitude, max_stars=max_stars)


def existing_stars_from_wcs(
    wcs_model: WCSModel, star_list: list[StarInSpace]
) -> list[StarInSpace]:
    """Update an existing list of stars to match a new WCS model.

    Args:
        wcs_model (WCSModel): The new WCS model to use
        star_list (list[StarInSpace]): The list of stars to update

    Returns:
        list[StarInSpace]: Updated list of stars with new pixel coordinates
    """
    # Convert WCS model to astropy WCS
    astropy_wcs = wcs_model.to_astropy_wcs()

    # Extract RA and Dec from all stars
    ra_dec_list = [(star.ra, star.dec) for star in star_list]

    # Convert all coordinates at once for efficiency. all_world2pix inverts the SIP distortion
    # iteratively and raises astropy's NoConvergence on a degenerate WCS; fall back to astropy's
    # best-effort solution so a single bad frame doesn't fail the whole detect (mirrors
    # runner.project_world_to_pixels).
    if ra_dec_list:
        try:
            pixel_coords = astropy_wcs.all_world2pix(ra_dec_list, 0)
        except NoConvergence as exc:
            logger.warning(
                "all_world2pix did not converge for %d/%d stars (degenerate WCS); using "
                "best-effort positions. %s",
                np.size(exc.divergent),
                len(ra_dec_list),
                exc,
            )
            pixel_coords = np.asarray(exc.best_solution)
    else:
        # Return empty list if no stars
        return []

    # Create new star list with updated pixel coordinates
    updated_stars = []
    for i, star in enumerate(star_list):
        # Create a new StarInSpace with the same celestial coordinates but updated pixel position
        updated_star = StarInSpace(
            ra=star.ra,
            dec=star.dec,
            magnitude=star.magnitude,
            catalog=star.catalog,
            catalog_id=star.catalog_id,
            x=float(pixel_coords[i][0]),
            y=float(pixel_coords[i][1]),
        )
        updated_stars.append(updated_star)

    return updated_stars


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
            "WCS status is not PIXEL_SHIFTED_WCS, skipping kernel convolution [call senpai.engine.utils.propagate_wcs.shift_wcs_by_pixel_shift first]"
        )
        raise ValueError("WCS status is not PIXEL_SHIFTED_WCS, skipping kernel convolution")

    # Get the kernel
    kernel = frame.streak.to_pyramoid()
    convolved_image = convolve(frame.frame.data, kernel, mode="same")

    # First pass: Get global shift using astrometric fit stars
    global_shift_x, global_shift_y = get_global_shift_from_astrometric_stars(
        frame, convolved_image
    )

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
        catalog_stars = catalog_stars_from_wcs(
            refined_wcs, limiting_magnitude=frame.starfield.limiting_magnitude
        )
        frame.starfield.catalog_stars = catalog_stars.stars

        if settings.plotting.debug:  # pragma: no cover
            plot_single_frame(
                frame.frame.data,
                starfield=frame.starfield,
                streak=frame.streak,
                output_file=Path(settings.plotting.output_dir)
                / f"{frame.index}_kernel_3_refined.png",
            )

    # Update WCS status
    frame.starfield.wcs_status = WCSStatus.KERNEL_REFINED_WCS

    # Ensure all stars have counts
    ensure_star_counts(frame)
    return refined_wcs is not None


def get_global_shift_from_astrometric_stars(
    frame: RateTrackFrame, convolved_image: np.ndarray
) -> tuple[float, float]:
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
        star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, temp_space_stars)

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

    logger.info(
        f"Global shift: x={median_x_shift:.2f}, y={median_y_shift:.2f} from {len(x_shifts)} matched stars"
    )

    return median_x_shift, median_y_shift


def refine_sidereal_frame(frame: SiderealFrame) -> None:
    """Refine WCS for sidereal frames using catalog stars from brightest to dimmest.

    Args:
        frame (SiderealFrame): The sidereal frame containing the stars.
    """
    convolved_image = convolve(
        frame.frame.data,
        sidereal_kernel(frame.starfield.detection_metadata.pixel_fwhm),
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
            output_file=Path(settings.plotting.output_dir)
            / f"{frame.index}_side_kernel_3_refit.png",
        )


def refine_sidereal_with_catalog_stars(
    frame: SiderealFrame, convolved_image: np.ndarray
) -> WCSModel:
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
    global_shift_x, global_shift_y = get_global_shift_from_astrometric_stars(
        frame, convolved_image
    )

    # Apply the global shift to the WCS
    original_wcs_model = frame.starfield.wcs
    updated_wcs_model = shift_wcs(original_wcs_model, -global_shift_x, -global_shift_y)

    if settings.plotting.debug:  # pragma: no cover
        plot_single_frame(
            frame.frame.data,
            starfield=frame.starfield,
            markersize=2 * frame.starfield.detection_metadata.pixel_fwhm,
            output_file=Path(settings.plotting.output_dir)
            / f"{frame.index}_side_kernel_0_init.png",
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
            output_file=Path(settings.plotting.output_dir)
            / f"{frame.index}_side_kernel_1_global.png",
        )

    logger.info("Refining WCS for sidereal frame with catalog stars")

    # Get catalog stars and sort by magnitude (brightest first)
    catalog_stars = frame.starfield.catalog_stars
    catalog_stars.sort(
        key=lambda star: star.magnitude if star.magnitude is not None else float("inf")
    )

    # Calculate proper SNRs using aperture photometry
    star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, catalog_stars)

    # Store SNR with each star for later use
    for star, snr, counts in star_snr_results:
        star.snr = snr
        star.counts = counts  # Store counts while we're at it

    # Filter stars by SNR
    min_snr = 8.0  # Minimum SNR threshold
    filtered_catalog_stars = [star for star, snr, _ in star_snr_results if snr >= min_snr]

    # Estimate limiting magnitude
    limiting_magnitude = estimate_limiting_magnitude_from_photometry(
        frame, star_snr_results, min_snr
    )

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
            dist = np.sqrt(
                (star.x - processed_detection.x) ** 2 + (star.y - processed_detection.y) ** 2
            )
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
        bg_aperture = CircularAnnulus(
            (measured_x, measured_y), r_in=aperture_radius * 1.5, r_out=aperture_radius * 2.5
        )

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

    logger.info(
        f"Found {len(filtered_star_data)} well-separated, high-SNR stars for WCS refinement"
    )

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
            output_file=Path(settings.plotting.output_dir)
            / f"{frame.index}_side_kernel_2_refit.png",
        )

    # Fit new WCS. Degenerate star geometry (e.g. matched pixel positions collapsing to ~one
    # point) makes fit_wcs_from_points raise "Initial guess is outside of provided bounds". That
    # is a recoverable refinement failure, not a fatal one: fall back to the global-shift WCS
    # (as the no-stars branch above does) rather than letting the ValueError fail the collect.
    # Mirrors refine_wcs_with_catalog_stars (the rate-frame sibling), which already catches this.
    try:
        refined_astropy_wcs = fit_wcs_from_points(
            (x_values, y_values), sky_coords, proj_point="center"
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
    catalog_stars.sort(
        key=lambda star: star.magnitude if star.magnitude is not None else float("inf")
    )

    # Calculate proper SNRs using aperture photometry
    star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, catalog_stars)

    # Store SNR with each star for later use
    for star, snr, counts in star_snr_results:
        star.snr = snr
        star.counts = counts  # Store counts while we're at it

    # Filter stars by SNR
    min_snr = 8.0  # Minimum SNR threshold
    filtered_catalog_stars = [star for star, snr, _ in star_snr_results if snr >= min_snr]

    # Estimate limiting magnitude
    limiting_magnitude = estimate_limiting_magnitude_from_photometry(
        frame, star_snr_results, min_snr
    )

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

    # Instead of just storing detections, store (detection, star, measured_x, measured_y) tuples
    # during the filtering process
    filtered_star_data = []

    # Process filtered catalog stars from brightest to dimmest
    for star in filtered_catalog_stars:
        # Skip stars that are too close to already processed stars
        too_close = False
        for processed_data in filtered_star_data:
            processed_detection = processed_data[0]  # Get the detection from the tuple
            dist = np.sqrt(
                (star.x - processed_detection.x) ** 2 + (star.y - processed_detection.y) ** 2
            )
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

    logger.info(
        f"Found {len(filtered_star_data)} well-separated, high-SNR stars for WCS refinement"
    )

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

    # Fit new WCS
    try:
        refined_astropy_wcs = fit_wcs_from_points(
            (x_values, y_values),
            sky_coords,
            proj_point="center",
            projection=frame.starfield.wcs.to_astropy_wcs(),
        )
    except ValueError as e:
        logger.warning(
            f"WCS could not be refined with catalog stars due to an error in fit_wcs_from_points: {e}."
        )
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
    star_snr_results = calculate_star_snrs_with_aperture_photometry(frame, stars_needing_counts)

    # Update the counts for each star
    for star, _, counts in star_snr_results:
        star.counts = counts

    logger.debug(f"Extracted counts for {len(star_snr_results)} stars in batch")


def shift_wcs_by_pixel_shift(senpai_run: SenpaiRun, frame_shift: FrameShift) -> None:
    """Propagate a source frame's WCS to a target frame via a pixel shift.

    Builds the target frame's starfield by shifting the source WCS by the frame
    shift's pixel offset and reprojecting the astrometric and catalog stars.

    Args:
        senpai_run (SenpaiRun): Run holding the frames, looked up by index.
        frame_shift (FrameShift): Source/target indices and the (x, y) pixel shift.

    Raises:
        ValueError: If the source frame has no WCS to shift.
    """
    # Get the source frame's WCS
    source_frame = senpai_run.get_frame_by_index(frame_shift.source_index)
    if source_frame.starfield.wcs_status == WCSStatus.NO_WCS:
        logger.error("Source frame WCS status is NO_WCS... no WCS to shift!")
        raise ValueError("Source frame WCS status is NO_WCS... no WCS to shift!")

    source_wcs_model = source_frame.starfield.wcs

    # Get the target frame
    target_frame = senpai_run.get_frame_by_index(frame_shift.target_index)

    # Get the pixel shifts
    shift_x = frame_shift.x_shift
    shift_y = frame_shift.y_shift

    target_wcs_model = shift_wcs(source_wcs_model, shift_x, shift_y)

    target_stars_astrometry = existing_stars_from_wcs(
        target_wcs_model, source_frame.starfield.astrometric_fit_stars
    )
    target_stars_catalog = catalog_stars_from_wcs(
        target_wcs_model, source_frame.starfield.limiting_magnitude
    )
    refined_image_metadata = target_stars_catalog.image_metadata
    refined_image_metadata.image_id = source_frame.starfield.image_metadata.image_id

    # Create the target starfield with the shifted WCS model
    target_frame.starfield = StarField(
        astrometric_fit_stars=target_stars_astrometry,
        catalog_stars=target_stars_catalog.stars,
        detections=[],
        image_metadata=refined_image_metadata,
        fit=True,
        wcs=target_wcs_model,
        wcs_metadata=WCSMetadata.from_wcsmodel(target_wcs_model),  # Keep the same metadata
        wcs_status=WCSStatus.PIXEL_SHIFTED_WCS,
        detection_metadata=source_frame.starfield.detection_metadata,
        astrometry=None,
        limiting_magnitude=source_frame.starfield.limiting_magnitude,
    )

    logger.info(
        f"Shifted WCS from frame {frame_shift.source_index} to {frame_shift.target_index} "
        f"by ({shift_x}, {shift_y}) pixels"
    )


def find_local_maxima(
    image: np.ndarray,
    min_distance: int = 30,
    threshold: float | None = None,
    max_detections: int | None = None,
) -> np.ndarray:
    """Find local maxima in an image with minimum separation distance.

    Args:
        image: 2D numpy array
        min_distance: Minimum pixel separation between maxima
        threshold: Optional intensity threshold
        max_detections: Maximum number of detections to return (returns brightest ones)

    Returns:
        Array of (y, x) coordinates of maxima
    """
    # Apply threshold if provided
    if threshold is not None:
        mask = image > threshold
        filtered_image = image * mask
    else:
        filtered_image = image.copy()

    # Find local maxima
    size = 2 * min_distance + 1

    # Apply maximum filter
    maximum_filtered = maximum_filter(filtered_image, size=size, mode="constant")

    # Find points that are local maxima
    maxima = (filtered_image == maximum_filtered) & (filtered_image > 0)

    # Get coordinates and values of maxima in one step
    y_coords, x_coords = np.where(maxima)

    if len(y_coords) == 0:
        return np.array([])

    # Get values at these coordinates
    values = filtered_image[y_coords, x_coords]

    # Sort by intensity (brightest first)
    sort_indices = np.argsort(-values)  # Negative for descending order

    # Limit to max_detections if specified
    if max_detections is not None and max_detections < len(sort_indices):
        sort_indices = sort_indices[:max_detections]

    # Return sorted coordinates
    return np.column_stack((y_coords[sort_indices], x_coords[sort_indices]))


def extract_counts_with_rectangular_aperture(
    image: np.ndarray,
    x: float,
    y: float,
    streak: StreakMetadata,
    background_annulus: bool = True,
) -> tuple[float, float]:
    """Extract counts from an image using a rectangular aperture aligned with a streak.

    Args:
        image: 2D numpy array containing the image data
        x: x-coordinate of the star center
        y: y-coordinate of the star center
        streak: Streak object containing length, width, and angle information
        background_annulus: Whether to subtract local background using an annulus

    Returns:
        counts: Background-subtracted counts within the aperture
        background: Local background level (per pixel)
    """
    # Create rectangular aperture aligned with the streak
    width = streak.fwhm * 4
    length = streak.pixel_length + streak.fwhm * 2
    theta = streak.radian_angle() + np.pi / 2  # Assuming angle is in radians

    # Create the aperture
    aperture = RectangularAperture((x, y), w=width, h=length, theta=theta)

    # Create background annulus if requested
    if background_annulus:
        # Make the annulus slightly larger than the aperture
        bg_aperture = RectangularAnnulus(
            (x, y), w_in=width, w_out=width + 4, h_in=length, h_out=length + 4, theta=theta
        )

    # Perform photometry

    phot_table = aperture_photometry(image, aperture)
    aperture_sum = float(phot_table["aperture_sum"][0])

    # Calculate background if requested
    if background_annulus:
        bg_phot_table = aperture_photometry(image, bg_aperture)
        bg_sum = float(bg_phot_table["aperture_sum"][0])
        bg_area = bg_aperture.area
        aperture_area = aperture.area

        # Calculate background per pixel
        background = bg_sum / bg_area

        # Subtract background from aperture sum
        counts = aperture_sum - (background * aperture_area)
    else:
        background = 0.0
        counts = aperture_sum

    return counts, background


def match_stars_to_detections(
    stars: list[StarInImage], detected_points: list[tuple[float, float]], max_distance: float = 20
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match catalog stars to detected points using bipartite matching.

    Args:
        stars: List of StarInImage objects
        detected_points: Array of (y, x) coordinates from local maxima detection
        max_distance: Maximum allowed matching distance in pixels

    Returns:
        matched_pairs: List of (star_idx, detection_idx) pairs
        unmatched_stars: List of star indices with no match
        unmatched_detections: List of detection indices with no match
    """
    if not stars or len(detected_points) == 0:
        return [], list(range(len(stars))), list(range(len(detected_points)))

    # Create distance matrix
    cost_matrix = np.zeros((len(stars), len(detected_points)))

    # Check if we have any valid matches at all
    all_infinite = True

    for i, star in enumerate(stars):
        if star is None:
            # If star is None, set all costs to infinity
            cost_matrix[i, :] = np.inf
            continue

        for j, (y, x) in enumerate(detected_points):
            # Calculate Euclidean distance
            dx = star.x - x
            dy = star.y - y
            distance = np.sqrt(dx**2 + dy**2)

            # Set cost to distance (no infinity cutoff here)
            cost_matrix[i, j] = distance
            all_infinite = False

    # If all costs are infinite, return empty matches
    if all_infinite:
        return [], list(range(len(stars))), list(range(len(detected_points)))

    # Solve the assignment problem
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        return [], list(range(len(stars))), list(range(len(detected_points)))

    # Filter out assignments with distance exceeding max_distance
    matched_pairs = []
    unmatched_stars = list(range(len(stars)))
    unmatched_detections = list(range(len(detected_points)))

    for i, j in zip(row_ind, col_ind, strict=False):
        if cost_matrix[i, j] <= max_distance:
            matched_pairs.append((i, j))
            unmatched_stars.remove(i)
            unmatched_detections.remove(j)

    return matched_pairs, unmatched_stars, unmatched_detections


def calculate_star_snrs_with_aperture_photometry(
    frame: RateTrackFrame | SiderealFrame, catalog_stars: list[StarInSpace]
) -> list[tuple[StarInSpace, float, float]]:
    """Calculate SNRs for catalog stars using proper aperture photometry.

    Args:
        frame: Frame containing image data (RateTrackFrame or SiderealFrame)
        catalog_stars: List of StarInSpace objects

    Returns:
        List of (star, snr, counts) tuples for all stars
    """
    # Determine frame type
    is_sidereal = isinstance(frame, SiderealFrame)

    # Filter stars with valid positions in the image bounds
    height, width = frame.frame.data.shape
    counts_array = frame.frame.data.copy()
    counts_array -= np.min(counts_array)
    margin = 10
    valid_stars = []
    positions = []

    for star in catalog_stars:
        if (
            star.x is not None
            and star.y is not None
            and margin <= star.x < width - margin
            and margin <= star.y < height - margin
        ):
            valid_stars.append(star)
            positions.append((star.x, star.y))

    if not valid_stars:
        return []

    results = []

    if is_sidereal:
        # For sidereal frames, use circular apertures (can process all at once)
        fwhm = frame.starfield.detection_metadata.pixel_fwhm
        radius = max(1.5 * fwhm, 3.0)  # Use at least 3 pixels radius

        apertures = CircularAperture(positions, r=radius)
        bg_apertures = CircularAnnulus(positions, r_in=radius * 1.5, r_out=radius * 2.5)

        # Perform photometry for all stars at once
        phot_table = aperture_photometry(counts_array, apertures)
        bg_phot_table = aperture_photometry(counts_array, bg_apertures)

        # Calculate background-subtracted counts and SNR for each star
        for i, star in enumerate(valid_stars):
            aperture_sum = float(phot_table["aperture_sum"][i])
            bg_sum = float(bg_phot_table["aperture_sum"][i])

            # Get areas - for multiple apertures, these are arrays
            bg_area = bg_apertures.area
            aperture_area = apertures.area

            # If we have multiple apertures, get the specific one for this star
            if hasattr(bg_area, "__len__"):
                bg_area = bg_area[i]
                aperture_area = aperture_area[i]

            # Calculate background per pixel and subtract from aperture
            bg_per_pixel = bg_sum / bg_area
            counts = aperture_sum - (bg_per_pixel * aperture_area)

            # Calculate noise (Poisson noise from source + background noise)

            bg_noise = np.sqrt(bg_per_pixel * aperture_area)

            source_noise = np.sqrt(max(0, counts))
            total_noise = np.sqrt(source_noise**2 + bg_noise**2)

            # Calculate SNR
            snr = counts / total_noise if total_noise > 0 else 0

            results.append((star, snr, counts))
    else:
        # For rate-track frames, process all stars at once with rotated apertures
        streak = frame.streak
        width_pixels = streak.fwhm * 4
        length_pixels = streak.pixel_length + streak.fwhm * 2
        theta = streak.radian_angle() + np.pi / 2  # photutils measurs angle differently than we do

        # Create apertures for all positions at once
        apertures = RectangularAperture(positions, w=width_pixels, h=length_pixels, theta=theta)
        bg_apertures = RectangularAnnulus(
            positions,
            w_in=width_pixels + 2,
            w_out=width_pixels + 6,
            h_in=length_pixels + 2,
            h_out=length_pixels + 6,
            theta=theta,
        )

        # Perform photometry for all stars at once. "exact" on these long rotated-rectangle
        # apertures is the slow path (~10-100x). method="subpixel", subpixels=10 reproduces the
        # exact catalog-star SNR closely enough that the count passing the min_snr gate (and hence
        # the WCS refit) matches exact, while staying ~10x faster on long streaks.
        phot_table = aperture_photometry(counts_array, apertures, method="subpixel", subpixels=10)
        bg_phot_table = aperture_photometry(
            counts_array, bg_apertures, method="subpixel", subpixels=10
        )

        # Calculate background-subtracted counts and SNR for each star
        for i, star in enumerate(valid_stars):
            aperture_sum = float(phot_table["aperture_sum"][i])
            bg_sum = float(bg_phot_table["aperture_sum"][i])

            # Get areas - for multiple apertures, these are arrays
            bg_area = bg_apertures.area
            aperture_area = apertures.area

            # If we have multiple apertures, get the specific one for this star
            if hasattr(bg_area, "__len__"):
                bg_area = bg_area[i]
                aperture_area = aperture_area[i]

            # Calculate background per pixel and subtract from aperture
            bg_per_pixel = bg_sum / bg_area
            counts = aperture_sum - (bg_per_pixel * aperture_area)

            # Calculate noise (Poisson noise from source + background noise)
            bg_noise = np.sqrt(bg_per_pixel * aperture_area)
            source_noise = np.sqrt(max(0, counts))
            total_noise = np.sqrt(source_noise**2 + bg_noise**2)

            # Calculate SNR
            snr = counts / total_noise if total_noise > 0 else 0

            results.append((star, snr, counts))
        if settings.plotting.debug:  # pragma: no cover
            plot_photometry_frame(
                counts_array,
                apertures=apertures,
                annuli=bg_apertures,
                output_file=Path(settings.plotting.output_dir)
                / f"frame_{frame.index}_aperture_photometry_stars.png",
            )

    return results


def estimate_limiting_magnitude_from_photometry(
    frame: RateTrackFrame | SiderealFrame,
    star_snr_results: list[tuple[StarInSpace, float, float]],
    min_snr: float = 5.0,
) -> float:
    """Estimate limiting magnitude using proper photometry results.

    Args:
        frame: Frame containing image data (RateTrackFrame or SiderealFrame)
        star_snr_results: List of (star, snr, counts) tuples
        min_snr: Minimum SNR threshold

    Returns:
        Estimated limiting magnitude
    """
    # Determine frame type
    is_rate_track = isinstance(frame, RateTrackFrame)

    # Extract magnitude and SNR pairs
    mag_snr_pairs = [
        (star.magnitude, snr) for star, snr, _ in star_snr_results if star.magnitude is not None
    ]

    if not mag_snr_pairs:
        return 15.0 if is_rate_track else 16.0  # Conservative default

    # Sort by magnitude
    mag_snr_pairs.sort(key=lambda x: x[0])

    # Try to fit a linear relationship between magnitude and log(SNR)
    magnitudes = np.array([m for m, _ in mag_snr_pairs])
    log_snrs = np.array([np.log10(max(s, 0.1)) for _, s in mag_snr_pairs])

    # Filter out stars with artificially capped SNR values (where log10(SNR) ≈ -1 from max(s, 0.1))
    valid_indices = [i for i, (_, snr) in enumerate(mag_snr_pairs) if snr > 0.1]
    if valid_indices:
        filtered_magnitudes = magnitudes[valid_indices]
        filtered_log_snrs = log_snrs[valid_indices]
    else:
        filtered_magnitudes = magnitudes
        filtered_log_snrs = log_snrs

    def _fallback_limiting_mag() -> float:
        """Conservative limiting magnitude used when a reliable linear fit isn't possible."""
        good_stars = [(m, s) for m, s in mag_snr_pairs if s >= min_snr]
        if good_stars:
            return max(m for m, _ in good_stars) + 0.5  # conservative margin
        return 12.0 if is_rate_track else 13.0

    # A degree-1 fit needs at least two stars spanning a range of magnitudes; degenerate
    # inputs make np.polyfit rank-deficient (and the fit unreliable), so fall back instead of
    # fitting them.
    if len(filtered_magnitudes) < 2 or np.ptp(filtered_magnitudes) == 0:
        return _fallback_limiting_mag()

    # Simple linear regression
    try:
        # Group stars by magnitude bins to calculate weights based on variance
        bin_width = 0.5  # magnitude bin width
        min_mag = np.floor(np.min(filtered_magnitudes))
        max_mag = np.ceil(np.max(filtered_magnitudes))
        bins = np.arange(min_mag, max_mag + bin_width, bin_width)

        # Initialize weights
        weights = np.ones_like(filtered_magnitudes)

        # Calculate variance in each bin and assign weights
        if len(filtered_magnitudes) > 10:  # Only do weighted fit if we have enough stars
            bin_indices = np.digitize(filtered_magnitudes, bins)

            # Track bin statistics for diagnostics
            bin_stats = []

            # Parameters to control weighting
            min_stars_per_bin = 3  # Minimum stars needed for reliable variance
            max_weight_factor = 10.0  # Cap on how much a bin's weight can exceed the median
            variance_floor = 0.01  # Minimum variance to prevent extreme weights

            # Calculate variances for all bins
            bin_variances = []
            bin_counts = []

            for bin_idx in range(1, len(bins)):
                bin_mask = bin_indices == bin_idx
                bin_count = np.sum(bin_mask)
                bin_counts.append(bin_count)

                if bin_count >= min_stars_per_bin:
                    bin_var = np.var(filtered_log_snrs[bin_mask])
                    # Add variance floor to prevent extreme weights
                    bin_var = max(bin_var, variance_floor)
                else:
                    # For sparse bins, use a high variance (low weight)
                    bin_var = 1.0  # Default high variance for sparse bins

                bin_variances.append(bin_var)
                bin_stats.append((bins[bin_idx - 1], bin_count, bin_var))

            # Calculate weights based on inverse variance
            if bin_variances:
                # Convert to inverse variance (higher for less scatter)
                inverse_variances = [1.0 / var for var in bin_variances]

                # Find median inverse variance for scaling
                median_inv_var = np.median(inverse_variances)

                # Apply weights to each bin, with capping
                for bin_idx in range(1, len(bins)):
                    bin_mask = bin_indices == bin_idx
                    if np.sum(bin_mask) > 0:
                        bin_var = bin_variances[bin_idx - 1]

                        # Weight is inverse of variance (higher weight = less scatter)
                        weight = 1.0 / bin_var

                        # Cap the weight relative to median
                        weight = min(weight, median_inv_var * max_weight_factor)

                        # Reduce weight for sparse bins
                        if bin_counts[bin_idx - 1] < min_stars_per_bin:
                            # Significantly downweight sparse bins
                            weight *= (bin_counts[bin_idx - 1] / min_stars_per_bin) ** 2

                        # Apply the weight to all stars in this bin
                        weights[bin_mask] = weight

        # Use weighted least squares for the fit
        coeffs = np.polyfit(filtered_magnitudes, filtered_log_snrs, 1, w=weights)
        slope, intercept = coeffs

        # Calculate magnitude where SNR drops to threshold
        limiting_mag = (np.log10(min_snr) - intercept) / slope
        limiting_mag = max(12, limiting_mag)

        if settings.plotting.debug:  # pragma: no cover
            # Create diagnostic plot
            from senpai.engine.plotting.images import plot_limiting_magnitude

            plot_limiting_magnitude(
                filtered_magnitudes=filtered_magnitudes,
                filtered_log_snrs=filtered_log_snrs,
                weights=weights,
                slope=slope,
                intercept=intercept,
                min_snr=min_snr,
                limiting_mag=limiting_mag,
                frame_index=frame.index,
            )

    except Exception:
        # Fallback if fitting fails
        return _fallback_limiting_mag()
    else:
        return limiting_mag
