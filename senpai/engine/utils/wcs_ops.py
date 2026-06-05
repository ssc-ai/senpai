"""Pure WCS manipulation primitives — no image processing."""

import logging
import math

from senpai.catalog.runner import query_catalog
from senpai.engine.detection.jacobian import wcs_distortion_metrics
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.starfield import StarInSpace, StarListSpace

logger = logging.getLogger(__name__)


def shift_wcs(wcs_model: WCSModel, shift_x: float, shift_y: float) -> WCSModel:
    # Create a new WCSModel by copying the source model and updating CRPIX values
    # Use model_dump() and model_validate() to create a copy with updated values
    wcs_data = wcs_model.model_dump()

    # Update the CRPIX values with the shifts
    if hasattr(wcs_model, "CRPIX1") and hasattr(wcs_model, "CRPIX2"):
        wcs_data["CRPIX1"] = wcs_model.CRPIX1 - shift_x
        wcs_data["CRPIX2"] = wcs_model.CRPIX2 - shift_y

    # Create new WCS model from the updated data
    return WCSModel.model_validate(wcs_data)


def compute_wcs_distortion_metrics(
    wcs_model: WCSModel,
    image_shape: tuple[int, int],
    nx: int = 5,
    ny: int = 5,
) -> dict[str, float] | None:
    """Compute compact WCS distortion metrics for a given WCSModel.

    This wraps :func:`wcs_distortion_metrics` to work with our WCSModel and
    keeps only summary scalars suitable for logging and config-based decisions.
    """
    try:
        astropy_wcs = wcs_model.to_astropy_wcs()
        if astropy_wcs is None:
            return None

        # Ensure the WCS is aware of the image dimensions for grid sampling
        height, width = image_shape
        astropy_wcs.array_shape = (height, width)

        # Use a nominal unit rate in RA; metrics are relative and do not depend
        # sensitively on the absolute rate when used just for distortion gating.
        metrics = wcs_distortion_metrics(
            astropy_wcs, rate_ra=1.0, rate_dec=0.0, nx=nx, ny=ny
        )

        return {
            "delta_J": float(metrics["delta_J"]),
            "max_angle_variation_deg": float(metrics["max_angle_variation_deg"]),
            "max_length_variation_fraction": float(
                metrics["max_length_variation_fraction"]
            ),
        }
    except Exception as e:
        logger.warning(f"Failed to compute WCS distortion metrics: {e}")
        return None


def catalog_stars_from_wcs(
    wcs_model: WCSModel, limiting_magnitude: float | None = None
) -> StarListSpace:
    return query_catalog(wcs_model, faint_lim=limiting_magnitude)


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

    # Convert all coordinates at once for efficiency
    if ra_dec_list:
        pixel_coords = astropy_wcs.all_world2pix(ra_dec_list, 0)
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
            magnitudes=star.magnitudes,
            catalog=star.catalog,
            catalog_id=star.catalog_id,
            counts=star.counts,
            snr=star.snr,
            x=float(pixel_coords[i][0]),
            y=float(pixel_coords[i][1]),
        )
        updated_stars.append(updated_star)

    return updated_stars


def shift_wcs_by_pixel_shift(senpai_run, frame_shift):
    """Shift WCS from a source frame to a target frame using pixel offsets.

    Args:
        senpai_run: SenpaiRun containing all frames.
        frame_shift: FrameShift with source/target indices and pixel shifts.
    """
    from senpai.core.config import get_config
    from senpai.engine.models.astrometry import WCSMetadata, WCSStatus
    from senpai.engine.models.starfield import StarField

    # Get the source frame's WCS
    logger.info(
        f"Shifting WCS from frame {frame_shift.source_index} to {frame_shift.target_index}"
    )
    source_frame = senpai_run.get_frame_by_index(frame_shift.source_index)
    if source_frame.starfield.wcs_status == WCSStatus.NO_WCS:
        logger.error("Source frame WCS status is NO_WCS... no WCS to shift!")
        raise ValueError("Source frame WCS status is NO_WCS... no WCS to shift!")

    source_wcs_model = source_frame.starfield.wcs

    # Debug: Show initial WCS quality
    logger.info(
        f"Source frame WCS CRPIX: ({source_wcs_model.CRPIX1:.2f}, {source_wcs_model.CRPIX2:.2f})"
    )
    logger.info(
        f"Source frame has {len(source_frame.starfield.astrometric_fit_stars)} astrometric fit stars"
    )
    logger.info(
        f"Source frame has {len(source_frame.starfield.catalog_stars)} catalog stars"
    )

    # Get the target frame
    target_frame = senpai_run.get_frame_by_index(frame_shift.target_index)

    # Get the pixel shifts
    shift_x = frame_shift.x_shift
    shift_y = frame_shift.y_shift

    logger.info(f"Applying shift: ({shift_x:.2f}, {shift_y:.2f}) pixels")

    target_wcs_model = shift_wcs(source_wcs_model, shift_x, shift_y)

    logger.info(
        f"Target frame WCS CRPIX after shift: ({target_wcs_model.CRPIX1:.2f}, {target_wcs_model.CRPIX2:.2f})"
    )

    target_stars_astrometry = existing_stars_from_wcs(
        target_wcs_model, source_frame.starfield.astrometric_fit_stars
    )

    # Propagate catalog stars from source frame by re-projecting RA/Dec
    # to target pixel coords. This preserves the full catalog; re-querying
    # with a shifted WCS can return fewer stars due to catalog boundary effects,
    # leaving bright stars unaccounted for in downstream rejection.
    target_stars_catalog_list = existing_stars_from_wcs(
        target_wcs_model, source_frame.starfield.catalog_stars
    )
    # Filter to stars within image bounds
    if target_frame.frame is not None and hasattr(target_frame.frame, "data"):
        h, w = target_frame.frame.data.shape[:2]
        target_stars_catalog_list = [
            s for s in target_stars_catalog_list
            if 0 <= s.x < w and 0 <= s.y < h
        ]
    logger.info(
        "Propagated %d catalog stars from source frame (source had %d)",
        len(target_stars_catalog_list),
        len(source_frame.starfield.catalog_stars),
    )

    # Apply radius filtering if configured
    config = get_config()
    if config.astrometry.reduce_field_by_radius is not None:
        from senpai.engine.models.starfield import StarListSpace
        wrapped = StarListSpace(
            stars=target_stars_catalog_list,
            image_metadata=source_frame.starfield.image_metadata,
        )
        wrapped = filter_catalog_stars_by_radius(
            wrapped,
            target_frame.frame.metadata,
            config.astrometry.reduce_field_by_radius,
        )
        target_stars_catalog_list = wrapped.stars
        logger.info(
            "Filtered catalog stars to %i stars within %.2f%% of image circle",
            len(target_stars_catalog_list),
            config.astrometry.reduce_field_by_radius * 100,
        )

    # Build image metadata from source frame (same image dimensions, similar boresight)
    refined_image_metadata = source_frame.starfield.image_metadata.model_copy()
    refined_image_metadata.image_id = source_frame.starfield.image_metadata.image_id

    # Create the target starfield with the shifted WCS model
    target_frame.starfield = StarField(
        astrometric_fit_stars=target_stars_astrometry,
        catalog_stars=target_stars_catalog_list,
        detections=[],
        image_metadata=refined_image_metadata,
        fit=True,
        wcs=target_wcs_model,
        wcs_metadata=WCSMetadata.from_wcsmodel(
            target_wcs_model
        ),  # Keep the same metadata
        wcs_status=WCSStatus.PIXEL_SHIFTED_WCS,
        detection_metadata=source_frame.starfield.detection_metadata,
        astrometry=None,
        limiting_magnitude=source_frame.starfield.limiting_magnitude,
        fwhm_stats=source_frame.starfield.fwhm_stats,
    )

    logger.info(
        f"Shifted WCS from frame {frame_shift.source_index} to {frame_shift.target_index} "
        f"by ({shift_x}, {shift_y}) pixels"
    )


def scale_wcs_solution(wcs_model: WCSModel, scale_factor: float) -> WCSModel:
    """Scale a WCS solution back to original image dimensions.

    This function adjusts the WCS solution to account for image scaling,
    ensuring coordinates map correctly to the original unscaled image.

    Args:
        wcs_model (WCSModel): The WCS model to scale
        scale_factor (float): The factor by which the image was scaled down

    Returns:
        WCSModel: A new WCS model scaled to the original image dimensions
    """
    # Get the original WCS
    astropy_wcs = wcs_model.to_astropy_wcs()

    # Scale the CDELT values (degrees per pixel)
    astropy_wcs.wcs.cdelt /= scale_factor

    # Scale the CD matrix if it exists
    if hasattr(astropy_wcs.wcs, "cd"):
        astropy_wcs.wcs.cd /= scale_factor

    # Scale the CRPIX reference pixel coordinates
    astropy_wcs.wcs.crpix *= scale_factor

    # Create new WCS model with scaled parameters
    new_wcs_model = WCSModel.from_astropy_wcs(
        astropy_wcs,
        image_shape=(
            int(wcs_model.NAXIS2 * scale_factor),
            int(wcs_model.NAXIS1 * scale_factor),
        ),
    )

    return new_wcs_model


def filter_catalog_stars_by_radius(
    catalog_stars: StarListSpace, image_metadata, radius_factor: float | None
) -> StarListSpace:
    """
    Filter catalog stars to only include those within a specified radius of the image center.

    Args:
        catalog_stars: StarListSpace object containing catalog stars
        image_metadata: ImageMetadata object containing width and height
        radius_factor: Radius as a fraction of the image circle (0.0 to 1.0), or None for no filtering

    Returns:
        StarListSpace: Filtered catalog stars
    """
    if radius_factor is None:
        return catalog_stars

    if radius_factor <= 0.0 or radius_factor > 1.0:
        raise ValueError(
            f"radius_factor must be between 0.0 and 1.0, got {radius_factor}"
        )

    # Calculate image center
    center_x = image_metadata.width / 2.0
    center_y = image_metadata.height / 2.0

    # Calculate the radius of the circle contained by the image
    # For non-square frames, use the average of width and height
    avg_dimension = (image_metadata.width + image_metadata.height) / 2.0
    max_radius = avg_dimension / 2.0

    # Apply the radius factor
    filter_radius = max_radius * radius_factor

    # Filter catalog stars
    filtered_stars = []
    for star in catalog_stars.stars:
        if star.x is None or star.y is None:
            continue

        # Calculate distance from center
        dx = star.x - center_x
        dy = star.y - center_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= filter_radius:
            filtered_stars.append(star)

    return StarListSpace(
        stars=filtered_stars, image_metadata=catalog_stars.image_metadata
    )
