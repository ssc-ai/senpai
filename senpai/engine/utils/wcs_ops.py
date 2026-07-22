"""WCS manipulation primitives.

The shared registration primitives (``shift_wcs``, ``shift_wcs_by_pixel_shift``,
``catalog_stars_from_wcs``, ``existing_stars_from_wcs``) are re-exported from
:mod:`senpai.engine.utils.propagate_wcs` (the ported engine implementations); the
distortion/scaling/radius helpers unique to this module keep their bodies here.
"""

import logging
import math

from senpai.engine.detection.jacobian import wcs_distortion_metrics
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.starfield import StarListSpace
from senpai.engine.utils.propagate_wcs import (  # noqa: F401  # re-exported API
    catalog_stars_from_wcs,
    existing_stars_from_wcs,
    shift_wcs,
    shift_wcs_by_pixel_shift,
)

logger = logging.getLogger(__name__)


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
    catalog_stars: StarListSpace, image_metadata: ImageMetadata, radius_factor: float | None
) -> StarListSpace:
    """Filter catalog stars to only those within a radius of the image center.

    Args:
        catalog_stars: StarListSpace object containing catalog stars.
        image_metadata: ImageMetadata object containing width and height.
        radius_factor: Radius as a fraction of the image circle (0.0 to 1.0),
            or None for no filtering.

    Returns:
        StarListSpace: Filtered catalog stars.

    Raises:
        ValueError: If ``radius_factor`` is not None and falls outside
            ``(0.0, 1.0]``.
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
