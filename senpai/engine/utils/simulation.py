"""Synthetic frame generation helpers (Gaussian PSF rendering and star models)."""

import logging

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.stats import sigma_clipped_stats

from senpai.engine.models.starfield import StarField

logger = logging.getLogger(__name__)


def add_gaussian(source: dict, image: np.ndarray) -> None:
    """Render a 2D Gaussian source and add it into an image in place.

    Args:
        source: Gaussian parameters with keys ``x_mean``, ``y_mean``,
            ``x_stddev``, ``y_stddev`` and ``amplitude``.
        image: The 2D image array to add the rendered Gaussian into; modified
            in place within a bounding box of +/- 6 sigma around the center.
    """
    x_mean, y_mean = source["x_mean"], source["y_mean"]
    x_stddev, y_stddev = source["x_stddev"], source["y_stddev"]
    amplitude = source["amplitude"]

    # Calculate the bounding box (consider 3 sigma to cover 99.7% of the Gaussian)
    x_min = max(int(x_mean - 6 * x_stddev), 0)
    x_max = min(int(x_mean + 6 * x_stddev) + 1, image.shape[1])
    y_min = max(int(y_mean - 6 * y_stddev), 0)
    y_max = min(int(y_mean + 6 * y_stddev) + 1, image.shape[0])

    # Generate x and y indices
    x, y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing="xy")

    # Create Gaussian model and add to the image
    gaussian = Gaussian2D(amplitude, x_mean, y_mean, x_stddev, y_stddev)
    image[y_min:y_max, x_min:x_max] += gaussian(x, y)


def simulated_sidereal_frame(
    starfield: StarField,
    stddev: float = 1.0,
    max_stars: int = 20,
    constant_signal: bool = False,
) -> np.ndarray:
    """Generate a simulated sidereal frame from a StarField object.

    Args:
        starfield: StarField object containing star information.
        stddev: Standard deviation for the Gaussian PSF.
        max_stars: Maximum number of stars to simulate.
        constant_signal: Whether to use constant flux for all stars.

    Returns:
        Simulated image as a numpy array.
    """
    logger.info("Generating simulated sidereal frame")

    # Use current dimensions from metadata
    height = starfield.image_metadata.height
    width = starfield.image_metadata.width
    shape = (height, width)

    # Use current (scaled) FWHM
    if starfield.detection_metadata and starfield.detection_metadata.pixel_fwhm:
        fwhm = starfield.detection_metadata.pixel_fwhm
    else:
        # If we have a scale factor, use it to estimate scaled FWHM
        fwhm = 3.0

    # Convert FWHM to standard deviation
    stddev = fwhm / 2.355

    # Use current (scaled) star coordinates
    catalog_stars = [
        star
        for star in starfield.catalog_stars
        if star.x is not None and star.y is not None and star.magnitude is not None
    ]

    logger.info(f"Generating simulated frame with shape {shape}, FWHM={fwhm:.2f} pixels")

    # Sort stars by magnitude (brightest first) and limit to max_stars
    catalog_stars = sorted(catalog_stars, key=lambda star: star.magnitude)[:max_stars]

    # Create empty image
    image = np.zeros(shape, dtype=np.float32)

    # Add each star to the image
    for star in catalog_stars:
        x0, y0 = star.x, star.y

        # Skip stars outside the image bounds (with some margin)
        margin = 3 * stddev
        if x0 < -margin or x0 > width + margin or y0 < -margin or y0 > height + margin:
            continue

        # Scale intensity based on magnitude (arbitrary scale) or use constant signal
        intensity = 1000.0 if constant_signal else 10 ** (0.4 * (20 - star.magnitude))

        # Create source dictionary for add_gaussian
        source = {
            "x_mean": x0,
            "y_mean": y0,
            "x_stddev": stddev,
            "y_stddev": stddev,
            "amplitude": intensity,
        }

        try:
            add_gaussian(source, image)
        except Exception as e:
            logger.warning(f"Failed to add Gaussian source: {e}, source={source}")

    logger.info(f"Generated synthetic frame with shape: {image.shape}")
    return image


def build_star_model_image(
    image: np.ndarray,
    starfield: StarField,
    fwhm_override: float | None = None,
) -> np.ndarray:
    """Build a synthetic star-only image by measuring each catalog star in the data.

    For every catalog star with a pixel position, the function measures the peak
    amplitude from a small cutout in the real image and renders a Gaussian PSF
    at that location.  Subtracting the returned model from the real image yields
    a *residual* that highlights non-stellar signal (streaks, satellites, etc.).

    The approach is intentionally simple: even imperfect subtraction (80-90%) is
    fine because downstream directional-matched-filter detection rejects the
    symmetric residuals left by slightly mis-modelled point sources.

    Args:
        image: Real image data (2D numpy array).
        starfield: StarField with ``catalog_stars`` and ``detection_metadata``.
        fwhm_override: Override FWHM in pixels.  If *None*, uses
            ``starfield.detection_metadata.pixel_fwhm``.

    Returns:
        Synthetic star image (same shape as input, float64).
    """
    # Determine PSF width
    if fwhm_override is not None:
        fwhm = fwhm_override
    elif starfield.detection_metadata and starfield.detection_metadata.pixel_fwhm:
        fwhm = starfield.detection_metadata.pixel_fwhm
    else:
        fwhm = 3.0
        logger.warning("No FWHM available, using default %.1f pixels", fwhm)

    sigma = fwhm / 2.355

    # Robust background estimate (sigma-clipped median)
    _, bg_median, _ = sigma_clipped_stats(image, sigma=3.0, maxiters=5)

    stars = starfield.catalog_stars or []
    if not stars:
        logger.warning("No catalog stars available for star model")
        return np.zeros(image.shape, dtype=np.float64)

    model = np.zeros(image.shape, dtype=np.float64)
    box = max(int(np.ceil(3 * sigma)), 3)
    n_rendered = 0

    for star in stars:
        if star.x is None or star.y is None:
            continue

        ix, iy = round(star.x), round(star.y)

        # Skip stars outside image
        if ix < 0 or ix >= image.shape[1] or iy < 0 or iy >= image.shape[0]:
            continue

        # Measure peak amplitude from a small cutout
        y_min = max(0, iy - box)
        y_max = min(image.shape[0], iy + box + 1)
        x_min = max(0, ix - box)
        x_max = min(image.shape[1], ix + box + 1)

        cutout = image[y_min:y_max, x_min:x_max]
        if cutout.size == 0:
            continue

        amplitude = float(cutout.max()) - bg_median
        if amplitude <= 0:
            continue

        source = {
            "x_mean": star.x,
            "y_mean": star.y,
            "x_stddev": sigma,
            "y_stddev": sigma,
            "amplitude": amplitude,
        }
        try:
            add_gaussian(source, model)
            n_rendered += 1
        except Exception as e:
            logger.debug("Failed to render star model at (%.1f, %.1f): %s", star.x, star.y, e)

    logger.info(
        "Built star model image: %d/%d catalog stars rendered, FWHM=%.2f px",
        n_rendered,
        len(stars),
        fwhm,
    )
    return model
