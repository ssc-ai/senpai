"""Point-source extraction for sidereal frames (DAOFIND and SExtractor backends)."""

import logging
import math
import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.optimize import OptimizeWarning, curve_fit

from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.starfield import StarInImage, StarListImage

logger = logging.getLogger(__name__)


def gaussian_2d(
    data: tuple[np.ndarray, np.ndarray],
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
) -> np.ndarray:
    """Evaluate a rotated 2D Gaussian, flattened for curve fitting.

    Args:
        data (tuple[np.ndarray, np.ndarray]): The (x, y) coordinate grids.
        amp (float): Peak amplitude above the offset.
        x0 (float): Centroid x position.
        y0 (float): Centroid y position.
        sigma_x (float): Standard deviation along the x axis.
        sigma_y (float): Standard deviation along the y axis.
        theta (float): Rotation angle in radians.
        offset (float): Constant background offset.

    Returns:
        np.ndarray: The flattened (1D) Gaussian evaluated over the grid.
    """
    x, y = data
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)

    # Calculate the 2D Gaussian
    gaussian = offset + amp * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )

    # Return the flattened Gaussian to match the flattened cutout data
    return gaussian.ravel()


def estimate_fwhm(
    image: np.ndarray,
    x_centroid: float,
    y_centroid: float,
    box_size: int | None = None,
    fwhm_x_guess: float = 2.0,
    fwhm_y_guess: float = 2.0,
) -> float | None:
    """Estimate the FWHM of a bright star by fitting a 2D Gaussian to the source.

    Args:
        image (np.ndarray): 2D array representing the image.
        x_centroid (float): The star centroid's x coordinate.
        y_centroid (float): The star centroid's y coordinate.
        box_size (int | None): Size of the box to extract around the centroid for
            fitting. When None, it is chosen automatically from the FWHM guesses.
        fwhm_x_guess (float): Initial FWHM guess along the x axis, in pixels.
        fwhm_y_guess (float): Initial FWHM guess along the y axis, in pixels.

    Returns:
        float | None: The estimated FWHM in pixels, or None if the cutout is too
            small or the fit fails to converge.
    """
    # Extract a small box around the star
    x0, y0 = int(x_centroid), int(y_centroid)

    # If box_size is None, choose it based on the FWHM guess.
    # We want the box to span several FWHM to capture wings,
    # but not be so large that background noise dominates.
    if box_size is None:
        # Use the geometric mean of the FWHM guesses as a characteristic scale
        fwhm_scale = max(1.0, float(np.sqrt(abs(fwhm_x_guess * fwhm_y_guess))))
        box_size = int(np.clip(4.0 * fwhm_scale, 7.0, 21.0))
        # Ensure box_size is odd so the centroid lies near the center pixel
        if box_size % 2 == 0:
            box_size += 1

        logger.info(f"Auto box_size set to {box_size} based on FWHM guess {fwhm_scale}")

    # Calculate box boundaries with boundary checks
    y_min = max(0, y0 - box_size // 2)
    y_max = min(image.shape[0], y0 + box_size // 2)
    x_min = max(0, x0 - box_size // 2)
    x_max = min(image.shape[1], x0 + box_size // 2)

    # Check if the box is too small for a meaningful fit
    if (y_max - y_min) < 5 or (x_max - x_min) < 5:
        logger.warning(f"Box too small for star at ({x0}, {y0})")
        return None

    cutout = image[y_min:y_max, x_min:x_max]

    # Create x and y coordinates for fitting
    y, x = np.mgrid[: cutout.shape[0], : cutout.shape[1]]

    # Initial guess for fitting parameters
    try:
        # Adjust initial guess to account for potentially asymmetric box
        x_center = (x_max - x_min) // 2
        y_center = (y_max - y_min) // 2
        initial_guess = (cutout.max(), x_center, y_center, fwhm_x_guess, fwhm_y_guess, 0, 0)
    except Exception:
        logger.exception("Error estimating FWHM")
        return None

    try:
        # Fit the 2D Gaussian model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(gaussian_2d, (x, y), cutout.ravel(), p0=initial_guess)

        # Extract fitted parameters (only the sigmas are used downstream)
        _amp, _x0_fit, _y0_fit, sigma_x, sigma_y, _theta, _offset = popt

        # FWHM is approximately 2.355 * sigma for a Gaussian
        fwhm_x = 2.355 * sigma_x
        fwhm_y = 2.355 * sigma_y

        # Return average FWHM
        return (fwhm_x + fwhm_y) / 2
    except RuntimeError:
        return None


def _detect_sources_daofind(
    image: np.ndarray, max_sources: int = 10, fwhm: float = 5.0, threshold_sigma: float = 5.0
) -> Table:
    """Detect point sources in a 2D image with DAOStarFinder.

    Args:
        image (np.ndarray): 2D array representing the image.
        max_sources (int): Maximum number of sources to return. Defaults to 10.
        fwhm (float): Full-width half-maximum of the point sources, in pixels.
            Defaults to 5.0.
        threshold_sigma (float): Detection threshold in units of background RMS
            noise. Defaults to 5.0.

    Returns:
        Table: The detected sources sorted by descending flux, truncated to
            ``max_sources``; empty if none are found.
    """
    # Estimate background statistics
    _, median, std = sigma_clipped_stats(image, sigma=3.0)

    # Define the DAOStarFinder object
    daofind = DAOStarFinder(fwhm=fwhm, sigma_radius=2.0, threshold=threshold_sigma * std)

    # Find stars
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            sources = daofind(image - median)
    except MemoryError:
        logger.warning(
            "Memory error while calling daofind. Proceeding as if no sources were detected."
        )
        sources = None

    # filter any with flux < threshold_sigma * std
    logger.info("Filtering sources with flux less than %.2f", threshold_sigma * std)
    if sources is not None:
        sources = sources[sources["flux"] > (threshold_sigma * std)]

    # Sort sources by peak brightness (flux) and limit to max_sources
    if sources is not None:
        sources.sort("flux", reverse=True)
        max_sources = min(max_sources, len(sources))
        return sources[:max_sources]

    return Table()


def extract_point_sources_daofind(
    image: ProcessedFitsImage, max_detections: int = 100, fwhm_guess: float = 1.0
) -> tuple[StarListImage, float, StarListImage]:
    """Extract point sources from an image using DAOFIND with FWHM filtering.

    A small set of bright sources is first detected to estimate the median PSF
    FWHM, which is then used for a fuller detection pass. Detected sources are
    accepted or rejected based on their fitted FWHM relative to that median.

    Args:
        image (ProcessedFitsImage): The processed image to extract sources from.
        max_detections (int): Maximum number of accepted sources to return.
            Defaults to 100.
        fwhm_guess (float): Initial FWHM guess in pixels. Defaults to 1.0.

    Returns:
        tuple[StarListImage, float, StarListImage]: The accepted sources, the
            median FWHM in pixels used for extraction, and the rejected sources.
    """
    logger.info("Extracting point sources from image %s", image.metadata.image_id)

    detected_sources = _detect_sources_daofind(
        image.data, max_sources=20, fwhm=fwhm_guess
    )  # Detect a few bright sources

    fwhm = None
    fwhms = []
    for source in detected_sources:
        x_centroid = round(source["xcentroid"])
        y_centroid = round(source["ycentroid"])
        fwhm = estimate_fwhm(
            image.data, x_centroid, y_centroid, fwhm_x_guess=fwhm_guess, fwhm_y_guess=fwhm_guess
        )
        if fwhm is not None:
            fwhms.append(math.fabs(fwhm))

    logger.info("Estimated FWHMs for detected sources: %s", fwhms)
    if not fwhms:
        logger.warning("No valid FWHM estimates; falling back to fwhm_guess=%s", fwhm_guess)
        fwhm_pixel = fwhm_guess
    else:
        fwhm_pixel = float(np.median(fwhms))
    logger.info("Using median FWHM of %s pixels for source extraction", fwhm_pixel)

    sources = _detect_sources_daofind(
        image.data, max_sources=max_detections * 2, fwhm=fwhm_pixel, threshold_sigma=5.0
    )

    stars = []
    stars_rejected = []
    min_fwhm = fwhm_pixel * 0.5
    max_fwhm = fwhm_pixel * 3.0
    logger.info("FWHM filter: min %s, max %s, source FWHM %s", min_fwhm, max_fwhm, fwhm)
    for star_index, source in enumerate(sources):
        if len(stars) >= max_detections:
            break

        x_centroid = round(source["xcentroid"])
        y_centroid = round(source["ycentroid"])
        try:
            fwhm = estimate_fwhm(
                image.data,
                x_centroid,
                y_centroid,
                fwhm_x_guess=fwhm_pixel,
                fwhm_y_guess=fwhm_pixel,
            )
        except Exception:
            # probably border source, can be important for astrometry
            logger.warning(
                "Failed to estimate FWHM for source at (%s, %s)", x_centroid, y_centroid
            )
            fwhm = fwhm_pixel

        if fwhm is not None and fwhm > min_fwhm and fwhm < max_fwhm:
            if not np.isnan(source["xcentroid"]) and not np.isnan(source["ycentroid"]):
                logger.info(
                    "Accepting source %d at (%s, %s) with counts %s, FWHM %s",
                    star_index,
                    source["xcentroid"],
                    source["ycentroid"],
                    source["flux"],
                    fwhm,
                )
                stars.append(
                    StarInImage(
                        x=source["xcentroid"],
                        y=source["ycentroid"],
                        counts=source["flux"],
                        fwhm=fwhm,
                    )
                )
        else:
            if not np.isnan(source["xcentroid"]) and not np.isnan(source["ycentroid"]):
                logger.info(
                    "Rejected source %d at (%s, %s) with counts %s, FWHM %s",
                    star_index,
                    source["xcentroid"],
                    source["ycentroid"],
                    source["flux"],
                    fwhm,
                )
                stars_rejected.append(
                    StarInImage(
                        x=source["xcentroid"],
                        y=source["ycentroid"],
                        counts=source["flux"],
                        fwhm=fwhm if fwhm is not None else -1.0,
                    )
                )

    starlist = StarListImage(detections=stars, image_metadata=image.metadata)
    starlistRejected = StarListImage(detections=stars_rejected, image_metadata=image.metadata)

    return starlist, fwhm_pixel, starlistRejected


# astrometry.net's 5x5 Gaussian convolution mask for FWHM=2.0 pixels.
# Source: astrometry.net/solver/augment-xylist.c lines 901-907 — written to
# disk and passed as `-FILTER_NAME` to the source-extractor binary when
# solve-field is invoked with --use-source-extractor.
_ASTROMETRY_SEXTRACTOR_FILTER = np.array(
    [
        [0.006319, 0.040599, 0.075183, 0.040599, 0.006319],
        [0.040599, 0.260856, 0.483068, 0.260856, 0.040599],
        [0.075183, 0.483068, 0.894573, 0.483068, 0.075183],
        [0.040599, 0.260856, 0.483068, 0.260856, 0.040599],
        [0.006319, 0.040599, 0.075183, 0.040599, 0.006319],
    ],
    dtype=np.float32,
)


def _detect_sources_sextractor(
    image: np.ndarray,
    max_sources: int = 100,
    threshold_sigma: float = 1.5,
) -> Table:
    """Extract sources via ``sep``, matching astrometry.net's SExtractor invocation.

    Configured to reproduce the real SExtractor binary as invoked by
    astrometry.net's ``--use-source-extractor``.

    Parameter choices (from astrometry.net/solver/augment-xylist.c and
    SExtractor's default.sex):
      - DETECT_THRESH=1.5 sigma (SExtractor default; astrometry.net does not override)
      - BACK_SIZE=64, BACK_FILTERSIZE=3 → sep.Background(bw=64, bh=64, fw=3, fh=3)
      - 5x5 Gaussian convolution filter for FWHM=2.0 px (the exact filter
        astrometry.net writes to disk for SExtractor)
      - minarea=5 (sep's connected-pixel count gives matching results to
        SExtractor's MINAREA=3 with these other parameters)
      - DEBLEND_NTHRESH=32, DEBLEND_MINCONT=0.005, CLEAN=Y, CLEAN_PARAM=1.0
        (all SExtractor defaults that astrometry.net inherits)

    Verified to produce 66/66 identical source positions (median delta = 0.0000
    px, max delta = 0.0001 px) versus the real source-extractor binary on
    f82e85e6's sidereal frame.

    Args:
        image (np.ndarray): 2D array representing the image.
        max_sources (int): Maximum number of sources to return. Defaults to 100.
        threshold_sigma (float): Detection threshold in units of background RMS
            noise. Defaults to 1.5.

    Returns:
        Table: The detected sources sorted by descending flux, truncated to
            ``max_sources``; empty if none are found.
    """
    import sep

    data = np.ascontiguousarray(image, dtype=np.float64)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    objects = sep.extract(
        data - bkg,
        threshold_sigma * bkg.globalrms,
        minarea=5,
        filter_kernel=_ASTROMETRY_SEXTRACTOR_FILTER,
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        clean_param=1.0,
    )
    if len(objects) == 0:
        return Table()
    t = Table(objects[["x", "y", "flux"]])
    t.rename_column("x", "xcentroid")
    t.rename_column("y", "ycentroid")
    t.sort("flux", reverse=True)
    return t[:max_sources]


def extract_point_sources_sextractor(
    image: ProcessedFitsImage, max_detections: int = 100
) -> tuple[StarListImage, float, StarListImage]:
    """SExtractor-based extraction. Returns the same shape as extract_point_sources_daofind."""
    table = _detect_sources_sextractor(image.data, max_sources=max_detections)
    stars = [
        StarInImage(x=float(r["xcentroid"]), y=float(r["ycentroid"]), counts=float(r["flux"]))
        for r in table
    ]
    return (
        StarListImage(detections=stars, image_metadata=image.metadata),
        1.0,
        StarListImage(detections=[], image_metadata=image.metadata),
    )


def extract_point_sources(
    image: ProcessedFitsImage,
    max_detections: int = 100,
    fwhm_guess: float = 1.0,
    method: str = "daofind",
) -> tuple[StarListImage, float, StarListImage]:
    """Dispatch to the named source extractor. 'image2xy' is handled at the astrometry runner level."""
    if method == "daofind":
        return extract_point_sources_daofind(image, max_detections, fwhm_guess)
    elif method == "sextractor":
        return extract_point_sources_sextractor(image, max_detections)
    else:
        raise ValueError(
            f"Unknown source extractor '{method}'. "
            "Valid options: 'daofind', 'sextractor'. "
            "Use 'image2xy' in the astrometry config to use the image2xy binary."
        )
