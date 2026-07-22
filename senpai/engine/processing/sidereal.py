"""Sidereal frame processing pipeline.

Core building block: preprocessing → point source extraction → astrometry solve →
catalog query. Returns a StarField with solved WCS and catalog stars attached;
WCS refinement runs in the collect pipeline after the starfield is attached to
its frame. FWHM statistics (which power photometry, seeing, and auto-scaling)
are measured here only when a downstream feature needs them.

Photometry, streak detection, file I/O, and plotting are handled by the
collect pipeline (``senpai.engine.processing.collect``).
"""

import logging
from pathlib import Path

from senpai.astrometry import solve_field, solve_field_fits
from senpai.catalog.runner import query_catalog
from senpai.core.config import get_config
from senpai.engine.detection.point.fwhm import measure_fwhm_from_catalog_stars
from senpai.engine.detection.point.sidereal import (
    extract_point_sources as extract_sidereal_sources,
)
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import (
    DetectionMetadata,
    FWHMMetadata,
    ImageMetadata,
)
from senpai.engine.models.starfield import StarField, StarListImage
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.utils.fits_io import extract_boresight_from_header
from senpai.engine.utils.preprocessing import (
    background_subtract,
    remove_column_and_row_medians,
)
from senpai.exceptions import SiderealSolveError

logger = logging.getLogger(__name__)


def process_astrometry_json_sidereal(sources: StarListImage, wcs: WCSModel | None = None) -> StarField:
    """Solve astrometry for an externally supplied source list.

    Args:
        sources (StarListImage): detected sources with pixel coordinates.
        wcs: optional existing WCS to verify/refine.

    Returns:
        StarField: the solve result.
    """
    wcs_starfield = solve_field(sources, wcs)

    return wcs_starfield


def _measure_fwhm_stats(
    fits_image: ProcessedFitsImage,
    wcs_starfield: StarField,
    initial_fwhm: float,
    sat_level: float | None,
) -> None:
    """Attach FWHM statistics to a solved starfield (feature enrichment).

    Populates ``wcs_starfield.fwhm_stats`` and upgrades ``detection_metadata`` with the
    catalog-measured median FWHM. Only called when photometry or auto-scaling needs it.

    Args:
        fits_image (ProcessedFitsImage): the solved frame.
        wcs_starfield (StarField): the solved starfield with catalog stars attached.
        initial_fwhm (float): FWHM estimate from source extraction.
        sat_level (float | None): frame saturation level measured during extraction.
    """
    config = get_config()
    if wcs_starfield.catalog_stars:
        fwhm_stats = measure_fwhm_from_catalog_stars(
            fits_image,
            wcs_starfield.catalog_stars,
            initial_fwhm,
            config,
            sat_level=sat_level,
        )
        median_fwhm = fwhm_stats.median_fwhm
    else:
        median_fwhm = initial_fwhm
        fwhm_stats = FWHMMetadata(
            n_measurements=1,
            median_fwhm=median_fwhm,
            mean_fwhm=median_fwhm,
            std_fwhm=0.0,
            min_fwhm=median_fwhm,
            max_fwhm=median_fwhm,
            fwhm_vs_position=[],
            fwhm_vs_magnitude=[],
            fwhm_vs_counts=[],
            is_oversampled=median_fwhm > config.calibrations.target_fwhm,
            recommended_scale_factor=(
                median_fwhm / config.calibrations.target_fwhm
                if median_fwhm > config.calibrations.target_fwhm
                else None
            ),
        )

    wcs_starfield.fwhm_stats = fwhm_stats
    wcs_starfield.detection_metadata = DetectionMetadata(
        pixel_fwhm=median_fwhm, fwhm_stats=fwhm_stats
    )


def process_astrometry_fits_sidereal(
    fits_image: ProcessedFitsImage, subtract_background: bool = True
) -> StarField | None:
    """Plate-solve a single sidereal frame and attach catalog stars.

    Preprocesses the image, extracts point sources, runs the astrometric solve, and
    queries the star catalog for the solved field of view. WCS refinement is NOT
    performed here — the collect pipeline refines after attaching the starfield to
    its frame.

    Args:
        fits_image (ProcessedFitsImage): the sidereal frame to process.
        subtract_background (bool): whether to perform background subtraction before
            source extraction. Defaults to True.

    Returns:
        StarField | None: the solved StarField with catalog stars attached, or None if
            no astrometric fit could be obtained (when failures are not configured to
            raise).

    Raises:
        SiderealSolveError: if the plate solve fails and
            ``astrometry.error_on_plate_solve_failure`` is enabled.
    """
    config = get_config()

    fits_image = remove_column_and_row_medians(fits_image)

    if subtract_background:
        logger.info("Performing background subtraction")
        fits_image.data = background_subtract(
            fits_image.data, box_size=50, filter_size=3, sigma=3.0
        )

    sources, fwhm, _rejected = extract_sidereal_sources(
        fits_image, fwhm_guess=1.0, max_detections=config.astrometry.max_sources
    )

    detection_metadata = DetectionMetadata(pixel_fwhm=fwhm)

    if config.plotting.debug:  # pragma: no cover
        plot_single_frame(
            fits_image.data,
            starlist=sources,
            markersize=fwhm,
            output_file=Path(config.plotting.output_dir) / "sidereal_detections.png",
        )

    # Boresight hints: frame metadata is populated at load (models.images.from_fits);
    # fall back to the config-driven header mapping for sensors it doesn't cover.
    if fits_image.metadata.boresight_ra is None or fits_image.metadata.boresight_dec is None:
        boresight_ra, boresight_dec = extract_boresight_from_header(fits_image.header)
        if fits_image.metadata.boresight_ra is None:
            fits_image.metadata.boresight_ra = boresight_ra
        if fits_image.metadata.boresight_dec is None:
            fits_image.metadata.boresight_dec = boresight_dec

    wcs_starfield = solve_field_fits(fits_image)
    if wcs_starfield.wcs is None:
        msg = "plate solve failed on sidereal frame: cannot process image set"
        if config.astrometry.error_on_plate_solve_failure:
            # Expected outcome on frames with too few/poor stars -- raise the typed error so the
            # API boundary logs a clean message (no stack trace) rather than looking like a crash.
            raise SiderealSolveError(msg)
        logger.warning(msg)
        return None

    catalog = query_catalog(wcs_starfield.wcs, max_stars=1000)

    image_metadata = wcs_starfield.image_metadata.model_dump()
    # Only update with non-None values so we preserve original exposure_time, etc.
    for key, value in catalog.image_metadata.model_dump().items():
        if value is not None:
            image_metadata[key] = value
    wcs_starfield.catalog_stars = catalog.stars
    wcs_starfield.image_metadata = ImageMetadata(**image_metadata)
    wcs_starfield.detection_metadata = detection_metadata

    # FWHM statistics power photometry, seeing propagation, and auto-scaling; they are
    # measured only when one of those features is enabled.
    if config.photometry.enable or config.calibrations.auto_scale_images:
        _measure_fwhm_stats(fits_image, wcs_starfield, fwhm, getattr(sources, "sat_level", None))

    return wcs_starfield
