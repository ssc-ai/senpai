"""Sidereal frame processing pipeline.

Core building block: point source extraction → astrometry solve → catalog query
→ FWHM measurement → WCS refinement.  Returns a StarField with solved WCS,
catalog stars, and FWHM stats.

Photometry, streak detection, file I/O, and plotting are handled by the
collect pipeline (``senpai.engine.processing.collect``).
"""

import logging
from typing import Literal

from senpai.astrometry import solve_field
from senpai.catalog.runner import query_catalog
from senpai.core.config import get_config
from senpai.engine.detection.point.fwhm import measure_fwhm_from_catalog_stars
from senpai.engine.detection.point.sidereal import extract_point_sources
from senpai.engine.models.metadata import (
    DetectionMetadata,
    FrameMetadata,
    FWHMMetadata,
    ImageMetadata,
)
from senpai.engine.models.senpai import SiderealFrame
from senpai.engine.models.starfield import StarField, StarListImage
from senpai.engine.utils.fits_io import extract_boresight_from_header
from senpai.engine.utils.frame_organization import extract_uct_time_from_header
from senpai.engine.utils.propagate_wcs import refine_sidereal_frame

logger = logging.getLogger(__name__)


def process_astrometry_json_sidereal(
    sources: StarListImage, wcs=None
) -> StarField:
    wcs_starfield = solve_field(sources, wcs)

    return wcs_starfield


def process_astrometry_fits_sidereal(
    fits_image,
    pipeline_mode: Literal["full", "detect_solve", "detect"] | None = None,
) -> StarField:
    """Process a sidereal frame: detect sources, solve astrometry, query catalog, measure FWHM.

    Returns a StarField with solved WCS, catalog stars, detection metadata, and FWHM stats.
    Photometry, plotting, and file I/O are NOT performed here — the collect pipeline
    handles those downstream.

    ``pipeline_mode`` trims the work for callers that only need a per-frame FWHM:
    ``'detect_solve'`` stops after the plate solve, and ``'detect'`` never attempts
    one. Both land in the same ``else:`` fallback below that a failed solve uses
    today, so the returned shape (detection FWHM in ``detection_metadata`` and a
    synthesized ``fwhm_stats``) is identical — reached on purpose rather than by
    failure. See ``AstrometryConfig.pipeline_mode``.

    Passing it explicitly overrides the configured mode FOR THIS CALL ONLY, so one
    process can run reduced-mode batches (an autofocus focus sweep) alongside full
    science batches without mutating global config. Omit it — the default — to use
    ``config.astrometry.pipeline_mode`` exactly as before.
    """
    config = get_config()
    pipeline_mode = pipeline_mode or config.astrometry.pipeline_mode

    # The half-flux radius is a focus-sweep metric: always measured in 'detect' mode (which is
    # what an autofocus sweep requests per frame), otherwise only when detection.measure_hfr opts
    # in, so science runs don't pay its extra segmentation pass.
    sources, initial_fwhm = extract_point_sources(
        fits_image,
        max_detections=config.astrometry.max_sources,
        measure_hfr=pipeline_mode == "detect" or config.detection.measure_hfr,
    )

    boresight_ra_degrees, boresight_dec_degrees = extract_boresight_from_header(
        fits_image.header
    )

    sources.image_metadata.boresight_ra = boresight_ra_degrees
    sources.image_metadata.boresight_dec = boresight_dec_degrees

    if pipeline_mode == "detect":
        wcs_starfield = StarField(
            wcs=None,
            detections=sources.detections,
            image_metadata=sources.image_metadata,
        )
    else:
        wcs_starfield = solve_field(sources)

        if pipeline_mode == "detect_solve":
            # Keep the WCS — StarField's validator derives wcs_metadata from it, so
            # consumers still get a plate scale — but report fit=False. The fit was
            # never refined or checked against a catalog, and every downstream pass in
            # the collect pipeline gates on `.fit`, which is what keeps this mode cheap.
            wcs_starfield.fit = False

    wcs_starfield.detection_metadata = DetectionMetadata(pixel_fwhm=initial_fwhm)

    if wcs_starfield.wcs and pipeline_mode == "full":
        # Create a SiderealFrame to pass to refine_sidereal_frame. A header-sparse
        # frame (no date) keeps a None timestamp rather than crashing refinement.
        try:
            timestamp = extract_uct_time_from_header(fits_image.header)
        except AttributeError:
            timestamp = None
        frame_metadata = FrameMetadata.from_header(fits_image.header)
        sidereal_frame = SiderealFrame(
            frame=fits_image,
            index=0,  # Single frame processing, so index 0
            timestamp=timestamp,
            starfield=wcs_starfield,
            frame_metadata=frame_metadata,
        )
        refine_sidereal_frame(sidereal_frame)

        wcs_starfield.wcs = sidereal_frame.starfield.wcs

        wcs_starfield.detection_metadata = DetectionMetadata(pixel_fwhm=initial_fwhm)

        # Query catalog without magnitude limits - we need all stars for photometry
        # The limiting magnitude will be determined from photometry results
        catalog = query_catalog(wcs_starfield.wcs, max_stars=None)

        # Merge catalog image metadata into the existing image metadata without
        # overwriting valid values (e.g. exposure time) from the original image.
        base_metadata = wcs_starfield.image_metadata.model_dump()
        catalog_metadata = catalog.image_metadata.model_dump()

        for key, value in catalog_metadata.items():
            # Only update with non-None values so we preserve original exposure_time, etc.
            if value is not None:
                base_metadata[key] = value

        wcs_starfield.catalog_stars = catalog.stars
        wcs_starfield.image_metadata = ImageMetadata(**base_metadata)

        # Ensure catalog stars have pixel coordinates using the current WCS (with SIP)
        # query_catalog already does this, but this ensures consistency
        if wcs_starfield.catalog_stars and wcs_starfield.wcs:
            from senpai.engine.utils.propagate_wcs import existing_stars_from_wcs

            wcs_starfield.catalog_stars = existing_stars_from_wcs(
                wcs_starfield.wcs, wcs_starfield.catalog_stars
            )

        # Use the new function to measure FWHM
        fwhm_stats = measure_fwhm_from_catalog_stars(
            fits_image, catalog.stars, initial_fwhm, config,
            sat_level=sources.sat_level,
        )
        wcs_starfield.fwhm_stats = fwhm_stats
        median_fwhm = fwhm_stats.median_fwhm

    else:
        # Fallback if no WCS solution — also the deliberate landing point for the
        # reduced pipeline_modes, which skip the catalog-based FWHM measurement.
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

    detection_metadata = DetectionMetadata(
        pixel_fwhm=median_fwhm,
        fwhm_stats=fwhm_stats,
        pixel_hfr=sources.hfr,
        n_hfr_measurements=sources.n_hfr,
    )

    wcs_starfield.detection_metadata = detection_metadata

    return wcs_starfield
