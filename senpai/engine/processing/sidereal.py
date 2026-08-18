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
from senpai.engine.models.astrometry import WCSModel
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


def process_astrometry_json_sidereal(sources: StarListImage, wcs: WCSModel | None = None) -> StarField:
    """Solve a WCS from an already-detected source list, without touching pixels.

    This is the sources-only entry point: a caller that has done its own detection sends
    coordinates and gets a solved starfield back.
    """
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

    sextractor_mode = config.astrometry.source_extractor == "sextractor"
    if sextractor_mode:
        # Bayesian-engine sidereal solve (config-gated; default upstream = point_detector).
        # Column/row-median + box-50 background-subtract the frame IN PLACE. This is
        # load-bearing beyond detection: the subtracted frame is what the downstream
        # sidereal->rate registration cross-correlates against, so it fixes the WCS
        # anchor on background-limited fields (seen aliasing an anchor by ~2 arcmin).
        from senpai.engine.utils.preprocessing import (
            background_subtract,
            preprocess_float_dtype,
            remove_column_and_row_medians,
        )

        # Promote at the configured preprocessing precision: this frame feeds the
        # sidereal FWHM fit and (background-subtracted) the sidereal->rate
        # registration cross-correlation, both sensitive to float32's ~0.005 ADU
        # rounding at ADU scale (see calibrations.preprocess_float_dtype).
        fits_image = remove_column_and_row_medians(fits_image, dtype=preprocess_float_dtype())
        fits_image.data = background_subtract(fits_image.data, box_size=50, filter_size=3, sigma=3.0)

    # Point-detector pass: the upstream source list + the FWHM used for detection metadata
    # and downstream (seeing propagated to rate frames).
    sources, initial_fwhm = extract_point_sources(fits_image, max_detections=config.astrometry.max_sources)

    # Inputs to the catalog-median FWHM measurement (below). The upstream path uses the point
    # detector's FWHM as the fit seed and its measured saturation level.
    fwhm_seed = initial_fwhm
    fwhm_sat_level = sources.sat_level
    if sextractor_mode:
        # Feed the plate solve SExtractor's sources (1.5-sigma; astrometry.net's
        # --use-source-extractor parameters) instead of the point detector's — the
        # Bayesian engine's standard config. On background-limited fields the point
        # detector returns too few sources to solve; SExtractor recovers them. The
        # frame's image metadata (boresight, set below) is preserved.
        from senpai.engine.detection.point.sextractor import extract_sextractor_sources

        sextractor_detections = extract_sextractor_sources(
            fits_image, max_detections=config.astrometry.max_sources
        ).detections
        # Only replace the source list if SExtractor actually found something. It returns an
        # empty table when background estimation misbehaves, and assigning that unconditionally
        # would hand the solve zero sources on a frame where the point detector had some --
        # strictly worse than the default path. Unlikely at 1.5 sigma, but not impossible.
        if sextractor_detections:
            sources.detections = sextractor_detections
        else:
            logger.warning("SExtractor found no sources; keeping the point detector's source list for the solve")

        # Fork-faithful FWHM-measurement inputs. The Bayesian engine extracts sidereal
        # sources with daofind (fwhm_guess=1.0) and feeds THAT raw FWHM as the seed to the
        # catalog-median measurement, with sat_level=None (daofind sets none). The point
        # detector's seed + real sat_level shift the fitted median enough to move the
        # refined anchor by ~arcsec on marginal fields, which flips downstream
        # rate-registration CC peaks (seen as a backward-chain alias). Match the fork.
        from senpai.engine.detection.point.sidereal_daofind import (
            extract_point_sources_daofind,
        )

        try:
            _dfs, _df_fwhm, _ = extract_point_sources_daofind(fits_image, config.astrometry.max_sources, 1.0)
            if _df_fwhm and _df_fwhm > 0:
                fwhm_seed = _df_fwhm
        except Exception as exc:
            # Warn, do not whisper: this seed sizes the catalog-median FWHM fit, which sizes
            # the refinement kernel, which moves the refined anchor and with it the
            # downstream rate-registration correlation peak. Falling back still produces
            # output -- computed from a different seed than the configuration asked for --
            # so the run has to say so rather than leave it at debug level.
            logger.warning(f"daofind FWHM seed unavailable, keeping the configured seed: {exc}")
        fwhm_sat_level = None

    boresight_ra_degrees, boresight_dec_degrees = extract_boresight_from_header(fits_image.header)

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
        if sextractor_mode:
            # Fork-faithful sidereal refinement. The Bayesian engine runs with photometry
            # enabled, so its process_astrometry_fits_sidereal measures the catalog FWHM
            # and upgrades detection_metadata.pixel_fwhm to the catalog MEDIAN *before* its
            # (deferred) refine — i.e. the fork's refine kernel is sized on the median, not
            # the raw extraction FWHM. Reproduce that order here: query the catalog on the
            # solved (un-refined) WCS, measure the median, set it, then run the Bayesian
            # refine (the fork's aperture-SNR + fit_wcs_from_points path), not the port's
            # rewrite.
            _pre_catalog = query_catalog(wcs_starfield.wcs, max_stars=1000)
            _pre_stats = measure_fwhm_from_catalog_stars(
                fits_image,
                _pre_catalog.stars,
                fwhm_seed,
                config,
                sat_level=fwhm_sat_level,
            )
            wcs_starfield.detection_metadata = DetectionMetadata(pixel_fwhm=_pre_stats.median_fwhm)
            from senpai.engine.detection.streak.bayesian.wcs_refinement import (
                refine_sidereal_frame as bayesian_refine_sidereal_frame,
            )

            bayesian_refine_sidereal_frame(sidereal_frame)
        else:
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

            wcs_starfield.catalog_stars = existing_stars_from_wcs(wcs_starfield.wcs, wcs_starfield.catalog_stars)

        # Measure FWHM. The Bayesian engine measures fwhm_stats ONCE, on the
        # max_stars=1000 (brightest) catalog from the un-refined WCS — that same median feeds
        # the refine kernel, the rate_sidereal anchor kernel and the downstream seeing.
        # Re-measuring here on the max_stars=None (all-stars) refined-WCS catalog skews the
        # median high whenever the daofind seed is degenerate (a 24.5 px seed was measured on
        # one benchmark anchor frame): the extra faint stars fit badly to the huge seed, the
        # median inflates, the rate kernel oversizes, and the backward chain aliases. Reuse
        # that measurement in sextractor mode; the upstream path measures on the full
        # refined-WCS catalog.
        if sextractor_mode:
            fwhm_stats = _pre_stats
        else:
            fwhm_stats = measure_fwhm_from_catalog_stars(
                fits_image,
                catalog.stars,
                fwhm_seed,
                config,
                sat_level=fwhm_sat_level,
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
                median_fwhm / config.calibrations.target_fwhm if median_fwhm > config.calibrations.target_fwhm else None
            ),
        )
        wcs_starfield.fwhm_stats = fwhm_stats

    detection_metadata = DetectionMetadata(pixel_fwhm=median_fwhm, fwhm_stats=fwhm_stats)

    wcs_starfield.detection_metadata = detection_metadata

    return wcs_starfield
