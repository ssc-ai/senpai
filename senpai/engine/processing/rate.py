"""Rate-track frame processing pipeline."""

import json
import logging
from pathlib import Path

import numpy as np

from senpai.astrometry import solve_field
from senpai.catalog.runner import query_catalog
from senpai.core.config import get_config
from senpai.engine.detection.point.satellite import extract_point_sources
from senpai.engine.detection.streak.rate_extraction import (
    build_streak_metadata,
    extract_rate_streak_measurement,
    extract_streak_centers_as_sources,
)
from senpai.engine.models.metadata import (
    DetectionMetadata,
    FrameMetadata,
    ImageMetadata,
    SeeingModel,
)
from senpai.engine.models.senpai import RateTrackFrame, RateTrackFrameSerializable
from senpai.engine.models.starfield import StarListImage
from senpai.engine.photometry.utils import measure_detection_photometry, measure_rate_starfield_photometry
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.plotting.photometry import plot_photometry_summary
from senpai.engine.utils.fits_io import extract_boresight_from_header
from senpai.engine.utils.frame_organization import extract_uct_time_from_header
from senpai.engine.utils.serialization import fits_header_to_jsonable

logger = logging.getLogger(__name__)


def process_rate_fits_rate(
    fits_image,
    *,
    run_id: str = "rate",
    attempt_wcs: bool = True,
    max_sources: int = 200,
    n_streaks: int = 10,
    photometry: bool = True,
) -> RateTrackFrameSerializable:
    """Process a single rate-track frame (standalone).

    .. deprecated::
        Use ``process_senpai_collect([fits_image])`` instead, which routes
        through the unified collect pipeline with full feature support
        (streak detection, catalog filtering, stamp confirmation).
    """
    import warnings

    warnings.warn(
        "process_rate_fits_rate is deprecated — use process_senpai_collect([image]) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    config = get_config()

    timestamp = extract_uct_time_from_header(fits_image.header)
    frame_metadata = FrameMetadata.from_header(fits_image.header)

    rate_frame = RateTrackFrame(
        frame=fits_image,
        index=0,
        timestamp=timestamp,
        frame_metadata=frame_metadata,
    )

    # 1) Measure the characteristic star streak in this rate-track frame
    measurement, psf, measured_fwhm = extract_rate_streak_measurement(
        rate_frame, n_streaks=n_streaks, initial_fwhm=None
    )

    if measurement is None:
        logger.warning("Failed to measure streak parameters for this rate frame.")
    else:
        if measurement.fwhm is None and measured_fwhm is not None:
            measurement.fwhm = float(measured_fwhm)

        if measurement.fwhm is None:
            measurement.fwhm = 4.0

        rate_frame.streak = build_streak_metadata(measurement)
        rate_frame.seeing = SeeingModel(
            pixel_fwhm=float(rate_frame.streak.fwhm),
            pixel_fwhm_stdev=0.0,
            n_measurements=1,
        )

        # Track rate in pixels/s from streak length / exposure
        exp = frame_metadata.exposure_time_seconds or fits_image.header.get("EXPTIME", 0)
        if exp and exp > 0:
            rate_frame.pixel_track_rate_per_second = float(rate_frame.streak.pixel_length) / float(exp)

    # 2) Persist streak artifacts
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if measurement is not None:
        with open(output_dir / "streak_measurement.json", "w") as f:
            json.dump(measurement.model_dump(mode="json"), f, indent=4)

    if psf is not None and (config.plotting.debug or config.plotting.review):
        plot_single_frame(
            psf,
            scale=False,
            output_file=output_dir / "rate_streak_psf.png",
        )

    # 3) Optional WCS attempt using streak-centroids as pseudo stars
    wcs_starfield = None
    if attempt_wcs:
        boresight_ra_degrees, boresight_dec_degrees = extract_boresight_from_header(fits_image.header)

        sources = extract_streak_centers_as_sources(
            fits_image.data,
            streak=rate_frame.streak,
            max_sources=max_sources,
        )

        # Plot detections going into astrometry (similar to sidereal_detections.png)
        if (config.plotting.debug or config.plotting.review) and sources:
            sources_for_plot = StarListImage(
                detections=sources,
                image_metadata=ImageMetadata(
                    width=fits_image.data.shape[1],
                    height=fits_image.data.shape[0],
                ),
            )
            plot_single_frame(
                fits_image.data,
                starlist=sources_for_plot,
                streak=rate_frame.streak,
                markersize=(rate_frame.streak.fwhm * 2 if rate_frame.streak else 10),
                output_file=output_dir / "rate_detections.png",
            )

        if sources:
            image_metadata = ImageMetadata(
                width=fits_image.data.shape[1],
                height=fits_image.data.shape[0],
                boresight_ra=boresight_ra_degrees,
                boresight_dec=boresight_dec_degrees,
                exposure_time=(
                    float(frame_metadata.exposure_time_seconds)
                    if frame_metadata.exposure_time_seconds is not None
                    else None
                ),
            )
            starlist = StarListImage(detections=sources, image_metadata=image_metadata)

            logger.info(
                "Attempting WCS solve from %d streak-centroid pseudo sources",
                len(sources),
            )
            try:
                wcs_starfield = solve_field(starlist)
            except Exception as e:
                logger.warning("WCS solve failed for rate frame: %s", e)
                wcs_starfield = None

            if wcs_starfield and wcs_starfield.wcs:
                # Query catalog to enable overlays/plotting and to keep outputs similar to sidereal.
                try:
                    catalog = query_catalog(wcs_starfield.wcs, max_stars=None, apply_sip=True)
                    wcs_starfield.catalog_stars = catalog.stars

                    # Merge non-None catalog metadata into existing image metadata
                    base_metadata = wcs_starfield.image_metadata.model_dump()
                    catalog_metadata = catalog.image_metadata.model_dump()
                    for key, value in catalog_metadata.items():
                        if value is not None:
                            base_metadata[key] = value
                    wcs_starfield.image_metadata = ImageMetadata(**base_metadata)
                except Exception as e:
                    logger.warning("Catalog query failed for rate-frame WCS: %s", e)

                rate_frame.starfield = wcs_starfield

                with open(output_dir / "rate_starfield.json", "w") as f:
                    json.dump(wcs_starfield.model_dump(mode="json"), f, indent=4)

                # Perform object detection if requested
                if config.detection.detect:
                    try:
                        # Ensure detection_metadata is set on starfield (needed by extract_point_sources)
                        # Use FWHM from streak measurement
                        if (
                            wcs_starfield.detection_metadata is None
                            and rate_frame.streak is not None
                            and rate_frame.streak.fwhm is not None
                        ):
                            wcs_starfield.detection_metadata = DetectionMetadata(
                                pixel_fwhm=float(rate_frame.streak.fwhm)
                            )
                            logger.info(
                                f"Set detection_metadata.pixel_fwhm={rate_frame.streak.fwhm:.2f} from streak measurement"
                            )

                        logger.info("Extracting point sources (satellites/objects)...")
                        rate_frame.detections = extract_point_sources(rate_frame)
                        logger.info(
                            f"Detected {len(rate_frame.detections.detections) if rate_frame.detections else 0} point sources"
                        )
                    except Exception as e:
                        logger.warning(f"Point source detection failed: {e}", exc_info=True)

                # Perform photometry if requested and WCS solution is successful
                photometry_results = None
                photometry_summary = None
                if photometry and rate_frame.streak is not None:
                    try:
                        photometry_results, photometry_summary = measure_rate_starfield_photometry(
                            fits_image,
                            wcs_starfield,
                            rate_frame.streak,
                            config.photometry,
                        )
                        logger.info(f"Photometry summary: {photometry_summary}")

                        # Set limiting magnitude from photometry results
                        # Prefer limiting_magnitude_50 (50% completeness) if available,
                        # otherwise use limiting_magnitude
                        if photometry_summary.limiting_magnitude_50 is not None and not np.isnan(
                            photometry_summary.limiting_magnitude_50
                        ):
                            wcs_starfield.limiting_magnitude = photometry_summary.limiting_magnitude_50
                            logger.info(
                                f"Set limiting magnitude to {photometry_summary.limiting_magnitude_50:.2f} "
                                f"(50% completeness) from photometry"
                            )
                        else:
                            wcs_starfield.limiting_magnitude = photometry_summary.limiting_magnitude
                            logger.info(
                                f"Set limiting magnitude to {photometry_summary.limiting_magnitude:.2f} from photometry"
                            )

                        # Save photometry results to JSON
                        from senpai.cli.common import serialize_photometry_to_json

                        serialize_photometry_to_json(
                            photometry_results, photometry_summary, output_dir / "photometry_results.json"
                        )

                        # Create photometry plots if enabled
                        if config.plotting.photometry:
                            plot_photometry_summary(
                                photometry_results,
                                photometry_summary,
                                fits_image.data.shape,
                                output_dir,
                                fits_image,
                            )
                    except Exception as e:
                        logger.warning(f"Photometry failed: {e}", exc_info=True)

                # Store photometry summary on the frame for API access
                if photometry_summary is not None:
                    from dataclasses import asdict

                    rate_frame.photometry_summary = asdict(photometry_summary)

                # Detection photometry
                if (
                    rate_frame.detections
                    and rate_frame.detections.detections
                    and photometry_summary is not None
                    and photometry_summary.zero_point is not None
                ):
                    try:
                        measure_detection_photometry(
                            fits_image,
                            rate_frame.detections,
                            photometry_summary.zero_point,
                            photometry_summary.zero_point_err,
                            exposure_time=(frame_metadata.exposure_time_seconds if frame_metadata else None),
                            config=config.photometry,
                            multiband_calibration=photometry_summary.multiband_calibration,
                            observation_filter=(frame_metadata.observation_filter if frame_metadata else None),
                        )
                    except Exception as e:
                        logger.warning(f"Detection photometry failed: {e}", exc_info=True)

                # Plot final solved frame (after photometry and detection)
                if config.plotting.debug or config.plotting.review:
                    plot_single_frame(
                        fits_image.data,
                        starfield=wcs_starfield,
                        streak=rate_frame.streak,
                        detections=rate_frame.detections,
                        output_file=output_dir / "rate_solved.png",
                        show_undistorted_catalog=False,
                    )

        else:
            logger.warning("No usable streak-centroid sources found; skipping WCS attempt.")

    # 4) Serialize rate frame (without embedding full image data)
    serializable = RateTrackFrameSerializable(
        starfield=rate_frame.starfield,
        streak=rate_frame.streak,
        seeing=rate_frame.seeing,
        hardware=rate_frame.hardware,
        detections=rate_frame.detections,
        frame_metadata=rate_frame.frame_metadata,
        original_frame_path=getattr(fits_image, "file_path", None),
        processed_frame_path=getattr(fits_image, "processed_file_path", None),
        original_frame_header=(fits_header_to_jsonable(fits_image.header)),
        processing_history=getattr(fits_image, "processing_history", None),
        correction_frames=getattr(fits_image, "correction_frames", None),
        index=rate_frame.index,
        timestamp=rate_frame.timestamp.isoformat(),
        pixel_track_rate_per_second=rate_frame.pixel_track_rate_per_second,
        photometry_summary=rate_frame.photometry_summary,
    )

    with open(output_dir / "rateframe.json", "w") as f:
        json.dump(serializable.model_dump(mode="json"), f, indent=4)

    logger.info("Wrote outputs to %s", str(output_dir))
    return serializable
