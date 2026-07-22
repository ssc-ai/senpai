"""SENPAI collect pipeline — multi-frame sidereal+rate processing.

The core registration flow (sidereal solve → shift chain → per-frame WCS
refinement → point-source detection) is the ported detection-engine implementation; the
feature stages around it (calibration preprocessing, auto-scaling, photometry,
sidereal non-catalog detections, streak detection/correlation, review plots)
are config-gated and degrade gracefully when disabled.
"""

import logging
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from senpai.core.config import get_config
from senpai.engine.detection.jacobian import wcs_distortion_metrics
from senpai.engine.detection.point.satellite import extract_point_sources
from senpai.engine.detection.streak.frame_shift import (
    enforce_chain_consistency,
    solve_shift,
)
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import SeeingModel, TrackMode
from senpai.engine.models.senpai import RateTrackFrame, SenpaiRun, SiderealFrame
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.processing.sidereal import process_astrometry_fits_sidereal
from senpai.engine.utils.memory import reclaim_process_memory
from senpai.engine.utils.preprocessing import scale_starfield_coordinates
from senpai.engine.utils.propagate_wcs import (
    refine_sidereal_frame,
    refine_wcs_by_kernel_convolution,
    shift_wcs_by_pixel_shift,
)
from senpai.exceptions import SiderealSolveError, WcsPropagationError

if TYPE_CHECKING:
    from senpai.engine.detection.streak.sidereal_streak import StreakCandidate
    from senpai.engine.models.starfield import SatelliteInImage

logger = logging.getLogger(__name__)


def require_registered_rate_frames(senpai_run: SenpaiRun) -> None:
    """Verify the analysis chain registered at least one rate-track frame to a WCS solution.

    A single recoverable shift failure is routed around, but if a cascade leaves *no* rate frame
    registered the collect produced nothing usable. That is an unrecoverable outcome and must be
    surfaced as a meaningful error rather than a silently empty result.

    Args:
        senpai_run (SenpaiRun): The run to assess after its analysis chain has completed.

    Raises:
        WcsPropagationError: If the run has rate-track frames but none ended up registered to a
            WCS solution.
    """
    if not senpai_run.rate_track_frames:
        return

    registered = [
        frame
        for frame in senpai_run.rate_track_frames
        if frame.starfield is not None and frame.starfield.wcs is not None
    ]
    if registered:
        return

    failed_shifts = [
        f"{shift.source_index}->{shift.target_index}"
        for shift in senpai_run.frame_shifts_failed
        + [s for s in senpai_run.frame_shifts if s.processed and not s.is_valid]
    ]
    raise WcsPropagationError(
        "No rate-track frame could be registered to the sidereal WCS solution; the analysis "
        f"chain failed to propagate ({len(failed_shifts)} shift(s) failed: "
        f"{', '.join(failed_shifts) or 'none recorded'}). See warnings for per-shift causes."
    )


def _parse_image_set_id(header: Mapping[str, object]) -> str:
    """Resolve the image-set id from a frame's FITS header.

    Prefer an explicit id header when one is populated; otherwise parse it from the ``ORCHCOMM``
    provenance card, whose value is shaped ``&<image_set_id>@[sensor]#[frame:n]%[...]``. Returns
    ``"unknown"`` when neither yields an id.

    Args:
        header (Mapping[str, object]): the frame's FITS header (or an empty mapping).

    Returns:
        str: the image-set id, or ``"unknown"`` when none can be resolved.
    """
    explicit = next(
        (str(header[k]) for k in ("IMGSETID", "IMAGESETID", "IMAGEID") if header.get(k)),
        None,
    )
    if explicit:
        return explicit
    orchcomm = header.get("ORCHCOMM")
    if orchcomm and (match := re.match(r"\s*&([^@]+)@", str(orchcomm))):
        return match.group(1)
    return "unknown"


def _log_collect_summary(senpai_run: SenpaiRun, file_list: list[ProcessedFitsImage]) -> None:
    """Log one INFO line identifying the collect, for after-the-fact reproduction from logs.

    The image-set id is resolved from an explicit id header (``IMGSETID``/``IMAGESETID``/
    ``IMAGEID``) or, failing that, parsed from the ``ORCHCOMM`` provenance card. Sensor and
    observation-time range, plus site/object hints, round out the record. ``sensor`` is
    best-effort: frames may name the sensor under any of several headers, so the first
    populated candidate is used.

    Args:
        senpai_run (SenpaiRun): the organized run (frames carry parsed observation timestamps).
        file_list (list[ProcessedFitsImage]): the collect's frames (headers carry sensor/site info).
    """
    header = file_list[0].header if file_list else {}
    image_set_id = _parse_image_set_id(header)
    sensor = next(
        (
            str(header[k])
            for k in ("SENID", "TELESCOP", "OBSERVAT", "SENSOR", "ORIGIN", "INSTRUME")
            if header.get(k)
        ),
        "unknown",
    )
    timestamps = [
        f.timestamp
        for f in (*senpai_run.sidereal_frames, *senpai_run.rate_track_frames)
        if f.timestamp is not None
    ]
    obs_start = min(timestamps).isoformat() if timestamps else "unknown"
    obs_end = max(timestamps).isoformat() if timestamps else "unknown"
    logger.info(
        "Processing collect: image_set_id=%s frames=%d sensor=%s site=(%s, %s) object=%s "
        "DATE-OBS %s .. %s",
        image_set_id,
        senpai_run.num_frames,
        sensor,
        header.get("SITELAT", "?"),
        header.get("SITELONG", "?"),
        header.get("OBJECT", "?"),
        obs_start,
        obs_end,
    )


def _apply_auto_scaling(senpai_run: SenpaiRun, image_frame: SiderealFrame) -> None:
    """Apply FWHM-driven frame scaling to the whole run (``calibrations.auto_scale_images``).

    Scales every frame by the recommended factor from the solved frame's FWHM stats, records
    the actual scale factor at the run level, and re-solves the anchor frame at the new scale.

    Args:
        senpai_run (SenpaiRun): the organized run whose frames are scaled in place.
        image_frame (SiderealFrame): the solved sidereal frame carrying the FWHM stats.
    """
    config = get_config()
    fwhm_stats = image_frame.starfield.fwhm_stats if image_frame.starfield else None
    if not (fwhm_stats and fwhm_stats.recommended_scale_factor and fwhm_stats.recommended_scale_factor > 1.0):
        return

    scale_factor = fwhm_stats.recommended_scale_factor
    logger.info(f"Scaling all frames using FWHM stats from frame {image_frame.index}")
    logger.info(
        f"FWHM: {fwhm_stats.median_fwhm:.1f} -> {config.calibrations.target_fwhm:.1f} pixels "
        f"(factor: {scale_factor:.2f})"
    )
    logger.info(
        f"Method: {config.calibrations.scaling_method}, {fwhm_stats.n_measurements} FWHM measurements"
    )

    for frame in senpai_run.sidereal_frames + senpai_run.rate_track_frames:
        frame.frame.scale_frame(scale_factor, method=config.calibrations.scaling_method)
        if frame.starfield:
            frame.starfield = scale_starfield_coordinates(frame.starfield, scale_factor)

    # Get the actual scale factor used from the processing history
    # (for median_filter this is the rounded integer value)
    actual_scale_factor = scale_factor
    if config.calibrations.scaling_method == "median_filter":
        for frame in senpai_run.sidereal_frames + senpai_run.rate_track_frames:
            if frame.frame.processing_history:
                for step in reversed(frame.frame.processing_history):
                    if step.step_type.value == "fwhm_optimization":
                        actual_scale_factor = step.parameters.get("scale_factor", scale_factor)
                        break
                if actual_scale_factor != scale_factor:
                    break

    senpai_run.scale_factor = actual_scale_factor
    logger.info(
        f"Stored actual scale_factor {actual_scale_factor} at run level "
        f"(original recommended: {scale_factor:.2f})"
    )
    logger.info("All frames scaled successfully")

    # Re-run astrometry on the scaled anchor frame
    rescaled = process_astrometry_fits_sidereal(image_frame.frame)
    if rescaled is not None:
        image_frame.starfield = rescaled
        refine_sidereal_frame(image_frame)


def _attach_distortion_metrics(image_frame: SiderealFrame) -> None:
    """Measure field distortion via local Jacobians on a solved sidereal frame.

    Populates ``starfield.distortion_metrics`` with compact scalars used to decide whether
    variable kernels are needed for rate-track frames. Failures are logged, never raised.

    Args:
        image_frame (SiderealFrame): the solved sidereal frame.
    """
    try:
        if image_frame.starfield and image_frame.starfield.wcs is not None:
            astropy_wcs = image_frame.starfield.wcs.to_astropy_wcs()
            if astropy_wcs is not None:
                # Ensure array_shape is set so jacobian sampling covers the full detector
                height, width = image_frame.frame.data.shape
                astropy_wcs.array_shape = (height, width)

                # Use a nominal unit rate vector in RA; metrics are relative so the
                # specific rate magnitude is not critical for distortion assessment.
                metrics = wcs_distortion_metrics(astropy_wcs, rate_ra=1.0, rate_dec=0.0, nx=5, ny=5)

                image_frame.starfield.distortion_metrics = {
                    "delta_J": float(metrics["delta_J"]),
                    "max_angle_variation_deg": float(metrics["max_angle_variation_deg"]),
                    "max_length_variation_fraction": float(
                        metrics["max_length_variation_fraction"]
                    ),
                }

                logger.info(
                    "Sidereal WCS distortion (frame %d): "
                    "delta_J=%.3g, max_angle_variation_deg=%.3f, max_length_variation_fraction=%.3f",
                    image_frame.index,
                    image_frame.starfield.distortion_metrics["delta_J"],
                    image_frame.starfield.distortion_metrics["max_angle_variation_deg"],
                    image_frame.starfield.distortion_metrics["max_length_variation_fraction"],
                )
    except Exception as e:
        logger.warning(
            "Failed to compute WCS distortion metrics for sidereal frame %d: %s",
            image_frame.index,
            e,
        )


def _solve_rate_only_fallback(senpai_run: SenpaiRun) -> bool:
    """Attempt a WCS from streak centroids when no sidereal frame solved (rate-only input).

    Args:
        senpai_run (SenpaiRun): the organized run with unsolved rate frames.

    Returns:
        bool: True when a rate frame produced an accepted WCS solution.
    """
    config = get_config()

    from senpai.astrometry import solve_field
    from senpai.catalog.runner import query_catalog
    from senpai.engine.detection.streak.rate_extraction import (
        build_streak_metadata,
        extract_rate_streak_measurement,
        extract_streak_centers_as_sources,
    )
    from senpai.engine.models.metadata import DetectionMetadata, ImageMetadata
    from senpai.engine.models.starfield import StarListImage
    from senpai.engine.utils.fits_io import extract_boresight_from_header

    for image_frame in senpai_run.rate_track_frames:
        measurement, _psf, measured_fwhm = extract_rate_streak_measurement(
            image_frame, n_streaks=10, initial_fwhm=None
        )
        if measurement is None:
            continue

        if measurement.fwhm is None and measured_fwhm is not None:
            measurement.fwhm = float(measured_fwhm)
        if measurement.fwhm is None:
            measurement.fwhm = 4.0

        image_frame.streak = build_streak_metadata(measurement)
        image_frame.seeing = SeeingModel(
            pixel_fwhm=float(image_frame.streak.fwhm),
            pixel_fwhm_stdev=0.0,
            n_measurements=1,
        )

        sources = extract_streak_centers_as_sources(
            image_frame.frame.data,
            streak=image_frame.streak,
            max_sources=200,
        )
        if not sources:
            continue

        # Diagnostic overlay: show streak centroids handed to the solver,
        # so a WCS failure still produces a visible debug artifact.
        if config.plotting.debug or config.plotting.review:
            sources_for_plot = StarListImage(
                detections=sources,
                image_metadata=ImageMetadata(
                    width=image_frame.frame.data.shape[1],
                    height=image_frame.frame.data.shape[0],
                ),
            )
            plot_single_frame(
                image_frame.frame.data,
                starlist=sources_for_plot,
                streak=image_frame.streak,
                markersize=(image_frame.streak.fwhm * 2 if image_frame.streak else 10),
                output_file=Path(config.runtime.output_dir)
                / f"rate_detections_{image_frame.index}.png",
            )

        boresight_ra, boresight_dec = extract_boresight_from_header(image_frame.frame.header)
        frame_meta = image_frame.frame_metadata
        img_meta = ImageMetadata(
            width=image_frame.frame.data.shape[1],
            height=image_frame.frame.data.shape[0],
            boresight_ra=boresight_ra,
            boresight_dec=boresight_dec,
            exposure_time=(
                float(frame_meta.exposure_time_seconds)
                if frame_meta and frame_meta.exposure_time_seconds
                else None
            ),
        )
        starlist = StarListImage(detections=sources, image_metadata=img_meta)

        try:
            wcs_starfield = solve_field(starlist)
        except Exception as e:
            logger.warning("WCS solve failed for rate frame %d: %s", image_frame.index, e)
            continue

        if wcs_starfield and wcs_starfield.wcs:
            try:
                catalog = query_catalog(wcs_starfield.wcs, max_stars=None, apply_sip=True)
                wcs_starfield.catalog_stars = catalog.stars
                base_md = wcs_starfield.image_metadata.model_dump()
                for k, v in catalog.image_metadata.model_dump().items():
                    if v is not None:
                        base_md[k] = v
                wcs_starfield.image_metadata = ImageMetadata(**base_md)
            except Exception as e:
                logger.warning("Catalog query failed for rate frame %d: %s", image_frame.index, e)

            wcs_starfield.detection_metadata = DetectionMetadata(
                pixel_fwhm=float(image_frame.streak.fwhm)
            )
            image_frame.starfield = wcs_starfield

            # Track rate in pixels/s
            exp = frame_meta.exposure_time_seconds if frame_meta else None
            if exp and exp > 0:
                image_frame.pixel_track_rate_per_second = float(
                    image_frame.streak.pixel_length
                ) / float(exp)

            logger.info(
                "Rate-only mode: WCS solved from rate frame %d streak centroids",
                image_frame.index,
            )
            return True

    return False


def _run_photometry_stage(senpai_run: SenpaiRun) -> None:
    """Per-frame photometry: zero point, limiting magnitude, and detection photometry.

    Gated by ``photometry.enable``. Failures are logged per frame, never raised.

    Args:
        senpai_run (SenpaiRun): the completed-detection run to measure.
    """
    from dataclasses import asdict

    from senpai.engine.photometry.utils import (
        measure_detection_photometry,
        measure_rate_starfield_photometry,
        measure_simple_starfield_photometry,
    )

    config = get_config()

    # Sidereal frames: simple circular aperture photometry
    for image_frame in senpai_run.sidereal_frames:
        if image_frame.starfield is None or not image_frame.starfield.fit:
            continue
        try:
            _, summary = measure_simple_starfield_photometry(
                image_frame.frame,
                image_frame.starfield,
                config.photometry,
                frame_index=image_frame.index,
            )
            image_frame.photometry_summary = asdict(summary)
            if summary.limiting_magnitude_50 is not None:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude_50
            elif summary.limiting_magnitude:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude
            logger.info(
                f"Sidereal frame {image_frame.index}: photometry ZP={summary.zero_point}, "
                f"limiting_mag={image_frame.starfield.limiting_magnitude}"
            )
        except Exception as e:
            logger.warning(f"Photometry failed for sidereal frame {image_frame.index}: {e}")

    # Rate-track frames: rectangular aperture photometry + detection photometry
    for image_frame in senpai_run.rate_track_frames:
        if image_frame.starfield is None or not image_frame.starfield.fit:
            continue
        if image_frame.streak is None:
            continue
        try:
            _, summary = measure_rate_starfield_photometry(
                image_frame.frame,
                image_frame.starfield,
                image_frame.streak,
                config.photometry,
                frame_index=image_frame.index,
            )
            image_frame.photometry_summary = asdict(summary)
            if summary.limiting_magnitude_50 is not None:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude_50
            elif summary.limiting_magnitude:
                image_frame.starfield.limiting_magnitude = summary.limiting_magnitude
            logger.info(
                f"Rate frame {image_frame.index}: photometry ZP={summary.zero_point}, "
                f"limiting_mag={image_frame.starfield.limiting_magnitude}"
            )

            # Detection photometry if we have detections and a valid zero point
            if (
                image_frame.detections
                and image_frame.detections.detections
                and summary.zero_point is not None
            ):
                try:
                    exp_time = (
                        image_frame.frame_metadata.exposure_time_seconds
                        if image_frame.frame_metadata
                        else None
                    )
                    measure_detection_photometry(
                        image_frame.frame,
                        image_frame.detections,
                        summary.zero_point,
                        summary.zero_point_err,
                        exposure_time=exp_time,
                        config=config.photometry,
                        multiband_calibration=summary.multiband_calibration,
                        observation_filter=(
                            image_frame.frame_metadata.observation_filter
                            if image_frame.frame_metadata
                            else None
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        f"Detection photometry failed for rate frame {image_frame.index}: {e}"
                    )
        except Exception as e:
            logger.warning(f"Photometry failed for rate frame {image_frame.index}: {e}")


def _flag_sidereal_non_catalog_detections(senpai_run: SenpaiRun) -> None:
    """Flag bright non-catalog point sources on solved sidereal frames as detections.

    Gated by ``detection.sidereal_point_detections``. Matches extracted sources against
    catalog stars; unmatched sources at least as bright as the median matched source and
    passing local shape validation become candidate detections.

    Args:
        senpai_run (SenpaiRun): the run whose sidereal frames are examined.
    """
    from senpai.engine.detection.point.sidereal_extra import validate_point_detection
    from senpai.engine.models.starfield import SatelliteInImage, SatelliteListImage

    for image_frame in senpai_run.sidereal_frames:
        if image_frame.starfield is None or not image_frame.starfield.fit:
            continue
        if not image_frame.starfield.catalog_stars or not image_frame.starfield.detections:
            continue

        fwhm = 4.0
        if (
            image_frame.starfield.detection_metadata
            and image_frame.starfield.detection_metadata.pixel_fwhm
        ):
            fwhm = image_frame.starfield.detection_metadata.pixel_fwhm

        match_radius_sq = (2 * fwhm) ** 2

        catalog_positions = [
            (s.x, s.y)
            for s in image_frame.starfield.catalog_stars
            if s.x is not None and s.y is not None
        ]
        if not catalog_positions:
            continue

        catalog_xy = np.array(catalog_positions)

        # Separate matched vs unmatched detections and compute a brightness
        # threshold.  Only flag unmatched detections that are at least as bright
        # as the MEDIAN of matched (catalog-confirmed) detections. The lower
        # quartile of matched counts sits at the noise floor (the deep catalog
        # matches plenty of barely-detected stars), so a p25 floor let dozens
        # of noise peaks through per frame (one sensor, frame 9: 38 at p25, 2 at
        # p50). Fainter unmatched sources are overwhelmingly noise peaks.
        matched_counts = []
        unmatched = []
        for det in image_frame.starfield.detections:
            if det.x is None or det.y is None:
                continue
            dists_sq = (catalog_xy[:, 0] - det.x) ** 2 + (catalog_xy[:, 1] - det.y) ** 2
            if np.min(dists_sq) <= match_radius_sq:
                if det.counts is not None:
                    matched_counts.append(det.counts)
            else:
                unmatched.append(det)

        if not unmatched or not matched_counts:
            continue

        min_counts = float(np.percentile(matched_counts, 50))

        non_catalog = [
            det for det in unmatched if det.counts is not None and det.counts >= min_counts
        ]

        # Shape/locality vetting: the brightness threshold above compares
        # globally background-subtracted counts, so detections sitting on
        # amplifier glow or edge glare look "bright" while being mere noise
        # wiggles on an elevated background.  Require each flagged detection
        # to be a significant point source at its LOCAL scale.
        n_before_validation = len(non_catalog)
        frame_data = image_frame.frame.data
        non_catalog = [
            det for det in non_catalog if validate_point_detection(frame_data, det.x, det.y, fwhm)
        ]
        n_rejected_shape = n_before_validation - len(non_catalog)

        if non_catalog:
            satellites = []
            for det in non_catalog:
                ra_val, dec_val = None, None
                if image_frame.starfield.wcs:
                    try:
                        wcs = image_frame.starfield.wcs.to_astropy_wcs()
                        sky = wcs.pixel_to_world(det.x, det.y)
                        ra_val = float(sky.ra.deg)
                        dec_val = float(sky.dec.deg)
                    except Exception as err:
                        logger.debug("Point detection WCS conversion failed: %s", err)
                satellites.append(
                    SatelliteInImage(
                        x=det.x,
                        y=det.y,
                        snr=det.snr,
                        ra=ra_val,
                        dec=dec_val,
                        pixel_fwhm=fwhm,
                        detection_type="point",
                    )
                )

            img_meta = image_frame.starfield.image_metadata
            if image_frame.detections is None:
                image_frame.detections = SatelliteListImage(
                    detections=satellites,
                    image_metadata=img_meta,
                )
            else:
                image_frame.detections.detections.extend(satellites)

        logger.info(
            "Sidereal frame %d: %d non-catalog point detections "
            "(%d unmatched, %d below brightness threshold, "
            "%d rejected by shape/local-significance, counts_thresh=%.0f)",
            image_frame.index,
            len(non_catalog),
            len(unmatched),
            len(unmatched) - n_before_validation,
            n_rejected_shape,
            min_counts,
        )


def _run_streak_stage(senpai_run: SenpaiRun) -> None:
    """Streak detection, cross-frame correlation, and point/streak deduplication.

    Gated by ``detection.detect_streaks``.

    Args:
        senpai_run (SenpaiRun): the run to detect and correlate streaks in.
    """
    from senpai.engine.processing.rate_scan_confirmation import confirm_streaks_via_rate_scan
    from senpai.engine.processing.streak_correlation import (
        correlate_rate_to_sidereal,
        detect_streaks_in_rate_frames,
        detect_streaks_in_sidereal_frames,
    )

    de_data = detect_streaks_in_sidereal_frames(senpai_run)
    detect_streaks_in_rate_frames(senpai_run)
    senpai_run.correlated_streaks = confirm_streaks_via_rate_scan(senpai_run, de_data)
    if senpai_run.rate_track_frames and senpai_run.sidereal_frames:
        correlate_rate_to_sidereal(senpai_run)

    # Clear unconfirmed rate-frame streak candidates — they are single-frame
    # detections that didn't pass multi-frame confirmation and would show
    # as false positives in annotations/output.
    for frame in senpai_run.rate_track_frames:
        frame.streak_candidates = []

    # Deduplicate: remove point-type detections that overlap with streak
    # detections or streak candidates (a streak's peak is usually also
    # picked up by the point finder; the streak report supersedes it)
    for frame in senpai_run.sidereal_frames + senpai_run.rate_track_frames:
        if frame.detections is None:
            continue
        streak_dets = [
            d for d in frame.detections.detections if getattr(d, "detection_type", None) == "streak"
        ]
        streak_dets.extend(frame.streak_candidates or [])
        if not streak_dets:
            continue
        fwhm = 4.0
        if (
            frame.starfield
            and frame.starfield.detection_metadata
            and frame.starfield.detection_metadata.pixel_fwhm
        ):
            fwhm = frame.starfield.detection_metadata.pixel_fwhm
        radius = 2 * fwhm

        def _near_streak(
            d: "SatelliteInImage",
            s: "SatelliteInImage | StreakCandidate",
            radius: float = radius,
        ) -> bool:
            # Distance from the point to the streak SEGMENT (not just its
            # center) so detections anywhere along the streak are caught.
            dx, dy = d.x - s.x, d.y - s.y
            angle = getattr(s, "angle_deg", None)
            half_len = (getattr(s, "length_pixels", 0) or 0) / 2
            if angle is None or half_len <= 0:
                return dx * dx + dy * dy < radius * radius
            angle_rad = np.radians(angle)
            ux, uy = np.cos(angle_rad), np.sin(angle_rad)
            along = np.clip(dx * ux + dy * uy, -half_len, half_len)
            return (dx - along * ux) ** 2 + (dy - along * uy) ** 2 < radius * radius

        cleaned = []
        for d in frame.detections.detections:
            if getattr(d, "detection_type", None) == "point" and any(
                _near_streak(d, s) for s in streak_dets
            ):
                continue
            cleaned.append(d)
        n_removed = len(frame.detections.detections) - len(cleaned)
        if n_removed:
            frame.detections.detections = cleaned
            logger.info(
                "Frame %d: removed %d point detections overlapping streak detections",
                frame.index,
                n_removed,
            )


def process_senpai_collect(
    file_list: list[ProcessedFitsImage],
    id: str = "senpai",  # noqa: A002  # public keyword arg passed as id=... by CLI/API callers
    force_track_mode: TrackMode | None = None,
) -> SenpaiRun:
    """Process a full SENPAI collect from sidereal solve through frame-by-frame shifts.

    Organizes the frames, finds a valid sidereal WCS solution, then propagates the WCS
    across the analysis chain, refining each frame and running point-source detection on
    rate-track frames. Feature stages (calibration preprocessing, photometry, streak
    detection, sidereal non-catalog detections) run config-gated around that core.

    Args:
        file_list (list[ProcessedFitsImage]): all frames in the collect.
        id (str): run identifier recorded on the result. Defaults to "senpai".
        force_track_mode (TrackMode | None): override the per-frame track-mode
            classification. Defaults to None (classify from headers/pixels).

    Returns:
        SenpaiRun: the populated run; ``completed`` is True on success, otherwise
            ``error_message`` records the failure (when failures are not configured
            to raise).

    Raises:
        SiderealSolveError: If no valid sidereal WCS solution could be found and
            ``astrometry.error_on_plate_solve_failure`` is enabled.
        WcsPropagationError: If the sidereal WCS could not be propagated to any rate
            frame and ``astrometry.error_on_plate_solve_failure`` is enabled.
    """
    try:
        t_start = time.time()
        config = get_config()

        # Calibration preprocessing (flats/darks/medians/background per config toggles)
        from senpai.engine.utils.preprocessing import preprocess_image

        logger.info("Applying preprocessing to all frames...")
        for frame in file_list:
            preprocess_image(frame, config, store_intermediates=False)

            # Save the processed frame data for later export (replot reads these;
            # full-night runs skip them — ~260 MB/frame dominates the output dir).
            if config.runtime.save_processed_fits and getattr(frame, "file_path", None):
                from astropy.io import fits

                processed_path = Path(frame.file_path)
                processed_filename = f"{processed_path.stem}_processed{processed_path.suffix}"
                processed_file_path = Path(config.runtime.output_dir) / processed_filename

                hdu = fits.PrimaryHDU(frame.data, frame.header)
                hdu.writeto(processed_file_path, overwrite=True)

                frame.processed_file_path = str(processed_file_path)
                logger.debug(f"Saved processed frame: {processed_file_path}")

        senpai_run = SenpaiRun.organize_senpai_frames(
            file_list, id=id, force_track_mode=force_track_mode
        )
        _log_collect_summary(senpai_run, file_list)

        valid_sidereal_frame = False
        for image_frame in senpai_run.sidereal_frames:
            sidereal_wcs_starfield = process_astrometry_fits_sidereal(image_frame.frame)
            if sidereal_wcs_starfield is None:
                continue

            image_frame.starfield = sidereal_wcs_starfield
            refine_sidereal_frame(image_frame)

            if image_frame.starfield.fit:
                logger.info(
                    f"Found valid WCS solution in frame {image_frame.index}, "
                    "initial sidereal processing complete"
                )
                valid_sidereal_frame = True

                # Seeing from FWHM stats (present when photometry/auto-scaling measured them);
                # propagate to all sidereal frames — they share the same optics.
                if image_frame.starfield.fwhm_stats:
                    image_frame.seeing = SeeingModel.from_fwhm_stats(
                        image_frame.starfield.fwhm_stats
                    )
                    for other_frame in senpai_run.sidereal_frames:
                        if other_frame.seeing is None:
                            other_frame.seeing = image_frame.seeing

                if config.calibrations.auto_scale_images:
                    _apply_auto_scaling(senpai_run, image_frame)

                _attach_distortion_metrics(image_frame)

                if config.plotting.review and config.plotting.debug:
                    # otherwise this'll be plotted at the end (review True, debug False)
                    plot_single_frame(
                        image_frame.frame.data,
                        starfield=image_frame.starfield,
                        detections=image_frame.detections,
                        output_file=Path(config.runtime.output_dir)
                        / f"final_{image_frame.index}.png",
                    )

                break

        if not valid_sidereal_frame and senpai_run.rate_track_frames:
            # Rate-only input — try WCS from the first rate frame's streak centroids
            valid_sidereal_frame = _solve_rate_only_fallback(senpai_run)

        if not valid_sidereal_frame:
            msg = "No valid WCS solution found"
            if config.astrometry.error_on_plate_solve_failure:
                raise SiderealSolveError(msg)
            senpai_run.error_message = msg
            logger.warning(senpai_run.error_message)
            return senpai_run

        # ok, find a valid path through all frames creating / using frame shifts
        senpai_run.create_valid_path()

        next_shift = senpai_run.get_next_shift()
        while next_shift is not None:
            if config.plotting.debug:  # pragma: no cover
                plot_single_frame(
                    senpai_run.get_frame_by_index(next_shift.target_index).frame.data,
                    output_file=Path(config.plotting.output_dir)
                    / f"{next_shift.target_index}_raw.png",
                )

            solve_shift(senpai_run, next_shift)

            # Backstop against a livelock: the loop pulls the next *unprocessed*
            # shift, so a solver that returns without setting processed=True hands
            # the same shift back forever. A solver must always mark the shift
            # processed; if one didn't, retire it as failed so the loop progresses.
            if not next_shift.processed:
                logger.error(
                    "Shift %d->%d returned unprocessed from solver; force-retiring "
                    "as failed to avoid livelock.",
                    next_shift.source_index,
                    next_shift.target_index,
                )
                next_shift.processed = True
                next_shift.is_valid = False
                next_shift.error_message = (
                    next_shift.error_message or "Solver returned without processing"
                )

            # Optional chain-consistency gate + failure rerouting (their v2.6 robustness
            # layer). Off by default in this flow, whose solvers handle rerouting
            # internally; when enabled, a rejected hop is routed around here.
            if config.chain_gate.enable:
                enforce_chain_consistency(senpai_run, next_shift)
                senpai_run.update_valid_path()

            logger.info("Shifting WCS by pixel shift")

            if next_shift.is_valid and next_shift.processed:
                shift_wcs_by_pixel_shift(senpai_run, next_shift)

                target = senpai_run.get_frame_by_index(next_shift.target_index)
                if isinstance(target, RateTrackFrame):
                    # Optionally reconcile a degenerate streak extraction with
                    # chain-derived geometry before the kernel refinement.
                    if config.streak.reconcile_with_chain:
                        from senpai.engine.utils.streak_chain import (
                            chain_drift_rates,
                            reconcile_streak_with_chain,
                        )

                        reconcile_streak_with_chain(
                            target,
                            chain_drift_rates(senpai_run),
                            config.streak.reconcile_length_tolerance,
                            config.streak.reconcile_angle_tolerance_deg,
                        )

                    logger.info("Refining WCS by kernel convolution")
                    wcs_refined = refine_wcs_by_kernel_convolution(target)

                    if config.detection.detect:
                        if not config.detection.require_wcs_refinement or wcs_refined:
                            target.detections = extract_point_sources(target)
                        else:
                            logger.warning(
                                f"Skipping detection for frame {target.index}: "
                                "catalog WCS refinement failed"
                            )

                if isinstance(target, SiderealFrame):
                    logger.info("Refining WCS by kernel convolution")
                    refine_sidereal_frame(target)

                if config.plotting.debug:  # pragma: no cover
                    plot_single_frame(
                        target.frame.data,
                        starfield=target.starfield,
                        detections=target.detections
                        if isinstance(target, RateTrackFrame)
                        else None,
                        streak=target.streak if isinstance(target, RateTrackFrame) else None,
                        output_file=Path(config.plotting.output_dir) / f"{target.index}.png",
                    )
                elif config.plotting.review:
                    plot_single_frame(
                        target.frame.data,
                        starfield=target.starfield,
                        detections=target.detections
                        if isinstance(target, RateTrackFrame)
                        else None,
                        streak=target.streak if isinstance(target, RateTrackFrame) else None,
                        output_file=Path(config.runtime.output_dir) / f"final_{target.index}.png",
                    )

            senpai_run.log_analysis_chain()
            next_shift = senpai_run.get_next_shift()

        # A cascade (e.g. a failed anchor) can leave the whole chain unregistered; surface
        # that as a meaningful error rather than returning an empty result.
        try:
            require_registered_rate_frames(senpai_run)
        except WcsPropagationError:
            if config.astrometry.error_on_plate_solve_failure:
                raise
            senpai_run.error_message = "No rate-track frame registered to the WCS solution"
            logger.warning(senpai_run.error_message)
            return senpai_run

        # --- Point source detection for rate-track frames that weren't shift targets ---
        # (e.g. the rate-only fallback path). Frames that WERE shift targets already had
        # their detection pass in the shift loop, gated on refinement success — a frame
        # left without detections there stays that way.
        if config.detection.detect:
            shift_target_indices = {
                shift.target_index
                for shift in senpai_run.frame_shifts + senpai_run.frame_shifts_failed
                if shift.processed
            }
            for image_frame in senpai_run.rate_track_frames:
                if image_frame.detections is not None:
                    continue
                if image_frame.index in shift_target_indices:
                    continue
                if image_frame.starfield is None or not image_frame.starfield.fit:
                    continue
                image_frame.detections = extract_point_sources(image_frame)

        # --- Feature stages (config-gated) ---
        if config.photometry.enable:
            _run_photometry_stage(senpai_run)

        if config.detection.sidereal_point_detections:
            _flag_sidereal_non_catalog_detections(senpai_run)

        if config.detection.detect and config.detection.detect_streaks:
            _run_streak_stage(senpai_run)

        senpai_run.completed = True
        senpai_run.error_message = None
        senpai_run.compute_seconds = round(time.time() - t_start, 2)
        logger.info(f"Time taken to process set: {senpai_run.compute_seconds} seconds")

        return senpai_run
    finally:
        reclaim_process_memory()


def _write_sequence_gif(image_paths: list, gif_path: Path) -> None:
    """Write a per-frame animation, padding frames to a common shape first.

    Mixed sidereal/rate batches render their ``final_*`` plots at different pixel
    sizes (different overlays/colorbars), so a naive ``np.stack`` of the frames
    raises "all input arrays must have the same shape". We pad each frame to the
    max height/width before stacking. The GIF is a diagnostic nicety, so any
    failure is logged and swallowed rather than failing the batch.

    Args:
        image_paths (list): paths of the per-frame PNGs, in order.
        gif_path: destination path for the GIF.
    """
    try:
        import imageio.v3 as iio

        images = [iio.imread(str(f)) for f in image_paths]
        if not images:
            return
        h = max(im.shape[0] for im in images)
        w = max(im.shape[1] for im in images)
        padded = []
        for im in images:
            pad = [(0, h - im.shape[0]), (0, w - im.shape[1])]
            pad += [(0, 0)] * (im.ndim - 2)
            padded.append(np.pad(im, pad, mode="constant"))
        iio.imwrite(gif_path, padded, duration=400, loop=0)
        logger.info(f"Created animation at {gif_path}")
    except Exception as e:
        logger.warning(f"Skipping animation {gif_path}: {e}")


def final_plots(senpai_run: SenpaiRun, output_dir: Path) -> None:
    """Render the per-frame review/PSF plots and sequence GIFs for a completed run.

    Args:
        senpai_run (SenpaiRun): the completed run to plot.
        output_dir (Path): directory the plots are written to.
    """
    config = get_config()

    run_id = config.runtime.run_id

    # Per-frame empirical PSF panels (stacked stars for sidereal, stacked streak
    # for rate). A small .npy stamp is saved next to each PNG so the panel can be
    # regenerated later (see engine.plotting.replot) without the raw FITS.
    if config.plotting.psfs:
        from senpai.engine.plotting.psf import plot_rate_frame, plot_sidereal_frame

        for f in senpai_run.sidereal_frames:
            png = output_dir / f"frame_{f.index}_psf.png"
            if not png.exists():
                try:
                    plot_sidereal_frame(f, png, output_dir / f"frame_{f.index}_psf.npy")
                except Exception as e:
                    logger.warning("PSF panel failed for sidereal frame %s: %s", f.index, e)
        for f in senpai_run.rate_track_frames:
            png = output_dir / f"frame_{f.index}_streak.png"
            if not png.exists():
                try:
                    plot_rate_frame(f, png, output_dir / f"frame_{f.index}_streak.npy")
                except Exception as e:
                    logger.warning("PSF panel failed for rate frame %s: %s", f.index, e)

    for image_frame in senpai_run.sidereal_frames:
        output_file = output_dir / f"final_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                starfield=image_frame.starfield,
                detections=image_frame.detections,
                streak_candidates=image_frame.streak_candidates or None,
                output_file=output_file,
            )
        output_file = output_dir / f"raw_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                output_file=output_file,
            )

    for image_frame in senpai_run.rate_track_frames:
        output_file = output_dir / f"final_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                starfield=image_frame.starfield,
                streak=image_frame.streak,
                detections=image_frame.detections,
                streak_candidates=image_frame.streak_candidates or None,
                output_file=output_file,
            )

        output_file = output_dir / f"raw_{image_frame.index}.png"
        if config.plotting.review and not output_file.exists():
            plot_single_frame(
                image_frame.frame.data,
                output_file=output_file,
            )

    if config.plotting.review:
        # Collect all plot filenames and sort by frame index
        plot_files = []
        plot_rate_files = []
        plot_raw_files = []
        for frame in sorted(
            senpai_run.sidereal_frames + senpai_run.rate_track_frames,
            key=lambda x: x.index,
        ):
            plot_file = output_dir / f"final_{frame.index}.png"
            if plot_file.exists():
                plot_files.append(plot_file)

                if isinstance(frame, RateTrackFrame):
                    plot_rate_files.append(plot_file)

            plot_file = output_dir / f"raw_{frame.index}.png"
            if plot_file.exists():
                plot_raw_files.append(plot_file)

        if plot_files:
            _write_sequence_gif(plot_files, output_dir / f"{run_id}_sequence.gif")
        if plot_rate_files:
            _write_sequence_gif(plot_rate_files, output_dir / f"{run_id}_sequence_rate.gif")
        if plot_raw_files:
            _write_sequence_gif(plot_raw_files, output_dir / f"{run_id}_sequence_raw.gif")
