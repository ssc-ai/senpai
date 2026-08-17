"""Bayesian-engine WCS pixel-shift propagation.

Unlike the upstream ``shift_wcs_by_pixel_shift`` (which re-projects the existing
catalog stars), this re-queries the catalog for the shifted field so the target
starfield carries fresh catalog stars for the new region. Selected by the
Bayesian registration engine; shares the upstream WCS helpers.
"""

import logging

from senpai.engine.models.astrometry import WCSMetadata, WCSStatus
from senpai.engine.models.senpai import FrameShift, SenpaiRun
from senpai.engine.models.starfield import StarField
from senpai.engine.utils.wcs_ops import (
    catalog_stars_from_wcs,
    existing_stars_from_wcs,
    shift_wcs,
)

logger = logging.getLogger(__name__)


def shift_wcs_by_pixel_shift(senpai_run: SenpaiRun, frame_shift: FrameShift) -> None:
    """Propagate a source frame's WCS to a target frame via a pixel shift.

    Builds the target frame's starfield by shifting the source WCS by the frame
    shift's pixel offset and reprojecting the astrometric stars + re-querying the
    catalog for the shifted field.

    Raises:
        ValueError: If the source frame has no WCS to shift.
    """
    source_frame = senpai_run.get_frame_by_index(frame_shift.source_index)
    if source_frame.starfield.wcs_status == WCSStatus.NO_WCS:
        logger.error("Source frame WCS status is NO_WCS... no WCS to shift!")
        raise ValueError("Source frame WCS status is NO_WCS... no WCS to shift!")

    source_wcs_model = source_frame.starfield.wcs
    target_frame = senpai_run.get_frame_by_index(frame_shift.target_index)
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

    target_frame.starfield = StarField(
        astrometric_fit_stars=target_stars_astrometry,
        catalog_stars=target_stars_catalog.stars,
        detections=[],
        image_metadata=refined_image_metadata,
        fit=True,
        wcs=target_wcs_model,
        wcs_metadata=WCSMetadata.from_wcsmodel(target_wcs_model),
        wcs_status=WCSStatus.PIXEL_SHIFTED_WCS,
        detection_metadata=source_frame.starfield.detection_metadata,
        astrometry=None,
        limiting_magnitude=source_frame.starfield.limiting_magnitude,
    )

    logger.info(
        f"Shifted WCS from frame {frame_shift.source_index} to {frame_shift.target_index} "
        f"by ({shift_x}, {shift_y}) pixels"
    )
