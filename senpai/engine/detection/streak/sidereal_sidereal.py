"""Solve the pixel shift between two sidereal-tracked frames via cross correlation.

Cross correlates a pair of (optionally synthetic) sidereal frames after removing
saturated and border-crossing streaks, then records the measured shift on the
provided frame shift.
"""

import logging

import numpy as np

from senpai.engine.detection.streak.extraction import (
    cross_corr,
    measure_gaussian_shift,
    prepare_sidereal_frame,
)
from senpai.engine.detection.streak.masking import (
    remove_border_crossing_streaks,
    remove_near_saturation_streaks,
)
from senpai.engine.models.senpai import FrameShift, SiderealFrame

logger = logging.getLogger(__name__)


def solve_sidereal_from_sidereal(
    frame_source: SiderealFrame, frame_target: SiderealFrame, frame_shift: FrameShift
) -> None:
    """Measure the pixel shift between two sidereal frames by cross correlation.

    Prepares both frames, strips near-saturation and border-crossing streaks, cross
    correlates them, and fits a Gaussian to the correlation peak to estimate the shift.
    The resulting shift is written to the provided frame shift in place.

    Args:
        frame_source: The reference sidereal frame.
        frame_target: The sidereal frame to align with the source.
        frame_shift: The frame shift to populate with the measured shift in place.

    Returns:
        None.
    """
    sidereal_source_data, source_is_synthetic = prepare_sidereal_frame(
        frame_source, allow_synthetic=False
    )
    sidereal_target_data, target_is_synthetic = prepare_sidereal_frame(
        frame_target, allow_synthetic=False
    )

    if source_is_synthetic and target_is_synthetic:
        # both frames have already been fit
        # extract shifts from the WCS
        pass

    if not source_is_synthetic:
        sidereal_source_data, _ = remove_near_saturation_streaks(
            sidereal_source_data, frame_source.frame.data_type
        )

    if not target_is_synthetic:
        sidereal_target_data, _ = remove_near_saturation_streaks(
            sidereal_target_data, frame_target.frame.data_type
        )

    sidereal_source_data = remove_border_crossing_streaks(sidereal_source_data)
    sidereal_target_data = remove_border_crossing_streaks(sidereal_target_data)

    sidereal_source_data = remove_border_crossing_streaks(sidereal_source_data)
    sidereal_target_data = remove_border_crossing_streaks(sidereal_target_data)

    # cross correlate
    cross_correlated_image = cross_corr(sidereal_target_data, sidereal_source_data)

    _fwhm, pixel_shift_rate_to_sidereal_xy = measure_gaussian_shift(cross_correlated_image)[::-1]

    pixel_shift_rate_to_sidereal = np.linalg.norm(pixel_shift_rate_to_sidereal_xy)

    logger.info(f"Pixel shift rate to sidereal: {pixel_shift_rate_to_sidereal:.1f} pixels.")

    frame_shift.x_shift = pixel_shift_rate_to_sidereal_xy[0]
    frame_shift.y_shift = pixel_shift_rate_to_sidereal_xy[1]
    frame_shift.is_valid = True
    frame_shift.processed = True

    return
