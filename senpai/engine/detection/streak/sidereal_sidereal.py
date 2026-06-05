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
):
    sidereal_source_data, source_is_synthetic = prepare_sidereal_frame(frame_source)
    sidereal_target_data, target_is_synthetic = prepare_sidereal_frame(frame_target)

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

    # cross correlate
    cross_correlated_image = cross_corr(sidereal_target_data, sidereal_source_data)

    # measure_gaussian_shift returns (shift_yx, fwhm) where shift_yx is (row, col)
    shift_yx, fwhm = measure_gaussian_shift(cross_correlated_image)

    pixel_shift_magnitude = np.linalg.norm(shift_yx)

    logger.info(
        f"Pixel shift sidereal to sidereal: {pixel_shift_magnitude:.1f} pixels."
    )

    frame_shift.x_shift = shift_yx[1]  # col offset = x
    frame_shift.y_shift = shift_yx[0]  # row offset = y
    frame_shift.is_valid = True
    frame_shift.processed = True
    frame_shift.error_message = None

    return
