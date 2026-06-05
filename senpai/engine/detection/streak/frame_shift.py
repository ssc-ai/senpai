import logging

from senpai.engine.detection.streak.rate_rate import solve_rate_from_rate
from senpai.engine.detection.streak.rate_sidereal import solve_rate_from_sidereal
from senpai.engine.detection.streak.sidereal_sidereal import solve_sidereal_from_sidereal
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.senpai import FrameShift, RateTrackFrame, SenpaiRun, SiderealFrame

logger = logging.getLogger(__name__)


def preprocess_for_shift(frame: ProcessedFitsImage) -> None:
    # Preprocessing should already be applied at frame load time
    # This function is kept for backward compatibility but should be a no-op
    logger.debug("Preprocessing already applied at frame load time")


def solve_shift(senpai_run: SenpaiRun, frame_shift: FrameShift) -> None:
    frame_source = senpai_run.get_frame_by_index(frame_shift.source_index)
    frame_target = senpai_run.get_frame_by_index(frame_shift.target_index)

    preprocess_for_shift(frame_source.frame)
    preprocess_for_shift(frame_target.frame)

    if isinstance(frame_source, SiderealFrame) and isinstance(frame_target, SiderealFrame):
        solver = solve_sidereal_from_sidereal
        solve_type = "sidereal to sidereal"

    elif isinstance(frame_source, RateTrackFrame) and isinstance(frame_target, RateTrackFrame):
        solver = solve_rate_from_rate
        solve_type = "rate to rate"

    elif isinstance(frame_source, SiderealFrame) and isinstance(frame_target, RateTrackFrame):
        solver = solve_rate_from_sidereal
        solve_type = "sidereal to rate"

    elif isinstance(frame_source, RateTrackFrame) and isinstance(frame_target, SiderealFrame):
        solver = solve_rate_from_sidereal
        solve_type = "rate to sidereal"

    else:
        raise ValueError(f"Invalid frame types: {type(frame_source)} and {type(frame_target)}")

    logger.info(f"Solving shift from {frame_source.index} to {frame_target.index} ({solve_type})")

    solver(frame_source, frame_target, frame_shift)
