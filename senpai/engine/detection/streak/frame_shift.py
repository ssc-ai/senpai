import logging

import numpy as np

from senpai.core.config import get_config
from senpai.engine.detection.streak.rate_rate import solve_rate_from_rate
from senpai.engine.detection.streak.rate_sidereal import solve_rate_from_sidereal
from senpai.engine.detection.streak.sidereal_sidereal import solve_sidereal_from_sidereal
from senpai.engine.detection.streak.validation import validate_proposed_shift
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


def _hop_rate(senpai_run: SenpaiRun, shift: FrameShift) -> tuple[float, float] | None:
    """Star-drift rate (px/s) implied by a solved hop, normalized by the
    *signed* time gap so hops solved in either temporal direction are
    comparable."""
    if shift.x_shift is None or shift.y_shift is None:
        return None
    source = senpai_run.get_frame_by_index(shift.source_index)
    target = senpai_run.get_frame_by_index(shift.target_index)
    if source.timestamp is None or target.timestamp is None:
        return None
    dt = (target.timestamp - source.timestamp).total_seconds()
    if abs(dt) < 0.1:
        return None
    return shift.x_shift / dt, shift.y_shift / dt


def enforce_chain_consistency(senpai_run: SenpaiRun, frame_shift: FrameShift) -> None:
    """Reject (or sign-repair) a solved hop that contradicts the accepted chain.

    Under rate tracking, star drift per second is nearly constant across an
    observation, so the solved shifts form a smooth chain. A hop whose rate
    reverses direction or deviates grossly from the accepted-chain median is
    a mis-solve — and because the WCS is propagated hop by hop, one such hop
    silently corrupts every frame beyond it. This gate runs after the solver
    and before WCS propagation; a rejected hop leaves its target frame
    without a WCS, which is strictly better than a confidently wrong one.
    """
    gate = get_config().chain_gate
    if not gate.enable or not (frame_shift.is_valid and frame_shift.processed):
        return

    source = senpai_run.get_frame_by_index(frame_shift.source_index)
    target = senpai_run.get_frame_by_index(frame_shift.target_index)
    # Only rate->rate hops drift at the steady tracking rate; the transition
    # to/from a sidereal frame includes mount settling and is exempt.
    if not (isinstance(source, RateTrackFrame) and isinstance(target, RateTrackFrame)):
        return

    history = []
    for prior in senpai_run.frame_shifts:
        if prior is frame_shift or not (prior.processed and prior.is_valid):
            continue
        a = senpai_run.get_frame_by_index(prior.source_index)
        b = senpai_run.get_frame_by_index(prior.target_index)
        if not (isinstance(a, RateTrackFrame) and isinstance(b, RateTrackFrame)):
            continue
        rate = _hop_rate(senpai_run, prior)
        if rate is not None:
            history.append(rate)

    if len(history) < gate.min_history_hops:
        return

    rate = _hop_rate(senpai_run, frame_shift)
    if rate is None:
        return

    median_rate = np.median(np.array(history), axis=0)
    median_mag = float(np.hypot(*median_rate))
    threshold = max(gate.max_rate_deviation_fraction * median_mag, gate.min_rate_deviation_px_s)

    def consistent(v: tuple[float, float]) -> bool:
        deviation = float(np.hypot(v[0] - median_rate[0], v[1] - median_rate[1]))
        reversed_dir = (v[0] * median_rate[0] + v[1] * median_rate[1]) < 0
        return deviation <= threshold and not reversed_dir

    if consistent(rate):
        return

    logger.warning(
        "Chain-consistency gate: hop %d->%d rate (%.1f, %.1f) px/s contradicts "
        "accepted chain median (%.1f, %.1f) px/s over %d hops",
        frame_shift.source_index,
        frame_shift.target_index,
        rate[0],
        rate[1],
        median_rate[0],
        median_rate[1],
        len(history),
    )

    # The rate-rate correlator has a known sign ambiguity: try the negated
    # hop before giving up, but only accept it if it independently validates.
    if consistent((-rate[0], -rate[1])):
        catalog_stars = source.starfield.catalog_stars if source.starfield else None
        if catalog_stars:
            fwhm = getattr(source.streak, "fwhm", None) if source.streak is not None else None
            valid, corr, _, _ = validate_proposed_shift(
                target,
                source,
                -frame_shift.x_shift,
                -frame_shift.y_shift,
                catalog_stars,
                trial=98,
                fwhm_exclusion=fwhm,
            )
            if valid:
                logger.warning(
                    "Chain-consistency gate: negated hop %d->%d is chain-consistent "
                    "and validates (corr=%.3f); flipping the shift sign",
                    frame_shift.source_index,
                    frame_shift.target_index,
                    corr,
                )
                frame_shift.x_shift = -frame_shift.x_shift
                frame_shift.y_shift = -frame_shift.y_shift
                return

    frame_shift.is_valid = False
    frame_shift.error_message = (
        f"Chain-consistency gate: hop rate ({rate[0]:.1f}, {rate[1]:.1f}) px/s "
        f"deviates from accepted chain median ({median_rate[0]:.1f}, {median_rate[1]:.1f}) px/s"
    )
