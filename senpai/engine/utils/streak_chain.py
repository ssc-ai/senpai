"""Reconcile per-frame streak models with the solved shift chain.

The streak extractor occasionally degenerates — the telltale signature is
``pixel_length == fwhm`` (it fit a blob, not a streak) — and such frames
carry lengths several times the truth, or angles unrelated to the drift.
Once the frame-to-frame shift chain is solved, the drift rate and direction
are known precisely, so the physical streak geometry (rate x exposure along
the drift axis) can overrule a deviant extraction. Star line training labels
and refinement kernels both inherit the corrected model.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from senpai.engine.models.senpai import (
        RateTrackFrame,
        RateTrackFrameSerializable,
        SenpaiRun,
        SenpaiRunResult,
        SiderealFrame,
        SiderealFrameSerializable,
    )

    # A per-frame model, either live (numpy/datetime fields) or serialized
    # (ISO-string timestamps); every accessor below is duck-typed via getattr.
    Frame = RateTrackFrame | SiderealFrame | RateTrackFrameSerializable | SiderealFrameSerializable
    Run = SenpaiRun | SenpaiRunResult

logger = logging.getLogger(__name__)


def _timestamp(frame: Frame) -> datetime | None:
    """Extract a frame's timestamp as a datetime, tolerating ISO strings.

    Args:
        frame: A live or serialized per-frame model.

    Returns:
        The frame timestamp as a datetime, or None if absent or unparseable.
    """
    ts = getattr(frame, "timestamp", None)
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return ts


def _exposure_seconds(frame: Frame) -> float | None:
    """Return a frame's exposure time in seconds, if available.

    Args:
        frame: A live or serialized per-frame model whose ``frame_metadata``
            may be a model or a plain dict.

    Returns:
        The exposure time in seconds, or None if unset or zero.
    """
    md = getattr(frame, "frame_metadata", None)
    exp = getattr(md, "exposure_time_seconds", None) if md is not None else None
    if exp is None and isinstance(md, dict):
        exp = md.get("exposure_time_seconds")
    return float(exp) if exp else None


def chain_drift_rates(senpai_run: Run) -> list[tuple[float, float]]:
    """Drift rate vectors (px/s) from the accepted hops of a run.

    Works on both live SenpaiRun objects and serialized SenpaiRunResult
    (ISO-string timestamps).

    Args:
        senpai_run: The run to read frames and accepted frame shifts from.

    Returns:
        One (x_rate, y_rate) pixel-per-second vector per valid, timed hop.
    """
    frames = {}
    for name in ("rate_track_frames", "sidereal_frames"):
        for fr in getattr(senpai_run, name, None) or []:
            frames[fr.index] = fr
    rates = []
    for shift in getattr(senpai_run, "frame_shifts", None) or []:
        if not (getattr(shift, "processed", True) and shift.is_valid):
            continue
        if shift.x_shift is None or shift.y_shift is None:
            continue
        a, b = frames.get(shift.source_index), frames.get(shift.target_index)
        if a is None or b is None:
            continue
        ta, tb = _timestamp(a), _timestamp(b)
        if ta is None or tb is None:
            continue
        dt = (tb - ta).total_seconds()
        if abs(dt) < 0.5:
            continue
        rates.append((shift.x_shift / dt, shift.y_shift / dt))
    return rates


def reconcile_streak_with_chain(
    frame: Frame,
    rates: list[tuple[float, float]],
    length_tolerance: float = 0.5,
    angle_tolerance_deg: float = 25.0,
) -> str | None:
    """Overrule a deviant streak model with chain-derived geometry.

    Only acts when the extraction is untrustworthy: the degenerate
    length==fwhm signature, a length off by more than *length_tolerance*
    (fractional), or an axis misaligned with the drift direction by more
    than *angle_tolerance_deg*.

    Args:
        frame: The rate-track frame whose ``streak`` model is corrected in place.
        rates: Chain-derived drift-rate vectors (px/s) from
            :func:`chain_drift_rates`; the median sets the expected geometry.
        length_tolerance: Fractional length deviation tolerated before the
            streak length is overruled.
        angle_tolerance_deg: Maximum angular misalignment (degrees) tolerated
            between the streak axis and the drift direction.

    Returns:
        A human-readable description of the corrections applied, or None when
        the model is trusted or there is no meaningful streak to reconcile.
    """
    streak = getattr(frame, "streak", None)
    if streak is None or not rates:
        return None
    exposure = _exposure_seconds(frame)
    if not exposure:
        return None

    r = np.array(rates, dtype=float)
    med = np.median(r, axis=0)
    rate_mag = float(np.hypot(*med))
    expected_length = rate_mag * exposure
    if expected_length < 2.0:
        return None  # near-sidereal: no meaningful streak to reconcile

    changes = []
    length = streak.pixel_length
    fwhm = streak.fwhm

    degenerate = (
        length is not None and fwhm is not None and abs(length - fwhm) < 0.01
    )
    length_off = (
        length is None
        or not np.isfinite(length)
        or abs(length - expected_length) > length_tolerance * expected_length
    )
    if degenerate or length_off:
        changes.append(f"length {length}->{expected_length:.1f}")
        streak.pixel_length = float(expected_length)
        if degenerate:
            seeing = getattr(frame, "seeing", None)
            seeing_fwhm = getattr(seeing, "pixel_fwhm", None) if seeing else None
            if seeing_fwhm and np.isfinite(seeing_fwhm):
                changes.append(f"fwhm {fwhm}->{seeing_fwhm:.1f}")
                streak.fwhm = float(seeing_fwhm)

    ca, sa = streak.cosine_angle, streak.sine_angle
    if ca is not None and sa is not None and np.isfinite([ca, sa]).all():
        axis = med / max(rate_mag, 1e-9)
        # sign-agnostic: a streak axis has a 180-degree ambiguity
        cos_mis = abs(ca * axis[0] + sa * axis[1]) / max(np.hypot(ca, sa), 1e-9)
        if cos_mis < np.cos(np.deg2rad(angle_tolerance_deg)):
            changes.append(
                f"angle ({ca:.2f},{sa:.2f})->({axis[0]:.2f},{axis[1]:.2f})"
            )
            streak.cosine_angle = float(axis[0])
            streak.sine_angle = float(axis[1])

    if changes:
        msg = "; ".join(changes)
        logger.info(
            "Reconciled streak model for frame %s with solved chain "
            "(rate=%.1f px/s): %s",
            getattr(frame, "index", "?"),
            rate_mag,
            msg,
        )
        return msg
    return None


def reconcile_run_streaks(senpai_run: Run, length_tolerance: float = 0.5,
                          angle_tolerance_deg: float = 25.0) -> int:
    """Reconcile every rate frame of a (possibly serialized) run in memory.

    Used at coco-export time so training labels get physical streak geometry
    even when the saved run predates in-pipeline reconciliation.

    Args:
        senpai_run: The run whose rate-track frames are reconciled in place.
        length_tolerance: Fractional length deviation tolerated before a
            streak length is overruled.
        angle_tolerance_deg: Maximum angular misalignment (degrees) tolerated
            between a streak axis and the drift direction.

    Returns:
        The number of frames whose streak model was changed.
    """
    rates = chain_drift_rates(senpai_run)
    if not rates:
        return 0
    n = 0
    for frame in getattr(senpai_run, "rate_track_frames", None) or []:
        if reconcile_streak_with_chain(
            frame, rates, length_tolerance, angle_tolerance_deg
        ):
            n += 1
    return n
