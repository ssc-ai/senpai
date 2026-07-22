"""Cross-frame streak correlation for multi-frame sidereal processing.

Given a SenpaiRun with multiple sidereal frames, this module:
1. Detects streaks in each sidereal frame that has a solved WCS.
2. Correlates streaks across frames to confirm real targets and resolve
   the 180-degree direction ambiguity.
3. (Optional) Correlates point-source detections in rate-track frames
   to predicted streak positions in sidereal frames.
"""

import logging
import uuid
from typing import TYPE_CHECKING

import numpy as np

from senpai.core.config import get_config
from senpai.engine.models.senpai import CorrelatedStreak, SenpaiRun, SiderealFrame

if TYPE_CHECKING:
    from senpai.engine.photometry.color_terms import MultiBandCalibration

logger = logging.getLogger(__name__)


def _deserialize_multiband(
    raw: "dict | MultiBandCalibration | None",
) -> "MultiBandCalibration | None":
    """Deserialize multiband_calibration from dict (photometry_summary) to MultiBandCalibration.

    Args:
        raw: The stored multiband calibration, either a plain dict from the
            photometry summary, an existing ``MultiBandCalibration``, or ``None``.

    Returns:
        The parsed ``MultiBandCalibration``, the value unchanged when it is
        already one, or ``None`` when absent or not parseable.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        try:
            from senpai.engine.photometry.color_terms import MultiBandCalibration

            return MultiBandCalibration.model_validate(raw)
        except Exception:
            return None
    return raw


# ---------------------------------------------------------------------------
# 1. Per-frame streak detection
# ---------------------------------------------------------------------------


def detect_streaks_in_sidereal_frames(
    senpai_run: SenpaiRun,
) -> dict[int, tuple[np.ndarray, float, np.ndarray]]:
    """Run streak detection + photometry on each sidereal frame with a solved WCS.

    Results are stored on ``frame.streak_candidates``.

    Returns:
        Dictionary mapping frame index to ``(directional_excess_map, noise_std, best_angle_deg)``
        for use in multi-frame DE-based confirmation.
    """
    from astropy.stats import sigma_clipped_stats

    from senpai.engine.detection.streak.sidereal_streak import (
        detect_streaks_in_sidereal,
        measure_streak_candidate_photometry,
    )

    config = get_config()
    de_data: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}

    for frame in senpai_run.sidereal_frames:
        if frame.starfield is None or not frame.starfield.fit:
            continue

        exposure_time = None
        if frame.frame_metadata and frame.frame_metadata.exposure_time_seconds:
            exposure_time = frame.frame_metadata.exposure_time_seconds

        try:
            candidates, directional_excess, best_angle_deg = detect_streaks_in_sidereal(
                frame.frame.data,
                frame.starfield,
                exposure_time=exposure_time,
            )
        except Exception as e:
            logger.warning("Streak detection failed for frame %d: %s", frame.index, e)
            continue

        # Save DE map, noise, and best_angle for multi-frame confirmation
        _, _, noise_std = sigma_clipped_stats(directional_excess, sigma=3.0, maxiters=5)
        de_data[frame.index] = (directional_excess, float(noise_std), best_angle_deg)

        # Measure photometry if we have a zero point
        if candidates and frame.photometry_summary:
            zp = frame.photometry_summary.get("zero_point")
            zp_err = frame.photometry_summary.get("zero_point_err")
            if zp is not None:
                fwhm = 4.0
                if frame.starfield.detection_metadata and frame.starfield.detection_metadata.pixel_fwhm:
                    fwhm = frame.starfield.detection_metadata.pixel_fwhm

                multiband_cal = _deserialize_multiband(frame.photometry_summary.get("multiband_calibration"))
                obs_filter = frame.frame_metadata.observation_filter if frame.frame_metadata else None

                try:
                    measure_streak_candidate_photometry(
                        frame.frame,
                        candidates,
                        zero_point=zp,
                        zero_point_err=zp_err,
                        exposure_time=exposure_time,
                        fwhm=fwhm,
                        gain=config.photometry.gain,
                        multiband_calibration=multiband_cal,
                        observation_filter=obs_filter,
                    )
                except Exception as e:
                    logger.warning("Streak photometry failed for frame %d: %s", frame.index, e)

        frame.streak_candidates = candidates
        logger.info("Frame %d: %d streak candidates detected", frame.index, len(candidates))

    return de_data


# ---------------------------------------------------------------------------
# 1b. Per-frame streak detection in rate-track frames
# ---------------------------------------------------------------------------


def detect_streaks_in_rate_frames(senpai_run: SenpaiRun) -> None:
    """Run streak detection on rate-track frames to find non-tracked objects.

    In a rate-track frame, stars appear as streaks at a consistent angle/length
    determined by the tracking rate and exposure time. Non-tracked objects
    (e.g., satellites at different angular rates) appear as streaks at different
    angles or lengths.

    This function:
    1. Runs directional matched filtering on each rate frame
    2. Filters out candidates matching the star streak angle+length (smeared stars)
    3. Stores remaining candidates on frame.streak_candidates
    """
    from senpai.engine.detection.streak.sidereal_streak import (
        detect_streaks_in_sidereal,
        measure_streak_candidate_photometry,
    )

    config = get_config()
    angle_tol = config.detection.streak_angle_tolerance_deg

    for frame in senpai_run.rate_track_frames:
        if frame.starfield is None or not frame.starfield.fit:
            continue
        if frame.streak is None:
            continue

        star_streak_angle = frame.streak.degree_angle()
        star_streak_length = frame.streak.pixel_length

        exposure_time = None
        if frame.frame_metadata and frame.frame_metadata.exposure_time_seconds:
            exposure_time = frame.frame_metadata.exposure_time_seconds

        try:
            # Star streaks (known angle/length from the tracking rate) are
            # excluded inside detection, before the expensive per-candidate
            # profile refinement.  The exclusion tolerance is wider than the
            # correlation tolerance: trail-residual angles are biased by
            # field aberration (coma tilts corner trails by ~15-20 deg).
            candidates, _, _ = detect_streaks_in_sidereal(
                frame.frame.data,
                frame.starfield,
                exposure_time=exposure_time,
                exclude_angle_deg=star_streak_angle,
                exclude_length_pixels=star_streak_length,
                exclude_angle_tol_deg=1.5 * angle_tol,
            )
        except Exception as e:
            logger.warning("Streak detection failed for rate frame %d: %s", frame.index, e)
            continue

        # Safety net: re-check the star-streak criterion on the surviving
        # candidates (refinement can update angle/length).
        non_star_candidates = []
        for c in candidates:
            angle_diff_val = _angle_diff(c.angle_deg, star_streak_angle)
            length_ratio = c.length_pixels / star_streak_length if star_streak_length > 0 else float("inf")

            # A star streak has consistent angle AND similar length
            is_star_streak = angle_diff_val < angle_tol and 0.5 < length_ratio < 2.0

            if not is_star_streak:
                non_star_candidates.append(c)

        # Measure photometry for remaining candidates
        if non_star_candidates and frame.photometry_summary:
            zp = frame.photometry_summary.get("zero_point")
            zp_err = frame.photometry_summary.get("zero_point_err")
            if zp is not None:
                fwhm = 4.0
                if frame.starfield.detection_metadata and frame.starfield.detection_metadata.pixel_fwhm:
                    fwhm = frame.starfield.detection_metadata.pixel_fwhm

                multiband_cal = _deserialize_multiband(frame.photometry_summary.get("multiband_calibration"))
                obs_filter = frame.frame_metadata.observation_filter if frame.frame_metadata else None

                try:
                    measure_streak_candidate_photometry(
                        frame.frame,
                        non_star_candidates,
                        zero_point=zp,
                        zero_point_err=zp_err,
                        exposure_time=exposure_time,
                        fwhm=fwhm,
                        gain=config.photometry.gain,
                        multiband_calibration=multiband_cal,
                        observation_filter=obs_filter,
                    )
                except Exception as e:
                    logger.warning("Streak photometry failed for rate frame %d: %s", frame.index, e)

        frame.streak_candidates = non_star_candidates
        logger.info(
            "Rate frame %d: %d non-star streak candidates (%d total, %d filtered as star streaks)",
            frame.index,
            len(non_star_candidates),
            len(candidates),
            len(candidates) - len(non_star_candidates),
        )


# ---------------------------------------------------------------------------
# 2. Cross-frame streak correlation
# ---------------------------------------------------------------------------


def _accumulate_shift(senpai_run: SenpaiRun, from_idx: int, to_idx: int) -> tuple[float, float] | None:
    """Accumulate pixel shifts along the frame_shifts chain from from_idx to to_idx.

    Returns (dx, dy) or None if no valid path exists.
    """
    if from_idx == to_idx:
        return 0.0, 0.0

    # Build adjacency from valid, processed shifts
    adj: dict[int, list[tuple[int, float, float]]] = {}
    for shift in senpai_run.frame_shifts:
        if shift.is_valid and shift.processed and shift.x_shift is not None:
            adj.setdefault(shift.source_index, []).append((shift.target_index, shift.x_shift, shift.y_shift))
            # Reverse direction
            adj.setdefault(shift.target_index, []).append((shift.source_index, -shift.x_shift, -shift.y_shift))

    # BFS to find path
    visited = {from_idx: (0.0, 0.0)}
    queue = [from_idx]
    while queue:
        current = queue.pop(0)
        if current == to_idx:
            return visited[to_idx]
        for neighbor, dx, dy in adj.get(current, []):
            if neighbor not in visited:
                cx, cy = visited[current]
                visited[neighbor] = (cx + dx, cy + dy)
                queue.append(neighbor)

    return None


def _angle_diff(a1: float, a2: float) -> float:
    """Minimum angular difference between two angles in [0, 180)."""
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)


def correlate_streaks_across_frames(senpai_run: SenpaiRun) -> list[CorrelatedStreak]:
    """Correlate streak candidates across sidereal frames.

    For each streak in frame N, predicts its position in frame N+k using
    measured frame shifts. Matches by position and angle. Resolves the
    180-degree direction ambiguity by checking which direction produces
    consistent motion across frames.

    Returns a list of CorrelatedStreak objects.
    """
    config = get_config()
    angle_tol = config.detection.streak_angle_tolerance_deg
    radius_fwhm = config.detection.streak_correlation_radius_fwhm

    # Collect all frames with streak candidates, sorted by index
    frames_with_streaks = [f for f in senpai_run.sidereal_frames if f.streak_candidates]
    frames_with_streaks.sort(key=lambda f: f.index)

    if len(frames_with_streaks) < 2:
        logger.info("Fewer than 2 frames with streaks, skipping correlation")
        # Return single-frame unconfirmed entries
        return _single_frame_streaks(frames_with_streaks)

    # Get FWHM for match radius
    fwhm = 4.0
    for f in frames_with_streaks:
        if f.starfield and f.starfield.detection_metadata and f.starfield.detection_metadata.pixel_fwhm:
            fwhm = f.starfield.detection_metadata.pixel_fwhm
            break

    match_radius = radius_fwhm * fwhm

    # Track which streaks have been claimed
    # Key: (frame_index, streak_list_index)
    claimed: set[tuple[int, int]] = set()
    correlated: list[CorrelatedStreak] = []

    # Sort all streaks by SNR (highest first) for greedy matching
    all_streaks = []
    for f in frames_with_streaks:
        for si, sc in enumerate(f.streak_candidates):
            snr = sc.peak_snr if hasattr(sc, "peak_snr") else 0
            all_streaks.append((snr, f.index, si, f, sc))
    all_streaks.sort(key=lambda x: x[0], reverse=True)

    for _, seed_fidx, seed_si, seed_frame, seed_streak in all_streaks:
        if (seed_fidx, seed_si) in claimed:
            continue

        # Start a chain from this streak
        chain_frames = [seed_frame]
        chain_streaks = [seed_streak]
        chain_indices = [(seed_fidx, seed_si)]
        claimed.add((seed_fidx, seed_si))

        # Try to extend to other frames
        for other_frame in frames_with_streaks:
            if other_frame.index == seed_fidx:
                continue

            # Accumulate shift from seed frame to other frame
            shift = _accumulate_shift(senpai_run, seed_fidx, other_frame.index)
            if shift is None:
                continue

            dx, dy = shift

            # Predicted position in other frame
            pred_x = seed_streak.x + dx
            pred_y = seed_streak.y + dy

            # Also account for streak motion: the streak moves during the
            # time between frames. With rate known, extrapolate position.
            if seed_streak.rate_pixels_per_sec and seed_frame.timestamp and other_frame.timestamp:
                dt = (other_frame.timestamp - seed_frame.timestamp).total_seconds()
                rate = seed_streak.rate_pixels_per_sec
                angle_rad = np.radians(seed_streak.angle_deg)
                # Try both directions (180 deg ambiguity)
                motion_dx_fwd = rate * dt * np.cos(angle_rad)
                motion_dy_fwd = rate * dt * np.sin(angle_rad)
            else:
                dt = 0
                motion_dx_fwd = 0
                motion_dy_fwd = 0

            # Search for matches in other frame
            best_match = None
            best_dist = match_radius
            best_direction = None  # "forward" or "reverse"

            for oi, other_streak in enumerate(other_frame.streak_candidates):
                if (other_frame.index, oi) in claimed:
                    continue

                # Check angle compatibility
                if _angle_diff(seed_streak.angle_deg, other_streak.angle_deg) > angle_tol:
                    continue

                # Try forward direction
                px_fwd = pred_x + motion_dx_fwd
                py_fwd = pred_y + motion_dy_fwd
                dist_fwd = np.sqrt((other_streak.x - px_fwd) ** 2 + (other_streak.y - py_fwd) ** 2)

                # Try reverse direction
                px_rev = pred_x - motion_dx_fwd
                py_rev = pred_y - motion_dy_fwd
                dist_rev = np.sqrt((other_streak.x - px_rev) ** 2 + (other_streak.y - py_rev) ** 2)

                if dist_fwd <= best_dist:
                    best_dist = dist_fwd
                    best_match = (other_frame, oi, other_streak)
                    best_direction = "forward"
                if dist_rev < best_dist:
                    best_dist = dist_rev
                    best_match = (other_frame, oi, other_streak)
                    best_direction = "reverse"

            if best_match is not None:
                mf, mi, ms = best_match
                chain_frames.append(mf)
                chain_streaks.append(ms)
                chain_indices.append((mf.index, mi))
                claimed.add((mf.index, mi))

        # Build CorrelatedStreak
        confirmed = len(chain_streaks) >= 2

        # Resolve direction
        direction_deg = None
        if confirmed and best_direction is not None:
            if best_direction == "forward":
                direction_deg = float(seed_streak.angle_deg % 360)
            else:
                direction_deg = float((seed_streak.angle_deg + 180) % 360)

        # Collect positions, timestamps, sky coords
        positions_x = [float(s.x) for s in chain_streaks]
        positions_y = [float(s.y) for s in chain_streaks]
        ra_list = [float(s.ra) for s in chain_streaks if s.ra is not None]
        dec_list = [float(s.dec) for s in chain_streaks if s.dec is not None]
        timestamps = [f.timestamp.isoformat() for f in chain_frames]
        frame_idxs = [f.index for f in chain_frames]

        # Best photometry: highest SNR streak in chain
        best = max(chain_streaks, key=lambda s: s.peak_snr)

        # Refine rate from multi-frame positions if confirmed
        refined_rate_pix = seed_streak.rate_pixels_per_sec
        refined_rate_arcsec = seed_streak.rate_arcsec_per_sec

        if confirmed and len(chain_frames) >= 2:
            # Use linear fit of position vs time
            times = [(f.timestamp - chain_frames[0].timestamp).total_seconds() for f in chain_frames]
            if max(times) > 0:
                xs = np.array(positions_x)
                ys = np.array(positions_y)
                ts = np.array(times)

                # Subtract frame shifts to isolate target motion
                target_xs = []
                target_ys = []
                for i, f in enumerate(chain_frames):
                    shift = _accumulate_shift(senpai_run, chain_frames[0].index, f.index)
                    if shift:
                        target_xs.append(xs[i] - shift[0])
                        target_ys.append(ys[i] - shift[1])
                    else:
                        target_xs.append(xs[i])
                        target_ys.append(ys[i])

                target_xs = np.array(target_xs)
                target_ys = np.array(target_ys)

                if len(ts) >= 2 and ts[-1] > 0:
                    dx_total = target_xs[-1] - target_xs[0]
                    dy_total = target_ys[-1] - target_ys[0]
                    total_motion = np.sqrt(dx_total**2 + dy_total**2)
                    refined_rate_pix = total_motion / ts[-1]

                    # Arcsec rate if we have plate scale
                    sf = chain_frames[0].starfield
                    if sf and sf.wcs_metadata and hasattr(sf.wcs_metadata, "x_ifov_arcsec"):
                        refined_rate_arcsec = refined_rate_pix * sf.wcs_metadata.x_ifov_arcsec

        cs = CorrelatedStreak(
            streak_id=str(uuid.uuid4())[:8],
            frame_indices=frame_idxs,
            positions_x=positions_x,
            positions_y=positions_y,
            ra=ra_list,
            dec=dec_list,
            timestamps_iso=timestamps,
            angle_deg=float(seed_streak.angle_deg),
            direction_deg=direction_deg,
            rate_pixels_per_sec=refined_rate_pix,
            rate_arcsec_per_sec=refined_rate_arcsec,
            confirmed=confirmed,
            best_snr=float(best.peak_snr),
            best_flux=best.flux,
            best_calibrated_magnitudes=best.calibrated_magnitudes,
            best_magnitude_errs=best.magnitude_errs,
        )
        correlated.append(cs)

    logger.info(
        "Streak correlation: %d correlated streaks (%d confirmed, %d single-frame)",
        len(correlated),
        sum(1 for c in correlated if c.confirmed),
        sum(1 for c in correlated if not c.confirmed),
    )

    return correlated


def _single_frame_streaks(frames_with_streaks: list[SiderealFrame]) -> list[CorrelatedStreak]:
    """Wrap single-frame streaks as unconfirmed CorrelatedStreak entries.

    Args:
        frames_with_streaks: Sidereal frames that carry streak candidates.

    Returns:
        One unconfirmed :class:`CorrelatedStreak` per streak candidate across
        the given frames.
    """
    result = []
    for frame in frames_with_streaks:
        for sc in frame.streak_candidates:
            cs = CorrelatedStreak(
                streak_id=str(uuid.uuid4())[:8],
                frame_indices=[frame.index],
                positions_x=[float(sc.x)],
                positions_y=[float(sc.y)],
                ra=[float(sc.ra)] if sc.ra is not None else [],
                dec=[float(sc.dec)] if sc.dec is not None else [],
                timestamps_iso=[frame.timestamp.isoformat()],
                angle_deg=float(sc.angle_deg),
                direction_deg=None,
                rate_pixels_per_sec=sc.rate_pixels_per_sec,
                rate_arcsec_per_sec=sc.rate_arcsec_per_sec,
                confirmed=False,
                best_snr=float(sc.peak_snr),
                best_flux=sc.flux,
                best_calibrated_magnitudes=sc.calibrated_magnitudes,
                best_magnitude_errs=sc.magnitude_errs,
            )
            result.append(cs)
    return result


# ---------------------------------------------------------------------------
# 3. Rate-to-sidereal cross-mode correlation
# ---------------------------------------------------------------------------


def correlate_rate_to_sidereal(senpai_run: SenpaiRun) -> None:
    """Correlate point-source detections in rate frames to streaks in sidereal frames.

    For each detection in a rate-track frame, predicts the corresponding
    streak position/angle in sidereal frames using the tracking rate and WCS.
    Matches against existing streak_candidates, creating or updating
    CorrelatedStreak entries.

    The target MOVES between the two frames: its rate-frame RA/Dec must be
    extrapolated to the sidereal frame's epoch before comparison (at
    ~30 arcsec/s and tens of seconds between frames the target travels
    hundreds of pixels — an unextrapolated comparison can never match).

    The motion is measured from the DATA: rate-frame detections of a
    tracked target sit at nearly the same pixel position in every frame,
    so they are clustered by pixel proximity and a linear RA/Dec-vs-time
    fit gives the target's sky rate directly.  Header track rates are only
    a fallback for single-frame detections — their sign conventions vary
    by mount (a real frame set had the Dec rate sign flipped), so all four
    sign combinations are tried.  A match also resolves the streak's
    180-degree direction ambiguity.
    """
    config = get_config()
    angle_tol = config.detection.streak_angle_tolerance_deg
    radius_fwhm = config.detection.streak_correlation_radius_fwhm

    if not senpai_run.rate_track_frames or not senpai_run.sidereal_frames:
        return

    # Get FWHM
    fwhm = 4.0
    for f in senpai_run.sidereal_frames:
        if f.starfield and f.starfield.detection_metadata and f.starfield.detection_metadata.pixel_fwhm:
            fwhm = f.starfield.detection_metadata.pixel_fwhm
            break

    base_radius = radius_fwhm * fwhm
    n_matches = 0

    # ---- 1. Collect rate-frame detections and cluster by pixel position --
    # (the mount tracks the target, so the same object stays at ~the same
    # pixel across rate frames)
    entries = []  # (frame, detection, t_seconds_rel)
    t_ref = None
    for rate_frame in senpai_run.rate_track_frames:
        if not rate_frame.detections or not rate_frame.detections.detections:
            continue
        if not rate_frame.starfield or not rate_frame.starfield.wcs:
            continue
        if rate_frame.timestamp is None:
            continue
        if t_ref is None:
            t_ref = rate_frame.timestamp
        for det in rate_frame.detections.detections:
            if det.ra is None or det.dec is None:
                continue
            t = (rate_frame.timestamp - t_ref).total_seconds()
            entries.append((rate_frame, det, t))

    if not entries:
        logger.info("Rate-to-sidereal correlation: no rate detections to correlate")
        return

    cluster_radius = 3 * fwhm
    clusters: list[list[tuple]] = []
    for entry in entries:
        _, det, _ = entry
        for cluster in clusters:
            cx = np.mean([e[1].x for e in cluster])
            cy = np.mean([e[1].y for e in cluster])
            if np.hypot(det.x - cx, det.y - cy) <= cluster_radius:
                cluster.append(entry)
                break
        else:
            clusters.append([entry])

    # ---- 2. Predict each cluster's position in each sidereal frame -------
    for cluster in clusters:
        times = np.array([e[2] for e in cluster])
        ras = np.array([e[1].ra for e in cluster])
        decs = np.array([e[1].dec for e in cluster])
        time_span = times.max() - times.min()

        measured_rate = None  # (dra_deg_per_s, ddec_deg_per_s)
        if len(cluster) >= 2 and time_span >= 2.0:
            measured_rate = (
                float(np.polyfit(times, ras, 1)[0]),
                float(np.polyfit(times, decs, 1)[0]),
            )

        # Fallback rates from the header, all four sign conventions
        fallback_rates = []
        if measured_rate is None:
            fm = cluster[0][0].frame_metadata
            rate_ra = (fm.track_rate_ra_arcsec_per_second or 0) if fm else 0
            rate_dec = (fm.track_rate_dec_arcsec_per_second or 0) if fm else 0
            if rate_ra or rate_dec:
                fallback_rates = [
                    (sr * rate_ra / 3600.0, sd * rate_dec / 3600.0)
                    for sr in (1.0, -1.0)
                    for sd in (1.0, -1.0)
                ]

        rate_hypotheses = [measured_rate] if measured_rate else fallback_rates
        if not rate_hypotheses:
            continue

        # Anchor the prediction at the cluster's mean epoch/position
        t_mean = float(times.mean())
        ra_mean = float(ras.mean())
        dec_mean = float(decs.mean())
        anchor_det = cluster[0][1]

        for sid_frame in senpai_run.sidereal_frames:
            if not sid_frame.streak_candidates:
                continue
            if not sid_frame.starfield or not sid_frame.starfield.wcs:
                continue
            if sid_frame.timestamp is None:
                continue

            dt = (sid_frame.timestamp - t_ref).total_seconds() - t_mean

            try:
                sid_wcs = sid_frame.starfield.wcs.to_astropy_wcs()
            except Exception:  # noqa: S112
                continue

            ifov = None
            if sid_frame.starfield.wcs_metadata and hasattr(
                sid_frame.starfield.wcs_metadata, "x_ifov_arcsec"
            ):
                ifov = sid_frame.starfield.wcs_metadata.x_ifov_arcsec

            best = None  # (dist, sc, rate, pred_x, pred_y)
            for rate in rate_hypotheses:
                ra_pred = ra_mean + rate[0] * dt
                dec_pred = dec_mean + rate[1] * dt
                try:
                    px_coords = sid_wcs.all_world2pix([[ra_pred, dec_pred]], 0)
                    pred_x, pred_y = float(px_coords[0][0]), float(px_coords[0][1])
                except Exception:  # noqa: S112
                    continue

                # Rate uncertainty grows the position error linearly with
                # the extrapolated distance: allow 5% on top of the base
                # match radius.
                travel_arcsec = float(np.hypot(rate[0], rate[1])) * 3600.0 * abs(dt)
                travel_px = travel_arcsec / ifov if ifov else 0.0
                match_radius = base_radius + 0.05 * travel_px

                for sc in sid_frame.streak_candidates:
                    dist = float(np.hypot(sc.x - pred_x, sc.y - pred_y))
                    if dist < match_radius and (best is None or dist < best[0]):
                        best = (dist, sc, rate, pred_x, pred_y)

            if best is None:
                continue
            dist, sc, rate, pred_x, pred_y = best

            # The streak angle in the sidereal frame must match the motion
            # direction mapped through the WCS
            try:
                step = 1.0  # seconds of motion for the direction vector
                px2 = sid_wcs.all_world2pix(
                    [[ra_pred + rate[0] * step, dec_pred + rate[1] * step]], 0
                )
                dx = float(px2[0][0]) - pred_x
                dy = float(px2[0][1]) - pred_y
                motion_angle = float(np.degrees(np.arctan2(dy, dx)))
                predicted_angle = motion_angle % 180
            except Exception:  # noqa: S112
                continue

            angle_d = _angle_diff(sc.angle_deg, predicted_angle)
            if angle_d >= angle_tol:
                continue

            n_matches += 1
            logger.info(
                "Rate detection cluster at (%.1f,%.1f) (%d frames, %s rate) "
                "extrapolated %.1fs matches sidereal streak in frame %d at "
                "(%.1f,%.1f): dist=%.1f, angle_diff=%.1f",
                anchor_det.x,
                anchor_det.y,
                len(cluster),
                "measured" if measured_rate else "header",
                dt,
                sid_frame.index,
                sc.x,
                sc.y,
                dist,
                angle_d,
            )
            # Mark matching correlated streak as confirmed; the motion
            # direction that produced the match resolves the streak's
            # 180-degree direction ambiguity.
            for cs in senpai_run.correlated_streaks:
                if sid_frame.index in cs.frame_indices:
                    for px, py in zip(cs.positions_x, cs.positions_y, strict=False):
                        if abs(px - sc.x) < 1 and abs(py - sc.y) < 1:
                            cs.confirmed = True
                            if cs.direction_deg is None:
                                cs.direction_deg = motion_angle % 360
                            break

    logger.info("Rate-to-sidereal correlation: %d matches found", n_matches)
