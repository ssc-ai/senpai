"""Multi-frame streak confirmation via rate scanning across DE maps.

For each single-frame streak candidate (which provides position and angle),
scans trial rates in both forward/reverse directions across the directional
excess maps of all frames.  The rate that maximizes the multi-frame DE sum
is the detection.

This replaces the fragile single-pixel DE sampling and profile cross-correlation
approaches, which fail for faint streaks because:
- Single-pixel DE sampling is too noisy
- Rate errors from single-frame trace length compound across frames
- Direction ambiguity can't be resolved from noisy profiles

The rate scan is computationally trivial: ~200 trial rates × 2 directions
× N_frames pixel lookups per candidate.  The DE maps are already computed
during per-frame detection.
"""

import logging
import uuid
from typing import TYPE_CHECKING

import numpy as np

from senpai.engine.models.senpai import CorrelatedStreak, SenpaiRun, SiderealFrame

if TYPE_CHECKING:
    from senpai.engine.detection.streak.sidereal_streak import StreakCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame shift accumulation (shared with stamp_confirmation)
# ---------------------------------------------------------------------------


def _accumulate_shift(
    senpai_run: SenpaiRun, from_idx: int, to_idx: int
) -> tuple[float, float] | None:
    """Accumulate pixel shifts from from_idx to to_idx via BFS."""
    if from_idx == to_idx:
        return 0.0, 0.0

    adj: dict[int, list[tuple[int, float, float]]] = {}
    for shift in senpai_run.frame_shifts:
        if shift.is_valid and shift.processed and shift.x_shift is not None:
            adj.setdefault(shift.source_index, []).append(
                (shift.target_index, shift.x_shift, shift.y_shift)
            )
            adj.setdefault(shift.target_index, []).append(
                (shift.source_index, -shift.x_shift, -shift.y_shift)
            )

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


# ---------------------------------------------------------------------------
# Rate scan confirmation
# ---------------------------------------------------------------------------


def confirm_streaks_via_rate_scan(
    senpai_run: SenpaiRun,
    de_data: dict[int, tuple[np.ndarray, float, np.ndarray]],
) -> list[CorrelatedStreak]:
    """Confirm streak candidates by scanning trial rates across DE maps.

    For each candidate from per-frame detection:
    1. Use the candidate's position and angle (from the matched filter)
    2. Scan trial rates from 1-60 px/s in forward and reverse directions
    3. At each trial rate, sum DE values at predicted positions across frames
    4. The (rate, direction) that maximizes the sum is the best match
    5. Require signal in ≥2 frames and stacked SNR above threshold

    Returns confirmed CorrelatedStreak objects.
    """
    all_frames = [
        f for f in senpai_run.sidereal_frames if f.starfield is not None
    ]
    all_frames.sort(key=lambda f: f.index)

    frames_with_candidates = [f for f in all_frames if f.streak_candidates]

    if not frames_with_candidates:
        return []

    if len(all_frames) < 2:
        return _confirm_single_frame(frames_with_candidates)

    # FWHM for noise and position thresholds
    fwhm = 4.0
    for f in all_frames:
        sf = f.starfield
        if sf and sf.detection_metadata and sf.detection_metadata.pixel_fwhm:
            fwhm = sf.detection_metadata.pixel_fwhm
            break

    # Rate scan grid
    rate_step = 0.5  # px/s — gives ~1.5px precision at dt=3s
    trial_rates = np.arange(1.0, 61.0, rate_step)

    # Precompute frame shifts and timestamps relative to each reference
    frame_info: dict[int, dict] = {}
    for f in all_frames:
        frame_info[f.index] = {"frame": f, "timestamp": f.timestamp}

    confirmed: list[CorrelatedStreak] = []
    n_rejected = 0

    for ref_frame in frames_with_candidates:
        if ref_frame.index not in de_data:
            continue

        ref_de, ref_noise, _ref_ba = de_data[ref_frame.index]

        # Precompute other-frame data
        other_frames = []
        for f in all_frames:
            if f.index == ref_frame.index or f.index not in de_data:
                continue
            shift = _accumulate_shift(senpai_run, ref_frame.index, f.index)
            if shift is None:
                continue
            dt = 0.0
            if ref_frame.timestamp and f.timestamp:
                dt = (f.timestamp - ref_frame.timestamp).total_seconds()
            if abs(dt) < 0.1:
                continue
            other_de, other_noise, other_ba = de_data[f.index]
            # Build star mask as 2D boolean array (fast numpy-based lookup)
            star_mask = np.zeros(other_de.shape, dtype=bool)
            if f.starfield and f.starfield.catalog_stars:
                star_mask_r = int(np.ceil(fwhm * 3))
                oh, ow = other_de.shape
                for s in f.starfield.catalog_stars:
                    if s.x is not None and s.y is not None:
                        ix, iy = int(round(s.x)), int(round(s.y))
                        y_lo = max(0, iy - star_mask_r)
                        y_hi = min(oh, iy + star_mask_r + 1)
                        x_lo = max(0, ix - star_mask_r)
                        x_hi = min(ow, ix + star_mask_r + 1)
                        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
                        dist_sq = (xx - s.x)**2 + (yy - s.y)**2
                        star_mask[y_lo:y_hi, x_lo:x_hi] |= dist_sq <= star_mask_r**2
            other_frames.append({
                "frame": f,
                "de": other_de,
                "noise": other_noise,
                "best_angle_deg": other_ba,
                "shift": shift,
                "dt": dt,
                "star_mask": star_mask,
            })

        if not other_frames:
            continue

        for candidate in ref_frame.streak_candidates:
            result = _rate_scan_candidate(
                candidate=candidate,
                ref_de=ref_de,
                ref_noise=ref_noise,
                ref_frame=ref_frame,
                other_frames=other_frames,
                trial_rates=trial_rates,
                fwhm=fwhm,
                senpai_run=senpai_run,
                all_frames=all_frames,
                de_data=de_data,
            )
            if result is not None:
                confirmed.append(result)
            else:
                n_rejected += 1

    # Deduplicate: same streak detected from different reference frames
    deduped = _deduplicate(confirmed, fwhm)
    n_deduped = len(confirmed) - len(deduped)

    # Propagate confirmed streaks to per-frame detections
    for f in senpai_run.sidereal_frames:
        f.streak_candidates = []

    for cs in deduped:
        if cs.confirmed:
            _propagate_to_frames(cs, senpai_run, all_frames, fwhm)

    logger.info(
        "Rate scan confirmation: %d confirmed (%d deduped), %d rejected",
        sum(1 for c in deduped if c.confirmed),
        n_deduped, n_rejected,
    )
    return deduped


def _rate_scan_candidate(
    candidate: "StreakCandidate",
    ref_de: np.ndarray,
    ref_noise: float,
    ref_frame: SiderealFrame,
    other_frames: list[dict],
    trial_rates: np.ndarray,
    fwhm: float,
    senpai_run: SenpaiRun,
    all_frames: list,
    de_data: dict,
) -> CorrelatedStreak | None:
    """Run rate scan for a single candidate.

    Args:
        candidate: The streak candidate from the reference frame to confirm.
        ref_de: The reference frame's directional-excess map.
        ref_noise: Noise standard deviation of the reference DE map.
        ref_frame: The reference sidereal frame the candidate came from.
        other_frames: Per-frame DE data for the non-reference frames.
        trial_rates: Grid of trial rates (pixels/second) to scan.
        fwhm: Representative point-source FWHM in pixels.
        senpai_run: The run being processed (for WCS/timestamps).
        all_frames: All candidate frames participating in the scan.
        de_data: Per-frame directional-excess data keyed by frame index.

    Returns:
        A confirmed :class:`CorrelatedStreak` for the candidate, or ``None`` when
        it fails the confirmation criteria.
    """
    cx, cy = candidate.x, candidate.y
    h, w = ref_de.shape

    # Reject candidates near image border (edge artifacts)
    border = int(fwhm * 3)
    if cx < border or cx >= w - border or cy < border or cy >= h - border:
        logger.debug("Rejected (%.0f,%.0f): near border", cx, cy)
        return None

    # Reference frame DE at candidate position
    ref_val = _sample_de(ref_de, cx, cy)
    if ref_val < 5 * ref_noise:
        logger.debug("Rejected (%.0f,%.0f): ref DE %.2f < 5*noise %.2f", cx, cy, ref_val, 5*ref_noise)
        return None  # Not significant in reference frame

    n_other = len(other_frames)
    n_total = n_other + 1  # Including reference

    # Constrain angle search to the per-frame detection angle ± tolerance.
    # The matched filter angle is reliable for bright streaks and only slightly
    # off for faint/clipped ones.  Searching all 36 angles (8640 trials) causes
    # massive false positives because artifact-correlated angles (~0°/180°
    # from row median subtraction) systematically win.
    cand_angle = candidate.angle_deg
    angle_tolerance = 15.0  # degrees
    trial_angles = np.arange(
        cand_angle - angle_tolerance,
        cand_angle + angle_tolerance + 2.5,
        5.0,
    ) % 180

    # Vectorized rate scan: compute all (angle, rate, direction) at once
    # for each other frame, then sum across frames.
    n_angles = len(trial_angles)
    n_rates = len(trial_rates)
    directions = np.array([+1, -1])
    n_dirs = 2

    cos_angles = np.cos(np.radians(trial_angles))
    sin_angles = np.sin(np.radians(trial_angles))

    # Shape: (n_angles, n_rates, n_dirs) — total DE for each combination
    total_de = np.full((n_angles, n_rates, n_dirs), ref_val)
    # Per-frame DE values for the best combination
    per_frame_de = np.zeros((len(other_frames), n_angles, n_rates, n_dirs))
    # Validity mask (in-bounds)
    valid = np.ones((n_angles, n_rates, n_dirs), dtype=bool)

    for fi, ofd in enumerate(other_frames):
        dx, dy = ofd["shift"]
        dt = ofd["dt"]
        de_map = ofd["de"]
        oh, ow = de_map.shape

        # Predicted positions: (n_angles, n_rates, n_dirs)
        # pred_x = cx + dx + dir * rate * dt * cos_a
        pred_x = (cx + dx
                   + directions[None, None, :] * trial_rates[None, :, None] * dt
                   * cos_angles[:, None, None])
        pred_y = (cy + dy
                   + directions[None, None, :] * trial_rates[None, :, None] * dt
                   * sin_angles[:, None, None])

        # Bounds check
        oob = (pred_x < 0) | (pred_x >= ow) | (pred_y < 0) | (pred_y >= oh)
        valid &= ~oob

        # Sample DE at in-bounds positions using bilinear interpolation
        # Clip to valid range for sampling, then zero out OOB
        px_clipped = np.clip(pred_x, 0, ow - 1.001)
        py_clipped = np.clip(pred_y, 0, oh - 1.001)

        x0 = px_clipped.astype(np.int32)
        y0 = py_clipped.astype(np.int32)
        x1 = np.minimum(x0 + 1, ow - 1)
        y1 = np.minimum(y0 + 1, oh - 1)
        fx = px_clipped - x0
        fy = py_clipped - y0

        vals = (de_map[y0, x0] * (1 - fx) * (1 - fy)
                + de_map[y0, x1] * fx * (1 - fy)
                + de_map[y1, x0] * (1 - fx) * fy
                + de_map[y1, x1] * fx * fy)

        # Zero out OOB positions
        vals[oob] = 0.0

        # Zero out star mask positions using 2D boolean array
        star_mask = ofd["star_mask"]
        if star_mask.any():
            ipx = np.clip(np.round(pred_x).astype(np.int64), 0, ow - 1)
            ipy = np.clip(np.round(pred_y).astype(np.int64), 0, oh - 1)
            on_star = star_mask[ipy, ipx]
            vals[on_star] = 0.0

        per_frame_de[fi] = vals
        total_de += vals

    # Mask invalid combinations
    total_de[~valid] = -np.inf

    # Find the best (angle, rate, direction)
    best_idx = np.unravel_index(total_de.argmax(), total_de.shape)
    best_ai, best_ri, best_di = best_idx
    best_sum = float(total_de[best_ai, best_ri, best_di])
    best_rate = float(trial_rates[best_ri])
    best_dir = int(directions[best_di])
    best_angle = float(trial_angles[best_ai])
    best_other_vals = [float(per_frame_de[fi, best_ai, best_ri, best_di])
                       for fi in range(len(other_frames))]

    angle_deg = best_angle

    # --- Confirmation criteria ---
    # 1. Stacked SNR must be high
    noise_combined = ref_noise * np.sqrt(n_total)
    stacked_snr = best_sum / noise_combined

    # 2. Other frames must contribute real signal (not just reference)
    #    Require at least 1 other frame with DE > 3σ
    n_other_with_signal = sum(
        1 for v, ofd in zip(best_other_vals, other_frames, strict=False)
        if v > 3 * ofd["noise"]
    )

    # 3. The best rate must beat a "no motion" baseline
    #    (catches static artifacts that have signal at the same position in all frames)
    static_sum = ref_val
    for ofd in other_frames:
        dx, dy = ofd["shift"]
        static_x = cx + dx
        static_y = cy + dy
        static_sum += _sample_de(ofd["de"], static_x, static_y)

    motion_boost = best_sum - static_sum

    # With constrained angle search (~7 angles × 120 rates × 2 dirs ≈ 1680 trials),
    # the expected max under null is moderate.  Require stacked SNR > 12
    # for detection.  A real streak with 3 frames at ~10σ each gives
    # stacked_snr ≈ 17.
    all_vals = [ref_val, *best_other_vals]

    if stacked_snr < 12.0:
        logger.debug("Rejected (%.0f,%.0f): stacked_snr %.1f < 12 vals=%s", cx, cy, stacked_snr, [f"{v:.1f}" for v in all_vals])
        return None

    # Must have signal in at least half the other frames
    if n_other_with_signal < max(1, n_other // 2):
        logger.debug("Rejected (%.0f,%.0f): n_other_signal=%d < %d vals=%s", cx, cy, n_other_with_signal, max(1, n_other // 2), [f"{v:.1f}" for v in all_vals])
        return None

    # Multi-frame consistency: real streaks have similar DE in all frames.
    positive_vals = [v for v in all_vals if v > 0]
    if len(positive_vals) < 2:
        logger.debug("Rejected (%.0f,%.0f): < 2 positive vals, vals=%s", cx, cy, [f"{v:.1f}" for v in all_vals])
        return None  # Need at least 2 frames with positive DE

    # Check that the weakest frame has at least 25% of the strongest.
    min_positive = min(positive_vals)
    max_positive = max(positive_vals)
    if max_positive > 0 and min_positive / max_positive < 0.25:
        logger.debug("Rejected (%.0f,%.0f): consistency ratio %.2f < 0.25 vals=%s", cx, cy, min_positive/max_positive, [f"{v:.1f}" for v in all_vals])
        return None  # Too inconsistent — one frame dominates

    # 4. Angle validation: at predicted positions in other frames, check
    # that the DE map's best_angle is consistent with the candidate angle.
    # This rejects cases where the DE at a predicted position comes from
    # a different source (star residual) at a different angle.
    # Check at the peak DE position within the search radius (more robust
    # to sub-pixel position errors than checking the exact predicted pixel).
    best_cos_a = np.cos(np.radians(best_angle))
    best_sin_a = np.sin(np.radians(best_angle))
    n_angle_consistent = 0
    r_search = max(1, int(round(fwhm)))  # Search along streak direction
    for v, ofd in zip(best_other_vals, other_frames, strict=False):
        if v <= 3 * ofd["noise"]:
            continue
        dx, dy = ofd["shift"]
        dt = ofd["dt"]
        pred_x = cx + dx + best_dir * best_rate * dt * best_cos_a
        pred_y = cy + dy + best_dir * best_rate * dt * best_sin_a
        de_map = ofd["de"]
        oh, ow = de_map.shape
        ba_map = ofd.get("best_angle_deg")
        if ba_map is not None:
            # Check angle at predicted position and a few offsets along
            # the streak direction (more robust to position errors)
            angle_ok = False
            for offset in range(-r_search, r_search + 1):
                sx = pred_x + offset * best_cos_a
                sy = pred_y + offset * best_sin_a
                six, siy = int(round(sx)), int(round(sy))
                if 0 <= siy < oh and 0 <= six < ow and de_map[siy, six] > 3 * ofd["noise"]:
                    local_angle = float(ba_map[siy, six])
                    adiff = abs(local_angle - best_angle) % 180
                    adiff = min(adiff, 180 - adiff)
                    if adiff < 25:
                        angle_ok = True
                        break
            if angle_ok:
                n_angle_consistent += 1
        else:
            n_angle_consistent += 1  # No angle map, skip check

    # Require angle consistency in at least 1 other frame with signal
    if n_other_with_signal > 0 and n_angle_consistent == 0:
        logger.debug("Rejected (%.0f,%.0f): angle inconsistent (0/%d) at rate=%.1f angle=%.1f vals=%s", cx, cy, n_other_with_signal, best_rate, best_angle, [f"{v:.1f}" for v in all_vals])
        return None

    # 5. Perpendicular-angle check: a real streak at angle θ should NOT have
    # comparable DE when sampled at angle θ+90°.  Star residuals and artifacts
    # produce isotropic DE, so they'll score similarly at any angle.  Scan
    # the same best rate at the perpendicular angle — if it gives comparable
    # signal, this is not a real streak.
    perp_angle = (best_angle + 90) % 180
    perp_cos = np.cos(np.radians(perp_angle))
    perp_sin = np.sin(np.radians(perp_angle))
    perp_best = -np.inf
    for rate in [best_rate]:  # Just check at the best rate
        for direction in [+1, -1]:
            perp_total = ref_val  # Same reference
            for ofd in other_frames:
                dx, dy = ofd["shift"]
                dt = ofd["dt"]
                pred_x = cx + dx + direction * rate * dt * perp_cos
                pred_y = cy + dy + direction * rate * dt * perp_sin
                oh, ow = ofd["de"].shape
                if 0 <= pred_x < ow and 0 <= pred_y < oh:
                    perp_total += _sample_de(ofd["de"], pred_x, pred_y)
            perp_best = max(perp_best, perp_total)

    # If perpendicular gives ≥60% of the best angle's signal, reject
    if perp_best > 0 and best_sum > 0 and perp_best / best_sum > 0.6:
        logger.debug("Rejected (%.0f,%.0f): perp ratio %.2f > 0.6 vals=%s", cx, cy, perp_best/best_sum, [f"{v:.1f}" for v in all_vals])
        return None

    min_positive = min(positive_vals) if positive_vals else 0
    logger.debug(
        "Candidate (%.0f,%.0f) rate=%.1f angle=%.1f: vals=%s ratio=%.2f stacked=%.1f angle_ok=%d/%d perp_ratio=%.2f",
        cx, cy, best_rate, best_angle,
        [f"{v:.2f}" for v in all_vals],
        min_positive / max_positive if max_positive > 0 else 0,
        stacked_snr, n_angle_consistent, n_other_with_signal,
        perp_best / best_sum if best_sum > 0 else 0,
    )

    # If static positions give similar or better signal, it's a static artifact
    if motion_boost < 2 * ref_noise and best_rate > 2.0:
        logger.debug("Rejected (%.0f,%.0f): static artifact, motion_boost=%.2f < 2*noise=%.2f", cx, cy, motion_boost, 2*ref_noise)
        return None

    # --- Build the confirmed streak ---
    exposure_time = None
    if ref_frame.frame_metadata and ref_frame.frame_metadata.exposure_time_seconds:
        exposure_time = ref_frame.frame_metadata.exposure_time_seconds

    # Streak length from rate × exposure
    length = best_rate * exposure_time if exposure_time else best_rate

    # Direction in [0, 360)
    direction_deg = float(angle_deg % 360) if best_dir > 0 else float((angle_deg + 180) % 360)

    # Positions in each frame (use best_angle's cos/sin, not loop variable)
    frame_indices = [ref_frame.index]
    positions_x = [float(cx)]
    positions_y = [float(cy)]

    for ofd in other_frames:
        dx, dy = ofd["shift"]
        dt = ofd["dt"]
        pred_x = cx + dx + best_dir * best_rate * dt * best_cos_a
        pred_y = cy + dy + best_dir * best_rate * dt * best_sin_a
        frame_indices.append(ofd["frame"].index)
        positions_x.append(float(pred_x))
        positions_y.append(float(pred_y))

    # Sort by frame index
    order = np.argsort(frame_indices)
    frame_indices = [frame_indices[i] for i in order]
    positions_x = [positions_x[i] for i in order]
    positions_y = [positions_y[i] for i in order]

    # Sky coordinates and RA/Dec rates
    ra_list, dec_list = [], []
    rate_ra, rate_dec, rate_arcsec = None, None, None

    wcs_obj = None
    for fi, px, py in zip(frame_indices, positions_x, positions_y, strict=False):
        f = next((f for f in all_frames if f.index == fi), None)
        if f and f.starfield and f.starfield.wcs:
            try:
                wcs_obj = f.starfield.wcs.to_astropy_wcs()
                sky = wcs_obj.pixel_to_world(px, py)
                ra_list.append(float(sky.ra.deg))
                dec_list.append(float(sky.dec.deg))
            except Exception as err:
                logger.debug("WCS pixel_to_world failed for frame %s: %s", fi, err)

    if len(ra_list) >= 2:
        try:
            f0 = next(f for f in all_frames if f.index == frame_indices[0])
            fn = next(f for f in all_frames if f.index == frame_indices[-1])
            dt_total = (fn.timestamp - f0.timestamp).total_seconds()
            if dt_total > 0:
                dra = (ra_list[-1] - ra_list[0]) * 3600 * np.cos(np.radians(dec_list[0]))
                ddec = (dec_list[-1] - dec_list[0]) * 3600
                rate_ra = float(dra / dt_total)
                rate_dec = float(ddec / dt_total)
                rate_arcsec = float(np.sqrt(dra**2 + ddec**2) / dt_total)

                # Refine pixel-space angle from RA/Dec velocity
                if wcs_obj is not None:
                    import astropy.units as u
                    from astropy.coordinates import SkyCoord
                    sky0 = wcs_obj.pixel_to_world(positions_x[0], positions_y[0])
                    sky1 = SkyCoord(
                        ra=sky0.ra + (rate_ra / 3600 / np.cos(np.radians(dec_list[0]))) * u.deg,
                        dec=sky0.dec + (rate_dec / 3600) * u.deg,
                    )
                    px1 = wcs_obj.all_world2pix([[sky1.ra.deg, sky1.dec.deg]], 0)
                    dx_pix = float(px1[0][0]) - positions_x[0]
                    dy_pix = float(px1[0][1]) - positions_y[0]
                    refined_rate = float(np.sqrt(dx_pix**2 + dy_pix**2))
                    angle_deg = float(np.degrees(np.arctan2(dy_pix, dx_pix))) % 180
                    direction_deg = float(np.degrees(np.arctan2(dy_pix, dx_pix))) % 360
                    if exposure_time:
                        length = refined_rate * exposure_time
                    best_rate = refined_rate
        except Exception as e:
            logger.debug("RA/Dec rate computation failed: %s", e)

    if rate_arcsec is None:
        sf = ref_frame.starfield
        if sf and sf.wcs_metadata and hasattr(sf.wcs_metadata, "x_ifov_arcsec"):
            rate_arcsec = best_rate * sf.wcs_metadata.x_ifov_arcsec

    # Measure photometry on the reference frame
    cal_mags, mag_errs, flux = None, None, candidate.flux
    if ref_frame.photometry_summary:
        zp = ref_frame.photometry_summary.get("zero_point")
        zp_err = ref_frame.photometry_summary.get("zero_point_err")
        if zp is not None:
            from senpai.core.config import get_config
            from senpai.engine.detection.streak.sidereal_streak import (
                StreakCandidate,
                measure_streak_candidate_photometry,
            )
            config = get_config()
            obs_filter = ref_frame.frame_metadata.observation_filter if ref_frame.frame_metadata else None
            multiband = None
            multiband_raw = ref_frame.photometry_summary.get("multiband_calibration")
            if multiband_raw:
                try:
                    from senpai.engine.photometry.color_terms import MultiBandCalibration
                    multiband = MultiBandCalibration.model_validate(multiband_raw) if isinstance(multiband_raw, dict) else multiband_raw
                except Exception as err:
                    logger.debug("Multiband calibration validation failed: %s", err)
            phot_candidate = StreakCandidate(
                x=float(cx), y=float(cy),
                angle_deg=float(angle_deg),
                length_pixels=float(length),
                width_pixels=float(fwhm),
                peak_snr=float(stacked_snr),
                directional_excess=0.0, fractional_excess=0.0,
            )
            try:
                measure_streak_candidate_photometry(
                    ref_frame.frame, [phot_candidate],
                    zero_point=zp, zero_point_err=zp_err,
                    exposure_time=exposure_time, fwhm=fwhm,
                    gain=config.photometry.gain,
                    multiband_calibration=multiband,
                    observation_filter=obs_filter,
                )
                flux = phot_candidate.flux
                cal_mags = phot_candidate.calibrated_magnitudes
                mag_errs = phot_candidate.magnitude_errs
            except Exception as e:
                logger.debug("Streak photometry failed: %s", e)

    # Reject if photometric flux is not positive — star residuals and
    # artifacts produce zero or negative flux in oriented rectangular apertures
    if flux is not None and flux <= 0:
        return None

    timestamps = []
    for fi in frame_indices:
        f = next((f for f in all_frames if f.index == fi), None)
        if f and f.timestamp:
            timestamps.append(f.timestamp.isoformat())

    logger.info(
        "Rate scan confirmed streak at (%.0f,%.0f): rate=%.1f px/s, "
        "angle=%.1f°, dir=%.1f°, stacked_snr=%.1f, n_frames=%d",
        cx, cy, best_rate, angle_deg, direction_deg,
        stacked_snr, n_other_with_signal + 1,
    )

    return CorrelatedStreak(
        streak_id=str(uuid.uuid4())[:8],
        frame_indices=frame_indices,
        positions_x=positions_x,
        positions_y=positions_y,
        ra=ra_list,
        dec=dec_list,
        timestamps_iso=timestamps,
        angle_deg=float(angle_deg),
        direction_deg=direction_deg,
        length_pixels=float(length),
        rate_pixels_per_sec=float(best_rate),
        rate_arcsec_per_sec=rate_arcsec,
        rate_ra_arcsec_per_sec=rate_ra,
        rate_dec_arcsec_per_sec=rate_dec,
        confirmed=True,
        best_snr=float(stacked_snr),
        best_flux=flux,
        best_calibrated_magnitudes=cal_mags,
        best_magnitude_errs=mag_errs,
    )


def _sample_de(de_map: np.ndarray, x: float, y: float) -> float:
    """Sample DE map at a position using bilinear interpolation.

    Bilinear interpolation is more robust to sub-pixel position errors
    than nearest-neighbor, since the DE map is smooth (result of FFT
    convolution with FWHM-wide kernels).
    """
    h, w = de_map.shape
    # Bilinear interpolation
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        # Fall back to nearest-neighbor at boundaries
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < h and 0 <= ix < w:
            return float(de_map[iy, ix])
        return 0.0

    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0

    val = (de_map[y0, x0] * (1 - dx) * (1 - dy)
           + de_map[y0, x1] * dx * (1 - dy)
           + de_map[y1, x0] * (1 - dx) * dy
           + de_map[y1, x1] * dx * dy)
    return float(val)


def _confirm_single_frame(frames_with_candidates: list[SiderealFrame]) -> list[CorrelatedStreak]:
    """Confirm high-SNR candidates from single-frame scenarios.

    When only 1 frame is available, multi-frame confirmation is impossible.
    Instead, use stricter per-frame criteria: high SNR, high fractional excess,
    and consistent width to mark the best candidates as confirmed.

    Args:
        frames_with_candidates: Sidereal frames carrying streak candidates.

    Returns:
        The candidates that pass the stricter single-frame criteria, as
        confirmed :class:`CorrelatedStreak` entries.
    """
    result = []
    for frame in frames_with_candidates:
        for sc in frame.streak_candidates:
            # Require high SNR for single-frame confirmation
            # (multi-frame would have ~sqrt(N) noise reduction)
            snr = sc.peak_snr if hasattr(sc, "peak_snr") else 0
            frac = sc.fractional_excess if hasattr(sc, "fractional_excess") else 0

            # Single-frame confirmation needs very high confidence
            is_confirmed = snr >= 10.0 and frac >= 0.5

            ra = [float(sc.ra)] if sc.ra is not None else []
            dec = [float(sc.dec)] if sc.dec is not None else []
            ts = [frame.timestamp.isoformat()] if frame.timestamp else []
            result.append(CorrelatedStreak(
                streak_id=str(uuid.uuid4())[:8],
                frame_indices=[frame.index],
                positions_x=[float(sc.x)],
                positions_y=[float(sc.y)],
                ra=ra, dec=dec, timestamps_iso=ts,
                angle_deg=float(sc.angle_deg),
                direction_deg=None,
                rate_pixels_per_sec=sc.rate_pixels_per_sec,
                rate_arcsec_per_sec=sc.rate_arcsec_per_sec,
                confirmed=is_confirmed,
                best_snr=float(sc.peak_snr),
                best_flux=sc.flux,
                best_calibrated_magnitudes=sc.calibrated_magnitudes,
                best_magnitude_errs=sc.magnitude_errs,
            ))
    return result


def _wrap_unconfirmed(frames_with_candidates: list[SiderealFrame]) -> list[CorrelatedStreak]:
    """Wrap single-frame candidates as unconfirmed.

    Args:
        frames_with_candidates: Sidereal frames carrying streak candidates.

    Returns:
        One unconfirmed :class:`CorrelatedStreak` per streak candidate.
    """
    result = []
    for frame in frames_with_candidates:
        for sc in frame.streak_candidates:
            ra = [float(sc.ra)] if sc.ra is not None else []
            dec = [float(sc.dec)] if sc.dec is not None else []
            ts = [frame.timestamp.isoformat()] if frame.timestamp else []
            result.append(CorrelatedStreak(
                streak_id=str(uuid.uuid4())[:8],
                frame_indices=[frame.index],
                positions_x=[float(sc.x)],
                positions_y=[float(sc.y)],
                ra=ra, dec=dec, timestamps_iso=ts,
                angle_deg=float(sc.angle_deg),
                direction_deg=None,
                rate_pixels_per_sec=sc.rate_pixels_per_sec,
                rate_arcsec_per_sec=sc.rate_arcsec_per_sec,
                confirmed=False,
                best_snr=float(sc.peak_snr),
                best_flux=sc.flux,
                best_calibrated_magnitudes=sc.calibrated_magnitudes,
                best_magnitude_errs=sc.magnitude_errs,
            ))
    return result


def _deduplicate(
    streaks: list[CorrelatedStreak], fwhm: float
) -> list[CorrelatedStreak]:
    """Remove duplicate detections of the same streak."""
    match_radius_sq = (3 * fwhm) ** 2
    streaks.sort(key=lambda c: c.best_snr, reverse=True)

    kept: list[CorrelatedStreak] = []
    for cs in streaks:
        is_dup = False
        for existing in kept:
            for fi, px, py in zip(cs.frame_indices, cs.positions_x, cs.positions_y, strict=False):
                for efi, epx, epy in zip(
                    existing.frame_indices, existing.positions_x, existing.positions_y, strict=False
                ):
                    if fi == efi:
                        dist_sq = (px - epx) ** 2 + (py - epy) ** 2
                        if dist_sq < match_radius_sq:
                            is_dup = True
                            break
                if is_dup:
                    break
            if is_dup:
                break
        if not is_dup:
            kept.append(cs)

    return kept


def _propagate_to_frames(
    cs: CorrelatedStreak,
    senpai_run: SenpaiRun,
    all_frames: list,
    fwhm: float,
) -> None:
    """Add confirmed streak detections to per-frame detection lists."""
    if not cs.confirmed or not cs.frame_indices:
        return

    angle = cs.angle_deg
    rate = cs.rate_pixels_per_sec or 0.0
    direction = cs.direction_deg
    length = cs.length_pixels or (rate * 1.0)

    ref_fidx = cs.frame_indices[0]
    ref_x = cs.positions_x[0]
    ref_y = cs.positions_y[0]

    ref_frame = next((f for f in all_frames if f.index == ref_fidx), None)
    if ref_frame is None:
        return

    dir_rad = np.radians(direction) if direction is not None else np.radians(angle)
    cos_d = np.cos(dir_rad)
    sin_d = np.sin(dir_rad)

    from senpai.engine.models.starfield import SatelliteInImage, SatelliteListImage

    for frame in all_frames:
        if frame.index == ref_fidx:
            streak_x, streak_y = ref_x, ref_y
        else:
            shift = _accumulate_shift(senpai_run, ref_fidx, frame.index)
            if shift is None:
                continue
            dx, dy = shift
            dt = 0.0
            if ref_frame.timestamp and frame.timestamp:
                dt = (frame.timestamp - ref_frame.timestamp).total_seconds()
            streak_x = ref_x + dx + rate * dt * cos_d
            streak_y = ref_y + dy + rate * dt * sin_d

        # Sky coordinates
        ra_val, dec_val = None, None
        if frame.starfield and frame.starfield.wcs:
            try:
                wcs = frame.starfield.wcs.to_astropy_wcs()
                sky = wcs.pixel_to_world(streak_x, streak_y)
                ra_val = float(sky.ra.deg)
                dec_val = float(sky.dec.deg)
            except Exception as err:
                logger.debug("WCS pixel_to_world failed for frame %s: %s", frame.index, err)

        rate_arcsec = None
        if frame.starfield and frame.starfield.wcs_metadata:
            wm = frame.starfield.wcs_metadata
            if hasattr(wm, "x_ifov_arcsec"):
                rate_arcsec = rate * wm.x_ifov_arcsec

        detection = SatelliteInImage(
            x=float(streak_x),
            y=float(streak_y),
            snr=float(cs.best_snr),
            ra=ra_val,
            dec=dec_val,
            pixel_fwhm=float(fwhm),
            flux=cs.best_flux,
            detection_type="streak",
            angle_deg=float(angle),
            length_pixels=float(length),
            rate_pixels_per_sec=float(rate),
            rate_arcsec_per_sec=rate_arcsec,
        )

        if frame.detections is None:
            img_meta = frame.starfield.image_metadata if frame.starfield else None
            if img_meta is None:
                continue
            frame.detections = SatelliteListImage(
                detections=[], image_metadata=img_meta,
            )
        frame.detections.detections.append(detection)
