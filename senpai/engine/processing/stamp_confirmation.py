"""Multi-frame streak confirmation via postage-stamp stacking.

Instead of independently measuring streaks in each frame and then trying
to correlate measurements, this module uses multi-frame data directly:

1. For each streak candidate in a reference frame, predict its position
   in all other frames (both forward and reverse directions).
2. Extract oriented postage stamps from the original images at predicted
   positions, with catalog stars masked.
3. Collapse stamps perpendicular to the streak to get 1D along-streak
   profiles (the "squish").
4. Stack profiles from all frames for sqrt(N) SNR improvement.
5. Compare forward vs reverse stack to resolve 180-degree ambiguity.
6. Reject candidates where neither direction shows SNR boost (false positives).
7. Cross-correlate profiles to refine position/rate.

With a single frame, candidates are returned as unconfirmed (no confirmation
possible).  With 3+ frames, the SNR boost is even stronger.
"""

import logging
import uuid

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import correlate

from senpai.core.config import get_config
from senpai.engine.models.senpai import CorrelatedStreak, SenpaiRun

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stamp extraction
# ---------------------------------------------------------------------------


def _extract_1d_profile(
    image: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    half_length: float,
    fwhm: float,
    star_positions: list[tuple[float, float]] | None = None,
    star_mask_radius: float | None = None,
) -> np.ndarray | None:
    """Extract a 1D along-streak profile by collapsing perpendicular to the streak.

    Samples the image in a strip aligned with the streak direction,
    sums across the perpendicular width (±1.5 FWHM), producing a 1D
    profile with improved SNR ("squish").

    Catalog star positions within the strip are masked (set to zero)
    before summation so that stars don't contaminate the profile.

    Returns a 1D array of length ~2*half_length, or None if out of bounds.
    """
    h, w = image.shape
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Along-streak sample points at 1-pixel steps
    n_along = int(np.ceil(2 * half_length)) + 1
    t_along = np.linspace(-half_length, half_length, n_along)

    # Perpendicular sample points: ±1.5 FWHM at 1-pixel steps
    perp_half = 1.5 * fwhm
    n_perp = max(3, int(np.ceil(2 * perp_half)) + 1)
    t_perp = np.linspace(-perp_half, perp_half, n_perp)

    # Build 2D grid in streak coordinates
    T_along, T_perp = np.meshgrid(t_along, t_perp)

    # Map to image coordinates
    sx = cx + T_along * cos_a - T_perp * sin_a
    sy = cy + T_along * sin_a + T_perp * cos_a

    valid = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
    if valid.sum() < n_along:
        return None

    # Sample image
    stamp = np.zeros_like(T_along)
    stamp[valid] = map_coordinates(image, [sy[valid], sx[valid]], order=1)

    # Mask catalog stars: zero out pixels within star_mask_radius
    if star_positions and star_mask_radius:
        r_sq = star_mask_radius ** 2
        for star_x, star_y in star_positions:
            dist_sq = (sx - star_x) ** 2 + (sy - star_y) ** 2
            star_mask = dist_sq < r_sq
            stamp[star_mask] = 0.0
            # Also mark these as invalid so they don't contribute to the sum
            valid[star_mask] = False

    # Collapse perpendicular: sum along the perp axis (axis=0)
    # Only count valid pixels to avoid edge bias
    valid_count = valid.astype(np.float64).sum(axis=0)
    valid_count = np.maximum(valid_count, 1)  # Avoid division by zero
    profile = stamp.sum(axis=0) / valid_count * n_perp  # Normalize to total flux

    return profile


def _profile_snr(profile: np.ndarray, fwhm: float) -> float:
    """Measure peak SNR of a 1D along-streak profile.

    Estimates signal from the peak region and noise from the wings.
    """
    n = len(profile)
    if n < 10:
        return 0.0

    # Wings: outer 25% on each side
    wing_n = max(3, n // 4)
    wings = np.concatenate([profile[:wing_n], profile[-wing_n:]])
    bg = np.median(wings)
    noise = np.std(wings)
    if noise <= 0:
        return 0.0

    # Signal: peak above background
    signal = np.max(profile) - bg

    return float(signal / noise)


# ---------------------------------------------------------------------------
# Frame shift accumulation (reused from streak_correlation)
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
# Main entry point
# ---------------------------------------------------------------------------


def confirm_streaks_via_stamps(
    senpai_run: SenpaiRun,
    de_data: dict[int, tuple[np.ndarray, float]] | None = None,
) -> list[CorrelatedStreak]:
    """Confirm streak candidates using multi-frame stamp stacking.

    For each candidate in each frame, predicts its position in all other
    frames and extracts 1D along-streak profiles.  Stacking profiles from
    multiple frames gives sqrt(N) SNR improvement.  Comparing forward vs
    reverse stacks resolves the 180-degree direction ambiguity.

    Candidates where neither direction shows SNR boost (false positives).

    Additionally, if ``de_data`` is provided (directional excess maps from
    the detection step), uses the matched-filter DE values at predicted
    positions for a more sensitive multi-frame confirmation that preserves
    the full matched-filter SNR.

    With a single frame total, candidates are returned as unconfirmed.
    """
    config = get_config()
    min_snr_boost = 1.2  # Require at least 20% SNR improvement from stacking

    # Frames with independent streak detections — used as reference frames
    frames_with_candidates = [
        f for f in senpai_run.sidereal_frames if f.streak_candidates
    ]
    for f in senpai_run.rate_track_frames:
        if f.streak_candidates:
            frames_with_candidates.append(f)
    frames_with_candidates.sort(key=lambda f: f.index)

    if not frames_with_candidates:
        return []

    # ALL sidereal frames — used for multi-frame comparison even if they
    # didn't independently detect any streaks.  This is critical: faint
    # streaks may only exceed the single-frame threshold in one frame,
    # but the directional excess at the predicted position in other frames
    # still contains matched-filter signal that helps confirmation.
    all_comparison_frames = [
        f for f in senpai_run.sidereal_frames
        if f.starfield is not None
    ]
    for f in senpai_run.rate_track_frames:
        if f.streak_candidates and f not in all_comparison_frames:
            all_comparison_frames.append(f)
    all_comparison_frames.sort(key=lambda f: f.index)

    # Get FWHM
    fwhm = 4.0
    for f in all_comparison_frames:
        sf = f.starfield
        if sf and sf.detection_metadata and sf.detection_metadata.pixel_fwhm:
            fwhm = sf.detection_metadata.pixel_fwhm
            break

    star_mask_radius = fwhm * 3

    # Single frame total: return all as unconfirmed
    if len(all_comparison_frames) < 2:
        logger.info("Single frame — returning %d unconfirmed candidates",
                     sum(len(f.streak_candidates) for f in frames_with_candidates))
        return _wrap_single_frame(frames_with_candidates)

    # Multi-frame confirmation
    correlated: list[CorrelatedStreak] = []
    n_confirmed = 0
    n_rejected = 0

    for ref_frame in frames_with_candidates:
        if not ref_frame.streak_candidates:
            continue

        # Get background-subtracted image for reference frame
        ref_image = ref_frame.frame.data.astype(np.float64)
        ref_bg = np.median(ref_image)
        ref_image = ref_image - ref_bg

        # Catalog star positions for masking
        ref_stars = []
        if ref_frame.starfield and ref_frame.starfield.catalog_stars:
            ref_stars = [
                (s.x, s.y) for s in ref_frame.starfield.catalog_stars
                if s.x is not None and s.y is not None
            ]

        # Use ALL comparison frames (not just those with candidates)
        other_frames_data = []
        for other_frame in all_comparison_frames:
            if other_frame.index == ref_frame.index:
                continue

            shift = _accumulate_shift(
                senpai_run, ref_frame.index, other_frame.index
            )
            if shift is None:
                continue

            other_image = other_frame.frame.data.astype(np.float64)
            other_bg = np.median(other_image)
            other_image = other_image - other_bg

            other_stars = []
            if other_frame.starfield and other_frame.starfield.catalog_stars:
                other_stars = [
                    (s.x, s.y) for s in other_frame.starfield.catalog_stars
                    if s.x is not None and s.y is not None
                ]

            dt = 0.0
            if ref_frame.timestamp and other_frame.timestamp:
                dt = (other_frame.timestamp - ref_frame.timestamp).total_seconds()

            other_frames_data.append({
                "frame": other_frame,
                "image": other_image,
                "stars": other_stars,
                "shift": shift,
                "dt": dt,
            })

        if not other_frames_data:
            # No valid other frames to compare against
            for sc in ref_frame.streak_candidates:
                correlated.append(_make_unconfirmed(ref_frame, sc))
            continue

        for sc in ref_frame.streak_candidates:
            result = _confirm_single_candidate(
                ref_image=ref_image,
                ref_stars=ref_stars,
                ref_frame=ref_frame,
                candidate=sc,
                other_frames_data=other_frames_data,
                fwhm=fwhm,
                star_mask_radius=star_mask_radius,
                min_snr_boost=min_snr_boost,
                de_data=de_data,
                senpai_run=senpai_run,
            )
            if result is not None:
                correlated.append(result)
                if result.confirmed:
                    n_confirmed += 1
            else:
                n_rejected += 1

    # --- Deduplicate confirmed streaks ---
    deduped = _deduplicate_confirmed(correlated, fwhm)
    n_deduped = n_confirmed - sum(1 for c in deduped if c.confirmed)

    # --- Replace per-frame streak_candidates with only confirmed detections ---
    for f in senpai_run.sidereal_frames:
        f.streak_candidates = []
    for f in senpai_run.rate_track_frames:
        f.streak_candidates = []

    for cs in deduped:
        if not cs.confirmed:
            continue
        _propagate_to_frame_candidates(
            cs, senpai_run, all_comparison_frames, fwhm,
        )

    logger.info(
        "Stamp confirmation: %d confirmed (%d deduped), %d rejected, %d total",
        sum(1 for c in deduped if c.confirmed), n_deduped,
        n_rejected, len(deduped),
    )
    return deduped


def _confirm_single_candidate(
    ref_image: np.ndarray,
    ref_stars: list[tuple[float, float]],
    ref_frame,
    candidate,
    other_frames_data: list[dict],
    fwhm: float,
    star_mask_radius: float,
    min_snr_boost: float,
    de_data: dict[int, tuple[np.ndarray, float]] | None = None,
    senpai_run: SenpaiRun | None = None,
) -> CorrelatedStreak | None:
    """Confirm a single streak candidate using stamp stacking across frames.

    Uses two complementary confirmation approaches:
    1. Profile cross-correlation (CC): extracts 1D profiles from raw images
       at predicted positions and cross-correlates for shape matching.
    2. DE multi-frame SNR (when de_data provided): checks directional excess
       values at predicted positions in other frames' matched-filter maps.
       This preserves the full matched-filter SNR and catches faint streaks
       that the profile CC misses.

    Returns a CorrelatedStreak (confirmed or unconfirmed), or None if the
    candidate should be rejected.
    """
    angle_deg = candidate.angle_deg
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rate = candidate.rate_pixels_per_sec or 0.0

    # Half-length for stamp extraction: generous margin to tolerate rate
    # errors.  The measured rate can be off by 50%+, so the predicted
    # position in the other frame may be tens of pixels from truth.
    # Use a large stamp so cross-correlation can find the true offset.
    estimated_motion = abs(rate * max(abs(ofd["dt"]) for ofd in other_frames_data)) if other_frames_data else 0
    half_length = max(
        candidate.length_pixels,
        estimated_motion * 0.5,  # Accommodate ±50% rate error
        10 * fwhm,
    ) / 2 + 5 * fwhm

    # Reference profile
    ref_profile = _extract_1d_profile(
        ref_image, candidate.x, candidate.y,
        angle_deg, half_length, fwhm,
        ref_stars, star_mask_radius,
    )
    if ref_profile is None:
        return _make_unconfirmed(ref_frame, candidate)

    ref_snr = _profile_snr(ref_profile, fwhm)

    # Collect profiles from other frames for both directions
    fwd_profiles = [ref_profile]
    rev_profiles = [ref_profile]
    fwd_frames = [ref_frame]
    rev_frames = [ref_frame]
    fwd_offsets = [(0.0, 0.0)]  # (along-streak offset, peak value)
    rev_offsets = [(0.0, 0.0)]

    for ofd in other_frames_data:
        dx, dy = ofd["shift"]
        dt = ofd["dt"]

        # Predicted center in other frame
        base_x = candidate.x + dx
        base_y = candidate.y + dy

        # Motion offset (forward and reverse)
        if rate > 0 and dt != 0:
            motion_dx = rate * dt * cos_a
            motion_dy = rate * dt * sin_a
        else:
            motion_dx = 0.0
            motion_dy = 0.0

        # Skip if the predicted motion is too small — the stamps would
        # overlap almost entirely and any persistent feature (star residual,
        # hot pixel) would trivially correlate.  Require at least 2*FWHM
        # of motion for the comparison to be meaningful.
        total_motion = np.sqrt(
            (motion_dx + dx) ** 2 + (motion_dy + dy) ** 2
        )
        if total_motion < 2 * fwhm:
            continue

        # Forward prediction
        fwd_x = base_x + motion_dx
        fwd_y = base_y + motion_dy

        fwd_prof = _extract_1d_profile(
            ofd["image"], fwd_x, fwd_y,
            angle_deg, half_length, fwhm,
            ofd["stars"], star_mask_radius,
        )
        if fwd_prof is not None and len(fwd_prof) == len(ref_profile):
            offset, cc_peak = _find_profile_offset(ref_profile, fwd_prof)
            fwd_profiles.append(fwd_prof)
            fwd_frames.append(ofd["frame"])
            fwd_offsets.append((offset, cc_peak))

        # Reverse prediction
        rev_x = base_x - motion_dx
        rev_y = base_y - motion_dy

        rev_prof = _extract_1d_profile(
            ofd["image"], rev_x, rev_y,
            angle_deg, half_length, fwhm,
            ofd["stars"], star_mask_radius,
        )
        if rev_prof is not None and len(rev_prof) == len(ref_profile):
            offset, cc_peak = _find_profile_offset(ref_profile, rev_prof)
            rev_profiles.append(rev_prof)
            rev_frames.append(ofd["frame"])
            rev_offsets.append((offset, cc_peak))

    # Use cross-correlation peak as confirmation metric.
    # Normalized CC of two profiles containing the same streak shape
    # gives a high peak (>0.3).  For unrelated noise, the peak is low.
    # Average CC peaks across all frame pairs for robustness.
    # Thresholds for confirmation.
    # The boxcar scan optimizes over position AND length, inflating SNR for noise.
    # Require both a high CC peak (shape correlation) and a minimum profile SNR.
    min_cc_peak = 0.50
    min_confirmed_snr = 8.0

    fwd_cc_peaks = [p for _, p in fwd_offsets[1:]]  # Skip reference self-match
    rev_cc_peaks = [p for _, p in rev_offsets[1:]]

    fwd_cc = float(np.mean(fwd_cc_peaks)) if fwd_cc_peaks else 0.0
    rev_cc = float(np.mean(rev_cc_peaks)) if rev_cc_peaks else 0.0

    best_cc = max(fwd_cc, rev_cc)
    best_snr = max(
        _profile_snr(np.mean(fwd_profiles, axis=0), fwhm) if len(fwd_profiles) > 1 else 0.0,
        _profile_snr(np.mean(rev_profiles, axis=0), fwhm) if len(rev_profiles) > 1 else 0.0,
    )

    # --- Confirmation decision ---
    # Two complementary paths:
    # 1. CC-based: profile cross-correlation confirms shape match
    # 2. DE-based: directional excess at predicted positions confirms
    #    matched-filter signal in other frames (more sensitive for faint streaks)
    cc_confirmed = best_cc > min_cc_peak

    de_confirmed = False
    de_direction = None
    de_snr_fwd = 0.0
    de_snr_rev = 0.0
    if de_data is not None and senpai_run is not None and not cc_confirmed:
        # Gate: only try DE-based confirmation for candidates with strong
        # single-frame matched-filter detection.  The reference-frame DE
        # at the candidate position must exceed a high threshold.
        # This prevents marginal noise peaks (5-7σ) from being confirmed
        # by coincidental DE values in other frames.
        ref_de_map, ref_noise = de_data.get(ref_frame.index, (None, None))
        ref_de_snr = 0.0
        if ref_de_map is not None and ref_noise and ref_noise > 0:
            ref_de_val = _sample_de_at_position(ref_de_map, candidate.x, candidate.y)
            ref_de_snr = ref_de_val / ref_noise

        min_ref_de_snr = 8.0  # Strong single-frame detection required

        # Reject candidates near catalog stars: bright star halos create
        # high DE at the mask boundary that survives single-frame detection
        # and can produce spurious multi-frame correlation.
        near_catalog_star = False
        if ref_frame.starfield and ref_frame.starfield.catalog_stars:
            # Stars are now PSF-subtracted before the filter bank, so the
            # exclusion zone only needs to cover subtraction residuals.
            # Use 3*FWHM (matching the reduced star mask radius).
            proximity_radius_sq = (fwhm * 3) ** 2  # ~3*FWHM ≈ 8 pixels
            for s in ref_frame.starfield.catalog_stars:
                if s.x is not None and s.y is not None:
                    dx = candidate.x - s.x
                    dy = candidate.y - s.y
                    if dx * dx + dy * dy < proximity_radius_sq:
                        near_catalog_star = True
                        break

        if ref_de_snr >= min_ref_de_snr and not near_catalog_star:
            de_snr_fwd, de_snr_rev = _de_multiframe_snr(
                candidate=candidate,
                ref_frame=ref_frame,
                other_frames_data=other_frames_data,
                de_data=de_data,
                senpai_run=senpai_run,
                fwhm=fwhm,
            )
            # Other-frame DE SNR > 3.5: the matched-filter response at
            # predicted positions in other frames shows positive signal.
            # 3.5σ balances sensitivity (real faint streaks ~4-5σ) against
            # false positives from DE map systematics (star halo residuals,
            # background structure).
            min_de_other_snr = 3.5
            de_other = max(de_snr_fwd, de_snr_rev)
            if de_other > min_de_other_snr:
                de_confirmed = True
                de_direction = "forward" if de_snr_fwd >= de_snr_rev else "reverse"
                logger.info(
                    "DE confirmation for streak at (%.0f,%.0f): "
                    "ref_de=%.1f, other_fwd=%.2f, other_rev=%.2f",
                    candidate.x, candidate.y, ref_de_snr,
                    de_snr_fwd, de_snr_rev,
                )

    confirmed = cc_confirmed or de_confirmed

    if not confirmed:
        if ref_snr > 5.0:
            return _make_unconfirmed(ref_frame, candidate)
        return None

    # Resolve direction and refine streak from stacked profile
    if confirmed and de_confirmed and not cc_confirmed:
        # DE-based confirmation: use DE direction
        if de_direction == "forward":
            direction = "forward"
            direction_deg = float(angle_deg % 360)
        else:
            direction = "reverse"
            direction_deg = float((angle_deg + 180) % 360)
        matched_frames = fwd_frames if direction == "forward" else rev_frames
        matched_profiles = fwd_profiles if direction == "forward" else rev_profiles
        matched_offsets = fwd_offsets if direction == "forward" else rev_offsets
    elif fwd_cc >= rev_cc:
        direction = "forward"
        direction_deg = float(angle_deg % 360)
        matched_frames = fwd_frames
        matched_profiles = fwd_profiles
        matched_offsets = fwd_offsets
    else:
        direction = "reverse"
        direction_deg = float((angle_deg + 180) % 360)
        matched_frames = rev_frames
        matched_profiles = rev_profiles
        matched_offsets = rev_offsets

    # --- Refine streak measurements ---
    exposure_time = None
    if ref_frame.frame_metadata and ref_frame.frame_metadata.exposure_time_seconds:
        exposure_time = ref_frame.frame_metadata.exposure_time_seconds

    if de_confirmed and not cc_confirmed:
        # DE-confirmed faint streaks: the 1D profile boxcar scan is
        # unreliable (tends to find the brightest sub-section, giving
        # too-short lengths).  Use the candidate's original measurements
        # from the single-frame directional filter trace, which are
        # more robust for faint streaks.
        center_offset = 0.0
        refined_length = candidate.length_pixels
        refined_rate = rate  # Original rate from single-frame detection
        measured_snr = candidate.peak_snr
        best_snr = measured_snr
        logger.debug(
            "DE-confirmed streak at (%.0f,%.0f): using candidate len=%.1f, rate=%.1f",
            candidate.x, candidate.y, refined_length, refined_rate,
        )
    else:
        # CC-confirmed: refine from the boxcar scan on the reference profile
        center_offset, refined_length, measured_snr = _measure_streak_from_profile(
            ref_profile, fwhm
        )
        if exposure_time and exposure_time > 0 and refined_length > 0:
            refined_rate = refined_length / exposure_time
        else:
            refined_rate = rate

        best_snr = measured_snr
        logger.debug(
            "CC-confirmed streak at (%.0f,%.0f): refined len=%.1f (was %.1f), "
            "rate=%.1f (was %.1f), snr=%.1f, cc=%.3f",
            candidate.x, candidate.y, refined_length, candidate.length_pixels,
            refined_rate, rate, measured_snr, best_cc,
        )

        # For CC-confirmed, require minimum boxcar SNR.
        if best_snr < min_confirmed_snr:
            if ref_snr > 5.0:
                return _make_unconfirmed(ref_frame, candidate)
            return None

    # --- Refine positions using DE peak search + RA/Dec velocity ---
    # Instead of blindly predicting positions from single-frame angle+rate,
    # find the actual DE peak near the predicted position in each other
    # frame, then compute the RA/Dec velocity from the refined positions.
    refined_cx = candidate.x + center_offset * cos_a
    refined_cy = candidate.y + center_offset * sin_a

    # All frames sorted by index (reference + others)
    all_frame_positions = {ref_frame.index: (float(refined_cx), float(refined_cy))}

    sign = 1.0 if direction == "forward" else -1.0
    for ofd in other_frames_data:
        dx, dy = ofd["shift"]
        dt = ofd["dt"]
        frame_idx = ofd["frame"].index

        # Initial prediction from angle + rate
        motion_dx = sign * refined_rate * dt * cos_a if refined_rate > 0 and dt != 0 else 0.0
        motion_dy = sign * refined_rate * dt * sin_a if refined_rate > 0 and dt != 0 else 0.0
        pred_x = refined_cx + dx + motion_dx
        pred_y = refined_cy + dy + motion_dy

        # Refine position by searching along the streak direction in the
        # DE map.  Instead of a naive 2D max (which finds off-axis noise),
        # extract a 1D profile along the streak angle and find its peak.
        if de_data is not None and frame_idx in de_data:
            de_map, _ = de_data[frame_idx]
            refined_pos = _find_de_peak_along_streak(
                de_map, pred_x, pred_y, angle_deg, fwhm,
                search_half=fwhm * 5,
            )
            if refined_pos is not None:
                pred_x, pred_y = refined_pos

        all_frame_positions[frame_idx] = (float(pred_x), float(pred_y))

    # Order by frame index
    matched_frames_all = [ref_frame] + [ofd["frame"] for ofd in other_frames_data]
    matched_frames_all.sort(key=lambda f: f.index)
    frame_indices = [f.index for f in matched_frames_all if f.index in all_frame_positions]
    positions_x = [all_frame_positions[fi][0] for fi in frame_indices]
    positions_y = [all_frame_positions[fi][1] for fi in frame_indices]

    # Convert to RA/Dec and compute sky-plane velocity
    ra_list, dec_list = [], []
    timestamps = []
    wcs_obj = None
    for fi in frame_indices:
        frame_obj = next(f for f in matched_frames_all if f.index == fi)
        if frame_obj.timestamp:
            timestamps.append(frame_obj.timestamp.isoformat())
        if frame_obj.starfield and frame_obj.starfield.wcs:
            try:
                wcs_obj = frame_obj.starfield.wcs.to_astropy_wcs()
                px, py = all_frame_positions[fi]
                sky = wcs_obj.pixel_to_world(px, py)
                ra_list.append(float(sky.ra.deg))
                dec_list.append(float(sky.dec.deg))
            except Exception:
                pass

    # Compute RA/Dec rates from multi-frame sky positions
    rate_ra = None
    rate_dec = None
    rate_arcsec = None
    if len(ra_list) >= 2 and len(frame_indices) >= 2:
        try:
            t0 = next(f for f in matched_frames_all if f.index == frame_indices[0]).timestamp
            t1 = next(f for f in matched_frames_all if f.index == frame_indices[-1]).timestamp
            dt_total = (t1 - t0).total_seconds()
            if dt_total > 0:
                dra = (ra_list[-1] - ra_list[0]) * 3600 * np.cos(np.radians(dec_list[0]))
                ddec = (dec_list[-1] - dec_list[0]) * 3600
                rate_ra = float(dra / dt_total)
                rate_dec = float(ddec / dt_total)
                rate_arcsec = float(np.sqrt(dra ** 2 + ddec ** 2) / dt_total)

                # Recompute pixel-space rate and direction from RA/Dec rates
                # by projecting the sky velocity through the reference WCS
                if wcs_obj is not None:
                    ref_px, ref_py = all_frame_positions[frame_indices[0]]
                    sky0 = wcs_obj.pixel_to_world(ref_px, ref_py)
                    # Offset by the velocity * 1 second in RA/Dec
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    sky1 = SkyCoord(
                        ra=sky0.ra + (rate_ra / 3600 / np.cos(np.radians(dec_list[0]))) * u.deg,
                        dec=sky0.dec + (rate_dec / 3600) * u.deg,
                    )
                    px1 = wcs_obj.all_world2pix([[sky1.ra.deg, sky1.dec.deg]], 0)
                    dx_pix = float(px1[0][0]) - ref_px
                    dy_pix = float(px1[0][1]) - ref_py
                    refined_rate = float(np.sqrt(dx_pix ** 2 + dy_pix ** 2))
                    # Pixel-space angle of the velocity vector
                    velocity_angle = float(np.degrees(np.arctan2(dy_pix, dx_pix))) % 180
                    direction_deg = float(np.degrees(np.arctan2(dy_pix, dx_pix))) % 360

                    # Update the streak angle and length from the velocity
                    angle_deg = velocity_angle
                    if exposure_time and exposure_time > 0:
                        refined_length = refined_rate * exposure_time

                    logger.info(
                        "Multi-frame refined: rate=%.1f px/s (%.1f\"/s), "
                        "angle=%.1f°, direction=%.1f°, length=%.1f px",
                        refined_rate, rate_arcsec, velocity_angle,
                        direction_deg, refined_length,
                    )
        except Exception as e:
            logger.debug("RA/Dec rate computation failed: %s", e)

    if rate_arcsec is None:
        sf = ref_frame.starfield
        if sf and sf.wcs_metadata and hasattr(sf.wcs_metadata, "x_ifov_arcsec"):
            rate_arcsec = refined_rate * sf.wcs_metadata.x_ifov_arcsec

    # Measure photometry on the reference frame candidate
    cal_mags = None
    mag_errs = None
    flux = candidate.flux
    if ref_frame.photometry_summary and confirmed:
        zp = ref_frame.photometry_summary.get("zero_point")
        zp_err = ref_frame.photometry_summary.get("zero_point_err")
        multiband_raw = ref_frame.photometry_summary.get("multiband_calibration")
        # Deserialize multiband calibration if it's a dict
        multiband = None
        if multiband_raw is not None:
            try:
                from senpai.engine.photometry.color_terms import MultiBandCalibration
                if isinstance(multiband_raw, dict):
                    multiband = MultiBandCalibration.model_validate(multiband_raw)
                else:
                    multiband = multiband_raw
            except Exception:
                pass
        if zp is not None:
            from senpai.engine.detection.streak.sidereal_streak import (
                measure_streak_candidate_photometry,
            )
            from senpai.core.config import get_config

            config = get_config()
            obs_filter = ref_frame.frame_metadata.observation_filter if ref_frame.frame_metadata else None

            # Build a temporary candidate with refined measurements for photometry
            from senpai.engine.detection.streak.sidereal_streak import StreakCandidate
            phot_candidate = StreakCandidate(
                x=float(refined_cx), y=float(refined_cy),
                angle_deg=float(angle_deg),
                length_pixels=float(refined_length),
                width_pixels=float(fwhm),
                peak_snr=float(best_snr),
                directional_excess=0.0, fractional_excess=0.0,
            )
            try:
                measure_streak_candidate_photometry(
                    ref_frame.frame, [phot_candidate],
                    zero_point=zp, zero_point_err=zp_err,
                    exposure_time=exposure_time,
                    fwhm=fwhm, gain=config.photometry.gain,
                    multiband_calibration=multiband,
                    observation_filter=obs_filter,
                )
                flux = phot_candidate.flux
                cal_mags = phot_candidate.calibrated_magnitudes
                mag_errs = phot_candidate.magnitude_errs
            except Exception as e:
                logger.debug("Streak photometry failed: %s", e)

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
        length_pixels=float(refined_length),
        rate_pixels_per_sec=refined_rate,
        rate_arcsec_per_sec=rate_arcsec,
        rate_ra_arcsec_per_sec=rate_ra,
        rate_dec_arcsec_per_sec=rate_dec,
        confirmed=confirmed,
        best_snr=float(best_snr),
        best_flux=flux,
        best_calibrated_magnitudes=cal_mags,
        best_magnitude_errs=mag_errs,
    )


def _measure_streak_from_profile(
    profile: np.ndarray,
    fwhm: float,
) -> tuple[float, float, float]:
    """Measure streak length and center offset from a 1D along-streak profile.

    Scans boxcar widths from min_length to max_length and finds the
    (position, length) that maximizes SNR.  This is equivalent to
    fitting a flat-topped signal of unknown length — the optimal boxcar
    length equals the true streak length because:
    - Too short: misses signal  (SNR ~ sqrt(L))
    - Too long: dilutes with noise  (SNR ~ L_true / sqrt(L))
    - Just right: captures all signal  (SNR = L_true × signal / sqrt(L_true × noise²))

    Returns (center_offset, length, peak_snr) where center_offset is
    relative to the profile center (positive = shifted right).
    """
    n = len(profile)
    if n < 5:
        return 0.0, 0.0, 0.0

    # Background and noise from wings (outer 25% each side)
    wing_n = max(3, n // 4)
    wings = np.concatenate([profile[:wing_n], profile[-wing_n:]])
    bg = np.median(wings)
    noise = np.std(wings)
    if noise <= 0:
        return 0.0, 0.0, 0.0

    # Background-subtracted profile
    p = profile - bg

    # Cumulative sum for O(1) boxcar evaluation
    cumsum = np.concatenate([[0.0], np.cumsum(p)])

    # Scan boxcar lengths from ~2*FWHM to ~80% of profile length
    min_len = max(3, int(2 * fwhm))
    max_len = min(n - 2, int(0.8 * n))

    best_snr = 0.0
    best_pos = n // 2
    best_len = min_len

    for L in range(min_len, max_len + 1):
        # Slide the boxcar across all valid positions
        # signal[i] = mean of p[i:i+L] = (cumsum[i+L] - cumsum[i]) / L
        # snr[i] = signal[i] * sqrt(L) / noise
        #        = (cumsum[i+L] - cumsum[i]) / (noise * sqrt(L))
        noise_term = noise * np.sqrt(L)
        for i in range(0, n - L + 1):
            signal = cumsum[i + L] - cumsum[i]
            snr = signal / noise_term
            if snr > best_snr:
                best_snr = snr
                best_pos = i
                best_len = L

    center_offset = (best_pos + best_len / 2) - n / 2

    return float(center_offset), float(best_len), float(best_snr)


def _find_profile_offset(
    ref_profile: np.ndarray,
    other_profile: np.ndarray,
) -> tuple[float, float]:
    """Find the along-streak offset between two 1D profiles via cross-correlation.

    Returns (offset_pixels, peak_correlation_value).
    """
    # Normalize profiles (zero-mean, unit-variance)
    ref_norm = ref_profile - np.mean(ref_profile)
    ref_std = np.std(ref_norm)
    if ref_std > 0:
        ref_norm = ref_norm / ref_std

    other_norm = other_profile - np.mean(other_profile)
    other_std = np.std(other_norm)
    if other_std > 0:
        other_norm = other_norm / other_std

    # Cross-correlate
    cc = correlate(ref_norm, other_norm, mode="full")
    cc = cc / len(ref_profile)  # Normalize

    # Find peak within a reasonable search range (±20% of profile length)
    center = len(cc) // 2
    search_half = max(5, len(ref_profile) // 5)
    lo = max(0, center - search_half)
    hi = min(len(cc), center + search_half + 1)

    peak_idx = lo + np.argmax(cc[lo:hi])
    offset = float(peak_idx - center)
    peak_val = float(cc[peak_idx])

    return offset, peak_val


# ---------------------------------------------------------------------------
# DE-based multi-frame confirmation
# ---------------------------------------------------------------------------


def _de_multiframe_snr(
    candidate,
    ref_frame,
    other_frames_data: list[dict],
    de_data: dict[int, tuple[np.ndarray, float]],
    senpai_run: SenpaiRun,
    fwhm: float = 4.0,
) -> tuple[float, float]:
    """Compute multi-frame SNR from directional excess maps at predicted positions.

    For each other frame, samples the directional excess map at the position
    predicted by the candidate's rate and angle (both forward and reverse
    directions).  Returns the combined SNR of the OTHER frames only
    (excluding the reference frame where the candidate was detected).

    The DE map already integrates signal perpendicular to the streak via the
    matched filter kernel, so a single sample at the predicted centroid
    captures the full matched-filter response.

    Returns:
        ``(fwd_other_snr, rev_other_snr)`` — combined SNR of DE values
        in other frames for forward and reverse directions.
    """
    rate = candidate.rate_pixels_per_sec or 0.0
    angle_rad = np.radians(candidate.angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Minimum motion to ensure DE values in other frames are spatially
    # independent from the reference candidate.  The directional filter
    # kernel creates correlations over ~5*FWHM (the kernel length).
    # Require the predicted position to be at least this far from the
    # frame-shift-only position to avoid spurious correlation.
    min_motion_pixels = 5 * fwhm

    fwd_de_vals: list[float] = []
    rev_de_vals: list[float] = []
    noise_vals: list[float] = []

    for ofd in other_frames_data:
        frame_idx = ofd["frame"].index
        if frame_idx not in de_data:
            continue

        de_map, noise_std = de_data[frame_idx]
        if noise_std <= 0:
            continue

        dx, dy = ofd["shift"]
        dt = ofd["dt"]

        # Streak motion prediction
        if rate > 0 and dt != 0:
            motion_dx = rate * dt * cos_a
            motion_dy = rate * dt * sin_a
        else:
            motion_dx = 0.0
            motion_dy = 0.0

        # Check that the total motion (frame shift + streak motion) is large
        # enough that the DE value at the predicted position is independent
        # of the reference candidate's DE.  The directional filter creates
        # spatial correlations over ~kernel_length pixels.
        fwd_total_motion = np.sqrt((dx + motion_dx) ** 2 + (dy + motion_dy) ** 2)
        rev_total_motion = np.sqrt((dx - motion_dx) ** 2 + (dy - motion_dy) ** 2)

        if max(fwd_total_motion, rev_total_motion) < min_motion_pixels:
            continue

        # Star proximity check: if the predicted position is near a
        # catalog star, the DE may be contaminated by subtraction residuals.
        # With PSF subtraction, use a smaller proximity radius.
        other_stars = ofd.get("stars", [])
        star_proximity_sq = (3.0 * fwhm) ** 2  # Match the reduced star mask radius

        # Forward predicted position
        fwd_val = 0.0
        if fwd_total_motion >= min_motion_pixels:
            fwd_x = candidate.x + dx + motion_dx
            fwd_y = candidate.y + dy + motion_dy
            near_star = any(
                (fwd_x - sx) ** 2 + (fwd_y - sy) ** 2 < star_proximity_sq
                for sx, sy in other_stars
            )
            if not near_star:
                fwd_val = _sample_de_at_position(de_map, fwd_x, fwd_y)
        fwd_de_vals.append(fwd_val)

        # Reverse predicted position
        rev_val = 0.0
        if rev_total_motion >= min_motion_pixels:
            rev_x = candidate.x + dx - motion_dx
            rev_y = candidate.y + dy - motion_dy
            near_star = any(
                (rev_x - sx) ** 2 + (rev_y - sy) ** 2 < star_proximity_sq
                for sx, sy in other_stars
            )
            if not near_star:
                rev_val = _sample_de_at_position(de_map, rev_x, rev_y)
        rev_de_vals.append(rev_val)

        noise_vals.append(noise_std)

    if not noise_vals:
        return 0.0, 0.0

    # Combined SNR of other frames only (not including reference).
    # Under null hypothesis (no streak), DE values at predicted positions
    # are ~N(0, noise_std).  Combined SNR = sum(DE_i) / sqrt(sum(noise_i²)).
    noise_combined = float(np.sqrt(np.sum(np.array(noise_vals) ** 2)))
    if noise_combined <= 0:
        return 0.0, 0.0

    fwd_snr = float(np.sum(fwd_de_vals)) / noise_combined
    rev_snr = float(np.sum(rev_de_vals)) / noise_combined

    return fwd_snr, rev_snr


def _sample_de_at_position(
    de_map: np.ndarray,
    cx: float,
    cy: float,
) -> float:
    """Sample the DE map at the predicted streak centroid.

    Uses single-point sampling at the nearest pixel.  The DE map already
    has spatial extent ~FWHM per streak, so single-point sampling captures
    the signal if the position prediction is accurate to within ~1 FWHM.
    Taking max over a search region inflates the null distribution
    (expected max of N noise samples >> expected single sample), causing
    false positives.
    """
    h, w = de_map.shape
    ix, iy = int(round(cx)), int(round(cy))
    if 0 <= iy < h and 0 <= ix < w:
        return float(de_map[iy, ix])
    return 0.0


def _find_de_peak_along_streak(
    de_map: np.ndarray,
    pred_x: float,
    pred_y: float,
    angle_deg: float,
    fwhm: float,
    search_half: float = 15.0,
) -> tuple[float, float] | None:
    """Find the DE peak along the streak direction near a predicted position.

    Extracts a 1D DE profile along the streak angle from the DE map,
    then finds the peak.  This is more robust than a 2D max search
    because it constrains the search to the streak direction, avoiding
    off-axis noise peaks and star residuals.

    Returns (x, y) of the peak, or None if out of bounds.
    """
    h, w = de_map.shape
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Sample along streak direction at 1-pixel steps
    t_values = np.arange(-search_half, search_half + 0.5, 1.0)
    sx = pred_x + t_values * cos_a
    sy = pred_y + t_values * sin_a

    valid = (sx >= 1) & (sx < w - 1) & (sy >= 1) & (sy < h - 1)
    if valid.sum() < 5:
        return None

    from scipy.ndimage import map_coordinates
    profile = map_coordinates(de_map, [sy[valid], sx[valid]], order=1)
    t_valid = t_values[valid]

    # Use flux-weighted centroid instead of peak position.
    # For faint streaks, the DE profile has a flat plateau where noise
    # determines the peak location.  The centroid averages over the
    # plateau and is much more stable.
    median_val = np.median(profile)
    above_median = profile > median_val
    if not above_median.any():
        return None

    # Weight by (DE - median) for pixels above median
    weights = np.maximum(profile - median_val, 0)
    total_w = weights.sum()
    if total_w <= 0:
        return None

    centroid_t = np.sum(t_valid * weights) / total_w
    centroid_x = pred_x + centroid_t * cos_a
    centroid_y = pred_y + centroid_t * sin_a
    return float(centroid_x), float(centroid_y)


def _find_de_peak_near(
    de_map: np.ndarray,
    pred_x: float,
    pred_y: float,
    search_radius: float,
) -> tuple[float | None, float | None]:
    """Find the DE peak within search_radius of a predicted position.

    Returns the (x, y) of the peak, or (None, None) if the predicted
    position is out of bounds.
    """
    h, w = de_map.shape
    ix, iy = int(round(pred_x)), int(round(pred_y))
    ir = int(np.ceil(search_radius))

    y_lo = max(0, iy - ir)
    y_hi = min(h, iy + ir + 1)
    x_lo = max(0, ix - ir)
    x_hi = min(w, ix + ir + 1)

    if y_hi <= y_lo or x_hi <= x_lo:
        return None, None

    patch = de_map[y_lo:y_hi, x_lo:x_hi]
    by, bx = np.unravel_index(patch.argmax(), patch.shape)
    peak_x = float(x_lo + bx)
    peak_y = float(y_lo + by)

    # Only use the peak if it's within the search radius
    if (peak_x - pred_x) ** 2 + (peak_y - pred_y) ** 2 > search_radius ** 2:
        return float(pred_x), float(pred_y)  # Fall back to prediction

    return peak_x, peak_y


def _make_unconfirmed(frame, candidate) -> CorrelatedStreak:
    """Wrap a single-frame candidate as an unconfirmed CorrelatedStreak."""
    ra = [float(candidate.ra)] if candidate.ra is not None else []
    dec = [float(candidate.dec)] if candidate.dec is not None else []
    ts = [frame.timestamp.isoformat()] if frame.timestamp else []

    return CorrelatedStreak(
        streak_id=str(uuid.uuid4())[:8],
        frame_indices=[frame.index],
        positions_x=[float(candidate.x)],
        positions_y=[float(candidate.y)],
        ra=ra,
        dec=dec,
        timestamps_iso=ts,
        angle_deg=float(candidate.angle_deg),
        direction_deg=None,
        rate_pixels_per_sec=candidate.rate_pixels_per_sec,
        rate_arcsec_per_sec=candidate.rate_arcsec_per_sec,
        confirmed=False,
        best_snr=float(candidate.peak_snr),
        best_flux=candidate.flux,
        best_calibrated_magnitudes=candidate.calibrated_magnitudes,
        best_magnitude_errs=candidate.magnitude_errs,
    )


def _wrap_single_frame(frames_with_streaks) -> list[CorrelatedStreak]:
    """Wrap all single-frame candidates as unconfirmed."""
    result = []
    for frame in frames_with_streaks:
        for sc in frame.streak_candidates:
            result.append(_make_unconfirmed(frame, sc))
    return result


def _deduplicate_confirmed(
    correlated: list[CorrelatedStreak],
    fwhm: float,
) -> list[CorrelatedStreak]:
    """Remove duplicate confirmed detections of the same target.

    Two confirmed streaks are duplicates if they share a frame and their
    positions in that frame are within 3*FWHM.  Keeps the one with higher SNR.
    """
    match_radius_sq = (3 * fwhm) ** 2

    confirmed = [c for c in correlated if c.confirmed]
    unconfirmed = [c for c in correlated if not c.confirmed]

    # Sort confirmed by SNR descending
    confirmed.sort(key=lambda c: c.best_snr, reverse=True)

    kept: list[CorrelatedStreak] = []
    for cs in confirmed:
        is_dup = False
        for existing in kept:
            # Check if they share a frame with nearby positions
            for fi, px, py in zip(
                cs.frame_indices, cs.positions_x, cs.positions_y
            ):
                for efi, epx, epy in zip(
                    existing.frame_indices, existing.positions_x, existing.positions_y
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

    n_removed = len(confirmed) - len(kept)
    if n_removed > 0:
        logger.debug("Deduplicated %d confirmed streaks", n_removed)

    return kept + unconfirmed


def _propagate_to_frame_candidates(
    cs: CorrelatedStreak,
    senpai_run: SenpaiRun,
    frames_with_streaks: list,
    fwhm: float,
) -> None:
    """Propagate confirmed streak to per-frame detections with photometry.

    Uses the canonical measurement (length, angle, rate) from the confirmed
    CorrelatedStreak.  For each frame, computes the predicted position and
    measures photometry from the actual image.  Results go into
    ``frame.detections`` as ``SatelliteInImage`` entries with
    ``detection_type="streak"``.
    """
    if not cs.confirmed or not cs.frame_indices:
        return

    angle = cs.angle_deg
    rate = cs.rate_pixels_per_sec or 0.0
    direction = cs.direction_deg
    length = rate * 3.0 if rate > 0 else 0  # TODO: store refined_length on CorrelatedStreak

    # Use the first frame's position as the reference
    ref_fidx = cs.frame_indices[0]
    ref_x = cs.positions_x[0]
    ref_y = cs.positions_y[0]

    # Find the reference frame's timestamp
    ref_frame = None
    for f in frames_with_streaks:
        if f.index == ref_fidx:
            ref_frame = f
            break
    if ref_frame is None:
        return

    # Get exposure time for computing length from rate
    exposure_time = None
    if ref_frame.frame_metadata and ref_frame.frame_metadata.exposure_time_seconds:
        exposure_time = ref_frame.frame_metadata.exposure_time_seconds
    if exposure_time and rate > 0:
        length = rate * exposure_time

    # Compute direction vector
    if direction is not None:
        dir_rad = np.radians(direction)
    else:
        dir_rad = np.radians(angle)
    cos_d = np.cos(dir_rad)
    sin_d = np.sin(dir_rad)

    # For each frame, compute the streak position and add to frame.detections
    # with photometry measured from the actual frame image.
    from senpai.engine.models.starfield import SatelliteInImage, SatelliteListImage

    for frame in frames_with_streaks:
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
            except Exception:
                pass

        rate_arcsec = None
        if frame.starfield and frame.starfield.wcs_metadata:
            wm = frame.starfield.wcs_metadata
            if hasattr(wm, "x_ifov_arcsec"):
                rate_arcsec = rate * wm.x_ifov_arcsec

        # Measure photometry on this frame's actual image.
        # Fall back to the reference frame's photometry calibration if this
        # frame doesn't have its own (common for shifted-WCS frames).
        flux_val, flux_err_val = None, None
        inst_mag, cal_mags, mag_errs, obs_filter = None, None, None, None

        # Use this frame's photometry if it has a zero point, else fall back
        # to the reference frame's calibration (same telescope/conditions).
        phot_summary = frame.photometry_summary
        if not phot_summary or phot_summary.get("zero_point") is None:
            phot_summary = ref_frame.photometry_summary
        if phot_summary:
            zp = phot_summary.get("zero_point")
            zp_err = phot_summary.get("zero_point_err")
            if zp is not None:
                from senpai.engine.detection.streak.sidereal_streak import (
                    StreakCandidate as SC,
                    measure_streak_candidate_photometry,
                )
                from senpai.core.config import get_config

                config = get_config()
                obs_filter = frame.frame_metadata.observation_filter if frame.frame_metadata else None
                exp_time = frame.frame_metadata.exposure_time_seconds if frame.frame_metadata else None

                # Deserialize multiband calibration
                multiband = None
                multiband_raw = phot_summary.get("multiband_calibration")
                if multiband_raw:
                    try:
                        from senpai.engine.photometry.color_terms import MultiBandCalibration
                        multiband = MultiBandCalibration.model_validate(multiband_raw) if isinstance(multiband_raw, dict) else multiband_raw
                    except Exception:
                        pass

                tmp = SC(
                    x=float(streak_x), y=float(streak_y),
                    angle_deg=float(angle), length_pixels=float(length),
                    width_pixels=float(fwhm), peak_snr=0.0,
                    directional_excess=0.0, fractional_excess=0.0,
                )
                try:
                    measure_streak_candidate_photometry(
                        frame.frame, [tmp], zero_point=zp, zero_point_err=zp_err,
                        exposure_time=exp_time, fwhm=fwhm,
                        gain=config.photometry.gain,
                        multiband_calibration=multiband,
                        observation_filter=obs_filter,
                    )
                    flux_val = tmp.flux
                    flux_err_val = tmp.flux_err
                    inst_mag = tmp.instrumental_magnitude
                    cal_mags = tmp.calibrated_magnitudes
                    mag_errs = tmp.magnitude_errs
                except Exception:
                    pass

        detection = SatelliteInImage(
            x=float(streak_x),
            y=float(streak_y),
            snr=float(cs.best_snr),
            ra=ra_val,
            dec=dec_val,
            pixel_fwhm=float(fwhm),
            flux=flux_val,
            flux_err=flux_err_val,
            instrumental_magnitude=inst_mag,
            calibrated_magnitudes=cal_mags,
            magnitude_errs=mag_errs,
            observation_filter=obs_filter,
            detection_type="streak",
            angle_deg=float(angle),
            length_pixels=float(length),
            rate_pixels_per_sec=float(rate),
            rate_arcsec_per_sec=rate_arcsec,
        )

        # Add to frame.detections
        if frame.detections is None:
            img_meta = frame.starfield.image_metadata if frame.starfield else None
            if img_meta is None:
                continue
            frame.detections = SatelliteListImage(
                detections=[], image_metadata=img_meta,
            )
        frame.detections.detections.append(detection)
