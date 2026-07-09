import logging
import time

import numpy as np

from senpai.core.config import get_config
from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
from senpai.engine.models.starfield import StarInSpace
from senpai.engine.models.streak_measurement import StreakMeasurement

logger = logging.getLogger(__name__)


def extract_box_statistics(
    image: np.ndarray,
    x: float,
    y: float,
    box_size: int = 11,
) -> dict:
    """
    Extract simple statistics from a box around a position.
    Much faster than flood-fill for validation purposes.

    Args:
        image: The image data
        x, y: Center position
        box_size: Size of box to extract (should be odd)

    Returns:
        dict with 'max', 'sum', 'mean', 'valid' keys
    """
    x_int, y_int = int(round(x)), int(round(y))
    half_box = box_size // 2

    # Check bounds
    if (
        x_int - half_box < 0
        or x_int + half_box >= image.shape[1]
        or y_int - half_box < 0
        or y_int + half_box >= image.shape[0]
    ):
        return {"max": 0.0, "sum": 0.0, "mean": 0.0, "valid": False}

    # Extract box
    box = image[
        y_int - half_box : y_int + half_box + 1, x_int - half_box : x_int + half_box + 1
    ]

    return {
        "max": float(np.max(box)),
        "sum": float(np.sum(box)),
        "mean": float(np.mean(box)),
        "median": float(np.median(box)),
        "valid": True,
    }


def quick_correlation_from_boxes(
    target_frame: np.ndarray,
    source_frame: np.ndarray,
    shift_x: float,
    shift_y: float,
    catalog_stars: list[StarInSpace],
    box_size: int = 11,
    max_stars: int = 50,
    debug_label: str = "",
) -> tuple[float, int, list]:
    """
    Fast correlation calculation using box statistics instead of flood-fill.

    Args:
        target_frame: Target frame data
        source_frame: Source frame data
        shift_x, shift_y: Shift to apply to source positions
        catalog_stars: List of catalog stars
        box_size: Size of box around each star
        max_stars: Maximum number of stars to use
        debug_label: Optional label for debug logging

    Returns:
        tuple: (correlation, n_valid_stars, box_stats_list)
    """
    # Sort stars by magnitude (brightest first)
    sorted_stars = sorted(
        catalog_stars, key=lambda s: s.magnitude if hasattr(s, "magnitude") else 999
    )
    stars_to_test = sorted_stars[:max_stars]

    source_stats = []
    target_stats = []
    source_max_vals = []
    target_max_vals = []
    valid_count = 0

    for star in stars_to_test:
        # Extract source box stats
        source_box = extract_box_statistics(source_frame, star.x, star.y, box_size)
        if not source_box["valid"]:
            continue

        # Calculate shifted position
        x_shifted = star.x - shift_x
        y_shifted = star.y - shift_y

        # Extract target box stats
        target_box = extract_box_statistics(
            target_frame, x_shifted, y_shifted, box_size
        )
        if not target_box["valid"]:
            continue

        # Calculate background-subtracted fluxes (net flux)
        # This is more robust than raw sum which is affected by background levels
        # and helps reject noise (where net flux ~ 0)
        box_area = box_size * box_size
        source_net = source_box["sum"] - (source_box["median"] * box_area)
        target_net = target_box["sum"] - (target_box["median"] * box_area)

        # Calculate standard deviation estimate for noise rejection
        # For Gaussian noise, std ≈ (75th percentile - 25th percentile) / 1.35
        # But we'll use a simpler check: max - median should be significant
        source_snr = (
            (source_box["max"] - source_box["median"]) / max(source_box["median"], 1.0)
            if source_box["median"] > 0
            else source_box["max"]
        )
        target_snr = (
            (target_box["max"] - target_box["median"]) / max(target_box["median"], 1.0)
            if target_box["median"] > 0
            else target_box["max"]
        )

        # Require SIGNIFICANT signal above background in both frames
        # Real sources should have peak values well above median
        # For rate-tracked frames with streaks, we need at least 50% above median
        # This is strict but necessary to avoid correlating noise
        min_signal_ratio = 0.50  # Peak should be at least 50% above median

        if source_snr < min_signal_ratio:
            logger.debug(
                f"{debug_label}: Star at ({star.x:.1f}, {star.y:.1f}) rejected: "
                f"source SNR too low: {source_snr:.3f} < {min_signal_ratio:.2f} "
                f"(max={source_box['max']:.1f}, median={source_box['median']:.1f})"
            )
            continue

        if target_snr < min_signal_ratio:
            logger.debug(
                f"{debug_label}: Star at ({star.x:.1f}, {star.y:.1f}) rejected: "
                f"target SNR too low: {target_snr:.3f} < {min_signal_ratio:.2f} "
                f"(max={target_box['max']:.1f}, median={target_box['median']:.1f})"
            )
            continue

        # Filter out cases where the target signal is likely just noise
        # A real source should have positive net flux
        if target_net <= 0:
            logger.debug(
                f"{debug_label}: Star at ({star.x:.1f}, {star.y:.1f}) rejected: "
                f"target_net={target_net:.1f} <= 0"
            )
            continue

        # Also ensure source is positive (should be, as it's a catalog star)
        if source_net <= 0:
            logger.debug(
                f"{debug_label}: Star at ({star.x:.1f}, {star.y:.1f}) rejected: "
                f"source_net={source_net:.1f} <= 0"
            )
            continue

        source_stats.append(source_net)
        target_stats.append(target_net)
        source_max_vals.append(source_box["max"])
        target_max_vals.append(target_box["max"])
        valid_count += 1

        logger.debug(
            f"{debug_label}: Star {valid_count} at ({star.x:.1f}, {star.y:.1f}) -> ({x_shifted:.1f}, {y_shifted:.1f}): "
            f"source_net={source_net:.1f} (max={source_box['max']:.1f}, median={source_box['median']:.1f}), "
            f"target_net={target_net:.1f} (max={target_box['max']:.1f}, median={target_box['median']:.1f})"
        )

    # Need at least 3 stars for any correlation
    # We'll apply skepticism to high correlations with few stars later
    min_stars = 3
    if valid_count < min_stars:
        logger.debug(
            f"{debug_label}: Insufficient valid stars: {valid_count} < {min_stars}"
        )
        return 0.0, valid_count, []

    # Calculate correlation
    source_stats = np.array(source_stats)
    target_stats = np.array(target_stats)
    source_max_vals = np.array(source_max_vals)
    target_max_vals = np.array(target_max_vals)

    # Try multiple correlation metrics and use the best one
    # 1. Spearman on net flux (original)
    # 2. Spearman on max values (simpler, no background subtraction)
    # 3. Pearson on net flux (for comparison)

    from scipy.stats import pearsonr, spearmanr

    corr_spearman_net, _ = spearmanr(source_stats, target_stats)
    corr_spearman_max, _ = spearmanr(source_max_vals, target_max_vals)

    # Calculate Pearson only if we have variation in both arrays
    if np.std(source_stats) > 0 and np.std(target_stats) > 0:
        corr_pearson_net, _ = pearsonr(source_stats, target_stats)
    else:
        corr_pearson_net = 0.0

    if np.std(source_max_vals) > 0 and np.std(target_max_vals) > 0:
        corr_pearson_max, _ = pearsonr(source_max_vals, target_max_vals)
    else:
        corr_pearson_max = 0.0

    # Handle NaN cases
    corr_spearman_net = 0.0 if np.isnan(corr_spearman_net) else corr_spearman_net
    corr_spearman_max = 0.0 if np.isnan(corr_spearman_max) else corr_spearman_max
    corr_pearson_net = 0.0 if np.isnan(corr_pearson_net) else corr_pearson_net
    corr_pearson_max = 0.0 if np.isnan(corr_pearson_max) else corr_pearson_max

    # Use a weighted combination of correlations instead of just the maximum
    # Taking the max is too lenient - random noise can make ONE metric correlate
    # Instead, give primary weight to Spearman (robust) and secondary to Pearson
    # Also prefer net flux over raw max (better background handling)
    correlation = (
        0.4 * corr_spearman_net
        + 0.3 * corr_pearson_net
        + 0.2 * corr_spearman_max
        + 0.1 * corr_pearson_max
    )

    # Log all metrics for debugging
    logger.debug(
        f"{debug_label}: Correlations - Spearman(net)={corr_spearman_net:.3f}, "
        f"Spearman(max)={corr_spearman_max:.3f}, Pearson(net)={corr_pearson_net:.3f}, "
        f"Pearson(max)={corr_pearson_max:.3f}, WEIGHTED={correlation:.3f}"
    )

    # Also log when metrics strongly disagree (could indicate measurement issues)
    metric_std = np.std(
        [corr_spearman_net, corr_spearman_max, corr_pearson_net, corr_pearson_max]
    )
    if metric_std > 0.3:
        logger.warning(
            f"{debug_label}: Metrics disagree significantly (std={metric_std:.3f}) - "
            f"possible noise correlation OR measurement issues (e.g., box too small for streaks)"
        )
        # When metrics disagree, also return the best single metric for comparison
        best_single_metric = max(
            corr_spearman_net, corr_spearman_max, corr_pearson_net, corr_pearson_max
        )
        logger.info(
            f"{debug_label}: Best single metric: {best_single_metric:.3f} "
            f"(weighted gave {correlation:.3f})"
        )

    return correlation, valid_count, list(zip(source_stats, target_stats, strict=False))


def validate_shift_lightweight(
    target: RateTrackFrame | SiderealFrame,
    source: RateTrackFrame | SiderealFrame,
    shift_x: float,
    shift_y: float,
    catalog_stars: list[StarInSpace],
    trial: int = 1,
    streak_rotation_deg: float | None = None,
    fwhm_exclusion: float | None = None,
) -> tuple[bool, float, StreakMeasurement | None, tuple[float, float]]:
    """
    Lightweight validation using box statistics and random shift comparison.

    Args:
        target: The frame we're shifting to align with the source frame
        source: The reference frame
        shift_x: Proposed x shift (pixels)
        shift_y: Proposed y shift (pixels)
        catalog_stars: List of stars from the source frame
        trial: Trial number for debugging
        streak_rotation_deg: Optional streak rotation in degrees (deprecated - use fwhm_exclusion instead)
        fwhm_exclusion: Exclusion radius perpendicular to shift direction (pixels)
                       Random samples will be placed at least this far from the shift line

    Returns:
        tuple: (valid, correlation, streak_measurement, shift_correction)
    """
    config = get_config()
    target_frame = target.frame.data
    source_frame = source.frame.data

    start_time = time.time()

    # Get config parameters
    base_box_size = config.validation.box_size
    n_random_trials = config.validation.n_random_trials
    random_radius = config.validation.random_radius_pixels
    max_stars = config.validation.max_validation_stars

    # Adaptive box size: increase for wide streaks to capture full flux
    # fwhm_exclusion represents the streak width perpendicular to motion
    if fwhm_exclusion is not None and fwhm_exclusion > 8:
        # For wide streaks, use larger boxes (at least 2x FWHM, minimum base_box_size)
        box_size = max(base_box_size, int(fwhm_exclusion * 2.5))
        logger.info(
            f"Using adaptive box size {box_size}px (base={base_box_size}px) "
            f"for FWHM={fwhm_exclusion:.1f}px"
        )
    else:
        box_size = base_box_size

    logger.info(
        f"Lightweight validation: shift=({shift_x:.1f}, {shift_y:.1f}), "
        f"box_size={box_size}, random_trials={n_random_trials}"
    )

    # 1. Test proposed shift
    proposed_corr, proposed_n_stars, _ = quick_correlation_from_boxes(
        target_frame,
        source_frame,
        shift_x,
        shift_y,
        catalog_stars,
        box_size,
        max_stars,
        debug_label="PROPOSED",
    )

    logger.info(
        f"Proposed shift: correlation={proposed_corr:.3f}, n_stars={proposed_n_stars}"
    )

    # If we don't have enough stars, reject immediately
    if proposed_n_stars < 4:
        logger.warning(f"Insufficient stars for validation: {proposed_n_stars} < 4")
        return False, 0.0, None, (0.0, 0.0)

    # Calculate shift magnitude to determine if we should use perpendicular sampling
    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)

    # 1b. Direction-ambiguity guard: streak patterns correlate almost as well
    # under a sign-flipped shift, and a flipped hop silently reverses the WCS
    # chain. If the negated shift correlates decisively better, the proposed
    # shift is the wrong branch of that ambiguity — reject it here so the
    # solver can try again rather than poison every downstream frame.
    if config.validation.test_negated_shift and shift_magnitude > 5.0:
        negated_corr, negated_n_stars, _ = quick_correlation_from_boxes(
            target_frame,
            source_frame,
            -shift_x,
            -shift_y,
            catalog_stars,
            box_size,
            max_stars,
            debug_label="NEGATED",
        )
        if (
            negated_n_stars >= 4
            and negated_corr > proposed_corr * config.validation.negated_rejection_ratio
        ):
            logger.warning(
                "Rejecting proposed shift (%.1f, %.1f): its negation correlates "
                "better (%.3f vs %.3f) — direction ambiguity",
                shift_x,
                shift_y,
                negated_corr,
                proposed_corr,
            )
            return False, proposed_corr, None, (0.0, 0.0)

    # 2. Test random alternative shifts
    # Strategy: Sample perpendicular to the shift vector to avoid landing on the streak
    random_correlations = []
    random_shifts = []
    random_n_stars = []  # Track number of stars for each random trial

    # Use fwhm_exclusion if provided, otherwise derive from config or use a default
    if fwhm_exclusion is not None:
        min_perpendicular_offset = fwhm_exclusion
        max_perpendicular_offset = max(fwhm_exclusion * 3, random_radius)
        logger.info(
            f"Using FWHM-based exclusion: {fwhm_exclusion:.1f}px, "
            f"sampling {min_perpendicular_offset:.1f} to {max_perpendicular_offset:.1f}px perpendicular to shift"
        )
    elif streak_rotation_deg is not None:
        # Backward compatibility: use streak_rotation_deg if provided
        min_perpendicular_offset = random_radius * 0.5
        max_perpendicular_offset = random_radius
        logger.info(
            f"Using legacy streak rotation parameter ({streak_rotation_deg:.1f}°)"
        )
    else:
        # No exclusion info - use broader circular sampling
        min_perpendicular_offset = 0
        max_perpendicular_offset = random_radius

    # If shift is large enough (> 5 pixels), use the shift vector to define streak direction
    # Otherwise, fall back to circular sampling
    if shift_magnitude > 5.0 and fwhm_exclusion is not None:
        # Derive streak direction from shift vector
        shift_angle_rad = np.arctan2(shift_y, shift_x)
        perp_angle_rad = shift_angle_rad + np.pi / 2

        logger.info(
            f"Shift magnitude={shift_magnitude:.1f}px, angle={np.rad2deg(shift_angle_rad):.1f}°. "
            f"Sampling perpendicular to shift (at {np.rad2deg(perp_angle_rad):.1f}°)"
        )

        for i in range(n_random_trials):
            # Sample perpendicular to shift direction, avoiding the streak
            # Alternate between positive and negative offsets for better coverage
            sign = 1 if i % 2 == 0 else -1
            perp_offset = sign * np.random.uniform(
                min_perpendicular_offset, max_perpendicular_offset
            )

            # Add small random component along shift direction (to test slight position errors)
            # Keep this minimal to avoid landing on the streak
            along_shift_offset = np.random.uniform(
                -min_perpendicular_offset * 0.3, min_perpendicular_offset * 0.3
            )

            # Calculate random shift position
            rand_x = (
                shift_x
                + perp_offset * np.cos(perp_angle_rad)
                + along_shift_offset * np.cos(shift_angle_rad)
            )
            rand_y = (
                shift_y
                + perp_offset * np.sin(perp_angle_rad)
                + along_shift_offset * np.sin(shift_angle_rad)
            )

            rand_corr, rand_n_stars_val, _ = quick_correlation_from_boxes(
                target_frame,
                source_frame,
                rand_x,
                rand_y,
                catalog_stars,
                box_size,
                max_stars,
                debug_label=f"RANDOM_{i+1}",
            )
            random_correlations.append(rand_corr)
            random_shifts.append((rand_x, rand_y))
            random_n_stars.append(rand_n_stars_val)

            logger.info(
                f"Random trial {i+1}: shift=({rand_x:.1f}, {rand_y:.1f}), "
                f"perp_offset={perp_offset:.1f}, along_offset={along_shift_offset:.1f}, "
                f"corr={rand_corr:.3f}, n_stars={rand_n_stars_val}"
            )
    elif streak_rotation_deg is not None:
        # Legacy mode: use provided streak rotation
        streak_angle_rad = np.deg2rad(streak_rotation_deg)
        perp_angle_rad = streak_angle_rad + np.pi / 2

        logger.info(
            f"Using provided streak rotation ({streak_rotation_deg:.1f}°) for perpendicular sampling"
        )

        for i in range(n_random_trials):
            sign = 1 if i % 2 == 0 else -1
            perp_offset = sign * np.random.uniform(
                min_perpendicular_offset, max_perpendicular_offset
            )
            along_streak_offset = np.random.uniform(
                -min_perpendicular_offset * 0.3, min_perpendicular_offset * 0.3
            )

            rand_x = (
                shift_x
                + perp_offset * np.cos(perp_angle_rad)
                + along_streak_offset * np.cos(streak_angle_rad)
            )
            rand_y = (
                shift_y
                + perp_offset * np.sin(perp_angle_rad)
                + along_streak_offset * np.sin(streak_angle_rad)
            )

            rand_corr, rand_n_stars_val, _ = quick_correlation_from_boxes(
                target_frame,
                source_frame,
                rand_x,
                rand_y,
                catalog_stars,
                box_size,
                max_stars,
                debug_label=f"RANDOM_{i+1}",
            )
            random_correlations.append(rand_corr)
            random_shifts.append((rand_x, rand_y))
            random_n_stars.append(rand_n_stars_val)

            logger.info(
                f"Random trial {i+1}: shift=({rand_x:.1f}, {rand_y:.1f}), "
                f"perp_offset={perp_offset:.1f}, corr={rand_corr:.3f}, n_stars={rand_n_stars_val}"
            )
    else:
        # Fallback: circular annulus sampling (no streak info available)
        logger.info(
            f"No streak information - using circular annulus sampling "
            f"({min_perpendicular_offset:.1f} to {max_perpendicular_offset:.1f}px)"
        )

        for i in range(n_random_trials):
            # Generate random shift in an annulus around proposed shift
            angle = np.random.uniform(0, 2 * np.pi)
            # Use annulus (ring) instead of full circle to ensure separation
            if min_perpendicular_offset > 0:
                radius = np.random.uniform(
                    min_perpendicular_offset, max_perpendicular_offset
                )
            else:
                radius = np.random.uniform(0, max_perpendicular_offset)
            rand_x = shift_x + radius * np.cos(angle)
            rand_y = shift_y + radius * np.sin(angle)

            rand_corr, rand_n_stars_val, _ = quick_correlation_from_boxes(
                target_frame,
                source_frame,
                rand_x,
                rand_y,
                catalog_stars,
                box_size,
                max_stars,
                debug_label=f"RANDOM_{i+1}",
            )
            random_correlations.append(rand_corr)
            random_shifts.append((rand_x, rand_y))
            random_n_stars.append(rand_n_stars_val)

            logger.info(
                f"Random trial {i+1}: shift=({rand_x:.1f}, {rand_y:.1f}), "
                f"radius={radius:.1f}, corr={rand_corr:.3f}, n_stars={rand_n_stars_val}"
            )

    # 3. Find best shift among all tested (proposed + randoms)
    # Apply confidence weighting based on number of stars
    # More stars = more confidence = higher effective score
    random_correlations = np.array(random_correlations)
    random_n_stars = np.array(random_n_stars)

    all_correlations_raw = np.array([proposed_corr] + list(random_correlations))
    all_n_stars = np.array([proposed_n_stars] + list(random_n_stars))
    all_shifts = [(shift_x, shift_y)] + random_shifts

    # Calculate confidence-weighted scores
    # Confidence increases with sqrt(n) because standard error decreases with sqrt(n)
    # Use the maximum number of stars as the reference point
    max_n_stars = np.max(all_n_stars)

    # Calculate confidence weight for each correlation
    # Shifts with more stars get boosted, shifts with fewer stars get penalized
    # (Minimum of 5 stars is enforced above in quick_correlation_from_boxes)
    confidence_weights = np.sqrt(all_n_stars / max_n_stars)

    # Additional penalty for suspiciously perfect correlations with few stars
    # A correlation of 1.0 with only 4-5 stars is likely spurious
    all_correlations_weighted = []
    for i, (corr, n_stars, weight) in enumerate(
        zip(all_correlations_raw, all_n_stars, confidence_weights, strict=False)
    ):
        weighted_corr = corr * weight

        # Apply skepticism penalty based on correlation strength and star count
        # ONLY penalize suspiciously perfect correlations with VERY FEW stars (3-5)
        # 6+ stars with high correlation is trustworthy - don't penalize!
        skepticism_factor = 1.0

        # Strong skepticism ONLY for perfect correlations with 3-5 stars
        if corr >= 0.98 and n_stars <= 5:
            # Perfect correlation (>0.98) with 3-5 stars is suspicious
            skepticism_factor = 0.4 + 0.6 * (
                (n_stars - 3) / 2.0
            )  # 0.4 at 3 stars, 1.0 at 5 stars

        # Moderate skepticism for very high correlations with only 3-4 stars
        elif corr >= 0.95 and n_stars <= 4:
            # Very high correlation (0.95-0.98) with 3-4 stars
            skepticism_factor = 0.6 + 0.4 * (
                (n_stars - 3) / 1.0
            )  # 0.6 at 3 stars, 1.0 at 4 stars

        # Light skepticism for high correlations with only 3 stars
        elif corr >= 0.85 and n_stars == 3:
            # High correlation with only 3 stars
            skepticism_factor = 0.7

        # Additional skepticism if this shift finds WAY more stars than proposed
        # This can indicate noise matching in random shifts
        # DISABLED: Let confidence weighting handle this naturally
        # if i > 0:  # Only apply to random trials, not proposed
        #     star_ratio = n_stars / all_n_stars[0]  # Compare to proposed
        #     if star_ratio > 2.0:  # More than 2x the stars
        #         # Apply additional penalty - this might be spurious
        #         excess_penalty = 0.7 + 0.3 * min(
        #             1.0, 2.0 / star_ratio
        #         )  # Max 0.7 penalty
        #         skepticism_factor *= excess_penalty
        #         logger.info(
        #             f"Shift {i} (n={n_stars}): Excess star count vs proposed "
        #             f"({all_n_stars[0]} stars) - applying penalty {excess_penalty:.3f}"
        #         )

        if skepticism_factor < 1.0:
            weighted_corr *= skepticism_factor
            logger.info(
                f"Shift {i} (n={n_stars}): Final weighted score: "
                f"{corr:.3f} -> {corr * weight:.3f} -> {weighted_corr:.3f} "
                f"(confidence={weight:.3f}, skepticism={skepticism_factor:.3f})"
            )
        elif n_stars != max_n_stars:
            logger.debug(
                f"Shift {i} (n={n_stars}): corr={corr:.3f} -> weighted={weighted_corr:.3f} "
                f"(confidence weight={weight:.3f})"
            )

        all_correlations_weighted.append(weighted_corr)

    all_correlations_weighted = np.array(all_correlations_weighted)

    # Find the best shift using weighted correlations
    best_idx = np.argmax(all_correlations_weighted)
    best_corr_weighted = all_correlations_weighted[best_idx]
    best_corr_raw = all_correlations_raw[best_idx]
    best_n_stars = all_n_stars[best_idx]
    best_shift = all_shifts[best_idx]

    # Calculate proposed shift's weighted correlation and ratio to best
    proposed_corr_weighted = all_correlations_weighted[0]
    corr_ratio = (
        proposed_corr_weighted / best_corr_weighted if best_corr_weighted > 0 else 0.0
    )

    logger.info(
        f"Star counts: proposed={proposed_n_stars}, "
        f"random range=[{np.min(random_n_stars)}-{np.max(random_n_stars)}], "
        f"max_overall={max_n_stars}"
    )

    logger.info(
        f"Random correlations (raw): mean={np.mean(random_correlations):.3f}, "
        f"max={np.max(random_correlations):.3f}"
    )
    logger.info(
        f"Random correlations (weighted): mean={np.mean(all_correlations_weighted[1:]):.3f}, "
        f"max={np.max(all_correlations_weighted[1:]):.3f}"
    )
    logger.info(
        f"Best correlation: {best_corr_raw:.3f} (weighted: {best_corr_weighted:.3f}) "
        f"at shift ({best_shift[0]:.1f}, {best_shift[1]:.1f}) with {best_n_stars} stars"
    )
    logger.info(
        f"Proposed: correlation={proposed_corr:.3f} (weighted: {proposed_corr_weighted:.3f}) "
        f"with {proposed_n_stars} stars, ratio to best: {corr_ratio:.3f}"
    )

    if best_idx == 0:
        logger.info("Proposed shift IS the best among all tested shifts")
    else:
        logger.info(f"Best shift is random trial #{best_idx}")

    # 4. Validate based on "near-best" criterion
    # Accept if proposed is within a few percent of the best AND has reasonable absolute correlation
    min_corr_ratio = config.validation.min_correlation_ratio
    min_absolute_corr = config.validation.min_absolute_correlation
    lenient_absolute_corr = config.validation.lenient_absolute_correlation

    # SPECIAL CASE: If proposed is FAR better than ALL random trials (all negative/near-zero)
    # then we should be very lenient with absolute threshold - it's clearly the only real signal
    max_random_corr = np.max(random_correlations)
    mean_random_corr = np.mean(random_correlations)

    very_lenient_threshold = 0.35  # For when randoms are all terrible
    if max_random_corr < 0.15 and mean_random_corr < 0.05 and proposed_corr > 0.3:
        # All randoms are near-zero or negative, proposed is positive
        # This strongly suggests proposed is the only real signal
        logger.info(
            f"Proposed ({proposed_corr:.3f}) is FAR better than all randoms "
            f"(max={max_random_corr:.3f}, mean={mean_random_corr:.3f}). "
            f"Using very lenient threshold ({very_lenient_threshold:.2f}) - "
            f"proposed is clearly the only real signal"
        )
        lenient_absolute_corr = very_lenient_threshold
        # Also be more lenient with the ratio requirement since we're already the best
        min_corr_ratio = 0.95

    # Use lenient threshold when correlation ratio is high (>= 0.93) but absolute is slightly low
    # This handles cases with few stars where absolute correlation is lower but ratio is very close to best
    lenient_ratio_threshold = 0.93
    use_lenient = corr_ratio >= lenient_ratio_threshold

    if use_lenient:
        effective_min_absolute_corr = lenient_absolute_corr
        logger.info(
            f"Using lenient absolute correlation threshold ({lenient_absolute_corr:.2f}) "
            f"because correlation ratio ({corr_ratio:.3f}) >= {lenient_ratio_threshold:.2f}"
        )
    else:
        effective_min_absolute_corr = min_absolute_corr

    # Additional check: if proposed has significantly fewer stars than best, be more strict
    if proposed_n_stars < best_n_stars - 1:
        logger.warning(
            f"Proposed shift has fewer stars ({proposed_n_stars}) than best ({best_n_stars}), "
            f"requiring higher correlation ratio for validation"
        )
        # Require a bit closer to best when we have fewer stars. Config-driven
        # (was a hardcoded 0.99, which razor-thin-rejected correct shifts at
        # ratio ~0.987 → fell through to a flipped shift); the random-trials noise
        # guard below + the absolute-correlation floor still catch bad shifts.
        min_corr_ratio = max(min_corr_ratio, config.validation.fewer_stars_correlation_ratio)

    # CRITICAL: If proposed has significantly fewer stars than MULTIPLE random trials,
    # this strongly suggests we're matching noise and the proposed shift is wrong
    random_stars_above_proposed = sum(
        1 for n in random_n_stars if n > proposed_n_stars + 3
    )
    if random_stars_above_proposed >= 3:
        # At least 3 random trials found significantly more stars (4+ more)
        logger.error(
            f"SUSPICIOUS: {random_stars_above_proposed} random trials found >3 more stars "
            f"than proposed ({proposed_n_stars}). This suggests noise correlation!"
        )
        # Require much stricter validation (config-driven).
        min_corr_ratio = config.validation.noise_correlation_ratio
        effective_min_absolute_corr = max(
            effective_min_absolute_corr, config.validation.noise_min_absolute_correlation
        )

    valid = (corr_ratio >= min_corr_ratio) and (
        proposed_corr >= effective_min_absolute_corr
    )

    elapsed = time.time() - start_time

    # Log validation result with detailed reasoning
    if valid:
        logger.info(
            f"Validation PASSED in {elapsed:.2f}s "
            f"(proposed_corr={proposed_corr:.3f} >= {effective_min_absolute_corr:.2f}, "
            f"ratio_to_best={corr_ratio:.3f} >= {min_corr_ratio:.2f})"
        )
    else:
        # Detailed failure explanation
        ratio_ok = corr_ratio >= min_corr_ratio
        abs_ok = proposed_corr >= effective_min_absolute_corr

        failure_reasons = []
        if not ratio_ok:
            failure_reasons.append(f"ratio {corr_ratio:.3f} < {min_corr_ratio:.2f}")
        if not abs_ok:
            failure_reasons.append(
                f"correlation {proposed_corr:.3f} < {effective_min_absolute_corr:.2f}"
            )

        logger.warning(
            f"Validation FAILED in {elapsed:.2f}s: {' AND '.join(failure_reasons)}"
        )

        # Additional diagnostic: if proposed is clearly better than randoms but correlation is low,
        # this suggests measurement issues (box size, background, etc.) rather than bad alignment
        if max_random_corr < 0.2 and proposed_corr > 0.3:
            logger.warning(
                f"NOTE: Proposed ({proposed_corr:.3f}) is significantly better than "
                f"all randoms (max={max_random_corr:.3f}), suggesting this may be a "
                f"MEASUREMENT ISSUE (e.g., box size too small for streaks) rather than "
                f"a bad alignment. Consider reviewing box_size parameter."
            )

    # Debug plotting if enabled
    if config.plotting.debug:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        from senpai.engine.plotting.normalization import zscale

        # Scale target frame for visualization
        scaled_target = zscale(target_frame)
        vmin, vmax = np.percentile(scaled_target, [1, 99])
        scaled_target = np.clip((scaled_target - vmin) / (vmax - vmin), 0, 1)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(
            scaled_target, cmap="viridis", origin="upper", interpolation="nearest"
        )
        ax.set_title(
            f"Lightweight Validation: {source.index} -> {target.index} (trial {trial})\n"
            f"Proposed: corr={proposed_corr:.3f}, ratio={corr_ratio:.3f}, valid={valid}"
        )

        # Draw boxes around proposed shift positions
        # Get stars used in validation (sorted by magnitude, brightest first)
        sorted_stars = sorted(
            catalog_stars, key=lambda s: s.magnitude if hasattr(s, "magnitude") else 999
        )
        stars_to_plot = sorted_stars[:max_stars]

        box_half = box_size // 2
        for star in stars_to_plot:
            # Calculate shifted position in target frame
            x_shifted = star.x - shift_x
            y_shifted = star.y - shift_y

            # Check bounds
            if (
                x_shifted - box_half >= 0
                and x_shifted + box_half < target_frame.shape[1]
                and y_shifted - box_half >= 0
                and y_shifted + box_half < target_frame.shape[0]
            ):
                rect = Rectangle(
                    (x_shifted - box_half, y_shifted - box_half),
                    box_size,
                    box_size,
                    fill=False,
                    color="black",
                    linewidth=2,
                    alpha=0.7,
                    label="Proposed shift" if star == stars_to_plot[0] else "",
                )
                ax.add_patch(rect)

        # Plot random trial positions - show where a representative star would be placed
        # Use the first (brightest) star as a reference
        if len(random_shifts) > 0 and len(stars_to_plot) > 0:
            ref_star = stars_to_plot[0]
            random_corrs_array = np.array(random_correlations)
            # Normalize correlations for colormap (0 to 1 range)
            if random_corrs_array.max() > random_corrs_array.min():
                norm_corrs = (random_corrs_array - random_corrs_array.min()) / (
                    random_corrs_array.max() - random_corrs_array.min()
                )
            else:
                norm_corrs = np.ones_like(random_corrs_array)

            # Use a colormap to color by correlation value
            cmap = plt.cm.plasma  # Different colormap to distinguish from boxes
            for i, ((rand_shift_x, rand_shift_y), corr, norm_corr) in enumerate(
                zip(random_shifts, random_correlations, norm_corrs, strict=False)
            ):
                # Calculate where the reference star would be placed with this random shift
                rand_x_pos = ref_star.x - rand_shift_x
                rand_y_pos = ref_star.y - rand_shift_y

                # Check bounds
                if (
                    rand_x_pos >= 0
                    and rand_x_pos < target_frame.shape[1]
                    and rand_y_pos >= 0
                    and rand_y_pos < target_frame.shape[0]
                ):
                    color = cmap(norm_corr)
                    # Use different marker styles for better visibility
                    marker = "o" if i % 2 == 0 else "s"
                    ax.scatter(
                        rand_x_pos,
                        rand_y_pos,
                        c=[color],
                        s=150,
                        marker=marker,
                        edgecolors="white",
                        linewidths=1.5,
                        alpha=0.9,
                        label=(
                            f"Random {i+1}: {corr:.3f}" if i < 5 else ""
                        ),  # Only label first 5 to avoid clutter
                        zorder=8,
                    )

        # Mark the best shift position with a special marker
        # Show where the reference star would be with the best shift
        if len(stars_to_plot) > 0:
            ref_star = stars_to_plot[0]
            if best_idx == 0:
                # Best is the proposed shift
                best_x_pos = ref_star.x - shift_x
                best_y_pos = ref_star.y - shift_y
                label_text = "Best (proposed)"
            else:
                # Best is a random trial
                best_rand_shift_x, best_rand_shift_y = random_shifts[best_idx - 1]
                best_x_pos = ref_star.x - best_rand_shift_x
                best_y_pos = ref_star.y - best_rand_shift_y
                label_text = f"Best (random {best_idx})"

            if (
                best_x_pos >= 0
                and best_x_pos < target_frame.shape[1]
                and best_y_pos >= 0
                and best_y_pos < target_frame.shape[0]
            ):
                ax.scatter(
                    best_x_pos,
                    best_y_pos,
                    s=400,
                    marker="*",
                    color="yellow",
                    edgecolors="red",
                    linewidths=3,
                    label=label_text,
                    zorder=10,
                )

        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        output_file = (
            config.runtime.output_dir
            / f"lightweight_validation_{source.index}_to_{target.index}_trial_{trial}.png"
        )
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved lightweight validation debug plot to {output_file}")

    # For now, return no streak measurement or shift correction
    # (these were mainly used for debugging in the old system)
    return valid, proposed_corr, None, (0.0, 0.0)


def validate_proposed_shift(
    target: RateTrackFrame | SiderealFrame,
    source: RateTrackFrame | SiderealFrame,
    shift_x: float,
    shift_y: float,
    catalog_stars: list[StarInSpace],
    trial: int = 1,
    streak_rotation_deg: float | None = None,
    fwhm_exclusion: float | None = None,
) -> tuple[bool, float, StreakMeasurement | None, tuple[float, float]]:
    """
    Validate proposed shift via lightweight box-based correlation.

    Args:
        target: The frame we're shifting to align with the source frame
        source: The reference frame
        shift_x: Proposed x shift (pixels)
        shift_y: Proposed y shift (pixels)
        catalog_stars: List of stars from the source frame
        trial: Trial number for debugging
        streak_rotation_deg: Optional streak rotation in degrees (deprecated)
        fwhm_exclusion: Exclusion radius perpendicular to shift direction (pixels)

    Returns:
        tuple: (valid, correlation, streak_measurement, shift_correction)
    """
    logger.info(f"Using lightweight box-based validation (trial {trial})")
    return validate_shift_lightweight(
        target,
        source,
        shift_x,
        shift_y,
        catalog_stars,
        trial,
        streak_rotation_deg,
        fwhm_exclusion,
    )

