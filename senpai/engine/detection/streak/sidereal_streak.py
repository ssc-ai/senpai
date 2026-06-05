"""Streak detection in sidereal frames via directional matched filtering.

Use case: finding objects with unknown rates in sidereal-tracked imagery
(e.g. GEO survey).  Stars appear as point sources; targets of interest
appear as short-to-long streaks depending on their angular rate relative
to the sidereal background.

Pipeline
--------
1. Background-subtract the image.
2. Apply a bank of directional matched filters at evenly spaced angles.
3. At each pixel compute:
   - *isotropic response*: mean of all angles except the best (dominated by
     point sources, which light up all angles equally).
   - *directional excess*: best response minus isotropic.
   - *fractional excess*: directional excess / isotropic response.
     Stars have fractional excess ~5% (discretization noise).
     Streaks have fractional excess >> 30% (one angle dominates).
4. Pre-mask known catalog star regions in the directional excess map.
5. Threshold on BOTH absolute excess (above noise floor) AND fractional
   excess (filters out stars regardless of brightness).
6. For each detection hotspot, trace along the best filter-bank angle in
   the directional excess map to find streak endpoints ("squish" approach).
7. Refine angle/width against the original image.
"""

import logging

import numpy as np
from astropy.stats import sigma_clipped_stats
from pydantic import BaseModel, field_serializer
from scipy.ndimage import label, maximum_filter, map_coordinates
from scipy.optimize import curve_fit

from senpai.engine.detection.kernels import build_directional_filter_bank
from senpai.engine.detection.streak.masking import analyze_source_shape_fwhm
from senpai.engine.models.starfield import StarField

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class StreakCandidate(BaseModel):
    """A detected streak candidate in a sidereal frame."""

    x: float
    y: float
    angle_deg: float  # Streak direction in [0, 180)
    length_pixels: float
    width_pixels: float  # FWHM perpendicular to streak
    peak_snr: float
    directional_excess: float  # max_response - isotropic_response
    fractional_excess: float  # directional_excess / isotropic_response
    n_pixels: int = 0
    ra: float | None = None
    dec: float | None = None
    rate_pixels_per_sec: float | None = None
    rate_arcsec_per_sec: float | None = None
    # Photometry (populated by measure_streak_candidate_photometry)
    flux: float | None = None
    flux_err: float | None = None
    instrumental_magnitude: float | None = None
    calibrated_magnitudes: dict[str, float] | None = None
    magnitude_errs: dict[str, float] | None = None
    observation_filter: str | None = None

    @field_serializer(
        "x", "y", "angle_deg", "length_pixels", "width_pixels",
        "peak_snr", "directional_excess", "fractional_excess",
    )
    def _round2(self, v: float) -> float:
        return round(v, 2)

    @field_serializer("ra", "dec")
    def _round4(self, v: float | None) -> float | None:
        return round(v, 4) if v is not None else None

    @field_serializer("rate_pixels_per_sec", "rate_arcsec_per_sec")
    def _round3(self, v: float | None) -> float | None:
        return round(v, 3) if v is not None else None


# ---------------------------------------------------------------------------
# Directional filter application  (memory-efficient: O(3*image), not O(N*image))
# ---------------------------------------------------------------------------


def _apply_directional_filters_fft(
    image: np.ndarray,
    fwhm: float,
    n_angles: int = 36,
    filter_length_fwhm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core FFT-based directional filter bank on an image at native resolution."""
    from scipy.fft import irfft2, rfft2

    kernels, angles = build_directional_filter_bank(fwhm, n_angles, filter_length_fwhm)

    max_response = np.full(image.shape, -np.inf, dtype=np.float64)
    best_angle_idx = np.zeros(image.shape, dtype=np.int32)
    sum_response = np.zeros(image.shape, dtype=np.float64)

    k0 = kernels[0]
    fft_shape = (
        image.shape[0] + k0.shape[0] - 1,
        image.shape[1] + k0.shape[1] - 1,
    )

    image_fft = rfft2(image, s=fft_shape)

    for i, kernel in enumerate(kernels):
        kernel_fft = rfft2(kernel, s=fft_shape)
        raw = irfft2(image_fft * kernel_fft, s=fft_shape)

        ky, kx = kernel.shape
        oy, ox = ky // 2, kx // 2
        response = raw[oy : oy + image.shape[0], ox : ox + image.shape[1]]

        better = response > max_response
        max_response[better] = response[better]
        best_angle_idx[better] = i
        sum_response += response

    isotropic = (sum_response - max_response) / (n_angles - 1)
    directional_excess = max_response - isotropic
    best_angle_deg = angles[best_angle_idx]

    return directional_excess, best_angle_deg, isotropic


def apply_directional_filters(
    image: np.ndarray,
    fwhm: float,
    n_angles: int = 36,
    filter_length_fwhm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convolve image with every filter in the bank, return directional excess.

    Computes responses incrementally — only keeps running max and sum, so
    memory usage is O(3 x image_size) regardless of the number of angles.

    Performance: precomputes the image FFT once and reuses it for all angles.

    Returns:
        ``(directional_excess, best_angle_deg, isotropic)`` arrays.
    """
    return _apply_directional_filters_fft(image, fwhm, n_angles, filter_length_fwhm)


# ---------------------------------------------------------------------------
# Per-component characterization
# ---------------------------------------------------------------------------


def _characterize_component(
    image: np.ndarray,
    component_mask: np.ndarray,
    best_angle_deg: np.ndarray,
    directional_excess: np.ndarray,
    fractional_excess: np.ndarray,
    noise_std: float,
    fwhm: float,
    starfield: StarField | None = None,
    exposure_time: float | None = None,
) -> StreakCandidate | None:
    """Measure streak parameters for a single connected component."""
    y_coords, x_coords = np.where(component_mask)
    if len(y_coords) < 3:
        return None

    # PCA-based shape analysis
    shape = analyze_source_shape_fwhm(image, y_coords, x_coords)
    length = shape["length"]
    width = shape["fwhm_minor"]
    centroid_y, centroid_x = shape["center"]

    # Reject if too compact
    if length < fwhm * 1.5:
        return None

    # A real streak is a PSF smeared along one axis.  Its perpendicular
    # width MUST be consistent with the seeing FWHM.  Too narrow = noise
    # artifact on a pixel grid.  Too wide = not a streak.
    if width < fwhm * 0.3 or width > fwhm * 2.5:
        return None

    # Best angle: weighted circular mean over the component pixels
    excess_vals = directional_excess[component_mask]
    angle_vals = best_angle_deg[component_mask]
    weights = np.maximum(excess_vals, 0)
    total_w = weights.sum()

    if total_w > 0:
        rads = np.radians(2 * angle_vals)
        mean_sin = np.sum(weights * np.sin(rads)) / total_w
        mean_cos = np.sum(weights * np.cos(rads)) / total_w
        angle = float(np.degrees(np.arctan2(mean_sin, mean_cos)) / 2) % 180
    else:
        angle = float(shape["orientation"]) % 180

    peak_snr = float(np.max(excess_vals) / noise_std) if noise_std > 0 else 0.0
    peak_excess = float(np.max(excess_vals))
    peak_frac = float(np.max(fractional_excess[component_mask]))

    candidate = StreakCandidate(
        x=float(centroid_x),
        y=float(centroid_y),
        angle_deg=angle,
        length_pixels=float(length),
        width_pixels=float(width),
        peak_snr=peak_snr,
        directional_excess=peak_excess,
        fractional_excess=peak_frac,
        n_pixels=len(y_coords),
    )

    # Sky coordinates
    if starfield and starfield.wcs:
        try:
            wcs = starfield.wcs.to_astropy_wcs()
            sky = wcs.pixel_to_world(centroid_x, centroid_y)
            candidate.ra = float(sky.ra.deg)
            candidate.dec = float(sky.dec.deg)
        except Exception:
            logger.debug(
                "WCS pixel_to_world failed for streak at (%.1f, %.1f)",
                centroid_x, centroid_y,
            )

    # Rate estimate
    if exposure_time and exposure_time > 0:
        rate_pix = float(length) / exposure_time
        candidate.rate_pixels_per_sec = rate_pix
        if (
            starfield
            and starfield.wcs_metadata
            and hasattr(starfield.wcs_metadata, "x_ifov_arcsec")
        ):
            candidate.rate_arcsec_per_sec = rate_pix * starfield.wcs_metadata.x_ifov_arcsec

    return candidate


# ---------------------------------------------------------------------------
# Image-based streak refinement
# ---------------------------------------------------------------------------


def _gauss1d(x, amp, mu, sig, bg):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sig ** 2)) + bg


def _fit_perp_profile(
    image: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    fwhm: float,
) -> tuple[float, float, float, float] | None:
    """Fit a 1D Gaussian to the perpendicular cross-section at a given angle.

    Returns ``(fitted_fwhm, amplitude, local_noise, residual_rms)`` or
    *None* if the fit fails.
    """
    sigma = fwhm / 2.355
    h, w = image.shape
    perp_half = int(max(4 * fwhm, 10))
    angle_rad = np.radians(angle_deg)

    t = np.arange(-perp_half, perp_half + 1, dtype=np.float64)
    sx = cx + t * (-np.sin(angle_rad))
    sy = cy + t * np.cos(angle_rad)

    valid = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
    if valid.sum() < 7:
        return None

    profile = map_coordinates(image, [sy[valid], sx[valid]], order=1)
    tv = t[valid]

    try:
        p0 = [profile.max() - profile.min(), 0.0, sigma, np.median(profile)]
        bounds = ([0, -perp_half, 0.3, -np.inf], [np.inf, perp_half, 5 * sigma, np.inf])
        popt, _ = curve_fit(_gauss1d, tv, profile, p0=p0, bounds=bounds, maxfev=200)
    except (RuntimeError, ValueError):
        return None

    fitted_fwhm = 2.355 * abs(popt[2])
    amplitude = popt[0]
    residual_rms = float(np.sqrt(np.mean((_gauss1d(tv, *popt) - profile) ** 2)))

    wing_mask = np.abs(tv) > 2 * fwhm
    local_noise = (
        float(np.std(profile[wing_mask])) if wing_mask.sum() >= 5
        else float(np.std(profile))
    )

    return fitted_fwhm, amplitude, local_noise, residual_rms


def _refine_streak_from_image(
    image: np.ndarray,
    candidate: StreakCandidate,
    fwhm: float,
    width_tolerance: float = 0.5,
) -> StreakCandidate | None:
    """Validate a streak candidate by checking its perpendicular profile.

    The candidate angle comes from the filter-bank weighted mean (which
    integrates over the full trace and is the most reliable angle for
    faint streaks).  This function does NOT change the angle — it only
    validates that the perpendicular width is consistent with the PSF FWHM
    and that the signal amplitude exceeds the local noise.

    Returns the candidate (with updated width), or *None* if validation fails.
    """
    cx, cy = candidate.x, candidate.y
    h, w = image.shape

    # ---- Validate perpendicular profile at the trace angle -------------
    best_angle = candidate.angle_deg

    # Fit profile at the trace angle, with ±3° fallback for faint streaks
    # where the filter-bank angle may be slightly off.
    best_fit = _fit_perp_profile(image, cx, cy, best_angle, fwhm)

    # If the primary fit fails or has poor width, try ±3° offsets
    for offset in [-3, 3]:
        test_angle = (best_angle + offset) % 180
        result = _fit_perp_profile(image, cx, cy, float(test_angle), fwhm)
        if result is None:
            continue
        fitted_fwhm_test, _, _, residual_rms = result
        if abs(fitted_fwhm_test - fwhm) / fwhm > width_tolerance:
            continue
        if best_fit is None or residual_rms < best_fit[3]:
            best_fit = result

    if best_fit is None:
        logger.debug(
            "Rejected streak at (%.0f,%.0f): no valid profile fit at any angle",
            cx, cy,
        )
        return None

    fitted_fwhm, fitted_amp, local_noise, _ = best_fit

    # Validate amplitude above noise
    if local_noise > 0 and fitted_amp < 3 * local_noise:
        logger.debug(
            "Rejected streak at (%.0f,%.0f): amp=%.2f < 3*noise=%.2f",
            cx, cy, fitted_amp, 3 * local_noise,
        )
        return None

    # ---- Along-streak profile (refines length) -------------------------
    angle_rad = np.radians(best_angle)
    along_half = int(max(candidate.length_pixels, 4 * fwhm))
    t_along = np.arange(-along_half, along_half + 1, dtype=np.float64)
    ax_x = cx + t_along * np.cos(angle_rad)
    ax_y = cy + t_along * np.sin(angle_rad)

    valid_a = (ax_x >= 0) & (ax_x < w - 1) & (ax_y >= 0) & (ax_y < h - 1)
    if valid_a.sum() >= 5:
        along_profile = map_coordinates(image, [ax_y[valid_a], ax_x[valid_a]], order=1)
        t_along_valid = t_along[valid_a]
        bg_level = np.median(along_profile)
        peak_level = along_profile.max()
        half_max = bg_level + 0.5 * (peak_level - bg_level)
        above_half = t_along_valid[along_profile > half_max]
        refined_length = (
            float(above_half[-1] - above_half[0]) if len(above_half) >= 2
            else candidate.length_pixels
        )
    else:
        refined_length = candidate.length_pixels

    candidate.angle_deg = best_angle
    candidate.width_pixels = float(fitted_fwhm)
    # Keep the traced length if along-streak refinement gives shorter —
    # the trace in the directional-excess map integrates more signal and
    # is more reliable for faint streaks than a single profile cut.
    candidate.length_pixels = float(max(refined_length, candidate.length_pixels))
    return candidate


# ---------------------------------------------------------------------------
# Star pre-masking
# ---------------------------------------------------------------------------


def _build_star_mask(
    shape: tuple[int, int],
    star_positions: list[tuple[float, float]],
    mask_radius: float,
) -> np.ndarray:
    """Build a boolean mask that is True where stars are located.

    Uses vectorized distance computation for efficiency.
    """
    if not star_positions:
        return np.zeros(shape, dtype=bool)

    star_mask = np.zeros(shape, dtype=bool)
    h, w = shape
    mask_radius_sq = mask_radius ** 2

    # Vectorized: for each star, mask a square region and check circular distance
    ir = int(np.ceil(mask_radius))
    for sx, sy in star_positions:
        y_lo = max(0, int(sy) - ir)
        y_hi = min(h, int(sy) + ir + 1)
        x_lo = max(0, int(sx) - ir)
        x_hi = min(w, int(sx) + ir + 1)
        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star_mask[y_lo:y_hi, x_lo:x_hi] |= dist_sq <= mask_radius_sq

    return star_mask


# ---------------------------------------------------------------------------
# PSF subtraction of catalog stars
# ---------------------------------------------------------------------------


def _subtract_catalog_stars(
    image: np.ndarray,
    catalog_stars: list,
    fwhm: float,
) -> np.ndarray:
    """Subtract Gaussian PSF models of catalog stars from the image.

    For each catalog star, fits the amplitude of a Gaussian PSF at the
    star's known position and subtracts it.  This removes the dominant
    star signal while preserving any streak signal passing through or
    near the star.

    The amplitude is estimated by matched filtering: the dot product of
    the image patch with the normalized PSF template, which is the
    maximum-likelihood amplitude estimate for known position and shape.

    Returns the image with stars subtracted.
    """
    sigma = fwhm / 2.355
    result = image.copy()
    h, w = image.shape

    # Build the PSF template (normalized to sum=1)
    half = int(np.ceil(3.5 * fwhm))
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    psf /= psf.sum()
    psf_sq_sum = np.sum(psf ** 2)

    n_subtracted = 0
    for star in catalog_stars:
        if star.x is None or star.y is None:
            continue

        ix, iy = int(round(star.x)), int(round(star.y))

        # Skip stars too close to the edge
        if iy - half < 0 or iy + half >= h or ix - half < 0 or ix + half >= w:
            continue

        # Extract patch at star position
        patch = result[iy - half : iy + half + 1, ix - half : ix + half + 1]

        # Maximum-likelihood amplitude: (patch · psf) / (psf · psf)
        amplitude = np.sum(patch * psf) / psf_sq_sum

        # Only subtract positive stars (negative amplitude = no star)
        if amplitude > 0:
            # Subpixel centering: shift the PSF by the fractional offset
            dx = star.x - ix
            dy = star.y - iy
            shifted_psf = np.exp(
                -((x - dx) ** 2 + (y - dy) ** 2) / (2 * sigma ** 2)
            )
            shifted_psf /= shifted_psf.sum()

            result[iy - half : iy + half + 1, ix - half : ix + half + 1] -= (
                amplitude * shifted_psf
            )
            n_subtracted += 1

    logger.info(
        "PSF-subtracted %d catalog stars (FWHM=%.2f, template=%dx%d)",
        n_subtracted, fwhm, 2 * half + 1, 2 * half + 1,
    )
    return result


# ---------------------------------------------------------------------------
# Streak tracing from hotspots
# ---------------------------------------------------------------------------


def _find_hotspots(
    directional_excess: np.ndarray,
    threshold: float,
    min_separation: float,
) -> list[tuple[int, int, float]]:
    """Find local maxima in directional excess above threshold.

    Returns list of (y, x, excess_value) sorted by excess descending.
    """
    size = max(3, int(2 * min_separation + 1))
    local_max = maximum_filter(directional_excess, size=size, mode="constant")
    peaks = (
        (directional_excess == local_max)
        & (directional_excess > threshold)
    )
    ys, xs = np.where(peaks)
    vals = directional_excess[ys, xs]
    # Sort by excess descending
    order = np.argsort(-vals)
    return [(int(ys[i]), int(xs[i]), float(vals[i])) for i in order]


def _trace_streak_profile(
    directional_excess: np.ndarray,
    hotspot_x: float,
    hotspot_y: float,
    angle_deg: float,
    fwhm: float,
    threshold: float,
) -> tuple[float, float, float, float, float] | None:
    """Trace along the best angle from a hotspot to find streak endpoints.

    Extracts a 1D profile along the streak direction from the directional
    excess map (which already integrates perpendicular to the streak —
    the "squish").  Finds endpoints where the profile drops below threshold.

    Returns ``(centroid_x, centroid_y, length, peak_excess, total_excess)``
    or None.  ``total_excess`` is the sum of DE values along the trace,
    used for comparing trace quality across directions.
    """
    h, w = directional_excess.shape
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Limit trace extent.  Streaks longer than half the image diagonal are
    # extremely rare; using the full half-image as extent creates unnecessarily
    # large profiles (thousands of samples) that slow map_coordinates.
    max_extent = min(int(max(h, w) / 2), 500)

    # Sample along streak direction at 0.5-pixel steps for sub-pixel accuracy
    t_values = np.arange(-max_extent, max_extent + 0.5, 0.5)
    sx = hotspot_x + t_values * cos_a
    sy = hotspot_y + t_values * sin_a

    valid = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
    if valid.sum() < 5:
        return None

    profile = map_coordinates(directional_excess, [sy[valid], sx[valid]], order=1)
    t_valid = t_values[valid]

    # Find the contiguous region above threshold that contains t=0 (the hotspot)
    above = profile > threshold
    if not above.any():
        return None

    # Find the index closest to t=0
    center_idx = np.argmin(np.abs(t_valid))
    if not above[center_idx]:
        return None

    # Walk outward from center to find endpoints
    left_idx = center_idx
    while left_idx > 0 and above[left_idx - 1]:
        left_idx -= 1

    right_idx = center_idx
    while right_idx < len(above) - 1 and above[right_idx + 1]:
        right_idx += 1

    t_left = t_valid[left_idx]
    t_right = t_valid[right_idx]
    length = t_right - t_left

    if length < 1.0:
        return None

    # Use geometric midpoint of the trace, not flux-weighted centroid.
    # Flux weighting biases toward streak endpoints where the directional
    # excess can be higher due to edge effects, but the profile fit at
    # endpoints is unreliable (asymmetric signal drop-off).
    t_centroid = (t_left + t_right) / 2

    centroid_x = hotspot_x + t_centroid * cos_a
    centroid_y = hotspot_y + t_centroid * sin_a
    peak_excess = float(profile[center_idx])
    total_excess = float(profile[left_idx : right_idx + 1].sum())

    return float(centroid_x), float(centroid_y), float(length), peak_excess, total_excess


def _weighted_circular_mean_angle(
    angles_deg: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted circular mean of angles in [0, 180) — handles 0/180 wraparound."""
    rads = np.radians(2 * angles_deg)  # Double to map [0,180) → [0,360)
    mean_sin = np.sum(weights * np.sin(rads)) / np.sum(weights)
    mean_cos = np.sum(weights * np.cos(rads)) / np.sum(weights)
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) / 2) % 180


def _trace_and_build_candidate(
    image: np.ndarray,
    directional_excess: np.ndarray,
    best_angle_deg: np.ndarray,
    fractional_excess: np.ndarray,
    hotspot_y: int,
    hotspot_x: int,
    fwhm: float,
    noise_std: float,
    threshold: float,
    starfield: StarField | None = None,
    exposure_time: float | None = None,
) -> StreakCandidate | None:
    """Trace a streak from a hotspot and build a StreakCandidate.

    Uses the best angle from the filter bank to trace along the streak
    direction in the directional excess map, finding endpoints where the
    signal drops below threshold.  Then refines the angle by computing
    a weighted circular mean of the filter-bank angles over the traced
    pixels, which is more robust than a single-point measurement.
    """
    initial_angle = float(best_angle_deg[hotspot_y, hotspot_x]) % 180

    result = _trace_streak_profile(
        directional_excess,
        float(hotspot_x),
        float(hotspot_y),
        initial_angle,
        fwhm,
        threshold,
    )
    if result is None:
        return None

    # After the initial trace, try the perpendicular angle from the CENTROID.
    # This catches faint streaks where preprocessing artifacts dominate the
    # per-pixel best_angle: the initial trace follows the artifact but still
    # finds the correct centroid.  Re-tracing perpendicular from the centroid
    # then finds the real streak.
    centroid_x_init, centroid_y_init = result[0], result[1]
    perp_angle = (initial_angle + 90) % 180
    used_perpendicular = False
    perp_result = _trace_streak_profile(
        directional_excess,
        centroid_x_init,
        centroid_y_init,
        perp_angle,
        fwhm,
        threshold,
    )
    # Compare TOTAL integrated DE, not trace length.  A real streak has
    # high DE along its full extent; an artifact or noise trace may be
    # longer but with lower integrated signal.  Comparing total_excess
    # prevents star-mask-truncated traces from losing to noise traces
    # that happen to be longer.
    if perp_result is not None and perp_result[4] > result[4]:
        result = perp_result
        initial_angle = perp_angle
        used_perpendicular = True

    centroid_x, centroid_y, length, peak_excess, _total = result

    # Refine the angle using the filter-bank angles over the traced extent.
    # Skip this when the perpendicular retrace was used — the per-pixel
    # best_angle values are dominated by the artifact that caused the wrong
    # initial angle, so the weighted mean would undo the correction.
    h, w = directional_excess.shape
    angle_rad = np.radians(initial_angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    half_len = length / 2

    # Use the DETECTION threshold (not the lower trace threshold) for
    # angle sampling.  Noise pixels at the streak edges have random
    # best_angle values that contaminate the weighted mean.  Only pixels
    # with reliable directional excess should contribute to the angle.
    if not used_perpendicular:
        # Refine angle from per-pixel best_angle along the trace.
        # Only when the initial angle was used — if perpendicular retrace
        # was needed, the per-pixel angles are unreliable (dominated by
        # the artifact that caused the wrong initial direction).
        angle_threshold = noise_std * 5.0
        sample_angles = []
        sample_weights = []
        for t in np.arange(-half_len, half_len + 0.5, 1.0):
            px = int(round(centroid_x + t * cos_a))
            py = int(round(centroid_y + t * sin_a))
            if 0 <= px < w and 0 <= py < h:
                de = directional_excess[py, px]
                if de > angle_threshold:
                    sample_angles.append(best_angle_deg[py, px])
                    sample_weights.append(de)

        if len(sample_angles) >= 3:
            angle = _weighted_circular_mean_angle(
                np.array(sample_angles), np.array(sample_weights)
            )
        else:
            angle = initial_angle
    else:
        angle = initial_angle

    peak_snr = peak_excess / noise_std if noise_std > 0 else 0.0
    peak_frac = float(fractional_excess[hotspot_y, hotspot_x])

    candidate = StreakCandidate(
        x=centroid_x,
        y=centroid_y,
        angle_deg=angle,
        length_pixels=length,
        width_pixels=fwhm,  # Will be refined by profile fitting
        peak_snr=peak_snr,
        directional_excess=peak_excess,
        fractional_excess=peak_frac,
        n_pixels=int(length * fwhm),  # Approximate
    )

    # Sky coordinates
    if starfield and starfield.wcs:
        try:
            wcs = starfield.wcs.to_astropy_wcs()
            sky = wcs.pixel_to_world(centroid_x, centroid_y)
            candidate.ra = float(sky.ra.deg)
            candidate.dec = float(sky.dec.deg)
        except Exception:
            pass

    # Rate estimate
    if exposure_time and exposure_time > 0:
        rate_pix = length / exposure_time
        candidate.rate_pixels_per_sec = rate_pix
        if (
            starfield
            and starfield.wcs_metadata
            and hasattr(starfield.wcs_metadata, "x_ifov_arcsec")
        ):
            candidate.rate_arcsec_per_sec = rate_pix * starfield.wcs_metadata.x_ifov_arcsec

    return candidate


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_streaks_in_sidereal(
    image: np.ndarray,
    starfield: StarField,
    *,
    detection_sigma: float = 5.0,
    min_fractional_excess: float = 0.5,
    n_angles: int = 36,
    filter_length_fwhm: float = 5.0,
    min_length_fwhm: float = 2.0,
    min_component_pixels: int = 10,
    exposure_time: float | None = None,
) -> tuple[list[StreakCandidate], np.ndarray]:
    """Detect streak candidates in a sidereal frame.

    Applies a bank of directional matched filters directly to the image.
    Point sources produce equal response at all angles; subtracting the
    isotropic (mean) response nulls them.  A *fractional excess* threshold
    further separates streaks from stars regardless of brightness.

    Args:
        image: Sidereal image data (2D array).
        starfield: Solved ``StarField`` with WCS, FWHM stats.
        detection_sigma: Threshold for absolute directional excess.
        min_fractional_excess: Minimum fractional excess (excess / isotropic).
            Stars are ~0.05, streaks are >0.3.
        n_angles: Number of angles in the filter bank.
        filter_length_fwhm: Filter length as multiple of FWHM.
        min_length_fwhm: Reject candidates shorter than this many FWHMs.
        min_component_pixels: Skip connected components smaller than this.
        exposure_time: Exposure time in seconds (enables rate estimation).

    Returns:
        ``(candidates, directional_excess_image)``
    """
    # ---- FWHM ----------------------------------------------------------
    if starfield.detection_metadata and starfield.detection_metadata.pixel_fwhm:
        fwhm = starfield.detection_metadata.pixel_fwhm
    else:
        fwhm = 4.0
        logger.warning("No FWHM in starfield, using default %.1f", fwhm)

    # ---- 1. Background subtract ----------------------------------------
    _, bg_median, _ = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    bg_subtracted = image.astype(np.float64) - bg_median

    # ---- 1b. PSF-subtract catalog stars --------------------------------
    # Instead of relying solely on masking (which destroys streak signal
    # near stars), subtract a Gaussian PSF model at each catalog star
    # position.  This preserves any streak signal passing through or near
    # the star while removing the dominant star contribution.
    if starfield.catalog_stars:
        bg_subtracted = _subtract_catalog_stars(
            bg_subtracted, starfield.catalog_stars, fwhm,
        )

    # ---- 2. Directional matched filter bank ----------------------------
    logger.info(
        "Applying %d-angle directional filter bank (FWHM=%.2f, length=%.1fxFWHM)",
        n_angles,
        fwhm,
        filter_length_fwhm,
    )
    directional_excess, best_angle_deg, isotropic = apply_directional_filters(
        bg_subtracted, fwhm, n_angles, filter_length_fwhm
    )

    # Noise estimates
    _, _, excess_noise = sigma_clipped_stats(directional_excess, sigma=3.0, maxiters=5)
    _, _, iso_noise = sigma_clipped_stats(isotropic, sigma=3.0, maxiters=5)
    logger.info("Directional excess noise sigma = %.4f", excess_noise)

    # Fractional excess: how much the peak exceeds isotropic, as a fraction
    # of the isotropic level.  Stars ~0.05, streaks >>0.3.
    # Use iso_noise as floor to avoid division by zero in background regions.
    fractional_excess = directional_excess / np.maximum(np.abs(isotropic), iso_noise)

    # ---- 3. Pre-mask known star regions --------------------------------
    # The directional filter kernel extends filter_length_fwhm/2 * FWHM in
    # each direction.  Bright stars create spurious directional excess in
    # their wings out to the kernel half-extent.  Mask these regions BEFORE
    # thresholding to prevent false positives.
    #
    # IMPORTANT: Only use catalog stars (known astronomical objects), NOT
    # detections from the point-source finder.  Detections may include
    # streak peaks, and masking those would remove the very targets we
    # are trying to find.
    star_positions = []
    if starfield.catalog_stars:
        for s in starfield.catalog_stars:
            if s.x is not None and s.y is not None:
                star_positions.append((s.x, s.y))

    # Mask radius for subtraction residuals.  Since catalog stars are now
    # PSF-subtracted before filtering, the mask only needs to cover
    # subtraction residuals (imperfect PSF model, saturation, etc.).
    # Use a smaller radius than before to preserve streak signal near stars.
    star_mask_radius = fwhm * 3.0
    star_mask = _build_star_mask(image.shape, star_positions, star_mask_radius)
    n_masked = int(star_mask.sum())

    # Zero out directional excess at star locations
    directional_excess[star_mask] = 0.0
    fractional_excess[star_mask] = 0.0
    logger.info(
        "Pre-masked %d pixels near %d known stars (radius=%.1f px)",
        n_masked,
        len(star_positions),
        star_mask_radius,
    )

    # ---- 4. Threshold --------------------------------------------------
    border = int(np.ceil(fwhm * filter_length_fwhm / 2 + 3 * fwhm / 2.355))
    border_mask = np.ones(image.shape, dtype=bool)
    border_mask[:border, :] = False
    border_mask[-border:, :] = False
    border_mask[:, :border] = False
    border_mask[:, -border:] = False

    # Require actual signal present (not just noise).  Background regions
    # have isotropic ~0, so fractional excess blows up there.  Requiring
    # isotropic > 3*iso_noise ensures we only look where there IS signal.
    iso_signal_threshold = 3 * iso_noise

    abs_threshold = detection_sigma * excess_noise
    detection_mask = (
        (directional_excess > abs_threshold)          # above noise floor
        & (fractional_excess > min_fractional_excess)  # not a point source
        & (isotropic > iso_signal_threshold)           # has real signal
        & border_mask
    )
    n_det = int(detection_mask.sum())
    logger.info(
        "Detection mask: %d pixels (excess>%.2f, frac>%.2f, iso>%.2f)",
        n_det,
        abs_threshold,
        min_fractional_excess,
        iso_signal_threshold,
    )

    if n_det == 0:
        return [], directional_excess

    # ---- 5. Find hotspots and trace streaks ----------------------------
    # Instead of connected-component PCA (which fails on faint/fragmented
    # streaks), find local maxima in directional_excess and trace along
    # the best filter-bank angle.  The directional excess map already
    # integrates perpendicular to the streak (the "squish"), so tracing
    # along it gives the full streak extent with high SNR.
    hotspots = _find_hotspots(
        directional_excess * detection_mask,
        abs_threshold,
        min_separation=fwhm * 2,
    )
    # Cap hotspots to avoid spending time tracing noise peaks.
    # Hotspots are sorted by DE descending so the strongest come first.
    MAX_HOTSPOTS = 150
    if len(hotspots) > MAX_HOTSPOTS:
        logger.info("Capping hotspots from %d to %d", len(hotspots), MAX_HOTSPOTS)
        hotspots = hotspots[:MAX_HOTSPOTS]
    logger.info("Found %d hotspots above threshold", len(hotspots))

    # Use the detection threshold for tracing.  A lower threshold extends
    # the trace into noise, giving random angles and wrong centroid/length.
    trace_threshold = abs_threshold

    # Track which hotspots have been claimed (avoid duplicate detections
    # of the same streak from multiple hotspots along its length)
    claimed_mask = np.zeros(image.shape, dtype=bool)

    candidates: list[StreakCandidate] = []
    n_rejected_short = 0
    n_rejected_duplicate = 0
    min_length = fwhm * min_length_fwhm

    for hy, hx, _ in hotspots:
        # Skip if this hotspot was already claimed by a brighter streak
        if claimed_mask[hy, hx]:
            n_rejected_duplicate += 1
            continue

        candidate = _trace_and_build_candidate(
            image=bg_subtracted,
            directional_excess=directional_excess,
            best_angle_deg=best_angle_deg,
            fractional_excess=fractional_excess,
            hotspot_y=hy,
            hotspot_x=hx,
            fwhm=fwhm,
            noise_std=excess_noise,
            threshold=trace_threshold,
            starfield=starfield,
            exposure_time=exposure_time,
        )
        if candidate is None:
            continue

        if candidate.length_pixels < min_length:
            n_rejected_short += 1
            continue

        # Mark the streak region as claimed to prevent duplicate detections.
        # Step at 2-pixel intervals (sufficient for claim_radius overlap).
        angle_rad = np.radians(candidate.angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        half_len = candidate.length_pixels / 2
        claim_radius = fwhm * 2
        ir = int(claim_radius)
        for t in np.arange(-half_len, half_len + 1, 2.0):
            px = int(round(candidate.x + t * cos_a))
            py = int(round(candidate.y + t * sin_a))
            if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                y_lo = max(0, py - ir)
                y_hi = min(image.shape[0], py + ir + 1)
                x_lo = max(0, px - ir)
                x_hi = min(image.shape[1], px + ir + 1)
                claimed_mask[y_lo:y_hi, x_lo:x_hi] = True

        candidates.append(candidate)

    # ---- 5b. Connected-component fallback for faint streaks --------------
    # The hotspot approach can miss faint streaks when preprocessing artifacts
    # dominate the per-pixel best_angle (the hotspot traces along the artifact
    # direction instead of the streak).  As a fallback, find above-threshold
    # connected components that weren't captured by any hotspot candidate.
    # Use PCA on the component shape to determine the angle — this is robust
    # because the streak creates a vertically/horizontally elongated detection
    # region regardless of per-pixel angle noise.
    n_component_candidates = 0
    labeled, n_labels = label(detection_mask & ~claimed_mask)
    # Cap the number of components to examine to avoid expensive scanning
    # of hundreds of noise blobs.  The most important components are the
    # larger ones; small components are filtered by min_component_pixels anyway.
    MAX_FALLBACK_COMPONENTS = 50
    for comp_id in range(1, min(n_labels + 1, MAX_FALLBACK_COMPONENTS + 1)):
        comp_ys, comp_xs = np.where(labeled == comp_id)
        n_pix = len(comp_ys)
        if n_pix < min_component_pixels:
            continue

        # PCA on the component pixel coordinates
        cx_mean = comp_xs.mean()
        cy_mean = comp_ys.mean()
        dx = comp_xs - cx_mean
        dy = comp_ys - cy_mean
        cov = np.array([[np.sum(dx * dx), np.sum(dx * dy)],
                        [np.sum(dx * dy), np.sum(dy * dy)]]) / n_pix
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Eigenvectors sorted ascending; last = major axis
        major = eigvecs[:, -1]
        pca_angle = float(np.degrees(np.arctan2(major[1], major[0]))) % 180

        # Require elongation (aspect ratio > 2:1)
        if eigvals[0] <= 0 or np.sqrt(eigvals[1] / eigvals[0]) < 2.0:
            continue

        # Length from the extent along the major axis
        projections = dx * major[0] + dy * major[1]
        pca_length = projections.max() - projections.min()
        if pca_length < min_length:
            continue

        # Trace from the component centroid at the PCA angle
        result = _trace_streak_profile(
            directional_excess,
            float(cx_mean),
            float(cy_mean),
            pca_angle,
            fwhm,
            trace_threshold,
        )
        if result is None or result[2] < min_length:
            continue

        candidate = _characterize_component(
            bg_subtracted,
            labeled == comp_id,
            best_angle_deg,
            directional_excess,
            fractional_excess,
            excess_noise,
            fwhm,
            starfield=starfield,
            exposure_time=exposure_time,
        )
        if candidate is not None:
            # Override angle with PCA angle (more reliable than per-pixel)
            candidate.angle_deg = pca_angle
            # Use trace centroid and length
            candidate.x = result[0]
            candidate.y = result[1]
            candidate.length_pixels = max(candidate.length_pixels, result[2])

            # Check not duplicate of existing candidates
            is_dup = False
            for existing in candidates:
                d = np.sqrt((candidate.x - existing.x) ** 2 + (candidate.y - existing.y) ** 2)
                if d < fwhm * 3:
                    is_dup = True
                    break
            if not is_dup:
                candidates.append(candidate)
                n_component_candidates += 1

    if n_component_candidates > 0:
        logger.info("Connected-component fallback added %d candidates", n_component_candidates)

    # ---- 6. Refine candidates against the original image ---------------
    # Go back to the actual image, fit the perpendicular cross-section,
    # and validate that the width matches the PSF FWHM.
    n_rejected_profile = 0
    refined = []
    for candidate in candidates:
        result = _refine_streak_from_image(bg_subtracted, candidate, fwhm)
        if result is not None:
            refined.append(result)
        else:
            n_rejected_profile += 1
    candidates = refined

    # Sort by SNR but return ALL candidates (no cap)
    candidates.sort(key=lambda c: c.peak_snr, reverse=True)
    logger.info(
        "Detected %d streak candidates (%d rejected short, "
        "%d rejected profile, %d duplicate hotspots, from %d hotspots)",
        len(candidates),
        n_rejected_short,
        n_rejected_profile,
        n_rejected_duplicate,
        len(hotspots),
    )

    return candidates, directional_excess, best_angle_deg


# ---------------------------------------------------------------------------
# Streak photometry
# ---------------------------------------------------------------------------


def measure_streak_candidate_photometry(
    image,
    candidates: list[StreakCandidate],
    zero_point: float,
    zero_point_err: float | None = None,
    exposure_time: float | None = None,
    fwhm: float = 4.0,
    gain: float = 1.0,
    read_noise: float = 0.0,
    multiband_calibration=None,
    observation_filter: str | None = None,
) -> None:
    """Measure photometry for streak candidates using rectangular apertures.

    Uses an oriented rectangular aperture aligned with the streak direction.
    Width = 2 x FWHM (captures the PSF cross-section), length = measured
    streak length + FWHM (captures full extent including PSF wings at ends).
    Background measured from a larger rectangular annulus.

    Updates each candidate in-place with flux, SNR, and magnitudes.
    """
    from photutils.aperture import RectangularAnnulus, RectangularAperture, aperture_photometry

    if exposure_time is None or exposure_time <= 0:
        exposure_time = 1.0

    image_data = image.data if hasattr(image, "data") else image

    for candidate in candidates:
        cx, cy = candidate.x, candidate.y
        h, w = image_data.shape

        # Skip if too close to edge
        margin = max(candidate.length_pixels, 5 * fwhm)
        if cx < margin or cx > w - margin or cy < margin or cy > h - margin:
            continue

        # Aperture dimensions
        ap_length = candidate.length_pixels + fwhm
        ap_width = 2 * fwhm
        # photutils angle convention: measured from +x axis, CCW
        theta = np.radians(candidate.angle_deg)

        aperture = RectangularAperture(
            (cx, cy), w=ap_length, h=ap_width, theta=theta
        )
        bg_annulus = RectangularAnnulus(
            (cx, cy),
            w_in=ap_length + fwhm,
            w_out=ap_length + 4 * fwhm,
            h_out=6 * fwhm,
            h_in=3 * fwhm,
            theta=theta,
        )

        try:
            phot = aperture_photometry(image_data, [aperture, bg_annulus])
        except Exception:
            logger.debug("Aperture photometry failed for streak at (%.0f, %.0f)", cx, cy)
            continue

        flux_ap = float(phot["aperture_sum_0"][0])
        flux_bg = float(phot["aperture_sum_1"][0])

        bg_area = bg_annulus.area
        ap_area = aperture.area
        bg_per_pixel = flux_bg / bg_area if bg_area > 0 else 0
        flux = flux_ap - bg_per_pixel * ap_area

        # Noise model (same as point source detection photometry)
        n_pix = ap_area
        source_e = max(flux, 0) * gain
        bg_e = max(bg_per_pixel, 0) * gain
        noise_e = np.sqrt(source_e + bg_e * n_pix + read_noise ** 2 * n_pix)
        flux_err = noise_e / gain if gain > 0 else 0

        candidate.flux = float(flux)
        candidate.flux_err = float(flux_err)
        candidate.peak_snr = float(flux / flux_err) if flux_err > 0 else 0.0

        if flux > 0:
            flux_per_sec = flux / exposure_time
            inst_mag = -2.5 * np.log10(flux_per_sec)
            candidate.instrumental_magnitude = float(inst_mag)

            mag_err_flux = 1.0857 * flux_err / flux
            zp_err = zero_point_err if zero_point_err else 0.0
            mag_err = float(np.sqrt(mag_err_flux ** 2 + zp_err ** 2))

            # Calibrated magnitude from zero point
            cal_mag = float(inst_mag + zero_point)

            if multiband_calibration is not None:
                # Use per-band zero points (no color term for unknown objects)
                mags, errs = {}, {}
                for band_name, band_cal in multiband_calibration.bands.items():
                    mags[band_name] = float(inst_mag + band_cal.zero_point)
                    band_err = float(np.sqrt(mag_err_flux ** 2 + band_cal.zero_point_err ** 2))
                    errs[band_name] = band_err
                candidate.calibrated_magnitudes = mags
                candidate.magnitude_errs = errs
            else:
                band = observation_filter or "instrumental"
                candidate.calibrated_magnitudes = {band: cal_mag}
                candidate.magnitude_errs = {band: mag_err}

            candidate.observation_filter = observation_filter

    n_measured = sum(1 for c in candidates if c.flux is not None)
    logger.info("Measured photometry for %d/%d streak candidates", n_measured, len(candidates))
