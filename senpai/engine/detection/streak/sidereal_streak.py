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
    from scipy.fft import irfft2, next_fast_len, rfft2

    kernels, angles = build_directional_filter_bank(fwhm, n_angles, filter_length_fwhm)

    # float32 halves FFT cost; response precision is bounded by image noise,
    # not arithmetic precision.
    image = np.ascontiguousarray(image, dtype=np.float32)

    max_response = np.full(image.shape, -np.inf, dtype=np.float32)
    best_angle_idx = np.zeros(image.shape, dtype=np.int32)
    sum_response = np.zeros(image.shape, dtype=np.float32)

    k0 = kernels[0]
    # Exact linear-convolution padding (image + kernel - 1) is often a
    # numerically awful FFT length (e.g. 2124 = 4*3*177); rounding up to the
    # next fast composite size is a large constant-factor win.
    fft_shape = (
        next_fast_len(image.shape[0] + k0.shape[0] - 1, real=True),
        next_fast_len(image.shape[1] + k0.shape[1] - 1, real=True),
    )

    image_fft = rfft2(image, s=fft_shape, workers=-1)

    for i, kernel in enumerate(kernels):
        kernel_fft = rfft2(kernel.astype(np.float32), s=fft_shape, workers=-1)
        raw = irfft2(image_fft * kernel_fft, s=fft_shape, workers=-1)

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


def _bin_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-average ``image`` by ``factor`` (trims edges that don't divide)."""
    h, w = image.shape
    hb, wb = h // factor, w // factor
    return (
        image[: hb * factor, : wb * factor]
        .reshape(hb, factor, wb, factor)
        .mean(axis=(1, 3))
    )


def _upsample_map(binned: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample of a binned map back to ``shape``.

    Edge rows/cols trimmed by binning are zero-padded; they fall inside the
    detection border margin anyway.
    """
    full = np.kron(binned, np.ones((factor, factor), dtype=binned.dtype))
    pad_y = shape[0] - full.shape[0]
    pad_x = shape[1] - full.shape[1]
    if pad_y or pad_x:
        full = np.pad(full, ((0, pad_y), (0, pad_x)))
    return full


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

    # Validate the perpendicular width against the PSF (same band as
    # _characterize_component): a real streak is the PSF smeared along one
    # axis.  Far too narrow = pixel-grid noise artifact; far too wide =
    # glare gradient or halo, not a streak.
    if fitted_fwhm < 0.3 * fwhm or fitted_fwhm > 2.5 * fwhm:
        logger.debug(
            "Rejected streak at (%.0f,%.0f): width=%.2f outside [0.3,2.5]*fwhm=%.2f",
            cx, cy, fitted_fwhm, fwhm,
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
        peak_idx = int(np.argmax(along_profile))
        peak_level = along_profile[peak_idx]
        half_max = bg_level + 0.5 * (peak_level - bg_level)
        above_half = t_along_valid[along_profile > half_max]
        if len(above_half) >= 2:
            # Only count the gap-free cluster of above-half samples around
            # the peak.  A raw first-to-last span bridges disjoint bright
            # features along the cut (e.g. a star blob plus glare 100 px
            # away) into one absurd length.
            peak_t = t_along_valid[peak_idx]
            gaps = np.where(np.diff(above_half) > fwhm)[0]
            starts = np.concatenate(([0], gaps + 1))
            ends = np.concatenate((gaps, [len(above_half) - 1]))
            refined_length = candidate.length_pixels
            for s, e in zip(starts, ends):
                if above_half[s] <= peak_t <= above_half[e]:
                    refined_length = float(above_half[e] - above_half[s])
                    break
        else:
            refined_length = candidate.length_pixels
    else:
        refined_length = candidate.length_pixels

    candidate.angle_deg = best_angle
    candidate.width_pixels = float(fitted_fwhm)
    # The traced length is measured on the directional-excess map, which is
    # smeared by the matched-filter kernel (~filter length beyond the true
    # ends for bright streaks).  When the streak is strong the image-space
    # along-profile is the unsmeared measurement — trust it.  This also
    # collapses star-residual blobs (compact flux, long smeared trace) to
    # their true image extent so the minimum-length filter rejects them.
    # For faint streaks the half-max crossing is unreliable, so keep the
    # trace, which integrates more signal.
    if local_noise > 0 and fitted_amp > 5 * local_noise:
        candidate.length_pixels = float(refined_length)
    else:
        candidate.length_pixels = float(max(refined_length, candidate.length_pixels))
    return candidate


# ---------------------------------------------------------------------------
# Star pre-masking
# ---------------------------------------------------------------------------


def _build_adaptive_star_mask(
    shape: tuple[int, int],
    star_amplitudes: list[tuple[float, float, float, float]],
    fwhm: float,
    image_noise: float,
    template_peak: float,
    streak_angle_deg: float | None = None,
    streak_length_pixels: float | None = None,
) -> np.ndarray:
    """Mask each subtracted star out to where its PSF model was significant.

    A blanket per-star radius is catastrophic on deep catalogs (a mag-21
    Gaia catalog covers essentially the whole frame); instead each star's
    mask radius is derived from its fitted amplitude: the radius where the
    model dropped below a fraction of the image noise.  Stars too faint to
    leave residuals get no mask at all.  Very bright stars leave structured
    non-model residuals (halos, saturation bleed, aberrated wings) far
    beyond the model radius, so they get a generous fixed radius.

    On rate-track frames (``streak_angle_deg``/``streak_length_pixels``
    given) stars are trailed, and the mask is a capsule along the trail
    segment instead of a disk.

    Args:
        shape: Image shape.
        star_amplitudes: ``(x, y, fitted_amplitude, measured_peak)`` per
            subtracted star.
        fwhm: PSF FWHM in pixels.
        image_noise: Robust image noise sigma (ADU).
        template_peak: Peak value of the sum-normalized subtraction
            template (converts amplitudes to model peak pixel values).
        streak_angle_deg: Star-trail angle on rate frames.
        streak_length_pixels: Star-trail length on rate frames.

    Returns:
        Boolean mask, True where candidate seeding should be suppressed.
    """
    star_mask = np.zeros(shape, dtype=bool)
    if not star_amplitudes or image_noise <= 0:
        return star_mask

    sigma = fwhm / 2.355
    h, w = shape

    if streak_angle_deg is not None and streak_length_pixels:
        angle_rad = np.radians(streak_angle_deg)
        ux, uy = np.cos(angle_rad), np.sin(angle_rad)
        half_trail = streak_length_pixels / 2.0
    else:
        ux = uy = 0.0
        half_trail = 0.0

    for sx, sy, amp, measured_peak in star_amplitudes:
        # The measured peak catches saturated and aberrated stars whose
        # model fit badly underestimates their real brightness.
        peak_snr = max(amp * template_peak, measured_peak) / image_noise
        if peak_snr < 3.0:
            continue  # subtraction residual is below the noise
        # Radius where a Gaussian cross-section falls to 0.25 * noise
        radius = sigma * np.sqrt(2 * np.log(4.0 * peak_snr))
        if peak_snr > 40.0:
            # Bright: the model is wrong in the wings; halos, bleed and
            # aberration residuals extend far beyond the model radius and
            # mimic streak-shaped signal.
            radius = max(radius, 6.0 * fwhm)

        ir = int(np.ceil(radius + half_trail))
        y_lo = max(0, int(sy) - ir)
        y_hi = min(h, int(sy) + ir + 1)
        x_lo = max(0, int(sx) - ir)
        x_hi = min(w, int(sx) + ir + 1)
        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        if half_trail > 0:
            # Distance to the trail segment: clamp the along-trail
            # projection to +-half_trail, then measure to the closest point.
            t = np.clip((xx - sx) * ux + (yy - sy) * uy, -half_trail, half_trail)
            dist_sq = (xx - sx - t * ux) ** 2 + (yy - sy - t * uy) ** 2
        else:
            dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star_mask[y_lo:y_hi, x_lo:x_hi] |= dist_sq <= radius * radius

    return star_mask


# ---------------------------------------------------------------------------
# PSF subtraction of catalog stars
# ---------------------------------------------------------------------------


def _subtract_catalog_stars(
    image: np.ndarray,
    catalog_stars: list,
    fwhm: float,
    streak_angle_deg: float | None = None,
    streak_length_pixels: float | None = None,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], float]:
    """Subtract PSF models of catalog stars from the image.

    For each catalog star, fits the amplitude of a PSF template at the
    star's known position and subtracts it.  This removes the dominant
    star signal while preserving any streak signal passing through or
    near the star.

    On sidereal frames the template is a round Gaussian.  On rate-track
    frames (``streak_angle_deg``/``streak_length_pixels`` given) every star
    is trailed by the tracking motion, and the template is the
    corresponding streaked PSF — subtracting round Gaussians from trailed
    stars removes almost nothing and floods the filter bank with trail
    residuals.

    The amplitude is estimated by matched filtering: the dot product of
    the image patch with the normalized template, which is the
    maximum-likelihood amplitude estimate for known position and shape.

    Returns:
        ``(subtracted_image, star_amplitudes, template_peak)`` where
        ``star_amplitudes`` is a list of ``(x, y, fitted_amplitude,
        measured_peak)`` per subtracted star and ``template_peak`` is the
        peak value of the sum-normalized template (converts fitted
        amplitudes to model peak pixel values).
    """
    from senpai.engine.detection.kernels import streak_matched_kernel

    sigma = fwhm / 2.355
    result = image.copy()
    h, w = image.shape

    trailed = streak_angle_deg is not None and streak_length_pixels
    if trailed:
        # Streaked PSF template at the tracking angle (normalized to sum=1)
        psf = streak_matched_kernel(
            round(float(fwhm), 2),
            round(float(streak_angle_deg), 1),
            round(float(streak_length_pixels) / fwhm, 2),
        )
        half = psf.shape[0] // 2
        gauss_grid = None
    else:
        # Round Gaussian template (normalized to sum=1)
        half = int(np.ceil(3.5 * fwhm))
        y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
        psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        psf /= psf.sum()
        gauss_grid = (y, x)

    psf_sq_sum = np.sum(psf ** 2)
    template_peak = float(psf.max())

    star_amplitudes: list[tuple[float, float, float, float]] = []
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
            # Measured peak near the star center: for saturated or aberrated
            # (non-Gaussian) stars the fitted model peak badly underestimates
            # the real brightness, and the residual mask needs to know.
            core = patch[half - 2 : half + 3, half - 2 : half + 3]
            measured_peak = float(core.max())

            if gauss_grid is not None:
                # Subpixel centering: shift the PSF by the fractional offset
                gy, gx = gauss_grid
                dx = star.x - ix
                dy = star.y - iy
                model = np.exp(
                    -((gx - dx) ** 2 + (gy - dy) ** 2) / (2 * sigma ** 2)
                )
                model /= model.sum()
            else:
                # Streaked template: pixel-centered subtraction.  The trail
                # is many pixels long, so the sub-pixel offset leaves only a
                # small dipole residual, covered by the residual mask.
                model = psf

            result[iy - half : iy + half + 1, ix - half : ix + half + 1] -= (
                amplitude * model
            )
            star_amplitudes.append((star.x, star.y, float(amplitude), measured_peak))

    logger.info(
        "PSF-subtracted %d catalog stars (FWHM=%.2f, template=%dx%d%s)",
        len(star_amplitudes), fwhm, 2 * half + 1, 2 * half + 1,
        f", trailed L={streak_length_pixels:.0f}px" if trailed else "",
    )
    return result, star_amplitudes, template_peak


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
    signal drops below threshold.  Then refines the angle with a
    DE-weighted PCA of the local above-threshold region (resolving below
    the filter-bank angle step) and re-traces along the refined angle.
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

    centroid_x, centroid_y, length, peak_excess, _total = result

    # Refine the angle with a DE-weighted PCA of the above-threshold pixels
    # around the trace.  The filter bank quantizes angles (typically 5deg
    # steps) and the per-pixel best_angle is noisy where the kernel smears
    # signal beyond the streak ends; the excess-weighted second moment of
    # the actual detection region resolves the angle below the bank step
    # and is insensitive to both.
    h, w = directional_excess.shape
    half_len = length / 2

    r = int(np.ceil(half_len + fwhm))
    y_lo = max(0, int(centroid_y) - r)
    y_hi = min(h, int(centroid_y) + r + 1)
    x_lo = max(0, int(centroid_x) - r)
    x_hi = min(w, int(centroid_x) + r + 1)
    local_de = directional_excess[y_lo:y_hi, x_lo:x_hi]
    angle_threshold = noise_std * 5.0
    sel = local_de > angle_threshold

    angle = initial_angle
    if sel.sum() >= 5:
        ys, xs = np.nonzero(sel)
        wts = local_de[ys, xs]
        wsum = wts.sum()
        mx = (wts * xs).sum() / wsum
        my = (wts * ys).sum() / wsum
        dx = xs - mx
        dy = ys - my
        cxx = (wts * dx * dx).sum() / wsum
        cyy = (wts * dy * dy).sum() / wsum
        cxy = (wts * dx * dy).sum() / wsum
        eigvals, eigvecs = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
        # Require clear elongation, otherwise the PCA direction is noise
        if eigvals[0] > 0 and np.sqrt(eigvals[1] / eigvals[0]) > 1.5:
            major = eigvecs[:, -1]
            angle = float(np.degrees(np.arctan2(major[1], major[0]))) % 180

    # If PCA moved the angle appreciably, re-trace along the refined angle:
    # the original trace direction was off by up to half the bank step and
    # its length/centroid degrade with angle error over long streaks.
    angle_change = abs(angle - initial_angle) % 180
    angle_change = min(angle_change, 180 - angle_change)
    if angle_change > 1.0:
        retrace = _trace_streak_profile(
            directional_excess, centroid_x, centroid_y, angle, fwhm, threshold,
        )
        if retrace is not None and retrace[4] >= result[4]:
            centroid_x, centroid_y, length, peak_excess, _total = retrace

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
    exclude_angle_deg: float | None = None,
    exclude_length_pixels: float | None = None,
    exclude_angle_tol_deg: float = 15.0,
) -> tuple[list[StreakCandidate], np.ndarray, np.ndarray]:
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
        exclude_angle_deg: Star-trail angle on rate-track frames.  When set
            (with ``exclude_length_pixels``), catalog stars are subtracted
            with a TRAILED PSF template, masked along the trail capsule,
            and candidates matching the trail angle (within
            ``exclude_angle_tol_deg``) are dropped when their length is
            0.4-4x the trail length or a significant star lies on their
            axis — before the expensive profile refinement.
        exclude_length_pixels: Star-trail length paired with
            ``exclude_angle_deg``.
        exclude_angle_tol_deg: Angle tolerance for the exclusion.

    Returns:
        ``(candidates, directional_excess_image, best_angle_deg_image)``
    """
    # ---- FWHM ----------------------------------------------------------
    if starfield.detection_metadata and starfield.detection_metadata.pixel_fwhm:
        fwhm = starfield.detection_metadata.pixel_fwhm
    else:
        fwhm = 4.0
        logger.warning("No FWHM in starfield, using default %.1f", fwhm)

    # ---- 1. Background subtract ----------------------------------------
    # Stats on a 4x4-subsampled view: identical medians to within noise at
    # a fraction of the cost on multi-megapixel frames.
    _, bg_median, image_noise = sigma_clipped_stats(
        image[::4, ::4], sigma=3.0, maxiters=5
    )
    bg_subtracted = image.astype(np.float64) - bg_median

    # ---- 1b. PSF-subtract catalog stars --------------------------------
    # Instead of relying solely on masking (which destroys streak signal
    # near stars), subtract a Gaussian PSF model at each catalog star
    # position.  This preserves any streak signal passing through or near
    # the star while removing the dominant star contribution.
    # On rate frames the exclusion model doubles as the star-trail model:
    # stars are subtracted with a trailed template and masked along the
    # trail capsule.
    star_amplitudes: list[tuple[float, float, float, float]] = []
    sigma = fwhm / 2.355
    template_peak = 1.0 / (2 * np.pi * sigma**2)
    if starfield.catalog_stars:
        bg_subtracted, star_amplitudes, template_peak = _subtract_catalog_stars(
            bg_subtracted,
            starfield.catalog_stars,
            fwhm,
            streak_angle_deg=exclude_angle_deg,
            streak_length_pixels=exclude_length_pixels,
        )

    # ---- 2. Directional matched filter bank ----------------------------
    # The PSF oversamples the pixel grid on most sensors; block-averaging
    # before the filter bank keeps the matched-filter SNR (bin << FWHM)
    # while cutting FFT cost by ~bin_factor^2.  Maps are upsampled back to
    # native resolution afterwards so tracing and refinement are unchanged.
    bin_factor = max(1, int(fwhm / 3.5))
    logger.info(
        "Applying %d-angle directional filter bank "
        "(FWHM=%.2f, length=%.1fxFWHM, bin=%d)",
        n_angles,
        fwhm,
        filter_length_fwhm,
        bin_factor,
    )
    if bin_factor > 1:
        filter_input = _bin_image(bg_subtracted, bin_factor)
        filter_fwhm = fwhm / bin_factor
    else:
        filter_input = bg_subtracted
        filter_fwhm = fwhm

    directional_excess, best_angle_deg, isotropic = apply_directional_filters(
        filter_input, filter_fwhm, n_angles, filter_length_fwhm
    )

    # Noise estimates (on the compact maps, before upsampling)
    _, _, excess_noise = sigma_clipped_stats(directional_excess, sigma=3.0, maxiters=5)
    _, _, iso_noise = sigma_clipped_stats(isotropic, sigma=3.0, maxiters=5)
    logger.info("Directional excess noise sigma = %.4f", excess_noise)

    if bin_factor > 1:
        directional_excess = _upsample_map(directional_excess, bin_factor, image.shape)
        best_angle_deg = _upsample_map(best_angle_deg, bin_factor, image.shape)
        isotropic = _upsample_map(isotropic, bin_factor, image.shape)

    # Fractional excess: how much the peak exceeds isotropic, as a fraction
    # of the isotropic level.  Stars ~0.05, streaks >>0.3.
    # Use iso_noise as floor to avoid division by zero in background regions.
    fractional_excess = directional_excess / np.maximum(np.abs(isotropic), iso_noise)

    # ---- 3. Build the star-residual seed mask ---------------------------
    # Imperfect PSF subtraction leaves residuals that light up the filter
    # bank.  Suppress candidate SEEDING there, but do NOT zero the
    # directional excess map itself: a streak passing near a star must
    # still trace through the region with its full signal, otherwise the
    # trace truncates (or wanders off along a noise direction) and the
    # candidate dies in refinement.
    star_mask = _build_adaptive_star_mask(
        image.shape,
        star_amplitudes,
        fwhm,
        float(image_noise),
        template_peak,
        streak_angle_deg=exclude_angle_deg,
        streak_length_pixels=exclude_length_pixels,
    )
    logger.info(
        "Star-residual seed mask: %d pixels (%.1f%% of frame) from %d stars",
        int(star_mask.sum()),
        100.0 * star_mask.sum() / star_mask.size,
        len(star_amplitudes),
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
        & ~star_mask                                   # no star residual seeds
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
    n_rejected_border = 0
    n_rejected_excluded = 0
    min_length = fwhm * min_length_fwhm

    # Stars whose trail residuals can seed candidates (used by
    # _is_excluded_star_streak on rate frames).  Trailing spreads a star's
    # flux over the trail, so even mag ~12 stars only peak at ~10 sigma —
    # the threshold here is deliberately low.
    trail_stars = [
        (sx, sy)
        for sx, sy, amp, measured_peak in star_amplitudes
        if max(amp * template_peak, measured_peak) > 8.0 * float(image_noise)
    ]

    def _is_excluded_star_streak(candidate: StreakCandidate) -> bool:
        """Candidate is a star trail (rate frames): matches the tracking angle
        and either the approximate trail length or a star on its axis."""
        if exclude_angle_deg is None or not exclude_length_pixels:
            return False
        diff = abs(candidate.angle_deg - exclude_angle_deg) % 180
        if min(diff, 180 - diff) > exclude_angle_tol_deg:
            return False
        # Wide band: the traced length of a trail residual is smeared by the
        # matched-filter kernel (and halos on bright stars), so it can come
        # out several times the nominal trail length.
        ratio = candidate.length_pixels / exclude_length_pixels
        if 0.4 < ratio < 4.0:
            return True
        # If a significant star sits on the candidate's axis, it IS that
        # star's trail regardless of the measured length.
        angle_rad = np.radians(candidate.angle_deg)
        ux, uy = np.cos(angle_rad), np.sin(angle_rad)
        reach = candidate.length_pixels / 2 + 6.0 * fwhm
        for sx, sy in trail_stars:
            dx = sx - candidate.x
            dy = sy - candidate.y
            along = dx * ux + dy * uy
            perp = abs(-dx * uy + dy * ux)
            if abs(along) <= reach and perp <= 1.5 * fwhm:
                return True
        return False

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

        # The trace can slide the centroid off the seed hotspot and into the
        # border zone (edge glare from bright sources just off-frame produces
        # strong directional gradients there).  Signal centered that close to
        # the edge cannot be validated; drop it.
        if (
            candidate.x < border
            or candidate.x >= image.shape[1] - border
            or candidate.y < border
            or candidate.y >= image.shape[0] - border
        ):
            n_rejected_border += 1
            continue

        # Mark the streak region as claimed to prevent duplicate detections.
        # Step at 2-pixel intervals (sufficient for claim_radius overlap).
        # Excluded star streaks claim their region too, so the remaining
        # hotspots along the same star streak are skipped cheaply.
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

        # On rate frames every star is a streak at the tracking angle/length;
        # drop those before the expensive profile refinement.
        if _is_excluded_star_streak(candidate):
            n_rejected_excluded += 1
            continue

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

            if _is_excluded_star_streak(candidate):
                n_rejected_excluded += 1
                continue

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
        # Refinement updates the length from the image profile; re-check the
        # minimum length so kernel-smear-only detections don't survive.
        if result is None or result.length_pixels < min_length:
            n_rejected_profile += 1
            continue
        # Refinement also updates the angle/length, so re-check the star
        # streak exclusion on rate frames.
        if _is_excluded_star_streak(result):
            n_rejected_excluded += 1
            continue
        # Rate estimates were derived from the kernel-smeared traced length;
        # recompute from the refined length.
        if exposure_time and exposure_time > 0:
            result.rate_pixels_per_sec = result.length_pixels / exposure_time
            if (
                starfield.wcs_metadata
                and hasattr(starfield.wcs_metadata, "x_ifov_arcsec")
            ):
                result.rate_arcsec_per_sec = (
                    result.rate_pixels_per_sec * starfield.wcs_metadata.x_ifov_arcsec
                )
        refined.append(result)
    candidates = refined

    # Sort by SNR but return ALL candidates (no cap)
    candidates.sort(key=lambda c: c.peak_snr, reverse=True)

    # Final dedup: hotspot- and component-seeded candidates can converge to
    # the same streak after angle refinement and re-tracing.  Keep the
    # highest-SNR instance.
    deduped: list[StreakCandidate] = []
    for candidate in candidates:
        if any(
            np.hypot(candidate.x - kept.x, candidate.y - kept.y) < fwhm * 2
            for kept in deduped
        ):
            continue
        deduped.append(candidate)
    candidates = deduped
    logger.info(
        "Detected %d streak candidates (%d rejected short, %d rejected profile, "
        "%d rejected border, %d excluded star streaks, "
        "%d duplicate hotspots, from %d hotspots)",
        len(candidates),
        n_rejected_short,
        n_rejected_profile,
        n_rejected_border,
        n_rejected_excluded,
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
