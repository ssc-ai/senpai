"""Satellite point-source detection in rate-track frames, assuming WCS is fit."""

import logging

import numpy as np
import sep
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter

from senpai.engine.detection.streak.masking import percent_difference
from senpai.engine.models.senpai import RateTrackFrame
from senpai.engine.models.starfield import SatelliteInImage, SatelliteListImage
from senpai.settings import settings

logger = logging.getLogger(__name__)


# Utility helpers


def cutout_gauss(sub_image: np.ndarray, pixel_seeing: float) -> tuple[float, float, float]:
    """Fit a 2D Gaussian to a sub-image and return FWHM measurements.

    Improvements over the original - Kevin:
    - sigma_init is derived from pixel_seeing and clamped to a safe range,
      preventing divergence on unusual inputs.
    - Explicit bounds on x_stddev / y_stddev keep the fitter within physical
      limits.
    - Convergence is checked via fit_info['ierr']; a ValueError is raised when
      the fit did not converge so the caller can handle it cleanly.

    Args:
        sub_image: Small image cutout centered on a detection.
        pixel_seeing: Expected seeing in pixels.

    Returns:
        Tuple of (fwhm_x, fwhm_y, fwhm_avg).

    Raises:
        ValueError: If the Gaussian fit does not converge.
    """
    size = sub_image.shape[0]

    # Remove background to improve Gaussian fitting
    sub_image = sub_image - np.median(sub_image)

    # Derive a physically sensible initial sigma and clamp to safe bounds
    sigma_seeing = pixel_seeing / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    min_sigma = 0.1
    max_sigma = size / 2.0
    sigma_init = float(np.clip(sigma_seeing, min_sigma, max_sigma))

    p_init = models.Gaussian2D(
        amplitude=np.max(sub_image),
        x_mean=size // 2,
        y_mean=size // 2,
        x_stddev=sigma_init,
        y_stddev=sigma_init,
    )
    p_init.x_stddev.bounds = (min_sigma, max_sigma)
    p_init.y_stddev.bounds = (min_sigma, max_sigma)

    fit_p = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:size, :size]
    fitted_p = fit_p(p_init, x, y, sub_image)

    # Reject fits that did not converge (ierr codes 1-4 indicate success)
    if fit_p.fit_info["ierr"] not in (1, 2, 3, 4):
        raise ValueError("Gaussian fit did not converge")

    fwhm_x = fitted_p.x_stddev.value * 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm_y = fitted_p.y_stddev.value * 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm_avg = (fwhm_x + fwhm_y) / 2.0

    return fwhm_x, fwhm_y, fwhm_avg


def find_two_brightest_points(arr: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the coordinates of the two brightest points in a 2D array.

    Args:
        arr: 2D numpy array.

    Returns:
        Coordinates of the two brightest points as ((y1, x1), (y2, x2)).
    """
    brightest_point_1 = np.unravel_index(np.argmax(arr), arr.shape)

    arr_copy = arr.copy()
    arr_copy[brightest_point_1] = np.min(arr)

    brightest_point_2 = np.unravel_index(np.argmax(arr_copy), arr_copy.shape)

    return brightest_point_1, brightest_point_2


def euclidean_distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    """Calculate the Euclidean distance between two points.

    Args:
        point1: Coordinates (y, x).
        point2: Coordinates (y, x).

    Returns:
        Euclidean distance.
    """
    return float(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def generate_cutout(frame: np.ndarray, detection: tuple[float, float], side: int) -> np.ndarray:
    """Generate a square cutout centered on a detection.

    Args:
        frame: Full image array.
        detection: (x, y) coordinates of the detection.
        side: Half-width of the cutout in pixels.

    Returns:
        Square cutout of the image.
    """
    x, y = detection
    y_min = max(0, round(y) - side)
    y_max = min(round(y) + side, frame.shape[0])
    x_min = max(0, round(x) - side)
    x_max = min(round(x) + side, frame.shape[1])

    return frame[y_min:y_max, x_min:x_max].copy()


# PSF-scale-aware concentration aperture radii, as multiples of the measured FWHM.
# Inner aperture captures the PSF core; outer aperture bounds the wings. For a
# Gaussian PSF the inner/outer flux ratio is ~constant (scale-invariant) and well
# above the area-ratio floor that pure noise produces.
_CONCENTRATION_INNER_FWHM = 0.75
_CONCENTRATION_OUTER_FWHM = 1.5

# Accept band for ``psf_flux_concentration``, derived from the OURSKY corpus:
#   pure noise ~0.26, faint SNR~7 point ~0.40 (synthetic) / ~0.32-0.40 (corpus),
#   bright real satellite ~0.61-0.72, hot-pixel / cosmic-ray spike -> ~1.0.
# The lower bound rejects pure noise while admitting faint point sources; the upper
# bound rejects single-pixel spikes while admitting the most concentrated real PSFs.
CONCENTRATION_MIN = 0.30
CONCENTRATION_MAX = 0.90


def _flux_concentration(cutout: np.ndarray) -> float:
    """Return the fraction of total flux contained in the central 3x3 core.

    Retained for backward compatibility; superseded by ``psf_flux_concentration``,
    which is PSF-scale-aware. A fixed 3x3 core under-captures real PSFs whose FWHM
    exceeds ~3 px, so this measure is not used by the detection filter anymore.

    Args:
        cutout: Background-subtracted, minimum-normalized cutout.

    Returns:
        core_flux / total_flux, or 0.0 if the cutout is too small.
    """
    cy, cx = np.array(cutout.shape) // 2
    if cy - 1 < 0 or cx - 1 < 0 or cy + 2 > cutout.shape[0] or cx + 2 > cutout.shape[1]:
        return 0.0
    core_flux = np.sum(cutout[cy - 1 : cy + 2, cx - 1 : cx + 2])
    total_flux = np.sum(cutout)
    return float(core_flux / total_flux) if total_flux > 0 else 0.0


def psf_flux_concentration(cutout: np.ndarray, pixel_fwhm: float) -> float:
    """Scale-invariant flux concentration: inner/outer PSF-sized aperture ratio.

    Measures the fraction of flux inside a PSF-core aperture (radius
    ``0.75 * FWHM``) relative to a larger aperture (radius ``1.5 * FWHM``), both
    circular and centred on the cutout. Unlike the legacy fixed-3x3 measure, this
    is (approximately) invariant to the PSF width, so a true point source yields a
    consistent value regardless of seeing:

    * Gaussian point source: ~0.6-0.8 (scale-invariant; faint SNR~7 ~0.40).
    * Pure noise: ~0.25, set by the aperture area ratio (0.75/1.5)^2.
    * Diffuse / extended blob: below the point-source band (flux spread to wings).
    * Single-pixel spike (hot pixel / cosmic ray): ~1.0 (all flux in the core).

    Args:
        cutout: Background-subtracted, minimum-normalized square cutout.
        pixel_fwhm: Measured PSF FWHM in pixels (the real seeing for this frame).

    Returns:
        inner_flux / outer_flux, or 0.0 when the cutout is too small to contain the
        outer aperture or the outer aperture holds no flux.
    """
    r_in = _CONCENTRATION_INNER_FWHM * pixel_fwhm
    r_out = _CONCENTRATION_OUTER_FWHM * pixel_fwhm

    cy, cx = np.array(cutout.shape) // 2
    # The cutout must be large enough to contain the full outer aperture.
    if (
        cy - r_out < 0
        or cx - r_out < 0
        or cy + r_out >= cutout.shape[0]
        or cx + r_out >= cutout.shape[1]
    ):
        return 0.0

    y, x = np.mgrid[: cutout.shape[0], : cutout.shape[1]]
    radius = np.hypot(x - cx, y - cy)

    inner_flux = float(np.sum(cutout[radius <= r_in]))
    outer_flux = float(np.sum(cutout[radius <= r_out]))

    return inner_flux / outer_flux if outer_flux > 0 else 0.0


# Fallback PSF FWHM (pixels) used only when no measured seeing is available on the
# frame. This is the legacy hard-coded value; preserved as a last resort.
_DEFAULT_PIXEL_FWHM = 3.0


def measured_pixel_fwhm(frame: RateTrackFrame) -> float:
    """Return the best available measured PSF FWHM (pixels) for a rate-track frame.

    The streak extractor measures the real PSF FWHM during ``solve_shift`` and stores
    it on ``frame.streak.fwhm`` before detection runs, so it is the most direct and
    frame-specific estimate. The catalogued sidereal seeing
    (``starfield.detection_metadata.pixel_fwhm``) is used as a fallback, and the
    legacy hard-coded value only when neither is present.

    Sources are validated to be positive and finite before use.

    Args:
        frame: The rate-track frame being processed.

    Returns:
        The measured PSF FWHM in pixels.
    """

    def _valid(value: float | None) -> bool:
        return value is not None and np.isfinite(value) and value > 0

    if frame.streak is not None and _valid(frame.streak.fwhm):
        return float(frame.streak.fwhm)

    detection_metadata = getattr(frame.starfield, "detection_metadata", None)
    if detection_metadata is not None and _valid(detection_metadata.pixel_fwhm):
        return float(detection_metadata.pixel_fwhm)

    logger.warning(
        "No measured PSF FWHM available on frame %s; falling back to %.1f px",
        frame.index,
        _DEFAULT_PIXEL_FWHM,
    )
    return _DEFAULT_PIXEL_FWHM


def _simple_mask(
    shape: tuple[int, int], center_yx: tuple[float, float], radius: int
) -> np.ndarray:
    """Create a boolean mask with a rectangular aperture around a centre.

    Args:
        shape: (rows, cols) of the image.
        center_yx: (y, x) centre coordinates.
        radius: Half-width of the masked region in pixels.

    Returns:
        Boolean array; True inside the aperture.
    """
    mask = np.zeros(shape, dtype=bool)
    cy, cx = int(center_yx[0]), int(center_yx[1])
    mask[
        max(0, cy - radius) : min(shape[0], cy + radius),
        max(0, cx - radius) : min(shape[1], cx + radius),
    ] = True
    return mask


def _centroid_guard_offset(fwhm: float, mode: str, value: float) -> float:
    """Max sub-pixel-centroid <-> brightest-pixel disagreement tolerated, in pixels.

    Beyond this the SEP sub-pixel centroid is treated as unreliable (saturation /
    trailing / a blended moment) and the brightest pixel is reported instead.

    - ``"fwhm"``: ``value * fwhm`` -- PSF-relative, self-scaling with seeing.
    - ``"fixed"``: ``value`` -- absolute pixels, independent of the PSF.
    - ``"none"``: ``inf`` -- never fall back; always report the sub-pixel centroid.

    See ``DetectionConfig.centroid_guard_mode`` / ``centroid_guard_value``.
    """
    if mode == "none":
        return float("inf")
    if mode == "fixed":
        return value
    if mode == "fwhm":
        return value * fwhm
    raise ValueError(f"Unknown centroid_guard_mode: {mode!r}")


def _report_centroid(
    masked_frame: np.ndarray,
    seed_xy: tuple[float, float],
    max_peak_offset: float = float("inf"),
) -> tuple[float, float]:
    """Choose the reported (x, y) for an accepted detection.

    Reports the SEP sub-pixel centroid (``seed_xy``, full-frame). When it disagrees with
    the masked brightest pixel by more than ``max_peak_offset`` px -- a sign the sub-pixel
    fit was pulled off by saturation, residual trailing or a blended moment -- the integer
    peak is reported instead. ``max_peak_offset=inf`` disables the fallback (always report
    the centroid). Raises ValueError when the masked region holds no signal.
    """
    if np.sum(masked_frame) <= 0:
        raise ValueError("Masked frame has no signal")
    x_det, y_det = seed_xy
    if max_peak_offset == float("inf"):
        return float(x_det), float(y_det)
    peak_y, peak_x = np.unravel_index(np.argmax(masked_frame), masked_frame.shape)
    if np.hypot(x_det - peak_x, y_det - peak_y) <= max_peak_offset:
        return float(x_det), float(y_det)
    return float(peak_x), float(peak_y)


# Filtering


def filter_point_sources(
    image: np.ndarray,
    detections: list[tuple[float, float]],
    pixel_seeing: float,
    hot_pixel_threshold: float = 0.35,
    centroid_guard_mode: str = "fwhm",
    centroid_guard_value: float = 0.4,
) -> list[tuple[float, float, float]]:
    """Filter detections, keeping only those consistent with a point-source PSF.

    Filter order (cheapest to most expensive):
      1. Edge check — non-square cutout means the source straddles the frame boundary.
      2. Zero-signal check.
      3. Hot-pixel check — a single pixel dominates the flux.
      4. Flux concentration check — PSF-scale-aware inner/outer aperture ratio must
         lie within [0.30, 0.90]. Runs before the Gaussian fit to avoid the expensive
         fitter on obvious non-point-sources.
      5. Multi-peak check — two brightest pixels must be within one seeing disc.
      6. Gaussian PSF shape check — FWHM within [seeing/2.5, seeing*3] and
         roundness within 55 % difference.
      7. Report position: the SEP sub-pixel centroid, guarded against unreliable fits
         by a configurable fallback to the brightest pixel (see _report_centroid).

    Args:
        image: Background-subtracted full image array (Used to be RateTrackFrame).
        detections: List of (x, y) initial detection coordinates.
        pixel_seeing: Expected PSF FWHM in pixels.
        hot_pixel_threshold: Max fraction of flux allowed in a single pixel.
        centroid_guard_mode: How the reported-position guard threshold is set --
            "fwhm" (value * FWHM), "fixed" (value px) or "none" (no fallback).
        centroid_guard_value: Threshold for ``centroid_guard_mode`` (PSF multiple or px).

    Returns:
        List of (x, y, fwhm) for accepted point sources.
    """
    filtered_detections = []
    cutout_size = int(3 * pixel_seeing)

    logger.info(f"Evaluating {len(detections)} detections")

    for idx, detection in enumerate(detections):
        x_det, y_det = detection

        # (1) Edge check
        cutout = generate_cutout(image, detection, cutout_size)
        if cutout.shape[0] != cutout.shape[1]:
            if settings.detection.verbose:
                logger.warning(f"[{idx + 1}] [FILTERING] Detection is on edge of image")
            continue

        # (2) Zero-signal check
        cutout = cutout - np.min(cutout)
        total_flux = np.sum(cutout)
        if total_flux == 0:
            if settings.detection.verbose:
                logger.warning(f"[{idx + 1}] [FILTERING] No signal in cutout")
            continue

        # (3) Hot-pixel check
        hot_pixel_concentration = np.max(cutout) / total_flux
        if hot_pixel_concentration > hot_pixel_threshold:
            if settings.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] Brightest pixel contains "
                    f"{hot_pixel_concentration:.2f} of total flux"
                )
            continue

        # (4) Flux concentration — cheap numpy gate before the Gaussian fit
        concentration = psf_flux_concentration(cutout, pixel_seeing)
        if concentration < CONCENTRATION_MIN or concentration > CONCENTRATION_MAX:
            if settings.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] PSF flux concentration "
                    f"{concentration:.2f} outside [{CONCENTRATION_MIN}, {CONCENTRATION_MAX}]"
                )
            continue

        # (5) Multi-peak check
        p1, p2 = find_two_brightest_points(cutout)
        dist = euclidean_distance(p1, p2)
        if dist > pixel_seeing:
            if settings.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] Two brightest pixels separated by "
                    f"{dist:.1f} px (seeing={pixel_seeing:.1f} px)"
                )
            continue

        # (6) Gaussian PSF shape check
        try:
            fx, fy, fcomb = cutout_gauss(cutout, pixel_seeing)
        except Exception as e:
            if settings.detection.verbose:
                logger.warning(f"[{idx + 1}] [FILTERING] Gaussian fit failed: {e}")
            continue

        if fx < pixel_seeing / 2.5 or fy < pixel_seeing / 2.5:
            if settings.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] PSF too narrow (FWHM={fcomb:.2f}, "
                    f"seeing={pixel_seeing:.2f})"
                )
            continue

        if fx > pixel_seeing * 3 or fy > pixel_seeing * 3:
            if settings.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] PSF too wide (FWHM={fcomb:.2f}, "
                    f"seeing={pixel_seeing:.2f})"
                )
            continue

        if percent_difference(fx, fy) > 55:
            if settings.detection.verbose:
                logger.warning(
                    f"[{idx + 1}] [FILTERING] PSF non-circular "
                    f"(Δ={percent_difference(fx, fy):.1f}%)"
                )
            continue

        # (7) Report position: the sub-pixel SEP centroid, with a PSF-relative
        #     fallback to the brightest pixel when the fit is unreliable
        #     (see _centroid_guard_offset / _report_centroid)
        try:
            mask = _simple_mask(image.shape, (y_det, x_det), cutout_size)
            masked_frame = image * mask
            offset = _centroid_guard_offset(fcomb, centroid_guard_mode, centroid_guard_value)
            x_cent, y_cent = _report_centroid(masked_frame, (x_det, y_det), offset)
        except Exception as e:
            if settings.detection.verbose:
                logger.warning(f"[{idx + 1}] [FILTERING] Centroid refinement failed: {e}")
            continue

        logger.info(
            f"[{idx + 1}] [ACCEPTING] FWHM={fcomb:.2f}, "
            f"core_concentration={concentration:.2f}, "
            f"hot_pixel_fraction={hot_pixel_concentration:.2f}"
        )
        filtered_detections.append([float(x_cent), float(y_cent), fcomb])

    return filtered_detections


# Main detection entry point


def extract_point_sources(frame: RateTrackFrame) -> SatelliteListImage:
    """Extract point sources from a rate-track frame via SEP and PSF filtering.

    Uses SEP for source detection and a multi-stage PSF filter for validation.

    Pipeline:
      1. Hot-pixel suppression via 3x3 median filter.
      2. Background estimation and subtraction with SEP.
      3. Adaptive-threshold source extraction (binary search, 50-300 sources).
      4. Multi-stage PSF filtering (see filter_point_sources).
      5. Proximity deduplication (< 1 px).
      6. SNR threshold cut.

    Args:
        frame: A RateTrackFrame containing the image data and metadata.

    Returns:
        A SatelliteListImage of detected point sources.
    """
    # Pre-processing
    image_data = median_filter(frame.frame.data, size=3)

    # SEP requires a C-contiguous float64 array
    image_data = np.ascontiguousarray(image_data, dtype=np.float64)

    bkg = sep.Background(image_data)
    image_sub = image_data - bkg

    # sigma_clipped_stats used only to obtain std for the convergence check
    _, _, std = sigma_clipped_stats(image_sub, sigma=3.0)

    # Adaptive threshold source extraction with SEP
    fwhm = measured_pixel_fwhm(frame)

    threshold_min = 3.0
    threshold_max = 50.0
    threshold = 5.0
    min_sources = 50
    max_sources = 300

    sources = None
    for attempt in range(10):
        sources = sep.extract(image_sub, threshold, err=bkg.globalrms)
        count = 0 if sources is None else len(sources)

        logger.info(f"Attempt {attempt + 1}: threshold={threshold:.2f}, found {count} sources")

        if sources is None or count < min_sources:
            threshold_max = threshold
            threshold = (threshold_min + threshold) / 2.0
            logger.info(f"Too few sources, decreasing threshold to {threshold:.2f}")
        elif count > max_sources:
            threshold_min = threshold
            threshold = (threshold + threshold_max) / 2.0
            logger.info(f"Too many sources ({count}), increasing threshold to {threshold:.2f}")
        else:
            logger.info(f"Found {count} sources at threshold {threshold:.2f}")
            break

        if abs(threshold_max - threshold_min) < 0.1 * std:
            logger.info(f"Threshold search converged at {threshold:.2f}")
            break

    if sources is None or len(sources) == 0:
        logger.info("No sources detected by SEP")
        return SatelliteListImage(detections=[], image_metadata=frame.starfield.image_metadata)

    initial_detections = [(float(src["x"]), float(src["y"])) for src in sources if src["flux"] > 0]
    logger.info(f"Initial extraction: {len(initial_detections)} candidate sources")

    # PSF filtering
    pixel_seeing = float(fwhm)
    logger.info(f"Measured PSF seeing for filtering: {pixel_seeing:.2f} pixels")

    filtered_detections = filter_point_sources(
        image=image_sub,
        detections=initial_detections,
        pixel_seeing=pixel_seeing,
        centroid_guard_mode=settings.detection.centroid_guard_mode,
        centroid_guard_value=settings.detection.centroid_guard_value,
    )
    logger.info(f"After PSF filtering: {len(filtered_detections)} sources remain")

    # Proximity deduplication
    deduplicated: list[list[float]] = []
    for det in filtered_detections:
        if not any(np.hypot(det[0] - ex[0], det[1] - ex[1]) < 1.0 for ex in deduplicated):
            deduplicated.append(det)

    if len(deduplicated) < len(filtered_detections):
        logger.info(f"After deduplication: {len(deduplicated)} sources remain")

    # SNR cut and WCS projection
    stars = []
    for x, y, pixel_fwhm in deduplicated:
        cutout = generate_cutout(image_sub, (x, y), int(pixel_fwhm * 2))
        if cutout.size == 0:
            continue
        snr = float(np.max(cutout) / std)

        ra, dec = None, None
        if frame.starfield.wcs is not None:
            ra, dec = frame.starfield.wcs.pix2world_0based(x, y)

        star = SatelliteInImage(
            x=float(x),
            y=float(y),
            ra=ra,
            dec=dec,
            snr=snr,
            pixel_fwhm=float(pixel_fwhm),
        )

        if (
            settings.detection.snr_threshold is not None
            and star.snr > settings.detection.snr_threshold
        ):
            stars.append(star)

    if settings.detection.snr_threshold:
        logger.info(f"After SNR filtering: {len(stars)} sources remain")

    return SatelliteListImage(detections=stars, image_metadata=frame.starfield.image_metadata)
