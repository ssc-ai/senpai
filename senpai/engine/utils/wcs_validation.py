"""Absolute, image-based WCS validation.

The refinement pipeline's existing checks are all *relative*: a refit is
compared against the WCS it started from, and a shift is compared against
perturbations of itself. A WCS poisoned by one bad frame-to-frame shift
passes every one of those checks. This module asks the only question that
settles it: is there actually star flux in the image at the positions the
WCS predicts for the brightest catalog stars?
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from senpai.core.config import get_config
from senpai.engine.models.astrometry import WCSQualityMetrics

if TYPE_CHECKING:
    from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame

logger = logging.getLogger(__name__)


def _coarse_background(image: np.ndarray, block: int) -> np.ndarray:
    """Blockwise median background, upsampled back to full resolution.

    Fast (no full-resolution filtering) and adequate for the smooth halo /
    moonlight gradients these frames carry; point sources and streaks barely
    move a 32x32-block median.
    """
    h, w = image.shape
    hb, wb = h // block, w // block
    trimmed = image[: hb * block, : wb * block]
    blocks = trimmed.reshape(hb, block, wb, block).swapaxes(1, 2)
    med = np.median(blocks, axis=(2, 3))
    bg = np.repeat(np.repeat(med, block, axis=0), block, axis=1)
    # Pad the right/bottom remainder with the nearest edge value
    if bg.shape != image.shape:
        bg = np.pad(bg, ((0, h - bg.shape[0]), (0, w - bg.shape[1])), mode="edge")
    return bg


def _box_sums(integral: np.ndarray, xs: np.ndarray, ys: np.ndarray, r: int) -> np.ndarray:
    """Background-subtracted flux in (2r x 2r) boxes via a summed-area table."""
    h, w = integral.shape[0] - 1, integral.shape[1] - 1
    x0 = np.clip(xs.astype(int) - r, 0, w)
    x1 = np.clip(xs.astype(int) + r, 0, w)
    y0 = np.clip(ys.astype(int) - r, 0, h)
    y1 = np.clip(ys.astype(int) + r, 0, h)
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def flux_significance_test(
    image: np.ndarray,
    positions: list[tuple[float, float]],
    box_radius: int,
    n_random: int = 500,
    significance_percentile: float = 99.0,
    control_offset_px: int = 300,
    background_block_px: int = 32,
    rng_seed: int = 0,
) -> dict:
    """Test predicted star positions for significant flux.

    Returns a dict with frac_significant (predictions), control_frac
    (predictions offset by control_offset_px), and n_tested. The control grid
    measures how often blank sky passes the same threshold, so a passing WCS
    must beat both the random null and its own offset control.
    """
    h, w = image.shape
    margin = box_radius + 10
    bgsub = image - _coarse_background(image, background_block_px)
    integral = np.zeros((h + 1, w + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(bgsub, axis=0), axis=1)

    pos = np.array(
        [(x, y) for x, y in positions if margin < x < w - margin and margin < y < h - margin]
    )
    if len(pos) == 0:
        return {"n_tested": 0, "frac_significant": 0.0, "control_frac": 0.0}

    rng = np.random.default_rng(rng_seed)
    rx = rng.uniform(margin, w - margin, n_random)
    ry = rng.uniform(margin, h - margin, n_random)
    null = _box_sums(integral, rx, ry, box_radius)
    threshold = float(np.percentile(null, significance_percentile))

    pred = _box_sums(integral, pos[:, 0], pos[:, 1], box_radius)

    # Control: same predictions pushed control_offset_px away (folded back into
    # bounds), i.e. what the score looks like when the WCS points at blank sky.
    cx = np.where(
        pos[:, 0] + control_offset_px < w - margin,
        pos[:, 0] + control_offset_px,
        pos[:, 0] - control_offset_px,
    )
    ctrl = _box_sums(integral, cx, pos[:, 1], box_radius)

    return {
        "n_tested": len(pos),
        "frac_significant": float(np.mean(pred > threshold)),
        "control_frac": float(np.mean(ctrl > threshold)),
    }


def _validation_box_radius(frame: SiderealFrame | RateTrackFrame) -> int:
    """Box radius matched to how a star appears in this frame.

    For rate-track frames the box is matched to the streak *width*, not its
    length: a box sum's signal-to-noise is flat in box size up to the streak
    length, while contamination from structured background (moon halo,
    vignetting) grows with box area — so a tight box centered mid-streak
    separates good from poisoned WCS far better than one covering the whole
    trail. Sidereal frames need only a few FWHM.
    """
    streak = getattr(frame, "streak", None)
    if streak is not None:
        fwhm = getattr(streak, "fwhm", None)
        if fwhm is not None and np.isfinite(fwhm):
            return int(np.clip(2 * fwhm, 8, 16))
        return 12
    md = frame.starfield.detection_metadata if frame.starfield else None
    fwhm = md.pixel_fwhm if md is not None and md.pixel_fwhm else 3.0
    return int(np.clip(3 * fwhm, 6, 15))


def validate_frame_wcs(
    frame: SiderealFrame | RateTrackFrame, refit_stats: dict | None = None
) -> WCSQualityMetrics | None:
    """Run absolute WCS validation on a frame and return the quality metrics.

    ``refit_stats`` (from :func:`fit_and_validate_wcs`) is folded into the
    metrics when the refinement produced a refit. Returns None (and logs) when
    validation is disabled or the frame has no usable catalog.
    """
    config = get_config().wcs_validation
    if not config.enable:
        return None

    starfield = frame.starfield
    if starfield is None or not starfield.catalog_stars:
        logger.warning(
            "WCS validation skipped for frame %d: no catalog stars", frame.index
        )
        return None

    stars = [
        s
        for s in starfield.catalog_stars
        if s.x is not None and s.y is not None and s.magnitude is not None
    ]
    stars.sort(key=lambda s: s.magnitude)
    positions = [(s.x, s.y) for s in stars[: config.n_stars]]

    box_radius = _validation_box_radius(frame)
    result = flux_significance_test(
        frame.frame.data,
        positions,
        box_radius=box_radius,
        n_random=config.n_random,
        significance_percentile=config.significance_percentile,
        control_offset_px=config.control_offset_px,
        background_block_px=config.background_block_px,
    )

    if result["n_tested"] < config.min_stars:
        passed = None
    else:
        passed = result["frac_significant"] >= max(
            config.min_frac_significant,
            result["control_frac"] + config.control_margin,
        )

    metrics = WCSQualityMetrics(
        n_stars_tested=result["n_tested"],
        box_radius_px=box_radius,
        frac_significant=result["frac_significant"],
        control_frac_significant=result["control_frac"],
        null_percentile=config.significance_percentile,
        passed=passed,
        refit_rms_px=(refit_stats or {}).get("rms_px"),
        refit_rms_arcsec=(refit_stats or {}).get("rms_arcsec"),
        n_refit_stars=(refit_stats or {}).get("n_stars"),
    )

    logger.info(
        "WCS validation frame %d: %d stars, frac_significant=%.2f (control=%.2f, "
        "box=%dpx) -> %s",
        frame.index,
        metrics.n_stars_tested,
        metrics.frac_significant,
        metrics.control_frac_significant,
        box_radius,
        {True: "PASS", False: "FAIL", None: "INDETERMINATE"}[passed],
    )
    return metrics
