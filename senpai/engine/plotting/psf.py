"""Per-frame empirical PSF panels (gated by ``config.plotting.psfs``).

Two panels, both built by median-stacking many bright, isolated, unsaturated
catalog stars straight from the frame pixels:

* **sidereal** — stars are points; stack them into a 2D PSF and read off the
  radial profile + RA/Dec cuts (so an elongation reads as a tracking error).
* **rate** — stars are streaks; stack oriented (streak-aligned) stamps and read
  off the along-streak and across-streak profiles, with the fitted length×width
  box overlaid.

The stacking is cosmic-ray robust (peak / centroid / SNR taken from a 3x3
median-filtered copy; the raw stamp is what gets stacked, so spikes wash out in
the median). Each panel also drops a small ``.npy`` of the stacked stamp next to
the PNG so the panel can be regenerated later without the raw FITS; replot falls
back to reloading the processed FITS and re-slicing when the .npy is absent.

This module is the shared home for the stacking/profile primitives; the
night-level observability plots can import them from here too.
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from astropy.wcs import WCS

    from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
    from senpai.engine.models.starfield import StarField

    # A duck-typed frame with a starfield/streak (either track flavor).
    Frame = SiderealFrame | RateTrackFrame
    # A catalog star as ``(x, y, magnitude)``; any element may be ``None``.
    StarTuple = tuple[float | None, float | None, float | None]

logger = logging.getLogger(__name__)

SAT_PEAK = 40000.0      # raw ADU; reject saturated stars (below the 65535 clip)
MAX_STARS = 200         # stamps to stack
MIN_STAMPS = 15         # min stacked stars for a usable panel
MIN_PEAK_SNR = 20.0     # preferred per-stamp peak/noise; used when enough stars clear it
MIN_PEAK_SNR_FLOOR = 5.0  # detection-grade floor; fall back to these on low-SNR frames
SIDEREAL_HALF = 30      # sidereal stamp is (2*half+1)^2 px
_GAUSS_W25_OVER_W50 = math.sqrt(math.log(4) / math.log(2))  # = sqrt(2)


# --------------------------------------------------------------------------
# profile primitives (shared with observability.calibration)
# --------------------------------------------------------------------------
def cut_width(profile: np.ndarray, level: float = 0.5) -> float:
    """Full width at ``level`` x peak from the outermost interpolated crossings.

    Args:
        profile: 1D intensity profile.
        level: Fraction of the peak at which to measure the width.

    Returns:
        The full width in samples, or ``nan`` if fewer than two samples exceed
        the threshold.
    """
    profile = np.asarray(profile, dtype=float)
    thr = profile.max() * level
    above = np.where(profile >= thr)[0]
    if len(above) < 2:
        return float("nan")
    lo, hi = int(above[0]), int(above[-1])
    left = (lo - (profile[lo] - thr) / (profile[lo] - profile[lo - 1])
            if lo > 0 and profile[lo] != profile[lo - 1] else float(lo))
    right = (hi + (profile[hi] - thr) / (profile[hi] - profile[hi + 1])
             if hi < len(profile) - 1 and profile[hi] != profile[hi + 1]
             else float(hi))
    return float(right - left)


def profile_shape(profile: np.ndarray) -> dict[str, float]:
    """Compute multi-level widths and a Gaussianity ``spike_index`` of a 1D cut.

    The ``spike_index`` is ~1 for a Gaussian, >>1 for a narrow core on a broad
    halo (FWHM then spurious), and <1 for a flat-top / donut profile.

    Args:
        profile: 1D intensity profile.

    Returns:
        A dict with keys ``fwhm``, ``fwqm``, ``fw3qm``, and ``spike_index``.
    """
    w50 = cut_width(profile, 0.5)
    w25 = cut_width(profile, 0.25)
    w75 = cut_width(profile, 0.75)
    ok = np.isfinite(w50) and w50 > 0 and np.isfinite(w25)
    idx = float((w25 / w50) / _GAUSS_W25_OVER_W50) if ok else float("nan")
    return {"fwhm": w50, "fwqm": w25, "fw3qm": w75, "spike_index": idx}


def radial_profile(
    stamp: np.ndarray, half: int, rstep: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an azimuthally-averaged (ring-median) radial profile.

    Args:
        stamp: Square 2D stamp centered on the source.
        half: Center coordinate (half the stamp size) in pixels.
        rstep: Radial bin width in pixels.

    Returns:
        A ``(radii, profile)`` pair where ``profile`` is peak-normalized.
    """
    n = stamp.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - half, yy - half)
    edges = np.arange(0.0, half + rstep, rstep)
    r, prof = [], []
    for lo, hi in itertools.pairwise(edges):
        m = (rr >= lo) & (rr < hi)
        if m.any():
            r.append((lo + hi) / 2)
            prof.append(float(np.median(stamp[m])))
    prof = np.array(prof)
    if prof.size and prof.max() > 0:
        prof = prof / prof.max()
    return np.array(r), prof


def sky_axes(astropy_wcs: WCS | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute pixel-space unit vectors pointing East (+RA) and North (+Dec).

    The vectors are evaluated at the frame center.

    Args:
        astropy_wcs: Astropy WCS for the frame, or ``None``.

    Returns:
        An ``(east, north)`` pair of unit vectors, or ``None`` if the WCS is
        missing or the projection is degenerate at the center.
    """
    if astropy_wcs is None:
        return None
    try:
        x0 = float(astropy_wcs.wcs.crpix[0]) - 1.0
        y0 = float(astropy_wcs.wcs.crpix[1]) - 1.0
        ra0, dec0 = (float(c) for c in astropy_wcs.all_pix2world(x0, y0, 0))
        dd = 1.0 / 3600.0
        xn, yn = (float(c) for c in astropy_wcs.all_world2pix(ra0, dec0 + dd, 0))
        xe, ye = (float(c) for c in astropy_wcs.all_world2pix(
            ra0 + dd / math.cos(math.radians(dec0)), dec0, 0))
        north = np.array([xn - x0, yn - y0])
        east = np.array([xe - x0, ye - y0])
        nn, ne = np.linalg.norm(north), np.linalg.norm(east)
        if not (np.isfinite(nn) and np.isfinite(ne) and nn > 0 and ne > 0):
            return None
        return east / ne, north / nn
    except Exception:
        return None


def _sample_line(stamp: np.ndarray, half: int, unit: np.ndarray) -> np.ndarray:
    """Sample a stamp along a line through its center in the ``unit`` direction.

    Args:
        stamp: Square 2D stamp.
        half: Center coordinate (half the stamp size) in pixels.
        unit: 2-element ``(x, y)`` direction unit vector.

    Returns:
        The interpolated intensity profile sampled at integer steps.
    """
    from scipy.ndimage import map_coordinates
    t = np.arange(-half, half + 1.0)
    return map_coordinates(stamp, [half + t * unit[1], half + t * unit[0]],
                           order=1, mode="constant", cval=0.0)


# --------------------------------------------------------------------------
# stacking
# --------------------------------------------------------------------------
def _ring_noise(ring: np.ndarray) -> float:
    """Estimate MAD-based sky scatter for a stamp's border ring.

    ``np.std`` is the wrong tool here: a bright star's own diffraction wings or a
    faint neighbor landing in the 4px border inflates the std by ~10x (measured:
    6500 vs a true scatter of ~500 ADU), which collapses the peak/noise ratio and
    makes every star fail the SNR gate — the panel then silently vanishes on
    perfectly good, bright frames. The MAD ignores those few contaminated pixels
    and recovers the real sky scatter.

    Args:
        ring: Pixel values sampled from the stamp's border ring.

    Returns:
        The robust sky scatter estimate (1.4826 x MAD), falling back to the
        standard deviation when the MAD is zero.
    """
    ring = np.asarray(ring, dtype=float)
    med = np.median(ring)
    mad = float(np.median(np.abs(ring - med)))
    return 1.4826 * mad if mad > 0 else float(np.std(ring))


def _isolated_order(
    xy: np.ndarray, mags: np.ndarray, iso_radius: float
) -> Iterator[int]:
    """Yield brightest-first indices of well-isolated stars.

    A star is isolated when it has no brighter-or-comparable neighbor within
    ``iso_radius``.

    Args:
        xy: ``(N, 2)`` array of star pixel positions.
        mags: Length-``N`` array of star magnitudes.
        iso_radius: Isolation radius in pixels.

    Yields:
        Indices into ``xy``/``mags`` in ascending-magnitude (brightest-first)
        order, skipping stars with a comparable brighter neighbor.
    """
    tree = cKDTree(xy)
    for i in np.argsort(mags):
        neigh = tree.query_ball_point(xy[i], iso_radius)
        if not any(j != i and mags[j] < mags[i] + 2.0 for j in neigh):
            yield int(i)


def stack_stars(
    data: np.ndarray,
    stars: list[StarTuple],
    fwhm: float,
    half: int | None = None,
    max_stars: int = MAX_STARS,
) -> tuple[np.ndarray | None, int]:
    """Build a median-stacked, peak-normalized point-source PSF (sidereal).

    Args:
        data: Full frame pixel data.
        stars: Catalog stars as ``(x, y, mag)`` tuples.
        fwhm: Estimated point-source FWHM in pixels; sets the stamp/isolation
            scaling.
        half: Stamp half-size in pixels; defaults to a FWHM-scaled value floored
            at :data:`SIDEREAL_HALF`.
        max_stars: Maximum number of stamps to stack.

    Returns:
        A ``(stamp, n)`` pair of the stacked PSF and the number of stars used, or
        ``(None, 0)`` if too few usable stars were found.
    """
    keep = [(s[0], s[1], s[2] if s[2] is not None else np.inf) for s in stars
            if s[0] is not None and s[1] is not None]
    if len(keep) < 20:
        return None, 0
    # The stamp must hold the whole PSF *plus* a clean sky border, because the
    # outer 4px ring is what sets the per-stamp background/noise and the
    # peak-SNR gate. A fixed 30px half is too small for a defocused or strongly
    # aberrated PSF (FWHM >~ 15px): its donut reaches the border, the "sky" ring
    # samples PSF wings, noise is overestimated and peak-SNR collapses — so
    # every stamp is rejected and the panel silently vanishes exactly when it is
    # most diagnostic. Scale the stamp (and the isolation radius) with FWHM,
    # floored at SIDEREAL_HALF for well-focused frames.
    if half is None:
        half = int(max(SIDEREAL_HALF, round(3.0 * fwhm)))
    isolation = max(60.0, half + fwhm)  # neighbors clear of the stamp footprint
    xy = np.array([(s[0], s[1]) for s in keep])
    mags = np.array([s[2] for s in keep])
    h, w = data.shape
    n = 2 * half + 1
    candidates = []  # (peak_snr, stamp), collected brightest-first
    for i in _isolated_order(xy, mags, isolation):
        if len(candidates) >= max_stars:
            break
        x, y = xy[i]
        if not (half + 2 < x < w - half - 2 and half + 2 < y < h - half - 2):
            continue
        xi, yi = int(round(x)), int(round(y))
        st = data[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(float)
        if st.shape != (n, n):
            continue
        ring = np.concatenate([st[0:4].ravel(), st[-4:].ravel(),
                               st[:, 0:4].ravel(), st[:, -4:].ravel()])
        noise = _ring_noise(ring)
        st = st - np.median(ring)
        sm = ndimage.median_filter(st, size=3)
        peak = float(sm.max())
        # Reject saturated stars and pure noise, but do NOT demand a *high* SNR
        # per stamp: the median stack of N marginal stamps recovers a clean PSF
        # (~sqrt(N) gain), so a hard SNR>=20 gate buys little quality while
        # silently deleting the whole panel on faint/low-SNR frames where every
        # unsaturated star sits below it. Keep everything above a detection-grade
        # floor; the high-SNR subset is preferred below when it exists.
        if peak <= 0 or peak > SAT_PEAK or noise <= 0 or peak < MIN_PEAK_SNR_FLOOR * noise:
            continue
        cy, cx = ndimage.center_of_mass(np.clip(sm, 0, None))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        if abs(cy - half) > fwhm or abs(cx - half) > fwhm:
            continue
        st = ndimage.shift(st, (half - cy, half - cx), order=3, mode="nearest")
        st /= peak
        candidates.append((peak / noise, st))
    # Prefer high-SNR stamps; fall back to the best available so a low-SNR frame
    # still yields a (noisier) panel instead of silently vanishing.
    strong = [s for snr, s in candidates if snr >= MIN_PEAK_SNR]
    chosen = strong if len(strong) >= MIN_STAMPS else [s for _, s in candidates]
    if len(chosen) < MIN_STAMPS:
        return None, 0
    if len(strong) < MIN_STAMPS:
        logger.info("psf: sidereal stack from %d low-SNR stars (only %d >= SNR %g)",
                    len(chosen), len(strong), MIN_PEAK_SNR)
    stamp = np.median(np.stack(chosen), axis=0)
    if stamp.max() > 0:
        stamp = stamp / stamp.max()
    return stamp, len(chosen)


def _oriented_stamp(
    data: np.ndarray,
    x: float,
    y: float,
    cos_a: float,
    sin_a: float,
    half_a: int,
    half_p: int,
) -> np.ndarray | None:
    """Sample a streak-aligned stamp at ``(x, y)``.

    The stamp rows run perpendicular to the streak and the columns run along it.

    Args:
        data: Full frame pixel data.
        x: Center x-coordinate in pixels.
        y: Center y-coordinate in pixels.
        cos_a: Cosine of the streak angle.
        sin_a: Sine of the streak angle.
        half_a: Along-streak half-size in pixels.
        half_p: Perpendicular half-size in pixels.

    Returns:
        The interpolated float stamp, or ``None`` if it falls out of bounds.
    """
    from scipy.ndimage import map_coordinates
    ta = np.arange(-half_a, half_a + 1.0)
    tp = np.arange(-half_p, half_p + 1.0)
    TA, TP = np.meshgrid(ta, tp)
    sx = x + TA * cos_a - TP * sin_a
    sy = y + TA * sin_a + TP * cos_a
    h, w = data.shape
    if (sx.min() < 1 or sx.max() > w - 2 or sy.min() < 1 or sy.max() > h - 2):
        return None
    return map_coordinates(data, [sy.ravel(), sx.ravel()], order=1).reshape(TA.shape)


def stack_streaks(
    data: np.ndarray,
    stars: list[StarTuple],
    fwhm: float,
    length: float,
    angle_deg: float,
    max_stars: int = MAX_STARS,
) -> tuple[np.ndarray | None, int, int, int]:
    """Build a median-stacked, peak-normalized streak PSF in streak-aligned coords.

    Each catalog star is a streak; oriented stamps centered on the bright
    isolated ones are stacked.

    Args:
        data: Full frame pixel data.
        stars: Catalog stars as ``(x, y, mag)`` tuples.
        fwhm: Cross-streak FWHM in pixels; sets the stamp/isolation scaling.
        length: Streak length in pixels.
        angle_deg: Streak position angle in degrees.
        max_stars: Maximum number of stamps to stack.

    Returns:
        A ``(stamp, half_along, half_perp, n)`` tuple where ``stamp`` is indexed
        ``[perp, along]`` and ``n`` is the number of streaks stacked; ``stamp``
        is ``None`` if too few usable streaks were found.
    """
    # The stamp must contain the whole streak *plus* a clean sky margin on every
    # side: the along ends should roll fully into background (so a 183px trail
    # reads off with sky on either side, not clipped flush to the border) and
    # the perpendicular edges are what set the noise / peak-SNR gate. Size both
    # axes off the fitted geometry; the caps are only a memory guard for
    # pathologically long streaks, set far above the typical footprint.
    half_a = int(min(400, max(20, round(length / 2 + 6 * fwhm))))
    half_p = int(min(120, max(12, round(5 * fwhm))))
    keep = [(s[0], s[1], s[2] if s[2] is not None else np.inf) for s in stars
            if s[0] is not None and s[1] is not None]
    if len(keep) < 20:
        return None, half_a, half_p, 0
    xy = np.array([(s[0], s[1]) for s in keep])
    mags = np.array([s[2] for s in keep])
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    iso = max(60.0, length + 4 * fwhm)
    candidates = []  # (peak_snr, stamp), collected brightest-first
    for i in _isolated_order(xy, mags, iso):
        if len(candidates) >= max_stars:
            break
        st = _oriented_stamp(data, xy[i][0], xy[i][1], cos_a, sin_a, half_a, half_p)
        if st is None:
            continue
        ring = np.concatenate([st[0:2].ravel(), st[-2:].ravel()])  # perp edges
        noise = _ring_noise(ring)
        st = st - np.median(ring)
        sm = ndimage.median_filter(st, size=3)
        peak = float(sm.max())
        # Detection-grade floor only; the median stack recovers SNR, so a hard
        # SNR>=20 gate would needlessly drop the panel on faint streaks. The
        # high-SNR subset is preferred below when there are enough of them.
        if peak <= 0 or peak > SAT_PEAK or noise <= 0 or peak < MIN_PEAK_SNR_FLOOR * noise:
            continue
        # Center perpendicular only (along position varies with where the catalog
        # point falls on the trail; the across profile is what we want centered).
        perp_prof = np.clip(sm, 0, None).sum(axis=1)
        cy = float(ndimage.center_of_mass(perp_prof)[0])
        if not np.isfinite(cy) or abs(cy - half_p) > 2 * fwhm:
            continue
        st = ndimage.shift(st, (half_p - cy, 0.0), order=3, mode="nearest")
        st /= peak
        candidates.append((peak / noise, st))
    strong = [s for snr, s in candidates if snr >= MIN_PEAK_SNR]
    chosen = strong if len(strong) >= MIN_STAMPS else [s for _, s in candidates]
    if len(chosen) < MIN_STAMPS:
        return None, half_a, half_p, 0
    if len(strong) < MIN_STAMPS:
        logger.info("psf: streak stack from %d low-SNR streaks (only %d >= SNR %g)",
                    len(chosen), len(strong), MIN_PEAK_SNR)
    stamp = np.median(np.stack(chosen), axis=0)
    if stamp.max() > 0:
        stamp = stamp / stamp.max()
    return stamp, half_a, half_p, len(chosen)


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------
def paper_ready_enabled() -> bool:
    """True when ``config.plotting.paper_ready`` is set — emit title-less copies."""
    try:
        from senpai.core.config import get_config
        return bool(getattr(get_config().plotting, "paper_ready", False))
    except Exception:
        return False


def strip_titles(fig: Figure) -> None:
    """Blank the figure suptitle and every axes title for a caption-ready copy.

    The figure caption replaces the on-figure title in a paper.

    Args:
        fig: The figure to strip titles from (modified in place).
    """
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        st.set_text("")
    for ax in fig.axes:
        if ax.get_title():
            ax.set_title("")


def clean_copy_path(path: Path | str) -> Path:
    """Derive the title-less paper-copy path (``foo.png`` -> ``foo_clean.png``).

    Args:
        path: The original PNG path.

    Returns:
        The path with a ``_clean`` suffix inserted before the extension.
    """
    path = Path(path)
    return path.with_name(f"{path.stem}_clean{path.suffix}")


def _save(fig: Figure, png_path: Path | str) -> None:
    """Save a figure to PNG, plus a title-less copy when paper-ready is enabled.

    Args:
        fig: The figure to save.
        png_path: Destination PNG path.
    """
    FigureCanvasAgg(fig)
    fig.savefig(str(png_path), dpi=130)
    if paper_ready_enabled():
        strip_titles(fig)
        fig.savefig(str(clean_copy_path(png_path)), dpi=130, bbox_inches="tight")


def render_sidereal_psf(
    stamp: np.ndarray,
    n_stars: int | None,
    axes: tuple[np.ndarray, np.ndarray] | None,
    meta: dict,
    png_path: Path | str,
) -> None:
    """Render the sidereal per-frame PSF panel.

    The panel is a 2D heatmap (with a 50% contour and N/E arrows) alongside the
    radial profile and orthogonal cuts.

    Args:
        stamp: Stacked, peak-normalized point-source PSF stamp.
        n_stars: Number of stars stacked, or ``None`` if unknown.
        axes: Optional ``(east, north)`` pixel-space unit vectors for RA/Dec cuts.
        meta: Frame metadata dict (``index``, ``exposure``, ``pixel_scale_arcsec``).
        png_path: Destination PNG path.
    """
    half = stamp.shape[0] // 2
    psc = meta.get("pixel_scale_arcsec")
    if axes is not None:
        east, north = axes
        cut_ra = _sample_line(stamp, half, east)
        cut_dec = _sample_line(stamp, half, north)
        a, b = "RA", "Dec"
    else:
        cut_ra, cut_dec = stamp[half, :], stamp[:, half]
        a, b = "x", "y"
    sh_ra, sh_dec = profile_shape(cut_ra), profile_shape(cut_dec)
    r, rad = radial_profile(stamp, half)
    _win = min(half, max(10.0, 3.0 * max(sh_ra["fwhm"], sh_dec["fwhm"])))
    # TEMP: lock the 2D stamp panel to a fixed +/-21px window so panels are
    # directly comparable across frames. Revert to `_win` (the adaptive window)
    # to restore auto-scaling.
    stamp_view = 21.0

    fig = Figure(figsize=(13, 4.6))
    ax0, ax1, ax2 = fig.subplots(1, 3)
    grid = np.linspace(-half, half, stamp.shape[0])
    ax0.imshow(np.arcsinh(np.clip(stamp, 0, None) / 0.02), origin="lower",
               extent=[-half, half, -half, half], cmap="inferno")
    ax0.contour(grid, grid, stamp, levels=[0.5], colors="cyan", linewidths=0.9)
    if axes is not None:
        for u, name, col in ((north, "N", "white"), (east, "E", "deepskyblue")):
            L = 0.7 * stamp_view
            ax0.annotate("", xy=(L * u[0], L * u[1]), xytext=(0, 0),
                         arrowprops={"arrowstyle": "->", "color": col, "lw": 1.4})
            ax0.text(L * 1.13 * u[0], L * 1.13 * u[1], name, color=col, fontsize=9,
                     ha="center", va="center")
    ax0.set_xlim(-stamp_view, stamp_view)
    ax0.set_ylim(-stamp_view, stamp_view)
    ax0.set_title("stacked PSF (50% contour)")
    ax0.set_xlabel("Δx (px)")
    ax0.set_ylabel("Δy (px)")

    ax1.plot(r, np.clip(rad, 1e-4, None), color="purple", lw=2)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 1.3)
    ax1.set_xlim(0, half)
    ax1.set_xlabel("radius (px)")
    ax1.set_ylabel("normalized flux")
    ax1.set_title("radial profile")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(np.arange(stamp.shape[0]) - half, cut_ra, color="tab:red", lw=2,
             label=f"{a} FWHM {sh_ra['fwhm']:.1f}px")
    ax2.plot(np.arange(stamp.shape[0]) - half, cut_dec, color="tab:blue", lw=1.6,
             ls="--", label=f"{b} FWHM {sh_dec['fwhm']:.1f}px")
    for lv in (0.25, 0.5, 0.75):
        ax2.axhline(lv, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax2.set_xlim(-stamp_view, stamp_view)  # TEMP: match the locked stamp window
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Δ from center (px)")
    ax2.set_title(f"{a} (solid) / {b} (dashed) cuts")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    spike = max(sh_ra["spike_index"], sh_dec["spike_index"])
    sec = f", {sh_ra['fwhm'] * psc:.1f}\"" if psc else ""
    n_txt = n_stars if n_stars is not None else "?"
    fig.suptitle(f"frame {meta.get('index', '?')} sidereal PSF — "
                 f"{meta.get('exposure', '?')}s, n={n_txt}, "
                 f"{a}/{b} FWHM={sh_ra['fwhm']:.1f}/{sh_dec['fwhm']:.1f}px{sec}"
                 f"{'  ⚠spike' if spike >= 1.3 else ''}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, png_path)


def render_streak_psf(
    stamp: np.ndarray,
    half_a: int,
    half_p: int,
    n_stars: int | None,
    length: float,
    fwhm: float,
    sky_in_streak: tuple[np.ndarray, np.ndarray] | None,
    meta: dict,
    png_path: Path | str,
) -> None:
    """Render the rate-track per-frame streak PSF panel.

    The panel is an oriented 2D stamp (with the fitted L×W box, a 50% contour,
    and N/E arrows) alongside the along-streak and across-streak profiles.

    Args:
        stamp: Stacked, peak-normalized streak stamp indexed ``[perp, along]``.
        half_a: Along-streak half-size in pixels.
        half_p: Perpendicular half-size in pixels.
        n_stars: Number of streaks stacked, or ``None`` if unknown.
        length: Fitted streak length in pixels.
        fwhm: Fitted cross-streak FWHM in pixels.
        sky_in_streak: Optional ``(east, north)`` unit vectors expressed in the
            streak-aligned frame.
        meta: Frame metadata dict (``index``, ``exposure``, ``pixel_scale_arcsec``).
        png_path: Destination PNG path.
    """
    psc = meta.get("pixel_scale_arcsec")
    along = stamp.sum(axis=0)        # collapse perpendicular
    along = along / along.max() if along.max() > 0 else along
    across = stamp[:, stamp.shape[1] // 2 - 2: stamp.shape[1] // 2 + 3].sum(axis=1)
    across = across / across.max() if across.max() > 0 else across
    sh_across = profile_shape(across)

    fig = Figure(figsize=(13, 4.6))
    ax0, ax1, ax2 = fig.subplots(1, 3)
    ext = [-half_a, half_a, -half_p, half_p]
    ax0.imshow(np.arcsinh(np.clip(stamp, 0, None) / 0.02), origin="lower",
               extent=ext, aspect="auto", cmap="inferno")
    gx = np.linspace(-half_a, half_a, stamp.shape[1])
    gy = np.linspace(-half_p, half_p, stamp.shape[0])
    ax0.contour(gx, gy, stamp, levels=[0.5], colors="cyan", linewidths=0.9)
    # fitted length x width box (streak-aligned frame: along = x, perp = y)
    from matplotlib.patches import Rectangle
    ax0.add_patch(Rectangle((-length / 2, -fwhm / 2), length, fwhm, fill=False,
                            edgecolor="lime", lw=1.4, ls="--"))
    if sky_in_streak is not None:
        east_s, north_s = sky_in_streak
        for u, name, col in ((north_s, "N", "white"), (east_s, "E", "deepskyblue")):
            L = 0.6 * half_p
            ax0.annotate("", xy=(L * u[0], L * u[1]), xytext=(0, 0),
                         arrowprops={"arrowstyle": "->", "color": col, "lw": 1.3})
            ax0.text(L * 1.2 * u[0], L * 1.2 * u[1], name, color=col, fontsize=9,
                     ha="center", va="center")
    ax0.set_xlabel("along streak (px)")
    ax0.set_ylabel("across (px)")
    ax0.set_title("stacked streak (lime = fitted L×W)")

    ax1.plot(np.arange(stamp.shape[1]) - half_a, along, color="darkorange", lw=2)
    ax1.axvline(-length / 2, color="lime", ls="--", lw=1)
    ax1.axvline(length / 2, color="lime", ls="--", lw=1)
    ax1.set_xlabel("along streak (px)")
    ax1.set_ylabel("normalized flux")
    ax1.set_title(f"along-streak profile (L={length:.1f}px)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(stamp.shape[0]) - half_p, across, color="tab:blue", lw=2)
    for lv in (0.25, 0.5, 0.75):
        ax2.axhline(lv, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax2.set_xlim(-half_p, half_p)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("across streak (px)")
    ax2.set_title(f"across-streak profile (FWHM={sh_across['fwhm']:.1f}px)")
    ax2.grid(True, alpha=0.3)

    sec = f", {fwhm * psc:.1f}\"" if psc else ""
    n_txt = n_stars if n_stars is not None else "?"
    fig.suptitle(f"frame {meta.get('index', '?')} rate streak PSF — "
                 f"{meta.get('exposure', '?')}s, n={n_txt}, "
                 f"length={length:.1f}px, width(FWHM)={sh_across['fwhm']:.1f}px{sec}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, png_path)


# --------------------------------------------------------------------------
# high-level entry points (data in -> npy + png out)
# --------------------------------------------------------------------------
def make_sidereal_psf(
    data: np.ndarray,
    stars: list[StarTuple],
    astropy_wcs: WCS | None,
    fwhm: float,
    meta: dict,
    png_path: Path | str,
    npy_path: Path | str | None = None,
) -> bool:
    """Stack stars into a sidereal PSF and render the panel to disk.

    Args:
        data: Full frame pixel data.
        stars: Catalog stars as ``(x, y, mag)`` tuples.
        astropy_wcs: Astropy WCS used to orient the RA/Dec cuts, or ``None``.
        fwhm: Estimated point-source FWHM in pixels.
        meta: Frame metadata dict (``index``, ``exposure``, ``pixel_scale_arcsec``).
        png_path: Destination PNG path for the panel.
        npy_path: Optional path to also save the stacked stamp as ``.npy``.

    Returns:
        ``True`` if a panel was rendered, ``False`` if too few stars were found.
    """
    stamp, n = stack_stars(data, stars, fwhm)
    if stamp is None:
        logger.info("psf: frame %s too few stars for sidereal PSF", meta.get("index"))
        return False
    if npy_path is not None:
        np.save(str(npy_path), stamp.astype(np.float32))
    render_sidereal_psf(stamp, n, sky_axes(astropy_wcs), meta, png_path)
    return True


def make_streak_psf(
    data: np.ndarray,
    stars: list[StarTuple],
    astropy_wcs: WCS | None,
    fwhm: float,
    length: float,
    angle_deg: float,
    meta: dict,
    png_path: Path | str,
    npy_path: Path | str | None = None,
) -> bool:
    """Stack streaks into a streak PSF and render the panel to disk.

    Args:
        data: Full frame pixel data.
        stars: Catalog stars as ``(x, y, mag)`` tuples.
        astropy_wcs: Astropy WCS used to orient the N/E arrows, or ``None``.
        fwhm: Cross-streak FWHM in pixels.
        length: Streak length in pixels.
        angle_deg: Streak position angle in degrees.
        meta: Frame metadata dict (``index``, ``exposure``, ``pixel_scale_arcsec``).
        png_path: Destination PNG path for the panel.
        npy_path: Optional path to also save the stacked stamp as ``.npy``.

    Returns:
        ``True`` if a panel was rendered, ``False`` if too few streaks were found.
    """
    stamp, half_a, half_p, n = stack_streaks(data, stars, fwhm, length, angle_deg)
    if stamp is None:
        logger.info("psf: frame %s too few streaks for streak PSF", meta.get("index"))
        return False
    if npy_path is not None:
        np.save(str(npy_path), stamp.astype(np.float32))
    # express N/E in the streak-aligned frame (rotate pixel axes by -angle)
    sis = None
    ax = sky_axes(astropy_wcs)
    if ax is not None:
        ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        rot = lambda u: np.array([ca * u[0] + sa * u[1], -sa * u[0] + ca * u[1]])
        sis = (rot(ax[0]), rot(ax[1]))
    render_streak_psf(stamp, half_a, half_p, n, length, fwhm, sis, meta, png_path)
    return True


# --------------------------------------------------------------------------
# in-memory frame adapters (duck-typed: SiderealFrame / RateTrackFrame)
# --------------------------------------------------------------------------
def _stars(sf: StarField) -> list[StarTuple]:
    """Extract ``(x, y, magnitude)`` tuples from a starfield's catalog stars.

    Args:
        sf: The solved starfield.

    Returns:
        A list of ``(x, y, magnitude)`` tuples (empty if no catalog stars).
    """
    return [(s.x, s.y, s.magnitude) for s in (sf.catalog_stars or [])]


def _astropy_wcs(sf: StarField) -> WCS | None:
    """Convert a starfield's WCS model to an astropy WCS.

    Args:
        sf: The solved starfield.

    Returns:
        The astropy WCS, or ``None`` if unset or the conversion fails.
    """
    try:
        return sf.wcs.to_astropy_wcs() if sf.wcs is not None else None
    except Exception:
        return None


def _plate_scale(astropy_wcs: WCS | None) -> float | None:
    """Compute the mean pixel scale in arcseconds from a WCS.

    Args:
        astropy_wcs: Astropy WCS, or ``None``.

    Returns:
        The mean plate scale in arcsec/pixel, or ``None`` if unavailable.
    """
    if astropy_wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        return float(np.mean(proj_plane_pixel_scales(astropy_wcs)) * 3600.0)
    except Exception:
        return None


def _exposure(frame: Frame) -> float | None:
    """Read the exposure time in seconds from a frame's metadata.

    Args:
        frame: A sidereal or rate-track frame.

    Returns:
        The exposure time in seconds, or ``None`` if unavailable.
    """
    fm = getattr(frame, "frame_metadata", None)
    return getattr(fm, "exposure_time_seconds", None) if fm else None


def plot_sidereal_frame(
    frame: Frame, png_path: Path | str, npy_path: Path | str | None = None
) -> bool:
    """Render the sidereal PSF panel for an in-memory frame.

    Args:
        frame: A sidereal frame carrying a solved starfield.
        png_path: Destination PNG path for the panel.
        npy_path: Optional path to also save the stacked stamp as ``.npy``.

    Returns:
        ``True`` if a panel was rendered, ``False`` otherwise.
    """
    sf = getattr(frame, "starfield", None)
    if sf is None or not sf.catalog_stars:
        return False
    wcs = _astropy_wcs(sf)
    fwhm = (getattr(getattr(frame, "seeing", None), "pixel_fwhm", None)
            or (sf.fwhm_stats.median_fwhm if sf.fwhm_stats else None) or 4.0)
    meta = {"index": frame.index, "exposure": _exposure(frame),
            "pixel_scale_arcsec": _plate_scale(wcs)}
    return make_sidereal_psf(frame.frame.data, _stars(sf), wcs, float(fwhm), meta,
                             png_path, npy_path)


def plot_rate_frame(
    frame: Frame, png_path: Path | str, npy_path: Path | str | None = None
) -> bool:
    """Render the streak PSF panel for an in-memory rate-track frame.

    Args:
        frame: A rate-track frame carrying a solved starfield and streak.
        png_path: Destination PNG path for the panel.
        npy_path: Optional path to also save the stacked stamp as ``.npy``.

    Returns:
        ``True`` if a panel was rendered, ``False`` otherwise.
    """
    sf = getattr(frame, "starfield", None)
    st = getattr(frame, "streak", None)
    if sf is None or not sf.catalog_stars or st is None or not st.pixel_length:
        return False
    wcs = _astropy_wcs(sf)
    meta = {"index": frame.index, "exposure": _exposure(frame),
            "pixel_scale_arcsec": _plate_scale(wcs)}
    return make_streak_psf(frame.frame.data, _stars(sf), wcs, float(st.fwhm),
                           float(st.pixel_length), float(st.degree_angle()), meta,
                           png_path, npy_path)


# --------------------------------------------------------------------------
# regenerate a panel from a saved .npy stamp (no raw FITS needed)
# --------------------------------------------------------------------------
def sidereal_from_stamp(
    stamp: np.ndarray, astropy_wcs: WCS | None, meta: dict, png_path: Path | str
) -> None:
    """Render a sidereal PSF panel from a previously saved stamp.

    Args:
        stamp: Stacked point-source PSF stamp loaded from ``.npy``.
        astropy_wcs: Astropy WCS used to orient the RA/Dec cuts, or ``None``.
        meta: Frame metadata dict (``index``, ``exposure``, ``pixel_scale_arcsec``).
        png_path: Destination PNG path.
    """
    render_sidereal_psf(stamp, None, sky_axes(astropy_wcs), meta, png_path)


def streak_from_stamp(
    stamp: np.ndarray,
    astropy_wcs: WCS | None,
    fwhm: float,
    length: float,
    angle_deg: float,
    meta: dict,
    png_path: Path | str,
) -> None:
    """Render a streak PSF panel from a previously saved stamp.

    Args:
        stamp: Stacked streak stamp loaded from ``.npy`` (indexed ``[perp, along]``).
        astropy_wcs: Astropy WCS used to orient the N/E arrows, or ``None``.
        fwhm: Fitted cross-streak FWHM in pixels.
        length: Fitted streak length in pixels.
        angle_deg: Streak position angle in degrees.
        meta: Frame metadata dict (``index``, ``exposure``, ``pixel_scale_arcsec``).
        png_path: Destination PNG path.
    """
    half_p = stamp.shape[0] // 2
    half_a = stamp.shape[1] // 2
    sis = None
    ax = sky_axes(astropy_wcs)
    if ax is not None:
        ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        rot = lambda u: np.array([ca * u[0] + sa * u[1], -sa * u[0] + ca * u[1]])
        sis = (rot(ax[0]), rot(ax[1]))
    render_streak_psf(stamp, half_a, half_p, None, length, fwhm, sis, meta, png_path)
