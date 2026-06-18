"""Measure detector gain (electrons per ADU) from raw frame pairs.

Photon transfer, the bias-free way. For two raw frames of the *same* field
(consecutive exposures in a burst), the difference cancels everything fixed --
pixel-response non-uniformity (PRNU), bias structure, and the stars themselves
(same pixels) -- leaving only shot + read noise. The per-pixel variance of that
difference is

    var_ADU = signal_ADU / gain + read_ADU**2 ,

where signal_ADU is the sky level above bias. Measuring (level, variance) across
many pairs spanning the night's range of sky levels and fitting a line gives

    gain = 1 / slope ,

which needs no bias frame and no knowledge of the offset: the bias only shifts
the line's intercept, not its slope. This is the classic photon-transfer curve
(PTC) restricted to its slope.

Tracking matters. A *sidereal* burst keeps stars on the same pixels, so they
cancel in the difference. A *rate-tracked* burst (tracking a moving target) lets
the stars move between exposures, so they do NOT cancel and leave residuals that
inflate the difference variance. The fix is to measure the variance only where
there are no stars: tile the difference into patches and keep the cleanest
(lowest-variance) ones -- pure sky shot+read noise -- which works regardless of
tracking mode and needs no source catalog. Each pair then yields one clean PTC
point, and a Theil-Sen fit over the night's range of sky levels gives the gain.
(In the in-pipeline per-frame estimate, where a WCS + source catalog exist,
masking sources and measuring the sky directly is the equivalent route.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# <timestamp>_<field tokens...>_f<index>.fits
_NAME_RE = re.compile(r"^(?P<ts>[^_]+)_(?P<field>.+)_f(?P<idx>\d+)$")


@dataclass
class FrameKey:
    path: Path
    timestamp: str
    field: str
    f_index: int


def parse_frame_key(path: str | Path) -> FrameKey | None:
    """Parse ``<timestamp>_<field>_f<index>.fits`` into its burst coordinates."""
    p = Path(path)
    m = _NAME_RE.match(p.stem)
    if not m:
        return None
    return FrameKey(p, m["ts"], m["field"], int(m["idx"]))


def find_burst_pairs(paths) -> list[tuple[Path, Path]]:
    """Consecutive same-field exposures (a burst) -> difference pairs.

    Two time-adjacent frames pair only when they share a field token and their
    f-index increments by one, i.e. ``..._f0`` then ``..._f1`` of the same
    target. Repeated ``_f0`` tiles at different times (e.g. a coverage scan) are
    *not* paired -- they are different fields and would not difference cleanly.
    """
    keys = [k for k in (parse_frame_key(p) for p in paths) if k is not None]
    keys.sort(key=lambda k: (k.timestamp, k.f_index))
    pairs: list[tuple[Path, Path]] = []
    for a, b in zip(keys, keys[1:]):
        if a.field == b.field and b.f_index == a.f_index + 1:
            pairs.append((a.path, b.path))
    return pairs


def _robust_sigma(a: np.ndarray) -> float:
    """MAD-based robust standard deviation (rejects stars / outliers)."""
    a = a[np.isfinite(a)]
    if a.size < 100:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))) * 1.4826)


def _clean_sky_sigma(diff: np.ndarray, patch: int = 128,
                     sky_pctile: float = 10.0) -> float:
    """Sky-only per-pixel std of a difference image, from the cleanest patches.

    Stars (and their non-cancelling residuals under rate tracking) are localized,
    so they raise the noise only in the patches that contain them. Tiling the
    difference into ``patch``×``patch`` blocks and taking a low percentile of the
    per-block robust std picks the star-free sky blocks -- the true shot+read
    noise -- regardless of how badly the stars cancelled. Falls back to a global
    MAD when the frame is too small to tile.
    """
    H, W = diff.shape
    p = max(8, min(int(patch), H // 8, W // 8))
    ny, nx = H // p, W // p
    if ny < 2 or nx < 2:
        return _robust_sigma(diff)
    blocks = (diff[:ny * p, :nx * p]
              .reshape(ny, p, nx, p).transpose(0, 2, 1, 3)
              .reshape(ny * nx, p * p))
    med = np.median(blocks, axis=1, keepdims=True)
    mad = np.median(np.abs(blocks - med), axis=1) * 1.4826
    mad = mad[np.isfinite(mad) & (mad > 0)]
    if mad.size < 4:
        return float("nan")
    return float(np.percentile(mad, sky_pctile))


def ptc_point(frame1: np.ndarray, frame2: np.ndarray,
              patch: int = 128) -> tuple[float, float] | None:
    """One PTC point ``(level_ADU, sky per-pixel var_ADU)`` from a same-field pair.

    The level is the mean of the two frame medians (sky); the variance is the
    star-free sky variance of the difference, ``var(frame1 - frame2) / 2``,
    measured from the cleanest patches (:func:`_clean_sky_sigma`) so star
    residuals -- which do not cancel under rate tracking -- are excluded. uint16
    frames are promoted before differencing so the subtraction does not wrap.
    Returns ``None`` for a degenerate pair.
    """
    a = np.asarray(frame1, dtype=np.float64)
    b = np.asarray(frame2, dtype=np.float64)
    if a.shape != b.shape:
        return None
    level = 0.5 * (float(np.median(a)) + float(np.median(b)))
    sigma_diff = _clean_sky_sigma(a - b, patch=patch)
    if not np.isfinite(sigma_diff) or sigma_diff <= 0.0 or level <= 0.0:
        return None
    var_pixel = 0.5 * sigma_diff * sigma_diff
    return level, var_pixel


@dataclass
class GainFit:
    gain: float                 # e-/ADU = 1 / slope
    gain_lo: float              # from the Theil-Sen 95% slope interval
    gain_hi: float
    slope: float                # var-vs-level slope = 1 / gain
    intercept: float            # = read_ADU**2 - bias_ADU / gain
    n_pairs: int
    levels: list[float] = field(default_factory=list)       # all PTC points
    variances: list[float] = field(default_factory=list)
    env_levels: list[float] = field(default_factory=list)   # lower-envelope pts
    env_variances: list[float] = field(default_factory=list)


def fit_gain(points: list[tuple[float, float]]) -> GainFit | None:
    """Theil-Sen fit of sky variance vs level over the PTC -> gain = 1/slope.

    With each point's variance already restricted to star-free sky
    (:func:`ptc_point`), the cloud collapses onto the photon-transfer line. A
    Theil-Sen fit gives a robust slope (and a 95% interval), then a one-sided
    sigma-clip drops points sitting ABOVE the line -- residual contamination
    only raises variance, so the clean shot line is the lower edge -- and refits.
    Needs points spanning a range of sky levels; a flat PTC cannot pin a slope.
    """
    pts = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)
           and x > 0 and y > 0]
    if len(pts) < 5:
        return None
    levels = np.array([p[0] for p in pts])
    variances = np.array([p[1] for p in pts])
    if float(levels.max() - levels.min()) < 1e-6:
        return None  # no lever arm in level -> slope undefined

    from scipy.stats import theilslopes
    # Contamination (residual stars, PRNU that doesn't fully cancel at high sky)
    # can only *raise* a point's variance, so the true shot line is the lower
    # edge. Iteratively reject points sitting ABOVE the fit (one-sided clip) so
    # those high-leverage, slightly-contaminated points don't steepen the slope.
    keep = np.ones(len(levels), dtype=bool)
    slope, intercept, lo_slope, hi_slope = theilslopes(variances, levels)
    for _ in range(3):
        fit_val = slope * levels + intercept
        # Fractional excess above the fit -- scale-invariant, since PTC variance
        # scatter grows with level (an absolute threshold set by the low-sky
        # cluster would clip every high-sky point). Points at/below the bias
        # floor (fit_val <= 0) are kept; only clearly-high points are dropped.
        excess = np.where(fit_val > 0, (variances - fit_val) / fit_val, -np.inf)
        new_keep = excess < 0.30
        if new_keep.sum() < 5 or new_keep.sum() == keep.sum():
            break
        keep = new_keep
        slope, intercept, lo_slope, hi_slope = theilslopes(
            variances[keep], levels[keep])
    if slope <= 0.0:
        return None
    gain = 1.0 / slope
    # Map the slope confidence interval to gain (slope and gain invert order).
    gain_lo = 1.0 / hi_slope if hi_slope > 0 else float("nan")
    gain_hi = 1.0 / lo_slope if lo_slope > 0 else float("nan")
    return GainFit(
        gain=gain, gain_lo=gain_lo, gain_hi=gain_hi,
        slope=slope, intercept=float(intercept), n_pairs=int(keep.sum()),
        levels=levels.tolist(), variances=variances.tolist(),
        env_levels=levels[keep].tolist(), env_variances=variances[keep].tolist(),
    )


def plot_ptc(fit: GainFit, output_path: str | Path,
             title: str = "detector gain (photon transfer)") -> Path:
    """Render the PTC: variance vs level, the fit line, and the recovered gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = np.array(fit.levels)
    kept = set(zip(fit.env_levels, fit.env_variances))
    is_kept = np.array([(x, y) in kept for x, y in zip(fit.levels, fit.variances)])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(levels[is_kept], np.array(fit.variances)[is_kept], s=22,
               alpha=0.7, color="tab:blue",
               label=f"fit pairs, star-free sky (n={fit.n_pairs})")
    if (~is_kept).any():
        ax.scatter(levels[~is_kept], np.array(fit.variances)[~is_kept], s=30,
                   facecolors="none", edgecolors="tab:red",
                   label="clipped (above line: residual contamination)")
    xs = np.linspace(levels.min(), levels.max(), 200)
    yfit = fit.slope * xs + fit.intercept
    inv = 1.0 / fit.gain
    pos = yfit > 0  # negative intercept -> line dips below 0 near the bias floor
    ax.plot(xs[pos], yfit[pos], "-", color="black", lw=1.8,
            label=f"fit: gain = {fit.gain:.3f} e-/ADU = {inv:.3f} ADU/e-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sky level (ADU)")
    ax.set_ylabel("difference variance / 2  (ADU²)")
    ax.set_title(title + "\nphoton transfer: slope = 1/gain "
                 "(star-free patches, one-sided clip)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
