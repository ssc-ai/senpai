"""Layered track-mode classification: metadata first, pixels only as a fallback.

A frame is either **sidereal** (mount tracks the stars; stars are round, any
target streaks) or **rate** (mount tracks a moving target; stars streak). We
decide in cheap-to-expensive tiers and stop at the first that is confident:

1. **TRKMODE** header — authoritative on burr/DAO (records what the mount
   actually did). If present and unambiguous, we're done; no pixels touched.
2. **RA/DEC rate magnitude** — if there's no TRKMODE but there are tracking
   rates, |rate| ~ 0 reads sidereal, otherwise rate.
3. **Pixels** — only when the header can't decide (no TRKMODE) do we look at the
   image: round sources -> sidereal, long *and mutually aligned* streaks -> rate.

Because rate header values can be unreliable, when we fall past TRKMODE the
**pixels arbitrate**: tiers 2 and 3 are both computed, and on disagreement the
pixels win (and we log it, which surfaces bad rate metadata). Tier 3 never runs
when a TRKMODE is present, so metadata-tagged frames pay nothing for it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from senpai.engine.models.metadata import TrackMode

logger = logging.getLogger(__name__)

# --- pixel-test tuning --------------------------------------------------------
_BRIGHT_SIGMA = 8.0        # sources are connected components above med + N*sigma
_MIN_BLOB_PX = 6           # reject cosmic rays / hot pixels
_MIN_SOURCES = 8           # too few measurable sources -> inconclusive
_MAX_SOURCES = 60          # measure at most this many (brightest first)
_ELONG_ROUND_MAX = 1.5     # median axis ratio <= this -> round -> sidereal
_ELONG_STREAK_MIN = 1.8    # median axis ratio >= this (and aligned) -> rate
_PA_ALIGN_MIN = 0.6        # axial concentration of streak position angles (0..1)
# Calibrated on DAO01 1s frames (median source axis ratio): sidereal ~1.08,
# slow-rate coverage ~1.22 (apparent motion < a pixel -> reads round, fine to
# treat as sidereal), moderate-rate coverage ~2.17, fast calsat streak ~20. The
# 1.5/1.8 band leaves the ambiguous slow-rate case as "round" and only calls
# rate once the trail is unmistakable. This path runs ONLY when a frame has no
# TRKMODE, so it never touches metadata-tagged (e.g. all burr) data.


@dataclass(frozen=True)
class ImageTrackVerdict:
    """Pixel-test outcome. ``mode`` is UNKNOWN when the image is inconclusive."""

    mode: TrackMode
    confidence: float
    n_sources: int
    median_elongation: float
    pa_alignment: float


@dataclass(frozen=True)
class TrackModeDecision:
    """Final classification plus *which* tier decided it (for logging)."""

    mode: TrackMode
    source: str   # "trkmode" | "rates" | "pixels" | "pixels>rates" | "unknown"
    detail: str = ""


def _blob_elongation(stamp: np.ndarray) -> tuple[float, float] | None:
    """Axis ratio (major/minor) and position angle (rad) from a blob's intensity
    second moments. Returns None for a degenerate blob."""
    tot = float(stamp.sum())
    if tot <= 0:
        return None
    h, w = stamp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = float((stamp * yy).sum() / tot)
    cx = float((stamp * xx).sum() / tot)
    mxx = float((stamp * (xx - cx) ** 2).sum() / tot)
    myy = float((stamp * (yy - cy) ** 2).sum() / tot)
    mxy = float((stamp * (xx - cx) * (yy - cy)).sum() / tot)
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + math.sqrt(disc)
    l2 = tr / 2.0 - math.sqrt(disc)
    if l2 <= 1e-6:
        return None
    elong = math.sqrt(l1 / l2)
    pa = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
    return elong, pa


def infer_track_mode_from_image(
    data: np.ndarray, max_sources: int = _MAX_SOURCES
) -> ImageTrackVerdict:
    """Round sources -> sidereal, mutually aligned streaks -> rate, else UNKNOWN.

    Rate-independent on purpose: it measures source *shape* directly, so it does
    not inherit any error in the header tracking rates."""
    a = np.asarray(data, dtype=np.float32)
    if a.ndim != 2:
        return ImageTrackVerdict(TrackMode.UNKNOWN, 0.0, 0, float("nan"), float("nan"))
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) or 1.0
    sigma = 1.4826 * mad
    mask = a > med + _BRIGHT_SIGMA * sigma
    if not mask.any():
        return ImageTrackVerdict(TrackMode.UNKNOWN, 0.0, 0, float("nan"), float("nan"))

    lbl, n = ndimage.label(mask)
    if n == 0:
        return ImageTrackVerdict(TrackMode.UNKNOWN, 0.0, 0, float("nan"), float("nan"))
    objs = ndimage.find_objects(lbl)
    a0 = a - med
    h, w = a.shape

    # rank components by peak brightness, measure the brightest up to max_sources
    cand = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        # drop edge-clipped blobs (a real streak/star fully in-frame is what we want)
        if ys.start == 0 or xs.start == 0 or ys.stop == h or xs.stop == w:
            continue
        sub = a0[sl]
        comp = lbl[sl] == i
        if int(comp.sum()) < _MIN_BLOB_PX:
            continue
        peak = float(sub[comp].max())
        cand.append((peak, sl, comp))
    cand.sort(key=lambda c: -c[0])

    elongs: list[float] = []
    angles: list[float] = []
    for _peak, sl, comp in cand[:max_sources]:
        stamp = np.where(comp, a0[sl], 0.0)
        res = _blob_elongation(stamp)
        if res is None:
            continue
        elong, pa = res
        elongs.append(elong)
        angles.append(pa)

    if len(elongs) < _MIN_SOURCES:
        return ImageTrackVerdict(TrackMode.UNKNOWN, 0.0, len(elongs),
                                 float("nan"), float("nan"))

    med_elong = float(np.median(elongs))
    # Position angle is axial (mod pi): use the mean resultant length of 2*PA as
    # an alignment score. Streaks from a single rate share a PA -> ~1; round
    # sources have random PA -> ~0.
    ang = np.asarray(angles)
    pa_align = float(abs(np.mean(np.exp(2j * ang))))

    if med_elong <= _ELONG_ROUND_MAX:
        conf = float(min(1.0, (_ELONG_ROUND_MAX - med_elong) / _ELONG_ROUND_MAX + 0.5))
        return ImageTrackVerdict(TrackMode.SIDEREAL, conf, len(elongs), med_elong, pa_align)
    if med_elong >= _ELONG_STREAK_MIN and pa_align >= _PA_ALIGN_MIN:
        conf = float(min(1.0, 0.5 + 0.5 * pa_align))
        return ImageTrackVerdict(TrackMode.RATE, conf, len(elongs), med_elong, pa_align)
    # elongated but not aligned (crowding/aberration) or mildly elongated: punt
    return ImageTrackVerdict(TrackMode.UNKNOWN, 0.0, len(elongs), med_elong, pa_align)


def _header_trkmode(header, mode_keys) -> TrackMode | None:
    """Explicit, unambiguous TRKMODE from the header, or None."""
    from senpai.engine.utils.fits_io import extract_header_value

    for k in mode_keys:
        v = extract_header_value(header, k)
        if v is None:
            continue
        s = str(v).strip().lower()
        has_rate, has_sid = "rate" in s, "sidereal" in s
        if has_rate and not has_sid:
            return TrackMode.RATE
        if has_sid and not has_rate:
            return TrackMode.SIDEREAL
        # 'fixed', 'both', '' -> ambiguous; let later tiers decide
        return None
    return None


def classify_track_mode(header, data=None, config=None) -> TrackModeDecision:
    """Classify a frame sidereal vs rate, cheapest evidence first (see module
    docstring). ``data`` (the 2-D image) enables the pixel arbiter; omit it to
    stay metadata-only."""
    from senpai.engine.utils.fits_io import extract_track_rates_from_header

    if config is None:
        from senpai.core.config import get_config
        config = get_config()
    tcfg = config.headers.tracking
    mode_keys = tcfg.track_mode_keys or ["TRKMODE"]

    # Tier 1 — authoritative TRKMODE.
    trk = _header_trkmode(header, mode_keys)
    if trk is not None:
        return TrackModeDecision(trk, "trkmode")

    # Tier 2 — rate magnitude (rates may be unreliable, so this only *proposes*).
    ra_rate, dec_rate, _ = extract_track_rates_from_header(header)
    rate_guess: TrackMode | None = None
    if ra_rate is not None and dec_rate is not None:
        thr = tcfg.sidereal_rate_threshold_arcsec_per_second
        mag = math.hypot(ra_rate, dec_rate)
        rate_guess = TrackMode.SIDEREAL if mag <= thr else TrackMode.RATE

    # Tier 3 — pixels arbitrate (rate-independent).
    pix: TrackMode | None = None
    pix_detail = ""
    if tcfg.data_fallback_enabled and data is not None:
        v = infer_track_mode_from_image(data)
        pix_detail = (f"elong={v.median_elongation:.2f} align={v.pa_alignment:.2f} "
                      f"n={v.n_sources}")
        if v.mode in (TrackMode.RATE, TrackMode.SIDEREAL):
            pix = v.mode

    if pix is not None and rate_guess is not None:
        if pix == rate_guess:
            return TrackModeDecision(pix, "pixels", pix_detail)
        logger.warning(
            "track-mode disagreement: rates say %s but pixels say %s (%s) — trusting "
            "pixels; check the frame's RA/DEC_RATE metadata",
            rate_guess.value, pix.value, pix_detail,
        )
        return TrackModeDecision(pix, "pixels>rates", pix_detail)
    if pix is not None:
        return TrackModeDecision(pix, "pixels", pix_detail)
    if rate_guess is not None:
        return TrackModeDecision(rate_guess, "rates",
                                 f"|rate|=({ra_rate:.2f},{dec_rate:.2f})\"/s")
    return TrackModeDecision(TrackMode.UNKNOWN, "unknown", pix_detail)


# --------------------------------------------------------------------------
# CLI: classify a frame and show each tier's evidence
#   python -m senpai.engine.detection.track_mode <fits...> [--from-data] [--raw]
# --------------------------------------------------------------------------
def _main(argv=None) -> int:
    import argparse
    import glob
    from pathlib import Path

    from astropy.io import fits

    from senpai.core.config import get_config, initialize_config
    from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
    from senpai.engine.utils.fits_io import extract_track_rates_from_header

    p = argparse.ArgumentParser(
        description="Classify a frame sidereal-vs-rate and show every tier's "
        "evidence: the TRKMODE header, the RA/DEC rates, and the from-data pixel "
        "measurement (source elongation + alignment).")
    p.add_argument("fits", nargs="+", help="FITS file(s) or glob(s).")
    p.add_argument("-c", "--config", default=str(LOCAL_APP_CONFIG_OVERRIDE),
                   help="Config YAML (header keys + thresholds).")
    p.add_argument("--from-data", action="store_true",
                   help="Force the decision from the pixel measurement, ignoring "
                        "TRKMODE/rates (see what the data alone says).")
    p.add_argument("--raw", action="store_true",
                   help="Skip the row/column-median subtraction the pipeline applies "
                        "before classifying (use the raw pixels).")
    args = p.parse_args(argv)

    initialize_config(Path(args.config))
    config = get_config()
    mode_keys = config.headers.tracking.track_mode_keys or ["TRKMODE"]
    thr = config.headers.tracking.sidereal_rate_threshold_arcsec_per_second

    paths: list[str] = []
    for f in args.fits:
        paths += sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f]
    if not paths:
        p.error("no FITS files matched")

    for path in paths:
        with fits.open(path) as hdul:
            header = hdul[0].header
            data = np.asarray(hdul[0].data, dtype=np.float32)
        if not args.raw:
            data = data - np.median(data, axis=0)[None, :]
            data = data - np.median(data, axis=1)[:, None]

        trk = _header_trkmode(header, mode_keys)
        ra, dec, _ = extract_track_rates_from_header(header)
        verdict = infer_track_mode_from_image(data)

        print(f"\n{Path(path).name}")
        print(f"  TRKMODE header : {str(header.get('TRKMODE')):>10}  -> "
              f"{trk.value if trk else '(unusable)'}")
        if ra is not None and dec is not None:
            mag = math.hypot(ra, dec)
            print(f"  RA/DEC rates   : ({ra:7.2f},{dec:7.2f})\"/s |rate|={mag:6.1f}  -> "
                  f"{'rate' if mag > thr else 'sidereal'}")
        else:
            print("  RA/DEC rates   :   (absent)")
        print(f"  pixel measure  : {verdict.mode.value:>10}  elong={verdict.median_elongation:.2f} "
              f"align={verdict.pa_alignment:.2f} n={verdict.n_sources}")

        if args.from_data:
            mode = verdict.mode if verdict.mode != TrackMode.UNKNOWN else TrackMode.SIDEREAL
            note = "" if verdict.mode != TrackMode.UNKNOWN else " (inconclusive->sidereal default)"
            print(f"  => DECISION    : {mode.value}  [forced from data]{note}")
        else:
            d = classify_track_mode(header, data, config)
            print(f"  => DECISION    : {d.mode.value}  via {d.source}"
                  + (f"  [{d.detail}]" if d.detail else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
