#!/usr/bin/env python3
r"""Measure detector gain (e-/ADU) from raw frame pairs via photon transfer.

Pairs consecutive same-field exposures (a burst), differences each pair to cancel
fixed pattern, and fits the lower envelope of difference-variance vs sky level --
``gain = 1 / slope`` -- which needs no bias frame and is robust to the moving
stars a rate-tracked burst leaves behind (they only add variance; see
:mod:`senpai.engine.observability.detector_gain`).

Usage:
    python -m senpai.cli.measure_gain /path/to/raw/frames \\
        --out gain_ptc.png --max-pairs 50

    # restrict to sidereal bursts (cleanest cancellation):
    python -m senpai.cli.measure_gain /path/to/frames --field-substr coverage
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from senpai.engine.observability.detector_gain import (
    find_burst_pairs,
    fit_gain,
    plot_ptc,
    ptc_point,
)

logger = logging.getLogger(__name__)


def _gather_frames(inputs: list[str]) -> list[str]:
    """Expand directories/globs into a sorted list of FITS paths."""
    out: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(str(x) for x in sorted(p.glob("*.fit*")))
        else:
            out.extend(sorted(glob.glob(item)))
    return sorted(set(out))


def _read_center(path: str, crop: int) -> np.ndarray | None:
    """Read a central ``crop``×``crop`` window.

    These frames carry BZERO/BSCALE (offset-stored uint16), which rules out a
    memory-mapped section read, so we load the (scaled) array and crop the
    centre — also avoiding edge vignetting.
    """
    try:
        with fits.open(path, memmap=False) as h:
            data = h[0].data
            ny, nx = data.shape
            half = min(crop, ny, nx) // 2
            cy, cx = ny // 2, nx // 2
            return np.asarray(
                data[cy - half:cy + half, cx - half:cx + half], dtype=np.float64)
    except Exception as e:  # unreadable / truncated frame
        logger.warning("skipping %s: %s", Path(path).name, e)
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the gain-measurement CLI over the given frames.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv`` when None.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    ap = argparse.ArgumentParser(
        description="Measure detector gain (e-/ADU) from raw frame pairs.")
    ap.add_argument("inputs", nargs="+",
                    help="directories and/or FITS globs of raw frames")
    ap.add_argument("--out", default="gain_ptc.png", help="output plot path")
    ap.add_argument("--max-pairs", type=int, default=50,
                    help="max same-field pairs to sample (spread over the night)")
    ap.add_argument("--crop", type=int, default=2048,
                    help="central window side in pixels read per frame")
    ap.add_argument("--patch", type=int, default=128,
                    help="patch side (px) for the star-free sky-variance estimate")
    ap.add_argument("--field-substr", default=None,
                    help="only use frames whose field token contains this string "
                         "(e.g. 'coverage' for sidereal bursts)")
    ap.add_argument("--write-night", default=None, metavar="NIGHT_JSON",
                    help="patch the measured gain into this night_calibration.json "
                         "(conditions.gain_e_per_adu_*), without reprocessing")
    ap.add_argument("--no-plot", action="store_true", help="skip the PTC plot")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s")

    frames = _gather_frames(args.inputs)
    if args.field_substr:
        frames = [f for f in frames if args.field_substr in Path(f).stem]
    pairs = find_burst_pairs(frames)
    if not pairs:
        logger.error("no same-field consecutive pairs found in %d frames",
                     len(frames))
        return 1

    # Sample evenly across the (time-sorted) pairs so the sky-level range of the
    # whole night is represented -- that range is what pins the PTC slope.
    if len(pairs) > args.max_pairs:
        idx = np.linspace(0, len(pairs) - 1, args.max_pairs).round().astype(int)
        pairs = [pairs[i] for i in dict.fromkeys(idx)]

    logger.info("found %d burst pairs in %d frames; measuring %d",
                len(find_burst_pairs(frames)), len(frames), len(pairs))

    points = []
    for i, (p1, p2) in enumerate(pairs, 1):
        a = _read_center(p1, args.crop)
        b = _read_center(p2, args.crop)
        if a is None or b is None or a.shape != b.shape:
            continue
        pt = ptc_point(a, b, patch=args.patch)
        if pt is not None:
            points.append(pt)
        if args.verbose and pt is not None:
            logger.debug("  pair %d/%d %-28s level=%.0f ADU  var=%.1f",
                         i, len(pairs), Path(p1).stem[:28], pt[0], pt[1])

    fit = fit_gain(points)
    if fit is None:
        logger.error("could not fit gain from %d usable points "
                     "(need a range of sky levels)", len(points))
        return 1

    lvls = np.array(fit.levels)
    inv = 1.0 / fit.gain
    inv_lo, inv_hi = 1.0 / fit.gain_hi, 1.0 / fit.gain_lo  # ADU/e- inverts order
    print("\n=== detector gain (photon transfer, frame-pair difference) ===")
    print(f"  usable pairs        : {fit.n_pairs}")
    print(f"  sky level range     : {lvls.min():.0f} – {lvls.max():.0f} ADU")
    print(f"  GAIN                : {fit.gain:.3f} e-/ADU   "
          f"(95% CI {fit.gain_lo:.3f}–{fit.gain_hi:.3f})")
    print(f"                      : {inv:.3f} ADU/e-   "
          f"(95% CI {inv_lo:.3f}–{inv_hi:.3f})")
    print(f"  PTC intercept       : {fit.intercept:.1f} ADU²  "
          f"(= read² − bias/gain; bias frames needed to split it)")

    if not args.no_plot:
        out = plot_ptc(fit, args.out,
                       title=f"detector gain: {Path(args.inputs[0]).name}")
        print(f"  plot                : {out}")

    if args.write_night:
        njson = Path(args.write_night)
        data = json.loads(njson.read_text())
        cond = data.setdefault("conditions", {})
        cond["gain_e_per_adu_median"] = round(fit.gain, 4)
        cond["gain_e_per_adu_std"] = round(0.5 * (fit.gain_hi - fit.gain_lo), 4)
        cond["n_frames_gain"] = fit.n_pairs
        cond["gain_method"] = "ptc_frame_pair_cli"
        njson.write_text(json.dumps(data, indent=2))
        print(f"  wrote gain to       : {njson} "
              f"(conditions.gain_e_per_adu_median = {fit.gain:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
