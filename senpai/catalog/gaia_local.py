"""Offline Gaia queries against a local mirror built by senpai.catalog.gaia_mirror.

Drop-in for senpai.catalog.gaia.query_by_ra_dec_bounds: same signature, same
star-dict shape (ra/dec in radians, mv, magnitudes incl. synthetic Johnson_V /
Sloan_r, source_id, proper motion), so it slots into catalog.runner unchanged and
the in-process sliver cache still wraps it. Reads only the HEALPix tiles whose
bbox (from index.json) overlaps the requested box — sub-second per field.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from astroeasy.catalog.mirror import query_mirror_box

logger = logging.getLogger(__name__)

# Bound dict-building on ultra-dense (galactic-plane) fields: a single frame
# there can contain millions of stars (observed 2.57M), and building a dict per
# row then projecting/isolating them all before the caller's magnitude-
# stratified max_stars_per_frame cap (applied downstream) peaked ~26 GB and
# drew the OOM killer. Keep the brightest MAX_LOCAL_ROWS here; the downstream
# stratified cap subsamples this for completeness. The cut is far fainter than
# any per-frame cap, so normal fields are untouched.
MAX_LOCAL_ROWS = 200_000


def query_by_ra_dec_bounds(
    min_ra: float, max_ra: float, min_dec: float, max_dec: float,
    faint_lim: float | None = None, bright_lim: float | None = None,
    primary_filter: str = "G", *, mirror_dir: str,
) -> list[dict[str, Any]]:
    """Stars from the local mirror within the RA/Dec box and magnitude limits.

    Tile selection and reading are delegated to astroeasy's mirror reader
    (astroeasy.catalog.mirror); this wrapper applies senpai's defaults and
    builds senpai's star dicts (see module docstring).
    """
    if faint_lim is None:
        faint_lim = 20.0
    if bright_lim is None:
        bright_lim = -32.0

    a = query_mirror_box(
        min_ra, max_ra, min_dec, max_dec,
        mirror_dir=mirror_dir,
        faint_limit=faint_lim,
        bright_limit=bright_lim,
        max_rows=MAX_LOCAL_ROWS,
    )
    return [_to_star(r, primary_filter, faint_lim) for r in a]


def _to_star(row: np.void, primary_filter: str, faint_lim: float) -> dict[str, Any]:
    """Build the same star dict as gaia.query_by_ra_dec_bounds.

    Coordinates are converted to radians, synthetic Johnson_V / Sloan_r are derived
    from BP-RP, and proper motion is expressed in rad/s.

    Args:
        row: One record of the mirror's ``MIRROR_DTYPE`` structured array (RA/Dec in
            degrees), as returned by ``query_mirror_box``.
        primary_filter: Gaia band to use as the primary magnitude ('G', 'BP', 'RP').
        faint_lim: Fallback magnitude assigned when no finite band magnitude is
            available for the star.

    Returns:
        Star dict matching ``gaia.query_by_ra_dec_bounds`` (ra/dec in radians, mv,
        magnitudes, catalog, source_id, proper motion, parallax).
    """
    from senpai.catalog.gaia_transforms import (
        gaia_bp_rp_to_johnson_v,
        gaia_bp_rp_to_sloan_r,
    )

    g, bp, rp = float(row["g"]), float(row["bp"]), float(row["rp"])
    magnitudes: dict[str, float] = {}
    if np.isfinite(g) and g < 32:
        magnitudes["Gaia_G"] = g
    if np.isfinite(bp) and bp < 32:
        magnitudes["Gaia_BP"] = bp
    if np.isfinite(rp) and rp < 32:
        magnitudes["Gaia_RP"] = rp
    if {"Gaia_G", "Gaia_BP", "Gaia_RP"} <= magnitudes.keys():
        bp_rp = magnitudes["Gaia_BP"] - magnitudes["Gaia_RP"]
        jv = gaia_bp_rp_to_johnson_v(magnitudes["Gaia_G"], bp_rp)
        if jv is not None:
            magnitudes["Johnson_V"] = jv
        sr = gaia_bp_rp_to_sloan_r(magnitudes["Gaia_G"], bp_rp)
        if sr is not None:
            magnitudes["Sloan_r"] = sr

    band = {"G": g, "BP": bp, "RP": rp}.get(primary_filter, g)
    primary_mag = band if (np.isfinite(band) and band < 32) else (
        g if (np.isfinite(g) and g < 32) else faint_lim
    )

    MAS2RAD = 4.84813681109535993589914102358e-9
    YEAR2SEC = 3.1556952e7
    pmra = float(row["pmra"])
    pmdec = float(row["pmdec"])
    ra_pm = pmra * MAS2RAD / YEAR2SEC if np.isfinite(pmra) else 0.0
    dec_pm = pmdec * MAS2RAD / YEAR2SEC if np.isfinite(pmdec) else 0.0

    return {
        "ra": np.radians(float(row["ra"])),
        "dec": np.radians(float(row["dec"])),
        "mv": primary_mag,
        "magnitudes": magnitudes,
        "catalog": "Gaia",
        "source_id": str(int(row["source_id"])),
        "ra_pm": ra_pm,
        "dec_pm": dec_pm,
        "parallax": 0.0,  # not stored in the trimmed mirror
    }
