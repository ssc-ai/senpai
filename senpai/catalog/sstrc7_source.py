"""SSTRC7 star catalog access, backed by the `sstrc7` package.

senpai vendored its own reader for this catalog through 2.6.x. The package
reads the same 1801 files (`sstrc.acc`, `s0000.cat`..`s1799.cat`) from the same
directory, so an existing local catalog works unchanged -- but it fixes two
bugs the vendored copy carried:

* **Proper motion was 9.77x too large.** The catalog stores proper motion in
  units of 0.32 mas/yr per count, so decoding multiplies by 0.32; the vendored
  scale divided instead (`(1 / 0.32) * mas2rad / year2sec`), which is off by
  exactly 1 / 0.32**2 = 9.7656. RA carried a further spurious `cos(dec)` on top
  of a value that is already a coordinate proper motion.
* **The zone selection dropped stars.** It truncated the high-RA edge of a
  field while over-returning several degrees on the low-RA side; roughly 16% of
  the stars inside a 0.5 degree field were missing from the result.

Records keep the star-dict shape shared by every senpai catalog backend (see
`gaia.py`, `sdss.py`): `ra`/`dec` in radians, `ra_pm`/`dec_pm` in radians per
second, `mv` plus a non-empty `magnitudes` dict, and a `catalog` provenance
string.
"""

import logging
from pathlib import Path

import numpy as np
import sstrc7

from senpai.core.config import settings

logger = logging.getLogger(__name__)

# 1 mas/yr expressed in radians per second.
MAS2RAD = 4.84813681109535993589914102358e-9
YEAR2SEC = 3.1556952e7
MAS_PER_YEAR_TO_RAD_PER_SEC = MAS2RAD / YEAR2SEC

# Magnitudes outside this range are sentinels, not measurements.
MAG_MIN = -32.0
MAG_MAX = 32.0
INVALID_MAG = 32.0
# Magnitudes are stored as integer millimags.
MAG_DECIMALS = 3

BAND_NAMES = tuple(sstrc7.BAND_NAMES)


def resolve_catalog_path(path: str | None = None) -> Path:
    """Resolve the catalog directory.

    An explicit path (senpai's `catalog.path` config) wins; otherwise the
    package resolves `$SSTRC7_PATH` and then `~/.sstrc7`.
    """
    return sstrc7.catalog_path(path)


def examine_catalog(path: str | None = None) -> bool:
    """Report whether the catalog on disk is complete.

    Sizes only -- hashing all 17.6 GB would make startup unusable.
    """
    status = sstrc7.status(path)
    if status.missing:
        logger.error(f"SSTRC7 catalog at {status.path} is missing {len(status.missing)} of {status.expected} files")
        logger.error(f"Run: python -m sstrc7 get --path {status.path}")
        return False
    if status.corrupt:
        logger.error(f"SSTRC7 catalog at {status.path} has {len(status.corrupt)} corrupt files")
        return False
    return True


def _catalog_labels(source_flags: np.ndarray) -> list[str]:
    """Provenance string per star, e.g. "Gaia Catalog, 2MASS Catalog"."""
    labels = {}
    for flags in np.unique(source_flags):
        decoded = sstrc7.decode_source_flags(int(flags))
        labels[int(flags)] = ", ".join(decoded) if decoded else "Unknown"
    return [labels[int(f)] for f in source_flags]


# Band order for the `open_first` priority ladder: the broad Gaia G response is
# the closest stand-in for an unfiltered silicon detector, so it leads.
OPEN_FIRST_BANDS = ("Gaia_G", "Johnson_R", "Sloan_r", "Johnson_V", "Johnson_B")


def _priority_ladder(field) -> np.ndarray:
    """Per-star visual magnitude under the configured band priority.

    `johnson_v_first` (the default) is the package's own `visual` ladder. NaN
    where no band of the ladder is usable, as `visual` is.
    """
    if settings.star_catalog.magnitude_band_priority != "open_first":
        return field.visual

    mags = field.mag
    ladder = np.full(len(mags), np.nan)
    # Lowest priority first, so each better band overwrites what precedes it.
    for name in reversed(OPEN_FIRST_BANDS):
        column = mags[:, BAND_NAMES.index(name)]
        usable = np.isfinite(column) & (column > MAG_MIN) & (column < MAG_MAX)
        ladder = np.where(usable, column, ladder)
    return ladder


def _star_records(field, filter_center: float | None) -> list[dict]:
    """Convert a package StarField into senpai's star dicts."""
    # The catalog stores magnitudes as integer millimags; the package hands
    # them back as float32, so 17.968 arrives as 17.968000411987305. Rounding
    # to the storage quantum recovers the exact catalog value rather than
    # propagating float32 noise into photometry.
    mags = np.round(field.mag.astype(np.float64), MAG_DECIMALS)
    valid = np.isfinite(mags) & (mags > MAG_MIN) & (mags < MAG_MAX)

    # The configured priority ladder: by default the package's `visual`, which
    # matches the vendored Johnson_V > Johnson_R > Sloan_r > Gaia_G > ... order
    # exactly. A star with no usable band gets the sentinel rather than NaN, so
    # the magnitude limits below and downstream `mv < 32` checks behave as before.
    visual = _priority_ladder(field)
    ladder = np.round(np.where(np.isfinite(visual), visual, INVALID_MAG).astype(np.float64), MAG_DECIMALS)

    if filter_center is not None:
        interpolated = field.at_wavelength(filter_center)
        primary = np.where(np.isfinite(interpolated), interpolated, np.nan)
    else:
        primary = ladder

    ra = field.ra_rad
    dec = field.dec_rad
    ra_pm = field.pm_ra * MAS_PER_YEAR_TO_RAD_PER_SEC
    dec_pm = field.pm_dec * MAS_PER_YEAR_TO_RAD_PER_SEC
    parallax = field.parallax * MAS2RAD
    catalogs = _catalog_labels(field.source_flags)

    stars = []
    for i in range(len(ra)):
        magnitudes = {BAND_NAMES[b]: float(mags[i, b]) for b in np.flatnonzero(valid[i])}

        mv = primary[i]
        if not np.isfinite(mv) or not (MAG_MIN < mv < MAG_MAX):
            # Fall back to the first available band, then to the sentinel --
            # `magnitudes` is never empty and `mv` is never None downstream.
            mv = next(iter(magnitudes.values()), None)
        if mv is None:
            mv = INVALID_MAG
            magnitudes["Invalid"] = INVALID_MAG

        stars.append(
            {
                "ra": float(ra[i]),
                "dec": float(dec[i]),
                "ra_pm": float(ra_pm[i]),
                "dec_pm": float(dec_pm[i]),
                "parallax": float(parallax[i]),
                "mv": float(mv),
                "magnitudes": magnitudes,
                "catalog": catalogs[i],
            }
        )
    return stars


def query_by_los_radec_with_rotation(
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rotation: float = 0.0,
    rootPath: str | None = None,  # camelCase kept for call-site compatibility
    filter_center: float | None = None,
    faint_lim: float | None = None,
    bright_lim: float | None = None,
    safety_margin: float = 0.1,
) -> list[dict]:
    """Query the catalog for stars around a line of sight.

    Args:
        y_fov: y field of view in degrees.
        x_fov: x field of view in degrees.
        ra: right ascension of the field center in degrees.
        dec: declination of the field center in degrees.
        rotation: field rotation in degrees. Accepted for call-site
            compatibility and deliberately unused -- the query is a cone about
            the boresight enclosing the field corners, which is the same set of
            stars at any rotation. Callers project and clip to the real focal
            plane afterwards.
        rootPath: catalog directory; None resolves from the environment.
        filter_center: wavelength in nm to interpolate `mv` at, instead of the
            broadband priority ladder.
        faint_lim: drop stars at or fainter than this magnitude.
        bright_lim: drop stars at or brighter than this magnitude.
        safety_margin: fraction to expand the field by before querying.

    Returns:
        A list of star dicts, in senpai's cross-catalog record shape.

    """
    margin = 1.0 + safety_margin
    # Half-diagonal of the expanded field: the smallest cone that contains the
    # focal plane whatever the rotation.
    radius = 0.5 * float(np.hypot(x_fov * margin, y_fov * margin))

    field = sstrc7.query_cone(ra, dec, radius, path=rootPath)

    # The vendored reader applied both limits against the priority-ladder
    # magnitude even when filter_center was set. Keep that.
    visual = _priority_ladder(field)
    ladder = np.where(np.isfinite(visual), visual, INVALID_MAG)
    keep = np.ones(len(ladder), dtype=bool)
    if faint_lim is not None:
        keep &= ladder < faint_lim
    if bright_lim is not None:
        keep &= ladder > bright_lim
    if not keep.all():
        field = sstrc7.StarField(field.records[keep])

    return _star_records(field, filter_center)
