"""Helpers for parsing celestial coordinates from FITS headers."""

import logging

import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits

logger = logging.getLogger(__name__)

_RA_HEADER_KEYS = ["RA", "RA_OBJ", "OBJCTRA", "TELRA", "CRVAL1"]
_DEC_HEADER_KEYS = ["DEC", "DEC_OBJ", "OBJCTDEC", "TELDEC", "CRVAL2"]


def parse_fits_coordinate(value: str | float | int, is_ra: bool) -> float | None:
    """Parse a FITS header coordinate value to decimal degrees.

    Handles plain floats/ints, decimal strings, and sexagesimal strings.
    For RA, sexagesimal is interpreted as hours-minutes-seconds; for Dec,
    as degrees-arcminutes-arcseconds.

    Args:
        value (str | float | int): raw header value.
        is_ra (bool): True if parsing RA (sexagesimal = HMS); False for Dec (DMS).

    Returns:
        float | None: decimal degrees, or None if parsing fails.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            pass
        try:
            unit = u.hourangle if is_ra else u.deg
            return float(Angle(value, unit=unit).deg)
        except Exception:
            logger.debug("Could not parse coordinate value %r", value)
    return None


def _find_coordinate(header: fits.Header, keys: list[str], is_ra: bool) -> float | None:
    """Return the first parseable coordinate found among the given header keys.

    Args:
        header (fits.Header): FITS header to search.
        keys (list[str]): Candidate header keywords, tried in order.
        is_ra (bool): True if parsing RA (sexagesimal = HMS); False for Dec (DMS).

    Returns:
        float | None: Decimal degrees of the first successfully parsed key, or
            None if no key yields a valid value.
    """
    for key in keys:
        raw = header.get(key)
        if raw is not None:
            val = parse_fits_coordinate(raw, is_ra=is_ra)
            if val is not None:
                return val
    return None


def read_boresight_from_header(
    header: fits.Header,
) -> tuple[float | None, float | None]:
    """Extract boresight RA and Dec from a FITS header.

    Tries each key in _RA_HEADER_KEYS / _DEC_HEADER_KEYS in order and returns the
    first successfully parsed value. Logs a warning if either coordinate is not found.

    Args:
        header (fits.Header): FITS header to search.

    Returns:
        tuple[float | None, float | None]: (ra_deg, dec_deg).
    """
    ra = _find_coordinate(header, _RA_HEADER_KEYS, is_ra=True)
    dec = _find_coordinate(header, _DEC_HEADER_KEYS, is_ra=False)

    if ra is None:
        logger.warning("Boresight RA not found in FITS header (tried: %s)", _RA_HEADER_KEYS)
    if dec is None:
        logger.warning("Boresight Dec not found in FITS header (tried: %s)", _DEC_HEADER_KEYS)

    return ra, dec
