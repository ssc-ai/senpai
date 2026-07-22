"""JSON serialization helpers for scientific/FITS data types."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astropy.io.fits import Header

logger = logging.getLogger(__name__)


def jsonable(value: object) -> object:
    """Best-effort conversion of common scientific/Python types to JSON-safe primitives.

    Args:
        value: The value to convert. May be a primitive, a numpy scalar, a
            list/tuple, an astropy FITS commentary card, or any other object.

    Returns:
        A JSON-serializable representation: primitives are returned unchanged,
        numpy scalars are converted to Python scalars, lists/tuples are
        recursed into, and any other object is stringified.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # numpy scalars -> Python scalars
    if isinstance(value, np.generic):
        return value.item()

    # lists/tuples -> recurse
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    # Astropy FITS commentary cards (COMMENT/HISTORY) are not JSON serializable
    try:
        from astropy.io.fits.header import _HeaderCommentaryCards  # type: ignore[reportPrivateImportUsage]

        if isinstance(value, _HeaderCommentaryCards):
            return [str(v) for v in value]
    except Exception as e:
        # If astropy isn't available or internals changed, fall back to string conversion below.
        logger.debug("Astropy FITS commentary conversion unavailable: %s", e)

    # Fallback: stringify unknown objects
    return str(value)


def fits_header_to_jsonable(header: Header | None) -> dict | None:
    """Convert an astropy FITS Header into a JSON-serializable plain dict.

    Args:
        header: The astropy FITS ``Header`` to convert, or None.

    Returns:
        A plain dict mapping each header keyword to a JSON-safe value, or None
        if ``header`` is None or serialization fails.
    """
    if header is None:
        return None

    out: dict[str, object] = {}
    try:
        for key in header:
            if key is None:
                continue
            out[str(key)] = jsonable(header[key])
    except Exception as e:
        logger.warning("Failed to serialize FITS header safely: %s", e)
        return None

    return out
