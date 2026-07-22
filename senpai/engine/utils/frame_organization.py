"""Frame timing and image-set organization helpers driven by FITS headers."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import arrow
from astropy.io import fits

from senpai.engine.constants import DATE_HEADERS, DATE_TIME_HEADERS, TIME_HEADERS

logger = logging.getLogger(__name__)


def _parse_date_string(date_str: str) -> arrow.Arrow:
    """Parse a date string into an Arrow instant.

    Tries arrow's default parsing first, then falls back to MM/DD/YY and
    MM/DD/YYYY formats.

    Args:
        date_str: The date (or datetime) string to parse.

    Returns:
        The parsed Arrow instant.
    """
    try:
        # First try arrow's default parsing (handles many formats)
        return arrow.get(date_str)
    except Exception:
        # If default parsing fails, try MM/DD/YY and MM/DD/YYYY formats
        try:
            return arrow.get(date_str, "M/D/YY")
        except Exception:
            try:
                return arrow.get(date_str, "MM/DD/YY")
            except Exception:
                try:
                    return arrow.get(date_str, "M/D/YYYY")
                except Exception:
                    return arrow.get(date_str, "MM/DD/YYYY")

    return None


def _parse_time_string(time_str: str) -> tuple[int, int, int, int] | None:
    """Parse a time string into its components.

    Args:
        time_str: The time string to parse.

    Returns:
        A (hour, minute, second, microsecond) tuple, or None if parsing fails.
    """
    from datetime import datetime

    # Try parsing with datetime first
    time_formats = [
        "%H:%M:%S.%f",  # 21:36:28.604000
        "%H:%M:%S",  # 21:36:28
        "%H:%M:%S.%f000",  # sometimes microseconds are padded
    ]

    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(time_str, fmt).time()
            return (parsed_time.hour, parsed_time.minute, parsed_time.second, parsed_time.microsecond)
        except ValueError:
            continue

    # Fallback: try arrow's natural parsing and extract time components
    try:
        arrow_time = arrow.get(time_str)
        return (arrow_time.hour, arrow_time.minute, arrow_time.second, arrow_time.microsecond)
    except Exception:  # noqa: S110  # best-effort time parse; None is returned on failure
        pass

    return None


def extract_uct_time_from_header(header: dict[str, Any]) -> datetime:
    """Extract the observation time from a FITS header.

    Tries the combined date-time headers first, then falls back to composing a
    separate date header with a time header.

    Args:
        header: The FITS header (or dict-like) to read time keywords from.

    Returns:
        The extracted observation time as a datetime.

    Raises:
        AttributeError: If no usable date/time header is present.
    """
    for header_key in DATE_TIME_HEADERS:
        if header_key in header:
            try:
                arrow_time = _parse_date_string(str(header[header_key]))
                return arrow_time.datetime
            except Exception as e:
                logger.error(f"failed to parse time from {header_key}: {e}")
                continue

    arrow_date = None
    time_components = None
    for header_key in DATE_HEADERS:
        if header_key in header:
            try:
                arrow_date = _parse_date_string(str(header[header_key]))
            except Exception as e:
                logger.error(f"failed to parse time from {header_key}: {e}")
                continue

    if arrow_date is not None:
        for header_key in TIME_HEADERS:
            if header_key in header:
                try:
                    time_components = _parse_time_string(str(header[header_key]))
                    break
                except Exception as e:
                    logger.error(f"failed to parse time from {header_key}: {e}")
                    continue

    # If we have both date and time, combine them
    if arrow_date is not None and time_components is not None:
        hour, minute, second, microsecond = time_components
        combined_datetime = arrow_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
        return combined_datetime.datetime

    # Debug-level: callers that can tolerate a missing time catch this raise and
    # log a clean, actionable warning themselves (see organize_senpai_frames /
    # extract_observation_time_from_header). Logging ERROR here made a handled,
    # expected condition look like a failure (6 lines x 2 calls per frame).
    logger.debug("no valid date header found in header")
    logger.debug(f"available header: {', '.join(list(header.keys()))}")
    logger.debug(f"coded DATE_TIME headers: {', '.join(DATE_TIME_HEADERS)}")
    logger.debug(f"coded DATE headers: {', '.join(DATE_HEADERS)}")
    logger.debug(f"coded TIME headers: {', '.join(TIME_HEADERS)}")
    logger.debug("YOU MUST HAVE A DATE_TIME or a DATE and a TIME")
    raise AttributeError(f"no valid date header found in {header}")


def get_imageset_by_filename(data_directory: Path, string_match: str) -> list[str]:
    """Find FITS files whose filename contains a substring.

    Args:
        data_directory: Directory searched recursively for ``*.fits`` files.
        string_match: Substring that must appear in the filename.

    Returns:
        Sorted paths (as strings) of the matching FITS files.
    """
    # Get all .fits files in directory that match the regex pattern
    fits_files = [str(f) for f in data_directory.glob("**/*.fits") if string_match in f.name]

    if not fits_files:
        logger.warning(f"No .fits files found matching '{string_match}'*.fits in {data_directory}")

    return sorted(fits_files)


def get_all_images_in_directory(data_directory: Path) -> list[str]:
    """Find every FITS file under a directory.

    Args:
        data_directory: Directory searched recursively for ``*.fits`` files.

    Returns:
        Sorted paths (as strings) of all FITS files found.
    """
    # Get all .fits files in directory and subdirectories
    fits_files = [str(f) for f in data_directory.glob("**/*.fits")]

    if not fits_files:
        logger.warning(f"No .fits files found in {data_directory}")

    return sorted(fits_files)


def extract_id_from_header(file: Path, header_key: str) -> str | None:
    """Extract an id value from a FITS file's primary header.

    For the ``ORCHCOMM`` key the embedded image-set id is parsed out of the
    ``&IMAGESETID@...`` structure; other keys are returned directly.

    Args:
        file: Path to the FITS file to read.
        header_key: The header keyword to extract.

    Returns:
        The extracted value, or None if the key is not present.
    """
    with fits.open(file) as hdul:
        header = hdul[0].header

    if header_key not in header:
        logger.warning(f"header key {header_key} not found in {file}")
        return None

    if header_key == "ORCHCOMM":
        # ORCHCOMM looks something like this: &IMAGESETID@[ukr]#[1:6]%[OPEN]
        return header["ORCHCOMM"].split("&")[1].split("@")[0]

    return header[header_key]


def header_key_matches(file: Path, header_key: str, value: str) -> bool:
    """Check whether a FITS file's header key matches a value.

    For the ``ORCHCOMM`` key the comparison is a substring match against the
    embedded image-set id; other keys are compared for equality.

    Args:
        file: Path to the FITS file to read.
        header_key: The header keyword to compare.
        value: The value to match against.

    Returns:
        True if the key is present and matches, otherwise False.
    """
    with fits.open(file) as hdul:
        header = hdul[0].header

    if header_key not in header:
        return False

    if header_key == "ORCHCOMM":
        # ORCHCOMM looks something like this: &IMAGESETID@[ukr]#[1:6]%[OPEN]
        return value in header["ORCHCOMM"].split("&")[1].split("@")[0]

    return header[header_key] == value


def get_imageset_by_id(data_directory: Path, imageset_id: str, header_id_key: str) -> list[str]:
    """Find FITS files whose header id key matches a value.

    Args:
        data_directory: Directory searched recursively for ``*.fits`` files.
        imageset_id: The image-set id value to match.
        header_id_key: The header keyword that carries the id.

    Returns:
        Sorted paths (as strings) of the matching FITS files.
    """
    # get all fits files in a directory that have the same value for the header_id_key
    fits_files = [
        str(f)
        for f in data_directory.glob("**/*.fits")
        if header_key_matches(f, header_id_key, imageset_id)
    ]

    if not fits_files:
        logger.warning(f"No .fits files found with ID {imageset_id} in {data_directory}")

    return sorted(fits_files)
