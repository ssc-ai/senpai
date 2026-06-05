import logging
from typing import Any

import numpy as np
from astroquery.sdss import SDSS

logger = logging.getLogger(__name__)


def query_by_ra_dec_bounds(
    min_ra: float,
    max_ra: float,
    min_dec: float,
    max_dec: float,
    faint_lim: float = None,
    bright_lim: float = None,
    primary_filter: str = "g",
) -> list[dict[str, Any]]:
    """Query SDSS catalog using explicit RA/DEC bounds.

    Args:
        min_ra: `float`, minimum right ascension in degrees
        max_ra: `float`, maximum right ascension in degrees
        min_dec: `float`, minimum declination in degrees
        max_dec: `float`, maximum declination in degrees
        faint_lim: `float`, faint magnitude limit (stars fainter than this are excluded)
        bright_lim: `float`, bright magnitude limit (stars brighter than this are excluded)
        primary_filter: `str`, SDSS band to use as primary magnitude ('u', 'g', 'r', 'i', 'z')

    Returns:
        A `list`, stars within the bounds of input parameters
    """
    # Set default magnitude limits
    if faint_lim is None:
        faint_lim = 23.0  # SDSS g-band limit
    if bright_lim is None:
        bright_lim = -32.0  # Include all bright stars

    # Normalize RA to [0, 360) range
    min_ra_normalized = np.mod(min_ra, 360.0)
    max_ra_normalized = np.mod(max_ra, 360.0)

    try:
        # Check if field crosses RA = 0/360 boundary
        ra_span = max_ra_normalized - min_ra_normalized
        crosses_zero = ra_span > 180.0 or (min_ra_normalized > max_ra_normalized and ra_span < 180.0)

        if crosses_zero:
            # Field crosses RA = 0/360 boundary - need two queries
            logger.info(
                f"Querying SDSS (crosses RA=0): "
                f"RA=[{min_ra_normalized:.3f}, 360.0] U [0.0, {max_ra_normalized:.3f}]°, "
                f"DEC=[{min_dec:.3f}, {max_dec:.3f}]°, "
                f"g=[{bright_lim:.1f}, {faint_lim:.1f}]"
            )

            # Query first range: high RA values (near 360)
            sql1 = f"""
            SELECT TOP 500000
                objid, ra, dec, u, g, r, i, z
            FROM PhotoPrimary
            WHERE g BETWEEN {bright_lim} AND {faint_lim}
            AND ra >= {min_ra_normalized} AND ra <= 360.0
            AND dec BETWEEN {min_dec} AND {max_dec}
            """

            # Query second range: low RA values (near 0)
            sql2 = f"""
            SELECT TOP 500000
                objid, ra, dec, u, g, r, i, z
            FROM PhotoPrimary
            WHERE g BETWEEN {bright_lim} AND {faint_lim}
            AND ra >= 0.0 AND ra <= {max_ra_normalized}
            AND dec BETWEEN {min_dec} AND {max_dec}
            """

            result1 = SDSS.query_sql(sql1)
            result2 = SDSS.query_sql(sql2)

            # Combine results using astropy Table vstack
            from astropy.table import vstack

            results_to_combine = []
            if result1 is not None and len(result1) > 0:
                results_to_combine.append(result1)
            if result2 is not None and len(result2) > 0:
                results_to_combine.append(result2)

            if len(results_to_combine) > 0:
                result = vstack(results_to_combine)
            else:
                result = None
        else:
            # Field doesn't cross boundary - simple case
            logger.info(
                f"Querying SDSS: RA=[{min_ra_normalized:.3f}, {max_ra_normalized:.3f}]°, "
                f"DEC=[{min_dec:.3f}, {max_dec:.3f}]°, "
                f"g=[{bright_lim:.1f}, {faint_lim:.1f}]"
            )

            sql = f"""
            SELECT TOP 500000
                objid, ra, dec, u, g, r, i, z
            FROM PhotoPrimary
            WHERE g BETWEEN {bright_lim} AND {faint_lim}
            AND ra BETWEEN {min_ra_normalized} AND {max_ra_normalized}
            AND dec BETWEEN {min_dec} AND {max_dec}
            """

            result = SDSS.query_sql(sql)

        if result is None or len(result) == 0:
            logger.info("No stars found in SDSS query")
            return []

        logger.info(f"SDSS returned {len(result)} stars")

        # Convert to star format matching SSTRC7 structure
        stars = []
        for row in result:
            magnitudes = {}
            # Add available SDSS magnitudes
            for band in ["u", "g", "r", "i", "z"]:
                if hasattr(row, band) and row[band] < 32:
                    magnitudes[f"Sloan_{band}"] = float(row[band])

            # Use requested primary filter band, fallback to g-band, then faint_lim
            if hasattr(row, primary_filter) and row[primary_filter] < 32:
                primary_mag = float(row[primary_filter])
            elif hasattr(row, "g") and row["g"] < 32:
                primary_mag = float(row["g"])
            else:
                primary_mag = faint_lim

            star = {
                "ra": np.radians(float(row["ra"])),
                "dec": np.radians(float(row["dec"])),
                "mv": primary_mag,
                "magnitudes": magnitudes,
                "catalog": "SDSS",
                "objid": str(row["objid"]),
                "ra_pm": 0.0,  # SDSS doesn't provide proper motions
                "dec_pm": 0.0,
                "parallax": 0.0,
            }
            stars.append(star)

        return stars

    except Exception as e:
        logger.error(f"SDSS query failed: {e}")
        return []


def query_by_bounds(
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rotation: float = 0.0,
    faint_lim: float = None,
    bright_lim: float = None,
    safety_margin: float = 0.1,
    primary_filter: str = "g",
) -> list[dict[str, Any]]:
    """Query SDSS catalog online based on field parameters.

    Args:
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        ra: `float`, right ascension of the field center in degrees
        dec: `float`, declination of the field center in degrees
        rotation: `float`, field rotation in degrees
        faint_lim: `float`, faint magnitude limit (stars fainter than this are excluded)
        bright_lim: `float`, bright magnitude limit (stars brighter than this are excluded)
        safety_margin: `float`, fraction to expand the search area by
        primary_filter: `str`, SDSS band to use as primary magnitude ('u', 'g', 'r', 'i', 'z')

    Returns:
        A `list`, stars within the bounds of input parameters
    """
    # Apply safety margin
    x_fov_with_margin = x_fov * (1 + safety_margin)
    y_fov_with_margin = y_fov * (1 + safety_margin)

    # Calculate the corners of the field in ra/dec space (accounting for rotation)
    half_width = x_fov_with_margin / 2
    half_height = y_fov_with_margin / 2

    # Define corners in pixel space (relative to center)
    corners_rel = np.array(
        [
            [-half_width, -half_height],  # bottom-left
            [half_width, -half_height],  # bottom-right
            [half_width, half_height],  # top-right
            [-half_width, half_height],  # top-left
        ]
    )

    # Apply rotation if provided
    if rotation != 0.0:
        rot_rad = np.radians(rotation)
        rot_matrix = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
        corners_rel = np.dot(corners_rel, rot_matrix.T)

    # Convert corners to RA/DEC (accounting for spherical geometry)
    # RA spacing shrinks with cos(dec)
    ra_corners = ra + corners_rel[:, 0] / np.cos(np.radians(dec))
    dec_corners = dec + corners_rel[:, 1]

    # Find bounding box that encompasses all corners
    min_dec = np.min(dec_corners)
    max_dec = np.max(dec_corners)

    # Normalize RA corners to [0, 360) range
    ra_corners_normalized = np.mod(ra_corners, 360.0)

    # Set default magnitude limits
    if faint_lim is None:
        faint_lim = 23.0  # SDSS g-band limit
    if bright_lim is None:
        bright_lim = -32.0  # Include all bright stars

    try:
        # Check if field crosses RA = 0/360 boundary
        ra_span = np.max(ra_corners_normalized) - np.min(ra_corners_normalized)
        crosses_zero = ra_span > 180.0  # If span > 180°, we must have crossed the boundary

        if crosses_zero:
            # Field crosses RA = 0/360 boundary - need two queries
            ra_low_side = ra_corners_normalized[ra_corners_normalized < 180.0]
            ra_high_side = ra_corners_normalized[ra_corners_normalized >= 180.0]

            if len(ra_low_side) > 0 and len(ra_high_side) > 0:
                min_ra_high = np.min(ra_high_side)
                max_ra_low = np.max(ra_low_side)

                # Query first range: high RA values (near 360)
                sql1 = f"""
                SELECT TOP 500000
                    objid, ra, dec, u, g, r, i, z
                FROM PhotoPrimary
                WHERE g BETWEEN {bright_lim} AND {faint_lim}
                AND ra BETWEEN {min_ra_high} AND 360.0
                AND dec BETWEEN {min_dec} AND {max_dec}
                """

                # Query second range: low RA values (near 0)
                sql2 = f"""
                SELECT TOP 500000
                    objid, ra, dec, u, g, r, i, z
                FROM PhotoPrimary
                WHERE g BETWEEN {bright_lim} AND {faint_lim}
                AND ra BETWEEN 0.0 AND {max_ra_low}
                AND dec BETWEEN {min_dec} AND {max_dec}
                """

                logger.info(
                    f"Querying SDSS (crosses RA=0): "
                    f"RA=[{min_ra_high:.3f}, 360.0] U [0.0, {max_ra_low:.3f}]°, "
                    f"DEC=[{min_dec:.3f}, {max_dec:.3f}]°, "
                    f"g=[{bright_lim:.1f}, {faint_lim:.1f}]"
                )

                result1 = SDSS.query_sql(sql1)
                result2 = SDSS.query_sql(sql2)

                # Combine results using astropy Table vstack
                from astropy.table import vstack

                results_to_combine = []
                if result1 is not None and len(result1) > 0:
                    results_to_combine.append(result1)
                if result2 is not None and len(result2) > 0:
                    results_to_combine.append(result2)

                if len(results_to_combine) > 0:
                    result = vstack(results_to_combine)
                else:
                    result = None
            else:
                # Fallback to simple query
                min_ra = np.min(ra_corners_normalized)
                max_ra = np.max(ra_corners_normalized)
                sql = f"""
                SELECT TOP 500000
                    objid, ra, dec, u, g, r, i, z
                FROM PhotoPrimary
                WHERE g BETWEEN {bright_lim} AND {faint_lim}
                AND ra BETWEEN {min_ra} AND {max_ra}
                AND dec BETWEEN {min_dec} AND {max_dec}
                """
                logger.info(
                    f"Querying SDSS: RA=[{min_ra:.3f}, {max_ra:.3f}]°, "
                    f"DEC=[{min_dec:.3f}, {max_dec:.3f}]°, "
                    f"g=[{bright_lim:.1f}, {faint_lim:.1f}]"
                )
                result = SDSS.query_sql(sql)
        else:
            # Field doesn't cross boundary - simple case
            min_ra = np.min(ra_corners_normalized)
            max_ra = np.max(ra_corners_normalized)

            sql = f"""
            SELECT TOP 500000
                objid, ra, dec, u, g, r, i, z
            FROM PhotoPrimary
            WHERE g BETWEEN {bright_lim} AND {faint_lim}
            AND ra BETWEEN {min_ra} AND {max_ra}
            AND dec BETWEEN {min_dec} AND {max_dec}
            """

            logger.info(
                f"Querying SDSS: RA=[{min_ra:.3f}, {max_ra:.3f}]°, "
                f"DEC=[{min_dec:.3f}, {max_dec:.3f}]°, "
                f"rotation={rotation:.1f}°, "
                f"g=[{bright_lim:.1f}, {faint_lim:.1f}]"
            )

            result = SDSS.query_sql(sql)

        # Check results after all query branches
        if result is None or len(result) == 0:
            logger.info("No stars found in SDSS query")
            return []

        logger.info(f"SDSS returned {len(result)} stars")

        # Convert to star format matching SSTRC7 structure
        stars = []
        for row in result:
            magnitudes = {}
            # Add available SDSS magnitudes
            for band in ["u", "g", "r", "i", "z"]:
                if hasattr(row, band) and row[band] < 32:
                    magnitudes[f"Sloan_{band}"] = float(row[band])

            # Use requested primary filter band, fallback to g-band, then faint_lim
            if hasattr(row, primary_filter) and row[primary_filter] < 32:
                primary_mag = float(row[primary_filter])
            elif hasattr(row, "g") and row["g"] < 32:
                primary_mag = float(row["g"])
            else:
                primary_mag = faint_lim

            star = {
                "ra": np.radians(float(row["ra"])),
                "dec": np.radians(float(row["dec"])),
                "mv": primary_mag,
                "magnitudes": magnitudes,
                "catalog": "SDSS",
                "objid": str(row["objid"]),
                "ra_pm": 0.0,  # SDSS doesn't provide proper motions
                "dec_pm": 0.0,
                "parallax": 0.0,
            }
            stars.append(star)

        return stars

    except Exception as e:
        logger.error(f"SDSS query failed: {e}")
        return []


def query_by_los_radec_with_rotation(
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rotation: float = 0.0,
    rootPath: str = None,
    filter_center: float = None,
    faint_lim: float = None,
    bright_lim: float = None,
    safety_margin: float = 0.1,
    primary_filter: str = "g",
) -> list[dict[str, Any]]:
    """Alias for query_by_bounds to match SSTRC7 interface."""
    return query_by_bounds(
        y_fov=y_fov,
        x_fov=x_fov,
        ra=ra,
        dec=dec,
        rotation=rotation,
        faint_lim=faint_lim,
        bright_lim=bright_lim,
        safety_margin=safety_margin,
        primary_filter=primary_filter,
    )
