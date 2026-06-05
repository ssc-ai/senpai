import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def query_by_ra_dec_bounds(
    min_ra: float,
    max_ra: float,
    min_dec: float,
    max_dec: float,
    faint_lim: float = None,
    bright_lim: float = None,
    primary_filter: str = "G",
) -> list[dict[str, Any]]:
    """Query Gaia catalog using explicit RA/DEC bounds.

    Args:
        min_ra: `float`, minimum right ascension in degrees
        max_ra: `float`, maximum right ascension in degrees
        min_dec: `float`, minimum declination in degrees
        max_dec: `float`, maximum declination in degrees
        faint_lim: `float`, faint magnitude limit (stars fainter than this are excluded)
        bright_lim: `float`, bright magnitude limit (stars brighter than this are excluded)
        primary_filter: `str`, Gaia band to use as primary magnitude ('G', 'BP', 'RP')

    Returns:
        A `list`, stars within the bounds of input parameters
    """
    from astroquery.gaia import Gaia

    # Set default magnitude limits
    if faint_lim is None:
        faint_lim = 21.0  # Gaia G-band limit (Gaia DR3 goes to ~21 mag)
    if bright_lim is None:
        bright_lim = -32.0  # Include all bright stars

    # Normalize RA to [0, 360) range
    min_ra_normalized = np.mod(min_ra, 360.0)
    max_ra_normalized = np.mod(max_ra, 360.0)

    try:
        # Check if field crosses RA = 0/360 boundary
        ra_span = max_ra_normalized - min_ra_normalized
        crosses_zero = ra_span > 180.0 or (min_ra_normalized > max_ra_normalized and ra_span < 180.0)

        # Build ADQL query for Gaia
        # Gaia uses ADQL (Astronomical Data Query Language) which is SQL-like
        if crosses_zero:
            # Field crosses RA = 0/360 boundary - need two queries
            logger.info(
                f"Querying Gaia (crosses RA=0): "
                f"RA=[{min_ra_normalized:.3f}, 360.0] U [0.0, {max_ra_normalized:.3f}]°, "
                f"DEC=[{min_dec:.3f}, {max_dec:.3f}]°, "
                f"{primary_filter}=[{bright_lim:.1f}, {faint_lim:.1f}]"
            )

            # Query first range: high RA values (near 360)
            adql1 = f"""
            SELECT TOP 500000
                source_id, ra, dec,
                phot_g_mean_mag as G,
                phot_bp_mean_mag as BP,
                phot_rp_mean_mag as RP,
                pmra, pmdec, parallax
            FROM gaiadr3.gaia_source
            WHERE phot_g_mean_mag BETWEEN {bright_lim} AND {faint_lim}
            AND ra >= {min_ra_normalized} AND ra <= 360.0
            AND dec BETWEEN {min_dec} AND {max_dec}
            """

            # Query second range: low RA values (near 0)
            adql2 = f"""
            SELECT TOP 500000
                source_id, ra, dec,
                phot_g_mean_mag as G,
                phot_bp_mean_mag as BP,
                phot_rp_mean_mag as RP,
                pmra, pmdec, parallax
            FROM gaiadr3.gaia_source
            WHERE phot_g_mean_mag BETWEEN {bright_lim} AND {faint_lim}
            AND ra >= 0.0 AND ra <= {max_ra_normalized}
            AND dec BETWEEN {min_dec} AND {max_dec}
            """

            result1 = Gaia.launch_job(adql1).get_results()
            result2 = Gaia.launch_job(adql2).get_results()

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
                f"Querying Gaia: RA=[{min_ra_normalized:.3f}, {max_ra_normalized:.3f}]°, "
                f"DEC=[{min_dec:.3f}, {max_dec:.3f}]°, "
                f"{primary_filter}=[{bright_lim:.1f}, {faint_lim:.1f}]"
            )

            adql = f"""
            SELECT TOP 500000
                source_id, ra, dec,
                phot_g_mean_mag as G,
                phot_bp_mean_mag as BP,
                phot_rp_mean_mag as RP,
                pmra, pmdec, parallax
            FROM gaiadr3.gaia_source
            WHERE phot_g_mean_mag BETWEEN {bright_lim} AND {faint_lim}
            AND ra BETWEEN {min_ra_normalized} AND {max_ra_normalized}
            AND dec BETWEEN {min_dec} AND {max_dec}
            """

            result = Gaia.launch_job(adql).get_results()

        if result is None or len(result) == 0:
            logger.info("No stars found in Gaia query")
            return []

        logger.info(f"Gaia returned {len(result)} stars")

        # Convert to star format matching SSTRC7 structure
        stars = []
        for row in result:
            magnitudes = {}
            # Add available Gaia magnitudes
            if "G" in result.colnames and not np.isnan(row["G"]) and row["G"] < 32:
                magnitudes["Gaia_G"] = float(row["G"])
            if "BP" in result.colnames and not np.isnan(row["BP"]) and row["BP"] < 32:
                magnitudes["Gaia_BP"] = float(row["BP"])
            if "RP" in result.colnames and not np.isnan(row["RP"]) and row["RP"] < 32:
                magnitudes["Gaia_RP"] = float(row["RP"])

            # Compute synthetic Johnson_V and Sloan_r from Gaia BP-RP
            if "Gaia_G" in magnitudes and "Gaia_BP" in magnitudes and "Gaia_RP" in magnitudes:
                from senpai.catalog.gaia_transforms import gaia_bp_rp_to_johnson_v, gaia_bp_rp_to_sloan_r

                bp_rp = magnitudes["Gaia_BP"] - magnitudes["Gaia_RP"]
                johnson_v = gaia_bp_rp_to_johnson_v(magnitudes["Gaia_G"], bp_rp)
                if johnson_v is not None:
                    magnitudes["Johnson_V"] = johnson_v
                sloan_r = gaia_bp_rp_to_sloan_r(magnitudes["Gaia_G"], bp_rp)
                if sloan_r is not None:
                    magnitudes["Sloan_r"] = sloan_r

            # Use requested primary filter band, fallback to G-band, then faint_lim
            if primary_filter == "G" and "G" in result.colnames and not np.isnan(row["G"]) and row["G"] < 32:
                primary_mag = float(row["G"])
            elif primary_filter == "BP" and "BP" in result.colnames and not np.isnan(row["BP"]) and row["BP"] < 32:
                primary_mag = float(row["BP"])
            elif primary_filter == "RP" and "RP" in result.colnames and not np.isnan(row["RP"]) and row["RP"] < 32:
                primary_mag = float(row["RP"])
            elif "G" in result.colnames and not np.isnan(row["G"]) and row["G"] < 32:
                primary_mag = float(row["G"])
            else:
                primary_mag = faint_lim

            # Get proper motion (convert from mas/yr to rad/s)
            ra_pm = 0.0
            dec_pm = 0.0
            if "pmra" in result.colnames and not np.isnan(row["pmra"]):
                # pmra is in mas/yr, convert to rad/s
                mas2rad = 4.84813681109535993589914102358e-9
                year2sec = 3.1556952e7
                ra_pm = float(row["pmra"]) * mas2rad / year2sec
            if "pmdec" in result.colnames and not np.isnan(row["pmdec"]):
                mas2rad = 4.84813681109535993589914102358e-9
                year2sec = 3.1556952e7
                dec_pm = float(row["pmdec"]) * mas2rad / year2sec

            # Get parallax (convert from mas to rad)
            parallax = 0.0
            if "parallax" in result.colnames and not np.isnan(row["parallax"]):
                mas2rad = 4.84813681109535993589914102358e-9
                parallax = float(row["parallax"]) * mas2rad

            star = {
                "ra": np.radians(float(row["ra"])),
                "dec": np.radians(float(row["dec"])),
                "mv": primary_mag,
                "magnitudes": magnitudes,
                "catalog": "Gaia",
                "source_id": str(row["source_id"]),
                "ra_pm": ra_pm,
                "dec_pm": dec_pm,
                "parallax": parallax,
            }
            stars.append(star)

        return stars

    except Exception as e:
        logger.error(f"Gaia query failed: {e}")
        import traceback

        traceback.print_exc()
        return []
