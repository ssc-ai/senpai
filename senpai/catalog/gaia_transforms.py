"""Gaia DR3 photometric transforms to standard bands.

Polynomial transforms from Gaia G, BP, RP to Johnson V and Sloan r
using BP-RP color index. Coefficients from Gaia DR3 documentation
(Riello et al. 2021, Table 5.7/5.9).
"""

import logging

logger = logging.getLogger(__name__)


def gaia_bp_rp_to_johnson_v(g: float, bp_rp: float) -> float | None:
    """Transform Gaia G magnitude to Johnson V using BP-RP color.

    Uses polynomial: V = G - (-0.02704 + 0.01424*X - 0.2156*X^2 + 0.01426*X^3)
    where X = BP - RP.

    Valid for -0.5 < BP-RP < 5.0.

    Parameters
    ----------
    g : float
        Gaia G-band magnitude
    bp_rp : float
        BP - RP color index

    Returns
    -------
    float or None
        Johnson V magnitude, or None if color is outside valid range
    """
    if bp_rp < -0.5 or bp_rp > 5.0:
        return None

    correction = -0.02704 + 0.01424 * bp_rp - 0.2156 * bp_rp**2 + 0.01426 * bp_rp**3
    return g - correction


def gaia_bp_rp_to_sloan_r(g: float, bp_rp: float) -> float | None:
    """Transform Gaia G magnitude to Sloan r using BP-RP color.

    Uses polynomial: r = G - (-0.09837 + 0.08592*X - 0.1907*X^2 + 0.01144*X^3)
    where X = BP - RP.

    Valid for -0.5 < BP-RP < 4.0.

    Parameters
    ----------
    g : float
        Gaia G-band magnitude
    bp_rp : float
        BP - RP color index

    Returns
    -------
    float or None
        Sloan r magnitude, or None if color is outside valid range
    """
    if bp_rp < -0.5 or bp_rp > 4.0:
        return None

    correction = -0.09837 + 0.08592 * bp_rp - 0.1907 * bp_rp**2 + 0.01144 * bp_rp**3
    return g - correction
