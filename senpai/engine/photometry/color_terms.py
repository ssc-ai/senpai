"""Multi-band zero point calibration with color term corrections.

Fits the relation: m_catalog - m_inst = ZP + C * color_index
for multiple target photometric bands using sigma-clipped linear regression.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from senpai.engine.models.starfield import StarField

logger = logging.getLogger(__name__)


@dataclass
class ColorTermFit:
    """Result of fitting m_target - m_inst = ZP + C * color_index."""

    band: str
    zero_point: float
    zero_point_err: float
    color_coefficient: float
    color_coefficient_err: float
    color_index_name: str
    n_stars: int
    rms_residual: float
    clipped_fraction: float

    def __post_init__(self):
        self.zero_point = round(self.zero_point, 3)
        self.zero_point_err = round(self.zero_point_err, 4)
        self.color_coefficient = round(self.color_coefficient, 4)
        self.color_coefficient_err = round(self.color_coefficient_err, 4)
        self.rms_residual = round(self.rms_residual, 4)
        self.clipped_fraction = round(self.clipped_fraction, 3)


@dataclass
class BandCalibration:
    """Calibration result for one target band."""

    band: str
    zero_point: float
    zero_point_err: float
    color_term: ColorTermFit | None = None
    method: str = "simple"

    def __post_init__(self):
        self.zero_point = round(self.zero_point, 3)
        self.zero_point_err = round(self.zero_point_err, 4)


@dataclass
class MultiBandCalibration:
    """Collection of calibrations across target bands."""

    bands: dict[str, BandCalibration] = field(default_factory=dict)
    observation_filter: str | None = None
    color_index_name: str = "BP-RP"


def fit_color_term(
    instrumental_mags: np.ndarray,
    catalog_mags: np.ndarray,
    color_indices: np.ndarray,
    band: str = "",
    color_index_name: str = "BP-RP",
    sigma_clip: float = 2.5,
    min_stars: int = 5,
) -> ColorTermFit | None:
    """Fit m_cat - m_inst = ZP + C * color via sigma-clipped linear regression.

    Parameters
    ----------
    instrumental_mags : np.ndarray
        Instrumental magnitudes (-2.5 * log10(flux/texp))
    catalog_mags : np.ndarray
        Catalog magnitudes in the target band
    color_indices : np.ndarray
        Color index values (e.g. BP - RP)
    band : str
        Name of the target band
    color_index_name : str
        Name of the color index used
    sigma_clip : float
        Sigma clipping threshold for outlier rejection
    min_stars : int
        Minimum number of stars required for fitting

    Returns
    -------
    ColorTermFit or None
        Fit result, or None if insufficient stars
    """
    if len(instrumental_mags) < min_stars:
        return None

    delta = catalog_mags - instrumental_mags
    n_initial = len(delta)

    # Mask non-finite values
    mask = np.isfinite(delta) & np.isfinite(color_indices)
    if np.sum(mask) < min_stars:
        return None

    x = color_indices[mask]
    y = delta[mask]

    # Iterative sigma-clipped linear fit
    for _ in range(3):
        if len(x) < min_stars:
            return None

        coeffs = np.polyfit(x, y, 1)
        color_coeff, zp = coeffs

        residuals = y - (zp + color_coeff * x)
        rms = float(np.std(residuals))

        if rms <= 0:
            break

        keep = np.abs(residuals) < sigma_clip * rms
        if np.sum(keep) < min_stars:
            break
        x = x[keep]
        y = y[keep]

    # Final fit on clipped data
    if len(x) < min_stars:
        return None

    coeffs = np.polyfit(x, y, 1)
    color_coeff, zp = coeffs

    residuals = y - (zp + color_coeff * x)
    rms = float(np.std(residuals))
    n_final = len(x)
    clipped_fraction = 1.0 - n_final / n_initial

    # Uncertainty estimates from residual scatter
    zp_err = rms / np.sqrt(n_final) if n_final > 0 else 0.0
    # Color coefficient uncertainty from polyfit covariance
    if n_final > 2:
        _, cov = np.polyfit(x, y, 1, cov=True)
        color_coeff_err = float(np.sqrt(cov[0, 0]))
        zp_err = float(np.sqrt(cov[1, 1]))
    else:
        color_coeff_err = 0.0

    return ColorTermFit(
        band=band,
        zero_point=float(zp),
        zero_point_err=float(zp_err),
        color_coefficient=float(color_coeff),
        color_coefficient_err=float(color_coeff_err),
        n_stars=n_final,
        rms_residual=rms,
        clipped_fraction=clipped_fraction,
        color_index_name=color_index_name,
    )


def calculate_multiband_calibration(
    results: list,
    starfield: StarField,
    target_bands: list[str],
    config,
    observation_filter: str | None = None,
) -> MultiBandCalibration | None:
    """Calculate zero points with color corrections for multiple target bands.

    Parameters
    ----------
    results : list[SimplePhotometryResult]
        Photometry results with instrumental magnitudes
    starfield : StarField
        Starfield containing catalog star magnitudes
    target_bands : list[str]
        Target band names (e.g. ["Johnson_V", "Sloan_r", "Gaia_G"])
    config : SimplePhotometryConfig
        Photometry configuration
    observation_filter : str or None
        Observation filter name (e.g. "Clear", "V")

    Returns
    -------
    MultiBandCalibration or None
        Multi-band calibration results, or None if no bands could be calibrated
    """
    if not results or not target_bands:
        return None

    # Get exposure time
    exposure_time = getattr(starfield.image_metadata, "exposure_time", None)
    if exposure_time is None or exposure_time <= 0:
        exposure_time = 1.0

    color_bp_key = "Gaia_BP"
    color_rp_key = "Gaia_RP"
    color_index_name = "BP-RP"

    calibration = MultiBandCalibration(
        observation_filter=observation_filter,
        color_index_name=color_index_name,
    )

    for band in target_bands:
        # Collect matched data: instrumental mag, catalog mag in band, color index
        inst_mags = []
        cat_mags = []
        colors = []

        for r in results:
            if not r.quality_flag or r.flux <= 0:
                continue

            inst_mag = -2.5 * np.log10(r.flux / exposure_time)

            star = r.star
            if not star.magnitudes:
                continue

            cat_mag = star.magnitudes.get(band)
            if cat_mag is None:
                continue

            bp = star.magnitudes.get(color_bp_key)
            rp = star.magnitudes.get(color_rp_key)
            color = None
            if bp is not None and rp is not None:
                color = bp - rp

            inst_mags.append(inst_mag)
            cat_mags.append(cat_mag)
            colors.append(color)

        if len(inst_mags) < 3:
            logger.debug(f"Skipping band {band}: only {len(inst_mags)} matched stars")
            continue

        inst_arr = np.array(inst_mags)
        cat_arr = np.array(cat_mags)

        # Try color term fit if we have enough stars with color info
        enable_color_terms = getattr(config, "enable_color_terms", True)
        color_fit = None
        has_colors = [c is not None for c in colors]
        n_with_color = sum(has_colors)

        if enable_color_terms and n_with_color >= 5:
            color_mask = np.array(has_colors)
            color_arr = np.array([c if c is not None else 0.0 for c in colors])

            color_fit = fit_color_term(
                instrumental_mags=inst_arr[color_mask],
                catalog_mags=cat_arr[color_mask],
                color_indices=color_arr[color_mask],
                band=band,
                color_index_name=color_index_name,
            )

        if color_fit is not None:
            calibration.bands[band] = BandCalibration(
                band=band,
                zero_point=color_fit.zero_point,
                zero_point_err=color_fit.zero_point_err,
                color_term=color_fit,
                method="color_term",
            )
            logger.info(
                f"Band {band}: ZP={color_fit.zero_point:.3f}+-{color_fit.zero_point_err:.3f}, "
                f"C={color_fit.color_coefficient:.4f}+-{color_fit.color_coefficient_err:.4f}, "
                f"N={color_fit.n_stars}, RMS={color_fit.rms_residual:.3f}"
            )
        else:
            # Simple zero point (no color term)
            delta = cat_arr - inst_arr
            finite = np.isfinite(delta)
            if np.sum(finite) < 3:
                continue
            delta = delta[finite]

            # Sigma clip
            med = float(np.median(delta))
            std = float(np.std(delta))
            if std > 0:
                keep = np.abs(delta - med) < 2.5 * std
                delta = delta[keep]

            if len(delta) < 3:
                continue

            zp = float(np.mean(delta))
            zp_err = float(np.std(delta) / np.sqrt(len(delta)))

            calibration.bands[band] = BandCalibration(
                band=band,
                zero_point=zp,
                zero_point_err=zp_err,
                method="simple",
            )
            logger.info(f"Band {band}: ZP={zp:.3f}+-{zp_err:.3f} (simple, N={len(delta)})")

    if not calibration.bands:
        return None

    return calibration
