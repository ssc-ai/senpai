"""Astrometry residual error diagnostics."""

import logging

import numpy as np

from senpai.engine.models.astrometry import WCSModel

logger = logging.getLogger(__name__)


def calculate_residual_errors(
    wcs_model: WCSModel, stars_with_radec_xy: list[tuple]
) -> dict:
    """Calculate residual errors between detected and WCS-predicted positions.

    Args:
        wcs_model: WCSModel to use for predictions
        stars_with_radec_xy: List of tuples (ra, dec, x_detected, y_detected)

    Returns:
        dict with 'x_errors', 'y_errors', 'radial_errors' and statistics
    """
    if not stars_with_radec_xy:
        return {}

    x_errors = []
    y_errors = []
    radial_errors = []

    for ra, dec, x_detected, y_detected in stars_with_radec_xy:
        # Convert RA/Dec to pixels using WCS
        x_wcs, y_wcs = wcs_model.world2pix_0based(ra, dec)

        # Calculate errors
        x_err = x_detected - x_wcs
        y_err = y_detected - y_wcs
        radial_err = np.sqrt(x_err**2 + y_err**2)

        x_errors.append(x_err)
        y_errors.append(y_err)
        radial_errors.append(radial_err)

    x_errors = np.array(x_errors)
    y_errors = np.array(y_errors)
    radial_errors = np.array(radial_errors)

    def calc_stats(errors):
        return {
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "std": float(np.std(errors)),
            "p50": float(np.percentile(errors, 50)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99)),
        }

    return {
        "x_errors": x_errors,
        "y_errors": y_errors,
        "radial_errors": radial_errors,
        "x_stats": calc_stats(x_errors),
        "y_stats": calc_stats(y_errors),
        "radial_stats": calc_stats(radial_errors),
    }


def log_residual_errors(phase_name: str, error_dict: dict):
    """Log residual error statistics in a formatted way.

    Args:
        phase_name: Name of the phase (e.g., "Phase 1 - Before SIP fit")
        error_dict: Dictionary returned from calculate_residual_errors()
    """
    if not error_dict:
        logger.warning(f"{phase_name}: No error data available")
        return

    x_stats = error_dict["x_stats"]
    y_stats = error_dict["y_stats"]
    radial_stats = error_dict["radial_stats"]

    logger.info(f"{phase_name} - Residual Errors:")
    logger.info(
        f"  X errors: mean={x_stats['mean']:.3f}, std={x_stats['std']:.3f}, "
        f"median={x_stats['median']:.3f}, p95={x_stats['p95']:.3f}, p99={x_stats['p99']:.3f} pixels"
    )
    logger.info(
        f"  Y errors: mean={y_stats['mean']:.3f}, std={y_stats['std']:.3f}, "
        f"median={y_stats['median']:.3f}, p95={y_stats['p95']:.3f}, p99={y_stats['p99']:.3f} pixels"
    )
    logger.info(
        f"  Radial errors: mean={radial_stats['mean']:.3f}, std={radial_stats['std']:.3f}, "
        f"median={radial_stats['median']:.3f}, p95={radial_stats['p95']:.3f}, p99={radial_stats['p99']:.3f} pixels"
    )
