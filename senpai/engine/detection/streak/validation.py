"""Validate a proposed inter-frame pixel shift by comparing catalog-star fluxes.

Provides the flux-correlation metric used to score candidate shifts during the
Bayesian shift optimization in the rate/sidereal streak solvers.
"""

import logging

import numpy as np

from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
from senpai.engine.models.starfield import StarInSpace
from senpai.settings import settings

logger = logging.getLogger(__name__)


def validate_proposed_shift(
    target: RateTrackFrame | SiderealFrame,
    source: RateTrackFrame | SiderealFrame,
    shift_x: float,
    shift_y: float,
    catalog_stars: list[StarInSpace],
) -> float:
    """Validate the proposed shift between frames by comparing star fluxes.

    For up to 50 brightest stars that remain in frame after applying the shift,
    sum up 7x7 boxes in the source and shifted target frames.

    Args:
        target: The frame we're shifting to align with the source frame
        source: The reference frame
        shift_x: Proposed x shift (pixels)
        shift_y: Proposed y shift (pixels)
        catalog_stars: List of stars from the source frame

    Returns:
        The Pearson correlation between source and shifted-target box fluxes, or 0.0
        when there are too few valid stars to compute a correlation.
    """
    target_frame = target.frame.data
    source_frame = source.frame.data

    # Sort stars by flux (brightest first)
    if not catalog_stars or len(catalog_stars) == 0:
        logger.warning("No catalog stars available for shift validation")
        return 0.0

    # Extract positions and fluxes, sort by brightness
    star_positions = []
    for star in catalog_stars:
        if hasattr(star, "x") and hasattr(star, "y") and hasattr(star, "magnitude"):
            # Try flipping x and y to see if that helps with alignment
            star_positions.append((star.y, star.x, star.magnitude))  # Flipped x and y

    if not star_positions:
        logger.warning("No valid star positions found in catalog")
        return 0.0

    # Sort by flux (brightest first) and take up to 50
    star_positions.sort(key=lambda s: s[2], reverse=False)
    star_positions = star_positions[:50]

    logger.debug(f"Validating shift with {len(star_positions)} brightest catalog stars")

    # Define box size for flux measurement
    box_size = 5
    half_box = box_size // 2

    # Height and width of frames
    h, w = source_frame.shape

    # Lists to store results
    source_fluxes = []
    target_fluxes = []
    valid_stars = []

    # For each star, measure flux in both frames
    for y, x, _ in star_positions:  # Note: x and y are flipped here
        # Calculate shifted position
        x_shifted = x - shift_x
        y_shifted = y - shift_y

        # Check if the box around the star is within frame boundaries in both frames
        if (
            x - half_box >= 0
            and x + half_box < w
            and y - half_box >= 0
            and y + half_box < h
            and x_shifted - half_box >= 0
            and x_shifted + half_box < w
            and y_shifted - half_box >= 0
            and y_shifted + half_box < h
        ):
            # Get integer coordinates for indexing
            x_int = int(x)
            y_int = int(y)
            x_shifted_int = int(x_shifted)
            y_shifted_int = int(y_shifted)

            # Sum the box in source frame
            source_box = source_frame[
                y_int - half_box : y_int + half_box + 1, x_int - half_box : x_int + half_box + 1
            ]
            source_flux = np.sum(source_box)

            # Sum the box in target frame at shifted position
            target_box = target_frame[
                y_shifted_int - half_box : y_shifted_int + half_box + 1,
                x_shifted_int - half_box : x_shifted_int + half_box + 1,
            ]
            target_flux = np.sum(target_box)

            # Store results
            source_fluxes.append(source_flux)
            target_fluxes.append(target_flux)
            valid_stars.append((x, y, x_shifted, y_shifted))

            logger.debug(
                f"Star at ({x:.1f}, {y:.1f}): source flux = {source_flux:.1f}, "
                f"target flux at ({x_shifted:.1f}, {y_shifted:.1f}) = {target_flux:.1f}"
            )

    # Log summary statistics
    correlation = 0.0
    if valid_stars:
        logger.debug(f"Validating shift with {len(valid_stars)} stars that remain in frame")
        source_fluxes = np.array(source_fluxes)
        target_fluxes = np.array(target_fluxes)

        # Calculate correlation between source and target fluxes
        median_ratio = 0.0
        ratio_std = 0.0
        if len(source_fluxes) > 1:
            correlation = np.corrcoef(source_fluxes, target_fluxes)[0, 1]
            logger.debug(f"Flux correlation between frames: {correlation:.3f}")

            # Calculate individual flux ratios and their consistency
            individual_ratios = target_fluxes / np.where(source_fluxes > 0, source_fluxes, 1)
            median_ratio = np.median(individual_ratios)
            ratio_std = np.std(individual_ratios)
    else:
        logger.warning("No stars remained in frame after applying shift")

    # Plot the target frame with boxes showing the shifted positions
    if settings.plotting.debug and valid_stars:  # pragma: no cover  # debug-only plotting branch
        from senpai.engine.plotting.images import plot_shift_validation

        plot_shift_validation(
            source_frame=source_frame,
            target_frame=target_frame,
            valid_stars=valid_stars,
            source_fluxes=source_fluxes,
            target_fluxes=target_fluxes,
            shift_x=shift_x,
            shift_y=shift_y,
            correlation=correlation,
            median_ratio=median_ratio,
            ratio_std=ratio_std,
            box_size=box_size,
            half_box=half_box,
            source_index=source.index,
            target_index=target.index,
        )

    return correlation
