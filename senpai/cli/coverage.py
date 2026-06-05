#!/usr/bin/env python3
"""
Sky Coverage Analysis Tool

Analyzes star coverage across the full sky for different FOVs and magnitude limits.
Uses convolution analysis to find min/max star counts per position.
"""

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Disable matplotlib logging - set before AND after import to catch all loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy import wcs  # noqa: E402
from scipy.signal import convolve2d  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Set again after import to ensure it sticks
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)
logging.getLogger("matplotlib.colorbar").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

import senpai.catalog.sstr7 as sstr7  # noqa: E402
from senpai.core.config import initialize_config  # noqa: E402
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE  # noqa: E402
from senpai.core.logging import set_log_level  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class CoverageStatistics:
    """Statistics for a single grid position, FOV, and magnitude threshold."""

    grid_position: Tuple[float, float]  # (ra, dec)
    fov: float
    magnitude_threshold: float
    min_stars: int
    max_stars: int
    mean_stars: float
    median_stars: float


@dataclass
class AggregatedStatistics:
    """Aggregated statistics across all sky positions for plotting."""

    fov: float
    magnitude_threshold: float
    global_min: int
    global_max: int
    mean_min: float
    mean_max: float
    percentiles: Dict[str, float]


@dataclass
class CorridorData:
    """Data for a single Earth-Moon corridor time step."""

    ra_center: float
    dec_center: float
    ra_earth: float
    dec_earth: float
    ra_moon: float
    dec_moon: float
    earth_moon_separation_deg: float
    time: str


def generate_sky_grid(
    max_fov: float,
    grid_spacing_mult: float = 2.0,
    test_mode: bool = False,
    degrees_off_geo_belt: float | None = None,
) -> List[Tuple[float, float]]:
    """
    Generate a grid of RA/Dec positions covering the sky.

    Args:
        max_fov: Maximum FOV in degrees
        grid_spacing_mult: Multiplier for grid spacing (default: 2.0, meaning 2*max_fov)
        test_mode: If True, create 3x3 grid around test location
        degrees_off_geo_belt: If specified, limit grid to Dec range [-degrees_off_geo_belt, +degrees_off_geo_belt]
            (geo belt = celestial equator at Dec=0). If None, covers full sky (avoiding poles).

    Returns:
        List of (ra, dec) tuples in degrees
    """
    if test_mode:
        # Create 3x3 grid around test location (RA=180, Dec=0)
        test_ra = 180.0
        test_dec = 0.0
        spacing = max_fov  # 3x3 grid with max_fov spacing

        grid_positions = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                ra = test_ra + i * spacing / np.cos(np.radians(test_dec))
                dec = test_dec + j * spacing
                # Normalize RA to [0, 360)
                ra = ra % 360.0
                grid_positions.append((ra, dec))
        return grid_positions

    # Full sky grid
    spacing = grid_spacing_mult * max_fov
    grid_positions = []

    # Determine Dec range based on degrees_off_geo_belt parameter
    half_fov = max_fov / 2.0
    if degrees_off_geo_belt is not None:
        # Limit to geo belt region (celestial equator ± degrees_off_geo_belt)
        # Clamp to avoid poles, but allow user to specify up to 90
        max_dec_offset = min(degrees_off_geo_belt, 90.0 - half_fov)
        dec_min = -max_dec_offset
        dec_max = max_dec_offset
        logger.info(
            f"Geo belt mode: limiting grid to Dec range [{dec_min:.2f}°, {dec_max:.2f}°] "
            f"(±{degrees_off_geo_belt}° from equator)"
        )
    else:
        # Full sky: avoid exact poles (±90°) since fields at poles have undefined RA
        # Instead, place grid positions at the edge of the pole coverage
        dec_min = -90.0 + half_fov  # Start half FOV away from South Pole
        dec_max = 90.0 - half_fov  # End half FOV away from North Pole
    dec_steps = int((dec_max - dec_min) / spacing) + 1

    for dec_idx in range(dec_steps):
        dec = dec_min + dec_idx * spacing
        if dec > dec_max:
            dec = dec_max

        # RA range: 0 to 360 degrees
        # Adjust RA spacing based on declination to maintain consistent angular separation
        ra_spacing = spacing / max(np.cos(np.radians(dec)), 0.01)  # Avoid division by zero
        ra_steps = int(360.0 / ra_spacing) + 1

        for ra_idx in range(ra_steps):
            ra = ra_idx * ra_spacing
            if ra >= 360.0:
                ra = 0.0
            grid_positions.append((ra, dec))

    return grid_positions


def query_stars_for_position(
    ra: float,
    dec: float,
    fov_size: float,
    catalog_path: str,
    faint_lim: float,
    bright_lim: float = None,
    x_fov: float | None = None,
    y_fov: float | None = None,
) -> List[Dict]:
    """
    Query catalog stars for a given sky position.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        fov_size: FOV size in degrees (used if x_fov/y_fov not provided,
            will query 2*fov_size to ensure complete coverage)
        catalog_path: Path to SSTR7 catalog
        faint_lim: Faint magnitude limit
        bright_lim: Bright magnitude limit (optional)
        x_fov: Custom RA width for query region (degrees, optional)
        y_fov: Custom Dec height for query region (degrees, optional)

    Returns:
        List of star dictionaries with 'ra', 'dec', 'mv' keys
    """
    # Use custom dimensions if provided, otherwise use 2*fov_size for both
    if x_fov is not None and y_fov is not None:
        query_x_fov = x_fov
        query_y_fov = y_fov
        # For compatibility with code that expects query_fov, use max dimension
        query_fov = max(query_x_fov, query_y_fov)
    else:
        # Query stars in a 2*fov_size field to ensure complete coverage for convolution
        query_fov = 2 * fov_size
        query_x_fov = query_fov
        query_y_fov = query_fov

    # Log what we're querying
    logger.debug(
        f"  Querying catalog: center=({ra:.2f}°, {dec:.2f}°), x_fov={query_x_fov:.2f}°, y_fov={query_y_fov:.2f}°"
    )

    stars = sstr7.query_by_los_radec_with_rotation(
        y_fov=query_y_fov,
        x_fov=query_x_fov,
        ra=ra,
        dec=dec,
        rotation=0.0,
        rootPath=catalog_path,
        faint_lim=faint_lim,
        bright_lim=bright_lim,
        safety_margin=0.1,
    )

    # Calculate what the bounding box should be (for debugging)
    safety_margin = 0.1
    query_x_fov_with_margin = query_x_fov * (1 + safety_margin)
    query_y_fov_with_margin = query_y_fov * (1 + safety_margin)
    half_x_fov = query_x_fov_with_margin / 2.0
    half_y_fov = query_y_fov_with_margin / 2.0

    # Expected bounds (approximate, accounting for spherical distortion)
    expected_ra_min = ra - half_x_fov / max(np.cos(np.radians(dec)), 0.01)
    expected_ra_max = ra + half_x_fov / max(np.cos(np.radians(dec)), 0.01)
    expected_dec_min = dec - half_y_fov
    expected_dec_max = dec + half_y_fov

    logger.debug(
        f"  Expected query bounds: RA=[{expected_ra_min:.2f}, {expected_ra_max:.2f}], "
        f"Dec=[{expected_dec_min:.2f}, {expected_dec_max:.2f}]"
    )

    # Convert to degrees and normalize, then filter using WCS to define field bounds
    # Convert all stars first (like test script does)
    for star in stars:
        ra_rad = star["ra"]
        dec_rad = star["dec"]

        # Convert from radians to degrees
        ra_deg = np.degrees(ra_rad)
        dec_deg = np.degrees(dec_rad)

        # Normalize RA to [0, 360) range
        ra_deg = ra_deg % 360.0

        # Ensure Dec is in valid range [-90, 90]
        if dec_deg > 90.0:
            dec_deg = 90.0
        elif dec_deg < -90.0:
            dec_deg = -90.0

        star["ra"] = ra_deg
        star["dec"] = dec_deg

    # Log converted bounds (after normalization, like test script)
    if stars:
        ra_values = [s["ra"] for s in stars]
        dec_values = [s["dec"] for s in stars]
        ra_min, ra_max = np.min(ra_values), np.max(ra_values)
        dec_min, dec_max = np.min(dec_values), np.max(dec_values)
        logger.debug(
            f"  Star coordinates (degrees, after conversion): RA=[{ra_min:.2f}, {ra_max:.2f}], "
            f"Dec=[{dec_min:.2f}, {dec_max:.2f}]"
        )

    # Now filter using WCS (or distance-based filtering for poles)
    filtered_stars = []

    # Special handling for poles: TAN projection breaks down at Dec = ±90°
    # At the pole, all RA values converge, so we use angular distance filtering instead
    is_pole = abs(dec) >= 89.0  # Within 1° of pole

    if is_pole:
        logger.debug(f"  Near pole (Dec={dec:.2f}°), using angular distance filtering instead of WCS")
        safety_margin = 0.1
        query_fov_with_margin = query_fov * (1 + safety_margin)
        max_angular_distance = query_fov_with_margin / 2.0  # Half FOV in degrees

        # At the pole, angular distance from the pole is simply the Dec difference
        # For South Pole (Dec=-90), we want stars with Dec from -90 to -90 + max_angular_distance
        # For North Pole (Dec=+90), we want stars with Dec from +90 - max_angular_distance to +90
        # Since we're querying around the pole, stars should already be in the right Dec range
        # We just need to filter by Dec difference (distance from pole)
        for star in stars:
            ra_deg = star["ra"]
            dec_deg = star["dec"]

            # Angular distance from pole is just the absolute Dec difference
            # For South Pole (dec=-90), dec_deg should be >= dec (more negative is closer to -90)
            # For North Pole (dec=+90), dec_deg should be <= dec (less positive is closer to +90)
            if dec < 0:  # South Pole
                # Distance from South Pole: stars should have Dec >= center Dec (closer to -90)
                dec_diff = abs(dec_deg - dec)  # This is the distance from -90
            else:  # North Pole
                # Distance from North Pole: stars should have Dec <= center Dec (closer to +90)
                dec_diff = abs(dec_deg - dec)  # This is the distance from +90

            if dec_diff <= max_angular_distance:
                filtered_stars.append(star)
    else:
        # Normal WCS-based filtering for non-pole positions
        safety_margin = 0.1  # This matches the safety_margin in the query call
        query_fov_with_margin = query_fov * (1 + safety_margin)

        # Create a WCS with center at (ra, dec) and FOV = query_fov_with_margin
        # Use a reasonable pixel scale (e.g., 0.01 degrees per pixel for a ~100x100 pixel field)
        pixel_scale = query_fov_with_margin / 100.0  # ~100 pixels across the field
        naxis = 100  # Number of pixels in each dimension

        # Create WCS with center at reference pixel
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [naxis / 2.0 + 1, naxis / 2.0 + 1]  # Center pixel (1-based)
        w.wcs.cdelt = [-pixel_scale, pixel_scale]  # Negative for RA (increases to west)
        w.wcs.crval = [ra, dec]  # Center coordinates
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        logger.debug(
            f"  Filtering bounds: query_fov={query_fov:.2f}°, with margin={query_fov_with_margin:.2f}° "
            f"(using WCS with {naxis}x{naxis} pixels, {pixel_scale:.4f}°/pixel)"
        )

        # Define field bounds in pixel space (with some padding for edge cases)
        # Field extends from pixel 0 to naxis in both dimensions
        pixel_tolerance = 5  # Allow stars slightly outside the field (5 pixels)
        min_pixel = -pixel_tolerance
        max_pixel = naxis + pixel_tolerance

        for star in stars:
            # Stars are already converted to degrees above
            ra_deg = star["ra"]
            dec_deg = star["dec"]

            # Convert star coordinates to pixel coordinates using WCS
            try:
                pix_x, pix_y = w.wcs_world2pix([[ra_deg, dec_deg]], 0)[0]

                # Check if star is within field bounds (with tolerance)
                if min_pixel <= pix_x <= max_pixel and min_pixel <= pix_y <= max_pixel:
                    filtered_stars.append(star)
            except Exception as e:
                # If coordinate conversion fails, skip this star
                logger.debug(f"  WCS conversion failed for star at ({ra_deg:.2f}, {dec_deg:.2f}): {e}")
                continue

    # Log filtering results
    if stars:
        original_count = len(stars)
        filtered_count = len(filtered_stars)
        if original_count != filtered_count:
            logger.info(
                f"  Filtered stars: {filtered_count}/{original_count} stars within WCS field bounds "
                f"(FOV={query_fov_with_margin:.2f}° with {pixel_tolerance} pixel tolerance)"
            )

        # If filtering removed all stars, log detailed diagnostics
        if filtered_count == 0 and original_count > 0:
            logger.error(
                f"  ERROR: All {original_count} stars filtered out! "
                f"WCS filtering issue at center=({ra:.2f}°, {dec:.2f}°)"
            )
            # Check a sample of stars to see where they're being placed
            sample_size = min(10, original_count)
            sample_indices = np.linspace(0, original_count - 1, sample_size, dtype=int)
            logger.debug(f"  Checking {sample_size} sample stars:")
            for idx in sample_indices:
                star = stars[idx]
                try:
                    pix_x, pix_y = w.wcs_world2pix([[star["ra"], star["dec"]]], 0)[0]
                    in_bounds = min_pixel <= pix_x <= max_pixel and min_pixel <= pix_y <= max_pixel
                    logger.debug(
                        f"    Star {idx}: RA={star['ra']:.2f}°, Dec={star['dec']:.2f}° -> "
                        f"pix=({pix_x:.2f}, {pix_y:.2f}) [bounds: {min_pixel} to {max_pixel}, in_bounds={in_bounds}]"
                    )
                except Exception as e:
                    logger.debug(f"    Star {idx}: WCS conversion failed: {e}")

        if filtered_stars:
            ra_values = [s["ra"] for s in filtered_stars]
            dec_values = [s["dec"] for s in filtered_stars]
            ra_min, ra_max = np.min(ra_values), np.max(ra_values)
            dec_min, dec_max = np.min(dec_values), np.max(dec_values)

            # Calculate actual RA span accounting for wraparound
            ra_values_array = np.array(ra_values)
            # Shift RA values to be relative to center, handling wraparound
            ra_relative = (ra_values_array - ra) % 360
            ra_relative = np.where(ra_relative > 180, ra_relative - 360, ra_relative)
            ra_span_actual = np.max(ra_relative) - np.min(ra_relative)

            # Calculate span ignoring wraparound (for display)
            ra_span_display = ra_max - ra_min
            if ra_span_display > 180:
                # Values wrap around - show both min/max and actual span
                logger.info(
                    f"  Filtered star coordinate bounds (degrees): RA=[{ra_min:.2f}, {ra_max:.2f}] "
                    f"(wraps around, actual span: {ra_span_actual:.2f}°), "
                    f"Dec=[{dec_min:.2f}, {dec_max:.2f}], Center=({ra:.2f}, {dec:.2f})"
                )
            else:
                logger.info(
                    f"  Filtered star coordinate bounds (degrees): RA=[{ra_min:.2f}, {ra_max:.2f}] "
                    f"(span: {ra_span_actual:.2f}°), "
                    f"Dec=[{dec_min:.2f}, {dec_max:.2f}] (span: {dec_max - dec_min:.2f}°), "
                    f"Center=({ra:.2f}, {dec:.2f})"
                )

            # Expected span should be roughly 2*query_fov (with some tolerance)
            expected_ra_span = query_fov * 2.2  # Allow 20% extra for rounding
            expected_dec_span = query_fov * 2.2

            if ra_span_actual > expected_ra_span:
                logger.warning(
                    f"  WARNING: RA span is {ra_span_actual:.2f}° (expected ~{expected_ra_span:.2f}°). "
                    f"This may indicate a filtering issue."
                )
            if dec_max - dec_min > expected_dec_span:
                logger.warning(
                    f"  WARNING: Dec span is {dec_max - dec_min:.2f}° (expected ~{expected_dec_span:.2f}°). "
                    f"This may indicate a filtering issue."
                )

            # Check if there are stars near the center
            center_stars = [
                s for s in filtered_stars if abs(((s["ra"] - ra) % 360)) <= 0.5 and abs(s["dec"] - dec) <= 0.5
            ]
            logger.debug(f"  Stars within 0.5° of center ({ra:.2f}°, {dec:.2f}°): {len(center_stars)}")

            # Check star density distribution
            if len(filtered_stars) > 0:
                # Divide into quadrants to see if distribution is uniform
                ra_values_array = np.array(ra_values)
                dec_values_array = np.array(dec_values)
                ra_relative = (ra_values_array - ra) % 360
                ra_relative = np.where(ra_relative > 180, ra_relative - 360, ra_relative)

                # Count stars in each quadrant relative to center
                q1 = np.sum((ra_relative >= 0) & (dec_values_array >= dec))  # Upper right
                q2 = np.sum((ra_relative < 0) & (dec_values_array >= dec))  # Upper left
                q3 = np.sum((ra_relative < 0) & (dec_values_array < dec))  # Lower left
                q4 = np.sum((ra_relative >= 0) & (dec_values_array < dec))  # Lower right

                logger.debug(
                    f"  Star distribution by quadrant (relative to center): UR={q1}, UL={q2}, LL={q3}, LR={q4}"
                )
        else:
            logger.warning(f"  WARNING: All {original_count} stars were filtered out! This may indicate a problem.")

    return filtered_stars


def analyze_fov_coverage(
    stars: List[Dict],
    center_ra: float,
    center_dec: float,
    fov_size: float,
    magnitude_thresholds: List[float],
    conv_resolution: float = 0.1,
    min_resolution: float = 0.01,
    min_fov: float | None = None,
) -> Dict[float, Dict[str, float]]:
    """
    Analyze star coverage for a given FOV using convolution.

    Args:
        stars: List of star dictionaries with 'ra', 'dec', 'mv' keys (in degrees)
        center_ra: Center RA of the field in degrees
        center_dec: Center Dec of the field in degrees
        fov_size: FOV size in degrees
        magnitude_thresholds: List of magnitude thresholds to analyze
        conv_resolution: Base grid resolution for convolution in degrees (default: 0.1)
            This is used as a fallback for backward compatibility.
        min_resolution: Minimum resolution in degrees (default: 0.01)
            Smaller FOVs will use this fine resolution for accuracy.
        min_fov: Minimum FOV size in degrees (optional)
            If provided, resolution scales proportionally: resolution = min_resolution * (fov_size / min_fov)

    Returns:
        Dictionary mapping magnitude_threshold -> statistics dict
    """
    if not stars:
        # Return empty statistics for all thresholds
        return {mag: {"min": 0, "max": 0, "mean": 0.0, "median": 0.0} for mag in magnitude_thresholds}

    # Adaptive resolution: scale proportionally with FOV size
    # If min_fov is provided, scale resolution: min_resolution for min_fov, scales linearly with FOV
    # This maintains constant pixel density (~10 pixels per FOV) for maximum speed
    # Example: min_fov=0.1° uses 0.01°, so 4.0° FOV uses 0.4° (40x coarser, same pixels per FOV)
    if min_fov is not None and min_fov > 0:
        # Scale resolution proportionally: resolution scales with FOV size
        adaptive_resolution = min_resolution * (fov_size / min_fov)
        # But still cap at a reasonable maximum (e.g., 0.5° for very large FOVs)
        adaptive_resolution = min(adaptive_resolution, 0.5)
        logger.debug(
            f"  Proportional scaling: min_fov={min_fov:.3f}°, "
            f"fov_size={fov_size:.3f}°, ratio={fov_size / min_fov:.1f}x, "
            f"resolution={adaptive_resolution:.4f}°"
        )
    else:
        # Fallback to original formula if min_fov not provided
        adaptive_resolution = max(min_resolution, fov_size / 100.0)
        logger.debug(
            f"  Fallback scaling: min_fov={min_fov}, "
            f"using formula max({min_resolution:.4f}, {fov_size:.3f}/100) = {adaptive_resolution:.4f}°"
        )

    # Log if resolution is significantly different from base
    if abs(adaptive_resolution - conv_resolution) > 0.001:
        logger.debug(
            f"  FOV {fov_size:.3f}°: Using adaptive resolution {adaptive_resolution:.4f}° "
            f"(base: {conv_resolution:.4f}°)"
        )

    # Create grid centered on center_ra, center_dec
    # Grid covers 2*fov_size field (stars were queried in this field)
    field_size = 2 * fov_size

    # Account for declination when calculating RA extent
    ra_extent = field_size / max(np.cos(np.radians(center_dec)), 0.01)
    dec_extent = field_size

    # Grid bounds centered on center position
    ra_min_grid = center_ra - ra_extent / 2
    ra_max_grid = center_ra + ra_extent / 2
    dec_min_grid = center_dec - dec_extent / 2
    dec_max_grid = center_dec + dec_extent / 2

    # Use adaptive resolution for grid and kernel
    ra_grid_size = int((ra_max_grid - ra_min_grid) / adaptive_resolution) + 1
    dec_grid_size = int((dec_max_grid - dec_min_grid) / adaptive_resolution) + 1

    # Create coordinate arrays
    ra_coords = np.linspace(ra_min_grid, ra_max_grid, ra_grid_size)
    dec_coords = np.linspace(dec_min_grid, dec_max_grid, dec_grid_size)

    # Create FOV kernel (all 1.0s)
    # Ensure kernel_size is at least 1, but we need to properly represent the FOV area
    kernel_size = max(1, int(np.ceil(fov_size / adaptive_resolution)))
    kernel = np.ones((kernel_size, kernel_size))

    # Log warning if kernel is too small to accurately represent FOV
    actual_fov_coverage = kernel_size * adaptive_resolution
    if actual_fov_coverage < fov_size * 0.9:  # If we're covering less than 90% of the FOV
        fov_pct = (actual_fov_coverage / fov_size) * 100
        logger.warning(
            f"FOV {fov_size:.3f}° is small relative to resolution {adaptive_resolution:.4f}°. "
            f"Kernel size {kernel_size} covers {actual_fov_coverage:.3f}° ({fov_pct:.1f}% of FOV)"
        )

    results = {}

    for mag_threshold in magnitude_thresholds:
        # Filter stars: magnitude <= threshold (cumulative: "this mag and brighter")
        # Include stars with valid magnitudes (mv < 32, where 32 means "no magnitude")
        filtered_stars = [s for s in stars if s["mv"] <= mag_threshold and s["mv"] < 32]

        if not filtered_stars:
            results[mag_threshold] = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
            continue

        # Create star density grid
        star_grid = np.zeros((dec_grid_size, ra_grid_size))

        # Filter stars to only include those within the grid bounds
        # This is important because we query stars in a 2*max_fov field, but for smaller FOVs
        # we only want to analyze stars within the 2*fov_size region
        # Add a small tolerance to account for rounding and coordinate transformations
        bounds_tolerance = adaptive_resolution * 0.5  # Allow stars slightly outside bounds
        stars_in_bounds = []
        for star in filtered_stars:
            star_ra = star["ra"]
            star_dec = star["dec"]

            # Check dec bounds with tolerance
            if not (dec_min_grid - bounds_tolerance <= star_dec <= dec_max_grid + bounds_tolerance):
                continue

            # Check RA bounds (handle wraparound) with tolerance
            # For RA, we need to handle wraparound more carefully
            ra_diff = (star_ra - center_ra) % 360
            if ra_diff > 180:
                ra_diff -= 360

            # Check if star is within RA extent (accounting for dec)
            ra_extent_with_tol = (ra_extent / 2) + bounds_tolerance
            if abs(ra_diff) <= ra_extent_with_tol:
                stars_in_bounds.append(star)

        # Debug logging for very small FOVs
        if fov_size < 0.5 and len(stars_in_bounds) == 0 and len(filtered_stars) > 0:
            logger.debug(
                f"  Warning: FOV {fov_size:.3f}° has 0 stars in bounds from {len(filtered_stars)} filtered stars. "
                f"Grid bounds: RA [{ra_min_grid:.3f}, {ra_max_grid:.3f}], Dec [{dec_min_grid:.3f}, {dec_max_grid:.3f}]"
            )

        # Place stars in grid
        for star in stars_in_bounds:
            # Find nearest grid position, handling RA wraparound
            star_ra = star["ra"]

            # Calculate distances accounting for wraparound
            ra_diffs = np.abs(ra_coords - star_ra)
            ra_diffs_wrap = np.abs(ra_coords - (star_ra + 360))
            ra_diffs_wrap2 = np.abs(ra_coords - (star_ra - 360))
            ra_diffs = np.minimum(ra_diffs, np.minimum(ra_diffs_wrap, ra_diffs_wrap2))

            ra_idx = np.argmin(ra_diffs)
            dec_idx = np.argmin(np.abs(dec_coords - star["dec"]))

            if 0 <= dec_idx < dec_grid_size and 0 <= ra_idx < ra_grid_size:
                star_grid[dec_idx, ra_idx] += 1.0

        # Convolve to get star counts at each position
        if kernel_size > 0:
            # Use 'same' mode to maintain grid size
            convolved = convolve2d(star_grid, kernel, mode="same", boundary="fill", fillvalue=0)

            # Exclude edge regions where kernel doesn't fully overlap
            # The kernel extends kernel_size/2 pixels on each side, so we exclude
            # those border pixels to only use completely valid, filled regions
            border = kernel_size // 2
            if border > 0 and convolved.shape[0] > 2 * border and convolved.shape[1] > 2 * border:
                # Extract the valid region (excluding borders where kernel extends beyond grid)
                valid_region = convolved[border:-border, border:-border]
            else:
                # If grid is too small or border is 0, use entire convolved region
                valid_region = convolved
        else:
            convolved = star_grid
            valid_region = convolved

        # Calculate statistics only over valid region where kernel fully overlaps
        # For sanity check: verify that max_stars makes physical sense for the FOV size
        # The maximum number of stars in a FOV should be roughly proportional to FOV area
        valid_counts = valid_region[valid_region > 0]
        if len(valid_counts) > 0:
            min_stars = int(np.min(valid_counts))
            max_stars_raw = int(np.max(valid_counts))
            mean_stars = float(np.mean(valid_region))
            median_stars = float(np.median(valid_region))

            # Sanity check: max_stars cannot exceed total stars in bounds
            # The convolution sums stars in overlapping regions, but each star
            # should only be counted once per FOV position
            if max_stars_raw > len(stars_in_bounds):
                logger.warning(
                    f"FOV {fov_size:.3f}°, mag {mag_threshold}: max_stars ({max_stars_raw}) "
                    f"exceeds stars in bounds ({len(stars_in_bounds)}). This suggests a bug."
                )
                # Cap at reasonable value
                max_stars = min(max_stars_raw, len(stars_in_bounds))
            else:
                max_stars = max_stars_raw
        else:
            min_stars = max_stars = 0
            mean_stars = median_stars = 0.0

        results[mag_threshold] = {
            "min": min_stars,
            "max": max_stars,
            "mean": mean_stars,
            "median": median_stars,
        }

    return results


def generate_fov_values(min_fov: float, max_fov: float, num_points: int) -> List[float]:
    """
    Generate FOV values with logarithmic spacing (roughly doubling).

    Args:
        min_fov: Minimum FOV in degrees
        max_fov: Maximum FOV in degrees
        num_points: Number of FOV values to generate

    Returns:
        List of FOV values in descending order (max to min)
    """
    if num_points < 2:
        return [max_fov, min_fov]

    # Generate log-spaced values
    log_min = np.log10(min_fov)
    log_max = np.log10(max_fov)
    log_values = np.linspace(log_max, log_min, num_points)

    # Convert back to linear space
    fov_values = 10**log_values

    # Ensure min and max are exactly included
    fov_values[0] = max_fov
    fov_values[-1] = min_fov

    # Round to reasonable precision
    fov_values = [round(f, 2) for f in fov_values]

    # Remove duplicates (in case of rounding)
    fov_values = sorted(set(fov_values), reverse=True)

    return fov_values


def process_single_corridor_time_step(
    corridor_data: Tuple[int, CorridorData],
    max_fov: float,
    min_fov: float,
    grid_spacing_mult: float,
    fov_num_points: int,
    bright_mag: float,
    faint_mag: float,
    mag_step: float,
    catalog_path: str,
    conv_resolution: float,
    fov_values: List[float],
    magnitude_thresholds: List[float],
    output_dir: Path | None,
    generate_debug_plots: bool = True,
) -> List[CoverageStatistics]:
    """
    Process a single Earth-Moon corridor time step.

    Queries stars once in a rectangular region covering the corridor, then scans across
    it with all FOVs.

    Args:
        corridor_data: Tuple of (time_step_num, CorridorData)
        max_fov: Maximum FOV in degrees
        min_fov: Minimum FOV in degrees
        grid_spacing_mult: Grid spacing multiplier
        fov_num_points: Number of FOV values
        bright_mag: Bright magnitude limit
        faint_mag: Faint magnitude limit
        mag_step: Step size for magnitude grid
        catalog_path: Path to SSTR7 catalog
        conv_resolution: Fine grid resolution for convolution
        fov_values: Pre-computed FOV values
        magnitude_thresholds: Pre-computed magnitude thresholds
        output_dir: Output directory (for debug plots)
        generate_debug_plots: Whether to generate debug plots

    Returns:
        List of CoverageStatistics for all positions and FOVs in this corridor
    """
    time_step_num, corridor = corridor_data
    all_statistics = []

    # Query stars once for the entire corridor region
    # Width = Earth-Moon separation, Height = grid_spacing * max_fov
    corridor_width = corridor.earth_moon_separation_deg
    corridor_height = grid_spacing_mult * max_fov

    # Calculate the diagonal of the corridor rectangle to ensure we query enough stars
    # The corridor is at an angle, so we need a square region large enough to encompass it
    corridor_diagonal = np.sqrt(corridor_width**2 + corridor_height**2)
    # Add margin to ensure we get all stars (50% margin)
    query_size = corridor_diagonal * 1.5

    logger.debug(
        f"  Corridor time step {time_step_num}: Querying square region "
        f"{query_size:.3f}° × {query_size:.3f}° (to encompass corridor: "
        f"{corridor_width:.3f}° × {corridor_height:.3f}° at angle) "
        f"centered at ({corridor.ra_center:.2f}°, {corridor.dec_center:.2f}°)"
    )

    stars = query_stars_for_position(
        ra=corridor.ra_center,
        dec=corridor.dec_center,
        fov_size=max_fov,  # Used for fallback, but overridden by x_fov/y_fov
        catalog_path=catalog_path,
        faint_lim=None,
        bright_lim=None,
        x_fov=query_size,
        y_fov=query_size,
    )

    if not stars:
        logger.warning(f"  No stars found for corridor time step {time_step_num}")
        # Create empty statistics for all FOVs and a default position
        default_pos = (round(corridor.ra_center, 2), round(corridor.dec_center, 2))
        for fov in fov_values:
            for mag in magnitude_thresholds:
                all_statistics.append(
                    CoverageStatistics(
                        grid_position=default_pos,
                        fov=fov,
                        magnitude_threshold=mag,
                        min_stars=0,
                        max_stars=0,
                        mean_stars=0.0,
                        median_stars=0.0,
                    )
                )
        return all_statistics

    # Debug plots: show star distribution for the corridor region
    if output_dir is not None and generate_debug_plots:
        try:
            # Create a custom plot showing the corridor region with Earth/Moon positions
            # Use the actual query size (square region)
            plot_star_distribution_debug_corridor(
                stars,
                corridor,
                time_step_num,
                output_dir,
                bright_mag,
                faint_mag,
                mag_step,
                query_x_fov=query_size,
                query_y_fov=query_size,
                corridor_width=corridor_width,
                corridor_height=corridor_height,
            )
        except Exception as e:
            logger.warning(f"  Failed to generate corridor debug plots for time step {time_step_num}: {e}")

    # Generate positions along the corridor ONCE using the smallest FOV
    # This ensures all FOVs are analyzed at the same positions, so coverage percentages are comparable
    smallest_fov = min(fov_values)
    spacing = smallest_fov / 2.0  # Use spacing based on smallest FOV for 50% overlap
    num_positions = max(1, int(np.ceil(corridor_width / spacing)) + 1)

    logger.debug(
        f"  Generating {num_positions} positions along corridor (spacing={spacing:.3f}° based on smallest FOV {smallest_fov:.3f}°)"
    )

    # Generate positions along the corridor (use same positions for all FOVs)
    corridor_positions = []
    ra_earth_rad = np.radians(corridor.ra_earth)
    ra_moon_rad = np.radians(corridor.ra_moon)

    # Unwrap RA to handle 0/360 boundary
    ra_values = np.array([ra_earth_rad, ra_moon_rad])
    ra_unwrapped = np.unwrap(ra_values)

    for i in range(num_positions):
        fraction = i / (num_positions - 1) if num_positions > 1 else 0.5

        # Interpolate RA (in unwrapped space)
        ra_interp_rad = ra_unwrapped[0] + fraction * (ra_unwrapped[1] - ra_unwrapped[0])
        ra_pos = np.degrees(ra_interp_rad) % 360.0

        # Interpolate Dec (straightforward)
        dec_pos = corridor.dec_earth + fraction * (corridor.dec_moon - corridor.dec_earth)

        corridor_positions.append((ra_pos, dec_pos))

    # For each FOV, analyze all positions along the corridor
    for fov in fov_values:
        logger.debug(f"  Analyzing FOV {fov:.3f}° at {num_positions} positions along corridor")

        # Analyze each position along the corridor with this FOV
        for ra_pos, dec_pos in corridor_positions:
            grid_pos = (round(ra_pos, 2), round(dec_pos, 2))

            # Analyze FOV coverage at this position using the queried stars
            fov_stats = analyze_fov_coverage(
                stars=stars,
                center_ra=ra_pos,
                center_dec=dec_pos,
                fov_size=fov,
                magnitude_thresholds=magnitude_thresholds,
                conv_resolution=conv_resolution,
                min_resolution=0.01,
                min_fov=min_fov,
            )

            # Store statistics for each magnitude threshold
            for mag, stats in fov_stats.items():
                all_statistics.append(
                    CoverageStatistics(
                        grid_position=grid_pos,
                        fov=fov,
                        magnitude_threshold=mag,
                        min_stars=stats["min"],
                        max_stars=stats["max"],
                        mean_stars=stats["mean"],
                        median_stars=stats["median"],
                    )
                )

    return all_statistics


def process_single_position(
    position_data: Tuple[int, Tuple[float, float]],
    max_fov: float,
    min_fov: float,
    fov_num_points: int,
    bright_mag: float,
    faint_mag: float,
    mag_step: float,
    catalog_path: str,
    conv_resolution: float,
    fov_values: List[float],
    magnitude_thresholds: List[float],
    output_dir: Path | None,
    generate_debug_plots: bool = True,
) -> Tuple[Tuple[float, float], List[CoverageStatistics]]:
    """
    Process a single sky position and return statistics.

    This is a worker function designed for multiprocessing.

    Args:
        position_data: Tuple of (position_num, (ra, dec))
        max_fov: Maximum FOV in degrees
        min_fov: Minimum FOV in degrees
        fov_num_points: Number of FOV values
        bright_mag: Bright magnitude limit
        faint_mag: Faint magnitude limit
        mag_step: Step size for magnitude grid
        catalog_path: Path to SSTR7 catalog
        conv_resolution: Fine grid resolution for convolution
        fov_values: Pre-computed FOV values
        magnitude_thresholds: Pre-computed magnitude thresholds
        output_dir: Output directory (for debug plots)
        generate_debug_plots: Whether to generate debug plots

    Returns:
        Tuple of (grid_position, list of CoverageStatistics)
    """
    position_num, (ra, dec) = position_data
    grid_pos = (round(ra, 2), round(dec, 2))

    position_statistics = []

    # Query stars for this position
    stars = query_stars_for_position(
        ra=ra,
        dec=dec,
        fov_size=max_fov,
        catalog_path=catalog_path,
        faint_lim=None,
        bright_lim=None,
    )

    # Debug plots: RA vs Dec scatter for brightest and faintest magnitude thresholds only
    if output_dir is not None and generate_debug_plots:
        try:
            query_fov = 2 * max_fov
            plot_star_distribution_debug(
                stars,
                ra,
                dec,
                position_num,
                output_dir,
                bright_mag,
                faint_mag,
                mag_step,
                query_fov=query_fov,
                only_extremes=True,
            )
        except Exception as e:
            logger.warning(f"  Failed to generate debug plots for position {position_num}: {e}")

    if not stars:
        # No stars found, create empty statistics
        for fov in fov_values:
            for mag in magnitude_thresholds:
                position_statistics.append(
                    CoverageStatistics(
                        grid_position=grid_pos,
                        fov=fov,
                        magnitude_threshold=mag,
                        min_stars=0,
                        max_stars=0,
                        mean_stars=0.0,
                        median_stars=0.0,
                    )
                )
        return grid_pos, position_statistics

    # Analyze each FOV
    for fov in fov_values:
        # Run convolution analysis for this FOV
        # Use min_resolution = 0.01° for small FOVs to maintain accuracy
        # Pass min_fov to enable proportional scaling
        fov_stats = analyze_fov_coverage(
            stars=stars,
            center_ra=ra,
            center_dec=dec,
            fov_size=fov,
            magnitude_thresholds=magnitude_thresholds,
            conv_resolution=conv_resolution,
            min_resolution=0.01,
            min_fov=min_fov,
        )

        # Store statistics for each magnitude threshold
        for mag, stats in fov_stats.items():
            position_statistics.append(
                CoverageStatistics(
                    grid_position=grid_pos,
                    fov=fov,
                    magnitude_threshold=mag,
                    min_stars=stats["min"],
                    max_stars=stats["max"],
                    mean_stars=stats["mean"],
                    median_stars=stats["median"],
                )
            )

    return grid_pos, position_statistics


def save_position_statistics(
    position_stats: List[CoverageStatistics],
    grid_pos: Tuple[float, float],
    position_num: int,
    output_dir: Path,
    save_parameters: Dict | None,
) -> None:
    """
    Save statistics for a single position to a per-position file.

    Args:
        position_stats: List of CoverageStatistics for this position
        grid_pos: Grid position tuple (ra, dec)
        position_num: Position number
        output_dir: Output directory
        save_parameters: Parameters dict
    """
    if save_parameters is None:
        return

    ra, dec = grid_pos
    # Create unique filename based on position
    # Format: coverage_statistics_pos_001_RA000.00_Dec-50.00.json
    pos_filename = f"coverage_statistics_pos_{position_num:03d}_RA{ra:06.2f}_Dec{dec:+06.2f}.json"
    pos_dir = output_dir / "per_position_statistics"
    pos_dir.mkdir(parents=True, exist_ok=True)
    pos_path = pos_dir / pos_filename

    pos_output = {
        "position": {"number": position_num, "ra": ra, "dec": dec},
        "parameters": save_parameters,
        "statistics": [asdict(s) for s in position_stats],
    }

    with open(pos_path, "w") as f:
        json.dump(pos_output, f, indent=2)


def load_position_statistics_from_files(
    output_dir: Path, expected_fovs: List[float] | None = None, expected_mags: List[float] | None = None
) -> Tuple[List[CoverageStatistics], Dict, set, Dict[Tuple[float, float], str]]:
    """
    Load all per-position statistics files and determine which positions have been processed.

    Args:
        output_dir: Output directory containing per_position_statistics/ subdirectory
        expected_fovs: List of expected FOV values (for validation)
        expected_mags: List of expected magnitude thresholds (for validation)

    Returns:
        Tuple of (statistics list, parameters dict, set of processed grid positions, dict of incomplete positions)
        The incomplete_positions dict maps grid_pos -> reason string
    """
    pos_dir = output_dir / "per_position_statistics"
    if not pos_dir.exists():
        return [], {}, set(), {}

    statistics = []
    processed_positions = set()
    incomplete_positions = {}  # grid_pos -> reason
    parameters = None

    # Load all position files
    for pos_file in sorted(pos_dir.glob("coverage_statistics_pos_*.json")):
        try:
            with open(pos_file, "r") as f:
                data = json.load(f)

            # Extract parameters from first file (should be same for all)
            if parameters is None:
                parameters = data.get("parameters", {})

            # Extract position info
            pos_info = data.get("position", {})
            ra = pos_info.get("ra")
            dec = pos_info.get("dec")
            grid_pos = (round(ra, 2), round(dec, 2))

            # Reconstruct CoverageStatistics objects
            pos_statistics = []
            statistics_data = data.get("statistics", [])

            # Check if statistics key exists but is empty vs missing
            if "statistics" not in data:
                logger.warning(f"Position file {pos_file.name} missing 'statistics' key")
                incomplete_positions[grid_pos] = "missing 'statistics' key in file"
                pos_file.unlink(missing_ok=True)
                logger.debug(f"Deleted invalid position file: {pos_file.name}")
                continue

            if len(statistics_data) == 0:
                logger.warning(f"Position file {pos_file.name} has empty statistics array")
                incomplete_positions[grid_pos] = "empty statistics array in file"
                pos_file.unlink(missing_ok=True)
                logger.debug(f"Deleted invalid position file: {pos_file.name}")
                continue

            logger.debug(f"Loading position file {pos_file.name}: found {len(statistics_data)} statistics")

            for stat_dict in statistics_data:
                try:
                    if isinstance(stat_dict.get("grid_position"), list):
                        stat_dict["grid_position"] = tuple(stat_dict["grid_position"])
                    pos_statistics.append(CoverageStatistics(**stat_dict))
                except Exception as e:
                    logger.warning(f"Failed to reconstruct statistic from {pos_file.name}: {e}")
                    logger.debug(f"  Problematic stat_dict: {stat_dict}")

            # Validate completeness if expected values provided
            if expected_fovs is not None and expected_mags is not None:
                expected_count = len(expected_fovs) * len(expected_mags)
                if len(pos_statistics) != expected_count:
                    incomplete_positions[grid_pos] = (
                        f"incomplete: {len(pos_statistics)}/{expected_count} statistics "
                        f"(expected {len(expected_fovs)} FOVs × {len(expected_mags)} mags). "
                        f"File: {pos_file.name}"
                    )
                    file_size = pos_file.stat().st_size if pos_file.exists() else 0
                    logger.warning(
                        f"Position {grid_pos} file is incomplete: {len(pos_statistics)}/{expected_count} statistics. "
                        f"File exists: {pos_file.exists()}, file size: {file_size} bytes. Will reprocess."
                    )
                    pos_file.unlink(missing_ok=True)
                    logger.debug(f"Deleted incomplete position file: {pos_file.name}")
                    continue

                # Verify we have all expected FOV/mag combinations
                found_combinations = set((s.fov, s.magnitude_threshold) for s in pos_statistics)
                expected_combinations = set((fov, mag) for fov in expected_fovs for mag in expected_mags)
                missing_combinations = expected_combinations - found_combinations
                if missing_combinations:
                    incomplete_positions[grid_pos] = (
                        f"missing {len(missing_combinations)} FOV/mag combinations: "
                        f"{sorted(list(missing_combinations))[:5]}"
                    )
                    logger.warning(
                        f"Position {grid_pos} file is missing {len(missing_combinations)} FOV/mag combinations. "
                        f"Will reprocess."
                    )
                    pos_file.unlink(missing_ok=True)
                    logger.debug(f"Deleted incomplete position file: {pos_file.name}")
                    continue

                # Validate that we don't have all zeros (which indicates a failed analysis)
                # Check if all min_stars are 0, or if bright magnitudes have 0 stars (physically impossible)
                all_min_stars = [s.min_stars for s in pos_statistics]
                if all(min_stars == 0 for min_stars in all_min_stars):
                    incomplete_positions[grid_pos] = "all statistics have 0 stars (likely failed analysis)"
                    logger.warning(
                        f"Position {grid_pos} has all zeros for min_stars. This indicates a failed analysis. "
                        f"Will reprocess."
                    )
                    pos_file.unlink(missing_ok=True)
                    logger.debug(f"Deleted invalid position file: {pos_file.name}")
                    continue

                # Check if bright magnitudes have 0 stars (should always have stars for bright mags)
                # Brightest magnitude threshold should have stars unless there's an issue
                brightest_mag = min(expected_mags)
                brightest_stats = [s for s in pos_statistics if s.magnitude_threshold == brightest_mag]
                if brightest_stats and all(s.min_stars == 0 for s in brightest_stats):
                    # Check if this is a real issue: even small FOVs should have stars for bright mags
                    # But very small FOVs at high declination might legitimately have no stars
                    # So we'll be conservative and only flag if multiple FOVs have zeros
                    max_fov = max(expected_fovs)
                    max_fov_stats = [s for s in brightest_stats if abs(s.fov - max_fov) < 0.01]
                    if max_fov_stats and all(s.min_stars == 0 for s in max_fov_stats):
                        incomplete_positions[grid_pos] = (
                            f"brightest magnitude ({brightest_mag}) has 0 stars even for largest FOV ({max_fov:.2f}°)"
                        )
                        logger.warning(
                            f"Position {grid_pos} has 0 stars for brightest magnitude ({brightest_mag}) "
                            f"even with largest FOV ({max_fov:.2f}°). This suggests a catalog query issue. "
                            f"Will reprocess."
                        )
                        pos_file.unlink(missing_ok=True)
                        logger.debug(f"Deleted invalid position file: {pos_file.name}")
                        continue

            # All statistics are valid, add them
            statistics.extend(pos_statistics)
            processed_positions.add(grid_pos)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse position file {pos_file} (corrupted JSON): {e}. Will reprocess.")
            pos_file.unlink(missing_ok=True)
            logger.debug(f"Deleted corrupted position file: {pos_file.name}")
            # Try to extract position info even from corrupted file
            try:
                # Read file to get position number from filename
                filename = pos_file.name
                # Extract RA and Dec from filename: coverage_statistics_pos_001_RA178.00_Dec+00.00.json
                if "_RA" in filename and "_Dec" in filename:
                    ra_str = filename.split("_RA")[1].split("_")[0]
                    dec_str = filename.split("_Dec")[1].split(".json")[0]
                    ra = float(ra_str)
                    dec = float(dec_str)
                    grid_pos = (round(ra, 2), round(dec, 2))
                    incomplete_positions[grid_pos] = "corrupted JSON file"
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Failed to load position file {pos_file}: {e}. Will reprocess.")
            pos_file.unlink(missing_ok=True)
            logger.debug(f"Deleted invalid position file: {pos_file.name}")
            # Try to extract position from filename
            try:
                filename = pos_file.name
                if "_RA" in filename and "_Dec" in filename:
                    ra_str = filename.split("_RA")[1].split("_")[0]
                    dec_str = filename.split("_Dec")[1].split(".json")[0]
                    ra = float(ra_str)
                    dec = float(dec_str)
                    grid_pos = (round(ra, 2), round(dec, 2))
                    incomplete_positions[grid_pos] = f"load error: {str(e)}"
            except Exception:
                pass

    return statistics, parameters or {}, processed_positions, incomplete_positions


def run_coverage_analysis(
    grid_positions: List[Tuple[float, float]],
    max_fov: float,
    min_fov: float,
    fov_num_points: int,
    bright_mag: float,
    faint_mag: float,
    mag_step: float,
    catalog_path: str,
    conv_resolution: float | None = None,
    existing_statistics: List[CoverageStatistics] | None = None,
    processed_positions: set | None = None,
    output_dir: Path | None = None,
    save_parameters: Dict | None = None,
    min_threshold: int = 4,
    coverage_threshold: int = 8,
    n_proc: int = 1,
) -> List[CoverageStatistics]:
    """
    Run coverage analysis for all grid positions, FOVs, and magnitude thresholds.

    Args:
        grid_positions: List of (ra, dec) grid positions
        max_fov: Maximum FOV in degrees
        min_fov: Minimum FOV in degrees
        fov_num_points: Number of FOV values
        bright_mag: Bright magnitude limit
        faint_mag: Faint magnitude limit
        mag_step: Step size for magnitude grid
        catalog_path: Path to SSTR7 catalog
        conv_resolution: Fine grid resolution for convolution
        existing_statistics: Existing statistics to append to (for resume)
        processed_positions: Set of already-processed positions (for resume)
        output_dir: Output directory for incremental saving (optional)
        save_parameters: Parameters dict for saving (optional)
        min_threshold: Minimum threshold for plotting (optional)
        coverage_threshold: Threshold for coverage percentage plot (optional)
        n_proc: Number of parallel processes (default: 1, sequential)

    Returns:
        List of CoverageStatistics objects
    """
    # Auto-determine convolution resolution based on smallest FOV
    # Use at least 10 pixels per FOV to ensure accurate representation
    if conv_resolution is None:
        conv_resolution = min_fov / 10.0
        logger.info(f"Auto-set convolution resolution to {conv_resolution:.4f}° (min_fov / 10)")
    else:
        # Warn if resolution is too coarse for smallest FOV
        pixels_per_fov = min_fov / conv_resolution
        if pixels_per_fov < 5:
            logger.warning(
                f"Convolution resolution {conv_resolution:.4f}° may be too coarse for "
                f"min_fov {min_fov:.4f}° (only {pixels_per_fov:.1f} pixels per FOV)"
            )

    # Start with existing statistics if provided
    all_statistics = existing_statistics.copy() if existing_statistics else []
    processed_positions = processed_positions or set()

    # Generate FOV values and magnitude thresholds (needed for both modes)
    fov_values = generate_fov_values(min_fov, max_fov, fov_num_points)
    magnitude_thresholds = np.arange(bright_mag, faint_mag + mag_step, mag_step).tolist()

    total_positions = len(grid_positions)
    total_fovs = len(fov_values)
    total_mags = len(magnitude_thresholds)
    total_combinations = total_positions * total_fovs * total_mags

    # Count remaining positions to process
    remaining_to_process = total_positions - len(processed_positions)
    if remaining_to_process < total_positions:
        logger.info(
            f"Resuming: {len(processed_positions)} positions already processed, "
            f"{remaining_to_process} remaining to process"
        )

    logger.info(f"Starting coverage analysis: {total_positions} positions, {total_fovs} FOVs, {total_mags} magnitudes")
    logger.info(f"Total combinations to process: {total_combinations}")

    start_time = time.time()

    # Filter out already processed positions
    positions_to_process = []
    missing_positions = []
    for pos_idx, (ra, dec) in enumerate(grid_positions):
        grid_pos = (round(ra, 2), round(dec, 2))
        if grid_pos not in processed_positions:
            positions_to_process.append((pos_idx + 1, (ra, dec)))
            missing_positions.append((pos_idx + 1, grid_pos))

    if not positions_to_process:
        logger.info("All positions already processed. Skipping analysis.")
        return all_statistics

    logger.info(f"Processing {len(positions_to_process)} positions with {n_proc} worker process(es)")
    if len(missing_positions) <= 20:
        logger.info("Positions to process:")
        for pos_num, grid_pos in missing_positions:
            logger.info(f"  Position {pos_num}: RA={grid_pos[0]:.2f}°, Dec={grid_pos[1]:.2f}°")
    else:
        logger.info("First 10 positions to process:")
        for pos_num, grid_pos in missing_positions[:10]:
            logger.info(f"  Position {pos_num}: RA={grid_pos[0]:.2f}°, Dec={grid_pos[1]:.2f}°")
        logger.info(f"  ... and {len(missing_positions) - 10} more positions")

    if n_proc > 1:
        # Multiprocessing mode
        logger.info(f"Using multiprocessing with {n_proc} workers")
        from functools import partial

        worker_func = partial(
            process_single_position,
            max_fov=max_fov,
            min_fov=min_fov,
            fov_num_points=fov_num_points,
            bright_mag=bright_mag,
            faint_mag=faint_mag,
            mag_step=mag_step,
            catalog_path=catalog_path,
            conv_resolution=conv_resolution,
            fov_values=fov_values,
            magnitude_thresholds=magnitude_thresholds,
            output_dir=output_dir,
            generate_debug_plots=output_dir is not None,
        )

        # Process positions in parallel
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            future_to_position = {executor.submit(worker_func, pos_data): pos_data for pos_data in positions_to_process}

            pbar = tqdm(
                total=len(positions_to_process),
                desc="Positions",
                unit="pos",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

            # Collect results as they complete
            for future in as_completed(future_to_position):
                pos_data = future_to_position[future]
                position_num, (ra, dec) = pos_data
                try:
                    grid_pos, position_stats = future.result()
                    all_statistics.extend(position_stats)
                    processed_positions.add(grid_pos)

                    # Save per-position statistics file
                    if output_dir is not None:
                        save_position_statistics(
                            position_stats,
                            grid_pos,
                            position_num,
                            output_dir,
                            save_parameters,
                        )

                    pbar.set_description(f"Position {position_num}/{total_positions} (RA={ra:.1f}°, Dec={dec:.1f}°)")
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing position {position_num}: {e}")
                    grid_pos = (round(ra, 2), round(dec, 2))
                    for fov in fov_values:
                        for mag in magnitude_thresholds:
                            all_statistics.append(
                                CoverageStatistics(
                                    grid_position=grid_pos,
                                    fov=fov,
                                    magnitude_threshold=mag,
                                    min_stars=0,
                                    max_stars=0,
                                    mean_stars=0.0,
                                    median_stars=0.0,
                                )
                            )
                    processed_positions.add(grid_pos)

                    # Save per-position statistics even for failed positions
                    if output_dir is not None:
                        failed_stats = [s for s in all_statistics if s.grid_position == grid_pos]
                        save_position_statistics(
                            failed_stats,
                            grid_pos,
                            position_num,
                            output_dir,
                            save_parameters,
                        )

                    pbar.update(1)

            pbar.close()

    else:
        # Sequential mode (original code)
        position_start_time = start_time
        positions_processed_this_run = 0

        pbar = tqdm(
            total=len(positions_to_process),
            desc="Positions",
            unit="pos",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for position_num, (ra, dec) in positions_to_process:
            grid_pos = (round(ra, 2), round(dec, 2))

            positions_processed_this_run += 1
            pbar.set_description(f"Position {position_num}/{total_positions} (RA={ra:.1f}°, Dec={dec:.1f}°)")
            logger.info(f"Processing position {position_num}/{total_positions}: RA={ra:.2f}°, Dec={dec:.2f}°")

            # Query stars for this position (use max_fov to determine query size)
            query_start = time.time()
            logger.debug(f"  Querying catalog for position {position_num}...")
            stars = query_stars_for_position(
                ra=ra,
                dec=dec,
                fov_size=max_fov,
                catalog_path=catalog_path,
                faint_lim=None,
                bright_lim=None,
            )
            query_time = time.time() - query_start
            logger.info(f"  Found {len(stars)} stars in {query_time:.2f}s")

            # Debug plots: RA vs Dec scatter for brightest and faintest magnitude thresholds only
            if output_dir is not None:
                try:
                    query_fov = 2 * max_fov
                    plot_star_distribution_debug(
                        stars,
                        ra,
                        dec,
                        position_num,
                        output_dir,
                        bright_mag,
                        faint_mag,
                        mag_step,
                        query_fov=query_fov,
                        only_extremes=True,
                    )
                except Exception as e:
                    logger.warning(f"  Failed to generate debug plots: {e}")

            if not stars:
                logger.warning(
                    f"  No stars found at position {position_num} (RA={ra:.2f}°, Dec={dec:.2f}°), "
                    f"creating empty statistics. This may indicate a catalog query or filtering issue."
                )
                processed_positions.add(grid_pos)
                for fov in fov_values:
                    for mag in magnitude_thresholds:
                        all_statistics.append(
                            CoverageStatistics(
                                grid_position=(ra, dec),
                                fov=fov,
                                magnitude_threshold=mag,
                                min_stars=0,
                                max_stars=0,
                                mean_stars=0.0,
                                median_stars=0.0,
                            )
                        )
                position_start_time = time.time()
                pbar.update(1)
                continue

            # Analyze each FOV
            fov_pbar = tqdm(
                enumerate(fov_values),
                total=len(fov_values),
                desc=f"  FOVs (pos {position_num})",
                leave=False,
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} FOVs",
            )
            for fov_idx, fov in fov_pbar:
                fov_num = fov_idx + 1
                fov_pbar.set_description(f"  FOV {fov_num}/{total_fovs}: {fov:.2f}°")
                logger.debug(
                    f"  Analyzing FOV {fov_num}/{total_fovs}: {fov:.2f}° (position {position_num}/{total_positions})"
                )

                fov_stats = analyze_fov_coverage(
                    stars=stars,
                    center_ra=ra,
                    center_dec=dec,
                    fov_size=fov,
                    magnitude_thresholds=magnitude_thresholds,
                    conv_resolution=conv_resolution,
                    min_resolution=0.01,
                    min_fov=min_fov,
                )

                for mag, stats in fov_stats.items():
                    all_statistics.append(
                        CoverageStatistics(
                            grid_position=grid_pos,
                            fov=fov,
                            magnitude_threshold=mag,
                            min_stars=stats["min"],
                            max_stars=stats["max"],
                            mean_stars=stats["mean"],
                            median_stars=stats["median"],
                        )
                    )

            position_time = time.time() - position_start_time
            logger.info(f"  Position {position_num} complete in {position_time:.2f}s")

            processed_positions.add(grid_pos)

            # Save per-position statistics file
            if output_dir is not None:
                position_stats = [s for s in all_statistics if s.grid_position == grid_pos]
                save_position_statistics(
                    position_stats,
                    grid_pos,
                    position_num,
                    output_dir,
                    save_parameters,
                )

            # Incremental save: update master aggregated file and plots (sequential mode only)
            if output_dir is not None and save_parameters is not None:
                aggregated_stats = aggregate_statistics(all_statistics)
                stats_output = {
                    "parameters": save_parameters,
                    "per_position_statistics": [asdict(s) for s in all_statistics],
                    "aggregated_statistics": [asdict(s) for s in aggregated_stats],
                }
                json_path = output_dir / "coverage_statistics.json"
                with open(json_path, "w") as f:
                    json.dump(stats_output, f, indent=2)
                logger.debug(f"  Incremental save: Master statistics saved to {json_path}")

                try:
                    plot_coverage_results(
                        json_path, output_dir, min_threshold=min_threshold, coverage_threshold=coverage_threshold
                    )
                    logger.debug("  Incremental save: Plot updated")
                except Exception as e:
                    logger.warning(f"  Failed to generate incremental plot: {e}")

            pbar.update(1)
            position_start_time = time.time()

        # Close progress bar
        pbar.close()

    # Save final results after all positions are processed (for both sequential and parallel modes)
    if output_dir is not None and save_parameters is not None:
        logger.info(f"Saving final results: {len(all_statistics)} statistics from {len(processed_positions)} positions")
        aggregated_stats = aggregate_statistics(all_statistics)
        stats_output = {
            "parameters": save_parameters,
            "per_position_statistics": [asdict(s) for s in all_statistics],
            "aggregated_statistics": [asdict(s) for s in aggregated_stats],
        }
        json_path = output_dir / "coverage_statistics.json"
        with open(json_path, "w") as f:
            json.dump(stats_output, f, indent=2)
        logger.info(f"Final statistics saved to {json_path}")

        # Generate final plot
        try:
            plot_coverage_results(
                json_path, output_dir, min_threshold=min_threshold, coverage_threshold=coverage_threshold
            )
            logger.info("Final plots generated")
        except Exception as e:
            logger.warning(f"Failed to generate final plots: {e}")

    total_time = time.time() - start_time
    logger.info(
        f"Coverage analysis complete: {len(all_statistics)} statistics collected in {total_time / 60:.2f} minutes"
    )
    return all_statistics


def aggregate_statistics(
    statistics: List[CoverageStatistics],
) -> List[AggregatedStatistics]:
    """
    Aggregate statistics across all sky positions for plotting.

    Returns:
        List of AggregatedStatistics objects
    """
    # Group by FOV and magnitude threshold
    grouped = {}
    for stat in statistics:
        key = (stat.fov, stat.magnitude_threshold)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(stat)

    aggregated = []
    for (fov, mag), stats_list in grouped.items():
        min_stars_list = [s.min_stars for s in stats_list]
        max_stars_list = [s.max_stars for s in stats_list]

        global_min = int(min(min_stars_list))
        global_max = int(max(max_stars_list))
        mean_min = float(np.mean(min_stars_list))
        mean_max = float(np.mean(max_stars_list))

        # Calculate percentiles
        percentiles = {
            "p10": float(np.percentile(min_stars_list, 10)),
            "p25": float(np.percentile(min_stars_list, 25)),
            "p50": float(np.percentile(min_stars_list, 50)),
            "p75": float(np.percentile(min_stars_list, 75)),
            "p90": float(np.percentile(min_stars_list, 90)),
        }

        aggregated.append(
            AggregatedStatistics(
                fov=fov,
                magnitude_threshold=mag,
                global_min=global_min,
                global_max=global_max,
                mean_min=mean_min,
                mean_max=mean_max,
                percentiles=percentiles,
            )
        )

    return aggregated


def load_statistics_from_json(json_path: Path) -> Tuple[List[AggregatedStatistics], Dict]:
    """
    Load aggregated statistics from JSON file.

    Args:
        json_path: Path to coverage_statistics.json file

    Returns:
        Tuple of (aggregated_stats list, parameters dict)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Reconstruct AggregatedStatistics objects
    aggregated_stats = []
    for stat_dict in data["aggregated_statistics"]:
        aggregated_stats.append(AggregatedStatistics(**stat_dict))

    return aggregated_stats, data.get("parameters", {})


def load_per_position_statistics_from_json(json_path: Path) -> Tuple[List[CoverageStatistics], Dict, set]:
    """
    Load per-position statistics from JSON file and determine which positions have been processed.

    Args:
        json_path: Path to coverage_statistics.json file

    Returns:
        Tuple of (statistics list, parameters dict, set of processed grid positions)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Reconstruct CoverageStatistics objects
    statistics = []
    for stat_dict in data.get("per_position_statistics", []):
        # Convert grid_position list back to tuple
        if isinstance(stat_dict["grid_position"], list):
            stat_dict["grid_position"] = tuple(stat_dict["grid_position"])
        statistics.append(CoverageStatistics(**stat_dict))

    # Determine which positions have been processed
    processed_positions = set()
    for stat in statistics:
        processed_positions.add(stat.grid_position)

    return statistics, data.get("parameters", {}), processed_positions


def load_corridor_data_from_file(positions_file: Path) -> List[CorridorData] | None:
    """
    Load Earth-Moon corridor data from a JSON file.

    Args:
        positions_file: Path to JSON file with positions

    Returns:
        List of CorridorData objects if corridor format detected, None otherwise
    """
    with open(positions_file, "r") as f:
        data = json.load(f)

    positions_data = data.get("positions", [])
    if not positions_data:
        return None

    # Check if we have the new format with Earth/Moon positions
    first_pos = positions_data[0]
    has_earth_moon = "ra_earth" in first_pos and "ra_moon" in first_pos

    if not has_earth_moon:
        return None

    corridor_data = []
    for pos_data in positions_data:
        corridor_data.append(
            CorridorData(
                ra_center=float(pos_data.get("ra_center", 0.0)) % 360.0,
                dec_center=float(pos_data.get("dec_center", 0.0)),
                ra_earth=float(pos_data["ra_earth"]) % 360.0,
                dec_earth=float(pos_data["dec_earth"]),
                ra_moon=float(pos_data["ra_moon"]) % 360.0,
                dec_moon=float(pos_data["dec_moon"]),
                earth_moon_separation_deg=float(pos_data.get("earth_moon_separation_deg", 0.0)),
                time=pos_data.get("time", ""),
            )
        )

    logger.info(f"Loaded {len(corridor_data)} corridor time steps from {positions_file}")
    return corridor_data


def load_positions_from_file(
    positions_file: Path,
    min_fov: float | None = None,
    max_fov: float | None = None,
    corridor_samples: int | None = None,
) -> List[Tuple[float, float]]:
    """
    Load RA/Dec positions from a JSON file (e.g., from l1_moon_search_zone.py).

    If the file contains Earth and Moon positions, generates positions along the
    Earth-Moon corridor for each time step. The number of samples is calculated
    based on Earth-Moon separation and FOV size to ensure complete coverage.

    Args:
        positions_file: Path to JSON file with positions
        min_fov: Minimum FOV in degrees (used to calculate corridor_samples dynamically)
            (preferred over max_fov to ensure coverage for all FOVs)
        max_fov: Maximum FOV in degrees (fallback if min_fov not provided)
        corridor_samples: Number of positions to sample along Earth-Moon corridor
            (if None and min_fov/max_fov provided, calculated automatically based on FOV;
            if None and no FOV provided, uses default of 5; including Earth and Moon endpoints)

    Returns:
        List of (ra, dec) tuples in degrees
    """
    with open(positions_file, "r") as f:
        data = json.load(f)

    positions = []
    positions_data = data.get("positions", [])

    if not positions_data:
        logger.warning(f"No positions found in {positions_file}")
        return positions

    # Check if we have the new format with Earth/Moon positions
    first_pos = positions_data[0]
    has_earth_moon = "ra_earth" in first_pos and "ra_moon" in first_pos

    if has_earth_moon:
        from astropy.coordinates import SkyCoord  # noqa: E402

        # Determine how to calculate samples
        use_fov_based = max_fov is not None and corridor_samples is None
        if use_fov_based:
            logger.info(
                f"Detected Earth-Moon corridor format. Will calculate samples per time step "
                f"based on Earth-Moon separation and FOV={max_fov:.3f}° (50% overlap for coverage)."
            )
        elif corridor_samples is None:
            corridor_samples = 5  # Default
            logger.info(f"Detected Earth-Moon corridor format. Using default {corridor_samples} samples per time step.")
        else:
            logger.info(f"Detected Earth-Moon corridor format. Using {corridor_samples} samples per time step.")

        total_samples_generated = 0
        for pos_data in positions_data:
            ra_earth = float(pos_data["ra_earth"])
            dec_earth = float(pos_data["dec_earth"])
            ra_moon = float(pos_data["ra_moon"])
            dec_moon = float(pos_data["dec_moon"])

            # Normalize RA to [0, 360)
            ra_earth = ra_earth % 360.0
            ra_moon = ra_moon % 360.0

            # Create SkyCoord objects for Earth and Moon
            earth_coord = SkyCoord(ra=ra_earth * u.deg, dec=dec_earth * u.deg, frame="icrs")
            moon_coord = SkyCoord(ra=ra_moon * u.deg, dec=dec_moon * u.deg, frame="icrs")

            # Calculate Earth-Moon separation
            total_separation = earth_coord.separation(moon_coord).deg

            # Calculate number of samples for this time step
            if use_fov_based:
                # Calculate samples based on FOV to ensure complete coverage
                # With 50% overlap, spacing = FOV/2, so we need ceil(separation / (FOV/2)) + 1 samples
                # This ensures FOVs placed at these positions cover the entire corridor
                spacing = max_fov / 2.0  # 50% overlap
                num_samples = max(1, int(np.ceil(total_separation / spacing)) + 1)
            elif corridor_samples == 1:
                num_samples = 1
            else:
                num_samples = corridor_samples

            # Generate positions along the great circle path from Earth to Moon
            if num_samples == 1:
                # Just use the center point
                if "ra_center" in pos_data:
                    ra_center = float(pos_data["ra_center"]) % 360.0
                    dec_center = float(pos_data["dec_center"])
                    positions.append((ra_center, dec_center))
                    total_samples_generated += 1
                else:
                    # Use midpoint if no center
                    midpoint = total_separation / 2.0
                    mid_coord = earth_coord.directional_offset_by(
                        position_angle=earth_coord.position_angle(moon_coord), distance=midpoint * u.deg
                    )
                    positions.append((mid_coord.ra.deg % 360.0, mid_coord.dec.deg))
                    total_samples_generated += 1
            else:
                # Generate multiple positions along the corridor
                for i in range(num_samples):
                    # Fraction along the path from Earth (0.0) to Moon (1.0)
                    fraction = i / (num_samples - 1) if num_samples > 1 else 0.5

                    # Use spherical interpolation
                    distance = total_separation * fraction
                    position_angle = earth_coord.position_angle(moon_coord)

                    # Get position at this fraction along the path
                    corridor_coord = earth_coord.directional_offset_by(
                        position_angle=position_angle, distance=distance * u.deg
                    )

                    ra = corridor_coord.ra.deg % 360.0
                    dec = corridor_coord.dec.deg
                    positions.append((ra, dec))
                    total_samples_generated += 1

        avg_samples = total_samples_generated / len(positions_data) if positions_data else 0
        logger.info(
            f"Generated {len(positions)} positions along Earth-Moon corridor "
            f"({len(positions_data)} time steps, avg {avg_samples:.1f} samples per step)"
        )
    else:
        # Old format: just use ra/dec or ra_center/dec_center
        logger.info("Using simple position format (no Earth-Moon corridor)")
        for pos_data in positions_data:
            if "ra_center" in pos_data:
                ra = float(pos_data["ra_center"])
                dec = float(pos_data["dec_center"])
            elif "ra" in pos_data:
                ra = float(pos_data["ra"])
                dec = float(pos_data["dec"])
            else:
                logger.warning(f"Position entry missing 'ra' or 'ra_center': {pos_data}")
                continue

            # Normalize RA to [0, 360)
            ra = ra % 360.0
            positions.append((ra, dec))

        logger.info(f"Loaded {len(positions)} positions from {positions_file}")

    if "metadata" in data:
        logger.info(f"  Metadata: {data['metadata'].get('description', 'N/A')}")
        logger.info(f"  Start date: {data['metadata'].get('start_date', 'N/A')}")
        logger.info(f"  Number of days: {data['metadata'].get('num_days', 'N/A')}")

    return positions


def check_parameters_match(params1: Dict, params2: Dict, tolerance: float = 1e-6) -> bool:
    """
    Check if two parameter dictionaries match (for resume capability).

    Args:
        params1: First parameter dictionary
        params2: Second parameter dictionary
        tolerance: Tolerance for floating point comparisons

    Returns:
        True if parameters match, False otherwise
    """
    # Keys that must match exactly
    required_keys = [
        "max_fov",
        "min_fov",
        "bright_mag",
        "faint_mag",
        "mag_step",
        "fov_num_points",
        "grid_spacing",
        "conv_resolution",
        "test_mode",
    ]

    for key in required_keys:
        if key not in params1 or key not in params2:
            return False
        val1 = params1[key]
        val2 = params2[key]

        # Handle floating point comparison
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                return False
        elif val1 != val2:
            return False

    return True


def plot_star_distribution_debug(
    stars: List[Dict],
    center_ra: float,
    center_dec: float,
    position_num: int,
    output_dir: Path,
    bright_mag: float,
    faint_mag: float,
    mag_step: float,
    query_fov: float | None = None,
    only_extremes: bool = False,
):
    """
    Create debug plots showing RA vs Dec scatter of stars for each magnitude bin.

    Args:
        stars: List of star dictionaries with 'ra', 'dec', 'mv' keys (in degrees)
        center_ra: Center RA of the position
        center_dec: Center Dec of the position
        position_num: Position number for labeling
        output_dir: Output directory for plots
        bright_mag: Bright magnitude limit
        faint_mag: Faint magnitude limit
        mag_step: Step size for magnitude bins
        query_fov: Query FOV for setting plot limits (optional)
        only_extremes: If True, only plot brightest and faintest thresholds (default: False)
    """
    if not stars:
        return

    # Create debug directory
    debug_dir = output_dir / "debug_star_distributions"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Generate magnitude bins
    if only_extremes:
        # Only plot brightest and faintest thresholds
        mag_bins = [bright_mag, faint_mag]
    else:
        # Plot all magnitude bins
        mag_bins = np.arange(bright_mag, faint_mag + mag_step, mag_step).tolist()

    # Create one plot per magnitude bin
    for mag_threshold in mag_bins:
        # Filter stars: magnitude <= threshold (cumulative: "this mag and brighter")
        filtered_stars = [s for s in stars if s["mv"] <= mag_threshold and s["mv"] < 32]

        if not filtered_stars:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract RA and Dec
        ra_values = np.array([s["ra"] for s in filtered_stars])
        dec_values = np.array([s["dec"] for s in filtered_stars])
        mag_values = np.array([s["mv"] for s in filtered_stars])

        # Center RA around the center position (handle wraparound)
        # Convert RA to relative coordinates centered at center_ra
        ra_diff = (ra_values - center_ra) % 360.0
        ra_diff = np.where(ra_diff > 180, ra_diff - 360, ra_diff)  # Convert to [-180, 180] range
        ra_plot = center_ra + ra_diff  # Now RA is centered around center_ra

        # Set plot limits based on query FOV if provided, otherwise use data extent
        if query_fov is not None:
            # Use the actual query FOV bounds
            half_fov = query_fov / 2.0
            # For RA, account for declination-dependent scaling
            ra_half_fov = half_fov / max(np.cos(np.radians(center_dec)), 0.01)
            ra_min_plot = center_ra - ra_half_fov
            ra_max_plot = center_ra + ra_half_fov
            dec_min_plot = center_dec - half_fov
            dec_max_plot = center_dec + half_fov
        else:
            # Fallback: determine plot range based on data extent
            ra_min_diff = np.min(ra_diff)
            ra_max_diff = np.max(ra_diff)
            ra_span = ra_max_diff - ra_min_diff
            # Use 10% margin or at least 2 degrees, but cap at reasonable maximum
            ra_margin = min(max(ra_span * 0.1, 2.0), 10.0)
            ra_min_plot = center_ra + ra_min_diff - ra_margin
            ra_max_plot = center_ra + ra_max_diff + ra_margin

            # Also check Dec range for plot limits
            dec_min_val = np.min(dec_values)
            dec_max_val = np.max(dec_values)
            dec_span = dec_max_val - dec_min_val
            # Use 10% margin or at least 1 degree for Dec
            dec_margin = min(max(dec_span * 0.1, 1.0), 5.0)
            dec_min_plot = dec_min_val - dec_margin
            dec_max_plot = dec_max_val + dec_margin

        # Create scatter plot
        scatter = ax.scatter(ra_plot, dec_values, c=mag_values, s=10, alpha=0.6, cmap="viridis_r")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Magnitude", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Mark the center position with a black cross (at center_ra, center_dec)
        ax.plot(center_ra, center_dec, "k+", markersize=20, markeredgewidth=3, label="Grid Center", zorder=10)

        # Set plot limits to show centered region
        ax.set_xlim(ra_min_plot, ra_max_plot)
        ax.set_ylim(dec_min_plot, dec_max_plot)

        ax.set_xlabel("RA (degrees)", fontsize=16)
        ax.set_ylabel("Dec (degrees)", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.set_title(
            f"Position {position_num}: Stars brighter than mag {mag_threshold:.1f}\n"
            f"Center: RA={center_ra:.2f}°, Dec={center_dec:.2f}°\n"
            f"Total stars: {len(filtered_stars)}",
            fontsize=16,
        )
        ax.grid(True, alpha=0.3)
        # Don't use equal aspect for RA/Dec plots as they have different scales
        # ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plot_path = debug_dir / f"position_{position_num:03d}_mag_{mag_threshold:.1f}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    logger.debug(f"  Debug plots saved to {debug_dir}")


def plot_star_distribution_debug_corridor(
    stars: List[Dict],
    corridor: CorridorData,
    time_step_num: int,
    output_dir: Path,
    bright_mag: float,
    faint_mag: float,
    mag_step: float,
    query_x_fov: float,
    query_y_fov: float,
    corridor_width: float | None = None,
    corridor_height: float | None = None,
):
    """
    Create debug plots for corridor mode showing star distribution with Earth/Moon positions.

    Args:
        stars: List of star dictionaries with 'ra', 'dec', 'mv' keys (in degrees)
        corridor: CorridorData object with Earth/Moon positions
        time_step_num: Time step number for labeling
        output_dir: Output directory for plots
        bright_mag: Bright magnitude limit
        faint_mag: Faint magnitude limit
        mag_step: Step size for magnitude bins
        query_x_fov: Query region width (RA direction, degrees)
        query_y_fov: Query region height (Dec direction, degrees)
        corridor_width: Actual corridor width (Earth-Moon separation, degrees, optional)
        corridor_height: Actual corridor height (grid_spacing * max_fov, degrees, optional)
    """
    if not stars:
        return

    # Create debug directory
    debug_dir = output_dir / "debug_star_distributions"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Only plot brightest and faintest thresholds
    mag_bins = [bright_mag, faint_mag]

    # Create one plot per magnitude bin
    for mag_threshold in mag_bins:
        # Filter stars: magnitude <= threshold (cumulative: "this mag and brighter")
        filtered_stars = [s for s in stars if s["mv"] <= mag_threshold and s["mv"] < 32]

        if not filtered_stars:
            continue

        fig, ax = plt.subplots(figsize=(12, 10))

        # Extract RA and Dec
        ra_values = np.array([s["ra"] for s in filtered_stars])
        dec_values = np.array([s["dec"] for s in filtered_stars])
        mag_values = np.array([s["mv"] for s in filtered_stars])

        # Center RA around the center position (handle wraparound)
        center_ra = corridor.ra_center
        center_dec = corridor.dec_center
        ra_diff = (ra_values - center_ra) % 360.0
        ra_diff = np.where(ra_diff > 180, ra_diff - 360, ra_diff)  # Convert to [-180, 180] range
        ra_plot = center_ra + ra_diff  # Now RA is centered around center_ra

        # Calculate Earth and Moon positions for plotting (handle RA wraparound)
        # Use the same unwrapping logic as for stars
        ra_earth_plot = corridor.ra_earth
        ra_moon_plot = corridor.ra_moon
        ra_earth_diff = (ra_earth_plot - center_ra) % 360.0
        ra_earth_diff = ra_earth_diff if ra_earth_diff <= 180 else ra_earth_diff - 360
        ra_earth_plot = center_ra + ra_earth_diff

        ra_moon_diff = (ra_moon_plot - center_ra) % 360.0
        ra_moon_diff = ra_moon_diff if ra_moon_diff <= 180 else ra_moon_diff - 360
        ra_moon_plot = center_ra + ra_moon_diff

        # Set plot limits based on query region, but ensure Earth and Moon are included
        half_x_fov = query_x_fov / 2.0
        half_y_fov = query_y_fov / 2.0
        ra_half_fov = half_x_fov / max(np.cos(np.radians(center_dec)), 0.01)

        # Calculate bounds to include query region, Earth, and Moon
        ra_min_query = center_ra - ra_half_fov
        ra_max_query = center_ra + ra_half_fov
        ra_min_plot = min(ra_min_query, ra_earth_plot, ra_moon_plot) - 0.1
        ra_max_plot = max(ra_max_query, ra_earth_plot, ra_moon_plot) + 0.1

        dec_min_query = center_dec - half_y_fov
        dec_max_query = center_dec + half_y_fov
        dec_min_plot = min(dec_min_query, corridor.dec_earth, corridor.dec_moon) - 0.1
        dec_max_plot = max(dec_max_query, corridor.dec_earth, corridor.dec_moon) + 0.1

        # Create scatter plot
        scatter = ax.scatter(ra_plot, dec_values, c=mag_values, s=10, alpha=0.6, cmap="viridis_r")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Magnitude", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Mark the center position with a black cross
        ax.plot(center_ra, center_dec, "k+", markersize=20, markeredgewidth=3, label="Center", zorder=10)

        # Mark Earth position
        ax.plot(
            ra_earth_plot,
            corridor.dec_earth,
            "go",
            markersize=15,
            markeredgewidth=2,
            markeredgecolor="darkgreen",
            label="Earth",
            zorder=10,
        )

        # Mark Moon position
        ax.plot(
            ra_moon_plot,
            corridor.dec_moon,
            "bo",
            markersize=15,
            markeredgewidth=2,
            markeredgecolor="darkblue",
            label="Moon",
            zorder=10,
        )

        # Draw line connecting Earth and Moon (use unwrapped coordinates)
        ax.plot(
            [ra_earth_plot, ra_moon_plot],
            [corridor.dec_earth, corridor.dec_moon],
            "r--",
            linewidth=2,
            alpha=0.5,
            label="Earth-Moon corridor",
            zorder=9,
        )

        # Set plot limits
        ax.set_xlim(ra_min_plot, ra_max_plot)
        ax.set_ylim(dec_min_plot, dec_max_plot)

        ax.set_xlabel("RA (degrees)", fontsize=16)
        ax.set_ylabel("Dec (degrees)", fontsize=16)
        ax.tick_params(labelsize=14)
        title_lines = [
            f"Corridor Time Step {time_step_num}: Stars brighter than mag {mag_threshold:.1f}",
            f"Center: RA={center_ra:.2f}°, Dec={center_dec:.2f}°",
            f"Query region: {query_x_fov:.2f}° × {query_y_fov:.2f}°",
        ]
        if corridor_width is not None and corridor_height is not None:
            title_lines.append(f"Corridor (at angle): {corridor_width:.3f}° × {corridor_height:.3f}°")
        title_lines.extend(
            [
                f"Earth-Moon separation: {corridor.earth_moon_separation_deg:.3f}°",
                f"Total stars: {len(filtered_stars)}",
            ]
        )
        ax.set_title("\n".join(title_lines), fontsize=16)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = debug_dir / f"corridor_time_step_{time_step_num:03d}_mag_{mag_threshold:.1f}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    logger.debug(f"  Corridor debug plots saved to {debug_dir}")


def plot_coverage_results(
    aggregated_stats_or_json: Union[List[AggregatedStatistics], Path, str],
    output_dir: Path,
    min_threshold: int = 4,
    coverage_threshold: int = 8,
):
    """
    Generate coverage plots from aggregated statistics or JSON file.

    Args:
        aggregated_stats_or_json: Either a list of AggregatedStatistics objects,
            or a Path/str to coverage_statistics.json file
        output_dir: Output directory for plots
        min_threshold: Minimum star count threshold for diagnostic plots
        coverage_threshold: Threshold for coverage percentage plot (default: 8 stars)
    """
    # Load statistics if JSON path provided
    per_position_stats = None
    if isinstance(aggregated_stats_or_json, (str, Path)):
        json_path = Path(aggregated_stats_or_json)
        if not json_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {json_path}")
        logger.info(f"Loading statistics from {json_path}")
        aggregated_stats, _ = load_statistics_from_json(json_path)
        # Also load per-position stats for coverage percentage calculation
        per_position_stats, _, _ = load_per_position_statistics_from_json(json_path)
        # Use JSON file's directory as output if not specified
        if output_dir is None:
            output_dir = json_path.parent
    else:
        aggregated_stats = aggregated_stats_or_json
        # If we have per-position stats, we need to load from JSON
        # For now, we'll require JSON path for coverage percentage plot
        logger.warning("Coverage percentage plot requires JSON file. Skipping this plot.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by FOV
    fov_groups = {}
    for stat in aggregated_stats:
        if stat.fov not in fov_groups:
            fov_groups[stat.fov] = []
        fov_groups[stat.fov].append(stat)

    # Sort FOVs from largest to smallest
    fovs_sorted = sorted(fov_groups.keys(), reverse=True)

    # Main coverage plot: Min/Max lines (original plot)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(fovs_sorted)))

    for fov, color in zip(fovs_sorted, colors, strict=True):
        fov_stats = sorted(fov_groups[fov], key=lambda x: x.magnitude_threshold)
        magnitudes = [s.magnitude_threshold for s in fov_stats]
        min_stars = [s.global_min for s in fov_stats]
        max_stars = [s.global_max for s in fov_stats]

        # Plot min and max lines for this FOV
        ax.plot(
            magnitudes,
            min_stars,
            "-",
            color=color,
            linewidth=2,
            label=f"FOV {fov:.2f}° (min)",
        )
        ax.plot(
            magnitudes,
            max_stars,
            "--",
            color=color,
            linewidth=2,
            label=f"FOV {fov:.2f}° (max)",
        )

    ax.set_xlabel("Limiting Magnitude Threshold", fontsize=16)
    ax.set_ylabel("Number of Stars", fontsize=16)
    ax.set_title("Star Coverage vs Magnitude by FOV (Min/Max)", fontsize=18, fontweight="bold")
    ax.set_yscale("log")
    ax.tick_params(labelsize=14)
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min([s.magnitude_threshold for s in aggregated_stats]))
    ax.set_ylim(bottom=1)  # Set to 1 for log scale instead of 0

    plt.tight_layout()
    main_plot_path = output_dir / "coverage_main.png"
    plt.savefig(main_plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Main coverage plot (min/max) saved to {main_plot_path}")
    plt.close()

    # New plot: Median min_stars with +/- 1 sigma error bars
    if per_position_stats is not None:
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(fovs_sorted)))

        # Group per-position stats by FOV and magnitude
        fov_mag_groups = {}
        for stat in per_position_stats:
            key = (stat.fov, stat.magnitude_threshold)
            if key not in fov_mag_groups:
                fov_mag_groups[key] = []
            fov_mag_groups[key].append(stat)

        for fov, color in zip(fovs_sorted, colors, strict=True):
            # Get all magnitude thresholds for this FOV
            fov_keys = [(f, m) for (f, m) in fov_mag_groups.keys() if f == fov]
            fov_keys_sorted = sorted(fov_keys, key=lambda x: x[1])

            magnitudes = []
            medians = []
            sigmas = []

            for _, mag in fov_keys_sorted:
                stats_list = fov_mag_groups[(fov, mag)]
                min_stars_list = [s.min_stars for s in stats_list]

                if len(min_stars_list) > 0:
                    median_val = float(np.median(min_stars_list))
                    sigma_val = float(np.std(min_stars_list))
                    # Ensure sigma is at least 1 for log scale
                    sigma_val = max(sigma_val, 0.1)

                    magnitudes.append(mag)
                    medians.append(median_val)
                    sigmas.append(sigma_val)

            if magnitudes:
                # Convert to numpy arrays for easier handling
                magnitudes_arr = np.array(magnitudes)
                medians_arr = np.array(medians)
                sigmas_arr = np.array(sigmas)

                # Plot scatter with error bars
                ax.errorbar(
                    magnitudes_arr,
                    medians_arr,
                    yerr=sigmas_arr,
                    fmt="o-",
                    color=color,
                    linewidth=2,
                    markersize=6,
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    label=f"FOV {fov:.2f}°",
                )

        ax.set_xlabel("Limiting Magnitude", fontsize=16)
        ax.set_ylabel("Number of Stars (Median min_stars)", fontsize=16)
        ax.set_title("Star Coverage vs Magnitude by FOV (Median with Error Bars)", fontsize=18, fontweight="bold")
        ax.set_yscale("log")
        ax.tick_params(labelsize=14)
        ax.legend(loc="best", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=min([s.magnitude_threshold for s in aggregated_stats]))
        ax.set_ylim(bottom=1)  # Set to 1 for log scale instead of 0

        plt.tight_layout()
        median_plot_path = output_dir / "coverage_median.png"
        plt.savefig(median_plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Median coverage plot (with error bars) saved to {median_plot_path}")
        plt.close()
    else:
        logger.warning("Per-position statistics not available. Skipping median plot.")

    # Coverage percentage plots: % of sky with min_stars > threshold
    # Generate plots for both threshold 3 and threshold 8 (or the specified threshold)
    if per_position_stats is not None:
        # Get total number of unique grid positions (should be the same for all FOV/mag combinations)
        # Count unique positions from all statistics
        all_positions = set(stat.grid_position for stat in per_position_stats)
        total_grid_positions = len(all_positions)
        logger.info(f"Total unique grid positions for coverage calculation: {total_grid_positions}")

        # Group per-position stats by FOV and magnitude
        fov_mag_groups = {}
        for stat in per_position_stats:
            key = (stat.fov, stat.magnitude_threshold)
            if key not in fov_mag_groups:
                fov_mag_groups[key] = []
            fov_mag_groups[key].append(stat)

        # Group by FOV for plotting
        fov_groups = {}
        for (fov, mag), stats_list in fov_mag_groups.items():
            if fov not in fov_groups:
                fov_groups[fov] = {}
            fov_groups[fov][mag] = stats_list

        # Generate plots for multiple thresholds
        thresholds_to_plot = [3, coverage_threshold] if coverage_threshold != 3 else [coverage_threshold]
        # Remove duplicates while preserving order
        thresholds_to_plot = list(dict.fromkeys(thresholds_to_plot))

        for threshold in thresholds_to_plot:
            _plot_coverage_percentage_single(
                fov_groups=fov_groups,
                aggregated_stats=aggregated_stats,
                output_dir=output_dir,
                threshold=threshold,
                total_grid_positions=total_grid_positions,
            )


def _plot_coverage_percentage_single(
    fov_groups: Dict,
    aggregated_stats: List[AggregatedStatistics],
    output_dir: Path,
    threshold: int,
    total_grid_positions: int,
) -> None:
    """
    Generate a single coverage percentage plot with error bars.

    Args:
        fov_groups: Dictionary mapping FOV -> {magnitude -> list of statistics}
        aggregated_stats: List of aggregated statistics (for x-axis limits)
        output_dir: Output directory for plots
        threshold: Threshold for min_stars (e.g., 3 or 8)
        total_grid_positions: Total number of grid positions
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort FOVs from smallest to largest (so smaller FOVs are plotted first, behind larger ones)
    fovs_sorted = sorted(fov_groups.keys(), reverse=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(fovs_sorted)))

    # Store data for marker plotting (will plot after all lines)
    marker_data = []

    # First pass: plot all lines
    for fov, color in zip(fovs_sorted, colors, strict=True):
        mags = sorted(fov_groups[fov].keys())
        coverage_percentages = []

        for mag in mags:
            stats_list = fov_groups[fov][mag]
            # Count positions where min_stars > threshold
            positions_above_threshold = sum(1 for s in stats_list if s.min_stars > threshold)
            # Use total_grid_positions instead of len(stats_list) to get correct percentage
            # Note: if some positions are missing statistics, len(stats_list) < total_grid_positions
            positions_with_stats = len(stats_list)
            if positions_with_stats < total_grid_positions:
                logger.warning(
                    f"FOV {fov:.2f}°, mag {mag:.1f}: Only {positions_with_stats}/{total_grid_positions} "
                    f"positions have statistics. Coverage may be underestimated."
                )
            if total_grid_positions > 0:
                coverage_pct = 100.0 * positions_above_threshold / total_grid_positions
            else:
                coverage_pct = 0.0

            coverage_percentages.append(coverage_pct)

        # Convert to numpy arrays for easier processing
        mags_arr = np.array(mags)
        coverage_arr = np.array(coverage_percentages)

        # Determine which points should have markers
        # If 3+ consecutive points are at the same level, only mark first and last
        marker_mask = np.ones(len(coverage_arr), dtype=bool)  # Start with all True
        tolerance = 0.01  # Tolerance for "same level" comparison (0.01% coverage)

        if len(coverage_arr) >= 3:
            i = 0
            while i < len(coverage_arr):
                # Find consecutive points at the same level
                start_idx = i
                current_value = coverage_arr[i]

                # Check how many consecutive points have the same value (within tolerance)
                j = i + 1
                while j < len(coverage_arr) and abs(coverage_arr[j] - current_value) < tolerance:
                    j += 1

                # If we have 3+ consecutive points at the same level, remove markers from middle points
                if j - start_idx >= 3:
                    # Keep marker at first point (start_idx) and last point (j-1)
                    # Remove markers from middle points (start_idx+1 to j-2)
                    marker_mask[start_idx + 1 : j - 1] = False

                i = j  # Move to next different value

        # Store marker data for later plotting
        marker_data.append((mags_arr, coverage_arr, marker_mask, color))

        # Plot the line (always plot full line)
        ax.plot(
            mags_arr,
            coverage_arr,
            "-",
            color=color,
            linewidth=2,
            label=f"{fov:.2f}",
        )

    # Second pass: plot all markers (on top of all lines)
    for mags_arr, coverage_arr, marker_mask, color in marker_data:
        if np.any(marker_mask):
            ax.plot(
                mags_arr[marker_mask],
                coverage_arr[marker_mask],
                "o",
                color=color,
                markersize=6,
                markeredgecolor=color,
                markeredgewidth=0,
            )

    ax.set_xlabel("Limiting Magnitude", fontsize=16)
    ax.set_ylabel("Sky Coverage (%)", fontsize=16)
    ax.set_title(f"Percentage of Sky with at least {threshold} stars", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14)
    ax.legend(loc="best", fontsize=14, title="FOV", title_fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-5, 105])
    ax.set_xlim(left=min([s.magnitude_threshold for s in aggregated_stats]))

    plt.tight_layout()
    coverage_plot_path = output_dir / f"coverage_percentage_threshold{threshold}.png"
    plt.savefig(coverage_plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Coverage percentage plot (threshold={threshold}) saved to {coverage_plot_path}")
    plt.close()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Analyze sky coverage for different FOVs and magnitude limits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments (unless --plot-only)
    parser.add_argument("--max-fov", type=float, required=False, help="Maximum FOV in degrees")
    parser.add_argument("--min-fov", type=float, required=False, help="Minimum FOV in degrees")

    # Optional arguments
    parser.add_argument("--faint-mag", type=float, default=19.0, help="Faint magnitude limit")
    parser.add_argument("--bright-mag", type=float, default=12.0, help="Bright magnitude limit")
    parser.add_argument("--mag-step", type=float, default=1.0, help="Step size for magnitude grid")
    parser.add_argument(
        "--fov-num-points",
        type=int,
        default=8,
        help="Number of FOV values to test (logarithmically spaced, roughly doubling)",
    )
    parser.add_argument(
        "--grid-spacing", type=float, default=2.0, help="Coarse grid spacing multiplier (2.0 means 2*max_fov)"
    )
    parser.add_argument(
        "--degrees-off-geo-belt",
        type=float,
        default=None,
        help="Limit grid to celestial equator ± this value in degrees (geo belt = Dec=0). "
        "If None, covers full sky. Example: --degrees-off-geo-belt 10 covers Dec -10° to +10°",
    )
    parser.add_argument(
        "--positions-file",
        type=str,
        default=None,
        help="Path to JSON file with RA/Dec positions (e.g., from l1_moon_search_zone.py). "
        "If provided, uses these positions instead of generating a grid. "
        "Mutually exclusive with --degrees-off-geo-belt and --test-mode.",
    )
    parser.add_argument(
        "--corridor-samples",
        type=int,
        default=None,
        help="Number of positions to sample along Earth-Moon corridor per time step "
        "(only used with --positions-file when Earth/Moon positions are present). "
        "If None, automatically calculates based on FOV size to ensure complete coverage "
        "(recommended). Default: None (auto-calculate based on max-fov).",
    )
    parser.add_argument(
        "--conv-resolution",
        type=float,
        default=None,
        help="Fine grid resolution for convolution in degrees (auto-determined from min_fov if not specified)",
    )
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode (3x3 grid around test location)")
    parser.add_argument(
        "--output-dir", type=str, default="coverage_results", help="Output directory for results and plots"
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help="Path to SSTR7 catalog (from config if not provided)",
    )
    parser.add_argument(
        "--min-threshold", type=int, default=4, help="Minimum star count threshold for diagnostic plots"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        default=8,
        help="Threshold for coverage percentage plot (min_stars > this value)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing coverage_statistics.json file (skip analysis)",
    )
    parser.add_argument(
        "--n-proc",
        type=int,
        default=1,
        help="Number of parallel processes to use (default: 1, sequential). "
        "Use CPU count for maximum speed: --n-proc $(nproc)",
    )
    parser.add_argument(
        "-c",
        "--config",
        help=f"Config file, defaults to {LOCAL_APP_CONFIG_OVERRIDE}",
        type=str,
        default=LOCAL_APP_CONFIG_OVERRIDE,
    )

    args = parser.parse_args()

    # Validate required arguments (unless plot-only mode)
    if not args.plot_only:
        if args.max_fov is None or args.min_fov is None:
            parser.error("--max-fov and --min-fov are required unless --plot-only is specified")

    # Validate that positions-file is mutually exclusive with degrees-off-geo-belt and test-mode
    if args.positions_file:
        if args.degrees_off_geo_belt is not None:
            parser.error("--positions-file cannot be used with --degrees-off-geo-belt")
        if args.test_mode:
            parser.error("--positions-file cannot be used with --test-mode")

    # Initialize configuration
    config = initialize_config(Path(args.config))
    set_log_level(config.logging.level)

    # Ensure matplotlib loggers stay at WARNING level after logging configuration
    # This must be after set_log_level() in case it resets logger levels
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.colorbar").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if plot-only mode
    json_path = output_dir / "coverage_statistics.json"

    if args.plot_only:
        # Plot-only mode: recreate master file from per-position files and generate plots
        pos_dir = output_dir / "per_position_statistics"

        # Check for per-position files first
        if pos_dir.exists() and any(pos_dir.glob("coverage_statistics_pos_*.json")):
            logger.info(f"Plot-only mode: Loading per-position statistics from {pos_dir}")
            try:
                statistics, parameters, _, incomplete_pos = load_position_statistics_from_files(output_dir)

                if not statistics:
                    logger.error("No valid statistics found in per-position files")
                    return 1

                logger.info(f"Found {len(statistics)} statistics from per-position files")

                # Aggregate statistics
                aggregated_stats = aggregate_statistics(statistics)
                logger.info(f"Aggregated into {len(aggregated_stats)} statistics")

                # Recreate master file
                stats_output = {
                    "parameters": parameters,
                    "per_position_statistics": [asdict(s) for s in statistics],
                    "aggregated_statistics": [asdict(s) for s in aggregated_stats],
                }

                with open(json_path, "w") as f:
                    json.dump(stats_output, f, indent=2)
                logger.info(f"Recreated master statistics file: {json_path}")

            except Exception as e:
                logger.error(f"Failed to load per-position statistics: {e}")
                return 1

        # Fall back to legacy master file if per-position files don't exist
        elif json_path.exists():
            logger.info(f"Plot-only mode: Loading statistics from legacy master file {json_path}")
        else:
            logger.error(f"Statistics file not found: {json_path}")
            logger.error("No per-position statistics files found in {pos_dir}")
            logger.error("Run without --plot-only to generate statistics first")
            return 1

        # Generate plots from master file
        logger.info(f"Generating plots from {json_path}")
        plot_coverage_results(
            json_path, output_dir, min_threshold=args.min_threshold, coverage_threshold=args.coverage_threshold
        )
        logger.info(f"Plots generated in {output_dir}")
        return 0

    # Full analysis mode
    # Get catalog path
    catalog_path = args.catalog_path
    if catalog_path is None:
        catalog_path = config.star_catalog.path
        if not catalog_path:
            logger.error("Catalog path not provided and not found in config")
            return 1

    logger.info(f"Using catalog path: {catalog_path}")

    # Prepare parameters dict
    current_parameters = {
        "max_fov": args.max_fov,
        "min_fov": args.min_fov,
        "bright_mag": args.bright_mag,
        "faint_mag": args.faint_mag,
        "mag_step": args.mag_step,
        "fov_num_points": args.fov_num_points,
        "grid_spacing": args.grid_spacing,
        "conv_resolution": args.conv_resolution,
        "test_mode": args.test_mode,
        "degrees_off_geo_belt": args.degrees_off_geo_belt,
        "positions_file": str(args.positions_file) if args.positions_file else None,
    }

    # Generate sky grid or load positions from file
    corridor_data_list = None
    grid_positions = None

    if args.positions_file:
        logger.info(f"Loading positions from file: {args.positions_file}")
        positions_file_path = Path(args.positions_file)
        if not positions_file_path.exists():
            logger.error(f"Positions file not found: {positions_file_path}")
            return 1

        # Check if this is a corridor file (Earth-Moon format)
        corridor_data_list = load_corridor_data_from_file(positions_file_path)
        if corridor_data_list is not None:
            logger.info("Detected Earth-Moon corridor format. Will use corridor processing mode.")
            # In corridor mode, we don't pre-generate grid positions
            # They are generated dynamically per FOV during processing
        else:
            # Regular positions file
            grid_positions = load_positions_from_file(
                positions_file_path, max_fov=args.max_fov, corridor_samples=args.corridor_samples
            )
    else:
        logger.info("Generating sky grid...")
        grid_positions = generate_sky_grid(
            max_fov=args.max_fov,
            grid_spacing_mult=args.grid_spacing,
            test_mode=args.test_mode,
            degrees_off_geo_belt=args.degrees_off_geo_belt,
        )
        logger.info(f"Generated {len(grid_positions)} grid positions")

    # Add num_positions to parameters
    if corridor_data_list is not None:
        current_parameters["num_positions"] = len(corridor_data_list)  # Number of time steps
        current_parameters["corridor_mode"] = True
    else:
        current_parameters["num_positions"] = len(grid_positions)
        current_parameters["corridor_mode"] = False

    # Generate expected FOVs and magnitude thresholds for validation
    expected_fovs = generate_fov_values(args.min_fov, args.max_fov, args.fov_num_points)
    expected_mags = np.arange(args.bright_mag, args.faint_mag + args.mag_step, args.mag_step).tolist()

    # Check for existing statistics file and resume if possible
    # Try per-position files first (more reliable for resume)
    existing_statistics = None
    processed_positions = set()
    incomplete_positions = {}

    pos_dir = output_dir / "per_position_statistics"
    if pos_dir.exists() and any(pos_dir.glob("coverage_statistics_pos_*.json")):
        logger.info(f"Found per-position statistics directory: {pos_dir}")
        try:
            existing_stats, existing_params, processed_pos, incomplete_pos = load_position_statistics_from_files(
                output_dir, expected_fovs=expected_fovs, expected_mags=expected_mags
            )
            if check_parameters_match(current_parameters, existing_params):
                logger.info("Parameters match! Resuming from per-position statistics files...")
                existing_statistics = existing_stats
                processed_positions = processed_pos
                incomplete_positions = incomplete_pos
                logger.info(
                    f"Found {len(existing_statistics)} existing statistics from "
                    f"{len(processed_positions)} complete positions"
                )
                if incomplete_positions:
                    logger.warning(
                        f"Found {len(incomplete_positions)} incomplete/corrupted position files "
                        f"that will be reprocessed:"
                    )
                    for grid_pos, reason in sorted(incomplete_positions.items())[:10]:  # Show first 10
                        logger.warning(f"  Position {grid_pos}: {reason}")
                    if len(incomplete_positions) > 10:
                        logger.warning(f"  ... and {len(incomplete_positions) - 10} more")
            else:
                logger.warning("Parameters don't match existing per-position files. Starting fresh analysis.")
                logger.warning(f"Existing: {existing_params}")
                logger.warning(f"Current: {current_parameters}")
        except Exception as e:
            logger.warning(f"Failed to load per-position statistics: {e}. Trying legacy master file...")

    # Fall back to legacy master JSON file if per-position files don't exist or failed
    if existing_statistics is None and json_path.exists():
        logger.info(f"Found existing master statistics file: {json_path}")
        try:
            existing_stats, existing_params, processed_pos = load_per_position_statistics_from_json(json_path)
            if check_parameters_match(current_parameters, existing_params):
                logger.info("Parameters match! Resuming from legacy master statistics file...")
                existing_statistics = existing_stats
                processed_positions = processed_pos
                logger.info(
                    f"Found {len(existing_statistics)} existing statistics from {len(processed_positions)} positions"
                )
            else:
                logger.warning("Parameters don't match existing file. Starting fresh analysis.")
                logger.warning(f"Existing: {existing_params}")
                logger.warning(f"Current: {current_parameters}")
        except Exception as e:
            logger.warning(f"Failed to load existing statistics: {e}. Starting fresh analysis.")

    # Run coverage analysis
    if corridor_data_list is not None:
        # Corridor mode: process each time step
        logger.info(f"Processing {len(corridor_data_list)} corridor time steps")
        from functools import partial

        # Generate FOV values and magnitude thresholds
        fov_values = generate_fov_values(args.min_fov, args.max_fov, args.fov_num_points)
        magnitude_thresholds = np.arange(args.bright_mag, args.faint_mag + args.mag_step, args.mag_step).tolist()

        # Auto-determine convolution resolution
        if args.conv_resolution is None:
            conv_resolution = args.min_fov / 10.0
            logger.info(f"Auto-set convolution resolution to {conv_resolution:.4f}° (min_fov / 10)")
        else:
            conv_resolution = args.conv_resolution

        # Process corridors
        all_statistics = existing_statistics.copy() if existing_statistics else []
        corridor_data_with_nums = [(i + 1, corridor) for i, corridor in enumerate(corridor_data_list)]

        if args.n_proc > 1:
            # Multiprocessing mode
            from concurrent.futures import ProcessPoolExecutor, as_completed

            worker_func = partial(
                process_single_corridor_time_step,
                max_fov=args.max_fov,
                min_fov=args.min_fov,
                grid_spacing_mult=args.grid_spacing,
                fov_num_points=args.fov_num_points,
                bright_mag=args.bright_mag,
                faint_mag=args.faint_mag,
                mag_step=args.mag_step,
                catalog_path=catalog_path,
                conv_resolution=conv_resolution,
                fov_values=fov_values,
                magnitude_thresholds=magnitude_thresholds,
                output_dir=output_dir,
                generate_debug_plots=output_dir is not None,
            )

            with ProcessPoolExecutor(max_workers=args.n_proc) as executor:
                future_to_corridor = {
                    executor.submit(worker_func, corridor_data): corridor_data
                    for corridor_data in corridor_data_with_nums
                }

                pbar = tqdm(
                    total=len(corridor_data_with_nums),
                    desc="Corridor time steps",
                    unit="step",
                    ncols=100,
                )

                for future in as_completed(future_to_corridor):
                    corridor_data = future_to_corridor[future]
                    time_step_num, corridor = corridor_data
                    try:
                        corridor_stats = future.result()
                        all_statistics.extend(corridor_stats)

                        # Save per-position statistics for this corridor time step
                        if output_dir is not None and current_parameters is not None:
                            # Group statistics by position for this time step
                            positions_in_time_step = {}
                            for stat in corridor_stats:
                                grid_pos = stat.grid_position
                                if grid_pos not in positions_in_time_step:
                                    positions_in_time_step[grid_pos] = []
                                positions_in_time_step[grid_pos].append(stat)

                            # Save each position as a separate file
                            for pos_idx, (grid_pos, pos_stats) in enumerate(positions_in_time_step.items()):
                                # Create a unique position number: time_step_num * 1000 + position_index
                                # This ensures unique numbers across time steps
                                position_num = time_step_num * 1000 + pos_idx + 1
                                save_position_statistics(
                                    pos_stats,
                                    grid_pos,
                                    position_num,
                                    output_dir,
                                    current_parameters,
                                )

                        pbar.set_description(
                            f"Time step {time_step_num}/{len(corridor_data_list)} "
                            f"(RA={corridor.ra_center:.1f}°, Dec={corridor.dec_center:.1f}°)"
                        )
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing corridor time step {time_step_num}: {e}")
                        pbar.update(1)

                pbar.close()
        else:
            # Sequential mode
            pbar = tqdm(
                total=len(corridor_data_with_nums),
                desc="Corridor time steps",
                unit="step",
                ncols=100,
            )

            for time_step_num, corridor in corridor_data_with_nums:
                pbar.set_description(
                    f"Time step {time_step_num}/{len(corridor_data_list)} "
                    f"(RA={corridor.ra_center:.1f}°, Dec={corridor.dec_center:.1f}°)"
                )
                corridor_stats = process_single_corridor_time_step(
                    (time_step_num, corridor),
                    max_fov=args.max_fov,
                    min_fov=args.min_fov,
                    grid_spacing_mult=args.grid_spacing,
                    fov_num_points=args.fov_num_points,
                    bright_mag=args.bright_mag,
                    faint_mag=args.faint_mag,
                    mag_step=args.mag_step,
                    catalog_path=catalog_path,
                    conv_resolution=conv_resolution,
                    fov_values=fov_values,
                    magnitude_thresholds=magnitude_thresholds,
                    output_dir=output_dir,
                    generate_debug_plots=output_dir is not None,
                )
                all_statistics.extend(corridor_stats)

                # Save per-position statistics for this corridor time step
                if output_dir is not None and current_parameters is not None:
                    # Group statistics by position for this time step
                    positions_in_time_step = {}
                    for stat in corridor_stats:
                        grid_pos = stat.grid_position
                        if grid_pos not in positions_in_time_step:
                            positions_in_time_step[grid_pos] = []
                        positions_in_time_step[grid_pos].append(stat)

                    # Save each position as a separate file
                    for pos_idx, (grid_pos, pos_stats) in enumerate(positions_in_time_step.items()):
                        # Create a unique position number: time_step_num * 1000 + position_index
                        # This ensures unique numbers across time steps
                        position_num = time_step_num * 1000 + pos_idx + 1
                        save_position_statistics(
                            pos_stats,
                            grid_pos,
                            position_num,
                            output_dir,
                            current_parameters,
                        )

                    # Incremental save: update master aggregated file and plots (sequential mode only)
                    aggregated_stats = aggregate_statistics(all_statistics)
                    stats_output = {
                        "parameters": current_parameters,
                        "per_position_statistics": [asdict(s) for s in all_statistics],
                        "aggregated_statistics": [asdict(s) for s in aggregated_stats],
                    }
                    json_path = output_dir / "coverage_statistics.json"
                    with open(json_path, "w") as f:
                        json.dump(stats_output, f, indent=2)
                    logger.debug(f"  Incremental save: Master statistics saved to {json_path}")

                    try:
                        plot_coverage_results(
                            json_path,
                            output_dir,
                            min_threshold=args.min_threshold,
                            coverage_threshold=args.coverage_threshold,
                        )
                        logger.debug("  Incremental save: Plot updated")
                    except Exception as e:
                        logger.warning(f"  Failed to generate incremental plot: {e}")

                pbar.update(1)

            pbar.close()

        statistics = all_statistics
    else:
        # Regular grid mode
        statistics = run_coverage_analysis(
            grid_positions=grid_positions,
            max_fov=args.max_fov,
            min_fov=args.min_fov,
            fov_num_points=args.fov_num_points,
            bright_mag=args.bright_mag,
            faint_mag=args.faint_mag,
            mag_step=args.mag_step,
            catalog_path=catalog_path,
            conv_resolution=args.conv_resolution,
            existing_statistics=existing_statistics,
            processed_positions=processed_positions,
            output_dir=output_dir,
            save_parameters=current_parameters,
            min_threshold=args.min_threshold,
            coverage_threshold=args.coverage_threshold,
            n_proc=args.n_proc,
        )

    # Final aggregation and save (already done incrementally, but do final one)
    logger.info(f"Final aggregation from {len(statistics)} individual statistics...")
    agg_start = time.time()
    aggregated_stats = aggregate_statistics(statistics)
    agg_time = time.time() - agg_start
    logger.info(f"Aggregation complete: {len(aggregated_stats)} aggregated statistics in {agg_time:.2f}s")

    # Final save (in case incremental saves missed anything)
    stats_output = {
        "parameters": current_parameters,
        "per_position_statistics": [asdict(s) for s in statistics],
        "aggregated_statistics": [asdict(s) for s in aggregated_stats],
    }

    with open(json_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    logger.info(f"Final statistics saved to {json_path}")

    # Generate final plots
    logger.info("Generating final plots...")
    plot_start = time.time()
    plot_coverage_results(
        json_path, output_dir, min_threshold=args.min_threshold, coverage_threshold=args.coverage_threshold
    )
    plot_time = time.time() - plot_start
    logger.info(f"Plot generation complete in {plot_time:.2f}s")

    logger.info(f"Coverage analysis complete! Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
