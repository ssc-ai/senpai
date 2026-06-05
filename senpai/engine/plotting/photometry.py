"""
Photometry plotting utilities for visualizing photometric results.

This module provides plotting functions for:
- Magnitude vs SNR scatter plots with completeness analysis
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from senpai.core.config import get_config
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.photometry.utils import (
    SimplePhotometryResult,
    SimplePhotometrySummary,
    _find_common_magnitude_system,
)

logger = logging.getLogger(__name__)


def plot_magnitude_vs_snr(
    results: List[SimplePhotometryResult],
    summary: SimplePhotometrySummary,
    output_file: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Create a magnitude vs SNR scatter plot.

    Parameters
    ----------
    results : List[SimplePhotometryResult]
        Photometry results for individual stars
    summary : SimplePhotometrySummary
        Summary statistics
    output_file : Path, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    # Use same magnitude selection logic as limiting magnitude calculation
    # to ensure consistency between plot and calculation AND to avoid mixing
    # different magnitude systems (e.g., Johnson_V vs Gaia_G)
    cfg = get_config()
    preferred_filters = (
        getattr(cfg.photometry, "preferred_filters", None)
        if hasattr(cfg, "photometry")
        else None
    )

    # Find a common magnitude system for all stars
    magnitudes = []
    snrs = []
    quality_flags = []

    try:
        result_stars = [r.star for r in results]
        common_filter = _find_common_magnitude_system(result_stars, preferred_filters)

        n_with_mag = sum(
            1
            for s in result_stars
            if hasattr(s, "magnitude") and s.magnitude is not None
        )
        n_with_mag_dict = sum(
            1
            for s in result_stars
            if hasattr(s, "magnitudes")
            and s.magnitudes is not None
            and len(s.magnitudes) > 0
        )
        logger.debug(
            f"Plotting: found common_filter={common_filter}, "
            f"n_stars={len(result_stars)}, "
            f"stars_with_magnitude={n_with_mag}, "
            f"stars_with_magnitudes_dict={n_with_mag_dict}"
        )

        for result in results:
            # Use consistent magnitude system for all stars
            mag = None
            if common_filter == "primary":
                # Use primary magnitude
                mag = (
                    result.star.magnitude if hasattr(result.star, "magnitude") else None
                )
            elif common_filter is not None:
                # Use the common filter
                if (
                    hasattr(result.star, "magnitudes")
                    and result.star.magnitudes is not None
                    and len(result.star.magnitudes) > 0
                ):
                    mag = result.star.magnitudes.get(common_filter)
            else:
                # Fallback: use primary magnitude if available
                mag = (
                    result.star.magnitude if hasattr(result.star, "magnitude") else None
                )

            if mag is not None:
                magnitudes.append(mag)
                snrs.append(result.snr)
                quality_flags.append(result.quality_flag)
    except Exception as e:
        logger.warning(
            f"Error finding common magnitude system, falling back to primary magnitude: {e}"
        )
        # Fallback: use primary magnitude
        for result in results:
            if hasattr(result.star, "magnitude") and result.star.magnitude is not None:
                magnitudes.append(result.star.magnitude)
                snrs.append(result.snr)
                quality_flags.append(result.quality_flag)

    if not magnitudes:
        ax.text(
            0.5,
            0.5,
            "No magnitude data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Magnitude vs SNR")
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Magnitude vs SNR plot saved to {output_file}")
        return fig

    magnitudes = np.array(magnitudes)
    snrs = np.array(snrs)
    quality_flags = np.array(quality_flags)

    # Create scatter plot with quality color coding
    quality_mask = quality_flags
    poor_mask = ~quality_flags

    # Plot quality measurements
    if np.any(quality_mask):
        ax.scatter(
            magnitudes[quality_mask],
            snrs[quality_mask],
            c="blue",
            alpha=0.6,
            s=30,
            label=f"Quality ({np.sum(quality_mask)})",
        )

    # Plot poor quality measurements
    if np.any(poor_mask):
        ax.scatter(
            magnitudes[poor_mask],
            snrs[poor_mask],
            c="red",
            alpha=0.6,
            s=30,
            label=f"Poor Quality ({np.sum(poor_mask)})",
        )

    # Add limiting SNR line (SNR threshold used for limiting magnitude)
    limiting_snr = summary.limiting_snr
    if limiting_snr is None:
        # Fallback to config if not stored in summary
        cfg = get_config()
        limiting_snr = getattr(cfg.photometry, "limiting_snr", 3.0)

    if limiting_snr > 0:
        ax.axhline(
            y=limiting_snr,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Limiting SNR: {limiting_snr:.1f}",
        )

    # Add limiting magnitude line for 50% completeness
    if summary.limiting_magnitude_50 is not None and summary.limiting_magnitude_50 > 0:
        ax.axvline(
            x=summary.limiting_magnitude_50,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Limiting Mag [50%]: {summary.limiting_magnitude_50:.1f}",
        )

    # Add limiting magnitude line for 90% completeness
    if summary.limiting_magnitude_90 is not None and summary.limiting_magnitude_90 > 0:
        ax.axvline(
            x=summary.limiting_magnitude_90,
            color="black",
            linestyle=":",
            alpha=0.7,
            label=f"Limiting Mag [90%]: {summary.limiting_magnitude_90:.1f}",
        )
    # Plot completeness vs magnitude on a secondary y-axis
    cfg = get_config()
    limiting_snr = getattr(cfg.photometry, "limiting_snr", 3.0)

    ax2 = ax.twinx()
    if len(magnitudes) > 0:
        # Use all catalog stars (quality + poor) for completeness, since we want
        # the fraction of *all* stars in each bin that are above the SNR
        # threshold.
        mags_all = magnitudes
        snrs_all = snrs
        bin_width = 0.25
        min_mag = float(np.floor(np.min(mags_all)))
        max_mag_actual = float(np.max(mags_all))  # Actual max magnitude in data
        max_mag = float(
            np.ceil(max_mag_actual)
        )  # For binning, use ceil to include last bin
        bins = np.arange(min_mag, max_mag + bin_width, bin_width)
        if len(bins) > 1 and len(mags_all) > 0:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            completeness = []
            bin_centers_filtered = []
            for i in range(len(bins) - 1):
                in_bin = (mags_all >= bins[i]) & (mags_all < bins[i + 1])
                n_bin = int(np.sum(in_bin))
                if n_bin == 0:
                    # Skip empty bins - don't plot them
                    continue
                else:
                    completeness.append(
                        float(np.mean(snrs_all[in_bin] >= limiting_snr))
                    )
                    bin_centers_filtered.append(bin_centers[i])

            # Only plot if we have completeness data
            if completeness:
                ax2.plot(
                    bin_centers_filtered,
                    completeness,
                    "ko-",
                    label="Completeness",
                    alpha=0.8,
                )
                ax2.set_ylim(0.0, 1.05)
                ax2.set_ylabel("Completeness (fraction)")

                # Warn if limiting magnitude is extrapolated beyond actual data range
                if summary.limiting_magnitude > max_mag_actual:
                    logger.warning(
                        f"Limiting magnitude ({summary.limiting_magnitude:.2f}) "
                        f"is extrapolated beyond actual data range (max sampled mag={max_mag_actual:.2f}). "
                        f"Completeness data only extends to {max_mag_actual:.2f} mag."
                    )

    # Format x-axis label with filter name
    if common_filter and common_filter != "primary":
        xlabel = f"Apparent Magnitude [{common_filter}]"
    else:
        xlabel = "Apparent Magnitude"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Signal-to-Noise Ratio")
    ax.set_title("Magnitude vs SNR")

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax.grid(True, alpha=0.3)

    # Set log scale for SNR if range is large
    if np.max(snrs) / np.min(snrs[snrs > 0]) > 10:
        ax.set_yscale("log")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Magnitude vs SNR plot saved to {output_file}")

    return fig


def plot_spatial_background_distribution(
    results: List[SimplePhotometryResult],
    image_shape: Tuple[int, int],
    output_file: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Create a spatial map of background level distribution.

    Parameters
    ----------
    results : List[SimplePhotometryResult]
        Photometry results for individual stars
    image_shape : tuple
        Shape of the image (height, width)
    output_file : Path, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract positions and background levels
    x_positions = []
    y_positions = []
    background_levels = []

    for result in results:
        if result.star.x is not None and result.star.y is not None:
            x_positions.append(result.star.x)
            y_positions.append(result.star.y)
            background_levels.append(result.background_level)

    if not x_positions:
        ax.text(
            0.5,
            0.5,
            "No position data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return fig

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    background_levels = np.array(background_levels)

    # Plot background level distribution
    scatter = ax.scatter(
        x_positions, y_positions, c=background_levels, cmap="viridis", s=50, alpha=0.7
    )
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Background Level Distribution")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Image coordinates: (0,0) at top-left
    plt.colorbar(
        scatter, ax=ax, label="Background Level (ADU/pixel)", shrink=0.6, aspect=20
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Background distribution plot saved to {output_file}")

    return fig


def plot_spatial_background_quality(
    results: List[SimplePhotometryResult],
    image_shape: Tuple[int, int],
    output_file: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Create a spatial map of background levels with quality coding.

    Parameters
    ----------
    results : List[SimplePhotometryResult]
        Photometry results for individual stars
    image_shape : tuple
        Shape of the image (height, width)
    output_file : Path, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract positions and background levels
    x_positions = []
    y_positions = []
    background_levels = []
    quality_flags = []

    for result in results:
        if result.star.x is not None and result.star.y is not None:
            x_positions.append(result.star.x)
            y_positions.append(result.star.y)
            background_levels.append(result.background_level)
            quality_flags.append(result.quality_flag)

    if not x_positions:
        ax.text(
            0.5,
            0.5,
            "No position data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return fig

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    background_levels = np.array(background_levels)
    quality_flags = np.array(quality_flags)

    # Get color scale
    vmin = np.min(background_levels)
    vmax = np.max(background_levels)

    # Plot quality and poor quality points separately with different edge colors
    quality_mask = quality_flags
    poor_mask = ~quality_flags

    # Plot quality points with black edges
    scatter_quality = None
    if np.any(quality_mask):
        scatter_quality = ax.scatter(
            x_positions[quality_mask],
            y_positions[quality_mask],
            c=background_levels[quality_mask],
            cmap="viridis",
            s=50,
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidths=1.0,
        )

    # Plot poor quality points with red edges
    scatter_poor = None
    if np.any(poor_mask):
        scatter_poor = ax.scatter(
            x_positions[poor_mask],
            y_positions[poor_mask],
            c=background_levels[poor_mask],
            cmap="viridis",
            s=50,
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
            edgecolors="red",
            linewidths=2.0,
        )

    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Background Level (Red edges = Poor Quality)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Image coordinates: (0,0) at top-left

    # Use the first available scatter plot for colorbar
    scatter_for_colorbar = (
        scatter_quality if scatter_quality is not None else scatter_poor
    )
    if scatter_for_colorbar is not None:
        plt.colorbar(
            scatter_for_colorbar,
            ax=ax,
            label="Background Level (ADU/pixel)",
            shrink=0.6,
            aspect=20,
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Background quality plot saved to {output_file}")

    return fig


def plot_background_contours_on_image(
    image: ProcessedFitsImage,
    results: List[SimplePhotometryResult],
    output_file: Optional[Path] = None,
) -> plt.Figure:
    """
    Create contour lines on the original image showing background distribution.

    Parameters
    ----------
    image : ProcessedFitsImage
        The original image
    results : List[SimplePhotometryResult]
        Photometry results for individual stars
    output_file : Path, optional
        Path to save the plot

    Returns
    -------
    plt.Figure
        The created figure
    """
    from senpai.engine.plotting.images import plot_single_frame

    # Use plot_single_frame to get the base image with proper axis
    fig, ax = plot_single_frame(image.data, output_file=None)

    # Extract positions and background levels
    x_positions = []
    y_positions = []
    background_levels = []

    for result in results:
        if result.star.x is not None and result.star.y is not None:
            x_positions.append(result.star.x)
            y_positions.append(result.star.y)
            background_levels.append(result.background_level)

    if not x_positions:
        ax.text(
            0.5,
            0.5,
            "No position data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return fig

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    background_levels = np.array(background_levels)

    # Create a finer grid for smoother interpolation
    height, width = image.data.shape
    x_grid = np.linspace(0, width - 1, width * 2)  # 2x resolution
    y_grid = np.linspace(0, height - 1, height * 2)  # 2x resolution
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate background levels onto the grid with cubic interpolation for smoother contours
    from scipy.interpolate import griddata

    points = np.column_stack((x_positions, y_positions))
    Z = griddata(points, background_levels, (X, Y), method="cubic", fill_value=np.nan)

    # Create fewer contour lines with thicker white lines
    contour_levels = np.linspace(
        np.nanmin(background_levels), np.nanmax(background_levels), 6
    )
    contours = ax.contour(
        X, Y, Z, levels=contour_levels, colors="white", linewidths=2.0, alpha=0.8
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    # Overlay star positions (no quality coding)
    ax.scatter(
        x_positions,
        y_positions,
        c="yellow",
        s=20,
        alpha=0.8,
        marker="o",
        label=f"Stars ({len(x_positions)})",
    )

    ax.set_title("Background Level Contours on Original Image")
    ax.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Background contours on image saved to {output_file}")

    return fig


def plot_spatial_instrumental_magnitude(
    results: List[SimplePhotometryResult],
    image_shape: Tuple[int, int],
    output_file: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create a spatial map of instrumental magnitudes.

    Parameters
    ----------
    results : List[SimplePhotometryResult]
        Photometry results for individual stars
    image_shape : tuple
        Shape of the image (height, width)
    output_file : Path, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract positions and instrumental magnitudes
    x_positions = []
    y_positions = []
    inst_mags = []
    catalog_mags = []
    quality_flags = []

    for result in results:
        if result.star.x is not None and result.star.y is not None:
            x_positions.append(result.star.x)
            y_positions.append(result.star.y)
            # Convert None to NaN for proper numpy handling
            inst_mag = (
                result.instrumental_magnitude
                if result.instrumental_magnitude is not None
                else np.nan
            )
            catalog_mag = (
                result.star.magnitude if result.star.magnitude is not None else np.nan
            )
            inst_mags.append(inst_mag)
            catalog_mags.append(catalog_mag)
            quality_flags.append(result.quality_flag)

    if not x_positions:
        ax1.text(
            0.5,
            0.5,
            "No position data available",
            transform=ax1.transAxes,
            ha="center",
            va="center",
        )
        ax2.text(
            0.5,
            0.5,
            "No position data available",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
        return fig

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    inst_mags = np.array(inst_mags)
    catalog_mags = np.array(catalog_mags)
    quality_flags = np.array(quality_flags)

    # Filter out NaN instrumental magnitudes
    valid_mask = ~np.isnan(inst_mags)
    if np.any(valid_mask):
        # Plot 1: Instrumental magnitude scatter
        scatter = ax1.scatter(
            x_positions[valid_mask],
            y_positions[valid_mask],
            c=inst_mags[valid_mask],
            cmap="plasma",
            s=50,
            alpha=0.7,
        )
        ax1.set_xlabel("X Position (pixels)")
        ax1.set_ylabel("Y Position (pixels)")
        ax1.set_title("Instrumental Magnitude Distribution")
        ax1.set_aspect("equal")
        ax1.invert_yaxis()  # Image coordinates: (0,0) at top-left
        plt.colorbar(scatter, ax=ax1, label="Instrumental Magnitude")
    else:
        ax1.text(
            0.5,
            0.5,
            "No valid instrumental magnitudes",
            transform=ax1.transAxes,
            ha="center",
            va="center",
        )

    # Plot 2: Magnitude difference (catalog - instrumental)
    # This shows where we might have obscurations or issues
    valid_catalog_mask = ~np.isnan(catalog_mags) & valid_mask
    if np.any(valid_catalog_mask):
        mag_diff = catalog_mags[valid_catalog_mask] - inst_mags[valid_catalog_mask]
        scatter2 = ax2.scatter(
            x_positions[valid_catalog_mask],
            y_positions[valid_catalog_mask],
            c=mag_diff,
            cmap="RdBu_r",
            s=50,
            alpha=0.7,
        )
        ax2.set_xlabel("X Position (pixels)")
        ax2.set_ylabel("Y Position (pixels)")
        ax2.set_title("Magnitude Difference (Catalog - Instrumental)")
        ax2.set_aspect("equal")
        ax2.invert_yaxis()  # Image coordinates: (0,0) at top-left
        plt.colorbar(scatter2, ax=ax2, label="Magnitude Difference")
    else:
        ax2.text(
            0.5,
            0.5,
            "No valid magnitude comparisons",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Spatial instrumental magnitude plot saved to {output_file}")

    return fig


def plot_spatial_snr(
    results: List[SimplePhotometryResult],
    image_shape: Tuple[int, int],
    output_file: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Create a spatial map of SNR values.

    Parameters
    ----------
    results : List[SimplePhotometryResult]
        Photometry results for individual stars
    image_shape : tuple
        Shape of the image (height, width)
    output_file : Path, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract positions and SNR values
    x_positions = []
    y_positions = []
    snrs = []
    quality_flags = []

    for result in results:
        if result.star.x is not None and result.star.y is not None:
            x_positions.append(result.star.x)
            y_positions.append(result.star.y)
            snrs.append(result.snr)
            quality_flags.append(result.quality_flag)

    if not x_positions:
        ax.text(
            0.5,
            0.5,
            "No position data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return fig

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    snrs = np.array(snrs)
    quality_flags = np.array(quality_flags)

    # Create SNR scatter plot
    scatter = ax.scatter(
        x_positions, y_positions, c=snrs, cmap="viridis", s=50, alpha=0.7
    )
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Signal-to-Noise Ratio Distribution")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Image coordinates: (0,0) at top-left
    plt.colorbar(scatter, ax=ax, label="SNR", shrink=0.6, aspect=20)

    # Add quality legend
    quality_mask = quality_flags
    poor_mask = ~quality_flags

    if np.any(quality_mask) and np.any(poor_mask):
        # Add legend for quality
        ax.scatter(
            [], [], c="blue", s=50, alpha=0.7, label=f"Quality ({np.sum(quality_mask)})"
        )
        ax.scatter(
            [],
            [],
            c="red",
            s=50,
            alpha=0.7,
            label=f"Poor Quality ({np.sum(poor_mask)})",
        )
        ax.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Spatial SNR plot saved to {output_file}")

    return fig


def plot_photometry_summary(
    results: List[SimplePhotometryResult],
    summary: SimplePhotometrySummary,
    image_shape: Tuple[int, int],
    output_dir: Path,
    image: Optional[ProcessedFitsImage] = None,
    output_file: Optional[Path] = None,
) -> None:
    """
    Create photometry magnitude vs SNR plot.

    Parameters
    ----------
    results : List[SimplePhotometryResult]
        Photometry results for individual stars
    summary : SimplePhotometrySummary
        Summary statistics
    image_shape : tuple
        Shape of the image (height, width)
    output_dir : Path
        Directory to save the plots
    image : ProcessedFitsImage, optional
        Original image for contour plots (not used)
    output_file : Path, optional
        Custom output file path. If None, uses default filename in output_dir
    """
    logger.info("Creating photometry summary plots...")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Magnitude vs SNR (only plot we want to keep)
    try:
        if output_file is not None:
            mag_snr_file = output_file
        else:
            mag_snr_file = output_dir / "photometry_magnitude_vs_snr.png"
        plot_magnitude_vs_snr(results, summary, mag_snr_file)
    except Exception as e:
        logger.error(f"Failed to create magnitude vs SNR plot: {e}", exc_info=True)

    logger.info(f"Photometry plots saved to {output_dir}")
