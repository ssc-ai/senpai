#!/usr/bin/env python3
"""Catalog magnitude-distribution analysis tool.

Samples random sky positions and analyzes the magnitude distribution of stars
in the catalog to determine the brightest and faintest sources.
"""

import argparse
import logging
from pathlib import Path

# Disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

import senpai.catalog.sstr7 as sstr7  # noqa: E402
from senpai.core.config import initialize_config  # noqa: E402
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE  # noqa: E402
from senpai.core.logging import set_log_level  # noqa: E402

logger = logging.getLogger(__name__)


def sample_random_sky_positions(num_samples: int) -> list[tuple[float, float]]:
    """Generate random sky positions uniformly distributed across the celestial sphere.

    Args:
        num_samples: Number of positions to generate

    Returns:
        List of (ra, dec) tuples in degrees
    """
    # Generate uniform random positions
    # For RA: uniform in [0, 360)
    # For Dec: uniform in cos(dec) to get uniform distribution on sphere
    ra_values = np.random.uniform(0, 360, num_samples)
    # Dec: uniform in cos(dec) from -90 to 90
    cos_dec_values = np.random.uniform(-1, 1, num_samples)
    dec_values = np.degrees(np.arccos(cos_dec_values)) - 90  # Convert to [-90, 90]

    positions = [(ra, dec) for ra, dec in zip(ra_values, dec_values, strict=True)]
    return positions


def query_stars_for_sample(
    ra: float, dec: float, fov_size: float, catalog_path: str
) -> list[dict]:
    """Query catalog stars for a given sky position.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        fov_size: FOV size in degrees
        catalog_path: Path to SSTR7 catalog

    Returns:
        List of star dictionaries with 'ra', 'dec', 'mv' keys (in degrees)
    """
    stars = sstr7.query_by_los_radec_with_rotation(
        y_fov=fov_size,
        x_fov=fov_size,
        ra=ra,
        dec=dec,
        rotation=0.0,
        rootPath=catalog_path,
        faint_lim=None,  # No limits - get all stars
        bright_lim=None,
        safety_margin=0.1,
    )

    # Convert to degrees and normalize
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

    return stars


def analyze_catalog_magnitudes(
    catalog_path: str,
    num_samples: int = 100,
    fov_size: float = 10.0,
    output_dir: Path | None = None,
) -> dict:
    """Analyze magnitude distribution of stars in the catalog.

    Args:
        catalog_path: Path to SSTR7 catalog
        num_samples: Number of random sky positions to sample
        fov_size: FOV size in degrees for each sample
        output_dir: Output directory for plots (optional)

    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Sampling {num_samples} random sky positions with {fov_size}° FOV")
    positions = sample_random_sky_positions(num_samples)

    all_magnitudes = []
    valid_magnitudes = []  # Exclude invalid magnitudes (mv >= 32)
    stars_per_sample = []

    logger.info("Querying catalog for each sample...")
    for ra, dec in tqdm(positions, desc="Sampling", unit="sample"):
        stars = query_stars_for_sample(ra, dec, fov_size, catalog_path)
        stars_per_sample.append(len(stars))

        for star in stars:
            mv = star.get("mv", None)
            if mv is not None:
                all_magnitudes.append(mv)
                # Valid magnitudes are < 32 (32 means "no magnitude")
                if mv < 32:
                    valid_magnitudes.append(mv)

    if not valid_magnitudes:
        logger.error("No valid magnitudes found! Check catalog path and query parameters.")
        return {}

    # Calculate statistics
    valid_magnitudes_array = np.array(valid_magnitudes)
    results = {
        "num_samples": num_samples,
        "fov_size": fov_size,
        "total_stars": len(all_magnitudes),
        "valid_stars": len(valid_magnitudes),
        "brightest_mag": float(np.min(valid_magnitudes_array)),
        "faintest_mag": float(np.max(valid_magnitudes_array)),
        "median_mag": float(np.median(valid_magnitudes_array)),
        "mean_mag": float(np.mean(valid_magnitudes_array)),
        "std_mag": float(np.std(valid_magnitudes_array)),
        "percentiles": {
            "1%": float(np.percentile(valid_magnitudes_array, 1)),
            "5%": float(np.percentile(valid_magnitudes_array, 5)),
            "25%": float(np.percentile(valid_magnitudes_array, 25)),
            "75%": float(np.percentile(valid_magnitudes_array, 75)),
            "95%": float(np.percentile(valid_magnitudes_array, 95)),
            "99%": float(np.percentile(valid_magnitudes_array, 99)),
        },
        "avg_stars_per_sample": float(np.mean(stars_per_sample)),
        "min_stars_per_sample": int(np.min(stars_per_sample)),
        "max_stars_per_sample": int(np.max(stars_per_sample)),
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Catalog Magnitude Distribution Analysis Results")
    logger.info("=" * 60)
    logger.info(f"Total stars sampled: {results['total_stars']:,}")
    logger.info(f"Valid magnitudes (mv < 32): {results['valid_stars']:,}")
    logger.info("\nMagnitude Statistics:")
    logger.info(f"  Brightest: {results['brightest_mag']:.2f} mag")
    logger.info(f"  Faintest:  {results['faintest_mag']:.2f} mag")
    logger.info(f"  Median:    {results['median_mag']:.2f} mag")
    logger.info(f"  Mean:      {results['mean_mag']:.2f} mag")
    logger.info(f"  Std Dev:   {results['std_mag']:.2f} mag")
    logger.info("\nPercentiles:")
    for pct, value in results["percentiles"].items():
        logger.info(f"  {pct:>4s}: {value:.2f} mag")
    logger.info("\nStars per sample:")
    logger.info(f"  Average: {results['avg_stars_per_sample']:.1f}")
    logger.info(f"  Min:     {results['min_stars_per_sample']:,}")
    logger.info(f"  Max:     {results['max_stars_per_sample']:,}")
    logger.info("=" * 60)

    # Create histogram plot
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Histogram of magnitudes
        ax1.hist(
            valid_magnitudes,
            bins=100,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.axvline(results["brightest_mag"], color="red", linestyle="--", linewidth=2, label="Brightest")
        ax1.axvline(results["faintest_mag"], color="orange", linestyle="--", linewidth=2, label="Faintest")
        ax1.axvline(results["median_mag"], color="green", linestyle="--", linewidth=2, label="Median")
        ax1.set_xlabel("Magnitude (mv)", fontsize=12)
        ax1.set_ylabel("Number of Stars", fontsize=12)
        ax1.set_title(
            f"Magnitude Distribution (n={results['valid_stars']:,} stars from {num_samples} samples)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Histogram of stars per sample
        ax2.hist(
            stars_per_sample,
            bins=min(50, len(set(stars_per_sample))),
            alpha=0.7,
            color="coral",
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.axvline(results["avg_stars_per_sample"], color="blue", linestyle="--", linewidth=2, label="Mean")
        ax2.set_xlabel("Number of Stars per Sample", fontsize=12)
        ax2.set_ylabel("Number of Samples", fontsize=12)
        ax2.set_title(
            f"Stars per Sample Distribution (FOV={fov_size}°)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "catalog_magnitude_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"\nPlot saved to: {plot_path}")
        plt.close()

    return results


def main() -> int:
    """Parse CLI arguments and run the catalog magnitude-distribution analysis.

    Returns:
        Process exit code (0 on success, 1 if no results were produced).
    """
    parser = argparse.ArgumentParser(
        description="Analyze magnitude distribution of stars in the SSTR7 catalog"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=LOCAL_APP_CONFIG_OVERRIDE or "resources/config/local.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help="Path to SSTR7 catalog (from config if not provided)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random sky positions to sample (default: 100)",
    )
    parser.add_argument(
        "--fov-size",
        type=float,
        default=10.0,
        help="FOV size in degrees for each sample (default: 10.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="catalog_analysis",
        help="Output directory for plots (default: catalog_analysis)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = initialize_config(Path(args.config))
    set_log_level(args.log_level)

    # Ensure matplotlib loggers stay at WARNING level
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.colorbar").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

    # Get catalog path
    catalog_path = args.catalog_path
    if catalog_path is None:
        catalog_path = config.star_catalog.path
        if not catalog_path:
            logger.error("Catalog path not provided and not found in config")
            return 1

    logger.info(f"Using catalog path: {catalog_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = analyze_catalog_magnitudes(
        catalog_path=catalog_path,
        num_samples=args.num_samples,
        fov_size=args.fov_size,
        output_dir=output_dir,
    )

    if not results:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())



