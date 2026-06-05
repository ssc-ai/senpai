#!/usr/bin/env python3
"""
CLI for photometry measurements on FITS images.

This CLI provides a command-line interface for performing comprehensive
photometry on astronomical images with WCS solutions.

Usage Examples:
--------------
Basic photometry on a single image:
    python -m senpai.cli.photometry -f image.fits -o photometry_results.json

Photometry with custom configuration:
    python -m senpai.cli.photometry -f image.fits -o results.json \\
        --min-snr 5.0 --gain 0.1

Photometry with detailed output:
    python -m senpai.cli.photometry -f image.fits -o results.json \\
        --verbose --save-plots --save-apertures
"""

import argparse
import json
import logging
from pathlib import Path

from senpai.cli.common import save_run_metadata
from senpai.core.config import initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import set_log_level

# Re-export for backward compatibility
from senpai.engine.processing.photometry_pipeline import process_image_photometry

logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Perform photometry on FITS images")

    # Required arguments
    parser.add_argument("-f", "--fits", required=True, help="Path to input FITS file")
    parser.add_argument("-o", "--output", required=True, help="Path to output JSON file")

    # Optional arguments
    parser.add_argument(
        "-c",
        "--config",
        help=f"Config file, defaults to {LOCAL_APP_CONFIG_OVERRIDE}",
        type=str,
        default=LOCAL_APP_CONFIG_OVERRIDE,
    )
    parser.add_argument("--output-dir", help="Output directory for plots", type=str)
    parser.add_argument("--save-plots", help="Save diagnostic plots", action="store_true")
    parser.add_argument("--save-apertures", help="Save aperture visualization", action="store_true")
    parser.add_argument("--verbose", help="Verbose output", action="store_true")

    # Photometry configuration (overrides the config file when given)
    parser.add_argument("--min-snr", type=float, default=None, help="Minimum signal-to-noise ratio")
    parser.add_argument("--max-crowding", type=float, default=None, help="Maximum crowding factor")
    parser.add_argument("--read-noise", type=float, default=None, help="Read noise in electrons")
    parser.add_argument("--gain", type=float, default=None, help="Gain in electrons per ADU")

    args = parser.parse_args()

    # Initialize configuration
    cfg = initialize_config(args.config)
    set_log_level(cfg.logging.level)

    # Create output directory
    output_path = Path(args.output)
    output_dir = Path(args.output_dir) if args.output_dir else output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    save_run_metadata(output_dir, "senpai.cli.photometry", cfg)

    # Photometry configuration: config file values with CLI overrides
    overrides = {
        key: value
        for key, value in {
            "min_snr": args.min_snr,
            "max_crowding": args.max_crowding,
            "read_noise": args.read_noise,
            "gain": args.gain,
        }.items()
        if value is not None
    }
    photometry_config = cfg.photometry.model_copy(update=overrides)

    # Process photometry
    try:
        results = process_image_photometry(
            args.fits,
            config=photometry_config,
            output_dir=output_dir,
            save_plots=args.save_plots,
            save_apertures=args.save_apertures,
            verbose=args.verbose,
        )

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Photometry results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error processing photometry: {e}")
        raise


if __name__ == "__main__":
    main()
