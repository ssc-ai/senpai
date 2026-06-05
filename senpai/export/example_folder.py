#!/usr/bin/env python3
"""
Example script for exporting SENPAI runs from a folder structure to COCO format.

This script demonstrates how to use the SENPAI export functionality to convert
SENPAI run data into individual COCO format files for each image.

Usage:
    python example_folder.py /path/to/senpai/data /path/to/output
"""

import argparse
import sys
from pathlib import Path

from senpai.export.cli import export_folder


def main():
    """Main function for the example script."""
    parser = argparse.ArgumentParser(
        description="Example script for exporting SENPAI runs to COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all runs from a data folder
  python example_folder.py /path/to/senpai/data /path/to/output

  # Export with custom settings
  python example_folder.py /path/to/senpai/data /path/to/output \\
    --max-runs 5 --write-fits --save-annotated-images --verbose

  # Export with filtering
  python example_folder.py /path/to/senpai/data /path/to/output \\
    --snr-cut 2.0 --mask-radius 1000
        """,
    )

    parser.add_argument("data_folder", help="Path to folder containing SENPAI runs")
    parser.add_argument("output_folder", help="Output directory for COCO files")
    parser.add_argument("--max-runs", type=int, help="Maximum number of runs to export")
    parser.add_argument("--write-png", action="store_true", default=False, help="Save PNG images (default: False)")
    parser.add_argument("--write-fits", action="store_true", default=True, help="Save FITS images (default: True)")
    parser.add_argument(
        "--save-annotated-images", action="store_true", default=False, help="Save annotated images (default: False)"
    )
    parser.add_argument(
        "--remove-median", action="store_true", default=False, help="Remove median from images (default: False)"
    )
    parser.add_argument("--snr-cut", type=float, default=0.5, help="Minimum SNR for annotations (default: 0.5)")
    parser.add_argument("--box-size", type=int, default=4, help="Bounding box size for point sources (default: 4)")
    parser.add_argument(
        "--streak-box-size", type=int, default=10, help="Bounding box size for satellites (default: 10)"
    )
    parser.add_argument("--mask-radius", type=float, help="Radius to mask around center (pixels)")
    parser.add_argument("--no-calibrations", action="store_true", default=False, help="Skip applying calibrations")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")

    args = parser.parse_args()

    # Validate inputs
    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)

    if not data_folder.exists():
        print(f"Error: Data folder {data_folder} does not exist")
        sys.exit(1)

    if not data_folder.is_dir():
        print(f"Error: {data_folder} is not a directory")
        sys.exit(1)

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Exporting SENPAI runs from {data_folder} to {output_folder}")
    print("Settings:")
    print(f"  Max runs: {args.max_runs or 'All'}")
    print(f"  Write PNG: {args.write_png}")
    print(f"  Write FITS: {args.write_fits}")
    print(f"  Save annotated images: {args.save_annotated_images}")
    print(f"  Remove median: {args.remove_median}")
    print(f"  SNR cut: {args.snr_cut}")
    print(f"  Box size: {args.box_size}")
    print(f"  Streak box size: {args.streak_box_size}")
    print(f"  Mask radius: {args.mask_radius or 'None'}")
    print(f"  Apply calibrations: {not args.no_calibrations}")
    print(f"  Verbose: {args.verbose}")
    print()

    try:
        # Export the folder
        export_folder(
            folder_path=str(data_folder),
            output_dir=str(output_folder),
            max_runs=args.max_runs,
            write_png=args.write_png,
            write_fits=args.write_fits,
            save_annotated_images=args.save_annotated_images,
            remove_median=args.remove_median,
            snr_cut=args.snr_cut,
            box_size=args.box_size,
            streak_box_size=args.streak_box_size,
            mask_radius=args.mask_radius,
            apply_calibrations=not args.no_calibrations,
            verbose=args.verbose,
        )

        print("\nExport completed successfully!")
        print(f"Output files are in: {output_folder}")

        # Show what was created
        if args.verbose:
            print("\nCreated files:")
            for file_path in sorted(output_folder.rglob("*")):
                if file_path.is_file():
                    print(f"  {file_path.relative_to(output_folder)}")

    except KeyboardInterrupt:
        print("\nExport interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nExport failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
