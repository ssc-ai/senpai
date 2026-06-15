"""Build a per-night photometric + observability calibration from processed
batches, and render the night-level plots.

Aggregates a night's per-batch ``senpai_*.json`` into ``night_calibration.json``
+ ``plot_data.json`` and renders the night plots (search rate, slew model, PSF
profile, PSF concentration, CCD temperature, ZP drift, extinction, …). Works on
any processed-night dir (output of a senpai night run); not burr-specific.

Usage::

    python -m senpai.cli.calibrate <processed_night_dir> [-o out] [--no-plots]
    python -m senpai.cli.calibrate <processed_night_dir> --from-plot-data

The pixel-level plots (psf_profile, psf_concentration) re-read each frame's raw
FITS via its stored path; if the raw frames are absent those two are skipped and
the rest still render. ``--from-plot-data`` re-renders everything from a saved
``plot_data.json`` with no reprocessing and no raw data.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from senpai.engine.observability.calibration import (
        analyze_night,
        load_plot_data,
        plot_calibration,
        save_calibration,
    )

    parser = argparse.ArgumentParser(
        description="Build a per-night calibration (JSON + plots) from processed "
                    "batches.",
    )
    parser.add_argument(
        "processed_night_dir",
        help="Processed-night dir (must contain manifest.json), e.g. the output "
             "of a senpai night run.",
    )
    parser.add_argument(
        "-o", "--output_dir", default=None,
        help="Output dir for calibration JSON + plots "
             "(default: <night_dir>/calibration/).",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip plot rendering.",
    )
    parser.add_argument(
        "--from-plot-data", action="store_true",
        help="Skip reprocessing: render plots from an existing "
             "<output>/plot_data.json instead of the batch JSONs.",
    )
    args = parser.parse_args(argv)

    night_dir = Path(args.processed_night_dir)
    out_dir = Path(args.output_dir) if args.output_dir else night_dir / "calibration"

    if args.from_plot_data:
        plot_calibration(load_plot_data(out_dir / "plot_data.json"), out_dir)
        return 0

    calib = analyze_night(night_dir)
    save_calibration(calib, out_dir)
    if not args.no_plots:
        plot_calibration(calib, out_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
