"""Single rate-track frame processing CLI.

.. deprecated::
    Use ``python -m senpai.cli.detect`` instead, which handles all frame types
    through the unified collect pipeline.
"""

import json
import logging
import warnings
from pathlib import Path

from senpai.cli.common import ensure_output_dir, save_run_metadata
from senpai.core.config import initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import set_log_level

# Re-exports for backward compatibility
from senpai.engine.detection.streak.rate_extraction import (
    build_streak_metadata,
    extract_rate_streak_measurement,
    extract_streak_centers_as_sources,
)
from senpai.engine.processing.rate import process_rate_fits_rate  # noqa: F401
from senpai.engine.utils.file_io import load_dng_file, load_fits_file, load_jpeg_file
from senpai.engine.utils.serialization import fits_header_to_jsonable, jsonable

logger = logging.getLogger(__name__)

# Keep old names for backward compatibility
_jsonable = jsonable
_fits_header_to_jsonable = fits_header_to_jsonable
_extract_rate_streak_measurement = extract_rate_streak_measurement
_build_streak_metadata = build_streak_metadata
_extract_streak_centers_as_sources = extract_streak_centers_as_sources


if __name__ == "__main__":
    warnings.warn(
        "senpai.cli.rate is deprecated — use senpai.cli.detect instead",
        DeprecationWarning,
        stacklevel=1,
    )

    default_output_dir = Path(".")

    import argparse

    from senpai.astrometry import enforce_indices, test_astrometry_install
    from senpai.catalog.runner import enforce_catalog

    parser = argparse.ArgumentParser(
        description="Process a single RATE-track frame (deprecated: use detect)"
    )
    parser.add_argument("-f", "--fits", help="Path to input FITS/DNG/JPG file", type=str)
    parser.add_argument(
        "-c", "--config",
        help=f"Config file, defaults to {LOCAL_APP_CONFIG_OVERRIDE}",
        type=str, default=LOCAL_APP_CONFIG_OVERRIDE,
    )
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str, default=default_output_dir)
    parser.add_argument("--no_wcs", help="Disable WCS attempt", action="store_true", default=False)
    parser.add_argument("--max_sources", help="Max pseudo-sources for astrometry", type=int, default=200)
    parser.add_argument("--n_streaks", help="Max streak candidates", type=int, default=10)
    parser.add_argument("-P", "--photometry", help="Perform photometry (default: True)", action="store_true", default=None)
    parser.add_argument("-D", "--detect", help="Detect point sources", action="store_true", default=False)
    args = parser.parse_args()

    if args.photometry is None:
        args.photometry = True
    if not args.fits:
        raise SystemExit("Must provide --fits")

    output_dir = ensure_output_dir(Path(args.output_dir), default_stem=Path(args.fits).stem)

    cfg = initialize_config(Path(args.config))
    cfg.runtime.output_dir = output_dir
    cfg.runtime.run_id = Path(args.fits).stem
    cfg.detection.detect = args.detect
    set_log_level(cfg.logging.level)
    save_run_metadata(output_dir, "senpai.cli.rate", cfg)

    if not args.no_wcs:
        test_astrometry_install()
        enforce_indices()
        enforce_catalog()

    if str(args.fits).endswith(".DNG"):
        image = load_dng_file(str(args.fits))
    elif str(args.fits).lower().endswith((".jpg", ".jpeg")):
        image = load_jpeg_file(args.fits)
    else:
        image = load_fits_file(str(args.fits))

    # Route through collect pipeline — CLI declares rate-track mode regardless of header
    from senpai.engine.models.metadata import TrackMode
    from senpai.engine.processing.collect import final_plots, process_senpai_collect

    senpai_run = process_senpai_collect(
        [image], id=cfg.runtime.run_id, force_track_mode=TrackMode.RATE
    )

    # Write results
    result = senpai_run.to_result()
    with open(output_dir / f"senpai_{result.id}.json", "w") as f:
        json.dump(result.model_dump(), f)

    summary = senpai_run.to_summary()
    with open(output_dir / f"senpai_{summary.id}_summary.json", "w") as f:
        json.dump(summary.model_dump(), f)

    # Write rateframe.json for backward compatibility
    if result.rate_track_frames:
        with open(output_dir / "rateframe.json", "w") as f:
            json.dump(result.rate_track_frames[0].model_dump(mode="json"), f)

    final_plots(senpai_run, output_dir)
