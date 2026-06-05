"""Single sidereal frame processing CLI.

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
from senpai.engine.detection.point.fwhm import measure_fwhm_from_catalog_stars  # noqa: F401
from senpai.engine.models.starfield import StarListImage
from senpai.engine.plotting.images import plot_single_frame  # noqa: F401
from senpai.engine.processing.sidereal import (
    process_astrometry_fits_sidereal,  # noqa: F401
    process_astrometry_json_sidereal,
)
from senpai.engine.utils.astrometry_diagnostics import (  # noqa: F401
    calculate_residual_errors,
    log_residual_errors,
)
from senpai.engine.utils.file_io import load_dng_file, load_fits_file, load_jpeg_file

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.warn(
        "senpai.cli.sidereal is deprecated — use senpai.cli.detect instead",
        DeprecationWarning,
        stacklevel=1,
    )

    default_output_dir = Path(".")

    import argparse

    from senpai.astrometry import enforce_indices, test_astrometry_install
    from senpai.catalog.runner import enforce_catalog

    parser = argparse.ArgumentParser(description="Process a single FITS or JSON file (deprecated: use detect)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--fits", help="Path to input FITS file", type=str)
    group.add_argument("-j", "--json", help="Path to input JSON file", type=str)
    parser.add_argument(
        "-c", "--config",
        help=f"Config file, defaults to {LOCAL_APP_CONFIG_OVERRIDE}",
        type=str, default=LOCAL_APP_CONFIG_OVERRIDE,
    )
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str, default=default_output_dir)
    parser.add_argument("-P", "--photometry", help="Perform photometry (default: True)", action="store_true", default=None)
    parser.add_argument("-S", "--detect_streaks", help="Detect streaks", action="store_true", default=False)
    parser.add_argument("--profile", help="Profile the processing run", action="store_true", default=False)
    args = parser.parse_args()

    if args.photometry is None:
        args.photometry = True

    output_dir = ensure_output_dir(
        Path(args.output_dir),
        default_stem=Path(args.fits).stem if args.fits else None,
    )

    cfg = initialize_config(Path(args.config))
    cfg.runtime.output_dir = output_dir
    cfg.detection.detect_streaks = args.detect_streaks
    save_run_metadata(output_dir, "senpai.cli.sidereal", cfg)
    set_log_level(cfg.logging.level)
    test_astrometry_install()
    enforce_indices()
    enforce_catalog()

    if args.json:
        # JSON path: can't go through collect, use legacy function
        with open(args.json) as fh:
            sources = StarListImage.model_validate_json(fh.read())
        wcs_starfield = process_astrometry_json_sidereal(sources)
        with open(Path(output_dir) / "starfield.json", "w") as fh:
            json.dump(wcs_starfield.model_dump(), fh, indent=4)
    else:
        # FITS path: route through collect pipeline
        if str(args.fits).endswith(".DNG"):
            image = load_dng_file(str(args.fits))
        elif str(args.fits).endswith((".jpg", ".jpeg")):
            image = load_jpeg_file(args.fits)
        else:
            image = load_fits_file(str(args.fits))

        from senpai.engine.models.metadata import TrackMode
        from senpai.engine.processing.collect import final_plots, process_senpai_collect

        cfg.runtime.run_id = Path(args.fits).stem
        # CLI declares sidereal mode regardless of header
        senpai_run = process_senpai_collect(
            [image], id=Path(args.fits).stem, force_track_mode=TrackMode.SIDEREAL
        )

        # Write results
        result = senpai_run.to_result()
        with open(output_dir / f"senpai_{result.id}.json", "w") as f:
            json.dump(result.model_dump(), f, indent=4)

        summary = senpai_run.to_summary()
        with open(output_dir / f"senpai_{summary.id}_summary.json", "w") as f:
            json.dump(summary.model_dump(), f, indent=4)

        # Also write starfield.json for backward compatibility
        if result.sidereal_frames:
            sf = result.sidereal_frames[0].starfield
            if sf:
                with open(output_dir / "starfield.json", "w") as f:
                    json.dump(sf.model_dump(mode="json"), f, indent=4)

        final_plots(senpai_run, output_dir)
