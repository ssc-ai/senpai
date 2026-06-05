"""Unified SENPAI detection CLI.

Accepts any mix of sidereal and rate-track frames (single file or directory).
All inputs go through the full collect pipeline, which handles N=1 correctly.

Usage::

    python -m senpai.cli.detect -f <fits_or_dir> -o <dir> [-c config] [-D] [-P]
"""

import json
import logging
from pathlib import Path

from senpai.cli.common import ensure_output_dir, save_run_metadata
from senpai.core.config import initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import set_log_level

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse

    from senpai.astrometry import enforce_indices, require_astrometry_install
    from senpai.catalog.runner import enforce_catalog

    parser = argparse.ArgumentParser(
        description="Unified SENPAI detection: auto-routes single/multi frame, sidereal/rate"
    )
    parser.add_argument(
        "-f",
        "--files",
        help="FITS file(s) or directory of FITS files",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        help=f"Config file (default: {LOCAL_APP_CONFIG_OVERRIDE})",
        type=str,
        default=LOCAL_APP_CONFIG_OVERRIDE,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-D",
        "--detect",
        help="Detect non-star objects (point sources in rate frames, streaks in sidereal/rate frames)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-P",
        "--photometry",
        help="Perform photometry",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    # Gather FITS files
    fits_paths = []
    for f in args.files:
        p = Path(f)
        if p.is_dir():
            fits_paths.extend(sorted(p.glob("*.fits")) + sorted(p.glob("*.fit")) + sorted(p.glob("*.fts")))
        elif p.is_file():
            fits_paths.append(p)
        else:
            logger.warning("Skipping non-existent path: %s", f)

    if not fits_paths:
        parser.error("No FITS files found")

    # Determine output dir
    default_stem = fits_paths[0].stem if len(fits_paths) == 1 else fits_paths[0].parent.name or "detect"
    output_dir = ensure_output_dir(Path(args.output_dir), default_stem=default_stem)

    # Init config
    config = initialize_config(Path(args.config))
    config.runtime.output_dir = output_dir
    config.runtime.run_id = default_stem
    config.detection.detect = args.detect
    config.detection.detect_streaks = args.detect
    set_log_level(config.logging.level)

    save_run_metadata(output_dir, "senpai.cli.detect", config)

    require_astrometry_install()
    enforce_indices()
    enforce_catalog()

    # Load files
    from senpai.engine.utils.file_io import load_fits_files

    file_list = load_fits_files(fits_paths)

    # All inputs go through the full collect pipeline (handles N=1 correctly)
    logger.info("Processing %d frame(s) through collect pipeline", len(file_list))
    from senpai.engine.processing.collect import final_plots, process_senpai_collect

    senpai_run = process_senpai_collect(file_list, id=default_stem)

    # Write results
    result = senpai_run.to_result()
    with open(output_dir / f"senpai_{result.id}.json", "w") as f:
        json.dump(result.model_dump(), f, indent=4)

    summary = senpai_run.to_summary()
    with open(output_dir / f"senpai_{summary.id}_summary.json", "w") as f:
        json.dump(summary.model_dump(), f, indent=4)

    # Per-frame JSONs
    for sid_frame in result.sidereal_frames:
        path = output_dir / f"frame_{sid_frame.index}_sidereal.json"
        with open(path, "w") as f:
            json.dump(sid_frame.model_dump(mode="json"), f, indent=4)

    for rt_frame in result.rate_track_frames:
        path = output_dir / f"frame_{rt_frame.index}_rate.json"
        with open(path, "w") as f:
            json.dump(rt_frame.model_dump(mode="json"), f, indent=4)

    # Correlated streaks
    if senpai_run.correlated_streaks:
        with open(output_dir / "correlated_streaks.json", "w") as f:
            json.dump(
                [cs.model_dump(mode="json") for cs in senpai_run.correlated_streaks],
                f,
                indent=4,
            )
        logger.info("Wrote %d correlated streaks", len(senpai_run.correlated_streaks))

    # Per-frame streak candidates (sidereal + rate)
    for frame in senpai_run.sidereal_frames + senpai_run.rate_track_frames:
        if frame.streak_candidates:
            path = output_dir / f"streak_candidates_{frame.index}.json"
            with open(path, "w") as f:
                json.dump(
                    [
                        sc.model_dump(mode="json") if hasattr(sc, "model_dump") else sc
                        for sc in frame.streak_candidates
                    ],
                    f,
                    indent=4,
                )

    final_plots(senpai_run, output_dir)

    logger.info(
        "Done: %d sidereal + %d rate frames, %d correlated streaks",
        len(result.sidereal_frames),
        len(result.rate_track_frames),
        len(senpai_run.correlated_streaks),
    )
