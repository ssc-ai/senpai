# this is a CLI for the SENPAI algorithm for collecting stars from a series of images
import json
import logging
import os
from pathlib import Path

import senpai
from senpai.astrometry import enforce_indices, require_astrometry_install
from senpai.catalog.runner import enforce_catalog
from senpai.cli.common import profile_run, save_run_metadata
from senpai.core.config import initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import set_log_level
from senpai.engine.models.images import ProcessedFitsImage

# Re-exports for backward compatibility
from senpai.engine.processing.collect import final_plots, process_senpai_collect
from senpai.engine.utils.file_io import load_fits_files
from senpai.engine.utils.frame_organization import (
    extract_id_from_header,
    get_all_images_in_directory,
    get_imageset_by_filename,
    get_imageset_by_id,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # python -m senpai.cli.single -f senpai/tests/data/7ee48b4c-de0e-4c8c-a44b-1c9e7254ae6c/47306_7ee48b4c-de0e-4c8c-a44b-1c9e7254ae6c_5.fits --plots
    # python -m senpai.cli.single -j senpai/tests/data/output_starlistinimage.json --plots -o tmp

    default_output_dir = Path(".")

    import argparse

    parser = argparse.ArgumentParser(description="Process a single FITS or JSON file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--id", help="ID of the imageset to process", type=str)
    group.add_argument(
        "-s", "--string_match", help="Regex to match imageset to process", type=str
    )
    group.add_argument(
        "-a", "--all", help="Process all images in data_directory", action="store_true"
    )
    parser.add_argument(
        "-d", "--data_directory", help="Path to input directory", type=str
    )
    parser.add_argument(
        "-D",
        "--detect",
        help="detect point sources",
        action="store_true",
        default=False,
    )
    # argument for whether or not to produce plots:
    parser.add_argument(
        "-c",
        "--config",
        help=f"Config file, defaults to {LOCAL_APP_CONFIG_OVERRIDE}",
        type=str,
        default=LOCAL_APP_CONFIG_OVERRIDE,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory",
        type=str,
        default=default_output_dir,
    )
    parser.add_argument(
        "-k", "--header_id_key", help="Header ID key", type=str, default=None
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = initialize_config(Path(args.config))
    set_log_level(level=config.logging.level)

    # Log variable-kernel configuration for traceability
    vk_cfg = config.streak.variable_kernel
    logger.info(
        "Variable streak kernels: enable=%s, angle_thresh_deg=%.2f, length_thresh_fraction=%.3f",
        vk_cfg.enable,
        vk_cfg.angle_thresh_deg,
        vk_cfg.length_thresh_fraction,
    )

    enforce_indices()
    enforce_catalog()
    require_astrometry_install()

    data_directory = Path(args.data_directory)

    run_id = f"senpai_{senpai.__version__}"
    if args.string_match:
        files = get_imageset_by_filename(data_directory, args.string_match)

    elif args.id:
        run_id = args.id
        if not args.header_id_key:
            raise ValueError(
                "Header ID key is required for ID-based collection (-k/--header_id_key)"
            )
        files = get_imageset_by_id(data_directory, args.id, args.header_id_key)
    elif args.all:
        files = get_all_images_in_directory(data_directory)
    else:
        raise ValueError("No input file selection method provided [id, regex, all]")

    if args.header_id_key:
        ids = [extract_id_from_header(file, args.header_id_key) for file in files]
        if len(ids) == 0:
            raise ValueError(
                f"No ID found in the header for the key {args.header_id_key}"
            )
        if len(set(ids)) > 1:
            raise ValueError(f"All files must have the same ID, I found: {set(ids)}")

        id_value = ids[0]
        if args.id is not None and args.id != id_value:
            raise ValueError(
                f"ID mismatch, I found: {id_value} but you requested: {args.id}"
            )
        run_id = id_value

    file_list: list[ProcessedFitsImage] = load_fits_files(files)

    config.runtime.run_id = run_id
    config.runtime.output_dir = output_dir
    config.detection.detect = args.detect

    save_run_metadata(output_dir, "senpai.cli.rate_sidereal", config)

    senpai_run = profile_run(process_senpai_collect, file_list, id=run_id, run_id=run_id)

    result = senpai_run.to_result()
    json.dump(
        result.model_dump(),
        open(output_dir / f"senpai_{result.senpai_version}_{result.id}.json", "w"),
        indent=4,
    )

    summary = senpai_run.to_summary()
    json.dump(
        summary.model_dump(),
        open(output_dir / f"senpai_{summary.senpai_version}_{summary.id}_summary.json", "w"),
        indent=4,
    )

    # Write per-frame JSON files
    for sid_frame in result.sidereal_frames:
        path = output_dir / f"frame_{sid_frame.index}_sidereal.json"
        with open(path, "w") as f:
            json.dump(sid_frame.model_dump(mode="json"), f, indent=4)

    for rt_frame in result.rate_track_frames:
        path = output_dir / f"frame_{rt_frame.index}_rate.json"
        with open(path, "w") as f:
            json.dump(rt_frame.model_dump(mode="json"), f, indent=4)

    logger.info(
        f"Wrote {len(result.sidereal_frames)} sidereal + {len(result.rate_track_frames)} rate per-frame JSON files"
    )

    final_plots(senpai_run, output_dir)
