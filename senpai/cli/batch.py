# CLI for batch processing of SENPAI algorithm across multiple date directories
import argparse
import concurrent.futures
import functools
import json
import logging
import multiprocessing
import os
import signal
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from senpai.astrometry import enforce_indices, require_astrometry_install
from senpai.catalog.runner import enforce_catalog
from senpai.cli.common import save_run_metadata, write_frame_quicklooks
from senpai.core.config import get_config, initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import set_log_level
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.processing.collect import final_plots, process_senpai_collect
from senpai.engine.utils.file_io import load_fits_files
from senpai.engine.utils.frame_organization import extract_id_from_header, get_imageset_by_id

logger = logging.getLogger(__name__)


def process_single_dataset(
    data_path: Path,
    id_value: str,
    header_id_key: str,
    output_dir: Path,
    detect: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Process a single dataset identified by its ID."""
    start_time = time.time()

    # Check if output already exists (match any version or legacy filename)
    dataset_output_dir = output_dir / id_value
    existing_results = list(dataset_output_dir.glob(f"senpai_*_{id_value}.json")) + list(
        dataset_output_dir.glob(f"{id_value}.json")
    )

    if skip_existing and existing_results:
        logger.info(f"Skipping dataset {id_value} - output already exists")
        return {
            "id": id_value,
            "status": "skipped",
            "error": None,
            "time_taken": time.time() - start_time,
            "data_path": str(data_path),
        }

    try:
        # Get files matching the ID
        files = get_imageset_by_id(data_path, id_value, header_id_key)
        if not files:
            return {
                "id": id_value,
                "status": "error",
                "error": f"No files found for ID {id_value}",
                "time_taken": time.time() - start_time,
                "data_path": str(data_path),
            }

        # Load FITS files
        file_list: list[ProcessedFitsImage] = load_fits_files(files)

        # Create dataset-specific output directory
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Update config for this run
        config = get_config()
        config.runtime.run_id = id_value
        config.runtime.output_dir = dataset_output_dir
        config.detection.detect = detect

        # Save run metadata for reproducibility
        save_run_metadata(dataset_output_dir, "senpai.cli.batch", config)

        # Process the dataset
        senpai_run = process_senpai_collect(file_list, id=id_value)

        result = senpai_run.to_result()
        with open(dataset_output_dir / f"senpai_{result.senpai_version}_{id_value}.json", "w") as f:
            json.dump(result.model_dump(), f)

        summary = senpai_run.to_summary()
        with open(dataset_output_dir / f"senpai_{summary.senpai_version}_{id_value}_summary.json", "w") as f:
            json.dump(summary.model_dump(), f)

        # Per-frame quick-look JSONs (detections + WCS, no bulk star arrays)
        write_frame_quicklooks(summary, dataset_output_dir)

        final_plots(senpai_run, dataset_output_dir)

        return {
            "id": id_value,
            "status": "success",
            "error": senpai_run.error_message,
            "time_taken": time.time() - start_time,
            "frames_processed": len(senpai_run.sidereal_frames) + len(senpai_run.rate_track_frames),
            "data_path": str(data_path),
        }

    except Exception as e:
        logger.exception(f"Error processing dataset {id_value}")
        return {
            "id": id_value,
            "status": "error",
            "error": str(e),
            "time_taken": time.time() - start_time,
            "data_path": str(data_path),
        }


# Move this function outside of discover_datasets
def extract_id_from_file(file_path, header_id_key):
    try:
        header_id = extract_id_from_header(file_path, header_id_key)
        if header_id is not False:
            return header_id
    except Exception as e:
        logger.warning(f"Error reading header from {file_path}: {e}")
    return None


def discover_datasets(
    base_dir: Path, header_id_key: str, max_datasets: int | None = None, n_proc: int = 1
) -> list[dict[str, Any]]:
    """
    Discover all datasets across date directories or in a flat directory structure.
    Returns a list of dataset information dictionaries.
    Uses parallel processing to speed up header extraction.
    """
    datasets = []

    # Check for subdirectories with FITS files
    date_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    # Also check if there are FITS files directly in the base directory
    base_fits_files = list(base_dir.glob("*.fits"))

    if date_dirs and not base_fits_files:
        # Traditional nested structure: date directories containing FITS files
        logger.info(f"Found {len(date_dirs)} date directories")
        dirs_to_process = date_dirs
    elif base_fits_files:
        # Flat structure: FITS files directly in base directory
        logger.info(f"Found {len(base_fits_files)} FITS files in base directory")
        dirs_to_process = [base_dir]
    else:
        # No FITS files found anywhere
        logger.info("No FITS files found in base directory or subdirectories")
        return datasets

    # Process each directory
    for current_dir in dirs_to_process:
        logger.info(f"Scanning directory: {current_dir}")

        # Get all FITS files in this directory
        if current_dir == base_dir and base_fits_files:
            # For base directory, use the files we already found
            fits_files = base_fits_files
        else:
            # For subdirectories, search recursively
            fits_files = list(current_dir.glob("**/*.fits"))

        if not fits_files:
            logger.info(f"No FITS files found in {current_dir}")
            continue

        # Extract unique IDs from headers in parallel
        unique_ids = set()
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
            # Use functools.partial to pass the header_id_key
            extract_func = functools.partial(extract_id_from_file, header_id_key=header_id_key)
            for header_id in tqdm(
                executor.map(extract_func, fits_files), total=len(fits_files), desc="Extracting IDs from headers"
            ):
                if header_id is not None:
                    unique_ids.add(header_id)

        # Add dataset info
        for id_value in unique_ids:
            datasets.append(
                {
                    "id": id_value,
                    "date_dir": current_dir,
                }
            )

        logger.info(f"Found {len(unique_ids)} unique datasets in {current_dir}")

        # Check if we've reached the maximum number of datasets
        if max_datasets and len(datasets) >= max_datasets:
            logger.info(f"Reached maximum number of datasets ({max_datasets})")
            break

    return datasets[:max_datasets] if max_datasets else datasets


def batch_process(
    base_dir: Path,
    output_dir: Path,
    header_id_key: str,
    n_proc: int = 1,
    max_datasets: int | None = None,
    detect: bool = False,
    skip_existing: bool = True,
    timeout: int | None = None,
) -> None:
    """
    Process multiple datasets in parallel.
    """
    start_time = time.time()

    # Discover all datasets
    datasets = discover_datasets(base_dir, header_id_key, max_datasets, n_proc)
    logger.info(f"Discovered {len(datasets)} datasets to process")

    # Create summary file
    summary_file = output_dir / "batch_summary.json"
    results = []

    # Use a dictionary to track processes and their start times
    processes = {}
    results_queue = multiprocessing.Queue()

    with tqdm(total=len(datasets), desc="Processing datasets") as pbar:
        # Start processes for each dataset, up to n_proc at a time
        active_processes = 0
        dataset_index = 0

        while dataset_index < len(datasets) or active_processes > 0:
            # Start new processes if we have capacity and datasets left
            while active_processes < n_proc and dataset_index < len(datasets):
                dataset = datasets[dataset_index]
                dataset_id = dataset["id"]
                dataset_dir = dataset["date_dir"]

                # Define a worker function that puts results in the queue
                def worker(dataset_dir, dataset_id, header_id_key, output_dir, detect, skip_existing, queue):
                    try:
                        result = process_single_dataset(
                            dataset_dir, dataset_id, header_id_key, output_dir, detect, skip_existing
                        )
                        queue.put(result)
                    except Exception as e:
                        queue.put(
                            {
                                "id": dataset_id,
                                "status": "error",
                                "error": str(e),
                                "time_taken": 0,
                                "data_path": str(dataset_dir),
                            }
                        )

                # Start a new process
                p = multiprocessing.Process(
                    target=worker,
                    args=(dataset_dir, dataset_id, header_id_key, output_dir, detect, skip_existing, results_queue),
                )
                p.start()
                processes[p.pid] = {
                    "process": p,
                    "start_time": time.time(),
                    "dataset_id": dataset_id,
                    "dataset_dir": dataset_dir,
                }
                active_processes += 1
                dataset_index += 1

            # Check for completed processes and timeouts
            for pid, process_info in list(processes.items()):
                p = process_info["process"]

                # Check if process has completed
                if not p.is_alive():
                    # Process completed, get result from queue if available
                    try:
                        if not results_queue.empty():
                            result = results_queue.get_nowait()
                            results.append(result)
                            with open(summary_file, "w") as f:
                                json.dump(results, f, indent=4)

                            if result["status"] == "success":
                                logger.info(
                                    f"Successfully processed dataset {result['id']} in {result['time_taken']:.2f} seconds"
                                )
                            elif result["status"] == "skipped":
                                logger.info(f"Skipped dataset {result['id']} - output already exists")
                            else:
                                logger.warning(f"Failed to process dataset {result['id']}: {result['error']}")
                    except Exception as e:
                        logger.exception(f"Error getting result for {process_info['dataset_id']}")
                        results.append(
                            {
                                "id": process_info["dataset_id"],
                                "status": "error",
                                "error": f"Error retrieving result: {e!s}",
                                "time_taken": time.time() - process_info["start_time"],
                                "data_path": str(process_info["dataset_dir"]),
                            }
                        )
                        with open(summary_file, "w") as f:
                            json.dump(results, f, indent=4)

                    # Clean up
                    del processes[pid]
                    active_processes -= 1
                    pbar.update(1)

                # Check for timeout
                elif timeout and (time.time() - process_info["start_time"] > timeout):
                    dataset_id = process_info["dataset_id"]
                    dataset_dir = process_info["dataset_dir"]
                    logger.warning(f"Dataset {dataset_id} timed out after {timeout} seconds")

                    # Terminate process
                    try:
                        # First try a gentle termination
                        p.terminate()
                        # Give it a moment to terminate
                        time.sleep(0.5)
                        # If still alive, force kill
                        if p.is_alive():
                            os.kill(pid, signal.SIGKILL)
                    except Exception:
                        logger.exception(f"Failed to terminate process {pid}")

                    # Record timeout
                    results.append(
                        {
                            "id": dataset_id,
                            "status": "timeout",
                            "error": f"Processing timed out after {timeout} seconds",
                            "time_taken": timeout,
                            "data_path": str(dataset_dir),
                        }
                    )
                    with open(summary_file, "w") as f:
                        json.dump(results, f, indent=4)

                    # Clean up
                    del processes[pid]
                    active_processes -= 1
                    pbar.update(1)

            # Small sleep to prevent CPU spinning
            time.sleep(0.1)

    # Final summary
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r["status"] == "success")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")

    logger.info("Batch processing complete:")
    logger.info(f"  Total datasets: {len(datasets)}")
    logger.info(f"  Successfully processed: {success_count}")
    logger.info(f"  Timed out: {timeout_count}")
    logger.info(f"  Failed: {len(datasets) - success_count - timeout_count}")
    logger.info(f"  Total time: {total_time:.2f} seconds")

    # Add summary stats to the summary file
    summary = {
        "total_datasets": len(datasets),
        "successful": success_count,
        "timed_out": timeout_count,
        "failed": len(datasets) - success_count - timeout_count,
        "total_time": total_time,
        "results": results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process multiple datasets with SENPAI")
    parser.add_argument(
        "-d", "--data_directory", help="Base directory containing date folders", type=str, required=True
    )
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str, required=True)
    parser.add_argument("-k", "--header_id_key", help="Header key for dataset ID", type=str, required=True)
    parser.add_argument("-n", "--n_proc", help="Number of concurrent processes", type=int, default=1)
    parser.add_argument("-m", "--max_datasets", help="Maximum number of datasets to process", type=int, default=None)
    parser.add_argument("-D", "--detect", help="Detect point sources", action="store_true", default=False)
    parser.add_argument(
        "--no-skip", help="Don't skip datasets with existing output", action="store_true", default=False
    )
    parser.add_argument(
        "-c",
        "--config",
        help=f"Config file, defaults to {LOCAL_APP_CONFIG_OVERRIDE}",
        type=str,
        default=LOCAL_APP_CONFIG_OVERRIDE,
    )
    parser.add_argument("-t", "--timeout", help="Timeout in seconds for each dataset", type=int, default=None)
    args = parser.parse_args()

    # Initialize paths
    base_dir = Path(args.data_directory)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize config
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
    require_astrometry_install()
    enforce_catalog()

    # Run batch processing
    batch_process(
        base_dir=base_dir,
        output_dir=output_dir,
        header_id_key=args.header_id_key,
        n_proc=args.n_proc,
        max_datasets=args.max_datasets,
        detect=args.detect,
        skip_existing=not args.no_skip,
        timeout=args.timeout,
    )
