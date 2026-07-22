r"""Dark frame utilities for creating and applying master dark corrections.

This module provides functions for:
1. Creating master dark frames from directories of dark frame FITS files
2. Applying dark subtraction to images
3. CLI interface for pre-generating master darks

Key differences from flats:
- Groups by BINNING, EXPTIME, and other relevant headers
- Combines using sigma-clipped mean (not median)
- No normalization (darks are subtracted, not divided)
- Exposure time scaling for different exposure times

CLI Usage Examples:
------------------

Basic master dark creation:
    python -m senpai.engine.utils.darks /path/to/darks/ -o master_dark.fits

Create master darks for different conditions (binning, exposure times):
    python -m senpai.engine.utils.darks /path/to/darks/ -o master_dark.fits \\
        --group-headers BINNING EXPTIME --create-all-groups

Custom quality filtering:
    python -m senpai.engine.utils.darks /path/to/darks/ -o master_dark.fits \\
        --max-counts 1000 --min-frames 5

Programmatic Usage:
------------------

Creating master darks:
    from senpai.engine.utils.darks import create_master_dark

    master_dark, header = create_master_dark(
        dark_directory="/path/to/darks/",
        output_path="/path/to/master_dark.fits",
        required_headers=["BINNING", "EXPTIME"]
    )

Applying dark corrections:
    from senpai.engine.utils.darks import apply_dark_subtraction

    corrected_image = apply_dark_subtraction(
        image=processed_fits_image,
        master_dark="/path/to/master_dark.fits"
    )
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip

from senpai.engine.models.images import ProcessedFitsImage, ProcessingStep


def create_master_dark(
    dark_directory: str | Path,
    output_path: str | Path | None = None,
    max_percentile_counts: float = 2000.0,
    percentile_threshold: float = 99.5,
    min_frames: int = 5,
    sigma: float = 3.0,
    maxiters: int = 5,
    required_headers: list[str] | None = None,
) -> tuple[np.ndarray, fits.Header]:
    """Create a master dark from a directory of dark frame FITS files.

    Args:
        dark_directory: Directory containing dark frame FITS files.
        output_path: Path to save the master dark. If None, returns the array
            and header only.
        max_percentile_counts: Maximum acceptable percentile value for the
            quality check (allows for hot pixels).
        percentile_threshold: Percentile to use for the quality check
            (e.g., 99.5 = 99.5th percentile).
        min_frames: Minimum number of frames required for combination.
        sigma: Sigma for sigma-clipped mean combination.
        maxiters: Maximum iterations for sigma clipping.
        required_headers: Header keywords that must be consistent across frames.

    Returns:
        A ``(master_dark, header)`` tuple: the master dark frame and the header
        from the first valid dark frame.

    Raises:
        ValueError: If no FITS files are found or fewer than ``min_frames``
            valid frames pass the quality check.
    """
    dark_directory = Path(dark_directory)

    # Find all FITS files
    fits_files = list(dark_directory.glob("*.fits")) + list(dark_directory.glob("*.fit"))
    if not fits_files:
        raise ValueError(f"No FITS files found in {dark_directory}")

    print(f"Found {len(fits_files)} FITS files in {dark_directory}")

    # Group frames by header consistency
    frame_groups = _group_frames_by_headers(fits_files, required_headers or ["BINNING", "EXPTIME"])

    if len(frame_groups) > 1:
        print(f"Found {len(frame_groups)} groups with different headers:")
        for i, (group_key, group_files) in enumerate(frame_groups.items()):
            header_desc = ", ".join(
                [f"{h}={v}" for h, v in zip(required_headers or ["BINNING", "EXPTIME"], group_key, strict=False)]
            )
            print(f"  Group {i + 1}: {header_desc} ({len(group_files)} files)")
        print("Processing the largest group. Consider running separately for each group.")
        print()

    # Use the largest group
    largest_group_files = max(frame_groups.values(), key=len)
    print(f"Processing {len(largest_group_files)} frames from the largest consistent group")

    # Load and validate frames
    valid_frames = []
    valid_headers = []

    for file_path in largest_group_files:
        try:
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float64)
                header = hdul[0].header

                # Check quality constraints using percentile instead of max
                frame_median = np.median(data)
                frame_percentile = np.percentile(data, percentile_threshold)
                frame_max = np.max(data)

                if frame_percentile < max_percentile_counts:
                    valid_frames.append(data)
                    valid_headers.append(header)
                    print(
                        f"✓ {file_path.name}: median={frame_median:.1f}, {percentile_threshold}th%={frame_percentile:.1f}, max={frame_max:.0f}"
                    )
                else:
                    print(
                        f"✗ {file_path.name}: median={frame_median:.1f}, {percentile_threshold}th%={frame_percentile:.1f}, max={frame_max:.0f} - rejected (percentile too high)"
                    )

        except Exception as e:
            print(f"✗ {file_path.name}: Error reading file - {e}")

    if len(valid_frames) < min_frames:
        raise ValueError(f"Need at least {min_frames} valid frames, found {len(valid_frames)}")

    print(f"Using {len(valid_frames)} valid frames for master dark")

    # Get image dimensions from first frame
    height, width = valid_frames[0].shape
    total_pixels = height * width

    # Estimate memory usage and decide on processing approach
    estimated_memory_gb = (len(valid_frames) * total_pixels * 8) / (1024**3)  # 8 bytes per float64
    print(f"Estimated memory usage: {estimated_memory_gb:.1f} GB")

    if estimated_memory_gb > 4.0:  # Use chunked processing for > 4GB
        print("Using memory-efficient chunked processing...")
        master_dark = _create_master_dark_chunked(valid_frames, sigma, maxiters)
    else:
        print("Using standard in-memory processing...")
        master_dark = _create_master_dark_standard(valid_frames, sigma, maxiters)

    # Create output header from first valid frame
    output_header = valid_headers[0].copy()
    output_header.add_history(f"Master dark created from {len(valid_frames)} frames")
    output_header.add_history(f"Sigma-clipped mean combination (sigma={sigma}, maxiters={maxiters})")
    output_header.add_history(f"Quality check: {percentile_threshold}th percentile < {max_percentile_counts}")

    # Add exposure time to header for scaling purposes
    exptime = output_header.get("EXPTIME", output_header.get("EXPOSURE", 1.0))
    output_header["EXPTIME"] = exptime
    output_header.add_history(f"Exposure time: {exptime} seconds")

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory if it doesn't exist
        hdu = fits.PrimaryHDU(data=master_dark.astype(np.float32), header=output_header)
        hdu.writeto(output_path, overwrite=True)
        print(f"Master dark saved to {output_path}")

    return master_dark, output_header


def apply_dark_subtraction(
    image: ProcessedFitsImage | np.ndarray,
    master_dark: str | Path | np.ndarray,
    dark_exposure_time: float | None = None,
    store_intermediates: bool = False,
) -> ProcessedFitsImage | np.ndarray:
    """Apply dark subtraction to an image.

    Args:
        image: Image to dark subtract.
        master_dark: Master dark frame. If a string/Path, it is loaded from file.
        dark_exposure_time: Exposure time of the master dark. If None, it is
            read from the dark's header (defaulting to 1.0).
        store_intermediates: Whether to store intermediate correction frames
            (only for ProcessedFitsImage).

    Returns:
        The dark-subtracted image, matching the input type (ProcessedFitsImage
        or np.ndarray).

    Raises:
        ValueError: If the image and dark frame shapes do not match.
    """
    # Load master dark if provided as file path. float32 throughout: master
    # darks are saved as float32 anyway, and the corrected frame's dtype is
    # inherited by the whole downstream pipeline (float64 doubles its cost).
    if isinstance(master_dark, (str, Path)):
        with fits.open(master_dark) as hdul:
            master_dark_data = hdul[0].data.astype(np.float32)
            dark_header = hdul[0].header
            if dark_exposure_time is None:
                dark_exposure_time = dark_header.get("EXPTIME", dark_header.get("EXPOSURE", 1.0))
    else:
        master_dark_data = master_dark.astype(np.float32)
        if dark_exposure_time is None:
            dark_exposure_time = 1.0  # Default if not provided

    # Clean the dark frame by removing hot pixels
    dark_median = np.median(master_dark_data)
    dark_std = np.std(master_dark_data)

    # Create a mask for hot pixels (pixels that are too high compared to the median)
    hot_pixel_threshold = dark_median + 5 * dark_std
    hot_pixel_mask = master_dark_data > hot_pixel_threshold

    # Replace hot pixels with the median value
    cleaned_dark = master_dark_data.copy()
    cleaned_dark[hot_pixel_mask] = dark_median

    n_hot_pixels = np.sum(hot_pixel_mask)
    if n_hot_pixels > 0:
        print(f"Cleaned {n_hot_pixels} hot pixels from dark frame (threshold: {hot_pixel_threshold:.1f} ADU)")
        print(
            f"Dark stats: median={dark_median:.1f}, std={dark_std:.1f}, max_before_cleaning={np.max(master_dark_data):.1f}"
        )

    # Handle ProcessedFitsImage objects
    if isinstance(image, ProcessedFitsImage):
        # Ensure shapes match
        if image.data.shape != cleaned_dark.shape:
            raise ValueError(f"Image shape {image.data.shape} doesn't match dark shape {cleaned_dark.shape}")

        # Get image exposure time for scaling
        image_exptime = image.header.get("EXPTIME", image.header.get("EXPOSURE", 1.0))

        # Scale dark if exposure times differ
        scaling_factor = 1.0  # Default to no scaling
        if abs(image_exptime - dark_exposure_time) > 0.1:  # Allow small tolerance
            scaling_factor = image_exptime / dark_exposure_time
            scaled_dark = cleaned_dark * scaling_factor
            print(f"Scaling dark by {scaling_factor:.3f} (image: {image_exptime}s, dark: {dark_exposure_time}s)")
        else:
            scaled_dark = cleaned_dark

        # Apply dark subtraction
        corrected_data = image.data.astype(np.float32) - scaled_dark

        # Store intermediate if requested
        if store_intermediates:
            if image.correction_frames is None:
                image.correction_frames = {}
            image.correction_frames[ProcessingStep.DARK_SUBTRACT] = scaled_dark

            if image.original_data is None:
                image.original_data = image.data.copy()

        # Update image
        image.data = corrected_data

        # Add processing metadata
        from senpai.engine.models.images import ProcessingMetadata

        # Build parameters dict, only including scaling if it's not 1.0
        parameters = {
            "master_dark_applied": True,
            "dark_exposure_time": dark_exposure_time,
            "image_exposure_time": image_exptime,
            "hot_pixels_cleaned": n_hot_pixels,
        }

        if scaling_factor != 1.0:
            parameters["exposure_time_scaling"] = scaling_factor

        dark_metadata = ProcessingMetadata(
            step_type=ProcessingStep.DARK_SUBTRACT,
            parameters=parameters,
        )
        image.processing_history.append(dark_metadata)

        return image

    else:
        # Handle numpy arrays
        if image.shape != cleaned_dark.shape:
            raise ValueError(f"Image shape {image.shape} doesn't match dark shape {cleaned_dark.shape}")

        # For numpy arrays, assume no scaling needed (user should handle this)
        return image.astype(np.float32) - cleaned_dark


def find_best_dark_for_exposure(
    dark_directory: str | Path,
    target_exptime: float,
    matching_headers: list[str] | None = None,
    max_exptime_ratio: float = 10.0,
) -> tuple[Path, float] | None:
    """Find the best dark frame for a given exposure time.

    Args:
        dark_directory: Directory containing dark frames.
        target_exptime: Target exposure time to match.
        matching_headers: Headers that must match (e.g., BINNING).
        max_exptime_ratio: Maximum ratio between target and dark exposure times.

    Returns:
        A ``(best_dark_path, dark_exptime)`` tuple for the closest matching
        dark frame, or None if the directory is missing or no frame is within
        ``max_exptime_ratio``.
    """
    if matching_headers is None:
        matching_headers = ["BINNING"]

    dark_directory = Path(dark_directory)
    if not dark_directory.exists():
        return None

    dark_files = list(dark_directory.glob("*.fits")) + list(dark_directory.glob("*.fit"))
    if not dark_files:
        return None

    best_match = None
    best_exptime = None
    best_ratio = float("inf")

    for dark_file in dark_files:
        try:
            with fits.open(dark_file) as hdul:
                header = hdul[0].header

                # Get exposure time
                dark_exptime = header.get("EXPTIME", header.get("EXPOSURE", 0.0))
                if dark_exptime <= 0:
                    continue

                # Check exposure time ratio
                ratio = max(target_exptime / dark_exptime, dark_exptime / target_exptime)
                if ratio > max_exptime_ratio:
                    continue

                # Check if this is a better match
                if ratio < best_ratio:
                    best_match = dark_file
                    best_exptime = dark_exptime
                    best_ratio = ratio

        except Exception as e:
            print(f"Warning: Could not read dark file {dark_file}: {e}")

    if best_match:
        print(
            f"Found dark frame: {best_match.name} ({best_exptime}s) for target {target_exptime}s (ratio: {best_ratio:.2f})"
        )
        return best_match, best_exptime

    return None


def _group_frames_by_headers(fits_files: list[Path], required_headers: list[str]) -> dict[tuple[str, ...], list[Path]]:
    """Group FITS files by consistent header values.

    Args:
        fits_files: List of FITS file paths.
        required_headers: Header keywords that must be consistent.

    Returns:
        A dict mapping group keys (tuples of header values) to lists of file
        paths.
    """
    if not required_headers:
        return {("all_files",): fits_files}

    groups = {}

    for file_path in fits_files:
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header

                # Create a key from the required header values
                key_values = []
                for header_key in required_headers:
                    value = header.get(header_key, "MISSING")
                    # For exposure time, round to avoid floating point issues
                    if header_key in ["EXPTIME", "EXPOSURE"] and isinstance(value, (int, float)):
                        value = round(float(value), 2)
                    key_values.append(str(value))

                group_key = tuple(key_values)

                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(file_path)

        except Exception as e:
            print(f"Warning: Could not read headers from {file_path}: {e}")

    return groups


def load_master_dark(file_path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Load a master dark from a FITS file.

    Args:
        file_path: Path to the master dark FITS file.

    Returns:
        A ``(master_dark, header)`` tuple: the master dark frame data and the
        FITS header.
    """
    with fits.open(file_path) as hdul:
        return hdul[0].data.astype(np.float64), hdul[0].header


def _create_descriptive_filename(
    base_output_path: str | Path,
    group_key: tuple[str, ...],
    header_names: list[str],
) -> Path:
    """Create a descriptive filename based on group characteristics.

    Args:
        base_output_path: Base output path (e.g., "master_dark.fits").
        group_key: Values from the header that define this group.
        header_names: Names of the headers corresponding to ``group_key``
            values.

    Returns:
        A descriptive filename incorporating the group characteristics.
    """
    base_path = Path(base_output_path)
    output_dir = base_path.parent
    output_stem = base_path.stem
    output_suffix = base_path.suffix

    # Create descriptive parts from header values
    descriptive_parts = []
    for header_name, value in zip(header_names, group_key, strict=False):
        # Clean up the value for filename use
        clean_value = str(value).replace("/", "-").replace("\\", "-").replace(" ", "")
        descriptive_parts.append(f"{header_name}-{clean_value}")

    # Create the descriptive filename
    descriptive_suffix = "_".join(descriptive_parts)
    descriptive_filename = f"{output_stem}_{descriptive_suffix}{output_suffix}"

    return output_dir / descriptive_filename


def main() -> None:
    """Run the CLI for creating master darks (or analyzing header variations)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Create master dark frames from a directory of dark frame FITS files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_dir", type=str, help="Directory containing dark frame FITS files")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for master dark FITS file (required unless using --analyze-headers)",
    )

    parser.add_argument(
        "--max-percentile-counts",
        type=float,
        default=2000.0,
        help="Maximum acceptable percentile value for quality check (allows for hot pixels)",
    )

    parser.add_argument(
        "--percentile-threshold",
        type=float,
        default=99.5,
        help="Percentile to use for quality check (e.g., 99.5 = 99.5th percentile)",
    )

    parser.add_argument("--min-frames", type=int, default=5, help="Minimum number of frames required for combination")

    parser.add_argument("--sigma", type=float, default=3.0, help="Sigma for sigma-clipped mean combination")

    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations for sigma clipping")

    parser.add_argument(
        "--group-headers",
        type=str,
        nargs="*",
        default=["BINNING", "EXPTIME"],
        help="FITS header keywords that must be consistent across frames",
    )

    parser.add_argument(
        "--create-all-groups",
        action="store_true",
        help="Create master darks for all header groups found, not just the largest",
    )

    parser.add_argument(
        "--analyze-headers",
        action="store_true",
        help="Analyze all files and show which headers vary (useful for determining --group-headers)",
    )

    args = parser.parse_args()

    # Analyze headers mode
    if args.analyze_headers:
        analyze_header_variations(args.input_dir)
        return

    # Require output path for master dark creation
    if not args.output:
        print("Error: --output is required when creating master darks")
        sys.exit(1)

    try:
        if args.create_all_groups:
            # Find all groups and create master darks for each
            dark_directory = Path(args.input_dir)
            fits_files = list(dark_directory.glob("*.fits")) + list(dark_directory.glob("*.fit"))

            if not fits_files:
                print(f"Error: No FITS files found in {dark_directory}")
                sys.exit(1)

            frame_groups = _group_frames_by_headers(fits_files, args.group_headers)

            if len(frame_groups) == 1:
                print("Only one group found, creating single master dark...")
            else:
                print(f"Found {len(frame_groups)} groups with different headers:")
                for i, (group_key, group_files) in enumerate(frame_groups.items()):
                    header_desc = ", ".join([f"{h}={v}" for h, v in zip(args.group_headers, group_key, strict=False)])
                    print(f"  Group {i + 1}: {header_desc} ({len(group_files)} files)")
                print()

            for i, (group_key, group_files) in enumerate(frame_groups.items()):
                if len(frame_groups) > 1:
                    # Multi-group naming with descriptive filenames
                    group_output = _create_descriptive_filename(args.output, group_key, args.group_headers)
                    header_desc = ", ".join([f"{h}={v}" for h, v in zip(args.group_headers, group_key, strict=False)])
                    print(f"\nProcessing group {i + 1}: {header_desc} ({len(group_files)} files) -> {group_output}")
                else:
                    group_output = args.output

                # Create master dark from this group's files
                master_dark, _header = _create_master_dark_from_files(
                    group_files,
                    group_output,
                    args.max_percentile_counts,
                    args.percentile_threshold,
                    args.min_frames,
                    args.sigma,
                    args.max_iterations,
                )

                print(f"✓ Master dark created: {group_output}")
        else:
            # Original single master dark creation
            master_dark, _header = create_master_dark(
                dark_directory=args.input_dir,
                output_path=args.output,
                max_percentile_counts=args.max_percentile_counts,
                percentile_threshold=args.percentile_threshold,
                min_frames=args.min_frames,
                sigma=args.sigma,
                maxiters=args.max_iterations,
                required_headers=args.group_headers,
            )

            print(f"✓ Master dark created successfully: {args.output}")
            print(f"  Shape: {master_dark.shape}")
            print(f"  Data range: {np.min(master_dark):.3f} - {np.max(master_dark):.3f}")

    except Exception as e:
        print(f"Error creating master dark: {e}")
        sys.exit(1)


def _create_master_dark_from_files(
    fits_files: list[Path],
    output_path: str | Path,
    max_percentile_counts: float,
    percentile_threshold: float,
    min_frames: int,
    sigma: float,
    maxiters: int,
) -> tuple[np.ndarray, fits.Header]:
    """Create a master dark from a specific list of FITS files.

    Args:
        fits_files: The dark-frame FITS files to combine.
        output_path: Path to write the resulting master dark to.
        max_percentile_counts: Maximum acceptable percentile value for the
            quality check.
        percentile_threshold: Percentile used for the quality check.
        min_frames: Minimum number of valid frames required.
        sigma: Sigma for sigma-clipped mean combination.
        maxiters: Maximum iterations for sigma clipping.

    Returns:
        A ``(master_dark, header)`` tuple: the master dark frame and its output
        header.

    Raises:
        ValueError: If fewer than ``min_frames`` valid frames pass the quality
            check.
    """
    # Load and validate frames
    valid_frames = []
    valid_headers = []

    for file_path in fits_files:
        try:
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float64)
                header = hdul[0].header

                # Check quality constraints using percentile instead of max
                frame_median = np.median(data)
                frame_percentile = np.percentile(data, percentile_threshold)
                frame_max = np.max(data)

                if frame_percentile < max_percentile_counts:
                    valid_frames.append(data)
                    valid_headers.append(header)
                    print(
                        f"  ✓ {file_path.name}: median={frame_median:.1f}, {percentile_threshold}th%={frame_percentile:.1f}, max={frame_max:.0f}"
                    )
                else:
                    print(
                        f"  ✗ {file_path.name}: median={frame_median:.1f}, {percentile_threshold}th%={frame_percentile:.1f}, max={frame_max:.0f} - rejected"
                    )

        except Exception as e:
            print(f"  ✗ {file_path.name}: Error reading file - {e}")

    if len(valid_frames) < min_frames:
        raise ValueError(f"Need at least {min_frames} valid frames, found {len(valid_frames)}")

    print(f"  Using {len(valid_frames)} valid frames for master dark")

    # Get image dimensions from first frame
    height, width = valid_frames[0].shape
    total_pixels = height * width

    # Estimate memory usage and decide on processing approach
    estimated_memory_gb = (len(valid_frames) * total_pixels * 8) / (1024**3)  # 8 bytes per float64
    print(f"  Estimated memory usage: {estimated_memory_gb:.1f} GB")

    if estimated_memory_gb > 4.0:  # Use chunked processing for > 4GB
        print("  Using memory-efficient chunked processing...")
        master_dark = _create_master_dark_chunked(valid_frames, sigma, maxiters)
    else:
        print("  Using standard in-memory processing...")
        master_dark = _create_master_dark_standard(valid_frames, sigma, maxiters)

    # Create output header from first valid frame
    output_header = valid_headers[0].copy()
    output_header.add_history(f"Master dark created from {len(valid_frames)} frames")
    output_header.add_history(f"Sigma-clipped mean combination (sigma={sigma}, maxiters={maxiters})")
    output_header.add_history(f"Quality check: {percentile_threshold}th percentile < {max_percentile_counts}")

    # Add exposure time to header for scaling purposes
    exptime = output_header.get("EXPTIME", output_header.get("EXPOSURE", 1.0))
    output_header["EXPTIME"] = exptime
    output_header.add_history(f"Exposure time: {exptime} seconds")

    # Save master dark
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory if it doesn't exist
    hdu = fits.PrimaryHDU(data=master_dark.astype(np.float32), header=output_header)
    hdu.writeto(output_path, overwrite=True)

    return master_dark, output_header


def _create_master_dark_standard(
    valid_frames: list[np.ndarray],
    sigma: float,
    maxiters: int,
) -> np.ndarray:
    """Standard in-memory processing for smaller datasets."""
    # Stack frames for sigma-clipped mean combination
    frame_stack = np.stack(valid_frames, axis=0)

    # Perform sigma-clipped mean combination (not median like flats)
    sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
    clipped_stack = sigma_clip(frame_stack, axis=0)

    # Convert MaskedArray to regular array and compute mean
    if hasattr(clipped_stack, "filled"):
        # It's a MaskedArray, convert to regular array
        master_dark = np.mean(clipped_stack.filled(np.nan), axis=0)
        # Replace any NaN values with the median of surrounding pixels
        if np.any(np.isnan(master_dark)):
            master_dark = np.where(np.isnan(master_dark), np.nanmedian(frame_stack, axis=0), master_dark)
    else:
        # It's already a regular array
        master_dark = np.mean(clipped_stack, axis=0)

    return master_dark


def _create_master_dark_chunked(
    valid_frames: list[np.ndarray],
    sigma: float,
    maxiters: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Memory-efficient chunked processing for larger datasets."""
    height, width = valid_frames[0].shape
    master_dark = np.zeros((height, width), dtype=np.float64)

    # Process in chunks to reduce memory usage
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk_height = end_row - start_row

        print(f"    Processing rows {start_row}-{end_row - 1} ({chunk_height} rows)")

        # Extract chunk from each frame
        chunk_stack = np.zeros((len(valid_frames), chunk_height, width), dtype=np.float64)
        for i, frame in enumerate(valid_frames):
            chunk_stack[i] = frame[start_row:end_row]

        # Perform sigma-clipped mean combination on this chunk
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        clipped_chunk = sigma_clip(chunk_stack, axis=0)

        # Convert MaskedArray to regular array and compute mean
        if hasattr(clipped_chunk, "filled"):
            # It's a MaskedArray, convert to regular array
            chunk_result = np.mean(clipped_chunk.filled(np.nan), axis=0)
            # Replace any NaN values with the median
            if np.any(np.isnan(chunk_result)):
                chunk_result = np.where(np.isnan(chunk_result), np.nanmedian(chunk_stack, axis=0), chunk_result)
        else:
            # It's already a regular array
            chunk_result = np.mean(clipped_chunk, axis=0)

        # Store result
        master_dark[start_row:end_row] = chunk_result

        # Explicitly delete chunk arrays to free memory
        del chunk_stack, clipped_chunk, chunk_result

    return master_dark


def analyze_header_variations(directory: str | Path) -> None:
    """Analyze all FITS files in a directory and report which headers vary.

    This helps users determine which headers they should use for grouping.

    Args:
        directory: Directory containing FITS files to analyze.
    """
    directory = Path(directory)
    fits_files = list(directory.glob("*.fits")) + list(directory.glob("*.fit"))

    if not fits_files:
        print(f"No FITS files found in {directory}")
        return

    print(f"Analyzing {len(fits_files)} FITS files in {directory}")
    print("=" * 60)

    # Collect all headers from all files
    all_headers = {}  # header_key -> set of values
    file_count = 0

    for file_path in fits_files:
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                file_count += 1

                for key in header:
                    # Skip standard FITS keywords that don't affect calibration
                    if key in [
                        "SIMPLE",
                        "BITPIX",
                        "NAXIS",
                        "NAXIS1",
                        "NAXIS2",
                        "EXTEND",
                        "COMMENT",
                        "HISTORY",
                        "",
                        "CHECKSUM",
                        "DATASUM",
                    ]:
                        continue

                    value = header.get(key)

                    # Convert to string for comparison, handle special cases
                    if isinstance(value, (int, float)):
                        if key in ["EXPTIME", "EXPOSURE"] and isinstance(value, (int, float)):
                            # Round exposure times to avoid floating point differences
                            value_str = f"{float(value):.3f}"
                        else:
                            value_str = str(value)
                    else:
                        value_str = str(value).strip() if value is not None else "None"

                    if key not in all_headers:
                        all_headers[key] = set()
                    all_headers[key].add(value_str)

        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    print(f"Successfully analyzed {file_count} files")
    print()

    # Separate headers into constant and varying
    constant_headers = {}
    varying_headers = {}

    for key, values in all_headers.items():
        if len(values) == 1:
            constant_headers[key] = next(iter(values))
        else:
            varying_headers[key] = sorted(values)

    # Print results
    print("HEADERS THAT VARY (candidates for --group-headers):")
    print("-" * 50)
    if varying_headers:
        for key, values in sorted(varying_headers.items()):
            print(f"{key:15s}: {len(values)} unique values")
            for value in values[:10]:  # Show first 10 values
                print(f"                 {value}")
            if len(values) > 10:
                print(f"                 ... and {len(values) - 10} more")
            print()
    else:
        print("No varying headers found - all files have identical headers")

    print()
    print("CONSTANT HEADERS (same across all files):")
    print("-" * 40)
    if constant_headers:
        for key, value in sorted(constant_headers.items()):
            print(f"{key:15s}: {value}")
    else:
        print("No constant headers found")

    print()
    print("SUGGESTED --group-headers:")
    print("-" * 25)

    # Suggest common grouping headers
    suggested = []
    common_grouping_headers = [
        "BINNING",
        "XBINNING",
        "BIN",
        "CCDSUM",
        "EXPTIME",
        "EXPOSURE",
        "FILTER",
        "FILTNAM",
        "FILTER1",
        "INSTRUME",
        "DETECTOR",
        "READMODE",
    ]

    for header in common_grouping_headers:
        if header in varying_headers:
            suggested.append(header)

    if suggested:
        print("Based on common calibration grouping, try:")
        print(f"  --group-headers {' '.join(suggested)}")
        print()
        print("For dark frames specifically, typically use:")
        print("  --group-headers BINNING EXPTIME")
        print("For flat frames, typically use:")
        print("  --group-headers BINNING FILTER")
    else:
        print("No common grouping headers found in varying headers.")
        if varying_headers:
            # Suggest any varying headers
            suggested_any = list(varying_headers.keys())[:3]  # First 3 varying headers
            print(f"You might try: --group-headers {' '.join(suggested_any)}")
        else:
            print("All files appear to have identical headers.")

    print()
    print("Run with --group-headers followed by any of the varying headers above")
    print("to group frames by those parameters.")


if __name__ == "__main__":
    main()
