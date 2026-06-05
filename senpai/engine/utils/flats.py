"""
Flat field utilities for creating and applying master flat corrections.

This module provides functions for:
1. Creating master flat fields from directories of flat frame FITS files
2. Applying flat field corrections to images
3. CLI interface for pre-generating master flats

CLI Usage Examples:
------------------

Basic master flat creation:
    python -m senpai.engine.utils.flats /path/to/flats/ -o master_flat.fits

Create master flats for different conditions (binning, filters):
    python -m senpai.engine.utils.flats /path/to/flats/ -o master_flat.fits \\
        --group-headers BINNING FILTER --create-all-groups

Custom quality filtering:
    python -m senpai.engine.utils.flats /path/to/flats/ -o master_flat.fits \\
        --min-median 30000 --max-median 60000 --max-counts 65000 --max-percentile 99.5

Handle hot pixels by using percentile-based checks:
    python -m senpai.engine.utils.flats /path/to/flats/ -o master_flat.fits \\
        --max-percentile 99.9 --max-counts 60000

Programmatic Usage:
------------------

Creating master flats:
    from senpai.engine.utils.flats import create_master_flat
    
    master_flat, header = create_master_flat(
        flat_directory="/path/to/flats/",
        output_path="/path/to/master_flat.fits",
        required_headers=["BINNING", "FILTER"]
    )

Applying flat corrections:
    from senpai.engine.utils.preprocessing import apply_flat_field
    
    corrected_image = apply_flat_field(
        image=processed_fits_image,
        master_flat="/path/to/master_flat.fits"
    )

Auto-applying calibrations based on config:
    from senpai.engine.utils.preprocessing import auto_apply_calibrations
    
    calibrated_image = auto_apply_calibrations(processed_fits_image)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip

from senpai.engine.models.images import ProcessedFitsImage, ProcessingStep


def create_master_flat(
    flat_directory: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    min_median: float = 40000.0,
    max_median: float = 50000.0,
    max_counts: float = 50000.0,
    max_percentile: float = 99.9,
    sigma: float = 3.0,
    maxiters: int = 5,
    required_headers: Optional[List[str]] = None,
    dark_directory: Optional[Union[str, Path]] = None,
    max_dark_exptime_ratio: float = 10.0,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Create a master flat from a directory of flat field FITS files.

    Parameters
    ----------
    flat_directory : str or Path
        Directory containing flat field FITS files
    output_path : str or Path, optional
        Path to save the master flat. If None, returns array and header only
    min_median : float
        Minimum acceptable median value for flat frames
    max_median : float
        Maximum acceptable median value for flat frames
    max_counts : float
        Maximum acceptable pixel value for linearity check (applied to percentile, not max)
    max_percentile : float
        Percentile to use for linearity check instead of maximum (default 99.9)
    sigma : float
        Sigma for sigma-clipped median combination
    maxiters : int
        Maximum iterations for sigma clipping
    required_headers : list of str, optional
        Header keywords that must be consistent across frames
    dark_directory : str or Path, optional
        Directory containing dark frames for dark subtraction of flats
    max_dark_exptime_ratio : float
        Maximum ratio between flat and dark exposure times for scaling

    Returns
    -------
    master_flat : np.ndarray
        The master flat field normalized between 0 and 1
    header : fits.Header
        Header from the first valid flat frame
    """
    flat_directory = Path(flat_directory)

    # Find all FITS files
    fits_files = list(flat_directory.glob("*.fits")) + list(flat_directory.glob("*.fit"))
    if not fits_files:
        raise ValueError(f"No FITS files found in {flat_directory}")

    print(f"Found {len(fits_files)} FITS files in {flat_directory}")

    # Group frames by header consistency
    frame_groups = _group_frames_by_headers(fits_files, required_headers or [])

    if len(frame_groups) > 1:
        print(f"Found {len(frame_groups)} groups with different headers:")
        for i, (group_key, group_files) in enumerate(frame_groups.items()):
            header_desc = ", ".join([f"{h}={v}" for h, v in zip(required_headers or [], group_key, strict=False)])
            print(f"  Group {i + 1}: {header_desc} ({len(group_files)} files)")
        print("Processing the largest group. Consider running separately for each group.")
        print()

    # Use the largest group
    largest_group_files = max(frame_groups.values(), key=len)
    print(f"Processing {len(largest_group_files)} frames from the largest consistent group")

    # Load and validate frames
    valid_frames = []
    valid_headers = []
    dark_subtracted_count = 0

    for file_path in largest_group_files:
        try:
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float64)
                header = hdul[0].header

                # Apply dark subtraction if dark directory provided
                if dark_directory is not None:
                    flat_exptime = header.get("EXPTIME", header.get("EXPOSURE", 1.0))
                    dark_result = _find_and_apply_dark_to_flat(
                        data, header, dark_directory, flat_exptime, max_dark_exptime_ratio
                    )
                    if dark_result is not None:
                        data = dark_result
                        dark_subtracted_count += 1

                # Check linearity constraints using percentile instead of max to handle hot pixels
                frame_median = np.median(data)
                frame_percentile = np.percentile(data, max_percentile)
                frame_max = np.max(data)

                if min_median <= frame_median <= max_median and frame_percentile < max_counts:
                    valid_frames.append(data)
                    valid_headers.append(header)
                    print(
                        f"✓ {file_path.name}: median={frame_median:.1f}, {max_percentile:.1f}%ile={frame_percentile:.1f}, max={frame_max:.1f}"
                    )
                else:
                    print(
                        f"✗ {file_path.name}: median={frame_median:.1f}, {max_percentile:.1f}%ile={frame_percentile:.1f}, max={frame_max:.1f} - rejected"
                    )

        except Exception as e:
            print(f"✗ {file_path.name}: Error reading file - {e}")

    if len(valid_frames) < 3:
        raise ValueError(f"Need at least 3 valid frames, found {len(valid_frames)}")

    print(f"Using {len(valid_frames)} valid frames for master flat")
    if dark_directory is not None:
        print(f"Dark subtracted {dark_subtracted_count}/{len(valid_frames)} frames")

    # Get image dimensions from first frame
    height, width = valid_frames[0].shape
    total_pixels = height * width

    # Estimate memory usage and decide on processing approach
    estimated_memory_gb = (len(valid_frames) * total_pixels * 8) / (1024**3)  # 8 bytes per float64
    print(f"Estimated memory usage: {estimated_memory_gb:.1f} GB")

    if estimated_memory_gb > 4.0:  # Use chunked processing for > 4GB
        print("Using memory-efficient chunked processing...")
        master_flat = _create_master_flat_chunked(valid_frames, sigma, maxiters)
    else:
        print("Using standard in-memory processing...")
        master_flat = _create_master_flat_standard(valid_frames, sigma, maxiters)

    # Create output header from first valid frame
    output_header = valid_headers[0].copy()
    output_header.add_history(f"Master flat created from {len(valid_frames)} frames")
    output_header.add_history(f"Sigma-clipped median combination (sigma={sigma}, maxiters={maxiters})")
    output_header.add_history("Normalized to 0-1 range")
    if dark_directory is not None:
        output_header.add_history(f"Dark subtracted {dark_subtracted_count}/{len(valid_frames)} frames")

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory if it doesn't exist
        hdu = fits.PrimaryHDU(data=master_flat.astype(np.float32), header=output_header)
        hdu.writeto(output_path, overwrite=True)
        print(f"Master flat saved to {output_path}")

    return master_flat, output_header


def apply_flat_field(
    image: Union[ProcessedFitsImage, np.ndarray],
    master_flat: Union[str, Path, np.ndarray],
    store_intermediates: bool = False,
) -> Union[ProcessedFitsImage, np.ndarray]:
    """
    Apply flat field correction to an image.

    Parameters
    ----------
    image : ProcessedFitsImage or np.ndarray
        Image to flat field correct
    master_flat : str, Path, or np.ndarray
        Master flat field. If string/Path, will load from file
    store_intermediates : bool
        Whether to store intermediate correction frames (only for ProcessedFitsImage)

    Returns
    -------
    corrected_image : ProcessedFitsImage or np.ndarray
        Flat field corrected image
    """
    # Load master flat if provided as file path
    if isinstance(master_flat, (str, Path)):
        with fits.open(master_flat) as hdul:
            master_flat = hdul[0].data.astype(np.float64)
    else:
        master_flat = master_flat.astype(np.float64)

    # Handle ProcessedFitsImage objects
    if isinstance(image, ProcessedFitsImage):
        # Ensure shapes match
        if image.data.shape != master_flat.shape:
            raise ValueError(f"Image shape {image.data.shape} doesn't match flat shape {master_flat.shape}")

        # Avoid division by zero/very small values
        safe_flat = np.where(master_flat < 0.1, 1.0, master_flat)

        # Apply flat field correction
        corrected_data = image.data.astype(np.float64) / safe_flat

        # Store intermediate if requested
        if store_intermediates:
            if image.correction_frames is None:
                image.correction_frames = {}
            image.correction_frames[ProcessingStep.FLAT_DIVIDE] = master_flat

            if image.original_data is None:
                image.original_data = image.data.copy()

        # Update image
        image.data = corrected_data

        # Add processing metadata
        from senpai.engine.models.images import ProcessingMetadata

        flat_metadata = ProcessingMetadata(
            step_type=ProcessingStep.FLAT_DIVIDE, parameters={"master_flat_applied": True}
        )
        image.processing_history.append(flat_metadata)

        return image

    # Handle raw numpy arrays
    else:
        if image.shape != master_flat.shape:
            raise ValueError(f"Image shape {image.shape} doesn't match flat shape {master_flat.shape}")

        # Avoid division by zero/very small values
        safe_flat = np.where(master_flat < 0.1, 1.0, master_flat)

        return image.astype(np.float64) / safe_flat


def _group_frames_by_headers(fits_files: List[Path], required_headers: List[str]) -> Dict[Tuple[str, ...], List[Path]]:
    """
    Group FITS files by consistent header values.

    Parameters
    ----------
    fits_files : list of Path
        List of FITS file paths
    required_headers : list of str
        Header keywords that must be consistent

    Returns
    -------
    groups : dict
        Dictionary mapping group keys (tuples of header values) to lists of file paths
    """
    if not required_headers:
        # If no headers specified, return all files as one group
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
                    key_values.append(str(value))

                group_key = tuple(key_values)

                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(file_path)

        except Exception as e:
            print(f"Warning: Could not read headers from {file_path}: {e}")

    return groups


def _find_and_apply_dark_to_flat(
    flat_data: np.ndarray,
    flat_header: fits.Header,
    dark_directory: Union[str, Path],
    flat_exptime: float,
    max_exptime_ratio: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Find and apply dark subtraction to a flat frame.

    Parameters
    ----------
    flat_data : np.ndarray
        Flat frame data
    flat_header : fits.Header
        Flat frame header
    dark_directory : str or Path
        Directory containing dark frames
    flat_exptime : float
        Exposure time of the flat frame
    max_exptime_ratio : float
        Maximum ratio between flat and dark exposure times

    Returns
    -------
    dark_subtracted_flat : np.ndarray or None
        Dark-subtracted flat data, or None if no suitable dark found
    """
    from senpai.engine.utils.darks import find_best_dark_for_exposure

    # Try to find a suitable dark frame
    dark_result = find_best_dark_for_exposure(
        dark_directory=dark_directory,
        target_exptime=flat_exptime,
        matching_headers=["BINNING"],  # Only match binning for flats
        max_exptime_ratio=max_exptime_ratio,
    )

    if dark_result is None:
        print(f"    No suitable dark found for {flat_exptime}s flat")
        return None

    dark_path, dark_exptime = dark_result

    try:
        # Load the dark frame
        with fits.open(dark_path) as hdul:
            dark_data = hdul[0].data.astype(np.float64)

        # Check if shapes match
        if flat_data.shape != dark_data.shape:
            print(f"    Dark shape mismatch: flat {flat_data.shape} vs dark {dark_data.shape}")
            return None

        # Scale dark if exposure times differ
        if abs(flat_exptime - dark_exptime) > 0.1:
            scaling_factor = flat_exptime / dark_exptime
            scaled_dark = dark_data * scaling_factor
            print(f"    Scaled dark by {scaling_factor:.3f} for {flat_exptime}s flat")
        else:
            scaled_dark = dark_data
            print(f"    Applied dark ({dark_exptime}s) to flat ({flat_exptime}s)")

        # Subtract dark from flat
        return flat_data - scaled_dark

    except Exception as e:
        print(f"    Error applying dark: {e}")
        return None


def load_master_flat(file_path: Union[str, Path]) -> Tuple[np.ndarray, fits.Header]:
    """
    Load a master flat from a FITS file.

    Parameters
    ----------
    file_path : str or Path
        Path to the master flat FITS file

    Returns
    -------
    master_flat : np.ndarray
        The master flat field data
    header : fits.Header
        The FITS header
    """
    with fits.open(file_path) as hdul:
        return hdul[0].data.astype(np.float64), hdul[0].header


def _create_descriptive_filename(
    base_output_path: Union[str, Path],
    group_key: Tuple[str, ...],
    header_names: List[str],
) -> Path:
    """
    Create a descriptive filename based on group characteristics.

    Parameters
    ----------
    base_output_path : str or Path
        Base output path (e.g., "master_flat.fits")
    group_key : tuple of str
        Values from the header that define this group
    header_names : list of str
        Names of the headers corresponding to group_key values

    Returns
    -------
    Path
        Descriptive filename incorporating group characteristics
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


def _create_master_flat_standard(
    valid_frames: List[np.ndarray],
    sigma: float,
    maxiters: int,
) -> np.ndarray:
    """Standard in-memory processing for smaller datasets."""
    # Normalize each frame by its median
    normalized_frames = []
    for frame in valid_frames:
        frame_median = np.median(frame)
        normalized_frame = frame / frame_median
        normalized_frames.append(normalized_frame)

    # Stack frames for sigma-clipped median combination
    frame_stack = np.stack(normalized_frames, axis=0)

    # Perform sigma-clipped median combination
    sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
    clipped_stack = sigma_clip(frame_stack, axis=0)

    # Convert MaskedArray to regular array and compute median
    if hasattr(clipped_stack, "filled"):
        # It's a MaskedArray, convert to regular array
        master_flat = np.median(clipped_stack.filled(np.nan), axis=0)
        # Replace any NaN values with the median of surrounding pixels
        if np.any(np.isnan(master_flat)):
            master_flat = np.where(np.isnan(master_flat), np.nanmedian(frame_stack, axis=0), master_flat)
    else:
        # It's already a regular array
        master_flat = np.median(clipped_stack, axis=0)

    # Normalize to 0-1 range
    master_flat = (master_flat - np.min(master_flat)) / (np.max(master_flat) - np.min(master_flat))

    return master_flat


def _create_master_flat_chunked(
    valid_frames: List[np.ndarray],
    sigma: float,
    maxiters: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Memory-efficient chunked processing for larger datasets."""
    height, width = valid_frames[0].shape
    master_flat = np.zeros((height, width), dtype=np.float64)

    # Normalize each frame by its median first
    normalized_frames = []
    for frame in valid_frames:
        frame_median = np.median(frame)
        normalized_frame = frame / frame_median
        normalized_frames.append(normalized_frame)

    # Process in chunks to reduce memory usage
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk_height = end_row - start_row

        print(f"    Processing rows {start_row}-{end_row - 1} ({chunk_height} rows)")

        # Extract chunk from each normalized frame
        chunk_stack = np.zeros((len(normalized_frames), chunk_height, width), dtype=np.float64)
        for i, frame in enumerate(normalized_frames):
            chunk_stack[i] = frame[start_row:end_row]

        # Perform sigma-clipped median combination on this chunk
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        clipped_chunk = sigma_clip(chunk_stack, axis=0)

        # Convert MaskedArray to regular array and compute median
        if hasattr(clipped_chunk, "filled"):
            # It's a MaskedArray, convert to regular array
            chunk_result = np.median(clipped_chunk.filled(np.nan), axis=0)
            # Replace any NaN values with the median
            if np.any(np.isnan(chunk_result)):
                chunk_result = np.where(np.isnan(chunk_result), np.nanmedian(chunk_stack, axis=0), chunk_result)
        else:
            # It's already a regular array
            chunk_result = np.median(clipped_chunk, axis=0)

        # Store result
        master_flat[start_row:end_row] = chunk_result

        # Explicitly delete chunk arrays to free memory
        del chunk_stack, clipped_chunk, chunk_result

    # Normalize to 0-1 range
    master_flat = (master_flat - np.min(master_flat)) / (np.max(master_flat) - np.min(master_flat))

    return master_flat


def main():
    """CLI interface for creating master flat fields."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Create master flat fields from a directory of flat field FITS files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_dir", type=str, help="Directory containing flat field FITS files")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for master flat FITS file (required unless using --analyze-headers)",
    )

    parser.add_argument(
        "--min-median-counts",
        type=float,
        default=40000.0,
        help="Minimum median counts for linearity check",
    )

    parser.add_argument(
        "--max-median-counts",
        type=float,
        default=50000.0,
        help="Maximum median counts for linearity check",
    )

    parser.add_argument(
        "--max-counts",
        type=float,
        default=50000.0,
        help="Maximum counts for linearity check (applied to percentile, not max)",
    )

    parser.add_argument(
        "--max-percentile",
        type=float,
        default=99.9,
        help="Percentile to use for linearity check instead of maximum (handles hot pixels)",
    )

    parser.add_argument("--min-frames", type=int, default=5, help="Minimum number of frames required for combination")

    parser.add_argument("--sigma", type=float, default=3.0, help="Sigma for sigma-clipped median combination")

    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations for sigma clipping")

    parser.add_argument(
        "--group-headers",
        type=str,
        nargs="*",
        default=["BINNING", "FILTER"],
        help="FITS header keywords that must be consistent across frames",
    )

    parser.add_argument(
        "--create-all-groups",
        action="store_true",
        help="Create master flats for all header groups found, not just the largest",
    )

    parser.add_argument(
        "--dark-directory",
        type=str,
        help="Directory containing dark frames for subtraction from flats",
    )

    parser.add_argument(
        "--max-dark-exptime-ratio",
        type=float,
        default=10.0,
        help="Maximum ratio between flat and dark exposure times for scaling",
    )

    parser.add_argument(
        "--analyze-headers",
        action="store_true",
        help="Analyze all files and show which headers vary (useful for determining --group-headers)",
    )

    args = parser.parse_args()

    # Analyze headers mode
    if args.analyze_headers:
        from senpai.engine.utils.darks import analyze_header_variations

        analyze_header_variations(args.input_dir)
        return

    # Require output path for master flat creation
    if not args.output:
        print("Error: --output is required when creating master flats")
        sys.exit(1)

    try:
        if args.create_all_groups:
            # Find all groups and create master flats for each
            flat_directory = Path(args.input_dir)
            fits_files = list(flat_directory.glob("*.fits")) + list(flat_directory.glob("*.fit"))

            if not fits_files:
                print(f"Error: No FITS files found in {flat_directory}")
                sys.exit(1)

            frame_groups = _group_frames_by_headers(fits_files, args.group_headers)

            if len(frame_groups) == 1:
                print("Only one group found, creating single master flat...")
                output_path = args.output
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

                # Create master flat from this group's files
                master_flat, header = _create_master_flat_from_files(
                    group_files,
                    group_output,
                    args.min_median_counts,
                    args.max_median_counts,
                    args.max_counts,
                    args.max_percentile,
                    args.min_frames,
                    args.sigma,
                    args.max_iterations,
                    args.dark_directory,
                    args.max_dark_exptime_ratio,
                )

                print(f"✓ Master flat created: {group_output}")
        else:
            # Original single master flat creation
            master_flat, header = create_master_flat(
                flat_directory=args.input_dir,
                output_path=args.output,
                min_median=args.min_median_counts,
                max_median=args.max_median_counts,
                max_counts=args.max_counts,
                max_percentile=args.max_percentile,
                sigma=args.sigma,
                maxiters=args.max_iterations,
                required_headers=args.group_headers,
                dark_directory=args.dark_directory,
                max_dark_exptime_ratio=args.max_dark_exptime_ratio,
            )

            print(f"✓ Master flat created successfully: {args.output}")
            print(f"  Shape: {master_flat.shape}")
            print(f"  Data range: {np.min(master_flat):.3f} - {np.max(master_flat):.3f}")

    except Exception as e:
        print(f"Error creating master flat: {e}")
        sys.exit(1)


def _create_master_flat_from_files(
    fits_files: List[Path],
    output_path: Union[str, Path],
    min_median: float,
    max_median: float,
    max_counts: float,
    max_percentile: float,
    min_frames: int,
    sigma: float,
    maxiters: int,
    dark_directory: Optional[Union[str, Path]] = None,
    max_dark_exptime_ratio: float = 10.0,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Helper function to create master flat from a specific list of files.
    """
    # Load and validate frames
    valid_frames = []
    valid_headers = []
    dark_subtracted_count = 0

    for file_path in fits_files:
        try:
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float64)
                header = hdul[0].header

                # Apply dark subtraction if dark directory provided
                if dark_directory is not None:
                    flat_exptime = header.get("EXPTIME", header.get("EXPOSURE", 1.0))
                    dark_result = _find_and_apply_dark_to_flat(
                        data, header, dark_directory, flat_exptime, max_dark_exptime_ratio
                    )
                    if dark_result is not None:
                        data = dark_result
                        dark_subtracted_count += 1

                # Check linearity constraints using percentile instead of max to handle hot pixels
                frame_median = np.median(data)
                frame_percentile = np.percentile(data, max_percentile)
                frame_max = np.max(data)

                if min_median <= frame_median <= max_median and frame_percentile < max_counts:
                    valid_frames.append(data)
                    valid_headers.append(header)
                    print(
                        f"  ✓ {file_path.name}: median={frame_median:.1f}, {max_percentile:.1f}%ile={frame_percentile:.1f}, max={frame_max:.1f}"
                    )
                else:
                    print(
                        f"  ✗ {file_path.name}: median={frame_median:.1f}, {max_percentile:.1f}%ile={frame_percentile:.1f}, max={frame_max:.1f} - rejected"
                    )

        except Exception as e:
            print(f"  ✗ {file_path.name}: Error reading file - {e}")

    if len(valid_frames) < min_frames:
        raise ValueError(f"Need at least {min_frames} valid frames, found {len(valid_frames)}")

    print(f"  Using {len(valid_frames)} valid frames for master flat")
    if dark_directory is not None:
        print(f"  Dark subtracted {dark_subtracted_count}/{len(valid_frames)} frames")

    # Get image dimensions from first frame
    height, width = valid_frames[0].shape
    total_pixels = height * width

    # Estimate memory usage and decide on processing approach
    estimated_memory_gb = (len(valid_frames) * total_pixels * 8) / (1024**3)  # 8 bytes per float64
    print(f"Estimated memory usage: {estimated_memory_gb:.1f} GB")

    if estimated_memory_gb > 4.0:  # Use chunked processing for > 4GB
        print("Using memory-efficient chunked processing...")
        master_flat = _create_master_flat_chunked(valid_frames, sigma, maxiters)
    else:
        print("Using standard in-memory processing...")
        master_flat = _create_master_flat_standard(valid_frames, sigma, maxiters)

    # Create output header from first valid frame
    output_header = valid_headers[0].copy()
    output_header.add_history(f"Master flat created from {len(valid_frames)} frames")
    output_header.add_history(f"Sigma-clipped median combination (sigma={sigma}, maxiters={maxiters})")
    output_header.add_history("Normalized to 0-1 range")
    if dark_directory is not None:
        output_header.add_history(f"Dark subtracted {dark_subtracted_count}/{len(valid_frames)} frames")

    # Save master flat
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory if it doesn't exist
    hdu = fits.PrimaryHDU(data=master_flat.astype(np.float32), header=output_header)
    hdu.writeto(output_path, overwrite=True)

    return master_flat, output_header


if __name__ == "__main__":
    main()
