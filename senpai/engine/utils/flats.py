r"""Flat field utilities for creating and applying master flat corrections.

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip

from senpai.engine.models.images import ProcessedFitsImage, ProcessingStep


@dataclass
class _FlatSource:
    """A validated flat frame, referenced by path.

    Data is re-read lazily chunk-by-chunk during combination so full frames
    never co-reside in memory (a night of unbinned 8120^2 twilight flats is
    ~50 GB).
    """

    path: Path
    median: float  # frame median (after dark subtraction); normalizes the frame
    dark_path: Path | None = None
    dark_scale: float = 1.0


def create_master_flat(
    flat_directory: str | Path,
    output_path: str | Path | None = None,
    min_median: float = 40000.0,
    max_median: float = 50000.0,
    max_counts: float = 50000.0,
    max_percentile: float = 99.9,
    sigma: float = 3.0,
    maxiters: int = 5,
    required_headers: list[str] | None = None,
    dark_directory: str | Path | None = None,
    max_dark_exptime_ratio: float = 10.0,
) -> tuple[np.ndarray, fits.Header]:
    """Create a master flat from a directory of flat field FITS files.

    Args:
        flat_directory: Directory containing flat field FITS files.
        output_path: Path to save the master flat. If None, returns the array
            and header only.
        min_median: Minimum acceptable median value for flat frames.
        max_median: Maximum acceptable median value for flat frames.
        max_counts: Maximum acceptable pixel value for the linearity check
            (applied to the percentile, not the max).
        max_percentile: Percentile to use for the linearity check instead of
            the maximum (default 99.9).
        sigma: Sigma for sigma-clipped median combination.
        maxiters: Maximum iterations for sigma clipping.
        required_headers: Header keywords that must be consistent across frames.
        dark_directory: Directory containing dark frames for dark subtraction
            of flats.
        max_dark_exptime_ratio: Maximum ratio between flat and dark exposure
            times for scaling.

    Returns:
        A ``(master_flat, header)`` tuple: the master flat field normalized to
        median = 1.0 (a photometric flat, so dividing a science frame by it
        preserves the flux scale) and the header from the first valid flat
        frame.

    Raises:
        ValueError: If no FITS files are found or fewer than 3 valid frames
            pass the quality check.
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

    valid_sources, valid_headers, dark_subtracted_count = _validate_flat_sources(
        largest_group_files,
        min_median=min_median,
        max_median=max_median,
        max_counts=max_counts,
        max_percentile=max_percentile,
        dark_directory=dark_directory,
        max_dark_exptime_ratio=max_dark_exptime_ratio,
    )

    if len(valid_sources) < 3:
        raise ValueError(f"Need at least 3 valid frames, found {len(valid_sources)}")

    print(f"Using {len(valid_sources)} valid frames for master flat")
    if dark_directory is not None:
        print(f"Dark subtracted {dark_subtracted_count}/{len(valid_sources)} frames")

    master_flat = _combine_flat_sources(valid_sources, sigma, maxiters)

    # Create output header from first valid frame
    output_header = valid_headers[0].copy()
    output_header.add_history(f"Master flat created from {len(valid_sources)} frames")
    output_header.add_history(f"Sigma-clipped median combination (sigma={sigma}, maxiters={maxiters})")
    output_header.add_history("Normalized to median = 1.0")
    if dark_directory is not None:
        output_header.add_history(f"Dark subtracted {dark_subtracted_count}/{len(valid_sources)} frames")

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory if it doesn't exist
        hdu = fits.PrimaryHDU(data=master_flat.astype(np.float32), header=output_header)
        hdu.writeto(output_path, overwrite=True)
        print(f"Master flat saved to {output_path}")

    return master_flat, output_header


def apply_flat_field(
    image: ProcessedFitsImage | np.ndarray,
    master_flat: str | Path | np.ndarray,
    store_intermediates: bool = False,
) -> ProcessedFitsImage | np.ndarray:
    """Apply flat field correction to an image.

    Args:
        image: Image to flat field correct.
        master_flat: Master flat field. If a string/Path, it is loaded from
            file.
        store_intermediates: Whether to store intermediate correction frames
            (only for ProcessedFitsImage).

    Returns:
        The flat-field-corrected image, matching the input type
        (ProcessedFitsImage or np.ndarray).

    Raises:
        ValueError: If the image and flat frame shapes do not match.
    """
    # Load master flat if provided as file path. float32 throughout: master
    # flats are saved as float32 anyway, and the corrected frame's dtype is
    # inherited by the whole downstream pipeline (float64 doubles its cost).
    if isinstance(master_flat, (str, Path)):
        with fits.open(master_flat) as hdul:
            master_flat = hdul[0].data.astype(np.float32)
    else:
        master_flat = master_flat.astype(np.float32)

    # Handle ProcessedFitsImage objects
    if isinstance(image, ProcessedFitsImage):
        # Ensure shapes match
        if image.data.shape != master_flat.shape:
            raise ValueError(f"Image shape {image.data.shape} doesn't match flat shape {master_flat.shape}")

        # Avoid division by zero/very small values
        safe_flat = np.where(master_flat < 0.1, 1.0, master_flat)

        # Apply flat field correction
        corrected_data = image.data.astype(np.float32) / safe_flat

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

        return image.astype(np.float32) / safe_flat


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


def load_master_flat(file_path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Load a master flat from a FITS file.

    Args:
        file_path: Path to the master flat FITS file.

    Returns:
        A ``(master_flat, header)`` tuple: the master flat field data and the
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
        base_output_path: Base output path (e.g., "master_flat.fits").
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


def _find_dark_for_flat(
    dark_directory: str | Path,
    flat_exptime: float,
    max_exptime_ratio: float = 10.0,
) -> tuple[Path, float] | None:
    """Find the best-matching dark for a flat exposure. Returns (path, dark_exptime)."""
    from senpai.engine.utils.darks import find_best_dark_for_exposure

    return find_best_dark_for_exposure(
        dark_directory=dark_directory,
        target_exptime=flat_exptime,
        matching_headers=["BINNING"],  # Only match binning for flats
        max_exptime_ratio=max_exptime_ratio,
    )


def _validate_flat_sources(
    fits_files: list[Path],
    min_median: float,
    max_median: float,
    max_counts: float,
    max_percentile: float,
    dark_directory: str | Path | None = None,
    max_dark_exptime_ratio: float = 10.0,
    indent: str = "",
) -> tuple[list[_FlatSource], list[fits.Header], int]:
    """Quality-filter flat frames from subsampled stats only.

    Full frames are never held in memory; stats come from an 8x-strided
    subsample read via ``hdu.section`` (which does partial reads even on
    BSCALE/BZERO-scaled integer FITS, where memmap is unavailable). The
    combination step re-reads accepted frames chunk-by-chunk.
    """
    valid_sources: list[_FlatSource] = []
    valid_headers: list[fits.Header] = []
    dark_subtracted_count = 0

    for file_path in fits_files:
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                sample = np.asarray(hdul[0].section[::8, ::8], dtype=np.float64)

            dark_path: Path | None = None
            dark_scale = 1.0
            if dark_directory is not None:
                flat_exptime = header.get("EXPTIME", header.get("EXPOSURE", 1.0))
                dark_result = _find_dark_for_flat(
                    dark_directory, flat_exptime, max_dark_exptime_ratio
                )
                if dark_result is None:
                    print(f"{indent}    No suitable dark found for {flat_exptime}s flat")
                else:
                    dark_path, dark_exptime = dark_result
                    dark_path = Path(dark_path)
                    if abs(flat_exptime - dark_exptime) > 0.1:
                        dark_scale = flat_exptime / dark_exptime
                    with fits.open(dark_path) as dh:
                        sample = sample - dark_scale * np.asarray(
                            dh[0].section[::8, ::8], dtype=np.float64
                        )
                    dark_subtracted_count += 1

            # Check linearity constraints using percentile instead of max to handle hot pixels
            frame_median = float(np.median(sample))
            frame_percentile = float(np.percentile(sample, max_percentile))

            if min_median <= frame_median <= max_median and frame_percentile < max_counts:
                valid_sources.append(
                    _FlatSource(Path(file_path), frame_median, dark_path, dark_scale)
                )
                valid_headers.append(header)
                print(
                    f"{indent}✓ {file_path.name}: median={frame_median:.1f}, "
                    f"{max_percentile:.1f}%ile={frame_percentile:.1f}"
                )
            else:
                print(
                    f"{indent}✗ {file_path.name}: median={frame_median:.1f}, "
                    f"{max_percentile:.1f}%ile={frame_percentile:.1f} - rejected"
                )

        except Exception as e:
            print(f"{indent}✗ {file_path.name}: Error reading file - {e}")

    return valid_sources, valid_headers, dark_subtracted_count


def _combine_flat_sources(
    sources: list[_FlatSource],
    sigma: float,
    maxiters: int,
    chunk_size: int = 512,
) -> np.ndarray:
    """Sigma-clipped median combination, streamed in row chunks.

    Each frame is normalized by its own median (so auto-exposed twilight
    flats at different sky levels combine as relative response maps), then
    the per-pixel sigma-clipped median is taken across frames. Drifting
    stars and saturated pixels appear at any given pixel in only a few
    frames and are rejected by the clip. The result is normalized to
    median = 1.0 — a photometric flat (min-max scaling would be an affine
    distortion of the response and would zero out the dimmest pixels).
    """
    with fits.open(sources[0].path) as hdul:
        height, width = hdul[0].shape

    master_flat = np.empty((height, width), dtype=np.float64)
    sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)

    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        print(f"    Combining rows {start_row}-{end_row - 1}")

        chunk_stack = np.empty((len(sources), end_row - start_row, width), dtype=np.float32)
        dark_chunks: dict[Path, np.ndarray] = {}
        for i, src in enumerate(sources):
            with fits.open(src.path) as hdul:
                chunk = np.asarray(hdul[0].section[start_row:end_row, :], dtype=np.float32)
            if src.dark_path is not None:
                if src.dark_path not in dark_chunks:
                    with fits.open(src.dark_path) as dh:
                        dark_chunks[src.dark_path] = np.asarray(
                            dh[0].section[start_row:end_row, :], dtype=np.float32
                        )
                chunk = chunk - np.float32(src.dark_scale) * dark_chunks[src.dark_path]
            chunk_stack[i] = chunk / np.float32(src.median)

        clipped = sigma_clip(chunk_stack, axis=0)
        if np.ma.isMaskedArray(clipped):
            chunk_result = np.ma.median(clipped, axis=0)
            # Pixels masked in every frame (shouldn't happen) fall back to
            # the unclipped median.
            chunk_result = np.asarray(
                chunk_result.filled(np.nan), dtype=np.float64
            )
            bad = np.isnan(chunk_result)
            if bad.any():
                chunk_result[bad] = np.median(chunk_stack, axis=0)[bad]
        else:
            chunk_result = np.median(clipped, axis=0).astype(np.float64)

        master_flat[start_row:end_row] = chunk_result
        del chunk_stack, clipped, chunk_result

    # Normalize to median = 1.0
    master_flat /= np.median(master_flat)

    return master_flat


def main() -> None:
    """Run the CLI for creating master flat fields (or analyzing headers)."""
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
                master_flat, _header = _create_master_flat_from_files(
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
            master_flat, _header = create_master_flat(
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
    fits_files: list[Path],
    output_path: str | Path,
    min_median: float,
    max_median: float,
    max_counts: float,
    max_percentile: float,
    min_frames: int,
    sigma: float,
    maxiters: int,
    dark_directory: str | Path | None = None,
    max_dark_exptime_ratio: float = 10.0,
) -> tuple[np.ndarray, fits.Header]:
    """Create a master flat from a specific list of FITS files.

    Args:
        fits_files: The flat-frame FITS files to combine.
        output_path: Path to write the resulting master flat to.
        min_median: Minimum acceptable median value for flat frames.
        max_median: Maximum acceptable median value for flat frames.
        max_counts: Maximum acceptable pixel value for the linearity check.
        max_percentile: Percentile used for the linearity check.
        min_frames: Minimum number of valid frames required.
        sigma: Sigma for sigma-clipped median combination.
        maxiters: Maximum iterations for sigma clipping.
        dark_directory: Directory containing dark frames for dark subtraction
            of flats.
        max_dark_exptime_ratio: Maximum ratio between flat and dark exposure
            times for scaling.

    Returns:
        A ``(master_flat, header)`` tuple: the master flat field and its output
        header.

    Raises:
        ValueError: If fewer than ``min_frames`` valid frames pass the quality
            check.
    """
    valid_sources, valid_headers, dark_subtracted_count = _validate_flat_sources(
        fits_files,
        min_median=min_median,
        max_median=max_median,
        max_counts=max_counts,
        max_percentile=max_percentile,
        dark_directory=dark_directory,
        max_dark_exptime_ratio=max_dark_exptime_ratio,
        indent="  ",
    )

    if len(valid_sources) < min_frames:
        raise ValueError(f"Need at least {min_frames} valid frames, found {len(valid_sources)}")

    print(f"  Using {len(valid_sources)} valid frames for master flat")
    if dark_directory is not None:
        print(f"  Dark subtracted {dark_subtracted_count}/{len(valid_sources)} frames")

    master_flat = _combine_flat_sources(valid_sources, sigma, maxiters)

    # Create output header from first valid frame
    output_header = valid_headers[0].copy()
    output_header.add_history(f"Master flat created from {len(valid_sources)} frames")
    output_header.add_history(f"Sigma-clipped median combination (sigma={sigma}, maxiters={maxiters})")
    output_header.add_history("Normalized to median = 1.0")
    if dark_directory is not None:
        output_header.add_history(f"Dark subtracted {dark_subtracted_count}/{len(valid_sources)} frames")

    # Save master flat
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory if it doesn't exist
    hdu = fits.PrimaryHDU(data=master_flat.astype(np.float32), header=output_header)
    hdu.writeto(output_path, overwrite=True)

    return master_flat, output_header


if __name__ == "__main__":
    main()
