"""Download, validate, and pare astrometry.net index files for SENPAI plate solving."""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from astropy.io import fits
from tqdm import tqdm

from senpai.astrometry.constants import (
    ASTROMETRY_4100_EXPECTED_STRUCTURE,
    ASTROMETRY_5200_EXPECTED_STRUCTURE,
    ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE,
    ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE,
    ASTROMETRY_INDICES_URL_4100,
    ASTROMETRY_INDICES_URL_5200,
    ASTROMETRY_INDICES_URL_5200_LITE,
    AstrometryIndexSeries,
)
from senpai.settings import settings

logger = logging.getLogger(__name__)


# Helper function for human-readable file sizes
def human_readable_size(size_bytes: float) -> str:
    """Convert size in bytes to human-readable format with appropriate units.

    Args:
        size_bytes (float): Size in bytes.

    Returns:
        str: The size formatted with two decimals and an appropriate unit
            (B, KB, MB, GB, or TB).
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def get_fits_files(base_url: str) -> list[str]:
    """List the downloadable ``.fits`` index files at a base URL.

    Scrapes the directory listing HTML at ``base_url`` for ``.fits`` links,
    excluding the ``index-##m#-*.fits`` (margin) variants.

    Args:
        base_url (str): HTTP(S) URL of the index directory listing.

    Returns:
        list[str]: Fully qualified URLs of the matching ``.fits`` files.

    Raises:
        ValueError: If ``base_url`` does not use an ``http:`` or ``https:`` scheme.
    """
    # Get list of files
    if not base_url.startswith(("http:", "https:")):
        raise ValueError("URL must start with 'http:' or 'https:'")
    with urlopen(base_url) as response:  # noqa: S310
        html = response.read().decode("utf-8")

    # Modified to exclude files matching index-##m#-*.fits pattern
    fits_files = [
        base_url + filename
        for filename in re.findall(r'href="([^"]+\.fits)"', html)
        if not re.match(r"index-\d+m\d+-.*\.fits", filename)
    ]

    return fits_files


def download_fits_files(
    base_url: str, output_dir: str | None = None, max_workers: int = 5
) -> None:
    """Download .fits files, skipping existing files of the same size.

    Args:
        base_url (str): The URL to download .fits files from
        output_dir (str, optional): Directory to save files to. Defaults to current directory.
        max_workers (int, optional): Number of concurrent downloads. Defaults to 5.

    Returns:
        None
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fits_files = get_fits_files(base_url)

    if not fits_files:
        logger.warning("No .fits files found!")
        return

    logger.info(f"Found {len(fits_files)} .fits files")

    def download_file(url: str) -> None:
        """Download a single index file, skipping it if an identical copy exists.

        Args:
            url (str): HTTP(S) URL of the file to download.

        Returns:
            None

        Raises:
            ValueError: If ``url`` does not use an ``http:`` or ``https:`` scheme.
        """
        if not url.startswith(("http:", "https:")):
            raise ValueError(f"Insecure URL scheme: {url}")

        try:
            filename = url.split("/")[-1]
            if output_dir:
                filename = os.path.join(output_dir, filename)

            # Check if file exists
            if os.path.exists(filename):
                # Get remote file size
                with urlopen(Request(url, method="HEAD")) as response:  # noqa: S310
                    remote_size = int(response.headers["Content-Length"])

                # Get local file size
                local_size = os.path.getsize(filename)

                if remote_size == local_size:
                    return
                else:
                    tqdm.write(f"Size mismatch for {filename}, downloading again...")
                    tqdm.write(f"Remote: {remote_size} bytes, Local: {local_size} bytes")
            else:
                tqdm.write(f"Downloading new file {filename}...")

            # Download with progress bar
            with urlopen(Request(url, method="HEAD")) as response:  # noqa: S310
                file_size = int(response.headers["Content-Length"])

            with (
                urlopen(url) as response,  # noqa: S310
                open(filename, "wb") as f,
                tqdm(
                    total=file_size, unit="B", unit_scale=True, desc=filename, leave=False
                ) as pbar,
            ):
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

            tqdm.write(f"Successfully downloaded {filename}")
        except Exception as e:
            tqdm.write(f"Error downloading {url}: {e}")

    # Overall progress bar for all files
    with (
        tqdm(total=len(fits_files), desc="Total progress", unit="file") as pbar,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        futures = []
        for url in fits_files:
            future = executor.submit(download_file, url)
            future.add_done_callback(lambda p: pbar.update())
            futures.append(future)


def extract_expected_structure(base_url: str) -> dict[str, int] | None:
    """Build a filename-to-size map for the index files at a base URL.

    Issues HTTP HEAD requests to read each file's ``Content-Length``.

    Args:
        base_url (str): HTTP(S) URL of the index directory listing.

    Returns:
        dict[str, int] | None: Mapping of filename to remote size in bytes, or
            ``None`` if no ``.fits`` files were found.

    Raises:
        ValueError: If any discovered URL does not use an ``http:`` or ``https:`` scheme.
    """
    # Get list of files
    fits_files = get_fits_files(base_url)

    if not fits_files:
        logger.warning("No .fits files found!")
        return

    logger.info(f"Found {len(fits_files)} .fits files")

    results_dict = {}

    for url in tqdm(fits_files, desc="Extracting expected filesizes", ascii=True):
        if not url.startswith(("http:", "https:")):
            raise ValueError(f"Insecure URL scheme: {url}")

        filename = url.split("/")[-1]

        with urlopen(Request(url, method="HEAD")) as response:  # noqa: S310
            remote_size = int(response.headers["Content-Length"])

            results_dict[filename] = remote_size

    return results_dict


def examine_by_path_and_structure(
    series: str, indices_path: str, expected_structure: dict
) -> bool:
    """Validate that an index series on disk matches its expected file structure.

    Compares each expected filename and size against the files present at
    ``indices_path``, logging any missing or size-mismatched indices. The custom
    series is treated as always valid.

    Args:
        series (str): Name of the index series being checked (used in log messages).
        indices_path (str): Directory containing the downloaded index files.
        expected_structure (dict): Mapping of expected filename to expected size in bytes.

    Returns:
        bool: ``True`` if the on-disk set is complete and valid (or the series is
            custom), ``False`` otherwise.
    """
    missing_indices = []
    size_mismatch_indices = []

    if series == AstrometryIndexSeries.SERIES_CUSTOM:
        logger.warning(
            f"[{series}] Astrometry indices are custom, skipping validation.  Please consider adding this to the codebase."
        )
        return True

    for filename, expected_size in expected_structure.items():
        filepath = Path(indices_path) / filename
        if filepath.exists():
            # Get local file size
            local_size = os.path.getsize(filepath)

            if expected_size != local_size:
                size_mismatch_indices.append(filename)
        else:
            missing_indices.append(filename)

    complete_set = len(size_mismatch_indices) + len(missing_indices) == 0

    if complete_set:
        logger.info(
            f"[{series}] Astrometry indices [{series}] are complete and valid [{indices_path}]"
        )
        return True

    if len(size_mismatch_indices) > 0:
        logger.warning(
            f"[{series}] Astrometry indices size mismatch "
            + f"for {', '.join(size_mismatch_indices)}"
        )
    if len(missing_indices) > 0:
        logger.warning(f"[{series}] Astrometry indices missing:{', '.join(missing_indices)}")

    logger.warning(f"[{series}] Astrometry indices are incomplete")
    logger.warning(
        f"[{series}] fix: python -m senpai.astrometry.indices download --series "
        + f"{series} --index_path {indices_path}"
    )

    return False


def pare_5200_to_SENPAI(series: str, indices_path: str | Path, output_path: str | Path) -> None:
    """Create the pared-down 5200-SENPAI series from a full 5200 index set.

    For each 5200 index file, keeps only the catalog columns SENPAI needs and writes
    the reduced file to ``output_path``, logging the size reduction. Files already
    present at the expected SENPAI size are skipped.

    Args:
        series (str): Source series; must be the 5200 series.
        indices_path (str | Path): Directory containing the full 5200 index files.
        output_path (str | Path): Directory to write the pared 5200-SENPAI files to.

    Returns:
        None

    Raises:
        ValueError: If ``series`` is not the 5200 series, or if the source indices
            are not valid.
    """
    if series != AstrometryIndexSeries.SERIES_5200:
        raise ValueError("Only series=5200 is supported for creating 5200-SENPAI")

    # ideally, first we'll check if output_path already is valid series, but we have to build it the first time before we write that code (need expected_structure)
    already_converted = examine_by_path_and_structure(
        "5200_SENPAI", output_path, ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE
    )
    if already_converted:
        logger.info("5200_SENPAI series already exists.")
        return

    source_indices_good = examine_by_path_and_structure(
        series, indices_path, ASTROMETRY_5200_EXPECTED_STRUCTURE
    )

    if not source_indices_good:
        raise ValueError("Source indices are not valid")

    columns_to_keep = ["ra", "dec", "mag", "ref_cat", "ref_id"]
    catalog_hdu_num = 13

    indices_path = Path(indices_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    total_original_size = 0
    total_new_size = 0

    for index_file in ASTROMETRY_5200_EXPECTED_STRUCTURE:
        index_file_path = indices_path / index_file
        output_file_path = output_path / index_file

        if output_file_path.exists() and (
            os.path.getsize(output_file_path)
            == ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE[index_file]
        ):
            logger.info(f"[{index_file}] already exists and is valid")
            continue

        with fits.open(index_file_path) as hdul:
            # get the catalog hdu
            catalog_hdu = hdul[catalog_hdu_num]

            # Create a new table with only the columns we want to keep
            new_data = fits.BinTableHDU.from_columns(
                [catalog_hdu.data.columns[col] for col in columns_to_keep]
            )

            # Copy over the header from the original HDU
            new_data.header["AN_FILE"] = "TAGALONG"

            # Replace the catalog HDU with our new one
            hdul[catalog_hdu_num] = new_data

            # Save modified FITS file to output path
            hdul.writeto(output_file_path, overwrite=True)

            # Calculate size reduction and log it
            original_size = ASTROMETRY_5200_EXPECTED_STRUCTURE[index_file]
            new_size = os.path.getsize(output_file_path)

            total_original_size += original_size
            total_new_size += new_size

            reduction = original_size - new_size
            reduction_percent = (reduction / original_size) * 100

            logger.info(
                f"{index_file} reduced by {human_readable_size(reduction)} ({reduction_percent:.1f}%) "
                f"from {human_readable_size(original_size)} to {human_readable_size(new_size)}"
            )

    if total_original_size > 0:
        # Calculate total reduction
        total_reduction = total_original_size - total_new_size
        total_reduction_percent = (total_reduction / total_original_size) * 100

        logger.info(
            f"Total size reduced by {human_readable_size(total_reduction)} ({total_reduction_percent:.1f}%) "
            f"from {human_readable_size(total_original_size)} to {human_readable_size(total_new_size)}"
        )

    print(json.dumps(ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE, indent=4))

    examine_by_path_and_structure(
        "5200-SENPAI", output_path, ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE
    )


def get_expected_structure(
    series: AstrometryIndexSeries,
) -> tuple[list[str], dict[str, int]]:
    """Resolve the download URLs and expected file structure for an index series.

    Args:
        series (AstrometryIndexSeries): The index series to resolve.

    Returns:
        tuple[list[str], dict[str, int]]: The base URL(s) to download from and the
            expected filename-to-size mapping for the series. Both are empty for the
            custom series, and the URL list is empty for the 5200-SENPAI series.

    Raises:
        AttributeError: If ``series`` is not a recognized series.
    """
    if series == AstrometryIndexSeries.SERIES_5200_SENPAI:
        base_urls = []
        expected_structure = ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_5200:
        base_urls = [ASTROMETRY_INDICES_URL_5200]
        expected_structure = ASTROMETRY_5200_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_5200_LITE:
        base_urls = [ASTROMETRY_INDICES_URL_5200_LITE]
        expected_structure = ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_4100:
        base_urls = [ASTROMETRY_INDICES_URL_4100]
        expected_structure = ASTROMETRY_4100_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_5200_LITE_4100:
        base_urls = [ASTROMETRY_INDICES_URL_5200_LITE, ASTROMETRY_INDICES_URL_4100]
        expected_structure = (
            ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE | ASTROMETRY_4100_EXPECTED_STRUCTURE
        )
    elif series == AstrometryIndexSeries.SERIES_CUSTOM:
        base_urls = []
        expected_structure = {}
    else:
        raise AttributeError(f"Unknown series {series}")

    return base_urls, expected_structure


def examine_indices() -> bool:
    """Validate the configured index series against the configured indices path.

    Returns:
        bool: ``True`` if the configured indices are complete and valid, else ``False``.
    """
    _, expected_structure = get_expected_structure(settings.astrometry.indices_series)

    indices_path = settings.astrometry.indices_path

    return examine_by_path_and_structure(
        settings.astrometry.indices_series, indices_path, expected_structure
    )


def enforce_indices() -> None:
    """Validate the configured indices and raise if they are incomplete.

    Returns:
        None

    Raises:
        RuntimeError: If the configured astrometry indices are missing or incomplete.
    """
    if not examine_indices():
        raise RuntimeError(
            f"Astrometry indices for {settings.astrometry.indices_series} are missing or incomplete"
        )


def check_indices_on_startup() -> None:
    """Check astrometry indices at startup, enforcing completeness if so configured.

    When ``require_complete_indices`` is set, raises on incomplete indices; otherwise
    only logs the validation result.

    Returns:
        None
    """
    if settings.astrometry.require_complete_indices:
        enforce_indices()
    else:
        examine_indices()


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Astrometry indices management")
    parser.add_argument(
        "action",
        choices=["download", "examine", "map_expected", "supported", "build_5200_senpai"],
        help="Action to perform",
    )
    parser.add_argument(
        "--series",
        type=AstrometryIndexSeries,
        choices=list(AstrometryIndexSeries),
        required=False,
        default=AstrometryIndexSeries.SERIES_5200,
        help="Index series to download",
    )
    parser.add_argument("--index_path", required=False, help="Path to the indices directory")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent downloads")

    args = parser.parse_args()

    base_urls, expected_structure = get_expected_structure(args.series)

    if args.action == "download":
        for base_url in base_urls:
            download_fits_files(base_url, output_dir=args.index_path, max_workers=args.workers)

    elif args.action == "examine":
        examine_by_path_and_structure(args.series, args.index_path, expected_structure)

    elif args.action == "supported":
        print(f"Supported indices: {', '.join(AstrometryIndexSeries)}")

    elif args.action == "map_expected":
        expected = {}

        for series, url in zip(
            ["4100", "5200", "5200_LITE"],
            [
                ASTROMETRY_INDICES_URL_4100,
                ASTROMETRY_INDICES_URL_5200,
                ASTROMETRY_INDICES_URL_5200_LITE,
            ],
            strict=False,
        ):
            expected[series] = extract_expected_structure(url)

        print(json.dumps(expected, indent=4))

    elif args.action == "build_5200_senpai":
        indices_path = Path(args.index_path)
        output_path = indices_path.parent / "5200-SENPAI"
        pare_5200_to_SENPAI(args.series, indices_path, output_path)
