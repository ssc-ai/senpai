import argparse
import hashlib
import json
import logging
import os
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

from senpai.catalog.constants import (
    SSTR7_EXPECTED_CHECKSUMS,
    SSTR7_EXPECTED_FILES,
    SSTR7_GITHUB_REPO,
    SSTR7_RELEASE_TAG,
)
from senpai.core.config import get_or_initialize_config

logger = logging.getLogger(__name__)


# Helper function for human-readable file sizes
def human_readable_size(size_bytes):
    """Convert size in bytes to human-readable format with appropriate units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def examine_sstrc7_by_path_and_structure(catalog_path: str, expected_files: dict) -> bool:
    """Examine SSTRC7 catalog files and check if they match expected structure.

    Args:
        catalog_path: Path to SSTRC7 catalog directory
        expected_files: Dictionary mapping filenames to expected file sizes

    Returns:
        True if all files exist and match expected sizes, False otherwise
    """
    missing_files = []
    size_mismatch_files = []

    catalog_path_obj = Path(catalog_path)
    if not catalog_path_obj.exists():
        logger.warning(f"SSTRC7 catalog path does not exist: {catalog_path}")
        return False

    if not catalog_path_obj.is_dir():
        logger.warning(f"SSTRC7 catalog path is not a directory: {catalog_path}")
        return False

    for filename, expected_size in expected_files.items():
        filepath = catalog_path_obj / filename
        if filepath.exists():
            # Get local file size
            local_size = os.path.getsize(filepath)

            if expected_size != local_size:
                size_mismatch_files.append(filename)
        else:
            missing_files.append(filename)

    complete_set = len(size_mismatch_files) + len(missing_files) == 0

    if complete_set:
        logger.info(f"SSTRC7 catalog is complete and valid [{catalog_path}]")
        return True

    if len(size_mismatch_files) > 0:
        logger.warning(f"SSTRC7 catalog size mismatch for {', '.join(size_mismatch_files)}")
    if len(missing_files) > 0:
        logger.warning(f"SSTRC7 catalog missing files: {', '.join(missing_files)}")

    logger.warning("SSTRC7 catalog is incomplete")
    logger.warning(
        f"Fix: python -m senpai.catalog.sstrc7_management download --catalog_path {catalog_path}"
    )

    return False


def get_github_release_assets(repo: str, tag: str):
    """Get release asset URLs from GitHub API.

    Args:
        repo: GitHub repository in format "owner/repo"
        tag: Release tag (e.g., "v1.0.0")

    Returns:
        List of dictionaries with asset information (name, url, size)
    """
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"

    try:
        req = Request(api_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urlopen(req) as response:
            release_data = json.loads(response.read().decode("utf-8"))

        assets = []
        for asset in release_data.get("assets", []):
            assets.append(
                {
                    "name": asset["name"],
                    "url": asset["browser_download_url"],
                    "size": asset["size"],
                }
            )

        return assets
    except Exception as e:
        logger.error(f"Failed to fetch GitHub release info: {e}")
        raise


def download_with_resume(url: str, filename: str, expected_size: int = None) -> bool:
    """Download file with resume capability.

    Args:
        url: URL to download from
        filename: Local filename to save to
        expected_size: Expected file size (optional, for validation)

    Returns:
        True on success, False on failure
    """
    try:
        # Check if file exists and get its size
        resume_pos = 0
        if os.path.exists(filename):
            resume_pos = os.path.getsize(filename)
            if expected_size and resume_pos == expected_size:
                logger.info(f"File already complete: {filename}")
                return True
            logger.info(f"Resuming download of {filename} from byte {resume_pos}")

        # Get file size from server
        req = Request(url, method="HEAD")
        with urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))

        # Create request for download
        if resume_pos > 0:
            req = Request(url)
            req.add_header("Range", f"bytes={resume_pos}-")
        else:
            req = Request(url)

        # Download with progress bar
        with urlopen(req) as response:
            # Handle 206 Partial Content for resume
            status_code = response.getcode()
            if status_code == 206:
                mode = "ab"  # Append mode for resume
            elif status_code == 200:
                mode = "wb"  # Write mode for new download
                resume_pos = 0
            else:
                logger.error(f"Unexpected HTTP status code: {status_code}")
                return False

            with open(filename, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=resume_pos,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(filename),
                    leave=False,
                ) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify final size if expected
        if expected_size:
            final_size = os.path.getsize(filename)
            if final_size != expected_size:
                logger.error(
                    f"Size mismatch for {filename}: expected {expected_size}, got {final_size}"
                )
                return False

        logger.info(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def verify_file_checksum(filename: str, expected_checksum: str) -> bool:
    """Verify SHA256 checksum of a file.

    Args:
        filename: Path to file to verify
        expected_checksum: Expected SHA256 checksum (hex string)

    Returns:
        True if checksum matches, False otherwise
    """
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    actual_checksum = sha256_hash.hexdigest()
    matches = actual_checksum.lower() == expected_checksum.lower()

    if not matches:
        logger.error(
            f"Checksum mismatch for {filename}: expected {expected_checksum}, got {actual_checksum}"
        )
    else:
        logger.info(f"Checksum verified for {filename}")

    return matches


def verify_sstrc7_checksums(part_files: list[str], expected_checksums: dict) -> bool:
    """Verify SHA256 checksums of downloaded part files.

    Args:
        part_files: List of paths to part files
        expected_checksums: Dictionary mapping filenames to expected checksums

    Returns:
        True if all checksums match, False otherwise
    """
    all_valid = True
    for part_file in part_files:
        filename = os.path.basename(part_file)
        if filename not in expected_checksums:
            logger.warning(f"No expected checksum for {filename}, skipping verification")
            continue

        if not verify_file_checksum(part_file, expected_checksums[filename]):
            all_valid = False

    return all_valid


def combine_and_extract_sstrc7(part_files: list[str], output_dir: str) -> None:
    """Combine part files and extract tar.gz archive.

    Args:
        part_files: List of part file paths in order (part00, part01, ...)
        output_dir: Directory to extract catalog to
    """
    # Sort part files to ensure correct order
    part_files_sorted = sorted(part_files, key=lambda x: int(x.split("part")[-1]))

    tar_filename = os.path.join(output_dir, "sstrc7.tar.gz")
    logger.info(f"Combining {len(part_files_sorted)} part files into {tar_filename}")

    # Combine parts
    with open(tar_filename, "wb") as outfile:
        for part_file in tqdm(part_files_sorted, desc="Combining parts"):
            with open(part_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    logger.info(f"Extracting {tar_filename} to {output_dir}")
    # Extract tar.gz
    with tarfile.open(tar_filename, "r:gz") as tar:
        tar.extractall(path=output_dir)

    # Clean up tar file
    os.remove(tar_filename)
    logger.info(f"Extraction complete, cleaned up {tar_filename}")


def download_sstrc7_from_github(output_dir: str, max_workers: int = 3, force: bool = False) -> None:
    """Download SSTRC7 catalog from GitHub release.

    Args:
        output_dir: Directory to download and extract catalog to
        max_workers: Number of concurrent downloads
        force: Force re-download even if files exist
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get release assets
    logger.info(f"Fetching release information for {SSTR7_GITHUB_REPO} tag {SSTR7_RELEASE_TAG}")
    assets = get_github_release_assets(SSTR7_GITHUB_REPO, SSTR7_RELEASE_TAG)

    # Filter for part files
    part_assets = [
        asset for asset in assets if asset["name"].startswith("sstrc7.tar.gz.part")
    ]
    part_assets.sort(key=lambda x: int(x["name"].split("part")[-1]))

    if not part_assets:
        logger.error("No part files found in release")
        return

    logger.info(f"Found {len(part_assets)} part files to download")

    # Download part files
    def download_part(asset):
        filename = os.path.join(output_dir, asset["name"])
        if os.path.exists(filename) and not force:
            # Check if file is complete
            local_size = os.path.getsize(filename)
            if local_size == asset["size"]:
                logger.info(f"File already exists and is complete: {filename}")
                return filename
        if download_with_resume(asset["url"], filename, asset["size"]):
            return filename
        return None

    part_files = []
    with tqdm(total=len(part_assets), desc="Downloading parts", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for asset in part_assets:
                future = executor.submit(download_part, asset)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)

            for future in futures:
                result = future.result()
                if result:
                    part_files.append(result)

    if len(part_files) != len(part_assets):
        logger.error("Not all part files downloaded successfully")
        return

    # Verify checksums
    logger.info("Verifying checksums...")
    if not verify_sstrc7_checksums(part_files, SSTR7_EXPECTED_CHECKSUMS):
        logger.error("Checksum verification failed")
        return

    # Combine and extract
    combine_and_extract_sstrc7(part_files, output_dir)

    # Clean up part files
    logger.info("Cleaning up part files...")
    for part_file in part_files:
        os.remove(part_file)

    # Verify extracted files
    logger.info("Verifying extracted catalog...")
    if examine_sstrc7_by_path_and_structure(output_dir, SSTR7_EXPECTED_FILES):
        logger.info("SSTRC7 catalog download and extraction complete!")
    else:
        logger.warning("Extracted catalog verification failed - some files may be missing")


def examine_sstrc7():
    """Examine SSTRC7 catalog using config."""
    config = get_or_initialize_config()
    catalog_path = config.star_catalog.path

    if not catalog_path:
        logger.error("SSTRC7 catalog path not configured")
        return False

    return examine_sstrc7_by_path_and_structure(catalog_path, SSTR7_EXPECTED_FILES)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="SSTRC7 catalog management")
    parser.add_argument(
        "action",
        choices=["download", "examine", "verify"],
        help="Action to perform",
    )
    parser.add_argument(
        "--catalog_path",
        required=False,
        help="Path to catalog directory (uses config if not provided)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent downloads (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    args = parser.parse_args()

    if args.action == "download":
        catalog_path = args.catalog_path
        if not catalog_path:
            config = get_or_initialize_config()
            catalog_path = config.star_catalog.path
            if not catalog_path:
                logger.error("catalog_path must be provided or configured")
                exit(1)

        download_sstrc7_from_github(catalog_path, max_workers=args.workers, force=args.force)

    elif args.action == "examine":
        catalog_path = args.catalog_path
        if not catalog_path:
            config = get_or_initialize_config()
            catalog_path = config.star_catalog.path
            if not catalog_path:
                logger.error("catalog_path must be provided or configured")
                exit(1)

        examine_sstrc7_by_path_and_structure(catalog_path, SSTR7_EXPECTED_FILES)

    elif args.action == "verify":
        catalog_path = args.catalog_path
        if not catalog_path:
            config = get_or_initialize_config()
            catalog_path = config.star_catalog.path
            if not catalog_path:
                logger.error("catalog_path must be provided or configured")
                exit(1)

        # Find part files in catalog directory (if they exist)
        catalog_path_obj = Path(catalog_path)
        part_files = list(catalog_path_obj.glob("sstrc7.tar.gz.part*"))
        if part_files:
            part_files_str = [str(f) for f in part_files]
            verify_sstrc7_checksums(part_files_str, SSTR7_EXPECTED_CHECKSUMS)
        else:
            logger.info("No part files found to verify. Catalog may already be extracted.")

