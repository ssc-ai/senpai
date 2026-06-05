import json
import logging
from pathlib import Path

import numpy as np
import rawpy
from astropy.io import fits
from fastapi import UploadFile

from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.senpai import SenpaiRunResult

logger = logging.getLogger(__name__)


def load_senpai_run(json_path: Path | str) -> SenpaiRunResult:
    """Load a SENPAI run from JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return SenpaiRunResult.model_validate(data)
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        raise


def load_jpeg_file(jpeg_file: Path | str) -> ProcessedFitsImage:
    """
    Loads a .JPEG file and returns:
      - grayscale numpy array
      - metadata header info (as dict)
    """
    logger.warning("untested")
    return ProcessedFitsImage.from_file_bytes(jpeg_file.read_bytes(), file_path=str(jpeg_file))


def load_dng_file(dng_file: Path | str) -> ProcessedFitsImage:
    """
    Loads a .DNG file and returns:
      - grayscale numpy array
      - metadata header info (as dict)
    """
    # Open and process the DNG (RAW) file
    with rawpy.imread(dng_file) as raw:
        # Get postprocessed RGB image (demosaiced)
        rgb = raw.postprocess()  # shape: (H, W, 3)

        # Convert to grayscale by summing all color channels
        gray = np.sum(rgb.astype(np.float32), axis=2)

        # Normalize to 0–1
        gray /= gray.max()
        # Extract header info (metadata) as FITS Header
        header = fits.Header()

        # Add basic image dimensions
        header["NAXIS1"] = gray.shape[1]
        header["NAXIS2"] = gray.shape[0]
        header["NAXIS"] = 2

        # Add camera metadata
        header["CAM_WB"] = str(raw.camera_whitebalance)
        header["DAY_WB"] = str(raw.daylight_whitebalance)
        header["NUM_COL"] = raw.num_colors
        header["COL_DESC"] = raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
        header["RAW_TYPE"] = str(raw.raw_type)
        header["WHITE_LVL"] = raw.white_level

        # Add array data as JSON strings (FITS headers have size limits)
        if hasattr(raw.color_matrix, "tolist"):
            header["COL_MAT"] = str(raw.color_matrix.tolist())
        else:
            header["COL_MAT"] = str(raw.color_matrix)

        if hasattr(raw.rgb_xyz_matrix, "tolist"):
            header["RGB_XYZ"] = str(raw.rgb_xyz_matrix.tolist())
        else:
            header["RGB_XYZ"] = str(raw.rgb_xyz_matrix)

        if hasattr(raw.black_level_per_channel, "tolist"):
            header["BLACK_LV"] = str(raw.black_level_per_channel.tolist())
        else:
            header["BLACK_LV"] = str(raw.black_level_per_channel)

        if hasattr(raw.camera_white_level_per_channel, "tolist"):
            header["CAM_WLV"] = str(raw.camera_white_level_per_channel.tolist())
        else:
            header["CAM_WLV"] = str(raw.camera_white_level_per_channel)

        if hasattr(raw.raw_pattern, "tolist"):
            header["RAW_PAT"] = str(raw.raw_pattern.tolist())
        else:
            header["RAW_PAT"] = str(raw.raw_pattern)

        # Add image dimensions info
        header["RAW_SHAP"] = str(raw.raw_image.shape)
        header["OUT_SHAP"] = str(rgb.shape)
        header["VIS_SHAP"] = str(raw.raw_image_visible.shape)

    metadata = ImageMetadata(
        image_id=Path(dng_file).stem,
        width=gray.shape[1],
        height=gray.shape[0],
        exposure_time=1.0,
    )

    return ProcessedFitsImage(
        data=gray,
        header=header,
        data_type=gray.dtype,
        metadata=metadata,
        file_path=str(dng_file),
    )


def load_fits_files(fits_files: list[Path | str]) -> list[ProcessedFitsImage]:
    fits_files = [Path(f) for f in fits_files]

    for fits_file in fits_files:
        if not fits_file.exists():
            raise FileNotFoundError(f"File {fits_file} does not exist")

    logger.info("Loading %d FITS files: %s", len(fits_files), ", ".join(f.name for f in fits_files))

    return [
        ProcessedFitsImage.from_file_bytes(fits_file.read_bytes(), file_path=str(fits_file)) for fits_file in fits_files
    ]


def load_fits_file(fits_file: Path | str) -> ProcessedFitsImage:
    logger.info("loading fits file from disk")
    fits_file = Path(fits_file)

    if not fits_file.exists():
        raise FileNotFoundError(f"File {fits_file} does not exist")

    return ProcessedFitsImage.from_file_bytes(fits_file.read_bytes(), file_path=str(fits_file))


async def load_uploaded_files(fits_files: list[UploadFile]) -> list[ProcessedFitsImage]:
    logger.info(f"loading {len(fits_files)} uploaded files")

    processed_files = []

    for file in fits_files:
        fits_file = ProcessedFitsImage.from_file_bytes(await file.read(), file_path=file.filename)
        processed_files.append(fits_file)

    return processed_files


def load_base64_files(base64_files: list[str]) -> list[ProcessedFitsImage]:
    logger.info(f"loading {len(base64_files)} base64 files")

    return [ProcessedFitsImage.from_base64_string(base64_file) for base64_file in base64_files]
