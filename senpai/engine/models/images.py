import base64
import logging
from enum import Enum
from io import BytesIO
from typing import Dict, List, Optional, Union

import numpy as np
from astropy.io import fits
from pydantic import BaseModel

from senpai.engine.models.metadata import ImageMetadata

logger = logging.getLogger(__name__)


class FitsImage(BaseModel, arbitrary_types_allowed=True):
    data: np.ndarray
    header: fits.Header


class ProcessingStep(Enum):
    DARK_SUBTRACT = "dark_subtract"
    FLAT_DIVIDE = "flat_divide"
    BACKGROUND_SUBTRACT = "background_subtract"
    ROW_MEDIAN_SUBTRACT = "row_median_subtract"
    COLUMN_MEDIAN_SUBTRACT = "column_median_subtract"
    FWHM_OPTIMIZATION = "fwhm_optimization"


class ProcessingMetadata(BaseModel):
    step_type: ProcessingStep
    parameters: Dict[str, Union[float, str, int]]  # Store relevant parameters for each step


class ProcessedFitsImage(BaseModel):
    # The processed image data
    data: np.ndarray

    # Original header plus any updates
    header: fits.Header

    # List of processing steps applied, in order
    processing_history: List[ProcessingMetadata] = []

    # Optional storage of intermediate data (like flat frames, background models, etc.)
    # Keys are ProcessingStep values, values are the corresponding correction arrays
    correction_frames: Optional[Dict[ProcessingStep, np.ndarray]] = None

    # processed data to store if FWHM_OPTIMIZATION is applied
    processed_unscaled_data: Optional[np.ndarray] = None

    # Original raw data (optional)
    original_data: Optional[np.ndarray] = None

    # data_type for input image
    data_type: np.dtype

    # metadata for image processing:
    metadata: ImageMetadata

    # file path of the original image
    file_path: str | None = None

    # file path of the processed image (saved during processing)
    processed_file_path: str | None = None

    class Config:
        arbitrary_types_allowed = True  # Needed for numpy arrays

    def scale_frame(self, scale_factor: float, method: str = "block_median") -> None:
        """Scale the frame data by the given factor.

        Args:
            scale_factor: Factor to scale the image by (e.g. 2.0 means downsample by factor of 2)
            method: Scaling method to use, one of ["block_median"]
        """
        if scale_factor <= 0:
            raise ValueError(f"Invalid scale factor: {scale_factor}")

        if scale_factor == 1.0:
            return  # No scaling needed

        # Store unscaled data if not already stored
        if self.processed_unscaled_data is None:
            self.processed_unscaled_data = self.data.copy()

        if method == "block_median":
            # Calculate new dimensions
            new_height = int(self.data.shape[0] / scale_factor)
            new_width = int(self.data.shape[1] / scale_factor)

            # Reshape array to perform block operations
            block_shape = (new_height, int(scale_factor), new_width, int(scale_factor))

            # Trim array if needed to make it divisible by scale_factor
            trim_height = new_height * int(scale_factor)
            trim_width = new_width * int(scale_factor)
            trimmed = self.data[:trim_height, :trim_width]

            # Reshape and take median of blocks
            blocks = trimmed.reshape(block_shape)
            self.data = np.median(blocks, axis=(1, 3))

            # Update header with new dimensions
            self.header["NAXIS1"] = new_width
            self.header["NAXIS2"] = new_height

            self.metadata.width = new_width
            self.metadata.height = new_height

            # Record the scaling step with trimming info
            self.processing_history.append(
                ProcessingMetadata(
                    step_type=ProcessingStep.FWHM_OPTIMIZATION,
                    parameters={
                        "scale_factor": float(scale_factor),
                        "method": method,
                        "original_width": int(trim_width),
                        "original_height": int(trim_height),
                        "trimmed": True,
                    },
                )
            )
        elif method == "median_filter":
            # Median filter reduction - apply median filter then downsample
            # Round scale factor to nearest integer for median filter
            scale_factor_int = int(round(scale_factor))
            if scale_factor_int < 1:
                scale_factor_int = 1  # Ensure minimum scale factor of 1

            # Apply median filter with kernel size equal to scale factor
            from scipy.ndimage import median_filter

            filtered_data = median_filter(self.data, size=(scale_factor_int, scale_factor_int))

            # Downsample by taking every Nth pixel
            self.data = filtered_data[::scale_factor_int, ::scale_factor_int]

            # Update header with new dimensions
            new_width = self.data.shape[1]
            new_height = self.data.shape[0]
            self.header["NAXIS1"] = new_width
            self.header["NAXIS2"] = new_height

            self.metadata.width = new_width
            self.metadata.height = new_height

            # Record the scaling step
            self.processing_history.append(
                ProcessingMetadata(
                    step_type=ProcessingStep.FWHM_OPTIMIZATION,
                    parameters={
                        "scale_factor": float(scale_factor_int),  # Record the actual integer scale factor used
                        "original_scale_factor": float(scale_factor),  # Record the original recommended scale factor
                        "method": method,
                        "original_width": new_width * scale_factor_int,
                        "original_height": new_height * scale_factor_int,
                        "trimmed": False,
                    },
                )
            )
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    def get_scale_factor(self) -> float:
        """Get the current scale factor of the frame.

        Returns:
            float: The scale factor used to scale the frame. 1.0 if no scaling was applied.
        """
        # First check processing history
        for step in reversed(self.processing_history):
            if step.step_type == ProcessingStep.FWHM_OPTIMIZATION:
                return float(step.parameters["scale_factor"])

        # If no scaling step found in history, calculate from dimensions if we have unscaled data
        if self.processed_unscaled_data is not None:
            height_ratio = self.processed_unscaled_data.shape[0] / self.data.shape[0]
            width_ratio = self.processed_unscaled_data.shape[1] / self.data.shape[1]
            # Both ratios should be very close, take average
            return float((height_ratio + width_ratio) / 2)

        return 1.0  # No scaling applied

    @classmethod
    def from_fits(
        cls, fits_file: fits.ImageHDU | fits.PrimaryHDU, file_path: str | None = None
    ) -> "ProcessedFitsImage":
        # Extract exposure time from header
        exposure_time = None
        for key in ["EXPTIME", "EXPOSURE", "TELAPSE"]:
            if key in fits_file.header:
                exposure_time = float(fits_file.header[key])
                break

        if exposure_time is None:
            logger.warning(f"No exposure time found in header. Available keys: {list(fits_file.header.keys())}")
            for key in fits_file.header.keys():
                if "TIME" in key.upper() or "EXP" in key.upper():
                    logger.info(f"  {key}: {fits_file.header[key]}")

        metadata = ImageMetadata(
            image_id=fits_file.header.get("IMAGEID", "tmp"),
            width=fits_file.header["NAXIS1"],
            height=fits_file.header["NAXIS2"],
            exposure_time=exposure_time,
        )
        return cls(
            data=fits_file.data,
            header=fits_file.header,
            data_type=fits_file.data.dtype,
            metadata=metadata,
            file_path=file_path,
        )

    @classmethod
    def from_file_bytes(cls, file_bytes: bytes, file_path: str | None = None) -> "ProcessedFitsImage":
        hdul = fits.open(BytesIO(file_bytes))
        return cls.from_fits(hdul[0], file_path=file_path)

    @classmethod
    def from_base64_string(cls, base64_string: str) -> "ProcessedFitsImage":
        file_bytes = base64.b64decode(base64_string)
        return cls.from_file_bytes(file_bytes)
