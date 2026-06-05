from datetime import datetime
from enum import Enum

import numpy as np
from astropy.io.fits import Header
from pydantic import BaseModel

from senpai.engine.detection.kernels import rectangle_pyramoid


class TrackMode(Enum):
    RATE = "rate"
    SIDEREAL = "sidereal"
    UNKNOWN = "unknown"


# Define SiteMetadata first, before importing functions that use it
class SiteMetadata(BaseModel):
    name: str | None = None
    latitude: float
    longitude: float
    altitude_km: float | None = None


class FWHMMetadata(BaseModel):
    """Detailed FWHM statistics collected from star detections."""

    n_measurements: int
    median_fwhm: float
    mean_fwhm: float
    std_fwhm: float
    min_fwhm: float
    max_fwhm: float
    # Individual measurements for analysis
    fwhm_vs_position: list[tuple[float, float, float]]  # [(x, y, fwhm), ...]
    fwhm_vs_magnitude: list[tuple[float, float]]  # [(magnitude, fwhm), ...]
    fwhm_vs_counts: list[tuple[float, float]]  # [(counts, fwhm), ...]
    # Spatial analysis
    has_spatial_gradient: bool = False
    spatial_gradient_info: dict | None = None
    # Scaling information
    is_oversampled: bool = False
    recommended_scale_factor: float | None = None


class DetectionMetadata(BaseModel):
    pixel_fwhm: float
    fwhm_stats: FWHMMetadata | None = None


class CollectionMetadata(BaseModel):
    pixel_rate_per_second: float | None = None


class ImageMetadata(BaseModel):
    image_id: str | None = None
    width: int
    height: int
    boresight_ra: float | None = None
    boresight_dec: float | None = None
    fov_min_degrees: float | None = None
    fov_max_degrees: float | None = None
    exposure_time: float | None = None  # Exposure time in seconds


class SeeingMetadata(BaseModel):
    arcsec: float | None = None
    arcsec_stdev: float | None = None
    n_measurements: int | None = None
    pixel: float
    pixel_stdev: float | None = None


class SeeingModel(BaseModel):
    pixel_fwhm: float
    pixel_fwhm_stdev: float
    n_measurements: int

    @classmethod
    def from_fwhm_stats(cls, fwhm_stats: FWHMMetadata) -> "SeeingModel":
        return cls(
            pixel_fwhm=fwhm_stats.median_fwhm,
            pixel_fwhm_stdev=fwhm_stats.std_fwhm,
            n_measurements=fwhm_stats.n_measurements,
        )


class StarMetadata(BaseModel):
    ra: float
    dec: float
    magnitude: float
    magnitude_stdev: float
    n_measurements: int


class StreakMetadata(BaseModel):
    pixel_length: float
    sine_angle: float
    cosine_angle: float
    fwhm: float
    # Whether to use variable, distortion-aware kernels for this streak
    use_variable_kernel: bool = False

    def degree_angle(self) -> float:
        return np.rad2deg(self.radian_angle())

    def radian_angle(self) -> float:
        return np.arctan2(self.sine_angle, self.cosine_angle)

    def to_pyramoid(self) -> np.ndarray:
        kernel = rectangle_pyramoid(self.pixel_length, self.sine_angle, self.cosine_angle, self.fwhm)

        return kernel


class FrameMetadata(BaseModel):
    exposure_time_seconds: float
    observation_time: datetime
    site: SiteMetadata | None = None
    track_mode: TrackMode | None = None
    track_rate_ra_arcsec_per_second: float | None = None
    track_rate_dec_arcsec_per_second: float | None = None
    boresight_ra_degrees: float | None = None
    boresight_dec_degrees: float | None = None
    observation_filter: str | None = None

    def to_serializable(self) -> "FrameMetadata":
        """Create a copy of this FrameMetadata with datetime converted to ISO format string."""
        data = self.dict()
        if self.observation_time:
            data["observation_time"] = self.observation_time.isoformat()
        return FrameMetadata(**data)

    @classmethod
    def from_header(cls, header: Header) -> "FrameMetadata":
        # avoid circular import
        from senpai.engine.utils.fits_io import (
            extract_boresight_from_header,
            extract_exposure_time_from_header,
            extract_filter_from_header,
            extract_observation_time_from_header,
            extract_observing_site_from_header,
            extract_track_rates_from_header,
        )

        site = extract_observing_site_from_header(header)
        boresight_ra, boresight_dec = extract_boresight_from_header(header)
        exposure_time = extract_exposure_time_from_header(header)
        observation_time = extract_observation_time_from_header(header)
        track_rate_ra, track_rate_dec, track_mode = extract_track_rates_from_header(header)
        observation_filter = extract_filter_from_header(header)

        return cls(
            site=site,
            boresight_ra_degrees=boresight_ra,
            boresight_dec_degrees=boresight_dec,
            exposure_time_seconds=exposure_time,
            observation_time=observation_time,
            track_mode=track_mode,
            track_rate_ra_arcsec_per_second=track_rate_ra,
            track_rate_dec_arcsec_per_second=track_rate_dec,
            observation_filter=observation_filter,
        )


class CameraMetadata(BaseModel):
    model: str
    pixel_size: float
    binning: int


class TelescopeMetadata(BaseModel):
    model: str
    aperture: float
    site: SiteMetadata
    camera: CameraMetadata
