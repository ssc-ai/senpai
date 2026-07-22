"""Pydantic metadata models describing frames, sites, optics, and detections.

Collects the lightweight data models used throughout the engine to carry
header-derived and computed metadata: observing site and telescope/camera
descriptions, per-frame observation metadata, seeing/FWHM statistics, and
per-detection star and streak descriptors.
"""

import logging
from datetime import datetime
from enum import Enum

import numpy as np
from astropy.io.fits import Header
from pydantic import BaseModel

from senpai.engine.detection.kernels import rectangle_pyramoid


class TrackMode(Enum):
    """Telescope tracking mode during an exposure."""

    RATE = "rate"
    SIDEREAL = "sidereal"
    UNKNOWN = "unknown"


# Define SiteMetadata first, before importing functions that use it
class SiteMetadata(BaseModel):
    """Geographic location of an observing site.

    Attributes:
        name: Human-readable site name, if known.
        latitude: Site latitude in degrees.
        longitude: Site longitude in degrees.
        altitude_km: Site altitude in kilometers, if known.
    """

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
    """Summary of the point-source detection characteristics for a frame.

    Attributes:
        pixel_fwhm: Representative point-source FWHM in pixels.
        fwhm_stats: Detailed FWHM statistics, if available.
    """

    pixel_fwhm: float
    fwhm_stats: FWHMMetadata | None = None


class CollectionMetadata(BaseModel):
    """Metadata describing a multi-frame collection.

    Attributes:
        pixel_rate_per_second: Apparent object rate across the collection in
            pixels per second, if known.
    """

    pixel_rate_per_second: float | None = None


class ImageMetadata(BaseModel):
    """Per-image geometry and pointing metadata.

    Attributes:
        image_id: Image identifier, if known.
        width: Image width in pixels.
        height: Image height in pixels.
        boresight_ra: Boresight right ascension in degrees, if known.
        boresight_dec: Boresight declination in degrees, if known.
        fov_min_degrees: Smaller field-of-view extent in degrees, if known.
        fov_max_degrees: Larger field-of-view extent in degrees, if known.
        exposure_time: Exposure time in seconds, if known.
    """

    image_id: str | None = None
    width: int
    height: int
    boresight_ra: float | None = None
    boresight_dec: float | None = None
    fov_min_degrees: float | None = None
    fov_max_degrees: float | None = None
    exposure_time: float | None = None  # Exposure time in seconds


class SeeingMetadata(BaseModel):
    """Seeing (atmospheric blur) measurement for a frame.

    Attributes:
        arcsec: Seeing FWHM in arcseconds, if computed.
        arcsec_stdev: Standard deviation of the arcsecond seeing, if computed.
        n_measurements: Number of measurements contributing to the estimate.
        pixel: Seeing FWHM in pixels.
        pixel_stdev: Standard deviation of the pixel seeing, if computed.
    """

    arcsec: float | None = None
    arcsec_stdev: float | None = None
    n_measurements: int | None = None
    pixel: float
    pixel_stdev: float | None = None


class SeeingModel(BaseModel):
    """Compact seeing model derived from FWHM statistics.

    Attributes:
        pixel_fwhm: Representative point-source FWHM in pixels.
        pixel_fwhm_stdev: Standard deviation of the pixel FWHM.
        n_measurements: Number of measurements contributing to the model.
    """

    pixel_fwhm: float
    pixel_fwhm_stdev: float
    n_measurements: int

    @classmethod
    def from_fwhm_stats(cls, fwhm_stats: FWHMMetadata) -> "SeeingModel":
        """Build a SeeingModel from collected FWHM statistics.

        Args:
            fwhm_stats: Aggregated FWHM statistics for the frame.

        Returns:
            A seeing model populated from the median FWHM, its standard
            deviation, and the measurement count.
        """
        return cls(
            pixel_fwhm=fwhm_stats.median_fwhm,
            pixel_fwhm_stdev=fwhm_stats.std_fwhm,
            n_measurements=fwhm_stats.n_measurements,
        )


class StarMetadata(BaseModel):
    """Photometric summary for a matched catalog star.

    Attributes:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        magnitude: Estimated magnitude.
        magnitude_stdev: Standard deviation of the magnitude estimate.
        n_measurements: Number of measurements contributing to the estimate.
    """

    ra: float
    dec: float
    magnitude: float
    magnitude_stdev: float
    n_measurements: int


class StreakMetadata(BaseModel):
    """Geometry of a detected streak used to build a matched-filter kernel.

    The orientation is stored as a sine/cosine pair to avoid angle wraparound.

    Attributes:
        pixel_length: Streak length in pixels.
        sine_angle: Sine of the streak orientation angle.
        cosine_angle: Cosine of the streak orientation angle.
        fwhm: Cross-sectional FWHM of the streak in pixels.
        use_variable_kernel: Whether to use variable, distortion-aware kernels
            for this streak.
    """

    pixel_length: float
    sine_angle: float
    cosine_angle: float
    fwhm: float
    # Whether to use variable, distortion-aware kernels for this streak
    use_variable_kernel: bool = False

    def degree_angle(self) -> float:
        """Return the streak orientation in degrees.

        Returns:
            The orientation angle in degrees.
        """
        return np.rad2deg(self.radian_angle())

    def radian_angle(self) -> float:
        """Return the streak orientation in radians.

        Returns:
            The orientation angle in radians, from the sine/cosine pair.
        """
        return np.arctan2(self.sine_angle, self.cosine_angle)

    def to_pyramoid(self) -> np.ndarray:
        """Build the rectangular-pyramoid matched-filter kernel for the streak.

        Returns:
            The kernel array sized and oriented to match this streak's length,
            orientation, and FWHM.
        """
        kernel = rectangle_pyramoid(self.pixel_length, self.sine_angle, self.cosine_angle, self.fwhm)

        return kernel


class FrameMetadata(BaseModel):
    """Header-derived observation metadata for a single frame.

    Every field is optional so frames with sparse or absent headers (e.g. a raw
    focus frame with only NAXIS) still build a FrameMetadata; downstream features
    that need a value gate on its presence (see :meth:`missing_capabilities`)
    rather than crashing the run.

    Attributes:
        exposure_time_seconds: Exposure time in seconds.
        observation_time: UTC observation time.
        site: Observing site location.
        track_mode: Telescope tracking mode during the exposure.
        track_rate_ra_arcsec_per_second: RA tracking rate in arcsec/second.
        track_rate_dec_arcsec_per_second: Dec tracking rate in arcsec/second.
        boresight_ra_degrees: Boresight right ascension in degrees.
        boresight_dec_degrees: Boresight declination in degrees.
        observation_filter: Photometric filter/band name.
    """

    # Optional so frames with sparse/absent headers (e.g. a raw focus frame with
    # only NAXIS) still build a FrameMetadata. Downstream features that need a
    # value gate on its presence (see FrameMetadata.missing_capabilities) rather
    # than crashing the run.
    exposure_time_seconds: float | None = None
    observation_time: datetime | None = None
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

    def missing_capabilities(self) -> list[tuple[str, str]]:
        """Audit which header-derived values are absent and what each disables.

        Returns a list of ``(missing_data, disabled_capability)`` pairs so a
        caller can log verbosely *what* could not run and *why*. Empty list
        means every header-gated feature has the data it needs.
        """
        gaps: list[tuple[str, str]] = []
        if self.observation_time is None:
            gaps.append((
                "observation time (e.g. DATE-OBS)",
                "multi-frame time ordering falls back to input order; "
                "time-based streak/rate correlation is disabled",
            ))
        if self.exposure_time_seconds is None:
            gaps.append((
                "exposure time (e.g. EXPTIME)",
                "exposure-normalized photometry (per-second magnitudes in "
                "detection/forced photometry) and rate conversion (pixels/s -> "
                "arcsec/s) are disabled; the catalog zero-point and limiting "
                "magnitude are still computed (instrumental, count-based)",
            ))
        if self.boresight_ra_degrees is None or self.boresight_dec_degrees is None:
            gaps.append((
                "boresight pointing (RA/DEC or AZ/ALT)",
                "plate solve runs blind (no RA/Dec hint) — slower, no constrained refine tier",
            ))
        if self.site is None:
            gaps.append((
                "observing site (lat/long/elev)",
                "airmass / observability metrics are disabled",
            ))
        if self.observation_filter is None:
            gaps.append((
                "filter (e.g. FILTER)",
                "band-specific photometric calibration falls back to a generic band",
            ))
        return gaps

    def log_missing_capabilities(self, logger: logging.Logger, label: str = "frame") -> None:
        """Emit one warning per missing-header capability gap (verbose by design).

        Args:
            logger: Logger to emit the warnings through.
            label: Human-readable label identifying the frame in the log lines.
        """
        gaps = self.missing_capabilities()
        if not gaps:
            return
        logger.warning(
            "%s: %d header value(s) missing — degrading gracefully:", label, len(gaps)
        )
        for missing, disabled in gaps:
            logger.warning("  - missing %s -> %s", missing, disabled)

    @classmethod
    def from_header(cls, header: Header) -> "FrameMetadata":
        """Build a FrameMetadata by extracting values from a FITS header.

        Args:
            header: The FITS header to parse.

        Returns:
            A FrameMetadata populated with whatever site, pointing, timing,
            tracking, and filter values could be extracted; absent values are
            left as ``None``.
        """
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
    """Camera description.

    Attributes:
        model: Camera model name.
        pixel_size: Physical pixel size (typically in microns).
        binning: On-chip binning factor.
    """

    model: str
    pixel_size: float
    binning: int


class TelescopeMetadata(BaseModel):
    """Telescope description and associated site and camera.

    Attributes:
        model: Telescope model name.
        aperture: Aperture diameter (typically in meters).
        site: Observing site for the telescope.
        camera: Camera mounted on the telescope.
    """

    model: str
    aperture: float
    site: SiteMetadata
    camera: CameraMetadata
