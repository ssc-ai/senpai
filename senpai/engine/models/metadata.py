"""Per-frame metadata models: what the headers said, and what measurement found.

Two kinds of thing live here. Header-derived models (site, camera, telescope, frame) carry
what the FITS keywords provided, and record explicitly which capabilities are missing so a
degraded run is diagnosable rather than merely worse. Measurement-derived models (FWHM,
seeing, streak, detection) carry what the pipeline measured from the pixels.
"""

import logging
from datetime import datetime
from enum import Enum

import numpy as np
from astropy.io.fits import Header
from pydantic import BaseModel

from senpai.engine.detection.kernels import rectangle_pyramoid

#: Gap signatures already reported in this process, so a keyword a sensor never writes is
#: warned about once rather than once per frame. Workers recycle per collect, so this resets
#: with each collect.
_WARNED_CAPABILITY_GAPS: set[tuple[str, ...]] = set()


class TrackMode(Enum):
    """How the mount was driven while the frame was exposed."""

    RATE = "rate"
    SIDEREAL = "sidereal"
    UNKNOWN = "unknown"


# Define SiteMetadata first, before importing functions that use it
class SiteMetadata(BaseModel):
    """Observing-site position, needed for airmass and observability."""

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
    """The PSF scale detection measured, and the statistics behind it."""

    pixel_fwhm: float
    fwhm_stats: FWHMMetadata | None = None


class CollectionMetadata(BaseModel):
    """Identifiers tying a frame to the collect it belongs to."""

    pixel_rate_per_second: float | None = None


class ImageMetadata(BaseModel):
    """Frame geometry, timing and pointing, as the pipeline needs them."""

    image_id: str | None = None
    width: int
    height: int
    boresight_ra: float | None = None
    boresight_dec: float | None = None
    fov_min_degrees: float | None = None
    fov_max_degrees: float | None = None
    exposure_time: float | None = None  # Exposure time in seconds


class SeeingMetadata(BaseModel):
    """Seeing measured on a frame, in pixels and on the sky."""

    arcsec: float | None = None
    arcsec_stdev: float | None = None
    n_measurements: int | None = None
    pixel: float
    pixel_stdev: float | None = None


class SeeingModel(BaseModel):
    """Seeing summarised for reporting, derived from the FWHM statistics."""

    pixel_fwhm: float
    pixel_fwhm_stdev: float
    n_measurements: int

    @classmethod
    def from_fwhm_stats(cls, fwhm_stats: FWHMMetadata) -> "SeeingModel":
        """Summarise FWHM statistics into the reporting form."""
        return cls(
            pixel_fwhm=fwhm_stats.median_fwhm,
            pixel_fwhm_stdev=fwhm_stats.std_fwhm,
            n_measurements=fwhm_stats.n_measurements,
        )


class StarMetadata(BaseModel):
    """Counts of catalog and fit stars used on a frame."""

    ra: float
    dec: float
    magnitude: float
    magnitude_stdev: float
    n_measurements: int


class StreakMetadata(BaseModel):
    """A streak's measured geometry: length, orientation and width.

    The orientation is stored as its sine and cosine rather than an angle, so that a kernel
    can be built without a branch at the wrap point.
    """

    pixel_length: float
    sine_angle: float
    cosine_angle: float
    fwhm: float
    # Whether to use variable, distortion-aware kernels for this streak
    use_variable_kernel: bool = False

    def degree_angle(self) -> float:
        """Streak orientation in degrees."""
        return np.rad2deg(self.radian_angle())

    def radian_angle(self) -> float:
        """Streak orientation in radians, recovered from its sine and cosine."""
        return np.arctan2(self.sine_angle, self.cosine_angle)

    def to_pyramoid(self) -> np.ndarray:
        """Build the convolution kernel matching this streak's length, angle and width."""
        kernel = rectangle_pyramoid(self.pixel_length, self.sine_angle, self.cosine_angle, self.fwhm)

        return kernel


class FrameMetadata(BaseModel):
    """What one frame's FITS header provided, plus which capabilities it lacks."""

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
        """Return a copy whose fields are all JSON-encodable."""
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
            gaps.append(
                (
                    "observation time (e.g. DATE-OBS)",
                    "multi-frame time ordering falls back to input order; "
                    "time-based streak/rate correlation is disabled",
                )
            )
        if self.exposure_time_seconds is None:
            gaps.append(
                (
                    "exposure time (e.g. EXPTIME)",
                    "exposure-normalized photometry (per-second magnitudes in "
                    "detection/forced photometry) and rate conversion (pixels/s -> "
                    "arcsec/s) are disabled; the catalog zero-point and limiting "
                    "magnitude are still computed (instrumental, count-based)",
                )
            )
        if self.boresight_ra_degrees is None or self.boresight_dec_degrees is None:
            gaps.append(
                (
                    "boresight pointing (RA/DEC or AZ/ALT)",
                    "plate solve runs blind (no RA/Dec hint) — slower, no constrained refine tier",
                )
            )
        if self.site is None:
            gaps.append(
                (
                    "observing site (lat/long/elev)",
                    "airmass / observability metrics are disabled",
                )
            )
        if self.observation_filter is None:
            gaps.append(
                (
                    "filter (e.g. FILTER)",
                    "band-specific photometric calibration falls back to a generic band",
                )
            )
        return gaps

    def log_missing_capabilities(self, logger: logging.Logger, label: str = "frame") -> None:
        """Report which header-gated capabilities this frame cannot run, once per set of gaps.

        A sensor that omits a keyword omits it on every frame, so warning per frame says the same
        thing once per frame and buries everything else: on a 134-collect benchmark run this one
        call produced 2,880 of 4,537 warning lines, all the same two sentences. The first frame
        with a given set of gaps warns; identical sets afterwards log at debug.

        Because workers are recycled per collect, "first time" means first in this process, which
        works out to once per collect -- the granularity that matters, since each collect's
        degradations belong in its own log.
        """
        gaps = self.missing_capabilities()
        if not gaps:
            return
        signature = tuple(missing for missing, _ in gaps)
        first_time = signature not in _WARNED_CAPABILITY_GAPS
        _WARNED_CAPABILITY_GAPS.add(signature)
        emit = logger.warning if first_time else logger.debug
        emit("%s: %d header value(s) missing — degrading gracefully:", label, len(gaps))
        for missing, disabled in gaps:
            emit("  - missing %s -> %s", missing, disabled)

    @classmethod
    def from_header(cls, header: Header) -> "FrameMetadata":
        """Build frame metadata from a FITS header, tolerating absent keywords."""
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
    """Detector properties read from the header."""

    model: str
    pixel_size: float
    binning: int


class TelescopeMetadata(BaseModel):
    """Optics properties read from the header."""

    model: str
    aperture: float
    site: SiteMetadata
    camera: CameraMetadata
