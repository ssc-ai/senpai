"""Response models for the SENPAI API.

Clean, flat response models designed for agent/DOSSIER consumption.
RA/Dec is applied per detection (no raw WCS blobs). Calibrated magnitudes
are included where available. Astrometry, photometry, and seeing are
structured summaries rather than opaque headers.
"""

import contextlib
import logging
import math

import numpy as np
from pydantic import BaseModel, field_serializer

from senpai.engine.models.senpai import RateTrackFrame, RateTrackFrameSerializable, SiderealFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_float(v: float | None) -> float | None:
    """Return None for NaN/inf, otherwise return the float."""
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


# ---------------------------------------------------------------------------
# Per-frame sub-models
# ---------------------------------------------------------------------------


class AstrometrySummary(BaseModel):
    """Structured astrometry solution summary."""

    solved: bool = False
    crval_ra_deg: float | None = None
    crval_dec_deg: float | None = None
    pixel_scale_arcsec: float | None = None
    rotation_deg: float | None = None
    n_catalog_stars_matched: int | None = None
    rms_arcsec: float | None = None

    @field_serializer("crval_ra_deg", "crval_dec_deg", "pixel_scale_arcsec", "rotation_deg", "rms_arcsec")
    def serialize_floats(self, v: float | None) -> float | None:
        return round(v, 6) if v is not None else None


class PhotometryResult(BaseModel):
    """Photometric calibration for a frame."""

    zeropoint: float | None = None
    zeropoint_err: float | None = None
    limiting_mag_5sigma: float | None = None
    limiting_mag_3sigma: float | None = None
    # Per-frame completeness curve: parallel arrays (bright → faint)
    completeness_mag: list[float] | None = None
    completeness_pct: list[float] | None = None

    @field_serializer("zeropoint", "zeropoint_err", "limiting_mag_5sigma", "limiting_mag_3sigma")
    def serialize_floats(self, v: float | None) -> float | None:
        return round(v, 3) if v is not None else None


class SeeingResult(BaseModel):
    """Seeing measurement."""

    fwhm_arcsec: float | None = None
    fwhm_px: float | None = None

    @field_serializer("fwhm_arcsec", "fwhm_px")
    def serialize_floats(self, v: float | None) -> float | None:
        return round(v, 2) if v is not None else None


class FrameDetection(BaseModel):
    """A single detection in a frame."""

    x_px: float
    y_px: float
    ra_deg: float | None = None
    dec_deg: float | None = None
    snr: float | None = None
    mag: float | None = None
    mag_err: float | None = None
    mag_band: str | None = None
    fwhm_px: float | None = None
    is_streak: bool = False
    streak_length_arcsec: float = 0.0
    streak_angle_deg: float = 0.0
    streak_rate_arcsec_per_sec: float | None = None

    @field_serializer("x_px", "y_px")
    def serialize_px(self, v: float) -> float:
        return round(v, 2)

    @field_serializer("ra_deg", "dec_deg")
    def serialize_radec(self, v: float | None) -> float | None:
        return round(v, 6) if v is not None else None

    @field_serializer("snr", "fwhm_px", "streak_length_arcsec", "streak_angle_deg", "streak_rate_arcsec_per_sec")
    def serialize_floats(self, v: float | None) -> float | None:
        if v is None:
            return None
        return round(v, 2)

    @field_serializer("mag", "mag_err")
    def serialize_mag(self, v: float | None) -> float | None:
        return round(v, 3) if v is not None else None


# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------


class FrameResult(BaseModel):
    """Result for a single processed frame."""

    index: int
    tracking_mode: str | None = None  # "sidereal" | "rate"
    timestamp_utc: str | None = None
    exposure_time_s: float | None = None
    astrometry: AstrometrySummary = AstrometrySummary()
    photometry: PhotometryResult = PhotometryResult()
    seeing: SeeingResult = SeeingResult()
    detections: list[FrameDetection] = []


class DetectResponse(BaseModel):
    """Top-level response from detection endpoints."""

    frames: list[FrameResult] = []
    correlated_streaks: list[dict] = []


# ---------------------------------------------------------------------------
# Internal conversion helpers
# ---------------------------------------------------------------------------


def _astrometry_from_starfield(starfield) -> AstrometrySummary:
    """Build AstrometrySummary from a StarField."""
    if starfield is None:
        return AstrometrySummary()

    summary = AstrometrySummary(solved=starfield.fit)

    if starfield.wcs:
        wcs_model = starfield.wcs
        summary.crval_ra_deg = wcs_model.CRVAL1
        summary.crval_dec_deg = wcs_model.CRVAL2

        # Pixel scale from auto-computed WCSMetadata
        if starfield.wcs_metadata:
            summary.pixel_scale_arcsec = _safe_float(
                (starfield.wcs_metadata.x_ifov_arcsec + starfield.wcs_metadata.y_ifov_arcsec) / 2.0
            )

        # Rotation from PC matrix
        summary.rotation_deg = _safe_float(float(np.degrees(np.arctan2(wcs_model.PC2_1, wcs_model.PC1_1))))

    if starfield.astrometric_fit_stars:
        summary.n_catalog_stars_matched = len(starfield.astrometric_fit_stars)

    # RMS: compute residuals between astrometric fit stars and WCS predictions
    if starfield.astrometric_fit_stars and starfield.wcs:
        try:
            astropy_wcs = starfield.wcs.to_astropy_wcs()
            residuals_sq = []
            for star in starfield.astrometric_fit_stars:
                if star.x is not None and star.y is not None and star.ra is not None and star.dec is not None:
                    pred = astropy_wcs.all_pix2world([[star.x, star.y]], 0)
                    dra = (pred[0][0] - star.ra) * np.cos(np.radians(star.dec)) * 3600.0
                    ddec = (pred[0][1] - star.dec) * 3600.0
                    residuals_sq.append(dra**2 + ddec**2)
            if residuals_sq:
                summary.rms_arcsec = _safe_float(float(np.sqrt(np.mean(residuals_sq))))
        except Exception:
            logger.debug("Failed to compute astrometric RMS", exc_info=True)

    return summary


def _photometry_from_dict(summary_dict: dict | None) -> PhotometryResult:
    """Build PhotometryResult from a photometry_summary dict (from dataclass asdict)."""
    if summary_dict is None:
        return PhotometryResult()

    return PhotometryResult(
        zeropoint=_safe_float(summary_dict.get("zero_point")),
        zeropoint_err=_safe_float(summary_dict.get("zero_point_err")),
        limiting_mag_5sigma=_safe_float(summary_dict.get("limiting_magnitude")),
        limiting_mag_3sigma=_safe_float(summary_dict.get("limiting_magnitude_50")),
        completeness_mag=summary_dict.get("completeness_mag"),
        completeness_pct=summary_dict.get("completeness_pct"),
    )


def _photometry_from_summary(summary) -> PhotometryResult:
    """Build PhotometryResult from a SimplePhotometrySummary dataclass."""
    if summary is None:
        return PhotometryResult()

    return PhotometryResult(
        zeropoint=_safe_float(getattr(summary, "zero_point", None)),
        zeropoint_err=_safe_float(getattr(summary, "zero_point_err", None)),
        limiting_mag_5sigma=_safe_float(getattr(summary, "limiting_magnitude", None)),
        limiting_mag_3sigma=_safe_float(getattr(summary, "limiting_magnitude_50", None)),
        completeness_mag=getattr(summary, "completeness_mag", None),
        completeness_pct=getattr(summary, "completeness_pct", None),
    )


def _seeing_from_frame(frame, pixel_scale_arcsec: float | None = None) -> SeeingResult:
    """Build SeeingResult from a SiderealFrame or RateTrackFrame."""
    fwhm_px = None

    if hasattr(frame, "seeing") and frame.seeing is not None:
        fwhm_px = frame.seeing.pixel_fwhm
    elif hasattr(frame, "starfield") and frame.starfield and frame.starfield.detection_metadata:
        fwhm_px = frame.starfield.detection_metadata.pixel_fwhm

    fwhm_arcsec = None
    if fwhm_px is not None and pixel_scale_arcsec is not None:
        fwhm_arcsec = fwhm_px * pixel_scale_arcsec

    return SeeingResult(fwhm_px=_safe_float(fwhm_px), fwhm_arcsec=_safe_float(fwhm_arcsec))


def _best_mag(sat) -> tuple[float | None, float | None, str | None]:
    """Extract the best calibrated magnitude from a SatelliteInImage."""
    if sat.calibrated_magnitudes:
        for band in ["V", "R", "Clear"]:
            if band in sat.calibrated_magnitudes:
                mag = sat.calibrated_magnitudes[band]
                mag_err = sat.magnitude_errs.get(band) if sat.magnitude_errs else None
                return _safe_float(mag), _safe_float(mag_err), band
        # Fallback to first available band
        band = next(iter(sat.calibrated_magnitudes))
        mag = sat.calibrated_magnitudes[band]
        mag_err = sat.magnitude_errs.get(band) if sat.magnitude_errs else None
        return _safe_float(mag), _safe_float(mag_err), band

    if sat.instrumental_magnitude is not None:
        return _safe_float(sat.instrumental_magnitude), None, "instrumental"

    return None, None, None


def _rough_mag(counts: float | None, zp: float | None) -> float | None:
    """Compute rough calibrated magnitude from raw counts and zeropoint."""
    if counts is not None and counts > 0 and zp is not None:
        try:
            return _safe_float(float(-2.5 * math.log10(counts) + zp))
        except (ValueError, OverflowError):
            pass
    return None


# ---------------------------------------------------------------------------
# Public conversion: SiderealFrame → FrameResult
# ---------------------------------------------------------------------------


def frame_result_from_sidereal(frame: SiderealFrame) -> FrameResult:
    """Convert a SiderealFrame (from process_senpai_collect) to a FrameResult."""
    astrometry = _astrometry_from_starfield(frame.starfield)
    photometry = _photometry_from_dict(frame.photometry_summary)
    seeing = _seeing_from_frame(frame, astrometry.pixel_scale_arcsec)
    detections = _sidereal_detections(frame, astrometry.pixel_scale_arcsec)

    timestamp = frame.timestamp.isoformat() if frame.timestamp else None
    exposure = frame.frame_metadata.exposure_time_seconds if frame.frame_metadata else None
    obs_filter = frame.frame_metadata.observation_filter if frame.frame_metadata else None

    # Annotate star detections with observation filter as mag_band
    if obs_filter:
        for det in detections:
            if det.mag is not None and det.mag_band is None:
                det.mag_band = obs_filter

    return FrameResult(
        index=frame.index,
        tracking_mode="sidereal",
        timestamp_utc=timestamp,
        exposure_time_s=exposure,
        astrometry=astrometry,
        photometry=photometry,
        seeing=seeing,
        detections=detections,
    )


def _sidereal_detections(
    frame: SiderealFrame,
    pixel_scale_arcsec: float | None,
) -> list[FrameDetection]:
    detections: list[FrameDetection] = []

    astropy_wcs = None
    if frame.starfield and frame.starfield.wcs:
        with contextlib.suppress(Exception):
            astropy_wcs = frame.starfield.wcs.to_astropy_wcs()

    field_fwhm = None
    if frame.starfield and frame.starfield.detection_metadata:
        field_fwhm = frame.starfield.detection_metadata.pixel_fwhm

    zp = None
    if frame.photometry_summary:
        zp = frame.photometry_summary.get("zero_point")

    # Point-source star detections
    if frame.starfield:
        for det in frame.starfield.detections:
            ra, dec = _pix2sky(astropy_wcs, det.x, det.y)
            detections.append(
                FrameDetection(
                    x_px=det.x,
                    y_px=det.y,
                    ra_deg=ra,
                    dec_deg=dec,
                    snr=_safe_float(det.snr),
                    mag=_rough_mag(det.counts, zp),
                    fwhm_px=_safe_float(field_fwhm),
                )
            )

    # Satellite / object detections (if any)
    if frame.detections:
        for sat in frame.detections.detections:
            mag, mag_err, mag_band = _best_mag(sat)
            detections.append(
                FrameDetection(
                    x_px=sat.x,
                    y_px=sat.y,
                    ra_deg=_safe_float(sat.ra),
                    dec_deg=_safe_float(sat.dec),
                    snr=_safe_float(sat.snr),
                    mag=mag,
                    mag_err=mag_err,
                    mag_band=mag_band,
                    fwhm_px=_safe_float(sat.pixel_fwhm),
                )
            )

    # Streak candidates from sidereal streak detection
    for sc in getattr(frame, "streak_candidates", []):
        # Handle both StreakCandidate objects and dicts
        if hasattr(sc, "x"):
            streak_length_arcsec = 0.0
            if pixel_scale_arcsec and hasattr(sc, "length_pixels"):
                streak_length_arcsec = sc.length_pixels * pixel_scale_arcsec

            # Best magnitude from calibrated magnitudes
            mag_val, mag_err_val, mag_band_val = None, None, None
            if sc.calibrated_magnitudes:
                mag_band_val = next(iter(sc.calibrated_magnitudes))
                mag_val = _safe_float(sc.calibrated_magnitudes[mag_band_val])
                if sc.magnitude_errs and mag_band_val in sc.magnitude_errs:
                    mag_err_val = _safe_float(sc.magnitude_errs[mag_band_val])

            detections.append(
                FrameDetection(
                    x_px=sc.x,
                    y_px=sc.y,
                    ra_deg=_safe_float(getattr(sc, "ra", None)),
                    dec_deg=_safe_float(getattr(sc, "dec", None)),
                    snr=_safe_float(sc.peak_snr),
                    mag=mag_val,
                    mag_err=mag_err_val,
                    mag_band=mag_band_val,
                    is_streak=True,
                    streak_length_arcsec=streak_length_arcsec,
                    streak_angle_deg=sc.angle_deg,
                    streak_rate_arcsec_per_sec=_safe_float(getattr(sc, "rate_arcsec_per_sec", None)),
                )
            )

    return detections


# ---------------------------------------------------------------------------
# Public conversion: RateTrackFrame → FrameResult
# ---------------------------------------------------------------------------


def frame_result_from_rate(frame: RateTrackFrame) -> FrameResult:
    """Convert a RateTrackFrame (from process_senpai_collect) to a FrameResult."""
    astrometry = _astrometry_from_starfield(frame.starfield)
    photometry = _photometry_from_dict(frame.photometry_summary)
    seeing = _seeing_from_frame(frame, astrometry.pixel_scale_arcsec)
    detections = _rate_detections(frame)

    timestamp = frame.timestamp.isoformat() if frame.timestamp else None
    exposure = frame.frame_metadata.exposure_time_seconds if frame.frame_metadata else None

    return FrameResult(
        index=frame.index,
        tracking_mode="rate",
        timestamp_utc=timestamp,
        exposure_time_s=exposure,
        astrometry=astrometry,
        photometry=photometry,
        seeing=seeing,
        detections=detections,
    )


def _rate_detections(frame: RateTrackFrame) -> list[FrameDetection]:
    detections: list[FrameDetection] = []

    if frame.detections:
        for sat in frame.detections.detections:
            mag, mag_err, mag_band = _best_mag(sat)
            detections.append(
                FrameDetection(
                    x_px=sat.x,
                    y_px=sat.y,
                    ra_deg=_safe_float(sat.ra),
                    dec_deg=_safe_float(sat.dec),
                    snr=_safe_float(sat.snr),
                    mag=mag,
                    mag_err=mag_err,
                    mag_band=mag_band,
                    fwhm_px=_safe_float(sat.pixel_fwhm),
                    # Satellites are point sources in rate-track mode
                )
            )

    return detections


# ---------------------------------------------------------------------------
# Public conversion: RateTrackFrameSerializable → FrameResult
#   (used by the standalone /rate endpoint)
# ---------------------------------------------------------------------------


def frame_result_from_rate_serializable(
    serializable: RateTrackFrameSerializable,
    frame_index: int = 0,
) -> FrameResult:
    """Convert a RateTrackFrameSerializable to a FrameResult."""
    astrometry = _astrometry_from_starfield(serializable.starfield)

    photometry = _photometry_from_dict(serializable.photometry_summary)
    # Fall back to starfield limiting_magnitude if photometry summary is empty
    if (
        photometry.limiting_mag_5sigma is None
        and serializable.starfield
        and serializable.starfield.limiting_magnitude is not None
    ):
        photometry.limiting_mag_5sigma = _safe_float(serializable.starfield.limiting_magnitude)

    fwhm_px = serializable.seeing.pixel_fwhm if serializable.seeing else None
    fwhm_arcsec = fwhm_px * astrometry.pixel_scale_arcsec if fwhm_px and astrometry.pixel_scale_arcsec else None
    seeing = SeeingResult(fwhm_px=_safe_float(fwhm_px), fwhm_arcsec=_safe_float(fwhm_arcsec))

    detections: list[FrameDetection] = []
    if serializable.detections:
        for sat in serializable.detections.detections:
            mag, mag_err, mag_band = _best_mag(sat)
            detections.append(
                FrameDetection(
                    x_px=sat.x,
                    y_px=sat.y,
                    ra_deg=_safe_float(sat.ra),
                    dec_deg=_safe_float(sat.dec),
                    snr=_safe_float(sat.snr),
                    mag=mag,
                    mag_err=mag_err,
                    mag_band=mag_band,
                    fwhm_px=_safe_float(sat.pixel_fwhm),
                )
            )

    return FrameResult(
        index=frame_index,
        tracking_mode="rate",
        timestamp_utc=serializable.timestamp,
        exposure_time_s=(serializable.frame_metadata.exposure_time_seconds if serializable.frame_metadata else None),
        astrometry=astrometry,
        photometry=photometry,
        seeing=seeing,
        detections=detections,
    )


# ---------------------------------------------------------------------------
# Public conversion: StarField → FrameResult
#   (used by the standalone /sidereal endpoint)
# ---------------------------------------------------------------------------


def frame_result_from_starfield(
    starfield,
    fits_image=None,
    photometry_summary=None,
    frame_index: int = 0,
) -> FrameResult:
    """Convert a StarField (from process_astrometry_fits_sidereal) to a FrameResult."""
    astrometry = _astrometry_from_starfield(starfield)
    photometry = _photometry_from_summary(photometry_summary)

    fwhm_px = None
    if starfield and starfield.detection_metadata:
        fwhm_px = starfield.detection_metadata.pixel_fwhm
    fwhm_arcsec = fwhm_px * astrometry.pixel_scale_arcsec if fwhm_px and astrometry.pixel_scale_arcsec else None
    seeing = SeeingResult(fwhm_px=_safe_float(fwhm_px), fwhm_arcsec=_safe_float(fwhm_arcsec))

    # Build detections
    detections: list[FrameDetection] = []
    astropy_wcs = None
    if starfield and starfield.wcs:
        with contextlib.suppress(Exception):
            astropy_wcs = starfield.wcs.to_astropy_wcs()

    zp = getattr(photometry_summary, "zero_point", None) if photometry_summary else None

    obs_filter = None
    if starfield:
        for det in starfield.detections:
            ra, dec = _pix2sky(astropy_wcs, det.x, det.y)
            detections.append(
                FrameDetection(
                    x_px=det.x,
                    y_px=det.y,
                    ra_deg=ra,
                    dec_deg=dec,
                    snr=_safe_float(det.snr),
                    mag=_rough_mag(det.counts, zp),
                    fwhm_px=_safe_float(fwhm_px),
                )
            )

    # Extract timestamp and exposure from FITS header
    timestamp = None
    exposure = None
    if fits_image and hasattr(fits_image, "header"):
        try:
            from senpai.engine.utils.fits_io import extract_exposure_time_from_header
            from senpai.engine.utils.frame_organization import extract_uct_time_from_header

            ts = extract_uct_time_from_header(fits_image.header)
            timestamp = ts.isoformat() if ts else None
            exposure = extract_exposure_time_from_header(fits_image.header)
            from senpai.engine.utils.fits_io import extract_filter_from_header

            obs_filter = extract_filter_from_header(fits_image.header)
        except Exception:
            logger.debug("Failed to extract header metadata", exc_info=True)

    if obs_filter:
        for det in detections:
            if det.mag is not None and det.mag_band is None:
                det.mag_band = obs_filter

    return FrameResult(
        index=frame_index,
        tracking_mode="sidereal",
        timestamp_utc=timestamp,
        exposure_time_s=exposure,
        astrometry=astrometry,
        photometry=photometry,
        seeing=seeing,
        detections=detections,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pix2sky(astropy_wcs, x: float, y: float) -> tuple[float | None, float | None]:
    """Convert pixel coords to RA/Dec using an astropy WCS. Returns (None, None) on failure."""
    if astropy_wcs is None:
        return None, None
    try:
        sky = astropy_wcs.all_pix2world([[x, y]], 0)
        return float(sky[0][0]), float(sky[0][1])
    except Exception:
        return None, None
