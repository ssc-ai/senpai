"""Tests for senpai.engine.models.

Validation rules, computed properties, and serialization round-trips for the
central pydantic data models.

No network, no Astrometry.net, no FITS files on disk: WCS models are built from
plain header dicts and exercised through astropy in-memory only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from senpai.engine.models.astrometry import WCSModel, WCSStatus
from senpai.engine.models.images import ImageMetadata
from senpai.engine.models.metadata import (
    FWHMMetadata,
    SeeingModel,
    StreakMetadata,
    TrackMode,
)
from senpai.engine.models.starfield import (
    SatelliteInImage,
    StarField,
    StarInImage,
    StarInSpace,
    StarListImage,
    StarListSpace,
)
from senpai.engine.models.streak_measurement import (
    StreakMeasurement,
    StreakMeasurements,
    angular_difference,
    normalize_angle,
)


# --------------------------------------------------------------------------- #
# A minimal but valid TAN WCS (no SIP) built from a header dict.
# --------------------------------------------------------------------------- #
def _wcs_model(naxis1: int = 100, naxis2: int = 100) -> WCSModel:
    """Build a minimal but valid TAN (no-SIP) WCSModel from a header dict.

    Args:
        naxis1: Image width in pixels (also the basis for CRPIX1).
        naxis2: Image height in pixels (also the basis for CRPIX2).

    Returns:
        A WCSModel centered at RA=150, Dec=2 with a 0.001 deg/pixel scale.
    """
    return WCSModel(
        WCSAXES=2,
        NAXIS1=naxis1,
        NAXIS2=naxis2,
        CRPIX1=naxis1 / 2,
        CRPIX2=naxis2 / 2,
        PC1_1=1.0,
        PC1_2=0.0,
        PC2_1=0.0,
        PC2_2=1.0,
        CDELT1=-0.001,
        CDELT2=0.001,
        CUNIT1="deg",
        CUNIT2="deg",
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
        CRVAL1=150.0,
        CRVAL2=2.0,
    )


# --------------------------------------------------------------------------- #
# StreakMetadata — angle computed properties
# --------------------------------------------------------------------------- #
def test_streak_metadata_radian_and_degree_angle() -> None:
    """StreakMetadata converts its sine/cosine to radian and degree angles."""
    # sine/cosine of 45 degrees
    s = StreakMetadata(
        pixel_length=10.0, sine_angle=math.sin(math.radians(45)), cosine_angle=math.cos(math.radians(45)), fwhm=2.0
    )
    assert s.radian_angle() == pytest.approx(math.radians(45))
    assert s.degree_angle() == pytest.approx(45.0)


def test_streak_metadata_negative_angle() -> None:
    """A negative sine yields a -90 degree angle and disables the variable kernel."""
    s = StreakMetadata(pixel_length=5.0, sine_angle=-1.0, cosine_angle=0.0, fwhm=1.5)
    assert s.degree_angle() == pytest.approx(-90.0)
    assert s.use_variable_kernel is False


# --------------------------------------------------------------------------- #
# SeeingModel.from_fwhm_stats
# --------------------------------------------------------------------------- #
def test_seeing_model_from_fwhm_stats() -> None:
    """SeeingModel.from_fwhm_stats copies median/std/count from the stats."""
    stats = FWHMMetadata(
        n_measurements=12,
        median_fwhm=3.1,
        mean_fwhm=3.2,
        std_fwhm=0.4,
        min_fwhm=2.5,
        max_fwhm=4.0,
        fwhm_vs_position=[],
        fwhm_vs_magnitude=[],
        fwhm_vs_counts=[],
    )
    seeing = SeeingModel.from_fwhm_stats(stats)
    assert seeing.pixel_fwhm == pytest.approx(3.1)
    assert seeing.pixel_fwhm_stdev == pytest.approx(0.4)
    assert seeing.n_measurements == 12


# --------------------------------------------------------------------------- #
# normalize_angle / angular_difference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "angle,expected",
    [(0, 0), (45, 45), (180, 0), (190, 10), (-10, 170), (360, 0)],
)
def test_normalize_angle(angle: float, expected: float) -> None:
    """normalize_angle folds an angle into the [0, 180) orientation range."""
    assert normalize_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a1,a2,expected",
    [
        (10, 10, 0),
        (0, 180, 0),  # 180 ambiguity: same orientation
        (10, 100, 90),
        (170, 10, 20),  # wraps past 180
        (5, 185, 0),  # 185 == 5
    ],
)
def test_angular_difference(a1: float, a2: float, expected: float) -> None:
    """angular_difference is the symmetric orientation gap modulo 180 degrees."""
    assert angular_difference(a1, a2) == pytest.approx(expected)
    # symmetric
    assert angular_difference(a2, a1) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# StreakMeasurement normalization + aggregation
# --------------------------------------------------------------------------- #
def test_streak_measurement_normalizes_rotation() -> None:
    """StreakMeasurement normalizes its rotation into the orientation range."""
    m = StreakMeasurement(rotation=200.0, length=10.0, fwhm=2.0)
    assert m.rotation == pytest.approx(20.0)


def test_streak_measurements_mean_and_median() -> None:
    """StreakMeasurements aggregates its members via mean and median."""
    ms = StreakMeasurements(
        header=StreakMeasurement(rotation=30.0, length=100.0, fwhm=2.0),
        frame_extraction=StreakMeasurement(rotation=32.0, length=102.0, fwhm=2.2),
        frame_to_frame=StreakMeasurement(rotation=31.0, length=101.0, fwhm=2.1),
    )
    mean = ms.mean_measurement()
    # mean rotation should be near 31 degrees, length near 101
    assert 28.0 < mean.rotation < 34.0
    assert 99.0 < mean.length < 103.0
    median = ms.median_measurement()
    assert 99.0 <= median.length <= 103.0


def test_streak_measurements_empty_returns_zeros() -> None:
    """An empty StreakMeasurements aggregates to zeros with no FWHM."""
    ms = StreakMeasurements()
    mean = ms.mean_measurement()
    assert mean.rotation == 0.0
    assert mean.length == 0.0
    assert mean.fwhm is None


def test_streak_measurements_sigma_clip() -> None:
    """Sigma-clipped aggregation rejects a gross length/rotation outlier."""
    ms = StreakMeasurements(
        header=StreakMeasurement(rotation=30.0, length=100.0, fwhm=2.0),
        cross_correlation=StreakMeasurement(rotation=31.0, length=101.0, fwhm=2.0),
        frame_extraction=StreakMeasurement(rotation=29.0, length=99.0, fwhm=2.0),
        frame_to_frame=StreakMeasurement(rotation=150.0, length=500.0, fwhm=8.0),  # outlier
    )
    clipped = ms.sigma_clipped_mean_measurement(sigma=2.0)
    # Outlier length should be rejected — result close to ~100, not pulled to 500
    assert clipped.length < 200.0


# --------------------------------------------------------------------------- #
# StarInImage / StarInSpace serialization (field_serializer rounding)
# --------------------------------------------------------------------------- #
def test_star_in_image_rounds_floats() -> None:
    """StarInImage rounds its float fields on serialization."""
    s = StarInImage(x=1.23456, y=2.98765, counts=100.5555, snr=12.3456)
    dumped = s.model_dump()
    assert dumped["x"] == 1.23
    assert dumped["y"] == 2.99
    assert dumped["counts"] == 100.56
    assert dumped["snr"] == 12.35


def test_star_in_image_none_passthrough() -> None:
    """Unset optional StarInImage fields serialize as None."""
    s = StarInImage(x=1.0, y=2.0)
    dumped = s.model_dump()
    assert dumped["counts"] is None
    assert dumped["snr"] is None


def test_star_in_space_radec_rounding_and_roundtrip() -> None:
    """StarInSpace rounds RA/Dec/magnitude and round-trips through validation."""
    s = StarInSpace(ra=150.123456789, dec=2.987654321, magnitude=15.55555)
    dumped = s.model_dump()
    assert dumped["ra"] == 150.1235
    assert dumped["dec"] == 2.9877
    assert dumped["magnitude"] == 15.56
    # round-trip back into the model
    again = StarInSpace.model_validate(dumped)
    assert again.ra == pytest.approx(150.1235)


def test_star_in_space_magnitudes_fallback_to_primary() -> None:
    """An empty magnitudes dict is synthesized to a Primary band on dump."""
    # magnitudes empty but magnitude set -> serializer synthesizes Primary
    s = StarInSpace(ra=10.0, dec=20.0, magnitude=14.0, magnitudes={})
    dumped = s.model_dump()
    assert dumped["magnitudes"] == {"Primary": 14.0}


def test_satellite_in_image_mag_dict_rounding() -> None:
    """SatelliteInImage rounds its magnitude dicts and instrumental magnitude."""
    sat = SatelliteInImage(
        x=5.0,
        y=6.0,
        calibrated_magnitudes={"Johnson_V": 12.123456},
        magnitude_errs={"Johnson_V": 0.0123456},
        instrumental_magnitude=-7.654321,
    )
    dumped = sat.model_dump()
    assert dumped["calibrated_magnitudes"]["Johnson_V"] == 12.123
    assert dumped["magnitude_errs"]["Johnson_V"] == 0.012
    assert dumped["instrumental_magnitude"] == -7.654


# --------------------------------------------------------------------------- #
# StarListSpace / StarListImage helpers
# --------------------------------------------------------------------------- #
def test_star_list_space_centers_radec() -> None:
    """StarListSpace.centers_radec returns an (N, 2) RA/Dec array."""
    meta = ImageMetadata(width=100, height=100)
    sl = StarListSpace(
        stars=[
            StarInSpace(ra=10.0, dec=20.0),
            StarInSpace(ra=11.0, dec=21.0),
        ],
        image_metadata=meta,
    )
    centers = sl.centers_radec()
    assert centers.shape == (2, 2)
    np.testing.assert_allclose(centers[0], [10.0, 20.0])


def test_star_list_image_centers_xy() -> None:
    """StarListImage.centers_xy returns an (N, 3) x/y/counts array."""
    meta = ImageMetadata(width=50, height=50)
    direct = StarListImage(
        detections=[
            StarInImage(x=3.0, y=4.0, counts=5.0),
            StarInImage(x=7.0, y=8.0, counts=9.0),
        ],
        image_metadata=meta,
    )
    centers = direct.centers_xy()
    assert centers.shape == (2, 3)
    np.testing.assert_allclose(centers[0], [3.0, 4.0, 5.0])


def test_star_list_image_from_starfield_builds_from_detections() -> None:
    """StarListImage.from_starfield builds the solve input from detections."""
    # from_starfield builds the solve input from ``starfield.detections``
    # (x, y, counts), skipping detections without pixel coordinates.
    meta = ImageMetadata(width=50, height=50)
    sf = StarField(
        detections=[
            StarInImage(x=1.0, y=2.0, counts=10.0),
            StarInImage(x=3.0, y=4.0, counts=20.0),
        ],
        image_metadata=meta,
        wcs=None,
    )
    sli = StarListImage.from_starfield(sf)
    assert len(sli.detections) == 2
    assert sli.detections[0].x == 1.0
    assert sli.detections[0].counts == 10.0
    assert sli.image_metadata is sf.image_metadata

def test_starfield_no_wcs_status_default() -> None:
    """A StarField with no WCS reports NO_WCS status and no WCS metadata."""
    meta = ImageMetadata(width=100, height=100)
    sf = StarField(detections=[], image_metadata=meta, wcs=None)
    assert sf.wcs_status == WCSStatus.NO_WCS
    assert sf.wcs_metadata is None


def test_starfield_creates_wcs_metadata_from_wcs() -> None:
    """Constructing a StarField with a WCS populates its WCS metadata."""
    meta = ImageMetadata(width=100, height=100)
    sf = StarField(detections=[], image_metadata=meta, wcs=_wcs_model())
    # model_validator(mode="after") populates wcs_metadata
    assert sf.wcs_metadata is not None
    assert sf.wcs_metadata.RA_center_deg == pytest.approx(150.0, abs=0.5)


def test_starfield_catalog_centers_xy_limiting_magnitude() -> None:
    """catalog_centers_xy applies an optional limiting-magnitude cutoff."""
    meta = ImageMetadata(width=100, height=100)
    sf = StarField(
        detections=[],
        image_metadata=meta,
        wcs=None,
        catalog_stars=[
            StarInSpace(ra=1.0, dec=1.0, x=10.0, y=10.0, magnitude=12.0),
            StarInSpace(ra=2.0, dec=2.0, x=20.0, y=20.0, magnitude=19.0),
        ],
    )
    # No limit -> both
    assert sf.catalog_centers_xy().shape[0] == 2
    # limit at 15 -> only the bright one
    assert sf.catalog_centers_xy(limiting_magnitude=15.0).shape[0] == 1


def test_starfield_catalog_centers_xy_none_when_no_catalog() -> None:
    """catalog_centers_xy returns None when there are no catalog stars."""
    meta = ImageMetadata(width=100, height=100)
    sf = StarField(detections=[], image_metadata=meta, wcs=None)
    assert sf.catalog_centers_xy() is None


# --------------------------------------------------------------------------- #
# WCSModel round-trips and coordinate transforms
# --------------------------------------------------------------------------- #
def test_wcs_model_to_astropy_and_back_pixel_world() -> None:
    """WCSModel round-trips pixel<->world coordinates through astropy."""
    wcs = _wcs_model()
    # center pixel maps near CRVAL
    ra, dec = wcs.pix2world_0based(49.0, 49.0)
    assert ra == pytest.approx(150.0, abs=0.5)
    assert dec == pytest.approx(2.0, abs=0.5)
    # round-trip world->pix->world
    x, y = wcs.world2pix_0based(ra, dec)
    ra2, dec2 = wcs.pix2world_0based(x, y)
    assert ra2 == pytest.approx(ra, abs=1e-3)
    assert dec2 == pytest.approx(dec, abs=1e-3)


def test_wcs_model_get_boresight() -> None:
    """WCSModel.get_boresight returns the CRVAL RA/Dec pair."""
    wcs = _wcs_model()
    assert wcs.get_boresight() == (150.0, 2.0)


def test_wcs_model_fov_and_dimensions() -> None:
    """get_fov_and_dimensions reports pixel dims and a proportional FOV."""
    wcs = _wcs_model(naxis1=100, naxis2=200)
    fov_w, fov_h, pw, ph = wcs.get_fov_and_dimensions()
    assert pw == 100
    assert ph == 200
    assert fov_w > 0 and fov_h > 0
    # height spans more pixels -> larger FOV
    assert fov_h > fov_w


def test_wcs_model_serialization_roundtrip() -> None:
    """WCSModel survives a model_dump / model_validate round-trip."""
    wcs = _wcs_model()
    dumped = wcs.model_dump()
    again = WCSModel.model_validate(dumped)
    assert again.CRVAL1 == wcs.CRVAL1
    assert again.NAXIS1 == wcs.NAXIS1


def test_wcs_model_extra_sip_coeffs_allowed() -> None:
    """WCSModel allows extra higher-order SIP coefficients."""
    # extra='allow' permits higher-order SIP coefficients beyond the declared ones
    wcs = WCSModel(**{**_wcs_model().model_dump(), "A_3_0": 1e-9})
    assert wcs.model_dump()["A_3_0"] == 1e-9


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
def test_track_mode_values() -> None:
    """The TrackMode enum exposes its expected string values."""
    assert TrackMode.RATE.value == "rate"
    assert TrackMode.SIDEREAL.value == "sidereal"
    assert TrackMode.UNKNOWN.value == "unknown"


# --------------------------------------------------------------------------- #
# FrameShift correlation + chain-status glyphs (EOS engine port)
# --------------------------------------------------------------------------- #
def test_frame_shift_correlation_field() -> None:
    """FrameShift carries optional correlation and error_message fields."""
    from senpai.engine.models.senpai import FrameShift

    shift = FrameShift(source_index=0, target_index=1)
    assert shift.correlation is None
    shift.correlation = 0.997
    assert shift.correlation == 0.997
    # their error_message field still present
    assert shift.error_message is None


@pytest.mark.parametrize(
    ("processed", "is_valid", "correlation", "expected"),
    [
        (False, True, None, "❓"),
        (True, False, None, "❌"),
        (True, True, None, "✅"),
        (True, True, 0.95, "✅"),
        (True, True, 0.42, "⚠️0.42"),
    ],
)
def test_shift_status_glyphs(
    processed: bool, is_valid: bool, correlation: float | None, expected: str
) -> None:
    """_shift_status maps a FrameShift's state to its status glyph."""
    from senpai.engine.models.senpai import FrameShift, _shift_status

    shift = FrameShift(
        source_index=0,
        target_index=1,
        processed=processed,
        is_valid=is_valid,
        correlation=correlation,
    )
    assert _shift_status(shift) == expected


def test_star_in_image_fwhm_rounds() -> None:
    """StarInImage rounds its FWHM on dump and leaves it None when unset."""
    star = StarInImage(x=1.0, y=2.0, fwhm=3.14159)
    assert star.model_dump()["fwhm"] == 3.14
    assert StarInImage(x=1.0, y=2.0).fwhm is None
