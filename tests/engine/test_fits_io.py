"""Behavioral tests for senpai.engine.utils.fits_io.

Covers the config-driven FITS header extraction helpers: coordinate parsing
(sexagesimal / float / NSEW), exposure & observation time, observing site
(sexagesimal + km/m altitude), boresight (RA/Dec and Alt/Az fallback),
tracking-rate unit normalization, track-mode classification, and filter
normalization. Configs are process-wide singletons, so we initialize in a
fixture and snapshot/restore any header-key lists we mutate per test.

All coordinate values below are synthetic round constants chosen only to
exercise the parsers; they are not real observing-site positions.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
from astropy.io.fits import Header

from senpai.core.config import AppConfig, HeadersConfig, get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.metadata import TrackMode
from senpai.engine.utils import fits_io


@pytest.fixture(scope="module", autouse=True)
def _config() -> AppConfig:
    """Initialize the process-wide config from local.yaml for the module.

    Returns:
        The initialized AppConfig singleton.
    """
    initialize_config(CONFIG_DIR / "local.yaml")
    return get_config()


@pytest.fixture
def restore_headers() -> Iterator[HeadersConfig]:
    """Snapshot the mutable header-key lists and restore them after the test.

    Tests that need multiple candidate keys / alternate formats mutate the
    in-memory config; this keeps the singleton clean for sibling tests.

    Yields:
        The live ``headers`` sub-config, which the test may mutate freely.
    """
    cfg = get_config().headers
    snapshot = {
        "exptime": list(cfg.exposure_time.exposure_time_keys),
        "obs_keys": list(cfg.observation_time.observation_time_keys),
        "obs_fmt": cfg.observation_time.format,
        "lat": list(cfg.site.site_latitude_keys),
        "lon": list(cfg.site.site_longitude_keys),
        "alt": list(cfg.site.site_altitude_keys),
        "alt_unit": cfg.site.altitude_unit,
        "pos_fmt": cfg.site.positional_format,
        "ra_keys": list(cfg.pointing.target_ra_keys),
        "dec_keys": list(cfg.pointing.target_dec_keys),
        "ra_dec_fmt": cfg.pointing.ra_dec_format,
        "ra_units": cfg.pointing.ra_units,
        "dec_units": cfg.pointing.dec_units,
        "az_keys": list(cfg.pointing.boresight_azimuth_keys),
        "alt_keys": list(cfg.pointing.boresight_altitude_keys),
        "ra_rate": list(cfg.tracking.track_ra_rate_keys),
        "dec_rate": list(cfg.tracking.track_dec_rate_keys),
        "ra_rate_unit": cfg.tracking.track_ra_rate_unit,
        "dec_rate_unit": cfg.tracking.track_dec_rate_unit,
        "mode_keys": list(cfg.tracking.track_mode_keys),
    }
    yield cfg
    cfg.exposure_time.exposure_time_keys = snapshot["exptime"]
    cfg.observation_time.observation_time_keys = snapshot["obs_keys"]
    cfg.observation_time.format = snapshot["obs_fmt"]
    cfg.site.site_latitude_keys = snapshot["lat"]
    cfg.site.site_longitude_keys = snapshot["lon"]
    cfg.site.site_altitude_keys = snapshot["alt"]
    cfg.site.altitude_unit = snapshot["alt_unit"]
    cfg.site.positional_format = snapshot["pos_fmt"]
    cfg.pointing.target_ra_keys = snapshot["ra_keys"]
    cfg.pointing.target_dec_keys = snapshot["dec_keys"]
    cfg.pointing.ra_dec_format = snapshot["ra_dec_fmt"]
    cfg.pointing.ra_units = snapshot["ra_units"]
    cfg.pointing.dec_units = snapshot["dec_units"]
    cfg.pointing.boresight_azimuth_keys = snapshot["az_keys"]
    cfg.pointing.boresight_altitude_keys = snapshot["alt_keys"]
    cfg.tracking.track_ra_rate_keys = snapshot["ra_rate"]
    cfg.tracking.track_dec_rate_keys = snapshot["dec_rate"]
    cfg.tracking.track_ra_rate_unit = snapshot["ra_rate_unit"]
    cfg.tracking.track_dec_rate_unit = snapshot["dec_rate_unit"]
    cfg.tracking.track_mode_keys = snapshot["mode_keys"]


# --------------------------------------------------------------------------
# Pure coordinate parsers (no config needed)
# --------------------------------------------------------------------------

def test_sexagesimal_degrees_positive() -> None:
    """A positive sexagesimal degrees string converts to decimal degrees."""
    assert fits_io.sexagesimal_to_decimal("+12 00 00", "degrees") == pytest.approx(12.0, abs=1e-5)


def test_sexagesimal_degrees_negative() -> None:
    """A negative sexagesimal degrees string keeps its sign in decimal degrees."""
    assert fits_io.sexagesimal_to_decimal("-07 30 00", "degrees") == pytest.approx(-7.5, abs=1e-5)


def test_sexagesimal_hours_multiplies_by_15() -> None:
    """A sexagesimal hours string is scaled by 15 into degrees."""
    # 6h -> degrees
    assert fits_io.sexagesimal_to_decimal("06 00 00", "hours") == pytest.approx(90.0, abs=1e-4)


def test_sexagesimal_colon_delimiters() -> None:
    """Colon-delimited sexagesimal strings parse correctly."""
    assert fits_io.sexagesimal_to_decimal("10:30:00", "degrees") == pytest.approx(10.5, abs=1e-9)


def test_sexagesimal_two_part_degrees_minutes() -> None:
    """A two-part degrees/minutes string parses with no seconds."""
    assert fits_io.sexagesimal_to_decimal("10 30", "degrees") == pytest.approx(10.5, abs=1e-9)


def test_sexagesimal_single_value_passthrough() -> None:
    """A single decimal value passes through unchanged."""
    assert fits_io.sexagesimal_to_decimal("42.5", "degrees") == pytest.approx(42.5, abs=1e-9)


def test_float_nsew_south_is_negative() -> None:
    """A South suffix makes the decimal value negative."""
    assert fits_io.float_nsew_to_decimal("10.0 S") == pytest.approx(-10.0)


def test_float_nsew_west_is_negative() -> None:
    """A West suffix makes the decimal value negative."""
    assert fits_io.float_nsew_to_decimal("20.0 W") == pytest.approx(-20.0)


def test_float_nsew_north_positive() -> None:
    """A North suffix keeps the decimal value positive."""
    assert fits_io.float_nsew_to_decimal("30.0 N") == pytest.approx(30.0)


def test_convert_to_decimal_degrees_float_hours() -> None:
    """Float-format hours are scaled by 15 into degrees."""
    assert fits_io.convert_to_decimal_degrees("1.0", fmt="float", units="hours") == pytest.approx(15.0)


def test_convert_to_decimal_degrees_float_degrees() -> None:
    """Float-format degrees pass through unchanged."""
    assert fits_io.convert_to_decimal_degrees("123.5", fmt="float", units="degrees") == pytest.approx(123.5)


def test_convert_to_decimal_degrees_unsupported_format_raises() -> None:
    """An unsupported coordinate format raises ValueError."""
    with pytest.raises(ValueError):
        fits_io.convert_to_decimal_degrees("1.0", fmt="bogus", units="degrees")


def test_convert_to_decimal_kilometers_from_meters() -> None:
    """Meters convert to kilometers."""
    assert fits_io.convert_to_decimal_kilometers("2000", units="meters") == pytest.approx(2.0)


def test_convert_to_decimal_kilometers_passthrough_km() -> None:
    """Kilometers pass through unchanged."""
    assert fits_io.convert_to_decimal_kilometers("1.5", units="kilometers") == pytest.approx(1.5)


def test_convert_to_decimal_kilometers_unknown_unit_raises() -> None:
    """An unknown altitude unit raises ValueError."""
    with pytest.raises(ValueError):
        fits_io.convert_to_decimal_kilometers("100", units="parsecs")


def test_extract_header_value_present_and_absent() -> None:
    """extract_header_value returns the value if present, else None."""
    h = Header()
    h["FOO"] = 7
    assert fits_io.extract_header_value(h, "FOO") == 7
    assert fits_io.extract_header_value(h, "MISSING") is None


# --------------------------------------------------------------------------
# Exposure time
# --------------------------------------------------------------------------

def test_exposure_time_basic() -> None:
    """A string EXPTIME header parses to a float exposure time."""
    h = Header()
    h["EXPTIME"] = "12.5"
    assert fits_io.extract_exposure_time_from_header(h) == pytest.approx(12.5)


def test_exposure_time_missing_returns_none() -> None:
    """A missing exposure-time header returns None."""
    assert fits_io.extract_exposure_time_from_header(Header()) is None


def test_exposure_time_multiple_candidate_keys(restore_headers: HeadersConfig) -> None:
    """The first present of several candidate exposure-time keys is used."""
    cfg = restore_headers
    cfg.exposure_time.exposure_time_keys = ["EXPOSURE", "EXPTIME"]
    h = Header()
    h["EXPTIME"] = 3.0  # only the second candidate present
    assert fits_io.extract_exposure_time_from_header(h) == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Observation time
# --------------------------------------------------------------------------

def test_observation_time_iso() -> None:
    """An ISO DATE-OBS header parses to a datetime."""
    h = Header()
    h["DATE-OBS"] = "2023-05-01T03:22:11.5"
    t = fits_io.extract_observation_time_from_header(h)
    assert isinstance(t, datetime)
    assert (t.year, t.month, t.day, t.hour, t.minute, t.second) == (2023, 5, 1, 3, 22, 11)


def test_observation_time_custom_format(restore_headers: HeadersConfig) -> None:
    """A configured non-ISO datetime format is honored."""
    cfg = restore_headers
    cfg.observation_time.format = "%Y/%m/%d %H:%M:%S"
    h = Header()
    h["DATE-OBS"] = "2021/12/25 18:30:00"
    t = fits_io.extract_observation_time_from_header(h)
    assert (t.year, t.month, t.day, t.hour, t.minute) == (2021, 12, 25, 18, 30)


def test_observation_time_falls_back_to_broad_parser() -> None:
    """Observation-time extraction falls back to the broad datetime parser."""
    # No configured DATE-OBS value, but a DATE_TIME header arrow can parse.
    h = Header()
    h["DATE-OBS"] = "2020-01-02T00:00:00"  # used by the broad fallback too
    t = fits_io.extract_observation_time_from_header(h)
    assert (t.year, t.month, t.day) == (2020, 1, 2)


# --------------------------------------------------------------------------
# Observing site (synthetic coordinates only)
# --------------------------------------------------------------------------

def test_site_sexagesimal_lat_lon() -> None:
    """Sexagesimal site lat/lon plus km altitude build an observing site."""
    h = Header()
    h["SITELAT"] = "10 00 00"
    h["SITELONG"] = "20 00 00"
    h["SITEALT"] = "1.5"  # config altitude_unit is kilometers
    site = fits_io.extract_observing_site_from_header(h)
    assert site is not None
    assert site.latitude == pytest.approx(10.0, abs=1e-5)
    assert site.longitude == pytest.approx(20.0, abs=1e-5)
    assert site.altitude_km == pytest.approx(1.5)


def test_site_altitude_meters_unit(restore_headers: HeadersConfig) -> None:
    """A meters altitude unit converts the site altitude to kilometers."""
    cfg = restore_headers
    cfg.site.altitude_unit = "meters"
    h = Header()
    h["SITELAT"] = "10 00 00"
    h["SITELONG"] = "20 00 00"
    h["SITEALT"] = "2000"
    site = fits_io.extract_observing_site_from_header(h)
    assert site.altitude_km == pytest.approx(2.0)


def test_site_missing_lat_lon_returns_none() -> None:
    """A site with altitude but no lat/lon returns None."""
    h = Header()
    h["SITEALT"] = "1.0"
    assert fits_io.extract_observing_site_from_header(h) is None


def test_site_float_positional_format(restore_headers: HeadersConfig) -> None:
    """A float positional format parses plain-decimal site lat/lon."""
    cfg = restore_headers
    cfg.site.positional_format = "float"
    cfg.site.altitude_unit = "kilometers"
    h = Header()
    h["SITELAT"] = "10.5"
    h["SITELONG"] = "20.5"
    site = fits_io.extract_observing_site_from_header(h)
    assert site.latitude == pytest.approx(10.5)
    assert site.longitude == pytest.approx(20.5)


# --------------------------------------------------------------------------
# Boresight (synthetic sky coordinates only)
# --------------------------------------------------------------------------

def test_boresight_ra_dec_sexagesimal_hours() -> None:
    """Sexagesimal-hours RA and degrees Dec yield the boresight in degrees."""
    h = Header()
    h["OBJCTRA"] = "06 00 00"
    h["OBJCTDEC"] = "+12 00 00"
    ra, dec = fits_io.extract_boresight_from_header(h)
    assert ra == pytest.approx(90.0, abs=1e-4)
    assert dec == pytest.approx(12.0, abs=1e-4)


def test_boresight_missing_returns_none_none() -> None:
    """A header with no boresight info returns (None, None)."""
    ra, dec = fits_io.extract_boresight_from_header(Header())
    assert ra is None and dec is None


def test_boresight_altaz_fallback(restore_headers: HeadersConfig) -> None:
    """Alt/Az plus site and time fall back to a valid RA/Dec boresight."""
    # local.yaml configures CENTAZ/CENTALT; supply az/alt + site + time and
    # verify Alt/Az -> RA/Dec produces a valid sky coordinate.
    h = Header()
    h["CENTAZ"] = "120.0"
    h["CENTALT"] = "45.0"
    h["SITELAT"] = "10 00 00"
    h["SITELONG"] = "20 00 00"
    h["SITEALT"] = "1.5"
    h["DATE-OBS"] = "2023-05-01T03:22:11.5"
    ra, dec = fits_io.extract_boresight_from_header(h)
    assert ra is not None and dec is not None
    assert 0.0 <= ra < 360.0
    assert -90.0 <= dec <= 90.0


# --------------------------------------------------------------------------
# Track rates / mode
# --------------------------------------------------------------------------

def test_track_rates_sidereal_from_mode_string() -> None:
    """A "Sidereal" mode string classifies as SIDEREAL with zero rates."""
    h = Header()
    h["TELTKRA"] = 0.0
    h["TELTKDEC"] = 0.0
    h["TRKMODE"] = "Sidereal"
    ra_rate, dec_rate, mode = fits_io.extract_track_rates_from_header(h)
    assert ra_rate == pytest.approx(0.0)
    assert dec_rate == pytest.approx(0.0)
    assert mode is TrackMode.SIDEREAL


def test_track_rates_rate_mode_from_string() -> None:
    """A "rate" mode string classifies as RATE with its nonzero rates."""
    h = Header()
    h["TELTKRA"] = 15.0
    h["TELTKDEC"] = -3.0
    h["TRKMODE"] = "rate"
    ra_rate, dec_rate, mode = fits_io.extract_track_rates_from_header(h)
    assert ra_rate == pytest.approx(15.0)
    assert dec_rate == pytest.approx(-3.0)
    assert mode is TrackMode.RATE


def test_track_mode_inferred_from_zero_rates_when_no_mode() -> None:
    """With no mode key, zero rates infer SIDEREAL."""
    h = Header()
    h["TELTKRA"] = 0.0
    h["TELTKDEC"] = 0.0
    _, _, mode = fits_io.extract_track_rates_from_header(h)
    assert mode is TrackMode.SIDEREAL


def test_track_mode_inferred_rate_from_nonzero_rates_when_no_mode() -> None:
    """With no mode key, a nonzero rate infers RATE."""
    h = Header()
    h["TELTKRA"] = 5.0
    h["TELTKDEC"] = 0.0
    _, _, mode = fits_io.extract_track_rates_from_header(h)
    assert mode is TrackMode.RATE


def test_track_mode_unknown_when_nothing_present() -> None:
    """An empty header yields an UNKNOWN track mode."""
    _, _, mode = fits_io.extract_track_rates_from_header(Header())
    assert mode is TrackMode.UNKNOWN


def test_track_rates_unit_conversion_degrees(restore_headers: HeadersConfig) -> None:
    """Degrees/second track rates convert to arcsec/second."""
    cfg = restore_headers
    cfg.tracking.track_ra_rate_unit = "degrees/second"
    cfg.tracking.track_dec_rate_unit = "degrees/second"
    h = Header()
    h["TELTKRA"] = 0.0112  # ~40.32 arcsec/s
    h["TELTKDEC"] = -0.0042  # ~-15.12 arcsec/s
    ra_rate, dec_rate, _ = fits_io.extract_track_rates_from_header(h)
    assert ra_rate == pytest.approx(40.32, rel=1e-6)
    assert dec_rate == pytest.approx(-15.12, rel=1e-6)


# --------------------------------------------------------------------------
# Filter normalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["open", "L", "lum", "clear", "none", ""])
def test_filter_clear_aliases_normalized(raw: str) -> None:
    """Various clear-filter aliases normalize to "Clear"."""
    h = Header()
    h["FILTER"] = raw
    assert fits_io.extract_filter_from_header(h) == "Clear"


def test_filter_named_passthrough() -> None:
    """A named filter passes through unchanged."""
    h = Header()
    h["FILTER"] = "Sloan_r"
    assert fits_io.extract_filter_from_header(h) == "Sloan_r"


def test_filter_missing_returns_none() -> None:
    """A missing FILTER header returns None."""
    assert fits_io.extract_filter_from_header(Header()) is None
