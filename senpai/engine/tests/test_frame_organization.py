"""Behavioral tests for senpai.engine.utils.frame_organization.

Covers the date/time header extraction (DATE-OBS combined timestamps, plus
separate DATE_OBS + TIME-OBS components, and MM/DD/YY fallbacks) and the
filesystem-organization helpers that group FITS frames by filename substring
or by a shared header-ID keyword (including the special ORCHCOMM encoding).

All FITS output goes to pytest's tmp_path; no network / Astrometry.net.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from astropy.io import fits

from senpai.engine.utils import frame_organization as fo


def _write_fits(path: Path, header_items: dict | None = None) -> Path:
    data = np.zeros((4, 4), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    for key, value in (header_items or {}).items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)
    return path


# --------------------------------------------------------------------------
# extract_uct_time_from_header
# --------------------------------------------------------------------------

def test_uct_time_from_combined_date_obs():
    header = {"DATE-OBS": "2023-05-01T03:22:11.5"}
    t = fo.extract_uct_time_from_header(header)
    assert isinstance(t, datetime)
    assert (t.year, t.month, t.day, t.hour, t.minute, t.second) == (2023, 5, 1, 3, 22, 11)


def test_uct_time_from_separate_date_and_time():
    # DATE_OBS (underscore) is a DATE header; TIME-OBS supplies the clock.
    header = {"DATE_OBS": "2022-11-09", "TIME-OBS": "21:36:28.604000"}
    t = fo.extract_uct_time_from_header(header)
    assert (t.year, t.month, t.day) == (2022, 11, 9)
    assert (t.hour, t.minute, t.second) == (21, 36, 28)
    assert t.microsecond == 604000


def test_uct_time_date_only_without_time_raises():
    # A DATE header with no usable DATE-TIME and no TIME header cannot combine.
    header = {"DATE_OBS": "2022-11-09"}
    with pytest.raises(AttributeError):
        fo.extract_uct_time_from_header(header)


def test_uct_time_no_date_headers_raises():
    with pytest.raises(AttributeError):
        fo.extract_uct_time_from_header({"FOO": "bar"})


def test_uct_time_prefers_date_time_over_separate():
    # DATE-OBS combined timestamp should win even if separate fields exist.
    header = {
        "DATE-OBS": "2019-03-04T10:11:12",
        "DATE_OBS": "2000-01-01",
        "TIME-OBS": "00:00:00",
    }
    t = fo.extract_uct_time_from_header(header)
    assert (t.year, t.month, t.day, t.hour, t.minute, t.second) == (2019, 3, 4, 10, 11, 12)


# --------------------------------------------------------------------------
# internal date/time string parsers
# --------------------------------------------------------------------------

def test_parse_date_string_iso():
    a = fo._parse_date_string("2021-07-04")
    assert (a.year, a.month, a.day) == (2021, 7, 4)


def test_parse_date_string_us_slash_format():
    a = fo._parse_date_string("3/14/21")
    assert (a.month, a.day, a.year) == (3, 14, 2021)


def test_parse_time_string_with_microseconds():
    assert fo._parse_time_string("21:36:28.604000") == (21, 36, 28, 604000)


def test_parse_time_string_without_microseconds():
    assert fo._parse_time_string("08:05:09") == (8, 5, 9, 0)


def test_parse_time_string_invalid_returns_none():
    assert fo._parse_time_string("not-a-time") is None


# --------------------------------------------------------------------------
# imageset selection by filename
# --------------------------------------------------------------------------

def test_get_imageset_by_filename_matches_substring(tmp_path):
    _write_fits(tmp_path / "targetA_001.fits")
    _write_fits(tmp_path / "targetA_002.fits")
    _write_fits(tmp_path / "targetB_001.fits")
    result = fo.get_imageset_by_filename(tmp_path, "targetA")
    assert [Path(p).name for p in result] == ["targetA_001.fits", "targetA_002.fits"]


def test_get_imageset_by_filename_recurses_subdirs(tmp_path):
    sub = tmp_path / "night1"
    sub.mkdir()
    _write_fits(sub / "set_003.fits")
    _write_fits(tmp_path / "set_001.fits")
    result = fo.get_imageset_by_filename(tmp_path, "set_")
    names = sorted(Path(p).name for p in result)
    assert names == ["set_001.fits", "set_003.fits"]


def test_get_imageset_by_filename_no_match(tmp_path):
    _write_fits(tmp_path / "alpha.fits")
    assert fo.get_imageset_by_filename(tmp_path, "zzz") == []


def test_get_all_images_in_directory_sorted(tmp_path):
    _write_fits(tmp_path / "b.fits")
    _write_fits(tmp_path / "a.fits")
    result = fo.get_all_images_in_directory(tmp_path)
    assert [Path(p).name for p in result] == ["a.fits", "b.fits"]


def test_get_all_images_in_directory_empty(tmp_path):
    assert fo.get_all_images_in_directory(tmp_path) == []


# --------------------------------------------------------------------------
# header ID extraction / matching
# --------------------------------------------------------------------------

def test_extract_id_from_header_plain_key(tmp_path):
    f = _write_fits(tmp_path / "img.fits", {"IMAGESET": "abc123"})
    assert fo.extract_id_from_header(f, "IMAGESET") == "abc123"


def test_extract_id_from_header_missing_key_returns_none(tmp_path):
    f = _write_fits(tmp_path / "img.fits", {"OTHER": "x"})
    assert fo.extract_id_from_header(f, "IMAGESET") is None


def test_extract_id_from_header_orchcomm_parsing(tmp_path):
    f = _write_fits(tmp_path / "img.fits", {"ORCHCOMM": "&MYSET01@[ukr]#[1:6]%[OPEN]"})
    assert fo.extract_id_from_header(f, "ORCHCOMM") == "MYSET01"


def test_header_key_matches_true_and_false(tmp_path):
    f = _write_fits(tmp_path / "img.fits", {"IMAGESET": "abc"})
    assert fo.header_key_matches(f, "IMAGESET", "abc") is True
    assert fo.header_key_matches(f, "IMAGESET", "xyz") is False


def test_header_key_matches_missing_key_false(tmp_path):
    f = _write_fits(tmp_path / "img.fits", {"OTHER": "v"})
    assert fo.header_key_matches(f, "IMAGESET", "abc") is False


def test_header_key_matches_orchcomm_substring(tmp_path):
    f = _write_fits(tmp_path / "img.fits", {"ORCHCOMM": "&SET007@[ukr]#[1:6]%[OPEN]"})
    assert fo.header_key_matches(f, "ORCHCOMM", "SET007") is True
    assert fo.header_key_matches(f, "ORCHCOMM", "SET999") is False


def test_get_imageset_by_id_groups_matching_headers(tmp_path):
    _write_fits(tmp_path / "f1.fits", {"IMAGESET": "night-A"})
    _write_fits(tmp_path / "f2.fits", {"IMAGESET": "night-A"})
    _write_fits(tmp_path / "f3.fits", {"IMAGESET": "night-B"})
    result = fo.get_imageset_by_id(tmp_path, "night-A", "IMAGESET")
    assert [Path(p).name for p in result] == ["f1.fits", "f2.fits"]


def test_get_imageset_by_id_no_match_empty(tmp_path):
    _write_fits(tmp_path / "f1.fits", {"IMAGESET": "night-A"})
    assert fo.get_imageset_by_id(tmp_path, "night-Z", "IMAGESET") == []
