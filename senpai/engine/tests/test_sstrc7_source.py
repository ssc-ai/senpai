"""SSTRC7 catalog reads, against a real local catalog.

Marked `requires_catalog`: these need the 17.6 GB catalog on disk and are
skipped in CI. Point `SSTRC7_PATH` at it (or let the package resolve
`~/.sstrc7`) to run them locally.
"""

import os
import types

import numpy as np
import pytest

from senpai.catalog import sstrc7_source
from senpai.core import config as config_module

RA, DEC, FOV = 245.45, 41.8, 0.5

# A proper motion this large does not exist: Barnard's Star, the fastest known,
# moves 10.4 arcsec/yr. The vendored reader this module replaced was 9.77x too
# large, which put ordinary stars past this bound.
MAX_PLAUSIBLE_ARCSEC_PER_YEAR = 11.0
RAD_PER_SEC_TO_ARCSEC_PER_YEAR = (180 / np.pi) * 3600 * 3.1556952e7


def _catalog_path():
    path = sstrc7_source.resolve_catalog_path(os.getenv("SSTRC7_PATH"))
    if not (path / "sstrc.acc").is_file():
        pytest.skip(f"no SSTRC7 catalog at {path}")
    return str(path)


@pytest.fixture(scope="module")
def stars():
    return sstrc7_source.query_by_los_radec_with_rotation(FOV, FOV, RA, DEC, rotation=0.0, rootPath=_catalog_path())


@pytest.mark.requires_catalog
def test_query_returns_senpai_star_records(stars):
    # The record shape is a contract shared with gaia.py and sdss.py.
    assert len(stars) > 0
    for star in stars[:50]:
        assert set(star) >= {"ra", "dec", "ra_pm", "dec_pm", "mv", "magnitudes", "catalog"}
        assert isinstance(star["magnitudes"], dict) and star["magnitudes"]
        assert star["mv"] is not None
        assert isinstance(star["catalog"], str) and star["catalog"]


@pytest.mark.requires_catalog
def test_positions_are_radians_around_the_requested_field(stars):
    ra = np.degrees([s["ra"] for s in stars])
    dec = np.degrees([s["dec"] for s in stars])
    # A cone enclosing the field corners, so half the diagonal plus the margin.
    radius = 0.5 * np.hypot(FOV * 1.1, FOV * 1.1)
    sep = np.hypot((ra - RA) * np.cos(np.radians(dec)), dec - DEC)
    assert sep.max() <= radius + 1e-6
    assert sep.min() < FOV / 2


@pytest.mark.requires_catalog
def test_proper_motion_scale_is_physical(stars):
    """Guards the 1/0.32**2 scale error the vendored reader carried."""
    pm = np.hypot([s["ra_pm"] for s in stars], [s["dec_pm"] for s in stars])
    assert (pm * RAD_PER_SEC_TO_ARCSEC_PER_YEAR).max() < MAX_PLAUSIBLE_ARCSEC_PER_YEAR
    assert pm.max() > 0  # a whole field with no measured motion means a decode bug


@pytest.mark.requires_catalog
def test_magnitudes_are_catalog_valued(stars):
    for star in stars[:200]:
        for band, value in star["magnitudes"].items():
            assert band in sstrc7_source.BAND_NAMES or band == "Invalid"
            assert -32 < value <= 32
            # Stored as integer millimags, so no float32 residue.
            assert value == round(value, sstrc7_source.MAG_DECIMALS)
        if star["mv"] < 32:
            assert star["mv"] in star["magnitudes"].values()


@pytest.mark.requires_catalog
def test_magnitude_limits_filter_the_result():
    path = _catalog_path()
    faint = sstrc7_source.query_by_los_radec_with_rotation(FOV, FOV, RA, DEC, rootPath=path, faint_lim=14.0)
    assert faint, "no stars brighter than 14th magnitude in this field"
    assert max(s["mv"] for s in faint) < 14.0

    bright = sstrc7_source.query_by_los_radec_with_rotation(FOV, FOV, RA, DEC, rootPath=path, bright_lim=14.0)
    assert bright
    assert min(s["mv"] for s in bright) > 14.0


@pytest.mark.requires_catalog
def test_filter_center_interpolates_a_different_magnitude(stars):
    at_2mass_j = sstrc7_source.query_by_los_radec_with_rotation(
        FOV, FOV, RA, DEC, rootPath=_catalog_path(), filter_center=1235.0
    )
    assert len(at_2mass_j) == len(stars)
    interpolated = np.array([s["mv"] for s in at_2mass_j])
    broadband = np.array([s["mv"] for s in stars])
    assert not np.allclose(interpolated, broadband)
    # Interpolating at 2MASS J's own center should reproduce that band.
    j_band = np.array([s["magnitudes"].get("2MASS_J", np.nan) for s in stars])
    measured = np.isfinite(j_band)
    assert np.allclose(interpolated[measured], j_band[measured], atol=1e-3)


@pytest.mark.requires_catalog
def test_examine_catalog_accepts_a_complete_catalog():
    assert sstrc7_source.examine_catalog(_catalog_path()) is True


# --------------------------------------------------------------------------- #
# Band-priority ladder (pure logic -- no catalog on disk).
# --------------------------------------------------------------------------- #
class _FakeField:
    """The two StarField attributes `_priority_ladder` reads."""

    def __init__(self, mag: np.ndarray, visual: np.ndarray) -> None:
        self.mag = mag
        self.visual = visual


def _fake_field(**bands: float) -> _FakeField:
    """One star, with the named bands set and every other band a sentinel."""
    mag = np.full((1, len(sstrc7_source.BAND_NAMES)), sstrc7_source.INVALID_MAG)
    for name, value in bands.items():
        mag[0, sstrc7_source.BAND_NAMES.index(name)] = value
    return _FakeField(mag, np.array([bands.get("Johnson_V", np.nan)]))


def _set_priority(monkeypatch, priority: str) -> None:

    fake = types.SimpleNamespace(star_catalog=types.SimpleNamespace(magnitude_band_priority=priority))
    monkeypatch.setattr(config_module, "_config_instance", fake)


def test_priority_ladder_default_is_the_package_visual(monkeypatch) -> None:
    _set_priority(monkeypatch, "johnson_v_first")
    field = _fake_field(Gaia_G=9.0, Johnson_V=11.0)
    assert sstrc7_source._priority_ladder(field) == pytest.approx([11.0])


def test_priority_ladder_open_first_leads_with_gaia_g(monkeypatch) -> None:
    _set_priority(monkeypatch, "open_first")
    field = _fake_field(Gaia_G=9.0, Johnson_R=10.0, Johnson_V=11.0)
    assert sstrc7_source._priority_ladder(field) == pytest.approx([9.0])


def test_priority_ladder_open_first_walks_down_the_order(monkeypatch) -> None:
    _set_priority(monkeypatch, "open_first")
    # Gaia_G unmeasured (sentinel), so Johnson_R is next in the order.
    field = _fake_field(Johnson_R=10.0, Johnson_V=11.0, Sloan_r=10.5)
    assert sstrc7_source._priority_ladder(field) == pytest.approx([10.0])


def test_priority_ladder_open_first_is_nan_with_no_usable_band(monkeypatch) -> None:
    _set_priority(monkeypatch, "open_first")
    field = _fake_field(**{"2MASS_J": 8.0})
    assert np.isnan(sstrc7_source._priority_ladder(field)).all()


def test_priority_ladder_requires_an_initialized_config(monkeypatch) -> None:
    """Reading the band priority before initialize_config raises rather than defaulting.

    A silent fallback would score a frame against a different band ladder than the one
    configured, which is indistinguishable from correct output.
    """

    monkeypatch.setattr(config_module, "_config_instance", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        sstrc7_source._priority_ladder(_fake_field(Gaia_G=9.0, Johnson_V=11.0))
