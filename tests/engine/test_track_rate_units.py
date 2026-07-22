"""Tests for senpai.engine.utils.fits_io._to_arcsec_per_second.

The track-rate unit conversion was a dead config field until burr-side work
exposed it: burr writes RA_RATE/DEC_RATE in degrees/second, while senpai's
classifier and downstream code assume arcsec/s. The helper normalizes so the
1 arcsec/s threshold in ``organize_senpai_frames`` works regardless of the
upstream unit convention.
"""


import pytest
from _pytest.python_api import ApproxBase

from senpai.engine.utils.fits_io import _to_arcsec_per_second


@pytest.mark.parametrize("unit,value,expected", [
    ("arcseconds/second", 5.0, 5.0),
    ("arcsec/second", 5.0, 5.0),
    ("arcsec/s", 5.0, 5.0),
    ("degrees/second", 0.0112, 40.32),   # burr's calsat rate
    ("deg/second", 1.0, 3600.0),
    ("deg/s", 0.001, 3.6),
    ("radians/second", 1e-5, pytest.approx(2.063, abs=1e-3)),
])
def test_known_units(unit: str, value: float, expected: float | ApproxBase) -> None:
    """Each supported unit string converts its value to arcsec/second."""
    assert _to_arcsec_per_second(value, unit) == pytest.approx(expected, rel=1e-6)


def test_case_insensitive() -> None:
    """Unit matching ignores letter case and surrounding whitespace."""
    assert _to_arcsec_per_second(1.0, "Degrees/Second") == pytest.approx(3600.0)
    assert _to_arcsec_per_second(1.0, "  arcsec/s  ") == pytest.approx(1.0)


def test_unknown_unit_passes_through_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """An unrecognized unit is passed through unchanged and logs a warning."""
    import logging
    with caplog.at_level(logging.WARNING):
        v = _to_arcsec_per_second(42.0, "furlongs/fortnight")
    assert v == 42.0
    assert "Unknown track-rate unit" in caplog.text


def test_negative_rates_preserve_sign() -> None:
    """A negative rate keeps its sign through the unit conversion."""
    # f3 of a burr calsat collection has RA_RATE=-0.0042 deg/s; we want the
    # converted value to keep its sign so downstream code can distinguish
    # tracking direction.
    assert _to_arcsec_per_second(-0.0042, "degrees/second") == pytest.approx(-15.12)
