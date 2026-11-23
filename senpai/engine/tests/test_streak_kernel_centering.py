"""The streak kernel must be centred on a pixel.

An even-sized kernel puts its centre between two pixels, so a ``mode="same"`` convolution moves
every response by half a pixel on that axis. Rate-frame WCS refinement takes each star's position
from such a convolution and refits the WCS to them, so the half pixel went into the WCS: measured on
a simulated GEO rate-track run (1.25 arcsec/px, 3 s exposures), 84% of rate frames
carried a +0.5 px offset on at least one axis, and the measured trail length decided which.
"""

import numpy as np
import pytest
from scipy.signal import convolve

from senpai.engine.detection.kernels import rectangle_pyramoid

# (length, rotation in degrees, fwhm) as senpai logs them for a vertical GEO trail.
LOGGED_STREAKS = [
    (34.0, 90.0, 4.0),
    (35.0, 90.0, 4.0),
    (34.0, 89.8, 4.0),
    (35.0, 90.0, 5.0),
    (34.0, 90.0, 3.0),
]


def _kernel(length: float, rotation_deg: float, fwhm: float) -> np.ndarray:
    rad = np.deg2rad(rotation_deg)
    rectangle_pyramoid.cache_clear()
    return rectangle_pyramoid(length, float(np.sin(rad)), float(np.cos(rad)), fwhm)


@pytest.mark.parametrize(("length", "rotation", "fwhm"), LOGGED_STREAKS)
def test_streak_kernel_has_odd_dimensions(length: float, rotation: float, fwhm: float) -> None:
    rows, cols = _kernel(length, rotation, fwhm).shape
    assert rows % 2 == 1
    assert cols % 2 == 1


@pytest.mark.parametrize(("length", "rotation", "fwhm"), LOGGED_STREAKS)
def test_streak_kernel_response_is_centred_on_the_source(
    length: float, rotation: float, fwhm: float
) -> None:
    image = np.zeros((121, 121))
    image[60, 60] = 1.0
    response = convolve(image, _kernel(length, rotation, fwhm), mode="same")
    rows, cols = np.indices(response.shape)
    total = response.sum()
    assert (response * cols).sum() / total == pytest.approx(60.0, abs=1e-3)
    assert (response * rows).sum() / total == pytest.approx(60.0, abs=1e-3)


