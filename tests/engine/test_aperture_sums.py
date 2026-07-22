"""Tests for the shared-shape aperture photometry helper.

``_shared_shape_aperture_sums`` replaced per-star photutils mask generation
with one cached mask per 1/8-px fractional offset. These pin its contract:
flux interchangeable with direct ``aperture_photometry`` (subpixel method)
even in crowded fields where neighbors cross aperture boundaries.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from photutils.aperture import (
    Aperture,
    CircularAnnulus,
    CircularAperture,
    RectangularAnnulus,
    RectangularAperture,
    aperture_photometry,
)

from senpai.engine.photometry.utils import _shared_shape_aperture_sums


def _streak_field(n: int = 400, size: int = 1500, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Build a noisy frame seeded with ``n`` rotated Gaussian streaks.

    Args:
        n: Number of streaks to inject.
        size: Side length of the square frame in pixels.
        seed: Seed for the random-number generator.

    Returns:
        The image data array and the ``(n, 2)`` array of streak centers.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 5.0, (size, size)).astype(np.float32)
    pos = np.column_stack(
        [rng.uniform(120, size - 120, n), rng.uniform(120, size - 120, n)]
    )
    sig_l, sig_w, ang = 16.0, 3.8, np.deg2rad(35)
    yy, xx = np.mgrid[-60:61, -60:61]
    for x, y in pos:
        ix, iy = round(x), round(y)
        fxs, fys = xx + ix - x, yy + iy - y
        uu = fxs * np.cos(ang) + fys * np.sin(ang)
        vv = -fxs * np.sin(ang) + fys * np.cos(ang)
        g = 5000.0 * np.exp(-(uu**2 / (2 * sig_l**2) + vv**2 / (2 * sig_w**2)))
        data[iy - 60:iy + 61, ix - 60:ix + 61] += g.astype(np.float32)
    return data, pos


def _assert_matches_photutils(
    data: np.ndarray,
    pos: np.ndarray,
    build: Callable[[np.ndarray], list[Aperture]],
) -> None:
    """Assert the shared-shape helper matches direct ``aperture_photometry``.

    Args:
        data: Image data to measure.
        pos: ``(n, 2)`` array of aperture centers.
        build: Factory returning ``[aperture, background_annulus]`` for positions.
    """
    ref = aperture_photometry(data, build(pos), method="subpixel", subpixels=5)
    ref0 = np.asarray(ref["aperture_sum_0"], dtype=float)
    got0, got1 = _shared_shape_aperture_sums(data, pos, build)

    rel = np.abs(got0 - ref0) / np.maximum(np.abs(ref0), 1.0)
    assert np.median(rel) < 1e-3
    assert np.percentile(rel, 99) < 5e-3
    # Background annulus sums are noise-scale; check they stay within a few
    # boundary-pixel flux units of the reference.
    ref1 = np.asarray(ref["aperture_sum_1"], dtype=float)
    assert np.median(np.abs(got1 - ref1)) < 0.05 * np.median(np.abs(ref1) + 1.0)


def test_rectangular_apertures_match_photutils() -> None:
    """Rotated rectangular aperture + annulus sums match photutils."""
    data, pos = _streak_field()
    w, h, theta = 36.0, 56.0, np.deg2rad(125.4)
    _assert_matches_photutils(
        data,
        pos,
        lambda p: [
            RectangularAperture(p, w=w, h=h, theta=theta),
            RectangularAnnulus(
                p, w_in=w + 18, w_out=w + 36, h_in=h + 18, h_out=h + 36, theta=theta
            ),
        ],
    )


def test_circular_apertures_match_photutils() -> None:
    """Circular aperture + annulus sums match photutils."""
    data, pos = _streak_field()
    _assert_matches_photutils(
        data,
        pos,
        lambda p: [
            CircularAperture(p, r=12.0),
            CircularAnnulus(p, r_in=18.0, r_out=30.0),
        ],
    )


def test_partially_off_frame_aperture_is_clipped_not_crashed() -> None:
    """Apertures whose bbox leaves the frame are clipped, yielding finite sums."""
    data, _ = _streak_field(n=10, size=600)
    # Centers close enough to the border that the annulus bbox leaves the
    # frame; callers margin-filter, but the helper must clip like photutils.
    pos = np.array([[15.0, 300.0], [300.0, 590.0], [595.0, 8.0]])
    def build(p: np.ndarray) -> list[Aperture]:
        """Return a circular aperture and background annulus for ``p``."""
        return [
            CircularAperture(p, r=12.0),
            CircularAnnulus(p, r_in=18.0, r_out=30.0),
        ]
    got0, got1 = _shared_shape_aperture_sums(data, pos, build)
    assert np.all(np.isfinite(got0)) and np.all(np.isfinite(got1))
