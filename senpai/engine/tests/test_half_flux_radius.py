"""Half-flux radius (measure_half_flux_radius) on synthetic in-focus and defocused stars.

The property that matters for focus sweeps: the half-flux radius keeps growing as a
star defocuses into a donut, where half-max measures around the centroid read noise.
"""

from __future__ import annotations

import itertools

import numpy as np

from senpai.engine.detection.point.fwhm import measure_half_flux_radius

SKY = 150.0
NOISE = 5.0


def _field(star, n_side: int = 4, size: int = 400, seed: int = 0) -> np.ndarray:
    """Sky plus an n_side x n_side grid of identical stars drawn by `star(dx, dy)`."""
    rng = np.random.default_rng(seed)
    image = SKY + rng.normal(0.0, NOISE, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    step = size // n_side
    for i in range(n_side):
        for j in range(n_side):
            image += star(xx - (i + 0.5) * step, yy - (j + 0.5) * step)
    return image.astype(np.float32)


def _gaussian(sigma: float, amp: float = 2000.0):
    return lambda dx, dy: amp * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))


def _donut(r_ring: float, width: float, amp: float = 400.0):
    return lambda dx, dy: amp * np.exp(-((np.hypot(dx, dy) - r_ring) ** 2) / (2 * width**2))


def test_gaussian_hfr_matches_analytic():
    # A Gaussian's half-flux radius is sigma * sqrt(2 ln 2) = FWHM / 2.
    sigma = 2.0
    hfr, n = measure_half_flux_radius(_field(_gaussian(sigma)), SKY, NOISE)
    assert n == 16
    assert abs(hfr - sigma * np.sqrt(2 * np.log(2))) < 0.3


def test_hfr_grows_monotonically_through_donut_defocus():
    stars = [_gaussian(1.5), _gaussian(3.0), _donut(6.0, 1.5), _donut(10.0, 1.5), _donut(14.0, 2.0)]
    radii = [measure_half_flux_radius(_field(s), SKY, NOISE)[0] for s in stars]
    assert all(r is not None for r in radii)
    assert all(b > a for a, b in itertools.pairwise(radii)), radii


def test_saturated_sources_are_excluded():
    # Clip every star's core to one value, as a saturated sensor does; the clip
    # level is found from the frame alone.
    image = _field(_gaussian(2.0, amp=5000.0))
    np.minimum(image, 3000.0, out=image)
    assert measure_half_flux_radius(image, SKY, NOISE) == (None, 0)


def test_unclipped_bright_stars_are_kept():
    # Identical bright stars share a near-equal peak but no pixel repeats the max.
    hfr, n = measure_half_flux_radius(_field(_gaussian(2.0, amp=5000.0)), SKY, NOISE)
    assert n == 16 and hfr is not None


def test_empty_frame_returns_none():
    rng = np.random.default_rng(1)
    blank = (SKY + rng.normal(0.0, NOISE, (200, 200))).astype(np.float32)
    assert measure_half_flux_radius(blank, SKY, NOISE) == (None, 0)
