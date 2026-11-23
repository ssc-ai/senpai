"""Rate-frame WCS refinement: the per-star search window.

When the propagated WCS is off by more than the 10 px search radius along the star trails, every
window misses its star's peak and every argmax lands on the window border. All stars then report
the same clipped shift, the MAD cut on shift magnitude rejects none of them, and the refit keeps an
error equal to the start error minus the radius -- reported as KERNEL_REFINED_WCS. Measured on a
simulated GEO rate-track collect (1.25 arcsec/px, 3 s exposures): a 5.4 px WCS error on the first
rate frame reached from the sidereal anchor.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import convolve

from senpai.engine.detection.kernels import rectangle_pyramoid
from senpai.engine.models.astrometry import WCSStatus
from senpai.engine.utils import wcs_refinement
from senpai.engine.utils.wcs_refinement import converge_search_windows, measure_window_peak


def _kernel(length: float, rotation_deg: float, fwhm: float) -> np.ndarray:
    rad = np.deg2rad(rotation_deg)
    rectangle_pyramoid.cache_clear()
    return rectangle_pyramoid(length, float(np.sin(rad)), float(np.cos(rad)), fwhm)



TRAIL_LENGTH = 35.0
TRAIL_FWHM = 4.0


def _trail_field(rng: np.random.Generator, n_stars: int = 40, size: int = 700):
    """A rate frame of vertical star trails, already convolved with the frame's streak kernel."""
    image = rng.normal(0.0, 1.0, (size, size))
    rows, cols = np.indices(image.shape)
    sigma = TRAIL_FWHM / 2.355
    truth: list[tuple[float, float]] = []
    while len(truth) < n_stars:
        x, y = rng.uniform(80, size - 80, 2)
        if any((x - a) ** 2 + (y - b) ** 2 < 60**2 for a, b in truth):
            continue
        truth.append((float(x), float(y)))
        across = np.exp(-0.5 * ((cols - x) / sigma) ** 2)
        along = np.clip(TRAIL_LENGTH / 2 - np.abs(rows - y), 0.0, 1.0)
        image += rng.uniform(20, 60) * across * along
    kernel = _kernel(TRAIL_LENGTH, 90.0, TRAIL_FWHM)
    return convolve(image, kernel, mode="same"), np.array(truth)


def test_measure_window_peak_flags_a_peak_outside_the_window() -> None:
    image = np.zeros((101, 101))
    image[50, 72] = 1.0  # 22 px right of the window centre: outside a radius-10 window
    image[50, 49] = 0.5  # a weaker peak inside the window
    x, y, on_border = measure_window_peak(image, 50.0, 50.0, radius=10)
    assert (x, y, on_border) == (49, 50, False)

    image[50, 49] = 0.0
    ramp = np.tile(np.arange(101, dtype=float), (101, 1))  # rises toward +x, peak beyond the window
    x, y, on_border = measure_window_peak(ramp, 50.0, 50.0, radius=10)
    assert x == 60
    assert on_border


def test_search_windows_stay_put_when_the_wcs_is_already_close() -> None:
    convolved, truth = _trail_field(np.random.default_rng(3))
    predicted = truth + np.array([0.4, -0.6])
    result = converge_search_windows(convolved, predicted, radius=10)
    assert result.converged
    assert result.offset == (0.0, 0.0)
    assert len(result.edge_fractions) == 1
    assert result.edge_fractions[0] < 0.5


@pytest.mark.parametrize("error", [(5.0, 15.0), (-6.4, -14.7), (2.0, -30.0)])
def test_search_windows_walk_to_the_stars_when_the_wcs_is_off_along_the_trails(
    error: tuple[float, float],
) -> None:
    convolved, truth = _trail_field(np.random.default_rng(11))
    predicted = truth + np.array(error)
    result = converge_search_windows(convolved, predicted, radius=10)
    assert result.converged
    assert result.edge_fractions[0] > 0.5  # the old single window missed nearly every star
    # Measured from the walked windows, the stars land on their true positions.
    measured = np.array(
        [measure_window_peak(convolved, x + result.offset[0], y + result.offset[1], 10)[:2]
         for x, y in predicted]
    )
    assert np.median(measured[:, 0] - truth[:, 0]) == pytest.approx(0.0, abs=0.5)
    assert np.median(measured[:, 1] - truth[:, 1]) == pytest.approx(0.0, abs=0.5)


def test_search_windows_report_failure_when_every_window_stays_on_its_border() -> None:
    ramp = np.tile(np.arange(700, dtype=float), (700, 1))  # no peak anywhere to converge on
    predicted = np.array([[100.0 + 40 * i, 350.0] for i in range(12)])
    result = converge_search_windows(ramp, predicted, radius=10, max_steps=4)
    assert not result.converged
    assert len(result.edge_fractions) == 5
    assert all(fraction > 0.5 for fraction in result.edge_fractions)


def test_unconverged_search_windows_demote_the_frame_without_a_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refit must not run on window-border measurements, and the frame must say so."""
    streak = SimpleNamespace(
        pixel_length=TRAIL_LENGTH, sine_angle=1.0, cosine_angle=0.0, fwhm=TRAIL_FWHM,
        to_pyramoid=lambda: _kernel(TRAIL_LENGTH, 90.0, TRAIL_FWHM),
    )
    starfield = SimpleNamespace(
        wcs_status=WCSStatus.PIXEL_SHIFTED_WCS, wcs=object(), catalog_stars=[],
        wcs_quality=None,
    )
    frame = SimpleNamespace(
        index=5, streak=streak, starfield=starfield, frame=SimpleNamespace(data=np.zeros((50, 50)))
    )
    config = SimpleNamespace(
        streak=SimpleNamespace(variable_kernel=SimpleNamespace(enable=False)),
        plotting=SimpleNamespace(debug=False),
    )
    refit_calls: list[object] = []
    monkeypatch.setattr(wcs_refinement, "get_config", lambda: config)
    monkeypatch.setattr(wcs_refinement, "get_global_shift_from_astrometric_stars", lambda f, c: (0.0, 0.0))
    monkeypatch.setattr(wcs_refinement, "shift_wcs", lambda wcs, dx, dy: wcs)
    monkeypatch.setattr(wcs_refinement, "update_starfield_wcs", lambda *a, **k: None)
    monkeypatch.setattr(wcs_refinement, "window_probe_positions", lambda f: np.array([[10.0, 10.0]]))
    monkeypatch.setattr(
        wcs_refinement,
        "converge_search_windows",
        lambda *a, **k: wcs_refinement.WindowSearch(
            offset=(0.0, 40.0), edge_fractions=[0.96, 0.95, 0.97, 0.96, 0.95], converged=False
        ),
    )
    monkeypatch.setattr(
        wcs_refinement,
        "refine_wcs_with_catalog_stars",
        lambda f, c: refit_calls.append(f) or (None, None),
    )
    monkeypatch.setattr(wcs_refinement, "validate_frame_wcs", lambda f, refit_stats=None: None)

    wcs_refinement.refine_wcs_by_kernel_convolution(frame)

    assert refit_calls == []
    assert frame.starfield.wcs_status == WCSStatus.REFINED_UNVALIDATED_WCS
