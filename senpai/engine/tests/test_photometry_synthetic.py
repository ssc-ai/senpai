"""Synthetic-image tests for the live photometry engine
(senpai.engine.photometry.utils).

We inject 2D Gaussian PSF stars of *known* total flux at *known* positions on a
flat + Gaussian-noise background (seeded) and check that:

  - circular-aperture photometry recovers the injected flux to a few percent,
  - SNR lands in a sensible ballpark and tracks brightness,
  - the quality flag honours min_snr / max_crowding,
  - the limiting-magnitude / completeness helpers and magnitude-selection /
    crowding helpers behave as documented.

The photometry functions read the process-wide config singleton via
get_config(); we initialise it from a real YAML in an autouse fixture (same
pattern as test_streak_extraction.py) and force plotting off so nothing touches
the filesystem.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import FWHMMetadata, ImageMetadata
from senpai.engine.models.starfield import StarField, StarInSpace
from senpai.engine.photometry.utils import (
    SimplePhotometryConfig,
    SimplePhotometryResult,
    _assess_simple_quality,
    _calculate_simple_crowding,
    _completeness_limits,
    _find_common_magnitude_system,
    _get_best_magnitude,
    _has_bright_neighbor,
    _isotonic_completeness,
    _precompute_star_magnitudes,
    compute_completeness_curve,
    measure_simple_star_photometry,
    measure_simple_starfield_photometry,
)

TRUE_ZP = 25.0
FWHM = 3.5
SIGMA = FWHM / 2.355


@pytest.fixture(scope="module", autouse=True)
def _config():
    """Process-wide config singleton, plotting off (no files written)."""
    initialize_config(CONFIG_DIR / "burr.yaml")
    get_config().plotting.photometry = False
    get_config().plotting.debug = False


def _cfg(**kw) -> SimplePhotometryConfig:
    """Photometry config with a simple, deterministic noise model.

    gain=1, read noise off -> flux_err is pure shot+sky Poisson in ADU, which
    keeps the expected SNR analytic for the assertions below.
    """
    base = {"gain": 1.0, "include_read_noise": False}
    base.update(kw)
    return SimplePhotometryConfig(**base)


def _inject_star(data, x0, y0, total_flux, sigma=SIGMA):
    """Add a normalised 2D Gaussian of given total flux in-place."""
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    psf = np.exp(-0.5 * (((xx - x0) ** 2 + (yy - y0) ** 2) / sigma**2))
    psf *= total_flux / psf.sum()
    data += psf


def _image(data, exposure_time=1.0):
    h, w = data.shape
    return ProcessedFitsImage(
        data=data.astype(np.float64),
        header=fits.Header(),
        data_type=np.dtype("float64"),
        metadata=ImageMetadata(image_id="t", width=w, height=h, exposure_time=exposure_time),
    )


def _flux_for_mag(mag):
    return 10 ** ((TRUE_ZP - mag) / 2.5)


def _single_star_image(total_flux, x0=100.3, y0=98.7, size=200, bg=100.0, sky=5.0, seed=0):
    rng = np.random.default_rng(seed)
    data = bg + rng.normal(0.0, sky, (size, size))
    _inject_star(data, x0, y0, total_flux)
    return _image(data), x0, y0


def _fwhm_stats(fwhm=FWHM):
    return FWHMMetadata(
        n_measurements=10,
        median_fwhm=fwhm,
        mean_fwhm=fwhm,
        std_fwhm=0.1,
        min_fwhm=fwhm - 0.5,
        max_fwhm=fwhm + 0.5,
        fwhm_vs_position=[],
        fwhm_vs_magnitude=[],
        fwhm_vs_counts=[],
    )


def _starfield(stars, fwhm=FWHM, size=400):
    return StarField.model_construct(
        catalog_stars=list(stars),
        fwhm_stats=_fwhm_stats(fwhm),
        image_metadata=ImageMetadata(image_id="t", width=size, height=size, exposure_time=1.0),
    )


# ---------------------------------------------------------------------------
# measure_simple_star_photometry — single Gaussian star
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_flux", [20000.0, 50000.0, 120000.0])
def test_single_star_recovers_injected_flux(total_flux):
    """Aperture (r = 2*FWHM ~ 7 px) captures ~99.7% of a FWHM-3.5 Gaussian."""
    img, x0, y0 = _single_star_image(total_flux)
    star = StarInSpace(ra=0, dec=0, x=x0, y=y0, magnitude=14.0)
    res = measure_simple_star_photometry(img, star, FWHM, _cfg())
    assert res is not None
    # r=2*FWHM aperture enclosed fraction is ~0.996; allow noise wiggle.
    assert res.flux == pytest.approx(total_flux, rel=0.02)


def test_single_star_snr_in_ballpark():
    """Background-limited SNR ~ flux / (sky_sigma * sqrt(N_pix)).

    The engine's reported SNR uses a shot+sky noise model, so it is >= the
    pure-sky estimate; we bound it from below and sanity-cap it.
    """
    img, x0, y0 = _single_star_image(50000.0, sky=5.0)
    star = StarInSpace(ra=0, dec=0, x=x0, y=y0, magnitude=14.0)
    res = measure_simple_star_photometry(img, star, FWHM, _cfg())
    assert res is not None
    npix = np.pi * (2 * FWHM) ** 2
    sky_limited = 50000.0 / (5.0 * np.sqrt(npix))
    assert res.snr > 0.5 * sky_limited
    assert res.snr < 5.0 * sky_limited


def test_brighter_star_has_higher_snr():
    img_f, x0, y0 = _single_star_image(20000.0, seed=7)
    img_b, _, _ = _single_star_image(120000.0, seed=7)
    star = StarInSpace(ra=0, dec=0, x=x0, y=y0, magnitude=14.0)
    faint = measure_simple_star_photometry(img_f, star, FWHM, _cfg())
    bright = measure_simple_star_photometry(img_b, star, FWHM, _cfg())
    assert bright.snr > faint.snr


def test_single_star_background_recovered():
    img, x0, y0 = _single_star_image(50000.0, bg=250.0, sky=4.0)
    star = StarInSpace(ra=0, dec=0, x=x0, y=y0, magnitude=14.0)
    res = measure_simple_star_photometry(img, star, FWHM, _cfg())
    assert res.background_level == pytest.approx(250.0, abs=2.0)
    assert res.background_std == pytest.approx(4.0, abs=1.5)


def test_instrumental_magnitude_matches_flux():
    img, x0, y0 = _single_star_image(50000.0)
    star = StarInSpace(ra=0, dec=0, x=x0, y=y0, magnitude=14.0)
    res = measure_simple_star_photometry(img, star, FWHM, _cfg())
    expected = -2.5 * np.log10(res.flux)
    assert res.instrumental_magnitude == pytest.approx(expected, abs=0.01)


def test_star_none_coordinates_returns_none():
    img, _, _ = _single_star_image(50000.0)
    star = StarInSpace(ra=0, dec=0, x=None, y=None, magnitude=14.0)
    assert measure_simple_star_photometry(img, star, FWHM, _cfg()) is None


def test_star_near_edge_returns_none():
    img, _, _ = _single_star_image(50000.0, size=200)
    star = StarInSpace(ra=0, dec=0, x=2.0, y=2.0, magnitude=14.0)
    assert measure_simple_star_photometry(img, star, FWHM, _cfg()) is None


def test_quality_flag_true_for_bright_isolated_star():
    img, x0, y0 = _single_star_image(80000.0)
    star = StarInSpace(ra=0, dec=0, x=x0, y=y0, magnitude=14.0)
    res = measure_simple_star_photometry(img, star, FWHM, _cfg())
    assert res.quality_flag is True


def test_quality_flag_false_when_min_snr_too_high():
    """Raise min_snr above the star's SNR -> quality flag flips off."""
    img, x0, y0 = _single_star_image(50000.0)
    star = StarInSpace(ra=0, dec=0, x=x0, y=y0, magnitude=14.0)
    res = measure_simple_star_photometry(img, star, FWHM, _cfg())
    high = SimplePhotometryConfig(gain=1.0, include_read_noise=False, min_snr=res.snr + 1000.0)
    res_high = measure_simple_star_photometry(img, star, FWHM, high)
    assert res.quality_flag is True
    assert res_high.quality_flag is False


# ---------------------------------------------------------------------------
# measure_simple_starfield_photometry — vectorized field
# ---------------------------------------------------------------------------


def _field_image_and_starfield(seed=1, size=400, sky=5.0):
    rng = np.random.default_rng(seed)
    data = 100.0 + rng.normal(0.0, sky, (size, size))
    positions = [(60, 60), (120, 90), (200, 150), (300, 250), (150, 300), (250, 80), (90, 200), (330, 330)]
    mags = [12.0, 13.0, 14.0, 15.0, 16.0, 12.5, 13.5, 14.5]
    stars = []
    for (px, py), m in zip(positions, mags, strict=False):
        _inject_star(data, px, py, _flux_for_mag(m))
        stars.append(
            StarInSpace(ra=0, dec=0, x=float(px), y=float(py), magnitude=float(m), magnitudes={"Johnson_V": float(m)})
        )
    return _image(data), _starfield(stars, size=size)


def test_starfield_recovers_all_star_fluxes():
    img, sf = _field_image_and_starfield()
    results, _ = measure_simple_starfield_photometry(img, sf, _cfg(), frame_index=None)
    assert len(results) == 8
    for r in results:
        expected = _flux_for_mag(r.star.magnitude)
        assert r.flux == pytest.approx(expected, rel=0.03)


def test_starfield_zero_point_recovered():
    img, sf = _field_image_and_starfield()
    _, summary = measure_simple_starfield_photometry(img, sf, _cfg(), frame_index=None)
    assert summary.zero_point is not None
    assert summary.zero_point == pytest.approx(TRUE_ZP, abs=0.05)


def test_starfield_summary_counts_and_snr():
    img, sf = _field_image_and_starfield()
    results, summary = measure_simple_starfield_photometry(img, sf, _cfg(), frame_index=None)
    assert summary.n_stars == len(results)
    assert summary.n_quality >= 1
    assert summary.median_snr > 0
    assert summary.stars_mag is not None and summary.stars_snr is not None
    assert len(summary.stars_mag) == len(summary.stars_snr)


def test_starfield_records_circular_aperture_geometry():
    """The summary carries the literal circular aperture/annulus pixel dims:
    radius/inner/outer = factor × FWHM, so a reader needn't re-derive them."""
    img, sf = _field_image_and_starfield()
    _, summary = measure_simple_starfield_photometry(img, sf, _cfg(), frame_index=None)
    geo = summary.aperture_geometry
    assert geo is not None and geo["shape"] == "circle"
    assert geo["fwhm_px"] == pytest.approx(FWHM)
    assert geo["aperture_radius_px"] == pytest.approx(2.0 * FWHM)  # aperture_radius_factor
    assert geo["bg_inner_px"] == pytest.approx(3.0 * FWHM)  # bg_inner_factor
    assert geo["bg_outer_px"] == pytest.approx(5.0 * FWHM)  # bg_outer_factor


def test_run_result_records_aperture_policy():
    """to_result() emits a run-level `photometry` block carrying the PSF-factor
    policy when the run measured photometry, so the output JSON documents how
    apertures were sized without the original config.yaml."""
    from senpai.engine.models.metadata import CollectionMetadata
    from senpai.engine.models.senpai import SenpaiRun, SiderealFrame

    frame = SiderealFrame(
        frame=_image(np.zeros((50, 50))),
        index=0,
        photometry_summary={"n_stars": 3, "aperture_geometry": {"shape": "circle"}},
    )
    run = SenpaiRun(
        id="t", num_frames=1, collect_metadata=CollectionMetadata(),
        sidereal_frames=[frame],
    )
    block = run.to_result().photometry
    pcfg = get_config().photometry
    assert block is not None
    assert block["aperture_radius_factor"] == pcfg.aperture_radius_factor
    assert block["bg_inner_factor"] == pcfg.bg_inner_factor
    assert block["bg_outer_factor"] == pcfg.bg_outer_factor
    assert "definition" in block


def test_run_result_omits_photometry_block_without_photometry():
    """A run that measured no photometry omits the run-level block entirely."""
    from senpai.engine.models.metadata import CollectionMetadata
    from senpai.engine.models.senpai import SenpaiRun, SiderealFrame

    frame = SiderealFrame(frame=_image(np.zeros((50, 50))), index=0, photometry_summary=None)
    run = SenpaiRun(
        id="t", num_frames=1, collect_metadata=CollectionMetadata(),
        sidereal_frames=[frame],
    )
    assert run.to_result().photometry is None


def test_starfield_no_fwhm_returns_empty():
    img, sf = _field_image_and_starfield()
    sf_no_fwhm = StarField.model_construct(
        catalog_stars=sf.catalog_stars,
        fwhm_stats=None,
        image_metadata=sf.image_metadata,
    )
    results, summary = measure_simple_starfield_photometry(img, sf_no_fwhm, _cfg(), frame_index=None)
    assert results == []
    assert summary.n_stars == 0


def test_starfield_no_catalog_stars_returns_empty():
    img, sf = _field_image_and_starfield()
    sf_empty = StarField.model_construct(
        catalog_stars=[],
        fwhm_stats=sf.fwhm_stats,
        image_metadata=sf.image_metadata,
    )
    results, summary = measure_simple_starfield_photometry(img, sf_empty, _cfg(), frame_index=None)
    assert results == []
    assert summary.n_stars == 0


# ---------------------------------------------------------------------------
# _assess_simple_quality — flag logic in isolation
# ---------------------------------------------------------------------------


def test_assess_quality_rejects_negative_flux():
    assert _assess_simple_quality(-1.0, 50.0, 0.0, _cfg()) is False


def test_assess_quality_rejects_low_snr():
    cfg = _cfg(min_snr=5.0)
    assert _assess_simple_quality(100.0, 2.0, 0.0, cfg) is False
    assert _assess_simple_quality(100.0, 10.0, 0.0, cfg) is True


def test_assess_quality_rejects_high_crowding():
    cfg = _cfg(max_crowding=0.3)
    assert _assess_simple_quality(100.0, 50.0, 0.5, cfg) is False
    assert _assess_simple_quality(100.0, 50.0, 0.1, cfg) is True


# ---------------------------------------------------------------------------
# crowding: _calculate_simple_crowding / _has_bright_neighbor
# ---------------------------------------------------------------------------


def test_crowding_zero_for_isolated_star():
    rng = np.random.default_rng(11)
    data = 100.0 + rng.normal(0.0, 3.0, (200, 200))
    _inject_star(data, 100, 100, 50000.0)
    crowd = _calculate_simple_crowding(data, 100.0, 100.0, 2 * FWHM)
    assert crowd < 0.05


def test_crowding_higher_with_bright_neighbor():
    rng = np.random.default_rng(12)
    iso = 100.0 + rng.normal(0.0, 3.0, (200, 200))
    _inject_star(iso, 100, 100, 50000.0)
    crowd_iso = _calculate_simple_crowding(iso, 100.0, 100.0, 2 * FWHM)

    crowded = iso.copy()
    # bright neighbour just outside the aperture (r=7), inside the 3x check ring
    _inject_star(crowded, 115, 100, 80000.0)
    crowd_neighbour = _calculate_simple_crowding(crowded, 100.0, 100.0, 2 * FWHM)
    assert crowd_neighbour > crowd_iso


def test_has_bright_neighbor_detects_blend():
    faint = StarInSpace(ra=0, dec=0, x=100.0, y=100.0, magnitude=18.0)
    bright = StarInSpace(ra=0, dec=0, x=103.0, y=100.0, magnitude=12.0)
    assert _has_bright_neighbor(faint, 18.0, [faint, bright], iso_radius_pix=10.0, delta_mag=2.0) is True


def test_has_bright_neighbor_ignores_distant_or_faint():
    faint = StarInSpace(ra=0, dec=0, x=100.0, y=100.0, magnitude=18.0)
    far_bright = StarInSpace(ra=0, dec=0, x=300.0, y=300.0, magnitude=12.0)
    near_similar = StarInSpace(ra=0, dec=0, x=103.0, y=100.0, magnitude=17.5)
    assert _has_bright_neighbor(faint, 18.0, [faint, far_bright], iso_radius_pix=10.0, delta_mag=2.0) is False
    assert _has_bright_neighbor(faint, 18.0, [faint, near_similar], iso_radius_pix=10.0, delta_mag=2.0) is False


def test_has_bright_neighbor_kdtree_matches_bruteforce():
    """KD-tree path and the O(n) fallback must agree."""
    from scipy.spatial import cKDTree

    faint = StarInSpace(ra=0, dec=0, x=100.0, y=100.0, magnitude=18.0)
    bright = StarInSpace(ra=0, dec=0, x=104.0, y=100.0, magnitude=12.0)
    stars = [faint, bright]
    positions = [[s.x, s.y] for s in stars]
    tree = (cKDTree(positions), [0, 1])
    brute = _has_bright_neighbor(faint, 18.0, stars, 10.0, 2.0)
    kd = _has_bright_neighbor(faint, 18.0, stars, 10.0, 2.0, kdtree=tree)
    assert brute == kd is True


# ---------------------------------------------------------------------------
# magnitude selection helpers
# ---------------------------------------------------------------------------


def test_get_best_magnitude_prefers_order():
    star = StarInSpace(ra=0, dec=0, magnitudes={"Sloan_r": 14.0, "Johnson_V": 13.5})
    assert _get_best_magnitude(star) == 13.5  # Johnson_V outranks Sloan_r


def test_get_best_magnitude_falls_back_to_primary():
    star = StarInSpace(ra=0, dec=0, magnitude=12.0)
    assert _get_best_magnitude(star) == 12.0


def test_get_best_magnitude_none_when_absent():
    star = StarInSpace(ra=0, dec=0)
    assert _get_best_magnitude(star) is None


def test_find_common_magnitude_system_picks_covered_filter():
    stars = [StarInSpace(ra=0, dec=0, magnitudes={"Johnson_V": float(m)}) for m in range(10)]
    assert _find_common_magnitude_system(stars) == "Johnson_V"


def test_find_common_magnitude_system_primary_fallback():
    stars = [StarInSpace(ra=0, dec=0, magnitude=float(m)) for m in range(5)]
    assert _find_common_magnitude_system(stars) == "primary"


def test_find_common_magnitude_system_empty():
    assert _find_common_magnitude_system([]) is None


def test_precompute_star_magnitudes_uses_common_system():
    stars = [StarInSpace(ra=0, dec=0, magnitudes={"Johnson_V": float(m), "Sloan_r": float(m) + 0.3}) for m in range(6)]
    cache = _precompute_star_magnitudes(stars)
    for s in stars:
        assert cache[id(s)] == s.magnitudes["Johnson_V"]


# ---------------------------------------------------------------------------
# completeness helpers
# ---------------------------------------------------------------------------


def test_isotonic_completeness_is_monotone_decreasing():
    mag = [10, 11, 12, 13, 14, 15, 16]
    # deliberately spiky input
    pct = [100, 90, 95, 70, 80, 30, 10]
    _, ys = _isotonic_completeness(mag, pct)
    assert all(ys[i] >= ys[i + 1] - 1e-9 for i in range(len(ys) - 1))


def test_completeness_limits_crossings():
    mag = [10, 11, 12, 13, 14, 15, 16, 17]
    pct = [100, 100, 100, 90, 60, 30, 10, 0]
    target, m50, m90 = _completeness_limits(mag, pct, target=0.5)
    assert m90 is not None and m50 is not None
    assert m90 < m50  # 90% completeness is reached at a brighter (smaller) mag
    assert 13.5 < m50 < 15.0
    assert target == pytest.approx(m50)


def test_completeness_limits_none_for_short_curve():
    assert _completeness_limits([10, 11], [100, 50]) == (None, None, None)


def test_completeness_limits_none_when_never_crosses():
    mag = [10, 11, 12, 13, 14]
    pct = [100, 100, 100, 100, 100]  # never drops below 50%
    _, m50, m90 = _completeness_limits(mag, pct, target=0.5)
    assert m50 is None and m90 is None


def test_compute_completeness_curve_rolls_over():
    """Build results spanning bright->faint with SNR decreasing; the curve
    should be ~100% at the bright end and drop toward 0 at the faint end."""
    results = []
    rng = np.random.default_rng(21)
    for mag in np.arange(10.0, 18.0, 0.25):
        for _ in range(6):
            # SNR falls steeply with magnitude; threshold is limiting_snr (3)
            snr = max(0.0, 10 ** ((15.0 - mag) / 2.5) + rng.normal(0, 0.5))
            star = StarInSpace(ra=0, dec=0, x=50.0, y=50.0, magnitude=float(mag), magnitudes={"Johnson_V": float(mag)})
            results.append(
                SimplePhotometryResult(
                    star=star, flux=max(snr, 0.1), flux_err=1.0, snr=snr,
                    background_level=0.0, background_std=1.0, aperture_radius=7.0,
                    crowding_factor=0.0, quality_flag=snr >= 3.0,
                )
            )
    sf = _starfield([r.star for r in results])
    comp_mag, comp_pct = compute_completeness_curve(results, sf, _cfg(), isolate=False)
    assert len(comp_mag) >= 3
    assert comp_pct[0] > comp_pct[-1]
    assert comp_pct[0] >= 90.0
    assert comp_pct[-1] <= 20.0


def test_compute_completeness_curve_empty_when_too_few():
    star = StarInSpace(ra=0, dec=0, x=50, y=50, magnitude=14.0, magnitudes={"Johnson_V": 14.0})
    res = [
        SimplePhotometryResult(
            star=star, flux=100.0, flux_err=1.0, snr=50.0, background_level=0.0,
            background_std=1.0, aperture_radius=7.0, crowding_factor=0.0, quality_flag=True,
        )
    ]
    sf = _starfield([star])
    comp_mag, comp_pct = compute_completeness_curve(res, sf, _cfg(), isolate=False)
    assert comp_mag == [] and comp_pct == []
