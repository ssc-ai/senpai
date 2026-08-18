"""Tests for multi-band calibration with color terms (senpai.engine.photometry.color_terms).

The relation fit is m_catalog - m_inst = ZP + C * color_index. We build synthetic stars that
obey that relation exactly for a chosen (ZP, C), then assert the fitter recovers both. We also
exercise the sigma clipping (a few gross outliers must be rejected), the min-star guards, and
the higher-level calculate_multiband_calibration() path (color-term vs simple-ZP fallback).
"""

from __future__ import annotations

import numpy as np
import pytest

from senpai.core.config import initialize_config, settings
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.starfield import StarField, StarInSpace
from senpai.engine.photometry.color_terms import (
    BandCalibration,
    ColorTermFit,
    MultiBandCalibration,
    calculate_multiband_calibration,
    fit_color_term,
)
from senpai.engine.photometry.utils import SimplePhotometryConfig, SimplePhotometryResult


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialise the process-wide config, which the photometry helpers read."""
    initialize_config(CONFIG_DIR / "burr.yaml")
    settings.plotting.photometry = False
    settings.plotting.debug = False


def _synth_arrays(
    zp: float, color_coeff: float, n: int = 40, noise: float = 0.01, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (inst, cat, color) obeying cat = zp + color_coeff*color + inst."""
    rng = np.random.default_rng(seed)
    color = rng.uniform(0.2, 2.0, n)
    inst = rng.uniform(-12.0, -7.0, n)
    cat = zp + color_coeff * color + inst + rng.normal(0.0, noise, n)
    return inst, cat, color


# ---------------------------------------------------------------------------
# fit_color_term
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zp,coeff", [(25.0, 0.15), (22.5, -0.10), (30.0, 0.0)])
def test_fit_recovers_zp_and_color_term(zp: float, coeff: float) -> None:
    """The fit recovers a known zero point and colour coefficient."""
    inst, cat, color = _synth_arrays(zp, coeff, noise=0.005, seed=1)
    fit = fit_color_term(inst, cat, color, band="Johnson_V")
    assert fit is not None
    assert fit.zero_point == pytest.approx(zp, abs=0.02)
    assert fit.color_coefficient == pytest.approx(coeff, abs=0.02)
    assert fit.color_index_name == "BP-RP"
    assert fit.band == "Johnson_V"


def test_fit_reports_low_residual_for_clean_data() -> None:
    """Noiseless input yields a residual near zero, so the residual is meaningful."""
    inst, cat, color = _synth_arrays(25.0, 0.12, noise=0.003, seed=2)
    fit = fit_color_term(inst, cat, color)
    assert fit.rms_residual < 0.02
    assert fit.zero_point_err < 0.02
    # sigma clipping may drop a borderline point; keep the bulk of the sample
    assert fit.n_stars >= len(inst) - 2


def test_fit_uncertainties_grow_with_noise() -> None:
    """Reported uncertainties grow with the injected noise.

    Without this the uncertainties could be constants and still look plausible.
    """
    inst1, cat1, color1 = _synth_arrays(25.0, 0.1, noise=0.005, seed=3)
    inst2, cat2, color2 = _synth_arrays(25.0, 0.1, noise=0.08, seed=3)
    clean = fit_color_term(inst1, cat1, color1)
    noisy = fit_color_term(inst2, cat2, color2)
    assert noisy.zero_point_err > clean.zero_point_err
    assert noisy.rms_residual > clean.rms_residual


def test_fit_returns_none_below_min_stars() -> None:
    """Too few stars yields no fit rather than one determined by a handful of points."""
    inst, cat, color = _synth_arrays(25.0, 0.1, n=4)
    assert fit_color_term(inst, cat, color, min_stars=5) is None


def test_fit_sigma_clips_outliers() -> None:
    """A handful of gross outliers must be clipped, leaving ZP/C near truth."""
    inst, cat, color = _synth_arrays(25.0, 0.15, n=40, noise=0.005, seed=4)
    cat = cat.copy()
    cat[0] += 3.0
    cat[1] -= 2.5
    cat[2] += 4.0
    fit = fit_color_term(inst, cat, color, sigma_clip=2.5)
    assert fit is not None
    assert fit.zero_point == pytest.approx(25.0, abs=0.05)
    assert fit.color_coefficient == pytest.approx(0.15, abs=0.05)
    assert fit.clipped_fraction > 0.0


def test_fit_handles_nonfinite_values() -> None:
    """Non-finite magnitudes are dropped rather than poisoning the fit."""
    inst, cat, color = _synth_arrays(25.0, 0.1, n=30, noise=0.005, seed=5)
    cat = cat.copy()
    color = color.copy()
    cat[0] = np.nan
    color[1] = np.inf
    fit = fit_color_term(inst, cat, color)
    assert fit is not None
    assert fit.zero_point == pytest.approx(25.0, abs=0.05)
    assert fit.n_stars <= len(inst)  # non-finite rows dropped


def test_fit_none_when_too_few_finite() -> None:
    """If dropping non-finite values leaves too few points, no fit is reported."""
    inst, cat, color = _synth_arrays(25.0, 0.1, n=10, noise=0.005, seed=6)
    cat = cat.copy()
    cat[5:] = np.nan  # only 5 finite, but mask drops them below min_stars=8
    assert fit_color_term(inst, cat, color, min_stars=8) is None


# ---------------------------------------------------------------------------
# dataclass rounding contracts
# ---------------------------------------------------------------------------


def test_colortermfit_rounds_fields() -> None:
    """The fit result rounds its coefficients to the precision they carry."""
    fit = ColorTermFit(
        band="V",
        zero_point=25.123456,
        zero_point_err=0.0123456,
        color_coefficient=0.151234,
        color_coefficient_err=0.004321,
        color_index_name="BP-RP",
        n_stars=20,
        rms_residual=0.012345,
        clipped_fraction=0.1234,
    )
    assert fit.zero_point == 25.123
    assert fit.zero_point_err == 0.0123
    assert fit.color_coefficient == 0.1512
    assert fit.rms_residual == 0.0123
    assert fit.clipped_fraction == 0.123


def test_bandcalibration_rounds_fields() -> None:
    """The band calibration rounds its zero point to the precision it carries."""
    cal = BandCalibration(band="V", zero_point=25.98765, zero_point_err=0.012399)
    assert cal.zero_point == 25.988
    assert cal.zero_point_err == 0.0124
    assert cal.method == "simple"


# ---------------------------------------------------------------------------
# calculate_multiband_calibration
# ---------------------------------------------------------------------------


def _make_results(
    zp: float,
    coeff: float,
    band: str = "Johnson_V",
    n: int = 20,
    with_color: bool = True,
    exposure_time: float | None = 1.0,
    seed: int = 10,
) -> list:
    """Build photometry results obeying a known zero point and colour term.

    Flux follows ``cat = zp + coeff*color + inst`` with ``inst = -2.5*log10(flux/texp)``, and
    colour comes from Gaia BP-RP.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        color = rng.uniform(0.3, 1.8)
        cat_mag = rng.uniform(12.0, 16.0)
        inst = cat_mag - zp - coeff * color
        flux = 10 ** (-inst / 2.5) * exposure_time
        mags = {band: cat_mag}
        if with_color:
            rp = 14.0
            mags["Gaia_BP"] = rp + color
            mags["Gaia_RP"] = rp
        star = StarInSpace(ra=0, dec=0, x=50.0, y=50.0, magnitude=cat_mag, magnitudes=mags)
        results.append(
            SimplePhotometryResult(
                star=star,
                flux=flux,
                flux_err=flux / 100.0,
                snr=100.0,
                background_level=0.0,
                background_std=1.0,
                aperture_radius=7.0,
                crowding_factor=0.0,
                quality_flag=True,
            )
        )
    return results


def _starfield_for(results: list, exposure_time: float | None = 1.0) -> StarField:
    """Wrap results in the StarField the multiband calibration reads."""
    return StarField.model_construct(
        catalog_stars=[r.star for r in results],
        image_metadata=ImageMetadata(image_id="t", width=100, height=100, exposure_time=exposure_time),
    )


def test_multiband_recovers_color_term() -> None:
    """A multiband calibration recovers the colour term per band."""
    results = _make_results(25.0, 0.12, n=25)
    sf = _starfield_for(results)
    cal = calculate_multiband_calibration(results, sf, ["Johnson_V"], SimplePhotometryConfig())
    assert cal is not None
    band = cal.bands["Johnson_V"]
    assert band.method == "color_term"
    assert band.zero_point == pytest.approx(25.0, abs=0.02)
    assert band.color_term.color_coefficient == pytest.approx(0.12, abs=0.02)


def test_multiband_simple_fallback_without_color() -> None:
    """No BP/RP -> color term disabled, falls back to simple per-band ZP.

    With coeff=0 the simple mean ZP still recovers the true value.
    """
    results = _make_results(24.0, 0.0, n=15, with_color=False)
    sf = _starfield_for(results)
    cal = calculate_multiband_calibration(results, sf, ["Johnson_V"], SimplePhotometryConfig())
    assert cal is not None
    band = cal.bands["Johnson_V"]
    assert band.method == "simple"
    assert band.zero_point == pytest.approx(24.0, abs=0.05)
    assert band.color_term is None


def test_multiband_respects_exposure_time() -> None:
    """A longer exposure does not shift the zero point.

    Instrumental magnitude is computed from flux per second, so as long as flux scales with
    exposure time the zero point is exposure-independent.
    """
    texp = 30.0
    results = _make_results(25.0, 0.1, n=20, exposure_time=texp)
    sf = _starfield_for(results, exposure_time=texp)
    cal = calculate_multiband_calibration(results, sf, ["Johnson_V"], SimplePhotometryConfig())
    band = cal.bands["Johnson_V"]
    assert band.zero_point == pytest.approx(25.0, abs=0.03)


def test_multiband_none_for_empty_inputs() -> None:
    """Empty inputs yield no calibration rather than an empty one that reads as valid."""
    sf = _starfield_for([])
    assert calculate_multiband_calibration([], sf, ["Johnson_V"], SimplePhotometryConfig()) is None
    results = _make_results(25.0, 0.1, n=5)
    sf2 = _starfield_for(results)
    assert calculate_multiband_calibration(results, sf2, [], SimplePhotometryConfig()) is None


def test_multiband_skips_band_with_too_few_matches() -> None:
    """A band with too few matched stars is skipped rather than fitted badly."""
    results = _make_results(25.0, 0.1, n=20)
    sf = _starfield_for(results)
    # No star carries the "Sloan_r" magnitude -> band skipped -> no bands -> None
    cal = calculate_multiband_calibration(results, sf, ["Sloan_r"], SimplePhotometryConfig())
    assert cal is None


def test_multiband_ignores_poor_quality_results() -> None:
    """Poor-quality results are excluded, and a band left below the minimum is skipped.

    Flagging all but two of twenty drops the band under its three-star minimum.
    """
    results = _make_results(25.0, 0.1, n=20)
    for r in results[2:]:
        r.quality_flag = False
    sf = _starfield_for(results)
    cal = calculate_multiband_calibration(results, sf, ["Johnson_V"], SimplePhotometryConfig())
    assert cal is None


def test_multiband_disable_color_terms_uses_simple() -> None:
    """enable_color_terms=False forces the simple-ZP branch even with colors."""
    results = _make_results(25.0, 0.0, n=20)  # coeff 0 so simple ZP == truth
    sf = _starfield_for(results)
    cfg = SimplePhotometryConfig(enable_color_terms=False)
    cal = calculate_multiband_calibration(results, sf, ["Johnson_V"], cfg)
    band = cal.bands["Johnson_V"]
    assert band.method == "simple"
    assert band.zero_point == pytest.approx(25.0, abs=0.05)


def test_multibandcalibration_default_fields() -> None:
    """A default multiband calibration carries no bands and no method claim."""
    cal = MultiBandCalibration()
    assert cal.bands == {}
    assert cal.color_index_name == "BP-RP"
    assert cal.observation_filter is None
