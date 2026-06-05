"""Robust zero-point selection (utils._calculate_simple_zero_point).

The ZP must come from well-measured stars only. A faint catalog tail — where
forced photometry latches onto a neighbour/trail and reports a high SNR with a
wildly wrong per-star ZP — used to bias the (mean) ZP up by ~1 mag on rate
frames. The fix: high-SNR selection + sigma-clipped median. These tests pin that.
"""

from __future__ import annotations

import math

from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.starfield import StarField, StarInSpace
from senpai.engine.photometry.utils import (
    SimplePhotometryConfig,
    SimplePhotometryResult,
    _calculate_simple_zero_point,
)

TRUE_ZP = 26.0


def _result(mag, flux, snr, crowding=0.0):
    star = StarInSpace(ra=0.0, dec=0.0, magnitude=mag, x=None, y=None)
    return SimplePhotometryResult(
        star=star, flux=flux, flux_err=flux / max(snr, 1e-6), snr=snr,
        background_level=0.0, background_std=1.0, aperture_radius=4.0,
        crowding_factor=crowding, quality_flag=True,
    )


def _clean(mag, snr=100.0):
    """A star whose flux is exactly consistent with TRUE_ZP at 1s exposure."""
    return _result(mag, flux=10 ** ((TRUE_ZP - mag) / 2.5), snr=snr)


def _starfield():
    # fwhm_stats=None → skip the KD-tree neighbour path; this isolates the
    # SNR-cut + sigma-clip robustness (the contamination defence under test).
    # model_construct: the ZP routine only reads .fwhm_stats, .catalog_stars and
    # .image_metadata; skip full StarField validation for this focused unit test.
    return StarField.model_construct(
        image_metadata=ImageMetadata(
            image_id="t", width=100, height=100, exposure_time=1.0
        ),
        fwhm_stats=None,
        catalog_stars=None,
    )


def _cfg():
    return SimplePhotometryConfig()


def test_zp_recovers_clean_zero_point():
    results = [_clean(m) for m in [12, 13, 14, 15, 16] * 3]
    zp, err = _calculate_simple_zero_point(results, _starfield(), _cfg())
    assert zp is not None
    assert abs(zp - TRUE_ZP) < 0.02
    assert err is not None and err < 0.05


def test_zp_ignores_contaminated_faint_tail():
    """30 clean stars at ZP 26 + 10 faint stars measuring a bright neighbour's
    flux (per-star ZP ~31). The robust median must stay at 26; the old mean
    would be dragged to ~27.2."""
    clean = [_clean(m) for m in [12, 13, 14, 15, 16] * 6]  # 30 stars, ZP=26
    # faint catalog mag ~20 but flux of a ~mag-15 source (gross contamination)
    bright_flux = 10 ** ((TRUE_ZP - 15.0) / 2.5)
    contam = [_result(20.5, flux=bright_flux, snr=120.0) for _ in range(10)]

    zp, _ = _calculate_simple_zero_point(clean + contam, _starfield(), _cfg())
    assert abs(zp - TRUE_ZP) < 0.1  # robust to the tail

    # sanity: the naive mean really would have been pulled up
    naive = sum(r.star.magnitude + 2.5 * math.log10(r.flux) for r in clean + contam) / 40
    assert naive > TRUE_ZP + 0.8


def test_zp_excludes_low_snr_stars():
    """Low-SNR stars are below zp_min_snr and must not contribute. Give them a
    biased ZP; the result must still come from the clean high-SNR sample."""
    clean = [_clean(m, snr=100.0) for m in [12, 13, 14, 15, 16] * 3]
    lowsnr = [_result(15.0, flux=10 ** ((28.0 - 15.0) / 2.5), snr=2.0) for _ in range(20)]
    zp, _ = _calculate_simple_zero_point(clean + lowsnr, _starfield(), _cfg())
    assert abs(zp - TRUE_ZP) < 0.1


def test_zp_none_when_too_few_stars():
    zp, err = _calculate_simple_zero_point([_clean(12)], _starfield(), _cfg())
    assert zp is None and err is None
