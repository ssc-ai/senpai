"""Tests for detector-gain measurement from raw frame pairs (photon transfer)."""

from __future__ import annotations

import numpy as np
import pytest

from senpai.engine.observability.detector_gain import (
    find_burst_pairs,
    fit_gain,
    parse_frame_key,
    ptc_point,
)


class TestParseAndPair:
    def test_parses_burst_coordinates(self):
        k = parse_frame_key("/x/20260613T041107_calsats_SAT_43873_f0.fits")
        assert k.field == "calsats_SAT_43873" and k.f_index == 0

    def test_returns_none_on_unmatched_name(self):
        assert parse_frame_key("/x/randomname.fits") is None

    def test_pairs_consecutive_same_field(self):
        names = [f"/x/20260613T04110{i}_calsats_SAT_1_f{i}.fits" for i in range(4)]
        assert len(find_burst_pairs(names)) == 3  # f0-f1, f1-f2, f2-f3

    def test_does_not_pair_repeated_f0_tiles(self):
        # Two coverage tiles, both _f0 at different times: different fields.
        names = ["/x/20260613T041148_coverage_11_f0.fits",
                 "/x/20260613T041219_coverage_11_f0.fits"]
        assert find_burst_pairs(names) == []

    def test_does_not_pair_across_fields(self):
        names = ["/x/20260613T041107_satA_f0.fits",
                 "/x/20260613T041115_satB_f1.fits"]
        assert find_burst_pairs(names) == []


def _sky_pair(level, gain, shape=(800, 800), seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.poisson(level * gain, shape) / gain
    f2 = rng.poisson(level * gain, shape) / gain
    return f1, f2


class TestPtcPoint:
    def test_recovers_level_and_variance(self):
        f1, f2 = _sky_pair(1500.0, 2.0, shape=(1200, 1200))
        level, var = ptc_point(f1, f2)
        assert level == pytest.approx(1500.0, rel=0.02)
        # var_pixel = level / gain = 750
        assert var == pytest.approx(750.0, rel=0.08)

    def test_patch_clean_ignores_localized_stars(self):
        """Bright stars that do NOT cancel (rate tracking) only spoil the patches
        they fall in; the clean-patch sky variance is unaffected."""
        rng = np.random.default_rng(3)
        f1, f2 = _sky_pair(1500.0, 2.0, shape=(1280, 1280))
        ys, xs = rng.integers(0, 1280, 60), rng.integers(0, 1280, 60)
        f1[ys, xs] += 40000.0  # uncancelled stars scattered across the frame
        _, var = ptc_point(f1, f2)
        assert var == pytest.approx(750.0, rel=0.10)

    def test_degenerate_pair_returns_none(self):
        assert ptc_point(np.ones((50, 50)), np.ones((50, 50))) is None


class TestFitGain:
    def _ptc(self, gain, levels, seed=0):
        return [ptc_point(*_sky_pair(L, gain, seed=seed + i))
                for i, L in enumerate(levels)]

    def test_recovers_known_gain(self):
        gain = 1.6
        pts = self._ptc(gain, np.linspace(500, 3000, 12))
        fit = fit_gain(pts)
        assert fit is not None
        assert fit.gain == pytest.approx(gain, rel=0.10)
        assert fit.gain_lo <= fit.gain <= fit.gain_hi

    def test_theilsen_rejects_bad_pairs(self):
        """A few whole-bad pairs (slew, total cloud) sit far above the line; the
        Theil-Sen slope shrugs them off."""
        gain = 1.6
        pts = self._ptc(gain, np.linspace(500, 3000, 12))
        pts += [(800.0, 60000.0), (1500.0, 90000.0)]  # outlier pairs
        fit = fit_gain(pts)
        assert fit is not None
        assert fit.gain == pytest.approx(gain, rel=0.15)

    def test_needs_level_range(self):
        flat = [(1500.0, 750.0)] * 10  # no lever arm in level
        assert fit_gain(flat) is None

    def test_too_few_points(self):
        assert fit_gain([(500.0, 300.0), (1000.0, 600.0)]) is None
