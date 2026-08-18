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
    """Parsing burst frames and pairing the consecutive ones a difference needs."""

    def test_parses_burst_coordinates(self) -> None:
        """A burst frame's field and index parse from its name."""
        k = parse_frame_key("/x/20260613T041107_calsats_SAT_43873_f0.fits")
        assert k.field == "calsats_SAT_43873" and k.f_index == 0

    def test_returns_none_on_unmatched_name(self) -> None:
        """A name that does not match yields None rather than a partial record."""
        assert parse_frame_key("/x/randomname.fits") is None

    def test_pairs_consecutive_same_field(self) -> None:
        """Two consecutive frames of the same field pair."""
        names = [f"/x/20260613T04110{i}_calsats_SAT_1_f{i}.fits" for i in range(4)]
        assert len(find_burst_pairs(names)) == 3  # f0-f1, f1-f2, f2-f3

    def test_does_not_pair_repeated_f0_tiles(self) -> None:
        # Two coverage tiles, both _f0 at different times: different fields.
        """Repeated first-frames of different pointings do not pair.

        A coverage scan revisits index zero at each tile, and differencing two of those measures the
        sky changing rather than the read noise.
        """
        names = ["/x/20260613T041148_coverage_11_f0.fits", "/x/20260613T041219_coverage_11_f0.fits"]
        assert find_burst_pairs(names) == []

    def test_does_not_pair_across_fields(self) -> None:
        """Frames from different fields never pair."""
        names = ["/x/20260613T041107_satA_f0.fits", "/x/20260613T041115_satB_f1.fits"]
        assert find_burst_pairs(names) == []


def _sky_pair(
    level: float, gain: float, shape: tuple[int, int] = (800, 800), seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build two sky frames at a known level whose difference carries the shot noise."""
    rng = np.random.default_rng(seed)
    f1 = rng.poisson(level * gain, shape) / gain
    f2 = rng.poisson(level * gain, shape) / gain
    return f1, f2


class TestPtcPoint:
    """Turning one frame pair into a level-and-variance point."""

    def test_recovers_level_and_variance(self) -> None:
        """The point recovers the pair's level and difference variance."""
        f1, f2 = _sky_pair(1500.0, 2.0, shape=(1200, 1200))
        level, var = ptc_point(f1, f2)
        assert level == pytest.approx(1500.0, rel=0.02)
        # var_pixel = level / gain = 750
        assert var == pytest.approx(750.0, rel=0.08)

    def test_patch_clean_ignores_localized_stars(self) -> None:
        """Non-cancelling stars spoil only the patches they fall in.

        On a rate-tracked pair the stars do not subtract out, but the clean patches still
        report the sky variance the gain is derived from.
        """
        rng = np.random.default_rng(3)
        f1, f2 = _sky_pair(1500.0, 2.0, shape=(1280, 1280))
        ys, xs = rng.integers(0, 1280, 60), rng.integers(0, 1280, 60)
        f1[ys, xs] += 40000.0  # uncancelled stars scattered across the frame
        _, var = ptc_point(f1, f2)
        assert var == pytest.approx(750.0, rel=0.10)

    def test_degenerate_pair_returns_none(self) -> None:
        """A degenerate pair yields no point rather than a meaningless one."""
        assert ptc_point(np.ones((50, 50)), np.ones((50, 50))) is None


class TestFitGain:
    """Fitting gain from the variance-versus-level line."""

    def _ptc(self, gain: float, levels: list[float], seed: int = 0) -> list:
        """Build photon-transfer points following a known gain."""
        return [ptc_point(*_sky_pair(L, gain, seed=seed + i)) for i, L in enumerate(levels)]

    def test_recovers_known_gain(self) -> None:
        """The fit recovers a known gain from clean photon-transfer points."""
        gain = 1.6
        pts = self._ptc(gain, np.linspace(500, 3000, 12))
        fit = fit_gain(pts)
        assert fit is not None
        assert fit.gain == pytest.approx(gain, rel=0.10)
        assert fit.gain_lo <= fit.gain <= fit.gain_hi

    def test_theilsen_rejects_bad_pairs(self) -> None:
        """A few wholly-bad pairs do not move the fitted slope.

        A slew or total cloud puts a pair far above the line, and the Theil-Sen estimator is
        chosen precisely so those cannot drag it.
        """
        gain = 1.6
        pts = self._ptc(gain, np.linspace(500, 3000, 12))
        pts += [(800.0, 60000.0), (1500.0, 90000.0)]  # outlier pairs
        fit = fit_gain(pts)
        assert fit is not None
        assert fit.gain == pytest.approx(gain, rel=0.15)

    def test_needs_level_range(self) -> None:
        """Points spanning too little level yield no fit.

        Without a lever arm the slope is set by noise, and gain is the reciprocal of that slope.
        """
        flat = [(1500.0, 750.0)] * 10  # no lever arm in level
        assert fit_gain(flat) is None

    def test_too_few_points(self) -> None:
        """Too few points yield no fit rather than a two-point line."""
        assert fit_gain([(500.0, 300.0), (1000.0, 600.0)]) is None
