"""Gaia sky-region cache (catalog.runner._query_gaia_sky).

Only the sky area not already cached is fetched online: a contained box reuses
the region (no query), a partial overlap fetches just the new sliver(s) and grows
coverage, a disjoint box starts a fresh region. Online calls are counted via a
monkeypatched query (no network); each fake "star" sits at the center of the
queried box so we can assert which sky was actually fetched.
"""

import numpy as np
import pytest

from senpai.catalog import runner


@pytest.fixture(autouse=True)
def _calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float, float, float]]:
    """Reset the sky cache and stub the online Gaia query, recording each fetch.

    Args:
        monkeypatch: Pytest fixture used to patch ``runner.gaia.query_by_ra_dec_bounds``.

    Returns:
        The list that accumulates one ``(min_ra, max_ra, min_dec, max_dec)`` tuple
        per online query, so tests can assert which sky was actually fetched.
    """
    runner._GAIA_SKY_CACHE.clear()
    calls: list[tuple[float, float, float, float]] = []

    def fake_query(
        min_ra: float,
        max_ra: float,
        min_dec: float,
        max_dec: float,
        faint_lim: float | None = None,
        bright_lim: float | None = None,
    ) -> list[dict[str, float | int]]:
        """Record the queried box and return a single star at its center."""
        calls.append((min_ra, max_ra, min_dec, max_dec))
        cra = np.deg2rad((min_ra + max_ra) / 2)
        cdec = np.deg2rad((min_dec + max_dec) / 2)
        return [{"ra": cra, "dec": cdec, "mv": 12.0, "source_id": len(calls)}]

    monkeypatch.setattr(runner.gaia, "query_by_ra_dec_bounds", fake_query)
    return calls


def test_first_query_pads_and_caches(_calls: list[tuple[float, float, float, float]]) -> None:
    """The first query fetches a pad-enlarged box and seeds one cached region."""
    runner._query_gaia_sky(255.0, 257.0, 43.0, 45.0, 21, -32)
    assert len(_calls) == 1
    pad = runner._GAIA_SKY_PAD_DEG
    assert _calls[0][0] == pytest.approx(255.0 - pad)  # padded fetch
    assert _calls[0][1] == pytest.approx(257.0 + pad)
    assert len(runner._GAIA_SKY_CACHE) == 1


def test_intra_batch_jitter_is_free(_calls: list[tuple[float, float, float, float]]) -> None:
    """Sub-pad shift plus a refinement nudge cost zero extra online queries.

    A frame shift of ~0.04 degrees and a refinement nudge both land inside the
    padded coverage, so no additional online query is issued.
    """
    runner._query_gaia_sky(255.0, 257.0, 43.0, 45.0, 21, -32)
    runner._query_gaia_sky(255.04, 257.04, 43.04, 45.04, 21, -32)
    runner._query_gaia_sky(255.001, 256.999, 43.0, 45.0, 21, -32)
    assert len(_calls) == 1  # only the initial fetch


def test_partial_overlap_fetches_only_sliver(_calls: list[tuple[float, float, float, float]]) -> None:
    """A real shift beyond the pad pulls only the new strip, not a full box."""
    runner._query_gaia_sky(255.0, 257.0, 43.0, 45.0, 21, -32)  # call 0: padded full
    runner._query_gaia_sky(255.5, 257.5, 43.0, 45.0, 21, -32)  # RA shift 0.5° -> 1 sliver
    assert len(_calls) == 2
    # the sliver is a thin RA strip on the right, NOT a ~2° full box
    smin_ra, smax_ra, smin_dec, smax_dec = _calls[1]
    assert (smax_ra - smin_ra) < 1.0       # thin in RA
    assert (smax_dec - smin_dec) > 1.5     # full height
    # one region, coverage grew to span both
    assert len(runner._GAIA_SKY_CACHE) == 1
    cov = runner._GAIA_SKY_CACHE[0]["box"]
    assert cov[0] <= 255.0 and cov[1] >= 257.5


def test_diagonal_shift_fetches_two_slivers(_calls: list[tuple[float, float, float, float]]) -> None:
    """A diagonal shift fetches the two strips of the L-shaped box difference."""
    runner._query_gaia_sky(255.0, 257.0, 43.0, 45.0, 21, -32)   # call 0
    runner._query_gaia_sky(255.5, 257.5, 43.5, 45.5, 21, -32)   # RA+Dec shift
    # box-difference of grown bbox vs prior coverage = an L = 2 strips
    assert len(_calls) == 3  # initial + 2 slivers
    assert len(runner._GAIA_SKY_CACHE) == 1


def test_disjoint_pointing_starts_new_region(_calls: list[tuple[float, float, float, float]]) -> None:
    """A far slew starts a second region rather than one giant spanning box."""
    runner._query_gaia_sky(255.0, 257.0, 43.0, 45.0, 21, -32)
    runner._query_gaia_sky(280.0, 282.0, 43.0, 45.0, 21, -32)  # far slew
    assert len(_calls) == 2
    assert len(runner._GAIA_SKY_CACHE) == 2  # not one giant box between them


def test_different_mag_limits_are_separate(_calls: list[tuple[float, float, float, float]]) -> None:
    """The same sky at a different faint limit is cached as its own region."""
    runner._query_gaia_sky(255.0, 257.0, 43.0, 45.0, 21, -32)
    runner._query_gaia_sky(255.0, 257.0, 43.0, 45.0, 18, -32)
    assert len(_calls) == 2
    assert len(runner._GAIA_SKY_CACHE) == 2


def test_ra_wrap_bypasses_cache(_calls: list[tuple[float, float, float, float]]) -> None:
    """A field crossing the RA=0 seam bypasses the cache entirely."""
    runner._query_gaia_sky(359.5, 360.5, 43.0, 45.0, 21, -32)  # crosses seam
    assert len(runner._GAIA_SKY_CACHE) == 0
