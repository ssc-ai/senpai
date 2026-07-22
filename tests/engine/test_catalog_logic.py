"""Pure-logic tests for senpai.catalog.

Query construction, mag-limit defaults, RA-wraparound handling, runner
box-region helpers, and the config faint-limit fallback. All network access
(astroquery Gaia/SDSS) is mocked; no query ever leaves the process.
"""

from __future__ import annotations

import sys
import types
from typing import NoReturn

import numpy as np
import pytest

from senpai.catalog import gaia, runner, sdss


# --------------------------------------------------------------------------- #
# Fake astroquery result table — a list of row-dicts that supports len() and
# colnames, mimicking the bits of astropy.table.Table the catalog code touches.
# --------------------------------------------------------------------------- #
class _FakeRow(dict):
    """Row supporting both row['k'] and hasattr(row, 'k') / row.k access."""

    def __getattr__(self, name: str) -> object:
        """Return the value stored under ``name`` as an attribute."""
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


class _FakeTable(list):
    """Minimal ``astropy.table.Table`` stand-in with ``len()`` and colnames."""

    @property
    def colnames(self) -> list[str]:
        """Return the column names inferred from the first row."""
        return list(self[0].keys()) if self else []


# --------------------------------------------------------------------------- #
# Gaia ADQL construction
# --------------------------------------------------------------------------- #
def _install_fake_gaia(
    monkeypatch: pytest.MonkeyPatch, capture: list[str], table: _FakeTable | None
) -> None:
    """Install a fake astroquery.gaia module that records launched ADQL.

    Args:
        monkeypatch: Fixture used to insert the fake module into ``sys.modules``.
        capture: List that each launched ADQL string is appended to.
        table: Canned result table returned by every job, or None.
    """

    class _Job:
        """Fake astroquery job wrapping a single ADQL query."""

        def __init__(self, adql: str) -> None:
            """Store the ADQL string this job was launched with."""
            self._adql = adql

        def get_results(self) -> _FakeTable | None:
            """Return the canned result table for this job."""
            return table

    class _Gaia:
        """Fake astroquery ``Gaia`` facade recording launched queries."""

        @staticmethod
        def launch_job(adql: str) -> _Job:
            """Record the ADQL and return a job wrapping it."""
            capture.append(adql)
            return _Job(adql)

    fake_mod = types.ModuleType("astroquery.gaia")
    fake_mod.Gaia = _Gaia
    monkeypatch.setitem(sys.modules, "astroquery.gaia", fake_mod)


def test_gaia_query_default_mag_limits_in_adql(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Gaia ADQL embeds the default mag limits, table, and box bounds."""
    captured: list[str] = []
    _install_fake_gaia(monkeypatch, captured, _FakeTable())
    gaia.query_by_ra_dec_bounds(150.0, 151.0, 2.0, 3.0)
    assert len(captured) == 1
    adql = captured[0]
    # default faint=21.0, bright=-32.0
    assert "BETWEEN -32.0 AND 21.0" in adql
    assert "gaiadr3.gaia_source" in adql
    assert "ra BETWEEN 150.0 AND 151.0" in adql
    assert "dec BETWEEN 2.0 AND 3.0" in adql


def test_gaia_query_custom_mag_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom faint/bright limits are embedded in the Gaia ADQL."""
    captured: list[str] = []
    _install_fake_gaia(monkeypatch, captured, _FakeTable())
    gaia.query_by_ra_dec_bounds(10.0, 11.0, 5.0, 6.0, faint_lim=18.0, bright_lim=6.0)
    assert "BETWEEN 6.0 AND 18.0" in captured[0]


def test_gaia_query_ra_wraparound_two_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A field straddling RA=0 splits into two seam-bounded ADQL queries."""
    captured: list[str] = []
    _install_fake_gaia(monkeypatch, captured, _FakeTable())
    # field straddling RA=0: min=359, max=1 -> crosses seam -> two ADQL queries
    gaia.query_by_ra_dec_bounds(359.0, 1.0, 2.0, 3.0)
    assert len(captured) == 2
    joined = " ".join(captured)
    assert "ra >= 359.0 AND ra <= 360.0" in joined
    assert "ra >= 0.0 AND ra <= 1.0" in joined


def test_gaia_query_parses_rows_and_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gaia rows parse into drop-in dicts with radians, bands, and proper motion."""
    captured: list[str] = []
    table = _FakeTable(
        [
            _FakeRow(
                source_id=42,
                ra=150.0,
                dec=2.0,
                G=15.0,
                BP=15.5,
                RP=14.5,
                pmra=10.0,
                pmdec=-5.0,
                parallax=2.0,
            )
        ]
    )
    _install_fake_gaia(monkeypatch, captured, table)
    stars = gaia.query_by_ra_dec_bounds(149.0, 151.0, 1.0, 3.0)
    assert len(stars) == 1
    star = stars[0]
    # ra/dec are returned in radians
    assert star["ra"] == pytest.approx(np.radians(150.0))
    assert star["catalog"] == "Gaia"
    assert star["mv"] == pytest.approx(15.0)  # primary G
    assert star["magnitudes"]["Gaia_G"] == 15.0
    # synthetic bands derived from BP-RP present
    assert "Johnson_V" in star["magnitudes"]
    assert "Sloan_r" in star["magnitudes"]
    # proper motion converted away from raw mas/yr
    assert star["ra_pm"] != 10.0


def test_gaia_query_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Gaia launch failure is swallowed and yields an empty list."""

    class _BoomGaia:
        """Fake Gaia facade whose ``launch_job`` always raises."""

        @staticmethod
        def launch_job(adql: str) -> NoReturn:
            """Raise to simulate a Gaia network failure."""
            raise RuntimeError("network down")

    fake_mod = types.ModuleType("astroquery.gaia")
    fake_mod.Gaia = _BoomGaia
    monkeypatch.setitem(sys.modules, "astroquery.gaia", fake_mod)
    assert gaia.query_by_ra_dec_bounds(10.0, 11.0, 1.0, 2.0) == []


# --------------------------------------------------------------------------- #
# SDSS SQL construction
# --------------------------------------------------------------------------- #
def test_sdss_query_default_mag_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDSS SQL embeds the default mag limits, table, and box bounds."""
    captured: list[str] = []

    def fake_query_sql(sql: str) -> _FakeTable:
        """Record the SQL and return an empty fake result table."""
        captured.append(sql)
        return _FakeTable()

    monkeypatch.setattr(sdss.SDSS, "query_sql", staticmethod(fake_query_sql))
    sdss.query_by_ra_dec_bounds(150.0, 151.0, 2.0, 3.0)
    assert len(captured) == 1
    sql = captured[0]
    # default faint=23.0, bright=-32.0
    assert "g BETWEEN -32.0 AND 23.0" in sql
    assert "FROM PhotoPrimary" in sql
    assert "ra BETWEEN 150.0 AND 151.0" in sql


def test_sdss_query_ra_wraparound_two_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A field straddling RA=0 splits into two seam-bounded SDSS queries."""
    captured: list[str] = []

    def fake_query_sql(sql: str) -> _FakeTable:
        """Record the SQL and return an empty fake result table."""
        captured.append(sql)
        return _FakeTable()

    monkeypatch.setattr(sdss.SDSS, "query_sql", staticmethod(fake_query_sql))
    sdss.query_by_ra_dec_bounds(359.0, 1.0, 2.0, 3.0)
    assert len(captured) == 2
    joined = " ".join(captured)
    assert "ra >= 359.0 AND ra <= 360.0" in joined
    assert "ra >= 0.0 AND ra <= 1.0" in joined


def test_sdss_query_parses_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDSS rows parse into drop-in dicts with a primary band and zero PM."""
    table = _FakeTable([_FakeRow(objid=7, ra=150.0, dec=2.0, u=18.0, g=16.0, r=15.5, i=15.0, z=14.8)])

    def fake_query_sql(sql: str) -> _FakeTable:
        """Return the canned fake result table."""
        return table

    monkeypatch.setattr(sdss.SDSS, "query_sql", staticmethod(fake_query_sql))
    stars = sdss.query_by_ra_dec_bounds(149.0, 151.0, 1.0, 3.0, primary_filter="r")
    assert len(stars) == 1
    star = stars[0]
    assert star["catalog"] == "SDSS"
    assert star["mv"] == pytest.approx(15.5)  # primary r
    assert star["magnitudes"]["Sloan_g"] == 16.0
    assert star["ra_pm"] == 0.0  # SDSS has no proper motion


def test_sdss_query_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An SDSS query failure is swallowed and yields an empty list."""

    def boom(sql: str) -> NoReturn:
        """Raise to simulate an SDSS query failure."""
        raise RuntimeError("boom")

    monkeypatch.setattr(sdss.SDSS, "query_sql", staticmethod(boom))
    assert sdss.query_by_ra_dec_bounds(1.0, 2.0, 1.0, 2.0) == []


def test_sdss_query_by_bounds_applies_safety_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    """query_by_bounds grows the box by a safety margin around the FOV."""
    captured: list[str] = []

    def fake_query_sql(sql: str) -> _FakeTable:
        """Record the SQL and return an empty fake result table."""
        captured.append(sql)
        return _FakeTable()

    monkeypatch.setattr(sdss.SDSS, "query_sql", staticmethod(fake_query_sql))
    # 1deg x 1deg field at dec=0 -> with 10% margin the RA/DEC span exceeds 1deg
    sdss.query_by_bounds(y_fov=1.0, x_fov=1.0, ra=150.0, dec=0.0)
    assert len(captured) == 1
    sql = captured[0]
    assert "FROM PhotoPrimary" in sql
    # bounding box should be slightly larger than 1 degree (margin applied)
    assert "g BETWEEN -32.0 AND 23.0" in sql


# --------------------------------------------------------------------------- #
# runner box-geometry helpers (pure functions)
# --------------------------------------------------------------------------- #
def test_box_overlap_and_contains() -> None:
    """_box_overlap and _box_contains classify inside/overlapping/disjoint boxes."""
    a = (0.0, 10.0, 0.0, 10.0)
    inside = (2.0, 4.0, 2.0, 4.0)
    overlapping = (5.0, 15.0, 5.0, 15.0)
    disjoint = (20.0, 30.0, 20.0, 30.0)
    assert runner._box_overlap(a, overlapping)
    assert runner._box_overlap(a, inside)
    assert not runner._box_overlap(a, disjoint)
    assert runner._box_contains(a, inside)
    assert not runner._box_contains(a, overlapping)


def test_box_difference_strips_ra_shift() -> None:
    """Growth only in RA yields exactly one right-hand difference strip."""
    # C inside U, only RA grew on the right -> exactly one right strip
    C = (0.0, 10.0, 0.0, 10.0)
    U = (0.0, 12.0, 0.0, 10.0)
    strips = runner._box_difference_strips(C, U)
    assert len(strips) == 1
    assert strips[0] == (10.0, 12.0, 0.0, 10.0)


def test_box_difference_strips_diagonal_two_strips() -> None:
    """Growth in both RA and Dec yields two difference strips."""
    C = (0.0, 10.0, 0.0, 10.0)
    U = (0.0, 12.0, 0.0, 12.0)  # grew in RA and Dec
    strips = runner._box_difference_strips(C, U)
    assert len(strips) == 2


def test_box_difference_strips_no_growth() -> None:
    """A box differenced against itself yields no strips."""
    C = (0.0, 10.0, 0.0, 10.0)
    assert runner._box_difference_strips(C, C) == []


def test_sky_dedup_key_prefers_source_id() -> None:
    """_sky_dedup_key keys on source_id, falling back to rounded ra/dec."""
    assert runner._sky_dedup_key({"source_id": 99, "ra": 1.0, "dec": 2.0}) == 99
    # falls back to rounded ra/dec when no source_id
    key = runner._sky_dedup_key({"source_id": None, "ra": 1.123456789, "dec": 2.0})
    assert key == (round(1.123456789, 8), 2.0)


# --------------------------------------------------------------------------- #
# faint_limit fallback in query_catalog_gaia (reads cfg.star_catalog.faint_limit)
# --------------------------------------------------------------------------- #
def test_query_catalog_gaia_faint_limit_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """query_catalog_gaia falls back to the configured faint limit.

    When ``faint_lim`` is None it pulls ``cfg.star_catalog.faint_limit`` via
    getattr and passes ``int(it)`` to the cached query.
    """
    captured = {}

    def fake_cached(
        wcs_tuple: tuple,
        faint_lim: int | None,
        bright_lim: int | None,
        pm_ts: float | None,
        max_stars: int | None,
    ) -> tuple[list[dict[str, object]], runner.ImageMetadata]:
        """Record the faint limit and return an empty result with metadata."""
        captured["faint_lim"] = faint_lim
        meta = runner.ImageMetadata(width=10, height=10, boresight_ra=1.0, boresight_dec=2.0)
        return [], meta

    monkeypatch.setattr(runner, "_query_catalog_gaia_cached", fake_cached)
    monkeypatch.setattr(runner, "_make_wcs_hashable", lambda wcs: ("k",))

    # Fake config with a star_catalog.faint_limit = 17.0
    fake_catalog = types.SimpleNamespace(faint_limit=17.0)
    fake_cfg = types.SimpleNamespace(star_catalog=fake_catalog)
    monkeypatch.setattr(runner, "get_config", lambda: fake_cfg)

    # Build a real-ish WCSModel only to satisfy validation downstream
    wcs = _wcs_model()
    runner.query_catalog_gaia(wcs, faint_lim=None)
    assert captured["faint_lim"] == 17  # int(17.0)


def test_query_catalog_gaia_faint_limit_none_when_config_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None configured faint limit stays None into the cached query."""
    captured = {}

    def fake_cached(
        wcs_tuple: tuple,
        faint_lim: int | None,
        bright_lim: int | None,
        pm_ts: float | None,
        max_stars: int | None,
    ) -> tuple[list[dict[str, object]], runner.ImageMetadata]:
        """Record the faint limit and return an empty result with metadata."""
        captured["faint_lim"] = faint_lim
        meta = runner.ImageMetadata(width=10, height=10, boresight_ra=1.0, boresight_dec=2.0)
        return [], meta

    monkeypatch.setattr(runner, "_query_catalog_gaia_cached", fake_cached)
    monkeypatch.setattr(runner, "_make_wcs_hashable", lambda wcs: ("k",))
    fake_cfg = types.SimpleNamespace(star_catalog=types.SimpleNamespace(faint_limit=None))
    monkeypatch.setattr(runner, "get_config", lambda: fake_cfg)

    runner.query_catalog_gaia(_wcs_model(), faint_lim=None)
    assert captured["faint_lim"] is None


# --------------------------------------------------------------------------- #
# Helper: build a tiny TAN WCSModel for runner tests.
# --------------------------------------------------------------------------- #
def _wcs_model() -> runner.WCSModel:
    """Build a tiny TAN WCSModel for the runner catalog-query tests."""
    from senpai.engine.models.astrometry import WCSModel

    return WCSModel(
        WCSAXES=2,
        NAXIS1=100,
        NAXIS2=100,
        CRPIX1=50.0,
        CRPIX2=50.0,
        PC1_1=1.0,
        PC1_2=0.0,
        PC2_1=0.0,
        PC2_2=1.0,
        CDELT1=-0.001,
        CDELT2=0.001,
        CUNIT1="deg",
        CUNIT2="deg",
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
        CRVAL1=150.0,
        CRVAL2=2.0,
    )


# --------------------------------------------------------------------------- #
# _validate_catalog_coverage RA-wraparound field-area logic (no logging assert,
# just that the function runs without error on a wrap field)
# --------------------------------------------------------------------------- #
def test_validate_coverage_collapses_large_positive_ra_span(caplog: pytest.LogCaptureFixture) -> None:
    """A >180 deg ascending RA span folds so sparse coverage still warns."""
    import logging

    caplog.set_level(logging.WARNING, logger="senpai.catalog.runner")
    # When min_ra/max_ra are given in ascending order, a >180deg span is folded to
    # 360-span (the wraparound branch). Here 10->300 (span 290) collapses to 70deg.
    runner._validate_catalog_coverage(
        stars_from_catalog=[{"ra": 0.0}] * 50,
        star_list=[],
        pixel_width=100,
        pixel_height=100,
        catalog_type="Gaia",
        min_ra=10.0,
        max_ra=300.0,
        min_dec=0.0,
        max_dec=1.0,
    )
    # span folds to 70deg * cos(0.5) * 1deg ~= 70 deg^2 -> 50/70 < 10 -> sparse warning
    assert "very sparse" in caplog.text


def test_validate_coverage_descending_ra_wrap_folds(caplog: pytest.LogCaptureFixture) -> None:
    """A descending RA-wrap field folds to a small area and does not warn."""
    # A descending wrap field (min_ra=359, max_ra=1) spans 2 degrees, so the
    # field area is ~4 deg^2 and 50 stars (12.5 stars/deg^2) is NOT sparse.
    # Before the (max_ra - min_ra) % 360 fix this computed ~715 deg^2 and
    # logged a spurious sparse-coverage warning.
    import logging

    with caplog.at_level(logging.WARNING):
        runner._validate_catalog_coverage(
            stars_from_catalog=[{"ra": 0.0}] * 50,
            star_list=[],
            pixel_width=100,
            pixel_height=100,
            catalog_type="Gaia",
            min_ra=359.0,
            max_ra=1.0,
            min_dec=0.0,
            max_dec=2.0,
        )
    assert "very sparse" not in caplog.text


def test_validate_coverage_empty_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    """An empty catalog result logs a NO-stars error."""
    import logging

    with caplog.at_level(logging.ERROR):
        runner._validate_catalog_coverage(
            stars_from_catalog=[],
            star_list=[],
            pixel_width=100,
            pixel_height=100,
            catalog_type="Gaia",
            min_ra=10.0,
            max_ra=11.0,
            min_dec=1.0,
            max_dec=2.0,
        )
    assert "NO stars" in caplog.text
