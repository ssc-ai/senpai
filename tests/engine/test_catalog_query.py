"""Unit tests for the star catalog query path (SSTR7 reader + catalog runner dispatch).

Covers ``senpai.catalog.sstr7`` (magnitude-band selection, WCS construction, RA-wrapping
zone selection, and the line-of-sight query functions) and the ``senpai.catalog.runner``
dispatch (implausible-FOV guardrail, unsupported-catalog-type rejection).

The fast tests use synthetic inputs and mocks. The line-of-sight query tests need local
SSTR7 catalog files; they resolve the catalog root from ``SSTRC7_PATH`` (default
``~/star-data/sstrc7``), carry the ``requires_catalog`` marker, and skip when the data is
absent (so they never run in CI).
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from senpai.catalog.sstr7 import get_star_mv, get_wcs, select_zone
from senpai.core.config import initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.astrometry import WCSModel


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialize the config singleton for the module."""
    initialize_config(CONFIG_DIR / "local.yaml")


def _catalog_path() -> str:
    """Resolve the local SSTR7 catalog root.

    Returns:
        The ``SSTRC7_PATH`` environment override, else the default under the home directory.
    """
    return os.environ.get("SSTRC7_PATH", str(Path.home() / "star-data" / "sstrc7"))


class TestGetStarMv:
    """Magnitude-band selection in ``get_star_mv``."""

    def test_open_band_takes_priority(self) -> None:
        """The open band (index 0) is preferred when populated, over Johnson_V (index 4)."""
        mv = np.full(18, 100.0)
        mv[0] = 10.5
        mv[4] = 9.0
        assert get_star_mv(mv) == pytest.approx(10.5)

    def test_gaia_g_fallback(self) -> None:
        """With no Johnson/Sloan-r coverage, the broad Gaia_G band (index 0) is used."""
        mv = np.full(18, 100.0)
        mv[0] = 10.5
        assert get_star_mv(mv) == pytest.approx(10.5)

    def test_johnson_r_fallback(self) -> None:
        """The Johnson_R band (index 5) is used when higher-priority bands are unpopulated."""
        mv = np.full(18, 100.0)
        mv[5] = 8.0
        assert get_star_mv(mv) == pytest.approx(8.0)

    def test_sloan_r_fallback(self) -> None:
        """The Sloan-r band (index 8) is used when higher-priority bands are unpopulated."""
        mv = np.full(18, 100.0)
        mv[8] = 7.5
        assert get_star_mv(mv) == pytest.approx(7.5)

    def test_johnson_v_fallback(self) -> None:
        """The Johnson_V band (index 4) is used when it is the only populated band."""
        mv = np.full(18, 100.0)
        mv[4] = 9.0
        assert get_star_mv(mv) == pytest.approx(9.0)

    def test_johnson_b_fallback(self) -> None:
        """The Johnson_B band (index 3) is used as a lower-priority fallback."""
        mv = np.full(18, 100.0)
        mv[3] = 11.0
        assert get_star_mv(mv) == pytest.approx(11.0)

    def test_all_saturated_returns_32(self) -> None:
        """With every band saturated, the sentinel magnitude 32 is returned."""
        mv = np.full(18, 100.0)
        assert get_star_mv(mv) == 32


class TestGetWcs:
    """WCS construction in ``get_wcs``."""

    def test_returns_wcs_object(self) -> None:
        """A WCS object is returned for valid inputs."""
        w = get_wcs(512, 512, 0.001, 0.001, 180.0, 0.0)
        assert w is not None

    def test_crval_matches_input(self) -> None:
        """The reference world coordinates match the requested centre."""
        w = get_wcs(100, 100, 0.005, 0.005, 45.0, 30.0)
        assert w.wcs.crval[0] == pytest.approx(45.0)
        assert w.wcs.crval[1] == pytest.approx(30.0)


class TestSelectZoneRaWrapping:
    """RA-wrapping branch coverage for ``select_zone``."""

    def _fake_index(self, num_ra: int = 10, num_dec: int = 10) -> list[list[dict]]:
        """Build a fake zone index grid.

        Args:
            num_ra: Number of RA zones per declination band.
            num_dec: Number of declination bands.

        Returns:
            A grid of zone descriptor dicts.
        """
        return [[{"pos": 0, "length": 0}] * num_ra for _ in range(num_dec)]

    def test_ra_wrapping_branch_executes(self) -> None:
        """An RA range that crosses zero (ra_min > ra_max) takes the wrapping branch."""
        result = select_zone(
            ra_min=6.2,
            ra_max=0.1,
            dec_min=-0.1,
            dec_max=0.1,
            zoneIndex=self._fake_index(),
            numRaZones=10,
            numDecZones=10,
        )
        assert isinstance(result, list)

    def test_normal_non_wrapping_query(self) -> None:
        """A standard non-wrapping RA range takes the normal branch."""
        result = select_zone(
            ra_min=0.5,
            ra_max=0.8,
            dec_min=-0.1,
            dec_max=0.1,
            zoneIndex=self._fake_index(),
            numRaZones=10,
            numDecZones=10,
        )
        assert isinstance(result, list)


def _make_wcs_model(
    ra: float = 180.0, dec: float = 0.0, scale: float = 0.001, width: int = 512, height: int = 512
) -> WCSModel:
    """Build a populated ``WCSModel``.

    Args:
        ra: Reference right ascension in degrees.
        dec: Reference declination in degrees.
        scale: Plate scale used in the PC matrix.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A populated ``WCSModel``.
    """
    return WCSModel(
        WCSAXES=2,
        NAXIS1=width,
        NAXIS2=height,
        CRPIX1=width / 2.0,
        CRPIX2=height / 2.0,
        PC1_1=-scale,
        PC1_2=0.0,
        PC2_1=0.0,
        PC2_2=scale,
        CDELT1=1.0,
        CDELT2=1.0,
        CUNIT1="deg",
        CUNIT2="deg",
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
        CRVAL1=ra,
        CRVAL2=dec,
    )


@pytest.mark.requires_catalog
class TestSstr7DataDriven:
    """Line-of-sight catalog queries against local SSTR7 files (skipped without data)."""

    def setup_method(self) -> None:
        """Skip the whole class when the local catalog is not available."""
        if not Path(_catalog_path()).exists():
            pytest.skip("SSTR7 catalog not available at " + _catalog_path())

    def test_query_by_los_returns_stars(self) -> None:
        """A line-of-sight query returns aligned row/column/magnitude arrays."""
        from senpai.catalog.sstr7 import query_by_los

        rr, cc, mm = query_by_los(
            height=512,
            width=512,
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            rootPath=_catalog_path(),
        )
        assert len(rr) > 0
        assert len(rr) == len(cc) == len(mm)

    def test_query_by_los_flipud_changes_rows(self) -> None:
        """Flipping up/down changes the returned row coordinates."""
        from senpai.catalog.sstr7 import query_by_los

        rr_normal, _, _ = query_by_los(
            height=512,
            width=512,
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            rootPath=_catalog_path(),
            flipud=False,
        )
        rr_flipped, _, _ = query_by_los(
            height=512,
            width=512,
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            rootPath=_catalog_path(),
            flipud=True,
        )
        assert not np.allclose(rr_normal, rr_flipped)

    def test_query_by_los_fliplr_changes_cols(self) -> None:
        """Flipping left/right changes the returned column coordinates."""
        from senpai.catalog.sstr7 import query_by_los

        _, cc_normal, _ = query_by_los(
            height=512,
            width=512,
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            rootPath=_catalog_path(),
            fliplr=False,
        )
        _, cc_flipped, _ = query_by_los(
            height=512,
            width=512,
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            rootPath=_catalog_path(),
            fliplr=True,
        )
        assert not np.allclose(cc_normal, cc_flipped)

    def test_query_by_los_radec_center_origin(self) -> None:
        """A centre-origin RA/Dec query returns stars."""
        from senpai.catalog.sstr7 import query_by_los_radec

        stars = query_by_los_radec(
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            origin="center",
            rootPath=_catalog_path(),
        )
        assert len(stars) > 0

    def test_query_by_los_radec_corner_origin(self) -> None:
        """A corner-origin RA/Dec query returns a list."""
        from senpai.catalog.sstr7 import query_by_los_radec

        stars = query_by_los_radec(
            y_fov=1.0,
            x_fov=1.0,
            ra=179.5,
            dec=-0.5,
            origin="corner",
            rootPath=_catalog_path(),
        )
        assert isinstance(stars, list)

    def test_query_by_los_radec_with_rotation(self) -> None:
        """A rotated RA/Dec query returns stars."""
        from senpai.catalog.sstr7 import query_by_los_radec_with_rotation

        stars = query_by_los_radec_with_rotation(
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            rotation=45.0,
            rootPath=_catalog_path(),
        )
        assert len(stars) > 0

    def test_query_with_filter_center(self) -> None:
        """A filter-centre query returns a list."""
        from senpai.catalog.sstr7 import query_by_los_radec_with_rotation

        stars = query_by_los_radec_with_rotation(
            y_fov=1.0,
            x_fov=1.0,
            ra=180.0,
            dec=0.0,
            rootPath=_catalog_path(),
            filter_center=600,
        )
        assert isinstance(stars, list)


class TestCatalogRunner:
    """Dispatch and guardrails in ``senpai.catalog.runner``."""

    def test_huge_fov_raises_sidereal_solve_error(self) -> None:
        """An implausible field of view is an unrecoverable solve failure (typed error)."""
        from senpai.catalog.runner import query_catalog_sstr7
        from senpai.exceptions import SiderealSolveError

        wcs = MagicMock()
        wcs.get_fov_and_dimensions.return_value = (100.0, 100.0, 512, 512)
        wcs.to_astropy_wcs.return_value = MagicMock()
        with pytest.raises(SiderealSolveError, match="Implausible sensor field of view"):
            query_catalog_sstr7(wcs, "/tmp/fake_catalog")  # noqa: S108

    def test_unsupported_catalog_type_raises(self) -> None:
        """An unsupported catalog type is rejected with a ``ValueError``."""
        from senpai.catalog.runner import query_catalog

        mock_config = MagicMock()
        mock_config.star_catalog.type = "unknown_catalog_type"
        mock_config.star_catalog.path = "/tmp"  # noqa: S108
        with (
            patch("senpai.catalog.runner.get_config", return_value=mock_config),
            pytest.raises(ValueError, match="not supported"),
        ):
            query_catalog(_make_wcs_model())

    @pytest.mark.requires_catalog
    def test_proper_motion_date_branch(self) -> None:
        """Supplying a proper-motion epoch exercises the proper-motion branch."""
        from senpai.catalog.runner import query_catalog_sstr7

        if not Path(_catalog_path()).exists():
            pytest.skip("catalog not available")
        result = query_catalog_sstr7(
            _make_wcs_model(),
            _catalog_path(),
            proper_motion_date=datetime(2020, 1, 1),
        )
        assert result is not None

    @pytest.mark.requires_catalog
    def test_bright_lim_filters_bright_stars(self) -> None:
        """A bright-magnitude limit removes stars brighter than the cutoff."""
        from senpai.catalog.runner import query_catalog_sstr7

        if not Path(_catalog_path()).exists():
            pytest.skip("catalog not available")
        result_all = query_catalog_sstr7(_make_wcs_model(), _catalog_path())
        result_bright_lim = query_catalog_sstr7(_make_wcs_model(), _catalog_path(), bright_lim=10)
        assert len(result_bright_lim.stars) <= len(result_all.stars)

    @pytest.mark.requires_catalog
    def test_max_stars_limits_output(self) -> None:
        """A max-stars cap bounds the number of returned stars."""
        from senpai.catalog.runner import query_catalog_sstr7

        if not Path(_catalog_path()).exists():
            pytest.skip("catalog not available")
        result = query_catalog_sstr7(_make_wcs_model(), _catalog_path(), max_stars=3)
        assert len(result.stars) <= 3
