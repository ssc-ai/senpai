"""Unit tests for engine utility and WCS-propagation helpers.

Covers synthetic-frame generation and star injection (``simulation``), the background
measurement fallback (``preprocessing``), FITS file loading (``file_io``), boresight
coordinate/header parsing (``coordinates``), the WCS-propagation edge cases in
``propagate_wcs`` (empty inputs, invalid status, limiting-magnitude estimation, and the
``NoConvergence`` fallbacks shared with ``senpai.astrometry.runner``), and the FITS
image-set-id parser.

All tests run on synthetic/deterministic data and mocks -- no network, astrometry, or
catalog access.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs.wcs import NoConvergence

from senpai.astrometry import runner
from senpai.core.config import initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.astrometry import WCSStatus
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.starfield import StarField, StarInSpace
from senpai.engine.utils.coordinates import parse_fits_coordinate, read_boresight_from_header
from senpai.engine.utils.file_io import load_fits_files
from senpai.engine.utils.preprocessing import measure_background
from senpai.engine.utils.propagate_wcs import (
    estimate_limiting_magnitude_from_photometry,
    existing_stars_from_wcs,
    find_local_maxima,
    get_global_shift_from_astrometric_stars,
    refine_wcs_by_kernel_convolution,
)
from senpai.engine.utils.simulation import add_gaussian, simulated_sidereal_frame


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialize the config singleton for the module."""
    initialize_config(CONFIG_DIR / "local.yaml")


def _make_starfield(
    catalog_stars: list[StarInSpace] | None = None,
    astrometric_fit_stars: list[StarInSpace] | None = None,
    width: int = 256,
    height: int = 256,
) -> StarField:
    """Build an unfit starfield with the given catalog and astrometric stars.

    Args:
        catalog_stars: Catalog stars for the field (may be None).
        astrometric_fit_stars: Astrometric fit stars for the field.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        An unfit ``StarField`` with no WCS.
    """
    return StarField(
        astrometric_fit_stars=astrometric_fit_stars or [],
        catalog_stars=catalog_stars,
        detections=[],
        image_metadata=ImageMetadata(width=width, height=height),
        fit=False,
        wcs=None,
    )


def _source(
    x: float = 64.0, y: float = 64.0, amplitude: float = 100.0, stddev: float = 2.0
) -> dict:
    """Build a Gaussian-source parameter dict for ``add_gaussian``.

    Args:
        x: Source centre x (column).
        y: Source centre y (row).
        amplitude: Peak amplitude.
        stddev: Isotropic standard deviation in pixels.

    Returns:
        A dict of Gaussian model parameters.
    """
    return {
        "x_mean": x,
        "y_mean": y,
        "x_stddev": stddev,
        "y_stddev": stddev,
        "amplitude": amplitude,
    }


class TestAddGaussian:
    """The ``add_gaussian`` in-place source injector."""

    def test_adds_positive_flux_at_center(self) -> None:
        """A source deposits positive flux at its centre pixel."""
        image = np.zeros((128, 128))
        add_gaussian(_source(x=64.0, y=64.0, amplitude=500.0), image)
        assert image[64, 64] > 0.0

    def test_modifies_array_in_place(self) -> None:
        """Injection mutates the supplied array rather than returning a new one."""
        image = np.zeros((128, 128))
        original_id = id(image)
        add_gaussian(_source(), image)
        assert id(image) == original_id

    def test_edge_clipping_does_not_raise(self) -> None:
        """A source near the image edge is clipped without raising."""
        image = np.zeros((128, 128))
        add_gaussian(_source(x=1.0, y=1.0, amplitude=100.0, stddev=3.0), image)

    def test_double_call_accumulates_flux(self) -> None:
        """Injecting the same source twice accumulates flux at the centre."""
        image = np.zeros((128, 128))
        add_gaussian(_source(x=64.0, y=64.0, amplitude=100.0), image)
        first_peak = image[64, 64]
        add_gaussian(_source(x=64.0, y=64.0, amplitude=100.0), image)
        assert image[64, 64] > first_peak


class TestSimulatedSiderealFrame:
    """The ``simulated_sidereal_frame`` catalog-star renderer."""

    def test_returns_nonzero_with_valid_catalog_stars(self) -> None:
        """A catalog star with a magnitude renders non-zero flux."""
        stars = [StarInSpace(ra=0.0, dec=0.0, magnitude=10.0, x=128.0, y=128.0)]
        sf = _make_starfield(catalog_stars=stars)
        result = simulated_sidereal_frame(sf)
        assert np.any(result > 0.0)

    def test_no_astrometric_star_fallback(self) -> None:
        """Simulation uses catalog stars only; stars without a magnitude are dropped.

        A catalog star lacking a magnitude yields an all-zero frame even when astrometric
        fit stars are present -- there is no fallback to the astrometric list. Downstream,
        ``prepare_sidereal_frame`` treats the all-zero result as "no synthetic frame".
        """
        catalog = [StarInSpace(ra=0.0, dec=0.0, magnitude=None, x=64.0, y=64.0)]
        astrometric = [StarInSpace(ra=0.0, dec=0.0, magnitude=8.0, x=64.0, y=64.0)]
        sf = _make_starfield(catalog_stars=catalog, astrometric_fit_stars=astrometric)
        result = simulated_sidereal_frame(sf)
        assert np.all(result == 0.0)

    def test_returns_zeros_when_no_valid_stars(self) -> None:
        """With no magnitude-bearing stars, the rendered frame is all zeros."""
        catalog = [StarInSpace(ra=0.0, dec=0.0, magnitude=None, x=64.0, y=64.0)]
        sf = _make_starfield(catalog_stars=catalog, astrometric_fit_stars=[])
        result = simulated_sidereal_frame(sf)
        assert np.all(result == 0.0)

    def test_output_shape_matches_metadata(self) -> None:
        """The rendered frame's shape follows the starfield's image metadata."""
        stars = [StarInSpace(ra=0.0, dec=0.0, magnitude=10.0, x=64.0, y=32.0)]
        sf = _make_starfield(catalog_stars=stars, width=200, height=100)
        result = simulated_sidereal_frame(sf)
        assert result.shape == (100, 200)

    def test_more_than_max_stars_does_not_error(self) -> None:
        """Rendering more candidate stars than ``max_stars`` completes without error."""
        stars = [
            StarInSpace(
                ra=float(i),
                dec=0.0,
                magnitude=float(i % 15 + 5),
                x=float(10 * i % 240 + 8),
                y=float(10 * i % 240 + 8),
            )
            for i in range(30)
        ]
        sf = _make_starfield(catalog_stars=stars)
        result = simulated_sidereal_frame(sf, max_stars=20)
        assert result.shape == (256, 256)

    def test_handles_gaussian_exception_gracefully(self) -> None:
        """An injector exception is swallowed, leaving an all-zero frame."""
        stars = [StarInSpace(ra=0.0, dec=0.0, magnitude=10.0, x=128.0, y=128.0)]
        sf = _make_starfield(catalog_stars=stars)
        with patch("senpai.engine.utils.simulation.add_gaussian", side_effect=ValueError("boom")):
            result = simulated_sidereal_frame(sf)
        assert np.all(result == 0.0)


class TestPreprocessing:
    """Background-measurement fallback behaviour."""

    def test_measure_background_fallback_on_tiny_image(self) -> None:
        """When ``Background2D`` fails, the background falls back to the global median."""
        img = np.ones((60, 60)) * 500.0
        with patch("senpai.engine.utils.preprocessing.Background2D", side_effect=ValueError):
            result = measure_background(img, box_size=5)
        assert result == pytest.approx(500.0)


class TestFileIO:
    """FITS file-loading error behaviour."""

    def test_load_fits_files_raises_on_nonexistent_path(self, tmp_path: Path) -> None:
        """Loading a missing path raises ``FileNotFoundError``.

        Args:
            tmp_path: Pytest temporary directory used to build a non-existent path.
        """
        with pytest.raises(FileNotFoundError):
            load_fits_files([tmp_path / "missing.fits"])


class TestParseFitsCoordinate:
    """The ``parse_fits_coordinate`` numeric/sexagesimal parser."""

    def test_float_returned_as_float(self) -> None:
        """A float coordinate is returned unchanged."""
        assert parse_fits_coordinate(30.0, is_ra=True) == pytest.approx(30.0)

    def test_int_returned_as_float(self) -> None:
        """An integer coordinate is coerced to a float."""
        result = parse_fits_coordinate(17, is_ra=False)
        assert result == pytest.approx(17.0)
        assert isinstance(result, float)

    def test_decimal_string_parsed(self) -> None:
        """A decimal string coordinate is parsed to a float."""
        assert parse_fits_coordinate("30.0", is_ra=True) == pytest.approx(30.0)

    def test_decimal_string_with_surrounding_whitespace(self) -> None:
        """Surrounding whitespace on a decimal string is tolerated."""
        assert parse_fits_coordinate("  17.5  ", is_ra=False) == pytest.approx(17.5)

    def test_ra_hms_converted_to_degrees(self) -> None:
        """An RA in HMS ('02 00 00') converts to decimal degrees."""
        result = parse_fits_coordinate("02 00 00", is_ra=True)
        assert result == pytest.approx(30.0, rel=1e-3)

    def test_dec_dms_positive_converted_to_degrees(self) -> None:
        """A positive Dec in DMS ('+12 00 00') converts to decimal degrees."""
        result = parse_fits_coordinate("+12 00 00", is_ra=False)
        assert result == pytest.approx(12.0, rel=1e-3)

    def test_dec_dms_negative_converted_to_degrees(self) -> None:
        """A negative Dec in DMS converts to decimal degrees."""
        result = parse_fits_coordinate("-45 30 00.00", is_ra=False)
        assert result == pytest.approx(-45.5)

    def test_unparseable_string_returns_none(self) -> None:
        """An unparseable coordinate string returns None."""
        assert parse_fits_coordinate("not-a-coordinate", is_ra=True) is None


class TestReadBoresightFromHeader:
    """The ``read_boresight_from_header`` key-priority and format handling."""

    def _make_header(self, **kwargs: object) -> fits.Header:
        """Build a FITS header from keyword arguments.

        Args:
            **kwargs: Header cards to set.

        Returns:
            A populated ``fits.Header``.
        """
        h = fits.Header()
        for k, v in kwargs.items():
            h[k] = v
        return h

    def test_decimal_ra_dec_keys(self) -> None:
        """Decimal RA/DEC keys are read directly."""
        h = self._make_header(RA=30.0, DEC=12.0)
        ra, dec = read_boresight_from_header(h)
        assert ra == pytest.approx(30.0)
        assert dec == pytest.approx(12.0)

    def test_sexagesimal_objctra_objctdec(self) -> None:
        """Sexagesimal OBJCTRA/OBJCTDEC cards are parsed to degrees."""
        h = self._make_header(OBJCTRA="02 00 00", OBJCTDEC="+12 00 00")
        ra, dec = read_boresight_from_header(h)
        assert ra == pytest.approx(30.0, rel=1e-3)
        assert dec == pytest.approx(12.0, rel=1e-3)

    def test_falls_back_to_crval_keys(self) -> None:
        """With no RA/DEC cards, the boresight falls back to CRVAL1/CRVAL2."""
        h = self._make_header(CRVAL1=15.0, CRVAL2=-30.0)
        ra, dec = read_boresight_from_header(h)
        assert ra == pytest.approx(15.0)
        assert dec == pytest.approx(-30.0)

    def test_ra_key_takes_priority_over_objctra(self) -> None:
        """The explicit RA card wins over OBJCTRA."""
        h = self._make_header(RA=10.0, OBJCTRA="02 00 00")
        ra, _ = read_boresight_from_header(h)
        assert ra == pytest.approx(10.0)

    def test_empty_header_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """An empty header returns (None, None) and warns for both axes.

        Args:
            caplog: Pytest log-capture fixture.
        """
        with caplog.at_level(logging.WARNING, logger="senpai.engine.utils.coordinates"):
            ra, dec = read_boresight_from_header(fits.Header())
        assert ra is None
        assert dec is None
        assert "Boresight RA not found" in caplog.text
        assert "Boresight Dec not found" in caplog.text

    def test_dec_missing_warns_only_for_dec(self, caplog: pytest.LogCaptureFixture) -> None:
        """A header with RA but no Dec warns only about the missing Dec.

        Args:
            caplog: Pytest log-capture fixture.
        """
        h = self._make_header(RA=180.0)
        with caplog.at_level(logging.WARNING, logger="senpai.engine.utils.coordinates"):
            ra, dec = read_boresight_from_header(h)
        assert ra == pytest.approx(180.0)
        assert dec is None
        assert "Boresight Dec not found" in caplog.text
        assert "Boresight RA not found" not in caplog.text


class TestPropagateWcsEdgeCases:
    """Edge-case behaviour of the WCS-propagation helpers."""

    def test_existing_stars_from_wcs_empty_returns_empty(self) -> None:
        """Projecting an empty star list returns an empty list."""
        result = existing_stars_from_wcs(MagicMock(), [])
        assert result == []

    def test_refine_wcs_invalid_status_raises(self) -> None:
        """Kernel-convolution refinement rejects a frame not in the pixel-shifted state."""
        mock_frame = MagicMock()
        mock_frame.starfield.wcs_status = WCSStatus.SIDEREAL_FIT_WCS
        with pytest.raises(ValueError, match="WCS status is not PIXEL_SHIFTED_WCS"):
            refine_wcs_by_kernel_convolution(mock_frame)

    def test_global_shift_no_astrometric_stars_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With no astrometric stars, the global shift warns and falls back.

        Args:
            caplog: Pytest log-capture fixture.
        """
        mock_frame = MagicMock()
        mock_frame.starfield.astrometric_fit_stars = []
        mock_frame.starfield.catalog_stars = []
        with caplog.at_level(logging.WARNING, logger="senpai.engine.utils.propagate_wcs"):
            get_global_shift_from_astrometric_stars(mock_frame, np.zeros((50, 50)))
        assert "No astrometric fit stars found" in caplog.text

    def test_global_shift_no_matches_returns_zero(self) -> None:
        """With no matched stars, the global shift is zero."""
        mock_frame = MagicMock()
        mock_frame.starfield.astrometric_fit_stars = []
        mock_frame.starfield.catalog_stars = []
        result = get_global_shift_from_astrometric_stars(mock_frame, np.zeros((50, 50)))
        assert result == (0.0, 0.0)

    def test_find_local_maxima_no_maxima_returns_empty(self) -> None:
        """A flat image has no local maxima."""
        result = find_local_maxima(np.zeros((50, 50)))
        assert len(result) == 0

    def test_estimate_limiting_mag_empty_returns_default(self) -> None:
        """With no photometry, the limiting magnitude is a default value."""
        result = estimate_limiting_magnitude_from_photometry(MagicMock(), [])
        assert result in (15.0, 16.0)

    def test_estimate_limiting_mag_fallback_unfiltered(self) -> None:
        """A sub-threshold SNR leaves no valid indices, so unfiltered magnitudes are used."""
        star = StarInSpace(ra=0.0, dec=0.0, magnitude=10.0, x=50.0, y=50.0)
        result = estimate_limiting_magnitude_from_photometry(MagicMock(), [(star, 0.05, 100.0)])
        assert isinstance(result, (int, float))

    def test_estimate_limiting_mag_polyfit_exception(self) -> None:
        """A polyfit failure falls back to the mean of the input magnitudes."""
        # Two distinct magnitudes pass the rank-deficiency guard so np.polyfit is reached
        # (and patched to fail), exercising the except fallback.
        star_a = StarInSpace(ra=0.0, dec=0.0, magnitude=10.0, x=50.0, y=50.0)
        star_b = StarInSpace(ra=0.0, dec=0.0, magnitude=11.0, x=60.0, y=60.0)
        with patch(
            "senpai.engine.utils.propagate_wcs.np.polyfit",
            side_effect=np.linalg.LinAlgError,
        ):
            result = estimate_limiting_magnitude_from_photometry(
                MagicMock(), [(star_a, 5.0, 100.0), (star_b, 5.0, 100.0)]
            )
        assert result == pytest.approx(11.5)


class TestParseImageSetId:
    """The ``_parse_image_set_id`` header/ORCHCOMM resolver."""

    def test_prefers_explicit_then_orchcomm(self) -> None:
        """The image-set id resolves from an explicit header first, else the ORCHCOMM card."""
        from senpai.engine.processing.collect import _parse_image_set_id

        set_id = "5873e160-1cfa-4757-8efd-ad1e6038c5ef"
        orchcomm = f"&{set_id}@[obs1]#[1:11]%[None]"

        # Explicit id header wins over ORCHCOMM.
        assert _parse_image_set_id({"IMGSETID": set_id, "ORCHCOMM": orchcomm}) == set_id
        # No explicit header (the prod/MDP case) -> parse it out of the ORCHCOMM card.
        assert _parse_image_set_id({"ORCHCOMM": orchcomm}) == set_id
        # Works against a real astropy FITS header, not just a dict.
        hdr = fits.Header()
        hdr["ORCHCOMM"] = orchcomm
        assert _parse_image_set_id(hdr) == set_id
        # Neither present -> "unknown".
        assert _parse_image_set_id({}) == "unknown"
        # Malformed ORCHCOMM (no &..@ delimiters) degrades gracefully -- never raises.
        assert _parse_image_set_id({"ORCHCOMM": "not-an-orchcomm-card"}) == "unknown"


# --------------------------------------------------------------------------- #
# NoConvergence fallback: world->pixel projection on a degenerate WCS
# --------------------------------------------------------------------------- #
class _DivergentWCS:
    """Fake astropy WCS whose ``all_world2pix`` always diverges."""

    def __init__(self, best: np.ndarray, divergent: np.ndarray) -> None:
        """Store the best-effort solution and divergence mask the failure carries.

        Args:
            best: The best-effort pixel solution reported by the exception.
            divergent: The per-point divergence mask.
        """
        self._best = best
        self._divergent = divergent

    def all_world2pix(self, world: np.ndarray, origin: int) -> np.ndarray:
        """Raise ``NoConvergence`` carrying the stored best-effort solution.

        Args:
            world: World coordinates to project (unused; always diverges).
            origin: Pixel origin convention (unused).

        Raises:
            NoConvergence: Always, to simulate a diverging SIP inverse.
        """
        raise NoConvergence(
            "SIP inverse diverged",
            best_solution=self._best,
            divergent=self._divergent,
        )


class _ConvergentWCS:
    """Fake astropy WCS whose ``all_world2pix`` returns a fixed result."""

    def __init__(self, result: np.ndarray) -> None:
        """Store the projection result to return.

        Args:
            result: The pixel coordinates to return from ``all_world2pix``.
        """
        self._result = result

    def all_world2pix(self, world: np.ndarray, origin: int) -> np.ndarray:
        """Return the stored projection result.

        Args:
            world: World coordinates to project (unused).
            origin: Pixel origin convention (unused).

        Returns:
            The stored pixel coordinates.
        """
        return self._result


def test_project_world_to_pixels_falls_back_on_noconvergence() -> None:
    """A diverging projection falls back to the exception's best-effort solution."""
    best = np.array([[10.0, 20.0], [30.0, 40.0]])
    wcs = _DivergentWCS(best=best, divergent=np.array([1]))
    out = runner.project_world_to_pixels(wcs, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(out, best)


def test_project_world_to_pixels_passthrough_when_converges() -> None:
    """A converging projection returns its result unchanged."""
    expected = np.array([[5.0, 6.0]])
    wcs = _ConvergentWCS(expected)
    out = runner.project_world_to_pixels(wcs, np.array([1.0]), np.array([2.0]))
    np.testing.assert_array_equal(out, expected)


class _DivergentWCSModel:
    """Duck-typed WCSModel whose ``to_astropy_wcs`` returns a divergent WCS."""

    def __init__(self, best: np.ndarray) -> None:
        """Store the best-effort solution the underlying divergent WCS reports.

        Args:
            best: The best-effort pixel solution reported on divergence.
        """
        self._best = best

    def to_astropy_wcs(self) -> _DivergentWCS:
        """Return a divergent WCS carrying the stored best-effort solution.

        Returns:
            A ``_DivergentWCS`` instance.
        """
        return _DivergentWCS(best=self._best, divergent=np.array([1]))


def test_existing_stars_from_wcs_falls_back_on_noconvergence() -> None:
    """On divergence, stars take the best-effort pixel positions with linkage preserved."""
    best = np.array([[10.0, 20.0], [30.0, 40.0]])
    stars = [
        StarInSpace(ra=1.0, dec=1.0, magnitude=5.0, catalog="gaia", catalog_id="a"),
        StarInSpace(ra=2.0, dec=2.0, magnitude=6.0, catalog="gaia", catalog_id="b"),
    ]
    updated = existing_stars_from_wcs(_DivergentWCSModel(best), stars)

    assert len(updated) == 2
    assert (updated[0].x, updated[0].y) == (10.0, 20.0)
    assert (updated[1].x, updated[1].y) == (30.0, 40.0)
    # Celestial coords and catalog linkage carried through unchanged.
    assert [s.catalog_id for s in updated] == ["a", "b"]
    assert [s.ra for s in updated] == [1.0, 2.0]


def test_existing_stars_from_wcs_empty_list_returns_empty() -> None:
    """Projecting an empty star list through a divergent WCS returns an empty list."""
    assert existing_stars_from_wcs(_DivergentWCSModel(np.empty((0, 2))), []) == []
