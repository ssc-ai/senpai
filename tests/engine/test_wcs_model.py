"""Unit tests for the WCS data model and the sidereal plate-solve seam.

Covers ``senpai.engine.models.astrometry``: ``WCSModel`` from/to-astropy conversion edge
cases (exception handling, pixel-shape inference, CRPIX rescaling) and
``WCSMetadata.from_wcsmodel`` derivation. Also covers the
``astrometry.error_on_plate_solve_failure`` raise-vs-warn seam of
``process_astrometry_fits_sidereal``.

All tests use synthetic WCS objects and mocks -- no network, astrometry, or catalog access.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.astrometry import WCSMetadata, WCSModel
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.starfield import StarField, StarListImage
from senpai.engine.processing.sidereal import process_astrometry_fits_sidereal


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialize the config singleton for the module."""
    initialize_config(CONFIG_DIR / "local.yaml")


def _make_wcs(
    ra: float = 45.0,
    dec: float = 30.0,
    crpix: float = 255.0,
    cdelt: float = 0.001,
    pixel_shape: tuple[int, int] | None = None,
) -> WCS:
    """Build a tangent-plane astropy WCS.

    Args:
        ra: Reference right ascension in degrees.
        dec: Reference declination in degrees.
        crpix: Reference pixel for both axes.
        cdelt: Plate scale in degrees per pixel.
        pixel_shape: Optional pixel shape to set on the WCS.

    Returns:
        A configured astropy ``WCS``.
    """
    w = WCS(naxis=2)
    w.wcs.crpix = [crpix, crpix]
    w.wcs.cdelt = [-cdelt, cdelt]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    if pixel_shape is not None:
        w.pixel_shape = pixel_shape
    return w


def _make_wcs_model_local(
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


class TestWCSModel:
    """Conversion edge cases for ``WCSModel``."""

    def test_to_astropy_wcs_returns_none_on_exception(self) -> None:
        """A failure constructing the astropy WCS yields None rather than raising."""
        m = _make_wcs_model_local()
        with patch("senpai.engine.models.astrometry.WCS", side_effect=Exception("boom")):
            result = m.to_astropy_wcs()
        assert result is None

    def test_from_astropy_wcs_without_image_shape_uses_pixel_shape(self) -> None:
        """With no image shape, the model infers dimensions from the WCS pixel shape."""
        w = _make_wcs(pixel_shape=(512, 512))
        m = WCSModel.from_astropy_wcs(w)
        assert m.NAXIS1 == 512
        assert m.NAXIS2 == 512
        assert pytest.approx(255.0, abs=0.1) == m.CRPIX1

    def test_from_astropy_wcs_close_pixel_shape_adjusts_crpix(self) -> None:
        """A pixel shape within tolerance of the image shape rescales CRPIX."""
        # pixel_shape (510,510) is within 10px of image_shape (512,512) -> CRPIX is scaled.
        w = _make_wcs(crpix=255.0, pixel_shape=(510, 510))
        m = WCSModel.from_astropy_wcs(w, image_shape=(512, 512))
        expected = 255.0 * (512.0 / 510.0)
        assert pytest.approx(expected, rel=1e-3) == m.CRPIX1


class TestWCSMetadata:
    """Derivation of ``WCSMetadata`` from a ``WCSModel``."""

    def test_from_wcsmodel_returns_metadata(self) -> None:
        """The derived metadata carries the field centre and a positive plate scale."""
        m = _make_wcs_model_local(ra=180.0, dec=0.0)
        metadata = WCSMetadata.from_wcsmodel(m)
        assert metadata.RA_center_deg == pytest.approx(180.0, abs=0.5)
        assert metadata.x_ifov_arcsec > 0


class TestErrorOnPlateSolveFailure:
    """The plate-solve failure seam of ``process_astrometry_fits_sidereal``.

    The solve entry point is ``solve_field_fits`` and the failure signal is a returned
    StarField with ``wcs is None``.
    """

    _MODULE = "senpai.engine.processing.sidereal"

    def _make_fits_image(self) -> ProcessedFitsImage:
        """Build a tiny blank frame.

        Returns:
            A 10x10 ``ProcessedFitsImage``.
        """
        header = fits.Header()
        header["NAXIS1"] = 10
        header["NAXIS2"] = 10
        return ProcessedFitsImage(
            data=np.zeros((10, 10)),
            header=header,
            data_type=np.dtype("float32"),
            metadata=ImageMetadata(width=10, height=10),
        )

    def _no_wcs_starfield(self) -> StarField:
        """Build the unsolved (no-WCS) starfield the failing solve returns.

        Returns:
            An unfit ``StarField`` with ``wcs is None``.
        """
        meta = ImageMetadata(width=10, height=10)
        return StarField(
            astrometric_fit_stars=[],
            detections=[],
            image_metadata=meta,
            fit=False,
            wcs=None,
        )

    def _mock_extract(self, img: ProcessedFitsImage) -> tuple[StarListImage, float, StarListImage]:
        """Build a stub source-extraction result for the given frame.

        Args:
            img: The frame whose metadata the stub detections carry.

        Returns:
            The ``(sources, seeing, sources)`` triple the extractor returns.
        """
        sl = StarListImage(detections=[], image_metadata=img.metadata)
        return (sl, 1.0, sl)

    def test_raises_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the flag set, an unsolvable frame raises ``SiderealSolveError``.

        Args:
            monkeypatch: Pytest fixture used to enable the raise flag.
        """
        from senpai.exceptions import SiderealSolveError

        monkeypatch.setattr(get_config().astrometry, "error_on_plate_solve_failure", True)
        img = self._make_fits_image()
        with (
            patch(f"{self._MODULE}.remove_column_and_row_medians", side_effect=lambda x: x),
            patch(
                f"{self._MODULE}.extract_sidereal_sources",
                return_value=self._mock_extract(img),
            ),
            patch(f"{self._MODULE}.solve_field_fits", return_value=self._no_wcs_starfield()),
            pytest.raises(SiderealSolveError, match="plate solve failed on sidereal frame"),
        ):
            process_astrometry_fits_sidereal(img, subtract_background=False)

    def test_returns_none_and_warns_when_not_configured(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With the flag unset, an unsolvable frame returns None and warns.

        Args:
            monkeypatch: Pytest fixture used to disable the raise flag.
            caplog: Pytest log-capture fixture.
        """
        monkeypatch.setattr(get_config().astrometry, "error_on_plate_solve_failure", False)
        img = self._make_fits_image()
        with (
            patch(f"{self._MODULE}.remove_column_and_row_medians", side_effect=lambda x: x),
            patch(
                f"{self._MODULE}.extract_sidereal_sources",
                return_value=self._mock_extract(img),
            ),
            patch(f"{self._MODULE}.solve_field_fits", return_value=self._no_wcs_starfield()),
            caplog.at_level(logging.WARNING, logger=self._MODULE),
        ):
            result = process_astrometry_fits_sidereal(img, subtract_background=False)
        assert result is None
        assert "plate solve failed on sidereal frame" in caplog.text
