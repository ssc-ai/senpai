"""Pydantic models for WCS astrometric solutions and related metadata."""

import logging
from enum import StrEnum

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from astropy.wcs import WCS
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from senpai.core.config import AppConfig
from senpai.settings import settings

logger = logging.getLogger(__name__)


class WCSStatus(StrEnum):
    """Provenance of a WCS solution, tracking how it was derived or refined."""

    NO_WCS = "NO_WCS"
    SIDEREAL_FIT_WCS = "SIDEREAL_FIT_WCS"
    PIXEL_SHIFTED_WCS = "PIXEL_SHIFTED_WCS"
    KERNEL_REFINED_WCS = "KERNEL_REFINED_WCS"
    ASTROMETRY_REFIT_WCS = "ASTROMETRY_REFIT_WCS"
    POST_REFIT_REFINED_WCS = "POST_REFIT_REFINED_WCS"
    # Refinement ran but the result failed the absolute image-based validation
    # (no significant star flux at the positions the WCS predicts). The WCS is
    # kept for inspection but must not be trusted for astrometry.
    REFINED_UNVALIDATED_WCS = "REFINED_UNVALIDATED_WCS"


class WCSQualityMetrics(BaseModel):
    """Absolute, image-based quality measurement of a frame's WCS.

    Measures background-subtracted flux at the pixel positions the WCS
    predicts for the brightest catalog stars, against a random-position null
    distribution and a deliberately offset control grid. A correct WCS puts
    real star flux at the predicted positions; a poisoned one lands on blank
    sky, which no relative (fallback-based) check can detect.
    """

    method: str = "flux_at_catalog_positions"
    n_stars_tested: int
    box_radius_px: int
    frac_significant: float = Field(
        description="Fraction of tested stars whose flux exceeds the null p-th percentile"
    )
    control_frac_significant: float = Field(
        description="Same fraction for the control grid (predictions offset by control_offset_px)"
    )
    null_percentile: float
    passed: bool | None = Field(
        description="True/False = validation verdict; None = too few testable stars to judge"
    )
    # Residuals of the WCS refit against the star positions it was fitted to
    # (absent when refinement fell back to the propagated WCS).
    refit_rms_px: float | None = None
    refit_rms_arcsec: float | None = None
    n_refit_stars: int | None = None

    @field_serializer("frac_significant", "control_frac_significant")
    def _ser_frac(self, v: float) -> float:
        """Round a fraction field to 3 decimal places.

        Args:
            v (float): the value to round.

        Returns:
            float: the rounded value.
        """
        return round(v, 3)

    @field_serializer("refit_rms_px", "refit_rms_arcsec")
    def _ser_rms(self, v: float | None) -> float | None:
        """Round an RMS field to 3 decimal places, passing through ``None``.

        Args:
            v (float | None): the value to round, or ``None``.

        Returns:
            float | None: the rounded value, or ``None`` if the input was ``None``.
        """
        return round(v, 3) if v is not None else None


class ReturnAstrometryConfig(BaseModel):
    """Astrometry solver configuration reported alongside a solved field."""

    indices_series: str = Field(description="Indices series (5200/5200_LITE/4100/5200_LITE_4100)")
    max_sources: int = Field(description="Maximum number of sources to solve for")
    min_width_degrees: float = Field(description="Minimum width in degrees")
    max_width_degrees: float = Field(description="Maximum width in degrees")

    @classmethod
    def from_settings(cls) -> "ReturnAstrometryConfig":
        """Build the config from the global astrometry settings.

        Returns:
            ReturnAstrometryConfig: The config populated from ``settings.astrometry``.
        """
        return cls.from_app_config(settings)

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "ReturnAstrometryConfig":
        """Build the config from an explicit :class:`AppConfig` instance.

        Args:
            config (AppConfig): the application config to read from.

        Returns:
            ReturnAstrometryConfig: The config populated from ``config.astrometry``.
        """
        return cls(
            indices_series=config.astrometry.indices_series,
            max_sources=config.astrometry.max_sources,
            min_width_degrees=config.astrometry.min_width_degrees,
            max_width_degrees=config.astrometry.max_width_degrees,
        )


class WCSModel(BaseModel):
    """Serializable representation of a FITS WCS header, including SIP distortion."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields for dynamic SIP coefficients

    WCSAXES: int
    NAXIS1: int
    NAXIS2: int
    CRPIX1: float
    CRPIX2: float
    PC1_1: float
    PC1_2: float
    PC2_1: float
    PC2_2: float
    CDELT1: float
    CDELT2: float
    CUNIT1: str
    CUNIT2: str
    CTYPE1: str
    CTYPE2: str
    CRVAL1: float
    CRVAL2: float
    LONPOLE: float | None = None
    LATPOLE: float | None = None
    EQUINOX: float | None = None

    # SIP distortion coefficients (common ones defined, but extra='allow' permits
    # higher-order coefficients)
    A_ORDER: int | None = None
    B_ORDER: int | None = None
    AP_ORDER: int | None = None
    BP_ORDER: int | None = None

    # Forward coefficients
    A_0_0: float | None = None
    A_0_1: float | None = None
    A_0_2: float | None = None
    A_1_0: float | None = None
    A_1_1: float | None = None
    A_2_0: float | None = None

    B_0_0: float | None = None
    B_0_1: float | None = None
    B_0_2: float | None = None
    B_1_0: float | None = None
    B_1_1: float | None = None
    B_2_0: float | None = None

    # Inverse coefficients
    AP_0_0: float | None = None
    AP_0_1: float | None = None
    AP_0_2: float | None = None
    AP_1_0: float | None = None
    AP_1_1: float | None = None
    AP_2_0: float | None = None

    BP_0_0: float | None = None
    BP_0_1: float | None = None
    BP_0_2: float | None = None
    BP_1_0: float | None = None
    BP_1_1: float | None = None
    BP_2_0: float | None = None

    @classmethod
    def from_astrometrydotnet(cls, astrometry_net_wcs: PrimaryHDU) -> "WCSModel":
        """Build a WCSModel from an astrometry.net WCS FITS HDU.

        Args:
            astrometry_net_wcs (PrimaryHDU): the astrometry.net ``wcs.fits`` primary HDU.

        Returns:
            WCSModel: the converted WCS model (CD-matrix headers are stored as PC with
                unit CDELT).
        """
        header = astrometry_net_wcs.header

        # Try PC matrix first
        if all(f"PC{i}_{j}" in header for i, j in [(1, 1), (1, 2), (2, 1), (2, 2)]):
            pc1_1 = header["PC1_1"]
            pc1_2 = header["PC1_2"]
            pc2_1 = header["PC2_1"]
            pc2_2 = header["PC2_2"]
            cdelt1 = header.get("CDELT1")
            cdelt2 = header.get("CDELT2")
        else:
            # If using CD matrix, we store the CD values in PC and set CDELT to 1
            # This preserves the transformation while fitting our model
            pc1_1 = header.get("CD1_1", 0)
            pc1_2 = header.get("CD1_2", 0)
            pc2_1 = header.get("CD2_1", 0)
            pc2_2 = header.get("CD2_2", 0)
            cdelt1 = 1.0
            cdelt2 = 1.0

        # Get SIP coefficients if they exist
        sip_params = {}
        for key in header:
            if key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"] or key.startswith(
                ("A_", "B_", "AP_", "BP_")
            ):
                sip_params[key] = header[key]

        return cls(
            WCSAXES=header["WCSAXES"],
            CRPIX1=header["CRPIX1"],
            CRPIX2=header["CRPIX2"],
            NAXIS1=header["IMAGEW"],
            NAXIS2=header["IMAGEH"],
            PC1_1=pc1_1,
            PC1_2=pc1_2,
            PC2_1=pc2_1,
            PC2_2=pc2_2,
            CDELT1=cdelt1,
            CDELT2=cdelt2,
            CUNIT1=header["CUNIT1"],
            CUNIT2=header["CUNIT2"],
            CTYPE1=header["CTYPE1"],
            CTYPE2=header["CTYPE2"],
            CRVAL1=header["CRVAL1"],
            CRVAL2=header["CRVAL2"],
            LONPOLE=header.get("LONPOLE"),
            LATPOLE=header.get("LATPOLE"),
            EQUINOX=header.get("EQUINOX"),
            # Add SIP parameters if they exist
            **sip_params,
        )

    def to_astrometrydotnet_fits(self, output_path: str) -> None:
        """Write this model as an astrometry.net-style WCS FITS file.

        Args:
            output_path (str): destination path for the FITS file.
        """
        values = self.model_dump()
        values["IMAGEW"] = values["NAXIS1"]
        values["IMAGEH"] = values["NAXIS2"]

        del values["NAXIS1"]
        del values["NAXIS2"]

        hdu = fits.PrimaryHDU()
        for key, value in values.items():
            hdu.header[key] = value

        hdu.writeto(output_path)

    @classmethod
    def from_astropy_wcs(
        cls, astropy_wcs: WCS, image_shape: tuple[int, int] | None = None
    ) -> "WCSModel":
        """Convert an Astropy WCS object to a WCSModel.

        Args:
            astropy_wcs (WCS): The Astropy WCS object to convert.
            image_shape (tuple[int, int] | None): The actual shape of the image as
                ``(height, width)``. If provided, this overrides the pixel_shape in the WCS.

        Returns:
            WCSModel: The converted WCS model.
        """
        # Use relax=True to include SIP keywords in header
        header = astropy_wcs.to_header(relax=True)

        # Use provided image_shape if available
        if image_shape is not None:
            # NumPy arrays are (height, width) = (NAXIS2, NAXIS1) in FITS convention
            naxis1, naxis2 = image_shape[1], image_shape[0]

            # If the WCS has a pixel_shape that's different from the image_shape,
            # we need to adjust the reference pixel (CRPIX) values
            if astropy_wcs.pixel_shape is not None:
                wcs_width, wcs_height = astropy_wcs.pixel_shape

                # Only adjust if the shapes are close but not identical
                # This handles cases where the WCS might be slightly off
                if abs(wcs_width - naxis1) < 10 and abs(wcs_height - naxis2) < 10:
                    # Scale factors to adjust CRPIX values
                    scale_x = naxis1 / wcs_width
                    scale_y = naxis2 / wcs_height

                    # Adjust CRPIX values to account for the difference in dimensions
                    crpix1 = header.get("CRPIX1", 0) * scale_x
                    crpix2 = header.get("CRPIX2", 0) * scale_y
                else:
                    # If dimensions are very different, use original values
                    crpix1 = header.get("CRPIX1", 0)
                    crpix2 = header.get("CRPIX2", 0)
            else:
                crpix1 = header.get("CRPIX1", 0)
                crpix2 = header.get("CRPIX2", 0)
        else:
            # If no image_shape provided, use WCS pixel_shape if available
            if astropy_wcs.pixel_shape is not None:
                naxis1, naxis2 = astropy_wcs.pixel_shape
            else:
                naxis1, naxis2 = header.get("NAXIS1", 0), header.get("NAXIS2", 0)

            crpix1 = header.get("CRPIX1", 0)
            crpix2 = header.get("CRPIX2", 0)

        # Extract values from header
        wcs_dict = {
            "WCSAXES": header.get("WCSAXES", 2),
            "NAXIS1": naxis1,
            "NAXIS2": naxis2,
            "CRPIX1": crpix1,
            "CRPIX2": crpix2,
            "PC1_1": header.get("PC1_1", 0),
            "PC1_2": header.get("PC1_2", 0),
            "PC2_1": header.get("PC2_1", 0),
            "PC2_2": header.get("PC2_2", 0),
            "CDELT1": header.get("CDELT1", 1.0),
            "CDELT2": header.get("CDELT2", 1.0),
            "CUNIT1": header.get("CUNIT1", "deg"),
            "CUNIT2": header.get("CUNIT2", "deg"),
            "CTYPE1": header.get("CTYPE1", ""),
            "CTYPE2": header.get("CTYPE2", ""),
            "CRVAL1": header.get("CRVAL1", 0),
            "CRVAL2": header.get("CRVAL2", 0),
            "LONPOLE": header.get("LONPOLE"),
            "LATPOLE": header.get("LATPOLE"),
            "EQUINOX": header.get("EQUINOX"),
        }

        # Get SIP coefficients if they exist
        # First check header (with relax=True, SIP keywords should be included)
        sip_found_in_header = False
        for key in header:
            if key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"] or key.startswith(
                ("A_", "B_", "AP_", "BP_")
            ):
                wcs_dict[key] = header[key]
                if key in ["A_ORDER", "B_ORDER"]:
                    sip_found_in_header = True

        # Also check if WCS has SIP directly (as a fallback)
        # This handles cases where fit_wcs_from_points creates SIP but header doesn't include it
        if not sip_found_in_header and hasattr(astropy_wcs, "sip") and astropy_wcs.sip is not None:
            sip = astropy_wcs.sip
            # Extract SIP coefficients from the sip object
            if hasattr(sip, "a_order") and sip.a_order is not None:
                wcs_dict["A_ORDER"] = int(sip.a_order)
            if hasattr(sip, "b_order") and sip.b_order is not None:
                wcs_dict["B_ORDER"] = int(sip.b_order)
            if hasattr(sip, "ap_order") and sip.ap_order is not None:
                wcs_dict["AP_ORDER"] = int(sip.ap_order)
            if hasattr(sip, "bp_order") and sip.bp_order is not None:
                wcs_dict["BP_ORDER"] = int(sip.bp_order)

            # Extract coefficient matrices
            if hasattr(sip, "a") and sip.a is not None:
                for i in range(sip.a.shape[0]):
                    for j in range(sip.a.shape[1]):
                        if sip.a[i, j] != 0:
                            wcs_dict[f"A_{i}_{j}"] = float(sip.a[i, j])
            if hasattr(sip, "b") and sip.b is not None:
                for i in range(sip.b.shape[0]):
                    for j in range(sip.b.shape[1]):
                        if sip.b[i, j] != 0:
                            wcs_dict[f"B_{i}_{j}"] = float(sip.b[i, j])
            if hasattr(sip, "ap") and sip.ap is not None:
                for i in range(sip.ap.shape[0]):
                    for j in range(sip.ap.shape[1]):
                        if sip.ap[i, j] != 0:
                            wcs_dict[f"AP_{i}_{j}"] = float(sip.ap[i, j])
            if hasattr(sip, "bp") and sip.bp is not None:
                for i in range(sip.bp.shape[0]):
                    for j in range(sip.bp.shape[1]):
                        if sip.bp[i, j] != 0:
                            wcs_dict[f"BP_{i}_{j}"] = float(sip.bp[i, j])

        return cls(**wcs_dict)

    def to_astropy_wcs(self) -> WCS | None:
        """Convert this model to a WCS object.

        Returns:
            astropy.wcs.WCS | None: A WCS object to do calcs with, or ``None``
                if construction failed.
        """
        try:
            # Filter out None values from the model dump
            header_dict = {k: v for k, v in self.model_dump().items() if v is not None}

            # Ensure CTYPE has -SIP suffix if SIP coefficients are present
            # This is required for astropy to apply SIP distortion
            if self.A_ORDER and self.A_ORDER > 0:
                ctype1 = header_dict.get("CTYPE1", "")
                ctype2 = header_dict.get("CTYPE2", "")
                if ctype1 and not ctype1.endswith("-SIP"):
                    header_dict["CTYPE1"] = ctype1 + "-SIP"
                if ctype2 and not ctype2.endswith("-SIP"):
                    header_dict["CTYPE2"] = ctype2 + "-SIP"

            # Use relax=True to allow SIP keywords (A_ORDER, B_ORDER, etc.)
            wcs = WCS(header=header_dict, relax=True)
        except Exception as e:
            logger.warning(f"Failed to create WCS from model: {e}")
            wcs = None

        return wcs

    def get_boresight(self) -> tuple[float, float]:
        """Get the boresight coordinates.

        Returns:
            tuple[float, float]: ``(RA, Dec)`` of the reference point, in degrees.
        """
        return self.CRVAL1, self.CRVAL2

    def pix2world_0based(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[float, float]:
        """Convert 0-based pixel coordinates to world coordinates (RA, Dec).

        Args:
            x (float | np.ndarray): 0-based x pixel coordinate(s)
            y (float | np.ndarray): 0-based y pixel coordinate(s)

        Returns:
            tuple[float, float]: (RA, Dec) in degrees
        """
        wcs = self.to_astropy_wcs()
        # Add 1 to convert from 0-based to 1-based coordinates
        # SIP distortion is handled natively by astropy WCS
        ra, dec = wcs.pixel_to_world(x + 1, y + 1).ra.deg, wcs.pixel_to_world(x + 1, y + 1).dec.deg
        return ra, dec

    def world2pix_0based(self, ra: float, dec: float) -> tuple[float, float]:
        """Convert world coordinates (RA, Dec) to 0-based pixel coordinates.

        Args:
            ra (float): Right Ascension in degrees
            dec (float): Declination in degrees

        Returns:
            tuple[float, float]: (x, y) pixel coordinates in 0-based indexing
        """
        wcs = self.to_astropy_wcs()

        # Create a SkyCoord object from the RA and Dec values
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(ra * u.deg, dec * u.deg)

        # Convert to pixel coordinates
        # SIP distortion is handled natively by astropy WCS
        x, y = wcs.world_to_pixel(coords)

        # Subtract 1 from result to convert from 1-based to 0-based coordinates
        return x - 1, y - 1

    def get_fov_and_dimensions(self) -> tuple[float, float, int, int]:
        """Calculate the field of view and pixel dimensions from this WCS model.

        Returns:
            tuple containing:
                - field of view width in degrees
                - field of view height in degrees
                - pixel width
                - pixel height
        """
        # Get pixel dimensions
        pixel_width = self.NAXIS1
        pixel_height = self.NAXIS2

        # Get the WCS object
        wcs = self.to_astropy_wcs()

        # Calculate the field of view width and height
        # Get the coordinates of the corners of the image
        corners = [
            [0, 0],  # bottom left
            [pixel_width - 1, 0],  # bottom right
            [0, pixel_height - 1],  # top left
            [pixel_width - 1, pixel_height - 1],  # top right
        ]

        # Convert to world coordinates
        world_corners = wcs.wcs_pix2world(corners, 0)

        # Calculate the width and height FOV
        from astropy.coordinates import SkyCoord

        sky_corners = SkyCoord(world_corners[:, 0], world_corners[:, 1], unit="deg")

        # Width: maximum separation between left and right edges
        width_separation1 = sky_corners[0].separation(sky_corners[1]).deg
        width_separation2 = sky_corners[2].separation(sky_corners[3]).deg
        fov_width = max(width_separation1, width_separation2)

        # Height: maximum separation between top and bottom edges
        height_separation1 = sky_corners[0].separation(sky_corners[2]).deg
        height_separation2 = sky_corners[1].separation(sky_corners[3]).deg
        fov_height = max(height_separation1, height_separation2)

        return fov_width, fov_height, pixel_width, pixel_height


class WCSMetadata(BaseModel):
    """Human-readable summary of a WCS solution (FOV, scale, and field center)."""

    x_ifov_arcsec: float
    y_ifov_arcsec: float
    x_fov_degrees: float
    y_fov_degrees: float
    RA_center_deg: float
    Dec_center_deg: float
    RA_center_HMS: str
    Dec_center_DMS: str

    @field_serializer("x_ifov_arcsec", "y_ifov_arcsec", "x_fov_degrees", "y_fov_degrees")
    def serialize_3digits(self, v: float) -> float:
        """Round a field-of-view value to 3 decimal places for serialization.

        Args:
            v (float): The value to round.

        Returns:
            float: The value rounded to 3 decimal places.
        """
        return round(v, 3)

    @field_serializer("RA_center_deg", "Dec_center_deg")
    def serialize_6digits(self, v: float) -> float:
        """Round a center-coordinate value to 6 decimal places for serialization.

        Args:
            v (float): The value to round.

        Returns:
            float: The value rounded to 6 decimal places.
        """
        return round(v, 6)

    @classmethod
    def from_wcs(cls, wcs: WCS) -> "WCSMetadata":
        """Derive metadata (scale, FOV, and center) from an Astropy WCS.

        Args:
            wcs (WCS): The WCS to summarize.

        Returns:
            WCSMetadata: The computed metadata for the WCS.
        """
        x_ifov, y_ifov = wcs.proj_plane_pixel_scales()
        x_fov_deg, y_fov_deg = np.array([x_ifov.value, y_ifov.value]) * np.array(wcs.pixel_shape)

        # Calculate center pixel coordinates (using 1-based FITS convention)
        center_x = (wcs.pixel_shape[0] + 1) / 2
        center_y = (wcs.pixel_shape[1] + 1) / 2

        # Transform center pixel coordinates to sky coordinates
        ra_center, dec_center = (
            wcs.pixel_to_world(center_x, center_y).ra.deg,
            wcs.pixel_to_world(center_x, center_y).dec.deg,
        )

        with np.errstate(invalid="ignore"):
            ra_hms = Angle(ra_center, unit=u.deg).to_string(unit=u.hour, sep=":")
            dec_dms = Angle(dec_center, unit=u.deg).to_string(unit=u.deg, sep=":")

        return cls(
            x_ifov_arcsec=x_ifov.to(u.arcsec).value,
            y_ifov_arcsec=y_ifov.to(u.arcsec).value,
            x_fov_degrees=x_fov_deg,
            y_fov_degrees=y_fov_deg,
            RA_center_deg=ra_center,
            Dec_center_deg=dec_center,
            RA_center_HMS=ra_hms,
            Dec_center_DMS=dec_dms,
        )

    @classmethod
    def from_wcsmodel(cls, wcs_model: WCSModel) -> "WCSMetadata":
        """Derive metadata from a WCSModel by converting it to an Astropy WCS.

        Args:
            wcs_model (WCSModel): The WCS model to summarize.

        Returns:
            WCSMetadata: The computed metadata for the model.
        """
        return cls.from_wcs(wcs_model.to_astropy_wcs())
