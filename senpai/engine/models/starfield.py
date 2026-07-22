"""Star, satellite, and starfield data models.

Defines the pixel-space and sky-space source records (stars and satellite/asteroid
detections), the list containers that carry them alongside image metadata, and the
:class:`StarField` aggregate that ties detections, catalog matches, the WCS
solution, and calibration/diagnostic products together for a single frame.
"""

import numpy as np
from pydantic import BaseModel, field_serializer, model_validator

from senpai.engine.models.astrometry import (
    ReturnAstrometryConfig,
    WCSMetadata,
    WCSModel,
    WCSQualityMetrics,
    WCSStatus,
)
from senpai.engine.models.metadata import DetectionMetadata, FWHMMetadata, ImageMetadata


class StarInImage(BaseModel):
    """A detected star in pixel space.

    Attributes:
        x: X pixel coordinate.
        y: Y pixel coordinate.
        counts: Integrated counts, if measured.
        snr: Signal-to-noise ratio, if measured.
        fwhm: Point-source FWHM in pixels, if measured.
    """

    x: float
    y: float
    counts: float | None = None
    snr: float | None = None
    fwhm: float | None = None

    @field_serializer("x", "y", "counts", "snr", "fwhm")
    def serialize_floats(self, v: float | None) -> float | None:
        """Round a float field to two decimals for serialization.

        Args:
            v: The field value, or ``None``.

        Returns:
            The value rounded to two decimals, or ``None``.
        """
        if v is None:
            return None
        return round(v, 2)


class SatelliteInImage(BaseModel):
    """A detected satellite/asteroid (point or streak) in pixel space.

    Holds the pixel position, optional sky coordinates and photometry, and
    streak-specific geometry that is ``None`` for point sources.

    Attributes:
        x: X pixel coordinate.
        y: Y pixel coordinate.
        snr: Signal-to-noise ratio, if measured.
        ra: Right ascension in degrees, if solved.
        dec: Declination in degrees, if solved.
        pixel_fwhm: Point-source FWHM in pixels, if measured.
        flux: Measured flux, if available.
        flux_err: Uncertainty on the flux, if available.
        instrumental_magnitude: Instrumental (uncalibrated) magnitude.
        calibrated_magnitudes: Calibrated magnitudes keyed by band.
        magnitude_errs: Magnitude uncertainties keyed by band.
        observation_filter: Filter/band the detection was observed in.
        detection_type: ``"point"`` or ``"streak"``.
        angle_deg: Streak orientation in degrees, in [0, 180).
        length_pixels: Streak length in pixels.
        rate_pixels_per_sec: Apparent rate in pixels per second.
        rate_arcsec_per_sec: Apparent rate in arcseconds per second.
    """

    x: float
    y: float
    snr: float | None = None
    ra: float | None = None
    dec: float | None = None
    pixel_fwhm: float | None = None
    flux: float | None = None
    flux_err: float | None = None
    instrumental_magnitude: float | None = None
    calibrated_magnitudes: dict[str, float] | None = None  # {band: mag}
    magnitude_errs: dict[str, float] | None = None  # {band: err}
    observation_filter: str | None = None  # e.g. "Clear", "V"
    # Streak-specific fields (null for point sources)
    detection_type: str | None = None  # "point" | "streak"
    angle_deg: float | None = None  # Streak angle [0, 180)
    length_pixels: float | None = None  # Streak length in pixels
    rate_pixels_per_sec: float | None = None
    rate_arcsec_per_sec: float | None = None

    @field_serializer("x", "y", "snr", "pixel_fwhm", "flux", "flux_err")
    def serialize_floats(self, v: float | None) -> float | None:
        """Round a float field to two decimals for serialization.

        Args:
            v: The field value, or ``None``.

        Returns:
            The value rounded to two decimals, or ``None``.
        """
        if v is None:
            return None
        return round(v, 2)

    @field_serializer("ra", "dec")
    def serialize_radec(self, v: float | None) -> float | None:
        """Round an RA/Dec field to four decimals for serialization.

        Args:
            v: The coordinate value in degrees, or ``None``.

        Returns:
            The value rounded to four decimals, or ``None``.
        """
        if v is None:
            return None
        return round(v, 4)

    @field_serializer("instrumental_magnitude")
    def serialize_instrumental_mag(self, v: float | None) -> float | None:
        """Round the instrumental magnitude to three decimals for serialization.

        Args:
            v: The instrumental magnitude, or ``None``.

        Returns:
            The value rounded to three decimals, or ``None``.
        """
        if v is None:
            return None
        return round(v, 3)

    @field_serializer("calibrated_magnitudes", "magnitude_errs")
    def serialize_mag_dicts(self, v: dict[str, float] | None) -> dict[str, float] | None:
        """Round each per-band magnitude value to three decimals for serialization.

        Args:
            v: A mapping of band name to magnitude value, or ``None``.

        Returns:
            A new mapping with each value rounded to three decimals, or ``None``.
        """
        if v is None:
            return None
        return {k: round(val, 3) for k, val in v.items()}


class StarInSpace(BaseModel):
    """A star in sky space, optionally projected into a specific image.

    Attributes:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        magnitude: Primary magnitude (kept for backward compatibility).
        magnitudes: All available magnitudes keyed by filter.
        catalog: Source catalog name (e.g. ``"sstrc7"``).
        catalog_id: Source catalog identifier for the star.
        x: X pixel coordinate when projected into an image, if available.
        y: Y pixel coordinate when projected into an image, if available.
        counts: Integrated counts, if measured.
        snr: Signal-to-noise ratio, if measured.
    """

    ra: float
    dec: float
    magnitude: float | None = None  # Primary magnitude (for backward compatibility)
    magnitudes: dict[str, float] | None = None  # All available magnitudes by filter
    catalog: str | None = None  # Source catalog name (e.g. "sstrc7")
    catalog_id: str | None = None  # Source catalog identifier for the star
    x: float | None = None
    y: float | None = None
    counts: float | None = None
    snr: float | None = None

    @field_serializer("x", "y", "snr", "counts", "magnitude")
    def serialize_floats(self, v: float | None) -> float | None:
        """Round a float field to two decimals for serialization.

        Args:
            v: The field value, or ``None``.

        Returns:
            The value rounded to two decimals, or ``None``.
        """
        if v is None:
            return None
        return round(v, 2)

    @field_serializer("ra", "dec")
    def serialize_radec(self, v: float | None) -> float | None:
        """Round an RA/Dec field to four decimals for serialization.

        Args:
            v: The coordinate value in degrees, or ``None``.

        Returns:
            The value rounded to four decimals, or ``None``.
        """
        if v is None:
            return None
        return round(v, 4)

    @field_serializer("magnitudes")
    def serialize_magnitudes(
        self, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        """Round per-band magnitudes, falling back to the primary magnitude.

        Args:
            v: A mapping of band name to magnitude value, or ``None``.

        Returns:
            A mapping with each value rounded to three decimals. When ``v`` is
            empty or ``None`` but a primary ``magnitude`` is set, a single
            ``{"Primary": magnitude}`` entry is returned; otherwise ``None``.
        """
        if v is None or len(v) == 0:
            # If magnitudes is empty but magnitude is set, create magnitudes with primary magnitude
            if self.magnitude is not None:
                return {"Primary": round(self.magnitude, 3)}
            return None
        return {k: round(v_val, 3) for k, v_val in v.items()}


class StarListSpace(BaseModel):
    """A list of sky-space stars with their associated image metadata.

    Attributes:
        stars: The stars in sky space.
        image_metadata: Metadata for the image the stars relate to.
    """

    stars: list[StarInSpace] = []
    image_metadata: ImageMetadata

    def centers_radec(self) -> np.ndarray:
        """Return the RA/Dec coordinates of all stars with valid positions.

        Returns:
            An ``(N, 2)`` array of ``[ra, dec]`` pairs in degrees for stars that
            have both coordinates set.
        """
        # Get all valid RA/Dec pairs
        return np.array(
            [
                [star.ra, star.dec]
                for star in self.stars
                if star.ra is not None and star.dec is not None
            ]
        )


class SatelliteListImage(BaseModel):
    """A list of satellite/asteroid detections with their image metadata.

    Attributes:
        detections: The satellite/asteroid detections in the image.
        image_metadata: Metadata for the image the detections came from.
    """

    detections: list[SatelliteInImage] = []
    image_metadata: ImageMetadata

    def centers_xy(self) -> np.ndarray:
        """Return the pixel positions and FWHM of all detections.

        Returns:
            An ``(N, 3)`` array of ``[x, y, pixel_fwhm]`` rows, one per detection.
        """
        return np.array(
            [
                [satellite.x, satellite.y, satellite.pixel_fwhm]
                for satellite in self.detections
            ]
        )


class StarListImage(BaseModel):
    """A list of detected stars in pixel space with image metadata.

    Attributes:
        detections: The detected stars in the image.
        image_metadata: Metadata for the image the detections came from.
        sat_level: Frame saturation level in ADU measured during detection, if
            known.
    """

    detections: list[StarInImage] = []
    image_metadata: ImageMetadata
    # Frame saturation level measured during detection (ADU). Downstream
    # FWHM measurement reuses it: estimating saturation from a
    # magnitude-sorted catalog sample is structurally unreliable (the
    # percentile lands in the faint bulk), whereas the detection-flux-sorted
    # sample measures it correctly.
    sat_level: float | None = None

    def centers_xy(self) -> np.ndarray:
        """Return the pixel positions and counts of all detected stars.

        Returns:
            An ``(N, 3)`` array of ``[x, y, counts]`` rows, one per detection.
        """
        return np.array([[star.x, star.y, star.counts] for star in self.detections])

    @classmethod
    def from_starfield(cls, starfield: "StarField") -> "StarListImage":
        """Build a StarListImage from a StarField's detected stars.

        Args:
            starfield: The starfield whose detections are extracted.

        Returns:
            A StarListImage containing the starfield's detections that have both
            pixel coordinates set, sharing the starfield's image metadata.
        """
        sources = [
            StarInImage(x=star.x, y=star.y, counts=star.counts)
            for star in starfield.detections
            if star.x is not None and star.y is not None
        ]

        return cls(detections=sources, image_metadata=starfield.image_metadata)


class StarField(BaseModel):
    """Aggregate of a frame's detections, catalog matches, WCS, and diagnostics.

    Ties together the detected sources, the catalog and astrometric-fit stars,
    the WCS solution and its metadata/quality, and per-frame calibration and
    distortion diagnostics.

    Attributes:
        astrometric_fit_stars: Stars used for the astrometric WCS fit.
        catalog_stars: Catalog stars projected into the frame.
        detections: Detected stars in pixel space.
        image_metadata: Metadata for the frame.
        fit: Whether a WCS fit has been performed.
        wcs: The WCS solution, if solved.
        wcs_metadata: Derived WCS metadata (FOV, plate scale, center).
        detection_metadata: Point-source detection metadata.
        astrometry: Astrometry backend configuration/result.
        wcs_status: Current WCS solution status.
        solver_tier: Escalation tier that produced the solve
            (T0/T1/T3; ``None`` if all failed or not applicable).
        solve_ms: Total wall time across attempted solver tiers, in ms.
        limiting_magnitude: Estimated limiting magnitude of the frame.
        fwhm_stats: Aggregated FWHM statistics for the frame.
        scale_factor: Scale factor applied to the image, if any.
        distortion_metrics: Optional scalar distortion diagnostics keyed by name.
        wcs_quality: Absolute image-based WCS validation result.
    """

    astrometric_fit_stars: list[StarInSpace] | None = None
    catalog_stars: list[StarInSpace] | None = None
    detections: list[StarInImage]
    image_metadata: ImageMetadata
    fit: bool = False
    wcs: WCSModel | None
    wcs_metadata: WCSMetadata | None = None
    detection_metadata: DetectionMetadata | None = None
    astrometry: ReturnAstrometryConfig | None = None
    wcs_status: WCSStatus = WCSStatus.NO_WCS
    # Cascade telemetry (solver_mode 'tetra3'/'chain'): which escalation tier produced the
    # solve (T0 boresight-refine / T1 tetra3 / T3 astrometry.net; None if all failed) and the
    # total wall time across all attempted tiers. None under 'dotnet'.
    solver_tier: str | None = None
    solve_ms: float | None = None
    limiting_magnitude: float | None = None
    fwhm_stats: FWHMMetadata | None = None
    scale_factor: float | None = None  # Track if image has been scaled
    # Optional per-field distortion diagnostics derived from the WCS
    # Keys are scalar metrics such as:
    #   - "delta_J"
    #   - "max_angle_variation_deg"
    #   - "max_length_variation_fraction"
    distortion_metrics: dict[str, float] | None = None
    # Absolute image-based WCS validation result (set after refinement)
    wcs_quality: WCSQualityMetrics | None = None

    @model_validator(mode="after")
    def create_wcs_metadata(self) -> "StarField":
        """Populate ``wcs_metadata`` from ``wcs`` when it is missing.

        Returns:
            The validated model, with WCS metadata derived from the WCS solution
            when a WCS is present but no metadata was supplied.
        """
        if self.wcs is not None and self.wcs_metadata is None:
            self.wcs_metadata = WCSMetadata.from_wcs(self.wcs.to_astropy_wcs())
        return self

    def centers_radec(self, centers: list[StarInSpace]) -> np.ndarray:
        """Return the RA/Dec coordinates of the given stars with valid positions.

        Args:
            centers: Stars to extract coordinates from.

        Returns:
            An ``(N, 2)`` array of ``[ra, dec]`` pairs in degrees for stars that
            have both coordinates set.
        """
        return np.array(
            [
                [star.ra, star.dec]
                for star in centers
                if star.ra is not None and star.dec is not None
            ]
        )

    def centers_xy(self, centers: list[StarInSpace]) -> np.ndarray:
        """Return the pixel coordinates of the given stars with valid positions.

        Args:
            centers: Stars to extract coordinates from.

        Returns:
            An ``(N, 2)`` array of ``[x, y]`` pairs for stars that have both
            pixel coordinates set.
        """
        return np.array(
            [
                [star.x, star.y]
                for star in centers
                if star.x is not None and star.y is not None
            ]
        )

    def astrometric_centers_radec(self) -> np.ndarray:
        """Return the RA/Dec coordinates of the astrometric-fit stars.

        Returns:
            An ``(N, 2)`` array of ``[ra, dec]`` pairs in degrees.
        """
        return self.centers_radec(self.astrometric_fit_stars)

    def astrometric_centers_xy(self) -> np.ndarray:
        """Return the pixel coordinates of the astrometric-fit stars.

        Returns:
            An ``(N, 2)`` array of ``[x, y]`` pairs.
        """
        return self.centers_xy(self.astrometric_fit_stars)

    def catalog_centers_radec(self) -> np.ndarray:
        """Return the RA/Dec coordinates of the catalog stars.

        Returns:
            An ``(N, 2)`` array of ``[ra, dec]`` pairs in degrees.
        """
        return self.centers_radec(self.catalog_stars)

    def catalog_centers_xy(self, limiting_magnitude: float | None = None) -> np.ndarray:
        """Return the pixel coordinates of the catalog stars, optionally filtered.

        Args:
            limiting_magnitude: Optional magnitude cutoff; only stars at least
                this bright are included. Falls back to the starfield's own
                ``limiting_magnitude`` when not provided.

        Returns:
            An ``(N, 2)`` array of ``[x, y]`` pairs, or ``None`` when the
            starfield has no catalog stars.
        """
        if self.catalog_stars is None:
            return None

        # Use StarField's limiting_magnitude if not explicitly provided
        if limiting_magnitude is None:
            limiting_magnitude = self.limiting_magnitude

        if limiting_magnitude is None:
            return self.centers_xy(self.catalog_stars)
        else:
            return self.centers_xy(
                [
                    star
                    for star in self.catalog_stars
                    if star.magnitude is not None
                    and star.magnitude <= limiting_magnitude
                ]
            )

    def detection_centers_xy(self) -> np.ndarray:
        """Return the pixel coordinates of the detected stars.

        Returns:
            An ``(N, 2)`` array of ``[x, y]`` pairs for detections with both
            pixel coordinates set.
        """
        return self.centers_xy(self.detections)
