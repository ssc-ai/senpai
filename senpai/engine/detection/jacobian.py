"""Local WCS Jacobians, and the streak geometry derived from them.

A rate-tracked streak is the image of a sky-motion vector under the local pixel-to-sky
mapping, so its direction and length vary across a distorted field. These helpers evaluate
that mapping by finite difference at a pixel and turn it into the streak vector, width and
convolution kernel appropriate to that position.
"""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

from senpai.engine.detection.kernels import rectangle_pyramoid

logger = logging.getLogger(__name__)


def local_jacobian(wcs: WCS, x: float, y: float, dsky: u.Quantity = 1 * u.arcsec) -> np.ndarray:
    """Compute the local Jacobian J = d(pixel)/d(sky) at pixel (x, y).

    Args:
        wcs: World coordinate solution to differentiate, distortion included.
        x: Pixel x (column) at which to evaluate the Jacobian.
        y: Pixel y (row) at which to evaluate the Jacobian.
        dsky: Finite-difference step on the sky.

    Returns:
        A 2x2 array ``[[dx/dRA, dx/dDec], [dy/dRA, dy/dDec]]`` in pixels per radian.

    """
    # Central sky coordinate
    sky0 = pixel_to_skycoord(x, y, wcs).transform_to("icrs")

    # Offset in RA
    sky_dra = SkyCoord(ra=sky0.ra + dsky, dec=sky0.dec, frame="icrs")
    x_dra, y_dra = skycoord_to_pixel(sky_dra, wcs)

    # Offset in Dec
    sky_ddec = SkyCoord(ra=sky0.ra, dec=sky0.dec + dsky, frame="icrs")
    x_ddec, y_ddec = skycoord_to_pixel(sky_ddec, wcs)

    # skycoord_to_pixel returns numpy arrays; cast to float scalars
    x_dra = float(x_dra)
    y_dra = float(y_dra)
    x_ddec = float(x_ddec)
    y_ddec = float(y_ddec)

    step_rad = dsky.to(u.rad).value
    dx_dra = (x_dra - x) / step_rad
    dy_dra = (y_dra - y) / step_rad
    dx_ddec = (x_ddec - x) / step_rad
    dy_ddec = (y_ddec - y) / step_rad

    return np.array([[dx_dra, dx_ddec], [dy_dra, dy_ddec]])


def streak_vector(J: np.ndarray, rate_ra: float, rate_dec: float) -> np.ndarray:
    """Map a sky rate through a Jacobian to a pixel-space streak vector.

    Args:
        J: Local Jacobian from :func:`local_jacobian`.
        rate_ra: Sky rate in RA, arcsec/s.
        rate_dec: Sky rate in Dec, arcsec/s.

    Returns:
        Pixel vector ``[vx, vy]`` per second.

    """
    rate = np.array([rate_ra, rate_dec]) * (np.pi / (180 * 3600))  # arcsec/s -> rad/s
    return J @ rate


def wcs_distortion_metrics(wcs: WCS, rate_ra: float, rate_dec: float, nx: int = 3, ny: int = 3) -> dict:
    """Measure how much the streak geometry varies across the field.

    Args:
        wcs: World coordinate solution to sample.
        rate_ra: Sky rate in RA, arcsec/s.
        rate_dec: Sky rate in Dec, arcsec/s.
        nx: Number of grid columns at which to evaluate the Jacobian.
        ny: Number of grid rows at which to evaluate the Jacobian.

    Returns:
        Mapping with ``delta_J`` (Jacobian variation relative to the field centre),
        ``max_angle_variation_deg``, ``max_length_variation_fraction``, and the raw
        ``jacobians`` and ``vectors`` arrays the metrics were derived from.

    """
    # Detector size
    ny_pix, nx_pix = wcs.array_shape
    xs = np.linspace(0, nx_pix - 1, nx)
    ys = np.linspace(0, ny_pix - 1, ny)

    jacobians = []
    vectors = []

    for y in ys:
        for x in xs:
            J = local_jacobian(wcs, x, y)
            v = streak_vector(J, rate_ra, rate_dec)
            jacobians.append(J)
            vectors.append(v)

    jacobians = np.array(jacobians)
    vectors = np.array(vectors)

    # Jacobian variation metric
    J0 = jacobians[((ny // 2) * nx + (nx // 2))]  # center
    delta_J = np.max(np.linalg.norm(jacobians - J0, axis=(1, 2)) / np.linalg.norm(J0))

    # Streak vector angle + magnitude
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    lengths = np.linalg.norm(vectors, axis=1)

    max_angle_var = np.max(np.abs(angles - angles[nx * ny // 2])) * 180 / np.pi
    max_len_var = np.max(np.abs(lengths / lengths[nx * ny // 2] - 1))

    return dict(
        delta_J=float(delta_J),
        max_angle_variation_deg=float(max_angle_var),
        max_length_variation_fraction=float(max_len_var),
        jacobians=jacobians,
        vectors=vectors,
    )


def compute_effective_sky_motion_vector(
    wcs: WCS,
    x_ref: float,
    y_ref: float,
    angle_rad: float,
    dsky: u.Quantity = 1 * u.arcsec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive an effective sky-motion direction vector from a central streak orientation.

    The result is a unit vector in sky-coordinate space (RA/Dec) such that
    J_ref @ rate_hat points along the observed pixel streak direction at (x_ref, y_ref),
    where J_ref is the local Jacobian at the reference pixel.

    Args:
        wcs: World coordinate solution to differentiate.
        x_ref: Reference pixel x at which the streak orientation was measured.
        y_ref: Reference pixel y at which the streak orientation was measured.
        angle_rad: Observed streak angle in pixel space, radians.
        dsky: Finite-difference step on the sky.

    Returns:
        ``(J_ref, rate_hat, e_perp)`` -- the reference Jacobian, the unit sky-motion
        direction, and the unit cross-track sky direction perpendicular to it.

    """
    J_ref = local_jacobian(wcs, x_ref, y_ref, dsky=dsky)

    # Unit vector in pixel space along the observed streak
    v_dir_pix = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    try:
        rate_dir = np.linalg.solve(J_ref, v_dir_pix)
    except np.linalg.LinAlgError:
        rate_dir = np.linalg.pinv(J_ref) @ v_dir_pix

    norm = np.linalg.norm(rate_dir)
    if norm == 0:
        # Fallback: pure RA direction
        rate_hat = np.array([1.0, 0.0])
    else:
        rate_hat = rate_dir / norm

    # Perpendicular sky direction (cross-track)
    e_perp = np.array([-rate_hat[1], rate_hat[0]])

    return J_ref, rate_hat, e_perp


def compute_local_streak_vector_from_jacobian(
    wcs: WCS,
    J_ref: np.ndarray,
    rate_hat: np.ndarray,
    pixel_length: float,
    x_ref: float,
    y_ref: float,
    x: float,
    y: float,
    dsky: u.Quantity = 1 * u.arcsec,
) -> tuple[np.ndarray, float, float]:
    """Compute the local pixel streak vector at (x, y) from WCS Jacobians.

    Returns:
        (v_local, length_local, angle_local)
            v_local: 2-element numpy array in pixel units
            length_local: scalar pixel length
            angle_local: scalar angle in radians (atan2(vy, vx))

    """
    # Central unit pixel vector implied by the sky rate
    v0_unit = J_ref @ rate_hat
    len0_unit = np.linalg.norm(v0_unit)
    if len0_unit == 0:
        scale = 0.0
    else:
        scale = pixel_length / len0_unit

    # Local Jacobian and unit pixel vector
    J_local = local_jacobian(wcs, x, y, dsky=dsky)
    v_local_unit = J_local @ rate_hat
    v_local = v_local_unit * scale

    length_local = float(np.linalg.norm(v_local))
    angle_local = float(np.arctan2(v_local[1], v_local[0]))

    return v_local, length_local, angle_local


def compute_local_streak_width_from_jacobian(
    wcs: WCS,
    J_ref: np.ndarray,
    e_perp: np.ndarray,
    base_fwhm: float,
    x_ref: float,
    y_ref: float,
    x: float,
    y: float,
    dsky: u.Quantity = 1 * u.arcsec,
) -> float:
    """Compute the local effective streak width at (x, y) from WCS Jacobians.

    The width is scaled relative to the reference pixel so that at (x_ref, y_ref)
    it equals base_fwhm, and elsewhere it expands or contracts according to the
    local stretching of a cross-track sky direction.
    """
    J0 = J_ref
    w0_unit = J0 @ e_perp
    len0_unit = np.linalg.norm(w0_unit)
    if len0_unit == 0:
        return float(base_fwhm)

    J_local = local_jacobian(wcs, x, y, dsky=dsky)
    w_local_unit = J_local @ e_perp
    len_local_unit = np.linalg.norm(w_local_unit)
    scale = len_local_unit / len0_unit

    return float(base_fwhm * scale)


def get_local_streak_kernel(
    wcs: WCS,
    streak_metadata: object,
    x: float,
    y: float,
    ref_xy: tuple[float, float] | None = None,
    scale_width: bool = True,
    upsample: int = 100,
    halo_fwhm: float | None = None,
    halo_level: float = 1e-3,
    dsky: u.Quantity = 1 * u.arcsec,
    verbose: bool = False,
) -> np.ndarray:
    """Generate a local streak kernel at (x, y) that follows the WCS distortion.

    Args:
        wcs: Astropy WCS object (with SIP/distortion applied).
        streak_metadata: StreakMetadata-like object with pixel_length, fwhm and
            radian_angle()/sine_angle/cosine_angle attributes.
        x: Pixel x at which the kernel should be evaluated.
        y: Pixel y at which the kernel should be evaluated.
        ref_xy: Optional reference pixel for defining the central streak vector.
            If None, uses the image center from wcs.pixel_shape/array_shape.
        scale_width: Whether to scale the streak width based on cross-track distortion.
        upsample: Sub-pixel sampling factor passed through to rectangle_pyramoid.
        halo_fwhm: Optional halo width passed through to rectangle_pyramoid.
        halo_level: Halo amplitude passed through to rectangle_pyramoid.
        dsky: Sky step used for finite-difference Jacobian evaluation.
        verbose: Whether to log the derived local geometry.

    Returns:
        2D numpy array representing the local streak kernel.

    """
    # Determine reference pixel (roughly image center) if not provided
    if ref_xy is None:
        if getattr(wcs, "pixel_shape", None) is not None:
            width, height = wcs.pixel_shape
            x_ref = (width - 1) / 2.0
            y_ref = (height - 1) / 2.0
        elif getattr(wcs, "array_shape", None) is not None:
            height, width = wcs.array_shape
            x_ref = (width - 1) / 2.0
            y_ref = (height - 1) / 2.0
        else:
            x_ref = 0.0
            y_ref = 0.0
    else:
        x_ref, y_ref = ref_xy

    angle_rad = streak_metadata.radian_angle()
    pixel_length = float(streak_metadata.pixel_length)
    base_width = float(streak_metadata.fwhm)

    # Prepare central Jacobian and sky basis vectors
    J_ref, rate_hat, e_perp = compute_effective_sky_motion_vector(wcs, x_ref, y_ref, angle_rad, dsky=dsky)

    # Local along-track vector and angle
    _, length_local, angle_local = compute_local_streak_vector_from_jacobian(
        wcs,
        J_ref,
        rate_hat,
        pixel_length,
        x_ref,
        y_ref,
        x,
        y,
        dsky=dsky,
    )

    # Local width scaling
    if scale_width:
        width_local = compute_local_streak_width_from_jacobian(
            wcs,
            J_ref,
            e_perp,
            base_fwhm=base_width,
            x_ref=x_ref,
            y_ref=y_ref,
            x=x,
            y=y,
            dsky=dsky,
        )
    else:
        width_local = base_width

    if verbose:
        # info, not debug: the caller passed verbose=True, so this was asked for explicitly
        # and should not then be filtered out by log level.
        logger.info(
            f"Local streak kernel at ({x:.1f}, {y:.1f}): "
            f"length={length_local:.2f} px, width={width_local:.2f} px, angle={np.rad2deg(angle_local):.2f} deg"
        )

    sinx = np.sin(angle_local)
    cosx = np.cos(angle_local)

    kernel = rectangle_pyramoid(
        length=length_local,
        sinx=sinx,
        cosx=cosx,
        width=width_local,
        upsample=upsample,
        pix_shift=None,
        halo_fwhm=halo_fwhm,
        halo_level=halo_level,
        verbose=verbose,
    )

    return kernel
