"""Convolution kernels for streak and sidereal point-source detection."""

import functools
import logging

import numpy as np
from scipy.ndimage import shift

logger = logging.getLogger(__name__)

# Upper bound on the supersampled streak-kernel grid (rows*ss * cols*ss elements). A real streak
# kernel is at most image-sized (a few thousand px per side -> well under this); exceeding it means
# the upstream streak-length/rate fit was degenerate (e.g. a wild FWHM on a noisy wide-FOV frame),
# which would otherwise allocate tens-to-hundreds of GiB of float32 and OOM the whole process.
# ~5e8 elements ~= 2 GiB of float32 -- generous for legitimate streaks, fatal only to garbage.
MAX_KERNEL_FINE_ELEMENTS = 500_000_000


def shift_filter_subpx(filter_array: np.ndarray, pix_shift: np.ndarray) -> np.ndarray:
    """Shift a filter kernel by a sub-pixel amount with edge padding.

    Args:
        filter_array (np.ndarray): 2D filter kernel to shift.
        pix_shift (np.ndarray): (row, column) sub-pixel shift to apply.

    Returns:
        np.ndarray: padded and shifted kernel, clipped to the range [0, 1] with
            near-zero values zeroed out.
    """
    pad = (pix_shift + 0.5).round().astype(int)

    padded = np.pad(filter_array, ((pad[0], pad[0]), (pad[1], pad[1])))
    shifted = shift(padded, pix_shift)

    shifted[np.where(np.abs(shifted) < 1e-4)] = 0.000
    shifted[np.where(shifted < 0)] = 0.001
    shifted[np.where(shifted > 1)] = 1

    return shifted


@functools.lru_cache(maxsize=32)
def rectangle_pyramoid(
    length: float,
    sinx: float,
    cosx: float,
    width: int = 4,
    pix_shift: tuple[float, float] | None = None,
    halo_fwhm: float | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Build a rotated rectangular streak kernel for streak detection.

    The kernel is a flat rotated rectangle of the given length and width with angle-aware
    one-pixel soft edges, evaluated directly on the output grid. This is the exact per-pixel
    area coverage of the rotated rectangle in the soft-edge limit, so it reproduces the
    previous supersample-rotate-downsample kernel (correlation >= 0.98 across lengths and
    rotations) while using only output-sized memory and no image-library calls.

    Args:
        length (float): streak length in pixels.
        sinx (float): sine of the streak orientation angle.
        cosx (float): cosine of the streak orientation angle.
        width (int): streak width in pixels. Defaults to 4.
        pix_shift (tuple[float, float] | None): optional (row, column) sub-pixel shift to
            apply to the final kernel. Defaults to None.
        halo_fwhm (float | None): if set, widens the kernel's zero border by roughly half
            this many pixels. Defaults to None.
        verbose (bool): if True, log a progress message. Defaults to False.

    Returns:
        np.ndarray: the rotated streak kernel at pixel resolution.
    """
    if verbose:
        logger.info("rectangle_pyramoid")

    width = max(1, int(width))
    length = max(1, int(length))

    # Size the output grid to the rotated rectangle's bounding box plus a small zero border.
    border = (int(halo_fwhm / 2) if halo_fwhm else 0) + 3
    n_cols = int(np.ceil(abs(length * cosx) + abs(width * sinx))) + 2 * border + 1
    n_rows = int(np.ceil(abs(length * sinx) + abs(width * cosx))) + 2 * border + 1

    # Exact per-pixel area coverage of the rotated rectangle, via supersampling: subdivide each
    # output pixel into ss x ss sub-samples, test each against the rectangle, and average. This
    # converges to the true area coverage (matching the previous 100x-supersample/area-downsample
    # kernel) -- a plain 1-px edge ramp was not faithful enough for the rate->rate streak estimate
    # and degraded registration on some collects (the c92a / object 39741 regression).
    ss = 4

    # Guard: a degenerate streak-length estimate (e.g. a wild rate/FWHM fit on a noisy wide-FOV
    # frame) would size the grid to tens of thousands of px and allocate 100+ GiB of float32,
    # OOM-killing the worker (observed on a wide-field frame: one worker hit 114 GiB RSS). Fail this
    # frame loudly and cheaply instead -- callers treat it as an unsolved frame, so one bad frame
    # can no longer crash the whole run.
    fine_elements = (n_rows * ss) * (n_cols * ss)
    if fine_elements > MAX_KERNEL_FINE_ELEMENTS:
        raise ValueError(
            f"streak kernel too large: {n_rows}x{n_cols} output px "
            f"(length={length}, width={width}, sinx={sinx:.3f}, cosx={cosx:.3f}) would allocate a "
            f"{fine_elements * 4 / 2**30:.1f} GiB supersampled grid "
            f"(> {MAX_KERNEL_FINE_ELEMENTS * 4 / 2**30:.1f} GiB cap); "
            "this indicates a degenerate streak-length estimate, not a real streak."
        )

    center_row, center_col = (n_rows - 1) / 2.0, (n_cols - 1) / 2.0
    offsets = (np.arange(ss, dtype=np.float32) + 0.5) / ss - 0.5  # sub-pixel sample centers
    fine_rows = (np.arange(n_rows, dtype=np.float32)[:, None] + offsets[None, :]).ravel()
    fine_cols = (np.arange(n_cols, dtype=np.float32)[:, None] + offsets[None, :]).ravel()
    d_row = fine_rows - center_row
    d_col = fine_cols - center_col

    # Project the fine samples onto the streak's own axes (along the length, across the width).
    along = d_col[None, :] * cosx + d_row[:, None] * sinx
    across = -d_col[None, :] * sinx + d_row[:, None] * cosx
    inside = ((np.abs(along) <= length / 2.0) & (np.abs(across) <= width / 2.0)).astype(np.float32)

    # Average the ss x ss sub-samples back down to one value per output pixel.
    pyramid = inside.reshape(n_rows, ss, n_cols, ss).mean(axis=(1, 3)).astype(np.float32)

    if pix_shift is not None:
        pyramid = shift_filter_subpx(pyramid, pix_shift)

    return pyramid


@functools.lru_cache(maxsize=32)
def sidereal_kernel(fwhm: float) -> np.ndarray:
    """Generate a 2D Gaussian kernel for sidereal star detection.

    Args:
        fwhm (float): Full width at half maximum of the Gaussian in pixels.

    Returns:
        np.ndarray: 2D Gaussian kernel normalized to sum to 1.
    """
    # Convert FWHM to sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Make kernel size odd and ~6 sigma
    size = int(np.ceil(6 * sigma))
    if size % 2 == 0:
        size += 1

    # Create coordinate grid
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # Generate 2D Gaussian
    kernel = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    # Normalize to sum to 1
    kernel = kernel / kernel.sum()

    return kernel


def streak_matched_kernel(
    fwhm: float, angle_deg: float, length_fwhm: float = 5.0
) -> np.ndarray:
    """Directional matched filter: Gaussian cross-section extruded along an angle.

    Used as part of a filter bank to detect streak-shaped signal in residual
    images (after PSF-model subtraction). The kernel is seeing-limited
    perpendicular to the streak and flat along it, with Gaussian taper at the
    ends to avoid ringing in FFT convolution.

    Args:
        fwhm: PSF full width at half maximum in pixels.
        angle_deg: Streak direction in degrees (0 = along x-axis).
        length_fwhm: Kernel length as a multiple of FWHM.

    Returns:
        2D kernel array normalized to sum to 1.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    length = fwhm * length_fwhm

    # Kernel must encompass the rotated streak + Gaussian wings on all sides
    size = int(np.ceil(length + 6 * sigma))
    if size % 2 == 0:
        size += 1

    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Project pixel coordinates onto streak direction and perpendicular
    along = x * cos_a + y * sin_a
    perp = -x * sin_a + y * cos_a

    # Gaussian profile perpendicular to streak (seeing-limited width)
    cross_section = np.exp(-(perp**2) / (2 * sigma**2))

    # Flat along streak body, Gaussian taper beyond the ends
    half_len = length / 2
    excess = np.maximum(np.abs(along) - half_len, 0)
    along_taper = np.exp(-(excess**2) / (2 * sigma**2))

    kernel = cross_section * along_taper

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def build_directional_filter_bank(
    fwhm: float, n_angles: int = 36, length_fwhm: float = 5.0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build a bank of directional matched filters at evenly spaced angles.

    Each filter is a :func:`streak_matched_kernel` at a different orientation.
    Together they form a filter bank that can detect streak-shaped signal at
    any angle by convolving the image with each filter and comparing responses.

    Args:
        fwhm: PSF FWHM in pixels.
        n_angles: Number of angles to sample in [0, 180) degrees.
        length_fwhm: Each filter's length as a multiple of FWHM.

    Returns:
        Tuple of (list of kernel arrays, array of angles in degrees).
    """
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    # Round for lru_cache friendliness
    fwhm_r = round(float(fwhm), 2)
    length_r = round(float(length_fwhm), 2)
    kernels = [streak_matched_kernel(fwhm_r, float(a), length_r) for a in angles]
    return kernels, angles
