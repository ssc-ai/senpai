"""Intensity-normalization stretches (zscale, histogram equalization) for display."""

import matplotlib
from astropy.visualization import ZScaleInterval

matplotlib.use("Svg")
import numpy as np


def zscale(data: np.ndarray, contrast: float = 0.2) -> np.ndarray:
    """Apply an astronomical zscale stretch.

    Args:
        data: Input image pixel values.
        contrast: ZScale contrast parameter controlling the stretch strength.

    Returns:
        The zscale-normalized image as a ``float32`` array.

    Notes:
        This function is used for *plotting*. Real-world FITS processing often
        introduces NaNs/Infs (e.g. divide-by-flat, masked pixels). Astropy's
        `ZScaleInterval` doesn't consistently handle non-finite values; if they
        leak through, `imshow` can show spurious dark/bright bands that are not
        present in the original data.

        To avoid plot artifacts, we replace non-finite values with the median of
        the finite pixels before applying zscale.
    """
    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)

    # Replace NaN/Inf with a robust fill so ZScaleInterval can compute limits.
    fill = float(np.median(arr[finite]))
    if not np.all(finite):
        arr = arr.copy()
        arr[~finite] = fill

    norm = ZScaleInterval(contrast=contrast)
    out = norm(arr)

    # Be defensive: ensure output is finite as well.
    out = np.asarray(out, dtype=np.float32)
    out_finite = np.isfinite(out)
    if not np.any(out_finite):
        return np.zeros_like(out)
    if not np.all(out_finite):
        out = out.copy()
        out[~out_finite] = float(np.median(out[out_finite]))

    return out


def histogram_equalization(img_in: np.ndarray, img_dtype: np.dtype = np.uint16) -> np.ndarray:
    """Apply histogram equalization to spread pixel intensities across the range.

    Args:
        img_in: Input image pixel values.
        img_dtype: Integer dtype defining the bit depth of the output histogram.

    Returns:
        The histogram-equalized image as an array of ``img_dtype``.
    """
    cast_img = img_in.astype(img_dtype)

    # Get bit depth from dtype
    bit_depth = 2 ** np.dtype(img_dtype).itemsize * 8

    # segregate color streams
    h_b, _bin_b = np.histogram(cast_img.flatten(), bit_depth, [0, bit_depth - 1])

    # calculate cdf
    cdf_b = np.cumsum(h_b)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * (bit_depth - 1) / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype(img_dtype)

    # Return the equalized image
    return cdf_final_b[cast_img]
