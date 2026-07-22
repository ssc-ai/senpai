"""Unit tests for photometric master-flat combination.

Covers the numeric core ``senpai.engine.utils.flats._combine_flat_sources``: each source
frame is normalized by its own median, the per-pixel sigma-clipped median is taken across
frames, and the result is renormalized to a median of 1.0 (a photometric flat).

The tests write small synthetic FITS frames to a temporary directory -- no network,
astrometry, or catalog access. The file-discovery wrapper around the combine core is
covered elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from senpai.engine.utils.flats import _combine_flat_sources, _FlatSource


def _write_flat(directory: Path, name: str, data: np.ndarray) -> _FlatSource:
    """Write ``data`` as a FITS frame and describe it as a flat source.

    Args:
        directory: Directory to write the frame into.
        name: File name for the written frame.
        data: Pixel values for the frame.

    Returns:
        A ``_FlatSource`` pointing at the written file with its median precomputed.
    """
    path = directory / name
    fits.PrimaryHDU(data=data.astype(np.float32)).writeto(path, overwrite=True)
    return _FlatSource(path=path, median=float(np.median(data)))


def _response_map() -> np.ndarray:
    """Build a non-uniform, strictly positive detector response map (a spatial ramp).

    Returns:
        A 16x16 array of strictly positive response values.
    """
    yy, xx = np.mgrid[0:16, 0:16].astype(np.float64)
    return 1000.0 + 5.0 * xx + 3.0 * yy


def test_combine_flat_sources_is_normalized_median_response(tmp_path: Path) -> None:
    """The combined master is the shared response map normalized to a median of 1.0.

    Args:
        tmp_path: Pytest temporary directory for the synthetic flat frames.
    """
    response = _response_map()
    # Frames are the same response map at different sky levels (auto-exposed
    # twilight flats). After per-frame median normalization they are identical,
    # so the combined master is the response map normalized to median 1.0.
    sources = [
        _write_flat(tmp_path, f"flat_{i}.fits", scale * response)
        for i, scale in enumerate((0.8, 1.0, 1.2, 0.9, 1.1))
    ]
    master = _combine_flat_sources(sources, sigma=3.0, maxiters=5)

    expected = response / np.median(response)
    assert master.shape == response.shape
    np.testing.assert_allclose(master, expected, rtol=1e-4, atol=1e-4)
    assert np.median(master) == pytest.approx(1.0, abs=1e-6)


def test_combine_flat_sources_sigma_clips_outlier(tmp_path: Path) -> None:
    """A hot pixel present in a single frame is rejected by the per-pixel sigma clip.

    Args:
        tmp_path: Pytest temporary directory for the synthetic flat frames.
    """
    response = _response_map()
    sources = []
    for i in range(7):
        frame = response.copy()
        if i == 3:
            frame[8, 8] = 1e6  # a hot pixel present in only one frame
        sources.append(_write_flat(tmp_path, f"flatb_{i}.fits", frame))

    master = _combine_flat_sources(sources, sigma=3.0, maxiters=5)

    # The lone outlier is rejected by the sigma clip: that pixel matches the
    # clean per-pixel response rather than being pulled up by the 1e6 spike.
    expected = response / np.median(response)
    assert master[8, 8] == pytest.approx(expected[8, 8], abs=1e-2)
