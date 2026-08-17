"""Guards on the PSF scale the Bayesian engine's kernels are sized from.

Both registration paths size a convolution kernel from the sidereal frame's measured PSF
scale. Neither can proceed without one, and both used to dereference it unguarded, so a
frame that reached them without a solved starfield raised AttributeError on None rather than
a diagnosable error. These assert the typed error instead -- matching the degenerate-timing
guard that already sat two lines away in `solve_rate_from_sidereal`.

Pure unit tests: the guards run before any image data is touched, so lightweight stand-ins
for the frames are enough and no astrometry or catalog access is required.
"""

import types
from datetime import datetime, timedelta

import pytest

from senpai.engine.detection.streak.bayesian.rate_sidereal import solve_rate_from_sidereal
from senpai.engine.detection.streak.bayesian.wcs_refinement import refine_sidereal_frame
from senpai.exceptions import WcsPropagationError


def _frame(
    starfield: types.SimpleNamespace | None,
    index: int = 0,
    exptime: float = 1.0,
    seconds: float = 0.0,
) -> types.SimpleNamespace:
    """Build a frame stand-in exposing only what the guards read before failing."""
    return types.SimpleNamespace(
        frame=types.SimpleNamespace(header={"EXPTIME": exptime}, data=None),
        index=index,
        timestamp=datetime(2026, 1, 1) + timedelta(seconds=seconds),
        starfield=starfield,
    )


@pytest.mark.parametrize(
    ("starfield", "expected"),
    [
        (None, "no solved starfield"),
        (types.SimpleNamespace(detection_metadata=None), "no detection metadata"),
    ],
)
def test_solve_rate_from_sidereal_requires_a_psf_scale(
    starfield: types.SimpleNamespace | None, expected: str
) -> None:
    """Registration without a measured PSF scale raises, rather than failing on None."""
    # 10 s apart with 1 s exposures, so the timing guard above it passes and we reach ours.
    sidereal = _frame(starfield, index=0, seconds=10.0)
    rate = _frame(types.SimpleNamespace(detection_metadata=None), index=1, seconds=0.0)
    shift = types.SimpleNamespace(source_index=0, target_index=1)

    with pytest.raises(WcsPropagationError, match=expected):
        solve_rate_from_sidereal(sidereal, rate, shift)


@pytest.mark.parametrize(
    ("starfield", "expected"),
    [
        (None, "no starfield"),
        (types.SimpleNamespace(detection_metadata=None), "no detection metadata"),
    ],
)
def test_refine_sidereal_frame_requires_a_psf_scale(
    starfield: types.SimpleNamespace | None, expected: str
) -> None:
    """Refinement without a measured PSF scale raises before convolving."""
    with pytest.raises(WcsPropagationError, match=expected):
        refine_sidereal_frame(_frame(starfield, index=7))
