"""Unit tests for rate-track to sidereal streak correlation.

Covers ``senpai.engine.processing.streak_correlation``: a tracked target moves hundreds
of pixels between a rate collect and a sidereal frame, so its rate-frame RA/Dec is
extrapolated to the sidereal frame's epoch (via a linear RA/Dec-vs-time fit over
clustered rate detections) before matching, and a match resolves the streak's
180-degree direction ambiguity.

The pure helpers (``_angle_diff``, ``_accumulate_shift``) and an end-to-end
``correlate_rate_to_sidereal`` pass are exercised on fully synthetic, deterministic
data -- no network, astrometry, or catalog access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import get_or_initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.streak.sidereal_streak import StreakCandidate
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import (
    CollectionMetadata,
    FrameMetadata,
    ImageMetadata,
)
from senpai.engine.models.senpai import (
    CorrelatedStreak,
    FrameShift,
    RateTrackFrame,
    SenpaiRun,
    SiderealFrame,
)
from senpai.engine.models.starfield import (
    SatelliteInImage,
    SatelliteListImage,
    StarField,
)
from senpai.engine.processing.streak_correlation import (
    _accumulate_shift,
    _angle_diff,
    correlate_rate_to_sidereal,
)


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialize the config singleton; correlation reads its detection tolerances from it."""
    get_or_initialize_config(CONFIG_DIR / "local.yaml")


@pytest.mark.parametrize(
    ("a1", "a2", "expected"),
    [
        (10.0, 20.0, 10.0),
        (0.0, 0.0, 0.0),
        (0.0, 179.0, 1.0),  # wraps: 179 deg ~= -1 deg
        (170.0, 10.0, 20.0),  # wraps around 180
        (90.0, 90.0, 0.0),
        (0.0, 180.0, 0.0),  # 180 identified with 0
    ],
)
def test_angle_diff_wraps_into_0_to_90(a1: float, a2: float, expected: float) -> None:
    """The angle difference folds into [0, 90] degrees and is symmetric in its arguments.

    Args:
        a1: First angle in degrees.
        a2: Second angle in degrees.
        expected: The expected folded absolute difference in degrees.
    """
    assert _angle_diff(a1, a2) == pytest.approx(expected)
    # Difference is symmetric and never exceeds 90 degrees.
    assert _angle_diff(a2, a1) == pytest.approx(expected)
    assert 0.0 <= _angle_diff(a1, a2) <= 90.0


def _run_with_shifts(shifts: list[FrameShift]) -> SenpaiRun:
    """Build a minimal run carrying only the given frame-shift edges.

    Args:
        shifts: The frame-shift edges to attach to the run.

    Returns:
        A ``SenpaiRun`` with no frames and the supplied shifts.
    """
    return SenpaiRun(
        id="shift-run",
        num_frames=0,
        collect_metadata=CollectionMetadata(),
        frame_shifts=shifts,
    )


def test_accumulate_shift_identity() -> None:
    """Accumulating a shift from a frame to itself is the zero shift."""
    run = _run_with_shifts([])
    assert _accumulate_shift(run, 5, 5) == (0.0, 0.0)


def test_accumulate_shift_chains_forward_and_reverse() -> None:
    """Shifts chain additively forward, and the reverse traversal negates the total."""
    run = _run_with_shifts(
        [
            FrameShift(source_index=0, target_index=1, x_shift=10.0, y_shift=5.0, processed=True),
            FrameShift(source_index=1, target_index=2, x_shift=3.0, y_shift=-2.0, processed=True),
        ]
    )
    assert _accumulate_shift(run, 0, 2) == (13.0, 3.0)
    # The reverse traversal negates the accumulated shift.
    assert _accumulate_shift(run, 2, 0) == (-13.0, -3.0)


def test_accumulate_shift_returns_none_without_path() -> None:
    """With no shift path between the two frames, the accumulation returns None."""
    run = _run_with_shifts(
        [FrameShift(source_index=0, target_index=1, x_shift=1.0, y_shift=1.0, processed=True)]
    )
    assert _accumulate_shift(run, 0, 99) is None


def test_accumulate_shift_ignores_unprocessed_or_invalid_shifts() -> None:
    """Unprocessed or invalid shift edges are not traversable, so no path is found."""
    run = _run_with_shifts(
        [
            FrameShift(source_index=0, target_index=1, x_shift=1.0, y_shift=1.0, processed=False),
            FrameShift(
                source_index=0, target_index=1, x_shift=1.0, y_shift=1.0,
                processed=True, is_valid=False,
            ),
        ]
    )
    assert _accumulate_shift(run, 0, 1) is None


def _wcs_model(ra: float = 45.0, dec: float = 30.0, scale: float = 0.0005) -> WCSModel:
    """Build a tangent-plane WCS model centred on a 1024x1024 image.

    Args:
        ra: Reference right ascension in degrees.
        dec: Reference declination in degrees.
        scale: Plate scale in degrees per pixel.

    Returns:
        A populated ``WCSModel``.
    """
    width = height = 1024
    return WCSModel(
        WCSAXES=2, NAXIS1=width, NAXIS2=height,
        CRPIX1=width / 2.0, CRPIX2=height / 2.0,
        PC1_1=-scale, PC1_2=0.0, PC2_1=0.0, PC2_2=scale,
        CDELT1=1.0, CDELT2=1.0, CUNIT1="deg", CUNIT2="deg",
        CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=ra, CRVAL2=dec,
    )


def _tiny_image() -> ProcessedFitsImage:
    """Build a placeholder frame whose header declares a 1024x1024 field.

    Returns:
        A ``ProcessedFitsImage`` with tiny backing data and 1024x1024 metadata.
    """
    header = fits.Header()
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    return ProcessedFitsImage(
        data=np.zeros((4, 4), dtype=np.float32),
        header=header,
        data_type=np.dtype("float32"),
        metadata=ImageMetadata(width=1024, height=1024),
    )


def _starfield_with_wcs() -> StarField:
    """Build a solved starfield carrying the standard synthetic WCS.

    Returns:
        A fitted ``StarField`` with no detections and a populated WCS.
    """
    return StarField(
        detections=[],
        image_metadata=ImageMetadata(width=1024, height=1024),
        wcs=_wcs_model(),
        fit=True,
    )


def test_correlate_rate_to_sidereal_extrapolates_and_confirms() -> None:
    """A tracked target's extrapolated rate-frame track lands on the sidereal streak.

    The rate-frame RA/Dec is fit vs time and extrapolated to the sidereal epoch, matched
    to the sidereal streak, confirming it and resolving its 180-degree direction ambiguity.
    """
    t_ref = datetime(2026, 1, 1, tzinfo=UTC)
    ra0, dec0 = 45.0, 30.0
    v_ra, v_dec = 0.001, 0.0008  # deg/s -- linear sky motion

    # Three rate frames, target held near the same pixel by the mount (so they
    # cluster), with linearly moving sky coordinates measured at t = 0, 5, 10 s.
    rate_frames = []
    for i, dt_s in enumerate((0.0, 5.0, 10.0)):
        det = SatelliteInImage(
            x=500.0 + 0.1 * i, y=500.0 + 0.1 * i,
            ra=ra0 + v_ra * dt_s, dec=dec0 + v_dec * dt_s, snr=10.0,
        )
        rate_frames.append(
            RateTrackFrame(
                frame=_tiny_image(),
                index=i,
                timestamp=t_ref + timedelta(seconds=dt_s),
                detections=SatelliteListImage(
                    detections=[det],
                    image_metadata=ImageMetadata(width=1024, height=1024),
                ),
                starfield=_starfield_with_wcs(),
                frame_metadata=FrameMetadata(),
            )
        )

    # The sidereal frame is 30 s after t_ref. Independently extrapolate the
    # target's sky position to that epoch and project it through the same WCS to
    # get where the streak must be found (this is exactly what the code derives
    # from the linear RA/Dec-vs-time fit).
    sid_dt = 30.0
    ra_pred = ra0 + v_ra * sid_dt
    dec_pred = dec0 + v_dec * sid_dt
    sid_wcs_model = _wcs_model()
    sid_astropy = sid_wcs_model.to_astropy_wcs()
    px = sid_astropy.all_world2pix([[ra_pred, dec_pred]], 0)
    pred_x, pred_y = float(px[0][0]), float(px[0][1])

    # The streak's angle must match the WCS-projected motion direction.
    px_step = sid_astropy.all_world2pix([[ra_pred + v_ra, dec_pred + v_dec]], 0)
    motion_angle = float(
        np.degrees(np.arctan2(px_step[0][1] - pred_y, px_step[0][0] - pred_x))
    )
    streak_angle = motion_angle % 180

    streak = StreakCandidate(
        x=pred_x, y=pred_y, angle_deg=streak_angle,
        length_pixels=20.0, width_pixels=4.0, peak_snr=15.0,
        directional_excess=5.0, fractional_excess=1.0,
        ra=ra_pred, dec=dec_pred,
    )
    sid_frame = SiderealFrame(
        frame=_tiny_image(),
        index=100,
        timestamp=t_ref + timedelta(seconds=sid_dt),
        starfield=StarField(
            detections=[],
            image_metadata=ImageMetadata(width=1024, height=1024),
            wcs=sid_wcs_model,
            fit=True,
        ),
        streak_candidates=[streak],
    )

    # An unconfirmed cross-frame streak at the streak's position, awaiting the
    # rate-frame confirmation + direction resolution.
    correlated = CorrelatedStreak(
        streak_id="seed0001",
        frame_indices=[100],
        positions_x=[pred_x],
        positions_y=[pred_y],
        angle_deg=streak_angle,
        direction_deg=None,
        confirmed=False,
        best_snr=15.0,
    )

    run = SenpaiRun(
        id="corr-run",
        num_frames=4,
        collect_metadata=CollectionMetadata(),
        rate_track_frames=rate_frames,
        sidereal_frames=[sid_frame],
        correlated_streaks=[correlated],
    )

    correlate_rate_to_sidereal(run)

    result = run.correlated_streaks[0]
    assert result.confirmed is True
    # 180-degree ambiguity resolved to a full-circle direction.
    assert result.direction_deg is not None
    assert result.direction_deg == pytest.approx(motion_angle % 360, abs=1e-3)


def test_correlate_rate_to_sidereal_no_rate_detections_is_noop() -> None:
    """With no rate frames to extrapolate from, correlation is a no-op and nothing confirms."""
    correlated = CorrelatedStreak(
        streak_id="seed0002", frame_indices=[0], positions_x=[1.0], positions_y=[1.0],
        angle_deg=10.0, confirmed=False, best_snr=1.0,
    )
    run = SenpaiRun(
        id="empty-run",
        num_frames=1,
        collect_metadata=CollectionMetadata(),
        sidereal_frames=[
            SiderealFrame(
                frame=_tiny_image(), index=0,
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                starfield=_starfield_with_wcs(),
                streak_candidates=[
                    StreakCandidate(
                        x=1.0, y=1.0, angle_deg=10.0, length_pixels=5.0,
                        width_pixels=2.0, peak_snr=5.0,
                        directional_excess=1.0, fractional_excess=1.0,
                    )
                ],
            )
        ],
        correlated_streaks=[correlated],
    )
    correlate_rate_to_sidereal(run)
    assert run.correlated_streaks[0].confirmed is False
