"""Regression tests for the production-stability guards on ``fix/prod-stability-guards``.

This branch hardens senpai v2.6.0 against a set of production failures observed on real
collects. Each guard is pinned here by one focused, mostly-synthetic test so that a
re-introduction (or a revert to ``origin/dev``) is caught immediately:

1. ``rate_rate.solve_rate_from_rate`` -- routes around a non-positive inter-frame gap
   (two rate frames sharing a timestamp) instead of dividing the track rate to infinity.
2. ``rate_rate`` and ``rate_sidereal`` -- coerce a string ``EXPTIME`` header to ``float``
   so the exposure arithmetic no longer raises ``TypeError`` on sensors that write ``'2.0'``.
3. ``wcs_helpers.fit_and_validate_wcs`` -- catches the ``ValueError`` a degenerate matched-star
   geometry raises inside ``fit_wcs_from_points`` and falls back to the provided WCS.
4. ``utils.memory.reclaim_process_memory`` -- new additive helper that trims glibc arenas
   after a run and no-ops safely off glibc (green-only; no meaningful red case).
5. ``kernels.rectangle_pyramoid`` -- rewritten as an exact area-coverage builder with a
   ``MAX_KERNEL_FINE_ELEMENTS`` cap that rejects a degenerate (giant) streak length instead of
   OOM-killing the worker (green-only; the degenerate case must never run against the old
   cv2 kernel, which would attempt a catastrophic allocation).
6. ``masking.analyze_source_shape_fwhm`` -- decomposes the symmetric pixel covariance with
   ``np.linalg.eigh`` instead of ``np.linalg.eig``, so streak geometry stays real-valued on
   numpy >= 2.5 (which no longer downcasts a real ``eig`` result to ``float64``).

Guards 1-3 and 6 have a real red case (a plain revert of the fixed file to ``origin/dev`` makes
the matching test fail); guards 4 and 5 are additive/rewrites and are asserted green-only.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.kernels import MAX_KERNEL_FINE_ELEMENTS, rectangle_pyramoid
from senpai.engine.detection.streak.masking import analyze_source_shape_fwhm
from senpai.engine.detection.streak.rate_rate import solve_rate_from_rate
from senpai.engine.detection.streak.rate_sidereal import solve_rate_from_sidereal
from senpai.engine.models.astrometry import WCSMetadata, WCSModel
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import DetectionMetadata, ImageMetadata
from senpai.engine.models.senpai import FrameShift, RateTrackFrame, SiderealFrame
from senpai.engine.models.starfield import StarField, StarInSpace
from senpai.engine.utils import memory
from senpai.engine.utils.wcs_helpers import fit_and_validate_wcs

_IMG = 400


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialise the process-wide config singleton and disable debug plotting.

    The streak solvers read ``get_config()`` (e.g. ``streak.symmetric_border_removal`` and
    ``plotting.debug``); a shipped YAML gives them a real, frozen config to read.
    """
    initialize_config(CONFIG_DIR / "burr.yaml")
    get_config().plotting.debug = False


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed the global numpy RNG.

    The shift validation path draws random alternative shifts via the global numpy RNG, so
    seeding keeps the synthetic-frame tests deterministic.
    """
    np.random.seed(1234)


# --------------------------------------------------------------------------------------
# Synthetic-frame helpers
# --------------------------------------------------------------------------------------
def _capture_exception(func: Callable[[], object]) -> BaseException | None:
    """Run ``func`` and return whatever exception it raises.

    Used by the ``EXPTIME`` coercion tests, which need to classify the *type* of any failure
    (a ``TypeError`` from the exposure arithmetic is the regression; other failures on the
    minimal synthetic fixture are acceptable).

    Args:
        func: Zero-argument callable to execute.

    Returns:
        The exception instance raised by ``func``, or ``None`` if it returned normally.
    """
    try:
        func()
    except Exception as exc:
        return exc
    return None


def _add_streak(
    img: np.ndarray,
    center_x: float,
    center_y: float,
    length: float,
    angle_deg: float,
    peak: float = 3000.0,
    fwhm: float = 6.0,
) -> None:
    """Add a rotated bar with a Gaussian cross-section (a synthetic satellite trail) in place.

    Args:
        img: Image array to draw into.
        center_x: Streak centre column.
        center_y: Streak centre row.
        length: Streak length in pixels (full extent along its axis).
        angle_deg: Streak orientation in degrees.
        peak: Peak intensity added at the streak core.
        fwhm: Cross-sectional full-width-at-half-maximum in pixels.
    """
    sigma = fwhm / 2.355
    cos_a, sin_a = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(float)
    dx, dy = xx - center_x, yy - center_y
    along = dx * cos_a + dy * sin_a
    across = -dx * sin_a + dy * cos_a
    inside = np.abs(along) <= length / 2.0
    img += peak * inside * np.exp(-0.5 * (across / sigma) ** 2)


def _rate_frame(
    index: int,
    timestamp: datetime,
    exptime: float | str,
    *,
    with_streaks: bool = True,
    seed: int = 0,
) -> RateTrackFrame:
    """Build a minimal rate-track frame, optionally seeded with several identical streaks.

    Args:
        index: Frame index within the run.
        timestamp: Frame timestamp.
        exptime: Value written to the ``EXPTIME`` header (a string exercises the coercion guard).
        with_streaks: If True, draw a cluster of near-identical streaks so the streak
            extractor returns a real measurement and the exposure arithmetic is reached.
        seed: RNG seed for the background noise and streak placement.

    Returns:
        A ``RateTrackFrame`` wrapping the synthetic image.
    """
    rng = np.random.default_rng(seed)
    data = np.full((_IMG, _IMG), 100.0, dtype=np.float32)
    data += rng.normal(0.0, 3.0, data.shape).astype(np.float32)
    if with_streaks:
        for _ in range(8):
            _add_streak(
                data,
                rng.uniform(80, _IMG - 80),
                rng.uniform(80, _IMG - 80),
                length=40.0,
                angle_deg=30.0,
            )
    header = fits.Header()
    header["NAXIS1"] = _IMG
    header["NAXIS2"] = _IMG
    header["EXPTIME"] = exptime
    frame = ProcessedFitsImage(
        data=data,
        header=header,
        data_type=np.dtype("uint16"),
        metadata=ImageMetadata(width=_IMG, height=_IMG),
    )
    return RateTrackFrame(frame=frame, index=index, timestamp=timestamp)


def _catalog_stars(seed: int = 5, n: int = 12) -> list[StarInSpace]:
    """Return a small list of catalog stars spread across the frame.

    The shift-validation path iterates ``starfield.catalog_stars``; a non-empty list lets it
    run to a graceful (invalid) result instead of raising on ``None``.

    Args:
        seed: RNG seed for the star positions.
        n: Number of stars to generate.

    Returns:
        A list of ``StarInSpace`` with pixel positions and magnitudes.
    """
    rng = np.random.default_rng(seed)
    stars = []
    for i in range(n):
        stars.append(
            StarInSpace(
                ra=10.0 + i * 0.01,
                dec=20.0 + i * 0.01,
                magnitude=10.0 + i * 0.1,
                x=float(rng.uniform(60, _IMG - 60)),
                y=float(rng.uniform(60, _IMG - 60)),
            )
        )
    return stars


def _wcs_metadata() -> WCSMetadata:
    """Return a minimal, valid ``WCSMetadata`` for the sidereal anchor frame.

    Returns:
        A ``WCSMetadata`` whose ``x_ifov_arcsec`` the rate-sidereal solver reads.
    """
    return WCSMetadata(
        x_ifov_arcsec=1.0,
        y_ifov_arcsec=1.0,
        x_fov_degrees=0.1,
        y_fov_degrees=0.1,
        RA_center_deg=10.0,
        Dec_center_deg=20.0,
        RA_center_HMS="00:40:00",
        Dec_center_DMS="+20:00:00",
    )


def _solved_sidereal_frame(index: int, timestamp: datetime, exptime: float | str) -> SiderealFrame:
    """Build a sidereal anchor frame carrying a minimal solved starfield.

    Args:
        index: Frame index within the run.
        timestamp: Frame timestamp.
        exptime: Value written to the ``EXPTIME`` header.

    Returns:
        A ``SiderealFrame`` with a starfield that has ``detection_metadata``, ``wcs_metadata``,
        and catalog stars -- everything ``solve_rate_from_sidereal`` reads before failing.
    """
    rng = np.random.default_rng(99)
    data = np.full((_IMG, _IMG), 100.0, dtype=np.float32)
    data += rng.normal(0.0, 3.0, data.shape).astype(np.float32)
    catalog = _catalog_stars()
    for star in catalog:
        _add_streak(data, star.x, star.y, length=6.0, angle_deg=0.0, peak=4000.0, fwhm=3.0)
    header = fits.Header()
    header["NAXIS1"] = _IMG
    header["NAXIS2"] = _IMG
    header["EXPTIME"] = exptime
    header["TRKMODE"] = "sidereal"
    frame = ProcessedFitsImage(
        data=data,
        header=header,
        data_type=np.dtype("uint16"),
        metadata=ImageMetadata(width=_IMG, height=_IMG),
    )
    sidereal = SiderealFrame(frame=frame, index=index, timestamp=timestamp)
    sidereal.starfield = StarField(
        detections=[],
        image_metadata=ImageMetadata(width=_IMG, height=_IMG),
        wcs=None,
        wcs_metadata=_wcs_metadata(),
        detection_metadata=DetectionMetadata(pixel_fwhm=3.0),
        catalog_stars=catalog,
        astrometric_fit_stars=_catalog_stars(),
    )
    return sidereal


# --------------------------------------------------------------------------------------
# Guard 1: rate_rate non-positive inter-frame gap is routed around (div-by-zero guard).
# --------------------------------------------------------------------------------------
def test_solve_rate_from_rate_routes_around_degenerate_timing() -> None:
    """Two rate frames sharing a timestamp are routed around, not fatal.

    A shared timestamp makes the inter-frame gap zero. On ``origin/dev`` the flow reached the
    shift-validation path (and, with no catalog stars, crashed there) or divided the track rate
    to infinity and crashed streak sizing (``int(inf)``). The early guard on this branch marks
    the shift processed-but-invalid and returns before any of that, so one degenerate pair can
    no longer kill the collect.
    """
    same_time = datetime(2024, 1, 1, 0, 0, 0)
    frame_a = _rate_frame(0, same_time, exptime=1.0, with_streaks=False, seed=1)
    frame_b = _rate_frame(1, same_time, exptime=1.0, with_streaks=False, seed=2)
    # The source frame needs a (minimal) solved starfield to pass the upstream-WCS check so we
    # actually reach the gap guard rather than the missing-starfield guard above it.
    frame_a.starfield = StarField(
        detections=[],
        image_metadata=ImageMetadata(width=_IMG, height=_IMG),
        wcs=None,
    )

    frame_shift = FrameShift(source_index=0, target_index=1)
    exc = _capture_exception(lambda: solve_rate_from_rate(frame_a, frame_b, frame_shift))

    assert exc is None, f"degenerate timing must be routed around, not raise: {exc!r}"
    assert frame_shift.is_valid is False
    assert frame_shift.processed is True


# --------------------------------------------------------------------------------------
# Guard 2: string EXPTIME headers are coerced to float in both streak solvers.
# --------------------------------------------------------------------------------------
def test_solve_rate_from_rate_handles_string_exptime() -> None:
    """A string ``EXPTIME`` must not crash the rate-to-rate exposure arithmetic.

    Some sensors write ``EXPTIME`` as ``'2.0'``. On ``origin/dev`` the raw string reached
    ``streak_mapping.length / rate_a_exposure_time`` -> ``TypeError``. The frames are given
    *different* timestamps (a positive gap) so the div-by-zero guard is skipped and the
    exposure arithmetic is genuinely exercised.
    """
    frame_a = _rate_frame(0, datetime(2024, 1, 1, 0, 0, 0), exptime="2.0", seed=1)
    frame_b = _rate_frame(1, datetime(2024, 1, 1, 0, 0, 2), exptime="2.0", seed=2)
    frame_a.starfield = StarField(
        detections=[],
        image_metadata=ImageMetadata(width=_IMG, height=_IMG),
        wcs=None,
        catalog_stars=_catalog_stars(),
        astrometric_fit_stars=_catalog_stars(),
    )

    frame_shift = FrameShift(source_index=0, target_index=1)
    exc = _capture_exception(lambda: solve_rate_from_rate(frame_a, frame_b, frame_shift))

    assert not isinstance(exc, TypeError), (
        f"string EXPTIME must be coerced, not raise TypeError from the exposure arithmetic: {exc!r}"
    )
    if exc is None:
        assert frame_shift.processed is True


def test_solve_rate_from_sidereal_handles_string_exptime() -> None:
    """A string ``EXPTIME`` must not crash the sidereal-to-rate exposure arithmetic.

    On ``origin/dev`` the raw string reached ``frame_extraction.length / rate_exposure_time``
    -> ``TypeError``. With ``EXPTIME`` coerced to ``float`` the solver instead runs to a
    graceful (invalid) result on this minimal fixture.
    """
    sidereal = _solved_sidereal_frame(0, datetime(2024, 1, 1, 0, 0, 0), exptime="2.0")
    rate = _rate_frame(1, datetime(2024, 1, 1, 0, 0, 2), exptime="2.0", seed=3)

    frame_shift = FrameShift(source_index=0, target_index=1)
    exc = _capture_exception(lambda: solve_rate_from_sidereal(sidereal, rate, frame_shift))

    assert not isinstance(exc, TypeError), (
        f"string EXPTIME must be coerced, not raise TypeError from the exposure arithmetic: {exc!r}"
    )
    if exc is None:
        assert frame_shift.processed is True


# --------------------------------------------------------------------------------------
# Guard 3: fit_and_validate_wcs falls back instead of propagating a degenerate-geometry error.
# --------------------------------------------------------------------------------------
def _identity_fallback_wcs() -> WCSModel:
    """Build a minimal but real ``WCSModel`` to hand ``fit_and_validate_wcs`` as the fallback.

    A small real rotation forces astropy to emit all four PC keywords (identity ones are
    omitted, and ``WCSModel.from_astropy_wcs`` would default the missing values to 0 -> a
    singular matrix).

    Returns:
        A ``WCSModel`` over a 200x200 image centred at (10, 20) deg.
    """
    theta = np.deg2rad(0.3)
    astropy_wcs = WCS(naxis=2)
    astropy_wcs.wcs.crpix = [100.0, 100.0]
    astropy_wcs.wcs.crval = [10.0, 20.0]
    astropy_wcs.wcs.cdelt = [-2.0e-4, 2.0e-4]
    astropy_wcs.wcs.pc = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    astropy_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return WCSModel.from_astropy_wcs(astropy_wcs, image_shape=(200, 200))


def test_fit_and_validate_wcs_falls_back_on_degenerate_geometry() -> None:
    """A matched-star geometry collapsed to one pixel falls back to the provided WCS.

    All pixel coordinates collapse to a single point while the sky coordinates are spread, so
    ``fit_wcs_from_points`` raises "Initial guess is outside of provided bounds". On
    ``origin/dev`` that ``ValueError`` escaped and killed the whole collect; the guard catches
    it and returns ``(fallback_wcs, None)`` -- the same result as a rejected validation.
    """
    fallback = _identity_fallback_wcs()
    rng = np.random.default_rng(0)
    world_coords = [(10.0 + rng.uniform(-0.1, 0.1), 20.0 + rng.uniform(-0.1, 0.1)) for _ in range(8)]
    pixel_coords = [(100.0, 100.0) for _ in range(8)]  # collapsed to a single point

    result_wcs, refit_stats = fit_and_validate_wcs(
        world_coords,
        pixel_coords,
        image_shape=(200, 200),
        fallback_wcs=fallback,
        sip_refit_order=2,
        sip_refit_enabled=False,
    )

    assert result_wcs is fallback, "must fall back to the provided WCS, not raise or return a new fit"
    assert refit_stats is None


# --------------------------------------------------------------------------------------
# Guard 4: reclaim_process_memory (additive; green-only).
# --------------------------------------------------------------------------------------
class _RecordingLibc:
    """Fake glibc handle that records ``malloc_trim`` invocations."""

    def __init__(self) -> None:
        """Initialise the empty call log."""
        self.trim_calls: list[int] = []

    def malloc_trim(self, pad: int) -> int:
        """Record a ``malloc_trim`` call and report success.

        Args:
            pad: The trim padding argument (senpai passes 0).

        Returns:
            1, mimicking glibc's "memory was actually released" return value.
        """
        self.trim_calls.append(pad)
        return 1


class _NoTrimLibc:
    """Fake handle without ``malloc_trim`` (models a non-glibc C library)."""

    def __getattr__(self, name: str) -> object:
        """Raise ``AttributeError`` for every symbol, as a missing glibc export would.

        Args:
            name: The attribute being looked up.

        Raises:
            AttributeError: Always -- the symbol is absent.
        """
        raise AttributeError(name)


def test_reclaim_process_memory_runs_without_raising() -> None:
    """The real helper runs to completion on the host (glibc ``malloc_trim`` or a safe no-op)."""
    memory.reclaim_process_memory()


def test_reclaim_process_memory_invokes_malloc_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    """On glibc the helper asks the allocator to trim its arenas via ``malloc_trim(0)``.

    Args:
        monkeypatch: Pytest fixture used to substitute a recording C-library handle.
    """
    fake = _RecordingLibc()
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda *args, **kwargs: fake)

    memory.reclaim_process_memory()

    assert fake.trim_calls == [0]


def test_reclaim_process_memory_noop_without_malloc_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper is a safe no-op when ``malloc_trim`` is unavailable (non-glibc).

    Args:
        monkeypatch: Pytest fixture used to substitute a C-library handle lacking the symbol.
    """
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda *args, **kwargs: _NoTrimLibc())

    memory.reclaim_process_memory()  # must swallow the AttributeError and not raise


# --------------------------------------------------------------------------------------
# Guard 5: rectangle_pyramoid area-coverage correctness + degenerate-size cap (green-only).
#
# NOTE: the cap test must NEVER run against origin/dev's cv2-based kernel -- that kernel would
# attempt a catastrophic allocation and OOM this shared host. It is green-on-this-branch only.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("length", "width", "angle_deg"),
    [(50.0, 4, 30.0), (80.0, 6, 15.0), (120.0, 8, 70.0)],
)
def test_rectangle_pyramoid_sums_to_rectangle_area(length: float, width: int, angle_deg: float) -> None:
    """The kernel's total coverage equals the rotated rectangle's area (length x width).

    The rewritten builder evaluates exact per-pixel area coverage, so summing every pixel must
    recover ``length * width`` to within a few percent regardless of rotation.

    Args:
        length: Streak length in pixels.
        width: Streak width in pixels.
        angle_deg: Streak orientation in degrees.
    """
    rectangle_pyramoid.cache_clear()
    kernel = rectangle_pyramoid(
        length=length,
        sinx=float(np.sin(np.deg2rad(angle_deg))),
        cosx=float(np.cos(np.deg2rad(angle_deg))),
        width=width,
    )

    assert kernel.max() == pytest.approx(1.0, abs=0.02)
    assert float(kernel.sum()) == pytest.approx(length * width, rel=0.05)


def test_rectangle_pyramoid_caps_degenerate_length() -> None:
    """A degenerate (huge) streak-length estimate raises instead of OOM-allocating a giant grid.

    Regression: a wild rate/FWHM fit on a noisy wide-FOV frame sized the supersampled kernel
    grid to 100+ GiB and OOM-killed the worker, breaking the whole process pool. The cap fires
    on the cheap size check, before any large array is allocated.
    """
    rectangle_pyramoid.cache_clear()
    with pytest.raises(ValueError, match="streak kernel too large"):
        rectangle_pyramoid(
            length=40000.0,
            sinx=float(np.sin(np.deg2rad(45.0))),
            cosx=float(np.cos(np.deg2rad(45.0))),
            width=8,
        )

    # A realistic large streak (image-scale, ~5000 px near-diagonal) stays well under the cap
    # and must still build normally -- the guard only rejects garbage, not long real streaks.
    rectangle_pyramoid.cache_clear()
    kernel = rectangle_pyramoid(
        length=5000.0,
        sinx=float(np.sin(np.deg2rad(45.0))),
        cosx=float(np.cos(np.deg2rad(45.0))),
        width=8,
    )
    assert kernel.ndim == 2
    assert kernel.size > 0
    assert (kernel.shape[0] * 4) * (kernel.shape[1] * 4) <= MAX_KERNEL_FINE_ELEMENTS


# --------------------------------------------------------------------------------------
# Guard 6: streak shape analysis stays real-valued under numpy >= 2.5.
# --------------------------------------------------------------------------------------
def test_analyze_source_shape_fwhm_returns_real_geometry() -> None:
    """Streak geometry must come back real, not ``complex128``.

    Regression: ``analyze_source_shape_fwhm`` decomposed the (symmetric) pixel covariance with
    ``np.linalg.eig`` -- the GENERAL solver. numpy < 2.5 downcast that result to ``float64``
    whenever every imaginary part was zero, so the complex branch was invisible. numpy 2.5.0
    dropped the downcast (``np.linalg.eig(np.eye(2))`` is now ``complex128``), so every streak
    length became complex, and ``extraction.extract_streak_dims_mapping``'s
    ``round(length / (length_std * 0.5))`` died with "type numpy.complex128 doesn't define
    __round__" -- taking down the whole collect. Every rate-frame observation on an affected
    sensor failed to solve.

    The covariance is ``[[xx, xy], [xy, yy]]`` -- symmetric by construction, so its eigenvalues
    are real by definition and ``eigh`` is both correct and version-independent. This pins the
    dtype rather than the numpy version: senpai declares ``numpy>=2.2.4`` with no upper bound.
    """
    image = np.zeros((64, 64), dtype=np.float64)
    angle_deg = 20.0
    for t in np.linspace(-15.0, 15.0, 120):
        y = 32.0 + t * np.sin(np.deg2rad(angle_deg))
        x = 32.0 + t * np.cos(np.deg2rad(angle_deg))
        image[round(y) - 1 : round(y) + 2, round(x) - 1 : round(x) + 2] = 100.0

    y_coords, x_coords = np.where(image > 0)
    result = analyze_source_shape_fwhm(image, y_coords, x_coords)

    for key in ("length", "fwhm_major", "fwhm_minor", "orientation"):
        value = result[key]
        assert not np.iscomplexobj(value), (
            f"{key} came back complex ({value!r}) -- the covariance decomposition must use "
            "np.linalg.eigh, not np.linalg.eig, which returns complex128 on numpy >= 2.5"
        )
        # round() is what the caller in extraction.py does; complex128 has no __round__.
        assert round(float(value), 6) == pytest.approx(float(value), abs=1e-6)

    # Sanity: the fix must not move the measurement, only its dtype.
    assert float(result["orientation"]) == pytest.approx(angle_deg, abs=2.0)
    assert float(result["length"]) == pytest.approx(32.0, rel=0.25)
    assert float(result["fwhm_major"]) > float(result["fwhm_minor"])
