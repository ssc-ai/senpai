"""Regression tests for previously-seen production failures.

Covers the streak-kernel performance/geometry contract, the small-gap sidereal->rate
failure fixes, rate->rate seed initialization and multi-peak validation, degenerate-WCS
fallbacks, and pipeline-level error routing. Each test targets one previously-observed
failure so that a re-introduction is caught immediately.

Most tests are fast and fully synthetic. The end-to-end memory-release test additionally
requires a local image-set fixture, an astrometry install, and a star catalog; it is
marked ``slow`` + ``requires_astrometry`` + ``requires_catalog`` and skips unless that
data is present.
"""

import glob
import logging
import random
import resource
import threading
import time
import tracemalloc
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from senpai.core.config import get_config, initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.detection.kernels import rectangle_pyramoid, sidereal_kernel
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.senpai import RateTrackFrame, SiderealFrame
from senpai.engine.processing.collect import process_senpai_collect

# Local image-set fixture for the end-to-end memory test (89 MB; not shipped in-repo).
_MEMORY_TEST_IMAGE_SET = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "image_sets"
    / "sample"
)


@pytest.fixture(scope="module", autouse=True)
def _config() -> None:
    """Initialize the config singleton for the module."""
    initialize_config(CONFIG_DIR / "local.yaml")


def _benchmark_function(
    func_to_test: Callable[..., object],
    time_limit: timedelta | None = None,
    mem_limit_mb: float | None = None,
    *args: object,
    **kwargs: object,
) -> tuple[timedelta, float]:
    """Benchmark a function's peak memory usage and wall-clock duration.

    Args:
        func_to_test: The function under examination.
        time_limit: Maximum tolerated duration; exceeding it raises. Defaults to None.
        mem_limit_mb: Maximum tolerated peak memory in MB; exceeding it raises.
            Defaults to None.
        *args: Positional arguments forwarded to ``func_to_test``.
        **kwargs: Keyword arguments forwarded to ``func_to_test``.

    Raises:
        ValueError: If either the time or memory tolerance is exceeded.

    Returns:
        The duration of the call and its peak memory use in MB.
    """
    # Start time and memory tracking
    start_time = time.monotonic()
    tracemalloc.start()

    # Execute the function under examination
    func_to_test(*args, **kwargs)

    # End the memory and time tracking
    _, f_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    f_duration = timedelta(seconds=time.monotonic() - start_time)

    # Test if the time tolerance was violated
    if time_limit and f_duration > time_limit:
        raise ValueError(
            f"Function executed in {f_duration.total_seconds():.2f} seconds, which is "
            f"above the tolerance of {time_limit.total_seconds():.2f} seconds"
        )

    # Test if the memory tolerance was violated
    f_peak_mb = f_peak_bytes / 10**6
    if mem_limit_mb and f_peak_mb > mem_limit_mb:
        raise ValueError(
            f"Memory peaked at {f_peak_mb:.2f} MB, which is above the tolerance of "
            f"{mem_limit_mb:.2f} MB"
        )

    # Return to the user the time and peak memory
    return f_duration, f_peak_mb


def test_rectangle_pyramoid_performance() -> None:
    """The streak kernel builds within nominal time and memory across random geometries."""
    # How many allocations should we try?
    n_trials = 30

    # Check performance with a nominal FWHM (< 10.0)
    duration_list = []
    memory_list = []
    for _ in range(n_trials):
        # Need some random parameters
        length = random.uniform(75.0, 125.0)
        rotation = random.uniform(0.0, 360.0)
        fwhm = random.uniform(5.0, 10.0)

        f_duration, f_peak_memory_mb = _benchmark_function(
            func_to_test=rectangle_pyramoid,
            time_limit=timedelta(seconds=30.0),  # 30 second limit
            mem_limit_mb=10000,  # 10 GB limit
            length=length,
            sinx=np.sin(np.deg2rad(rotation)),
            cosx=np.cos(np.deg2rad(rotation)),
            width=int(fwhm * 2),
            halo_fwhm=4,
        )
        duration_list.append(f_duration.total_seconds())
        memory_list.append(f_peak_memory_mb)

    assert np.mean(duration_list) < 2.0 * 10.0  # x10 headroom for loaded CI runners
    assert np.mean(memory_list) < 1000.0


def test_saturated_image_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully saturated collect cannot be plate-solved and raises ``SiderealSolveError``.

    The collect is unrecoverable, so with ``error_on_plate_solve_failure`` enabled the
    pipeline raises rather than returning an empty run.

    Args:
        monkeypatch: Pytest fixture used to enable the raise-on-failure flag.
    """
    settings = get_config()
    monkeypatch.setattr(settings.astrometry, "error_on_plate_solve_failure", True)

    height, width = 256, 256
    saturated_data = np.full((height, width), np.iinfo(np.uint16).max, dtype=np.uint16)
    file_list = []
    for i in range(5):
        header = fits.Header()
        header["NAXIS1"] = width
        header["NAXIS2"] = height
        header["DATE-OBS"] = f"2024-01-01T00:00:{i:02d}.000"
        header["EXPTIME"] = 1.0
        header["TRKMODE"] = "sidereal"
        file_list.append(
            ProcessedFitsImage(
                data=saturated_data.copy(),
                header=header,
                data_type=saturated_data.dtype,
                metadata=ImageMetadata(image_id=f"saturated_{i}", width=width, height=height),
            )
        )

    from senpai.exceptions import SiderealSolveError

    with pytest.raises(SiderealSolveError):
        process_senpai_collect(file_list)


@pytest.mark.slow
@pytest.mark.requires_astrometry
@pytest.mark.requires_catalog
def test_detection_releases_memory_after_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running one image set end-to-end returns its large transient memory to the OS.

    Requires a local image-set fixture at ``tests/data/image_sets/sample`` plus a local
    astrometry install and star catalog.

    Args:
        monkeypatch: Pytest fixture used to relax the plate-solve failure flag.
    """
    if not _MEMORY_TEST_IMAGE_SET.exists():
        pytest.skip(f"local image-set fixture not available at {_MEMORY_TEST_IMAGE_SET}")

    settings = get_config()
    monkeypatch.setattr(settings.astrometry, "error_on_plate_solve_failure", False)

    image_set = _MEMORY_TEST_IMAGE_SET
    file_list = [
        ProcessedFitsImage.from_file_bytes((image_set / f).read_bytes())
        for f in sorted(glob.glob("*.fits", root_dir=image_set))
    ]

    # Clear kernel caches so this run actually rebuilds (and frees) its big transient arrays.
    rectangle_pyramoid.cache_clear()
    sidereal_kernel.cache_clear()

    def rss_mb() -> float:
        """Return the process resident set size in MB."""
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * resource.getpagesize() / 1024 / 1024

    before = rss_mb()
    peak = before
    stop = threading.Event()

    def watch() -> None:
        """Track the peak resident set size until signalled to stop."""
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, rss_mb())
            time.sleep(0.05)

    t = threading.Thread(target=watch)
    t.start()
    process_senpai_collect(file_list)  # the real path; the fix reclaims inside here
    stop.set()
    t.join()
    after = rss_mb()

    transient = peak - before
    retained = after - before
    assert transient > 100, f"no measurable transient ({transient:.0f} MB); test ineffective"
    assert retained < 0.4 * transient, (
        f"freed memory not returned: peaked +{transient:.0f} MB, "
        f"still holding +{retained:.0f} MB after run"
    )


# --------------------------------------------------------------------------------------
# Regression tests for the small-gap sidereal->rate failures (SENPAI timed out / crashed
# on sim collects). Each targets one fix; all are fast and synthetic.
# --------------------------------------------------------------------------------------


def _make_rate_track_frame(
    index: int,
    data: np.ndarray,
    timestamp: datetime | None = None,
    exptime: float = 1.0,
) -> RateTrackFrame:
    """Build a minimal rate-track frame from synthetic image data.

    Args:
        index: Frame index within the run.
        data: Backing image array.
        timestamp: Frame timestamp; defaults to a 1 s cadence from a fixed epoch.
        exptime: Exposure time in seconds.

    Returns:
        A ``RateTrackFrame`` wrapping the data.
    """
    header = fits.Header()
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    header["EXPTIME"] = exptime
    frame = ProcessedFitsImage(
        data=data,
        header=header,
        data_type=data.dtype,
        metadata=ImageMetadata(width=data.shape[1], height=data.shape[0]),
    )
    if timestamp is None:
        timestamp = datetime(2024, 1, 1, 0, 0, index)
    return RateTrackFrame(frame=frame, index=index, timestamp=timestamp)


def _make_sidereal_frame(
    index: int,
    data: np.ndarray,
    timestamp: datetime | None = None,
    exptime: float = 1.0,
) -> SiderealFrame:
    """Build a minimal sidereal frame from synthetic image data (no starfield solved yet).

    Args:
        index: Frame index within the run.
        data: Backing image array.
        timestamp: Frame timestamp; defaults to a 1 s cadence from a fixed epoch.
        exptime: Exposure time in seconds.

    Returns:
        A ``SiderealFrame`` wrapping the data.
    """
    header = fits.Header()
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    header["EXPTIME"] = exptime
    header["TRKMODE"] = "sidereal"
    frame = ProcessedFitsImage(
        data=data,
        header=header,
        data_type=data.dtype,
        metadata=ImageMetadata(width=data.shape[1], height=data.shape[0]),
    )
    if timestamp is None:
        timestamp = datetime(2024, 1, 1, 0, 0, index)
    return SiderealFrame(frame=frame, index=index, timestamp=timestamp)


def _rotated_rect_reference(
    shape: tuple[int, int], length: float, sinx: float, cosx: float, width: float
) -> np.ndarray:
    """Compute an independent angle-aware rotated-rectangle coverage map on a centered grid.

    Args:
        shape: Output array shape (height, width).
        length: Rectangle length in pixels.
        sinx: Sine of the rotation angle.
        cosx: Cosine of the rotation angle.
        width: Rectangle width in pixels.

    Returns:
        A coverage map in [0, 1] over the grid.
    """
    height, wid = shape
    cy, cx = (height - 1) / 2.0, (wid - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:wid].astype(float)
    dx, dy = xx - cx, yy - cy
    u = dx * cosx + dy * sinx
    v = -dx * sinx + dy * cosx
    r = abs(cosx) + abs(sinx)
    return np.clip((length / 2.0 - np.abs(u)) / r + 0.5, 0.0, 1.0) * np.clip(
        (width / 2.0 - np.abs(v)) / r + 0.5, 0.0, 1.0
    )


def test_rate_rate_peak_search_handles_degenerate_seed() -> None:
    """The rate->rate correlation-peak search does not crash on a degenerate/negative seed.

    A negative expected shift previously produced an empty search window, raising
    "argmax of an empty sequence".
    """
    from senpai.engine.detection.streak.rate_rate import windowed_correlation_peaks

    rng = np.random.default_rng(0)
    cc = rng.random((201, 201)).astype(np.float32)
    for expected_shift in (None, -50.0, 0.0, 5.0):
        peak = windowed_correlation_peaks(cc, expected_shift, n_peaks=1)[0]
        peak = np.asarray(peak)
        assert peak.shape == (2,)
        assert 0 <= peak[0] < cc.shape[0] and 0 <= peak[1] < cc.shape[1]


def test_fit_wcs_from_points_raises_on_degenerate_pixels() -> None:
    """Document the prod root cause: degenerate pixel geometry violates fit_wcs bounds.

    ``fit_wcs_from_points`` raises "Initial guess is outside of provided bounds" when the
    matched pixel coordinates collapse to ~one point -- the exact ValueError that escaped
    the sidereal WCS-refine path in production.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points

    rng = np.random.default_rng(0)
    # All detections at the same pixel, but spread on the sky -> unsolvable, bounds violated.
    x = np.full(8, 100.0)
    y = np.full(8, 100.0)
    sky = SkyCoord(rng.uniform(10, 11, 8), rng.uniform(20, 21, 8), unit=u.deg)

    with pytest.raises(ValueError, match="Initial guess is outside of provided bounds"):
        fit_wcs_from_points((x, y), sky, proj_point="center")


def test_sidereal_refine_falls_back_when_fit_wcs_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sidereal WCS refine whose fit raises falls back to the global-shift WCS and warns.

    Degenerate geometry must not propagate the ValueError and fail the whole collect. The
    rate-frame sibling already caught this; the sidereal path previously did not (the
    uncaught "Initial guess is outside of provided bounds" tracebacks seen in production).

    Args:
        monkeypatch: Pytest fixture used to stub the heavy prerequisites and the failing fit.
        caplog: Pytest log-capture fixture.
    """
    from types import SimpleNamespace

    from astropy.wcs import WCS

    import senpai.engine.utils.propagate_wcs as pw
    from senpai.engine.models.astrometry import WCSModel
    from senpai.engine.models.metadata import DetectionMetadata
    from senpai.engine.models.starfield import StarField, StarInSpace

    # A minimal but real WCS so shift_wcs / WCSMetadata work on the real objects. A small real
    # rotation forces astropy to emit all four PC keywords (identity ones are omitted, and
    # WCSModel.from_astropy_wcs defaults missing PC values to 0 -> a singular matrix).
    theta = np.deg2rad(0.3)
    astropy_wcs = WCS(naxis=2)
    astropy_wcs.wcs.crpix = [60.0, 60.0]
    astropy_wcs.wcs.crval = [10.0, 20.0]
    astropy_wcs.wcs.cdelt = [-2.0e-4, 2.0e-4]
    astropy_wcs.wcs.pc = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    astropy_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_model = WCSModel.from_astropy_wcs(astropy_wcs, image_shape=(120, 120))

    frame = _make_sidereal_frame(0, (np.ones((120, 120)) * 100).astype(np.uint16))
    frame.starfield = StarField(
        astrometric_fit_stars=[],
        detections=[],
        image_metadata=ImageMetadata(width=120, height=120),
        wcs=wcs_model,
        detection_metadata=DetectionMetadata(pixel_fwhm=2.0),
    )

    # Two well-separated catalog stars so >=1 survives the SNR/separation filter and we
    # reach the fit.
    stars = [
        StarInSpace(ra=10.0, dec=20.0, magnitude=10.0, x=30.0, y=30.0),
        StarInSpace(ra=10.01, dec=20.01, magnitude=11.0, x=90.0, y=90.0),
    ]

    # Stub the heavy prerequisites (catalog query, photometry) so the test stays fast and
    # focused on the error-handling seam; make fit_wcs_from_points raise the real prod
    # ValueError.
    monkeypatch.setattr(pw, "get_global_shift_from_astrometric_stars", lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr(pw, "catalog_stars_from_wcs", lambda *a, **k: SimpleNamespace(stars=stars))
    monkeypatch.setattr(
        pw,
        "calculate_star_snrs_with_aperture_photometry",
        lambda _frame, cat: [(s, 20.0, 100.0) for s in cat],
    )
    monkeypatch.setattr(pw, "estimate_limiting_magnitude_from_photometry", lambda *a, **k: None)
    monkeypatch.setattr(
        pw, "aperture_photometry", lambda *a, **k: {"aperture_sum": np.array([100.0])}
    )

    def _raise_bounds(*_a: object, **_k: object) -> None:
        raise ValueError("Initial guess is outside of provided bounds")

    monkeypatch.setattr(pw, "fit_wcs_from_points", _raise_bounds)

    convolved = frame.frame.data.astype(float)
    with caplog.at_level(logging.WARNING):
        result = pw.refine_sidereal_with_catalog_stars(frame, convolved)  # must not raise

    assert isinstance(result, WCSModel), "must fall back to a usable WCS, not None/raise"
    assert result is frame.starfield.wcs, "fallback must be the global-shift WCS"
    assert any("fit_wcs_from_points" in r.getMessage() for r in caplog.records), (
        "the recoverable refine failure must be logged as a warning"
    )


def test_rate_rate_window_excludes_spurious_far_peak() -> None:
    """The rate->rate search window stays tight so a spurious far peak cannot win.

    The window is 1.2x the expected shift; the caller takes the brightest peak in it and
    only retries 3x against the corr>0.9 gate, so a wider window lets bright spurious peaks
    exhaust the retries and break registration (a 2x margin dropped the 28190/26360 MDP
    collects to zero). A faint true peak inside the window must beat a brighter peak outside.
    """
    from senpai.engine.detection.streak.rate_rate import windowed_correlation_peaks

    expected_shift = 50.0  # window = 1.2 * 50 = 60 px; central self-corr mask = 0.2 * 50 = 10 px
    rng = np.random.default_rng(0)
    cc = (rng.random((201, 201)).astype(np.float32)) * 0.01  # faint background
    # True peak 40 px from the (100, 100) center: inside the 60 px window, outside the mask.
    true_row, true_col = 140, 100
    cc[true_row, true_col] = 50.0
    # Brighter spurious peak 85 px out: outside the 60 px window, must be ignored.
    cc[100, 185] = 100.0

    peak = np.asarray(windowed_correlation_peaks(cc, expected_shift=expected_shift, n_peaks=1)[0])

    assert abs(int(peak[0]) - true_row) <= 2 and abs(int(peak[1]) - true_col) <= 2, (
        f"window admitted a spurious far peak: expected ~({true_row},{true_col}), got {tuple(peak)}"
    )


def test_aperture_photometry_uses_subpixel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rate-frame star-SNR photometry uses method='subpixel' with subpixels>=5.

    That reproduces the 'exact' method's catalog-star count at the min_snr gate.

    Args:
        monkeypatch: Pytest fixture used to capture the photometry call kwargs.
    """
    import senpai.engine.utils.propagate_wcs as pw
    from senpai.engine.models.metadata import StreakMetadata
    from senpai.engine.models.starfield import StarInSpace

    calls = []

    def fake_aperture_photometry(data: np.ndarray, aperture: object, **kwargs: object) -> dict:
        """Record the photometry kwargs and return unit flux per position."""
        calls.append(kwargs)
        n = len(np.atleast_2d(aperture.positions))
        return {"aperture_sum": np.ones(n)}

    monkeypatch.setattr(pw, "aperture_photometry", fake_aperture_photometry)

    frame = _make_rate_track_frame(0, (np.ones((120, 120)) * 100).astype(np.uint16))
    frame.streak = StreakMetadata(
        pixel_length=20.0, sine_angle=np.sin(0.5), cosine_angle=np.cos(0.5), fwhm=4.0
    )
    stars = [StarInSpace(ra=0.0, dec=0.0, magnitude=10.0, x=60.0, y=60.0) for _ in range(12)]

    pw.calculate_star_snrs_with_aperture_photometry(frame, stars)

    assert calls, "aperture_photometry was not called"
    assert all(c.get("method") == "subpixel" and c.get("subpixels", 0) >= 5 for c in calls), (
        f"rate-frame aperture photometry must use method='subpixel', subpixels>=5: {calls}"
    )


def test_rectangle_pyramoid_matches_reference_and_is_cheap() -> None:
    """The streak kernel builds cheaply while still matching the requested rotated rectangle."""
    rectangle_pyramoid.cache_clear()
    length, rotation, fwhm = 200.0, 35.0, 6.0
    sinx, cosx = np.sin(np.deg2rad(rotation)), np.cos(np.deg2rad(rotation))

    start = time.monotonic()
    kernel = rectangle_pyramoid(
        length=length,
        sinx=sinx,
        cosx=cosx,
        width=int(fwhm * 2),
        halo_fwhm=4,
    )
    duration = time.monotonic() - start

    assert duration < 0.5, f"kernel build too slow ({duration:.2f}s); expected a lightweight builder"
    assert kernel.max() == pytest.approx(1.0, abs=0.05)
    ref = _rotated_rect_reference(kernel.shape, length, sinx, cosx, int(fwhm * 2))
    corr = float(np.corrcoef(kernel.ravel(), ref.ravel())[0, 1])
    assert corr > 0.9, (
        f"kernel does not match a rotated rectangle of the requested geometry (corr={corr:.3f})"
    )


def test_rectangle_pyramoid_guards_degenerate_length() -> None:
    """A degenerate (huge) streak-length estimate raises instead of OOM-allocating a giant kernel.

    Regression: on a wide-FOV frame a wild FWHM/rate fit produced an enormous streak
    length, sizing the supersampled kernel grid to ~114 GiB and OOM-killing the worker, which
    broke the whole process pool. The guard fires on the size check, before any large array is
    allocated, so the caller can treat it as a single unsolved frame instead of crashing.
    """
    rectangle_pyramoid.cache_clear()
    # ~40000 px length at 45 deg -> ~28000x28000 output -> a >10 TiB supersampled grid.
    with pytest.raises(ValueError, match="streak kernel too large"):
        rectangle_pyramoid(
            length=40000.0,
            sinx=float(np.sin(np.deg2rad(45.0))),
            cosx=float(np.cos(np.deg2rad(45.0))),
            width=8,
        )

    # A realistic large streak (image-scale, ~5000 px near-diagonal) stays well under the cap and
    # must still build normally -- the guard only rejects garbage, not long real streaks.
    rectangle_pyramoid.cache_clear()
    kernel = rectangle_pyramoid(
        length=5000.0,
        sinx=float(np.sin(np.deg2rad(45.0))),
        cosx=float(np.cos(np.deg2rad(45.0))),
        width=8,
    )
    assert kernel.ndim == 2
    assert kernel.size > 0


def test_rate_rate_handles_unsolved_source_frame(caplog: pytest.LogCaptureFixture) -> None:
    """A rate->rate shift whose source frame was never solved is recoverable, not fatal.

    The chain routes around it: the solve logs a warning and marks the shift invalid --
    it does not raise and does not store an error string on the model.

    Args:
        caplog: Pytest log-capture fixture.
    """
    from senpai.engine.detection.streak.rate_rate import solve_rate_from_rate
    from senpai.engine.models.senpai import FrameShift

    rng = np.random.default_rng(0)
    source = _make_rate_track_frame(5, rng.integers(100, 500, (120, 120), dtype=np.uint16))
    target = _make_rate_track_frame(4, rng.integers(100, 500, (120, 120), dtype=np.uint16))
    assert source.starfield is None  # unsolved upstream
    fs = FrameShift(source_index=5, target_index=4)

    with caplog.at_level(logging.WARNING):
        solve_rate_from_rate(source, target, fs)  # must not raise

    assert fs.is_valid is False
    assert fs.processed is True
    assert any(record.levelno >= logging.WARNING for record in caplog.records), (
        "a recoverable rate->rate failure must be logged as a warning"
    )


def test_solve_rate_from_rate_routes_around_degenerate_timing() -> None:
    """Two rate frames sharing a timestamp are routed around, not fatal.

    A shared timestamp makes the exposure-midpoint elapsed time zero, which previously divided
    the pixel track rate to infinity and crashed streak sizing (``int(inf)`` -> OverflowError).
    The pair must be marked invalid and processed, not raise.
    """
    from senpai.engine.detection.streak.rate_rate import solve_rate_from_rate
    from senpai.engine.models.senpai import FrameShift
    from senpai.engine.models.starfield import StarField

    data = (np.ones((64, 64)) * 100).astype(np.uint16)
    same_time = datetime(2024, 1, 1, 0, 0, 0)
    frame_a = _make_rate_track_frame(0, data, timestamp=same_time)
    frame_b = _make_rate_track_frame(1, data, timestamp=same_time)
    # The source frame needs a (minimal) solved starfield to pass the upstream-WCS check.
    frame_a.starfield = StarField(
        detections=[],
        image_metadata=ImageMetadata(width=64, height=64),
        wcs=None,
    )

    fs = FrameShift(source_index=0, target_index=1)
    solve_rate_from_rate(frame_a, frame_b, fs)  # must not raise

    assert fs.is_valid is False
    assert fs.processed is True


def test_sidereal_to_rate_degenerate_timing_raises() -> None:
    """A degenerate sidereal->rate timing (overlapping exposures) fails fast.

    The sidereal->rate shift is the anchor that carries the solved WCS into the collect.
    When ``gap + 0.5*rate_exp <= 0`` it cannot be measured and nothing downstream can
    recover, so the solver raises ``WcsPropagationError`` rather than warn-and-return.
    """
    from senpai.engine.detection.streak.rate_sidereal import solve_rate_from_sidereal
    from senpai.engine.models.senpai import FrameShift
    from senpai.exceptions import WcsPropagationError

    data = np.zeros((64, 64), dtype=np.uint16)
    # |dt| = 1.26 s, EXPTIME = 3 s -> inter_frame_gap = -1.74 s -> gap + 0.5*exp = -0.24 s <= 0
    sidereal = _make_sidereal_frame(
        6, data, timestamp=datetime(2024, 1, 1, 0, 0, 1, 260000), exptime=3.0
    )
    rate = _make_rate_track_frame(5, data, timestamp=datetime(2024, 1, 1, 0, 0, 0), exptime=3.0)
    fs = FrameShift(source_index=6, target_index=5)

    with pytest.raises(WcsPropagationError):
        solve_rate_from_sidereal(sidereal, rate, fs)


def test_solve_rate_from_sidereal_handles_string_exptime() -> None:
    """A string EXPTIME (some sensors write ``'2.0'``) must not crash the sidereal->rate anchor.

    The exposure arithmetic previously did ``0.5 * ('2.0' + '2.0')`` -> ``TypeError``. With
    EXPTIME coerced to float, overlapping frame timing instead surfaces the meaningful
    ``WcsPropagationError``.
    """
    from senpai.engine.detection.streak.rate_sidereal import solve_rate_from_sidereal
    from senpai.engine.models.senpai import FrameShift
    from senpai.exceptions import WcsPropagationError

    data = (np.ones((64, 64)) * 100).astype(np.uint16)
    same_time = datetime(2024, 1, 1, 0, 0, 0)
    sidereal = _make_sidereal_frame(0, data, timestamp=same_time)
    rate = _make_rate_track_frame(1, data, timestamp=same_time)
    sidereal.frame.header["EXPTIME"] = "2.0"  # string, as some sensors write it
    rate.frame.header["EXPTIME"] = "2.0"

    fs = FrameShift(source_index=0, target_index=1)
    # Coercion must succeed (no TypeError); the overlapping timing then fails fast and typed.
    with pytest.raises(WcsPropagationError):
        solve_rate_from_sidereal(sidereal, rate, fs)


def test_streak_parameters_from_xcorr_returns_none_when_no_cluster() -> None:
    """A lone spike on a flat cross-correlation yields no measurable streak (returns None).

    The None is a routable sentinel the anchor solver turns into a ``WcsPropagationError``,
    rather than a bare ValueError that would be caught as control flow.
    """
    from senpai.engine.detection.streak.extraction import streak_parameters_from_xcorr

    image = np.zeros((201, 201), dtype=float)
    image[40, 40] = 100.0  # single bright pixel, no extended cluster

    result = streak_parameters_from_xcorr(image, plate_scale_arcsec=None, seeing_fwhm_pixels=2.0)
    assert result is None


def test_collect_backstop_raises_when_no_rate_frame_registered() -> None:
    """A collect that registers zero rate-track frames to a WCS raises ``WcsPropagationError``.

    This is the belt-and-suspenders backstop for cascades: an empty result is not returned
    silently -- a meaningful error is raised instead.
    """
    from senpai.engine.models.metadata import CollectionMetadata
    from senpai.engine.models.senpai import SenpaiRun
    from senpai.engine.processing.collect import require_registered_rate_frames
    from senpai.exceptions import WcsPropagationError

    data = np.zeros((64, 64), dtype=np.uint16)
    run = SenpaiRun(
        id="run_",
        num_frames=2,
        collect_metadata=CollectionMetadata(),
        rate_track_frames=[_make_rate_track_frame(1, data), _make_rate_track_frame(0, data)],
    )
    assert all(f.starfield is None for f in run.rate_track_frames)  # none registered

    with pytest.raises(WcsPropagationError):
        require_registered_rate_frames(run)


def test_from_file_bytes_raises_invalid_input_on_garbage() -> None:
    """Malformed (non-FITS) submitted bytes raise a typed ``InvalidInputError`` (API 422).

    An untyped astropy error would otherwise map to a 500.
    """
    from senpai.exceptions import InvalidInputError

    with pytest.raises(InvalidInputError):
        ProcessedFitsImage.from_file_bytes(b"this is definitely not a FITS file")


# --------------------------------------------------------------------------------------
# Rate->rate seed initialization. The first rate->rate pair has no measured rate yet; rather
# than back it out of the sidereal->rate registration offset (a star shift across a slew-mode
# change, which under-estimates the object rate), seed it from the rate frame's mount
# track-rate header, falling back to the measured streak length. Subsequent pairs re-measure
# their own rate.
# --------------------------------------------------------------------------------------


def _solved_rate_frame(
    index: int,
    data: np.ndarray,
    timestamp: datetime,
    exptime: float = 1.0,
    pixel_length: float = 30.0,
) -> RateTrackFrame:
    """Build a rate frame with a minimal real solved starfield (WCS + streak).

    Args:
        index: Frame index within the run.
        data: Backing image array.
        timestamp: Frame timestamp.
        exptime: Exposure time in seconds.
        pixel_length: Measured streak length in pixels.

    Returns:
        A ``RateTrackFrame`` with a solved starfield and a streak measurement.
    """
    from astropy.wcs import WCS

    from senpai.engine.models.astrometry import WCSModel
    from senpai.engine.models.metadata import DetectionMetadata, StreakMetadata
    from senpai.engine.models.starfield import StarField

    theta = np.deg2rad(0.3)
    w = WCS(naxis=2)
    w.wcs.crpix = [60.0, 60.0]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.cdelt = [-2.0e-4, 2.0e-4]
    w.wcs.pc = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_model = WCSModel.from_astropy_wcs(w, image_shape=tuple(data.shape))

    frame = _make_rate_track_frame(index, data, timestamp=timestamp, exptime=exptime)
    frame.starfield = StarField(
        astrometric_fit_stars=[],
        detections=[],
        image_metadata=ImageMetadata(width=data.shape[1], height=data.shape[0]),
        wcs=wcs_model,
        detection_metadata=DetectionMetadata(pixel_fwhm=2.0),
        catalog_stars=[],
    )
    frame.streak = StreakMetadata(
        pixel_length=pixel_length, sine_angle=0.0, cosine_angle=1.0, fwhm=4.0
    )
    return frame


def test_track_rate_from_header() -> None:
    """The mount track-rate header yields a pixel rate: great-circle arcsec/s over plate scale.

    TELTKRA is an RA rate, so it is scaled by cos(dec); the result is None when either header
    is absent or the plate scale is non-positive (so the caller can cleanly fall back).
    """
    from senpai.engine.detection.streak.rate_rate import track_rate_from_header

    gc = np.hypot(26.44 * np.cos(np.deg2rad(31.93)), 28.51)  # arcsec/s
    rate = track_rate_from_header({"TELTKRA": 26.44, "TELTKDEC": -28.51}, 0.9, 31.93)
    assert rate == pytest.approx(gc / 0.9, rel=1e-6)

    assert track_rate_from_header({"TELTKRA": 26.44}, 0.9, 31.93) is None  # missing TELTKDEC
    assert track_rate_from_header({}, 0.9, 31.93) is None
    assert track_rate_from_header({"TELTKRA": 26.44, "TELTKDEC": -28.51}, 0.0, 31.93) is None
    assert (
        track_rate_from_header({"TELTKRA": 0.0, "TELTKDEC": 0.0}, 0.9, 31.93) is None
    )  # non-positive


def test_initial_track_rate_prefers_header_then_streak() -> None:
    """The initial-rate helper prefers the header rate, then streak length, else None."""
    from senpai.engine.detection.streak.rate_rate import (
        _initial_track_rate,
        track_rate_from_header,
    )

    data = np.zeros((120, 120), dtype=np.uint16)
    frame = _solved_rate_frame(
        5, data, timestamp=datetime(2024, 1, 1, 0, 0, 5), exptime=2.0, pixel_length=30.0
    )
    frame.frame.header["TELTKRA"] = 26.44
    frame.frame.header["TELTKDEC"] = -28.51

    plate = frame.starfield.wcs_metadata.x_ifov_arcsec
    dec = float(frame.starfield.wcs.to_astropy_wcs().wcs.crval[1])
    expected_header = track_rate_from_header(frame.frame.header, plate, dec)
    assert _initial_track_rate(frame) == pytest.approx(expected_header, rel=1e-6)

    # No header -> streak length / exposure (30 px / 2 s = 15 px/s).
    del frame.frame.header["TELTKRA"]
    del frame.frame.header["TELTKDEC"]
    assert _initial_track_rate(frame) == pytest.approx(30.0 / 2.0, rel=1e-6)

    # Neither header nor streak -> None.
    frame.streak = None
    assert _initial_track_rate(frame) is None


def test_solve_rate_from_rate_seeds_window_from_track_rate_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first (unmeasured) rate->rate pair seeds its window from the header track rate.

    It is not left unseeded nor derived from a sidereal offset.

    Args:
        monkeypatch: Pytest fixture used to capture the expected shift and stub the optimizer.
    """
    import senpai.engine.detection.streak.rate_rate as rr
    from senpai.engine.models.senpai import FrameShift

    rng = np.random.default_rng(0)
    a = _solved_rate_frame(
        5,
        rng.integers(100, 500, (120, 120), dtype=np.uint16),
        timestamp=datetime(2024, 1, 1, 0, 0, 5),
        exptime=1.0,
    )
    a.frame.header["TELTKRA"] = 26.44
    a.frame.header["TELTKDEC"] = -28.51
    b = _solved_rate_frame(
        4,
        rng.integers(100, 500, (120, 120), dtype=np.uint16),
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        exptime=1.0,
    )
    b.starfield = None  # target is not yet solved; it's what we register
    assert a.pixel_track_rate_per_second is None and b.pixel_track_rate_per_second is None

    captured = {}

    def cap_peaks(cc: np.ndarray, expected_shift: float | None, n_peaks: int = 1) -> list[np.ndarray]:
        """Capture the expected shift and return a benign centre peak."""
        captured["expected_shift"] = expected_shift
        return [np.array(cc.shape) / 2.0] * max(1, n_peaks)  # center -> zero shift, ends benignly

    monkeypatch.setattr(rr, "windowed_correlation_peaks", cap_peaks)
    monkeypatch.setattr(rr, "bayesian_optimize_proposed_shift", lambda **_k: (0.0, 0.0, 0.95))

    rr.solve_rate_from_rate(a, b, FrameShift(source_index=5, target_index=4))

    plate = a.starfield.wcs_metadata.x_ifov_arcsec
    dec = float(a.starfield.wcs.to_astropy_wcs().wcs.crval[1])
    header_rate = rr.track_rate_from_header(a.frame.header, plate, dec)
    dt = 5.0  # |t_a - t_b| seconds; expected_shift = rate * (gap + 0.5*(exp_a+exp_b)) = rate * dt
    assert captured["expected_shift"] == pytest.approx(header_rate * dt, rel=1e-6)


def test_windowed_correlation_peaks_returns_distinct_candidates() -> None:
    """The multi-peak finder returns several distinct in-window peaks, brightest first.

    Each is separated from the others, so the caller can validate candidates beyond the
    brightest rather than re-finding one cluster (the 963e1bc5 / 28190 spurious-peak failure).
    """
    from senpai.engine.detection.streak.rate_rate import windowed_correlation_peaks

    rng = np.random.default_rng(0)
    cc = (rng.random((201, 201)).astype(np.float32)) * 0.01
    # Brightest peak (spurious-analog) and a fainter second peak, both inside the 1.2*50=60
    # window of the (100, 100) centre.
    cc[100, 150] = 100.0  # 50 px right of center
    cc[140, 100] = 60.0  # 40 px below center
    peaks = windowed_correlation_peaks(cc, expected_shift=50.0, n_peaks=3)

    assert len(peaks) == 3
    p0, p1 = np.asarray(peaks[0]), np.asarray(peaks[1])
    assert abs(p0[0] - 100) <= 2 and abs(p0[1] - 150) <= 2, "brightest peak returned first"
    assert abs(p1[0] - 140) <= 2 and abs(p1[1] - 100) <= 2, (
        "second peak is the distinct fainter one"
    )
    assert np.linalg.norm(p0 - p1) > 20, "candidates must be distinct, not the same cluster"


def test_solve_rate_from_rate_picks_corr_validated_peak_over_brighter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distinct fainter peak that validates is chosen over a brighter peak that fails the gate.

    When the brightest in-window peak fails the correlation gate (a spurious feature) but a
    fainter one validates, the validated peak is selected (963e1bc5: true peak is 2nd-brightest).

    Args:
        monkeypatch: Pytest fixture used to stub the peak finder and the correlation optimizer.
    """
    import senpai.engine.detection.streak.rate_rate as rr
    from senpai.engine.models.senpai import FrameShift

    rng = np.random.default_rng(1)
    a = _solved_rate_frame(
        5,
        rng.integers(100, 500, (120, 120), dtype=np.uint16),
        timestamp=datetime(2024, 1, 1, 0, 0, 5),
        exptime=1.0,
    )
    a.pixel_track_rate_per_second = 40.0  # seed present -> skip the header init path
    b = _solved_rate_frame(
        4,
        rng.integers(100, 500, (120, 120), dtype=np.uint16),
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        exptime=1.0,
    )
    b.starfield = None

    # Two candidate peaks as offsets from the cc centre (frames are cropped during
    # preprocessing, so derive peaks from the cc we're handed). First is the brighter
    # "spurious" (shift x=-50), second the "true" (shift x=-40).
    def fake_peaks(cc: np.ndarray, _es: float | None, n_peaks: int = 1) -> list[np.ndarray]:
        """Return two fixed candidate peaks derived from the cc centre."""
        c = np.array(cc.shape) / 2.0
        return [c + np.array([0, 50]), c + np.array([0, 40])][: max(1, n_peaks)]

    monkeypatch.setattr(rr, "windowed_correlation_peaks", fake_peaks)

    def fake_bayesian(**kw: object) -> tuple[float, float, float]:
        """Validate only the true peak (x ~ -40) with a high correlation."""
        mid_x = (kw["shift_x_low"] + kw["shift_x_high"]) / 2.0
        mid_y = (kw["shift_y_low"] + kw["shift_y_high"]) / 2.0
        corr = 0.99 if abs(mid_x - (-40.0)) < 3.0 else 0.3  # only the true peak validates
        return mid_x, mid_y, corr

    monkeypatch.setattr(rr, "bayesian_optimize_proposed_shift", fake_bayesian)

    fs = FrameShift(source_index=5, target_index=4)
    rr.solve_rate_from_rate(a, b, fs)

    # Selected shift should be the validated true peak (x ~ -40, minus the 1px convention).
    assert abs(fs.x_shift - (-41.0)) < 2.0, (
        f"expected the corr-validated peak (~-41), got {fs.x_shift}"
    )


def test_solve_rate_from_rate_falls_back_to_brightest_when_no_peak_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no candidate clears the correlation gate, the brightest (first) peak is kept.

    Wandering off the brightest peak to a marginally-better distinct peak misregisters the
    first pair and zeroes collects where the true peak is brightest but lands just under 0.9
    (c92a121e/39741, 3bb31699). Best-correlation and closest-to-seed selectors were each
    evaluated on the full MDP set; both recovered one collect but regressed more, so
    brightest-with-fallback was the empirical best net.

    Args:
        monkeypatch: Pytest fixture used to stub the peak finder and the correlation optimizer.
    """
    import senpai.engine.detection.streak.rate_rate as rr
    from senpai.engine.models.senpai import FrameShift

    rng = np.random.default_rng(2)
    a = _solved_rate_frame(
        5,
        rng.integers(100, 500, (120, 120), dtype=np.uint16),
        timestamp=datetime(2024, 1, 1, 0, 0, 5),
        exptime=1.0,
    )
    a.pixel_track_rate_per_second = 40.0  # seed present -> skip the header init path
    b = _solved_rate_frame(
        4,
        rng.integers(100, 500, (120, 120), dtype=np.uint16),
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        exptime=1.0,
    )
    b.starfield = None

    # First (brightest) candidate is the true peak (shift x=-40); the second is a spurious
    # distinct peak (shift x=-50) that correlates slightly better but still fails the 0.9 gate.
    def fake_peaks(cc: np.ndarray, _es: float | None, n_peaks: int = 1) -> list[np.ndarray]:
        """Return two fixed candidate peaks derived from the cc centre."""
        c = np.array(cc.shape) / 2.0
        return [c + np.array([0, 40]), c + np.array([0, 50])][: max(1, n_peaks)]

    monkeypatch.setattr(rr, "windowed_correlation_peaks", fake_peaks)

    def fake_bayesian(**kw: object) -> tuple[float, float, float]:
        """Report a sub-gate correlation for both peaks, higher for the spurious one."""
        mid_x = (kw["shift_x_low"] + kw["shift_x_high"]) / 2.0
        mid_y = (kw["shift_y_low"] + kw["shift_y_high"]) / 2.0
        corr = 0.8 if abs(mid_x - (-50.0)) < 3.0 else 0.6  # neither clears 0.9; spurious higher
        return mid_x, mid_y, corr

    monkeypatch.setattr(rr, "bayesian_optimize_proposed_shift", fake_bayesian)

    fs = FrameShift(source_index=5, target_index=4)
    rr.solve_rate_from_rate(a, b, fs)

    # No peak validated -> keep the brightest (x ~ -40, minus the 1px convention), NOT -50.
    assert abs(fs.x_shift - (-41.0)) < 2.0, (
        f"expected fallback to the brightest peak (~-41), got {fs.x_shift}"
    )
