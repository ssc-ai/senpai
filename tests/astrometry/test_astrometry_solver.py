"""Unit tests for the astrometry solver package.

Covers the FITS solve entry point (``senpai.astrometry.solve_field_fits`` and its source
extractor dispatch) plus the ``senpai.astrometry.runner`` resource-management internals:

  - Index page-cache release (``_release_index_page_cache`` / ``_index_files``): hands the
    astrometry index files' clean page cache back to the kernel via
    ``posix_fadvise(POSIX_FADV_DONTNEED)`` so a sky-sweeping run's working set stays flat.
  - ra/dec tag-along index guard (``_hdu13_has_radec_tagalong``): distinguishes 5200/Gaia
    indices (ra/dec catalog tag-along at HDU 13) from 4100/Tycho-2 indices (photometry
    only), so a blind solve winning on a 4100 quad degrades to "no catalog stars" rather
    than raising.
  - Per-solve SIGALRM timeout (``_solve_time_limit``): bounds a blocking C solve to a
    wall-clock limit via ``signal.setitimer``; a documented no-op off the main thread.

The tests use synthetic frames and mocks only -- no astrometry.net, index files, or
star catalogs.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from senpai.astrometry import runner
from senpai.core import config as cfg_mod
from senpai.core.config import AppConfig, get_or_initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.engine.models.images import ProcessedFitsImage
from senpai.exceptions import MissingDependencyError

_HAS_FADVISE = hasattr(os, "posix_fadvise")
_fadvise_only = pytest.mark.skipif(not _HAS_FADVISE, reason="os.posix_fadvise is Unix-only")


# --------------------------------------------------------------------------- #
# solve_field_fits entry point
# --------------------------------------------------------------------------- #
def _min_app(**overrides: object) -> dict:
    """Build a minimal ``AppConfig`` payload for the solver tests.

    Args:
        **overrides: Top-level keys to add to or replace in the base payload.

    Returns:
        A config dict suitable for constructing an ``AppConfig``.
    """
    data = {
        "version": "1.0.0",
        "astrometry": {
            "indices_series": "5200_LITE",
            "indices_path": "/nonexistent/idx",
            "max_sources": 100,
            "min_sources_for_attempt": 10,
            "min_width_degrees": 0.1,
            "max_width_degrees": 10.0,
            "cpulimit_seconds": 30,
            "docker_image": None,
        },
        "plotting": {"debug": False, "review": False},
        "star_catalog": {"type": "gaia"},
    }
    data.update(overrides)
    return data


def _blank_image(size: int = 64) -> ProcessedFitsImage:
    """Build a starless Gaussian-noise frame.

    Args:
        size: Side length of the square frame in pixels.

    Returns:
        A ``ProcessedFitsImage`` containing pure noise (no extractable sources).
    """
    rng = np.random.default_rng(42)
    data = rng.normal(100.0, 1.0, (size, size)).astype(np.float64)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["NAXIS1"] = size
    hdu.header["NAXIS2"] = size
    return ProcessedFitsImage.from_fits(hdu)


def test_solve_field_fits_empty_field_returns_unfit_starfield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A starless field yields an unfit StarField (no WCS) that still carries frame metadata.

    Args:
        monkeypatch: Pytest fixture used to install the minimal config.
    """
    from senpai.astrometry import solve_field_fits

    monkeypatch.setattr(cfg_mod, "_config_instance", AppConfig(**_min_app()))
    starfield = solve_field_fits(_blank_image())
    assert starfield.fit is False
    assert starfield.wcs is None
    # detections list mirrors whatever the extractor found (possibly empty)
    assert starfield.image_metadata.width == 64


def test_solve_field_fits_respects_source_extractor_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The solve entry dispatches to the sextractor extractor with the configured max sources.

    Args:
        monkeypatch: Pytest fixture used to install the config and stub the extractor.
    """
    import senpai.astrometry.astroeasy_backend as ast_mod

    monkeypatch.setattr(cfg_mod, "_config_instance", AppConfig(**_min_app()))

    called = {}

    def fake_extract(
        fits_img: ProcessedFitsImage,
        max_detections: int | None = None,
        method: str | None = None,
        **kwargs: object,
    ) -> tuple:
        called["method"] = method
        called["max_detections"] = max_detections
        from senpai.engine.models.starfield import StarListImage

        return StarListImage(detections=[], image_metadata=fits_img.metadata), 2.0, []

    monkeypatch.setattr(ast_mod, "extract_point_sources", fake_extract)
    ast_mod.solve_field_fits(_blank_image())
    assert called["method"] == "sextractor"
    assert called["max_detections"] == 100


# --------------------------------------------------------------------------- #
# Index page-cache release (_release_index_page_cache / _index_files)
# --------------------------------------------------------------------------- #
def _touch_index_files(directory: Path, n: int) -> list[Path]:
    """Create ``n`` small dummy index .fits files.

    Args:
        directory: Directory to create the files in.
        n: Number of files to create.

    Returns:
        The created file paths.
    """
    paths = []
    for i in range(n):
        p = directory / f"index-520{i}-00.fits"
        p.write_bytes(b"\x00" * 4096)
        paths.append(p)
    return paths


@_fadvise_only
def test_release_index_page_cache_hints_dontneed_on_each_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every index file receives a whole-file ``POSIX_FADV_DONTNEED`` hint.

    Args:
        tmp_path: Pytest temporary directory for the dummy index files.
        monkeypatch: Pytest fixture used to stub the index-file list and record fadvise calls.
    """
    files = _touch_index_files(tmp_path, 3)
    monkeypatch.setattr(runner, "_index_files", lambda: files)

    real_fadvise = os.posix_fadvise
    calls: list[tuple[int, int, int]] = []

    def _record(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((offset, length, advice))
        real_fadvise(fd, offset, length, advice)  # exercise the real syscall too

    monkeypatch.setattr(os, "posix_fadvise", _record)

    runner._release_index_page_cache()  # must not raise

    assert len(calls) == len(files)
    assert all(advice == os.POSIX_FADV_DONTNEED for _, _, advice in calls)
    # Whole-file hint: offset 0, length 0 ("to end of file").
    assert all(offset == 0 and length == 0 for offset, length, _ in calls)


@_fadvise_only
def test_release_index_page_cache_noop_when_no_index_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no index files configured, the release is a clean no-op.

    Args:
        monkeypatch: Pytest fixture used to stub an empty index-file list and record calls.
    """
    monkeypatch.setattr(runner, "_index_files", list)
    calls: list[int] = []
    monkeypatch.setattr(os, "posix_fadvise", lambda *a: calls.append(1))

    runner._release_index_page_cache()  # clean no-op

    assert calls == []


@_fadvise_only
def test_release_index_page_cache_survives_fadvise_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing cache hint is best-effort: logged and swallowed, never raised.

    Args:
        tmp_path: Pytest temporary directory for the dummy index files.
        monkeypatch: Pytest fixture used to stub the index-file list and a raising fadvise.
    """
    files = _touch_index_files(tmp_path, 2)
    monkeypatch.setattr(runner, "_index_files", lambda: files)

    def _boom(*_a: object) -> None:
        raise OSError("fadvise failed")

    monkeypatch.setattr(os, "posix_fadvise", _boom)

    runner._release_index_page_cache()  # does not propagate the OSError


def test_index_files_reads_configured_indices_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_index_files`` globs only ``*.fits`` under the configured indices path.

    Args:
        tmp_path: Pytest temporary directory used as the indices path.
        monkeypatch: Pytest fixture used to install a config pointing at ``tmp_path``.
    """
    _touch_index_files(tmp_path, 2)
    (tmp_path / "not_an_index.txt").write_bytes(b"x")

    base = get_or_initialize_config(CONFIG_DIR / "local.yaml")
    data = base.model_dump()
    data["astrometry"]["indices_path"] = str(tmp_path)
    app = AppConfig(**data)
    monkeypatch.setattr(cfg_mod, "_config_instance", app)

    found = runner._index_files()
    assert sorted(p.name for p in found) == ["index-5200-00.fits", "index-5201-00.fits"]


def test_index_files_empty_for_missing_indices_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-existent indices path yields an empty index-file list.

    Args:
        tmp_path: Pytest temporary directory whose non-existent child is the indices path.
        monkeypatch: Pytest fixture used to install the config.
    """
    base = get_or_initialize_config(CONFIG_DIR / "local.yaml")
    data = base.model_dump()
    data["astrometry"]["indices_path"] = str(tmp_path / "does_not_exist")
    app = AppConfig(**data)
    monkeypatch.setattr(cfg_mod, "_config_instance", app)

    assert runner._index_files() == []


# --------------------------------------------------------------------------- #
# External binary dependency checks (image2xy from astrometry.net)
# --------------------------------------------------------------------------- #
def test_examine_astrometry_install_false_when_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ``image2xy`` binary fails the install check even when index files are present.

    Args:
        tmp_path: Pytest temporary directory used as the indices path.
        monkeypatch: Pytest fixture used to install the config and stub PATH resolution.
    """
    _touch_index_files(tmp_path, 1)
    base = get_or_initialize_config(CONFIG_DIR / "local.yaml")
    data = base.model_dump()
    data["astrometry"]["indices_path"] = str(tmp_path)
    monkeypatch.setattr(cfg_mod, "_config_instance", AppConfig(**data))
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)

    assert runner.examine_astrometry_install() is False


def test_check_astrometry_dependencies_passes_when_binary_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No error is raised when every required binary resolves on ``PATH``.

    Args:
        monkeypatch: Pytest fixture used to stub PATH resolution to a present binary.
    """
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/image2xy")
    runner.check_astrometry_dependencies()


def test_check_astrometry_dependencies_raises_when_binary_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing binary raises the typed ``MissingDependencyError`` naming the binary.

    Args:
        monkeypatch: Pytest fixture used to stub PATH resolution to a missing binary.
    """
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    with pytest.raises(MissingDependencyError, match="image2xy"):
        runner.check_astrometry_dependencies()


def test_extract_stars_image2xy_raises_typed_error_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``image2xy`` call site converts a bare ``FileNotFoundError`` into a typed error.

    A missing binary surfaces from ``subprocess.run`` as ``FileNotFoundError``; the extractor
    must re-raise it as ``MissingDependencyError`` so the boundary reports an actionable install
    message rather than an opaque crash.

    Args:
        monkeypatch: Pytest fixture used to install the config and stub ``subprocess.run``.
    """
    monkeypatch.setattr(cfg_mod, "_config_instance", AppConfig(**_min_app()))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("image2xy")

    monkeypatch.setattr(runner.subprocess, "run", _boom)
    with pytest.raises(MissingDependencyError, match="image2xy"):
        runner._extract_stars_image2xy(_blank_image())


# --------------------------------------------------------------------------- #
# ra/dec tag-along index guard (_hdu13_has_radec_tagalong)
# --------------------------------------------------------------------------- #
def _hdulist_with_columns_at_13(*column_names: str) -> fits.HDUList:
    """Build an HDUList whose HDU 13 is a table with the given columns.

    Args:
        *column_names: Column names for the HDU-13 binary table.

    Returns:
        A 14-HDU ``HDUList`` whose index 13 is the populated table.
    """
    prim = fits.PrimaryHDU()
    filler = [fits.ImageHDU() for _ in range(12)]  # HDUs 1..12
    cols = fits.ColDefs(
        [fits.Column(name=name, format="D", array=np.array([1.0, 2.0])) for name in column_names]
    )
    bintable = fits.BinTableHDU.from_columns(cols)
    return fits.HDUList([prim, *filler, bintable])  # index 13 == bintable


def test_hdu13_tagalong_true_with_ra_dec() -> None:
    """An HDU-13 table carrying lowercase ra/dec columns is a catalog tag-along index."""
    hdul = _hdulist_with_columns_at_13("ra", "dec", "mag", "ref_id")
    assert runner._hdu13_has_radec_tagalong(hdul) is True


def test_hdu13_tagalong_false_without_ra_dec() -> None:
    """A photometry-only HDU-13 table (4100/Tycho-2 style) is not a tag-along index."""
    hdul = _hdulist_with_columns_at_13("MAG_BT", "MAG_VT")
    assert runner._hdu13_has_radec_tagalong(hdul) is False


def test_hdu13_tagalong_false_when_fewer_than_14_hdus() -> None:
    """An index with fewer than 14 HDUs cannot carry the tag-along table."""
    hdul = fits.HDUList([fits.PrimaryHDU(), *[fits.ImageHDU() for _ in range(4)]])
    assert len(hdul) < 14
    assert runner._hdu13_has_radec_tagalong(hdul) is False


def test_hdu13_tagalong_false_when_data_none() -> None:
    """A 14-HDU index whose HDU 13 has no table data is not a tag-along index."""
    hdul = fits.HDUList([fits.PrimaryHDU(), *[fits.ImageHDU() for _ in range(13)]])
    assert len(hdul) == 14
    assert hdul[13].data is None
    assert runner._hdu13_has_radec_tagalong(hdul) is False


# --------------------------------------------------------------------------- #
# Per-solve SIGALRM timeout (_solve_time_limit)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="signal.setitimer is Unix-only")
class TestSolveTimeLimit:
    """The ``_solve_time_limit`` context manager bounds a blocking call via SIGALRM."""

    def test_raises_timeout_on_overrun(self) -> None:
        """An overrunning call is interrupted with ``TimeoutError`` and the timer is disarmed."""
        start = time.monotonic()
        with pytest.raises(TimeoutError, match=r"exceeded 0\.2s"), runner._solve_time_limit(0.2):
            time.sleep(1.0)  # interrupted by SIGALRM at ~0.2s
        # The alarm fired well before the nominal 1s sleep completed.
        assert time.monotonic() - start < 0.9
        # And the interval timer is disarmed afterward.
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)

    def test_fast_op_completes_and_clears_alarm(self) -> None:
        """A call that finishes in time runs to completion and clears the alarm."""
        ran = False
        with runner._solve_time_limit(5):
            ran = True
        assert ran
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)

    def test_nonpositive_limit_disables(self) -> None:
        """A non-positive limit disables the guard so no timer is ever armed."""
        with runner._solve_time_limit(0):
            pass
        # No timer was ever armed.
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)

    def test_noop_off_main_thread(self) -> None:
        """Off the main thread the guard is a documented no-op (signals are undeliverable)."""
        outcome: list[str] = []

        def worker() -> None:
            """Run the guard on a worker thread and record its outcome."""
            try:
                with runner._solve_time_limit(1):
                    time.sleep(0.05)
                outcome.append("ok")
            except Exception as exc:  # record any escape from the guard
                outcome.append(f"error: {exc!r}")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert outcome == ["ok"]
