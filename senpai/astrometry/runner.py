"""Plate-solving via the astrometry.net Python package and SIP/WCS refinement."""

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import astrometry
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# ``NoConvergence`` is astropy's own exception, raised by ``all_world2pix`` when the iterative SIP
# inverse diverges. We catch it (rather than wrap it in a SenpaiError) because it is recoverable:
# we fall back to astropy's best-effort solution and warn, so nothing propagates to the boundary.
from astropy.wcs.wcs import NoConvergence
from scipy.spatial import cKDTree

from senpai.engine.detection.point.sidereal import extract_point_sources
from senpai.engine.models.astrometry import ReturnAstrometryConfig, WCSModel, WCSStatus
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.starfield import StarField, StarInImage, StarInSpace
from senpai.exceptions import MissingDependencyError
from senpai.settings import settings

logger = logging.getLogger(__name__)


# External command-line binaries the astrometry path shells out to. ``image2xy`` (from
# astrometry.net) is invoked unconditionally during WCS refinement, so its absence makes every
# solve fail. Kept as a module constant so callers can validate the environment up front.
REQUIRED_ASTROMETRY_BINARIES: tuple[str, ...] = ("image2xy",)


def _missing_astrometry_binaries() -> list[str]:
    """Return required astrometry CLI binaries (see ``REQUIRED_ASTROMETRY_BINARIES``) not on PATH.

    Returns:
        list[str]: Names of missing binaries; empty if all are present.
    """
    return [b for b in REQUIRED_ASTROMETRY_BINARIES if shutil.which(b) is None]


def check_astrometry_dependencies() -> None:
    """Verify the external binaries the astrometry solve/refine path needs are on ``PATH``.

    Call this before a run to fail fast with an actionable message rather than deep inside a solve.

    Raises:
        MissingDependencyError: If any required binary is missing.
    """
    missing = _missing_astrometry_binaries()
    if missing:
        raise MissingDependencyError(
            "Required astrometry binaries not found on PATH: "
            + ", ".join(missing)
            + ". These come from astrometry.net (Debian/Ubuntu: `apt-get install astrometry.net`, "
            "or build from https://github.com/dstndstn/astrometry.net); ensure its bin/ is on PATH."
        )


# Minimum increase in matched catalog stars required to advance to a finer (denser)
# index series in the post-refit diminishing-returns catalog selection loop.
# If going one level finer adds fewer than this many new matches, the current level
# has already captured all detectable stars in the image.
_MIN_MATCH_DELTA: int = 5


def _default_logodds_callback(logodds_list: list[float]) -> astrometry.Action:
    """Stop the solver as soon as a match exceeds the configured confidence threshold."""
    if logodds_list and max(logodds_list) >= settings.astrometry.min_logodds_threshold:
        return astrometry.Action.STOP
    return astrometry.Action.CONTINUE


def _index_files() -> list[Path]:
    """Return all .fits index files found at the configured indices path, sorted."""
    return sorted(Path(settings.astrometry.indices_path).glob("*.fits"))


def _release_index_page_cache() -> None:
    """Drop the astrometry index files' page cache via ``posix_fadvise(POSIX_FADV_DONTNEED)``.

    Each plate solve builds a fresh ``astrometry.Solver`` that mmaps index tiles (and the SIP/verify
    steps ``fits.open`` the matched index with ``memmap=True``); all of those mappings are released by
    the time a solve returns, but the pages they touched linger as file-backed page cache. Across a run
    that sweeps the sky this accumulates toward the full index size (~52 GiB), so the container's
    working set climbs like a leak even though the memory is reclaimable. Called after every solve, this
    hands the now-unmapped clean pages back to the kernel, keeping the working set flat. Best-effort: a
    failed hint is logged and ignored (the next solve just re-reads the tiles it needs from disk).
    """
    for path in _index_files():
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as exc:  # pragma: no cover - best-effort cache hint
            logger.debug("posix_fadvise(DONTNEED) failed for %s: %s", path, exc)


def _size_hint(image_width: int) -> astrometry.SizeHint:
    """Convert FOV bounds (degrees) to a per-pixel scale hint for the solver.

    Args:
        image_width (int): image width in pixels, used to derive arcsec/pixel bounds.

    Returns:
        astrometry.SizeHint: lower/upper arcsec-per-pixel bounds derived from
            settings.astrometry.min_width_degrees and max_width_degrees.
    """
    return astrometry.SizeHint(
        lower_arcsec_per_pixel=settings.astrometry.min_width_degrees * 3600 / image_width,
        upper_arcsec_per_pixel=settings.astrometry.max_width_degrees * 3600 / image_width,
    )


def _position_hint(ra: float | None, dec: float | None) -> astrometry.PositionHint | None:
    """Build a sky-position search constraint from a boresight coordinate.

    Args:
        ra (float | None): boresight right ascension in degrees.
        dec (float | None): boresight declination in degrees.

    Returns:
        astrometry.PositionHint | None: search constraint centered on (ra, dec) with
            radius settings.astrometry.search_radius_degrees, or None if either
            coordinate is missing.
    """
    if ra is None or dec is None:
        return None
    return astrometry.PositionHint(
        ra_deg=ra,
        dec_deg=dec,
        radius_deg=settings.astrometry.search_radius_degrees,
    )


def _solution_parameters(
    sip_order: int | None = None,
    min_logodds_threshold: float | None = None,
    tune_up_logodds_threshold: float | None = 14.0,
    parity: astrometry.Parity = astrometry.Parity.BOTH,
    positional_noise_pixels: float = 1.0,
    distractor_ratio: float = 0.25,
    code_tolerance_l2_distance: float = 0.01,
    minimum_quad_size_pixels: float | None = None,
    maximum_quads: int = 0,
    maximum_matches: int = 0,
    logodds_callback: Callable[[list[float]], astrometry.Action] | None = None,
) -> astrometry.SolutionParameters:
    """Build astrometry.net solver parameters.

    Keyword arguments override config/library defaults, enabling per-call tuning without
    changing global settings. sip_order and min_logodds_threshold fall back to
    settings.astrometry when None; all other parameters use astrometry.net library defaults.

    Args:
        sip_order (int | None): SIP distortion polynomial order; 0 disables SIP.
            Defaults to settings.astrometry.sip_order.
        min_logodds_threshold (float | None): minimum log-odds confidence for accepting
            a solution. Defaults to settings.astrometry.min_logodds_threshold.
        tune_up_logodds_threshold (float | None): log-odds threshold before SIP distortion
            fitting is applied. Defaults to 14.0.
        parity (astrometry.Parity): axis orientation — NORMAL, FLIP, or BOTH.
            BOTH doubles search time. Defaults to astrometry.Parity.BOTH.
        positional_noise_pixels (float): expected star position error in pixels.
            Defaults to 1.0.
        distractor_ratio (float): expected fraction of spurious detections.
            Defaults to 0.25.
        code_tolerance_l2_distance (float): hash space tolerance. Defaults to 0.01.
        minimum_quad_size_pixels (float | None): smallest quad size; None = auto.
            Defaults to None.
        maximum_quads (int): quads to attempt; 0 = unlimited. Defaults to 0.
        maximum_matches (int): match attempts; 0 = unlimited. Defaults to 0.
        logodds_callback (Callable | None): called after each match attempt with current
            logodds scores; return Action.STOP to halt early, Action.CONTINUE to keep going.
            Defaults to _default_logodds_callback (stops when threshold is reached).

    Returns:
        astrometry.SolutionParameters: fully populated solver parameter object.
    """
    resolved_sip_order = sip_order if sip_order is not None else settings.astrometry.sip_order
    resolved_tune_up = tune_up_logodds_threshold if resolved_sip_order > 0 else None

    return astrometry.SolutionParameters(
        sip_order=resolved_sip_order,
        output_logodds_threshold=(
            min_logodds_threshold
            if min_logodds_threshold is not None
            else settings.astrometry.min_logodds_threshold
        ),
        tune_up_logodds_threshold=resolved_tune_up,
        parity=parity,
        positional_noise_pixels=positional_noise_pixels,
        distractor_ratio=distractor_ratio,
        code_tolerance_l2_distance=code_tolerance_l2_distance,
        minimum_quad_size_pixels=minimum_quad_size_pixels,
        maximum_quads=maximum_quads,
        maximum_matches=maximum_matches,
        logodds_callback=logodds_callback
        if logodds_callback is not None
        else _default_logodds_callback,
    )


def _extract_stars(fits_img: ProcessedFitsImage) -> list[StarInImage]:
    """Extract point sources for plate solving using the configured extractor.

    Dispatches to image2xy or the in-process point-source extractor based on
    settings.astrometry.source_extractor.

    Args:
        fits_img (ProcessedFitsImage): preprocessed FITS image to extract sources from.

    Returns:
        list[StarInImage]: detected sources, capped at settings.astrometry.max_sources.
    """
    extractor = settings.astrometry.source_extractor
    if extractor == "image2xy":
        return _extract_stars_image2xy(fits_img)
    sources, _, _ = extract_point_sources(
        fits_img,
        max_detections=settings.astrometry.max_sources,
        method=extractor,
    )
    return sources.detections


def _extract_stars_image2xy(
    fits_img: ProcessedFitsImage, max_sources: int | None = None
) -> list[StarInImage]:
    """Run the image2xy binary on a temporary FITS file and return extracted positions.

    Args:
        fits_img (ProcessedFitsImage): preprocessed FITS image to extract sources from.
        max_sources (int | None): cap; None uses settings.astrometry.max_sources;
            a large value (e.g. 10000) effectively returns all detections.

    Returns:
        list[StarInImage]: list of StarInImage objects, capped at max_sources.
    """
    cap = settings.astrometry.max_sources if max_sources is None else max_sources
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        temp_fits = temp_dir / "image.fits"
        fits.PrimaryHDU(data=fits_img.data, header=fits_img.header).writeto(
            temp_fits, overwrite=True
        )
        try:
            subprocess.run(["image2xy", str(temp_fits)], check=True, capture_output=True)  # noqa: S603, S607
        except FileNotFoundError as exc:
            raise MissingDependencyError(
                "The 'image2xy' binary (from astrometry.net) is required for astrometry source "
                "extraction but was not found on PATH. Install astrometry.net (Debian/Ubuntu: "
                "`apt-get install astrometry.net`, or build from "
                "https://github.com/dstndstn/astrometry.net) and ensure its bin/ is on PATH."
            ) from exc
        with fits.open(temp_dir / "image.xy.fits") as hdul:
            data = hdul[1].data
            return [StarInImage(x=float(r["X"]), y=float(r["Y"]), counts=None) for r in data[:cap]]


def _hdu13_has_radec_tagalong(hdul: fits.HDUList) -> bool:
    """Whether an index file exposes a lowercase ra/dec catalog tag-along at HDU 13.

    The 5200/Gaia indices carry ``(ra, dec, mag, ...)`` in HDU 13, which the catalog
    readers below project through the WCS. The 4100/Tycho-2 series instead keeps star
    positions only in the kdtree and places photometry (``MAG_BT``, ...) in HDU 13, so
    reading ``ext13['ra']`` there raises ``KeyError``. When both index series share an
    indices directory the blind solver can win on a 4100 quad; callers use this guard
    to degrade gracefully (no catalog stars) instead of crashing the whole collect.
    """
    if len(hdul) <= 13 or hdul[13].data is None:
        return False
    names = hdul[13].data.names
    return "ra" in names and "dec" in names


def _solver_verify_sip_wcs_logodds(
    index_path: Path,
    wcs: WCS,
    detections_xy: np.ndarray,
    image_width: int,
    image_height: int,
    pixel_scale_arcsec: float,
    verify_pix: float = 1.5,
    distractors: float = 0.25,
    logodds_bail: float = -230.26,  # log(1e-100) — astrometry.net DEFAULT_BAIL_THRESHOLD
) -> float:
    """Pure-Python port of astrometry.net's `solver_verify_sip_wcs` -> verify_hit.

    Replicates the `--verify --tag-all` index selection path. Returns bestlogodds
    (max accumulated logodds during sequential matching of image detections
    against catalog stars projected through the given WCS). This is the same
    `K` the C verify returns from real_verify_star_lists; the index with the
    highest bestlogodds across all candidates is the one --tag-all uses for RDLS.

    Solver_verify_sip_wcs sets distance_from_quad_bonus=FALSE, so sigma is
    constant (no quad-distance gamma). The Gaussian foreground likelihood plus
    Bayesian distractor model exactly mirror real_verify_star_lists in
    astrometry.net/solver/verify.c.
    """
    with fits.open(index_path, memmap=True) as hdul:
        if not _hdu13_has_radec_tagalong(hdul):
            return float("-inf")
        ext13 = hdul[13].data
        cat_ra = ext13["ra"].astype(float)
        cat_dec = ext13["dec"].astype(float)
        # Per-index jitter (arcsec) — typically 0.1 for SENPAI indices
        index_jitter_arcsec = float(hdul[7].header.get("JITTER", 1.0)) if len(hdul) > 7 else 1.0

    cat_pix = wcs.wcs_world2pix(np.column_stack([cat_ra, cat_dec]), 0)
    in_bounds = (
        (cat_pix[:, 0] >= 0)
        & (cat_pix[:, 0] < image_width)
        & (cat_pix[:, 1] >= 0)
        & (cat_pix[:, 1] < image_height)
    )
    refxy = cat_pix[in_bounds]
    NR = len(refxy)
    NT = len(detections_xy)
    if NR == 0 or NT == 0:
        return float("-inf")

    sig2 = verify_pix**2 + (index_jitter_arcsec / pixel_scale_arcsec) ** 2
    rtree = cKDTree(refxy)
    effA = float(image_width * image_height)
    logbg = float(np.log(1.0 / effA))
    nn_radius = float(np.sqrt(25.0 * sig2))
    loggmax_const = float(np.log(max((1.0 - distractors) / (2.0 * np.pi * sig2 * NR), 1e-300)))

    rmatches = np.full(NR, -1, dtype=int)
    rprobs = np.full(NR, -np.inf)
    logodds = 0.0
    bestlogodds = -np.inf
    mu = 0
    for i in range(NT):
        testxy = detections_xy[i]
        mu_term = distractors + (1.0 - distractors) * mu / NR
        logd = float(np.log(max(mu_term, 1e-300))) + logbg
        d, refi = rtree.query(testxy, k=1, distance_upper_bound=nn_radius)
        logfg = -np.inf if not np.isfinite(d) or refi >= NR else loggmax_const - d * d / (2.0 * sig2)
        if logfg < logd:
            logfg = logd
        else:
            if rmatches[refi] != -1:
                # Conflict: keep-old vs upgrade-to-new (approximation of C upgrade logic).
                oldfg = rprobs[refi]
                keepfg = logd
                switchfg = logfg + (logd - oldfg)
                if switchfg > keepfg:
                    rmatches[refi] = i
                    rprobs[refi] = logfg
                    logfg = switchfg
                else:
                    logfg = keepfg
            else:
                rmatches[refi] = i
                rprobs[refi] = logfg
                mu += 1
        logodds += logfg - logbg
        if logodds > bestlogodds:
            bestlogodds = logodds
        if logodds < logodds_bail:
            break
    return float(bestlogodds)


@contextlib.contextmanager
def _solve_time_limit(seconds: int) -> Iterator[None]:
    """Bound a blocking solve to ``seconds`` of wall time via SIGALRM -- no worker thread.

    The astrometry package exposes no native solve timeout. Rather than run the solve in a throwaway
    thread (which cannot actually be cancelled -- a timed-out thread keeps the C solve running), we arm
    an interval timer whose SIGALRM handler raises, unwinding out of the C solve at the next interpreter
    check. The solver calls its Python ``logodds_callback`` frequently, so the handler fires promptly in
    practice. Signals are deliverable only on a process's main thread; the process-pool workers that run
    solves in production are always on their main thread, so this is a safe no-op elsewhere (e.g. under
    the in-process test executor) -- the request-level timeout still bounds those.

    Args:
        seconds (int): wall-clock limit; values <= 0 disable the limit.

    Yields:
        None.

    Raises:
        TimeoutError: if the guarded block runs longer than ``seconds`` (on the main thread).
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _on_alarm(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"astrometry solve exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _solve_with_timeout(
    stars: list[list[float]],
    size_hint: astrometry.SizeHint,
    position_hint: astrometry.PositionHint | None,
    solution_parameters: astrometry.SolutionParameters,
    index_files: list[Path] | None = None,
) -> astrometry.Solution | None:
    """Run the astrometry.net solver, bailing out after cpulimit_seconds (see ``_solve_time_limit``).

    Args:
        stars (list[list[float]]): list of [x, y] star pixel coordinates.
        size_hint (astrometry.SizeHint): FOV scale bounds for the solver.
        position_hint (astrometry.PositionHint | None): optional sky position constraint.
        solution_parameters (astrometry.SolutionParameters): solver tuning parameters.
        index_files (list[Path] | None): explicit index files to load; None uses
            the default set from _index_files().

    Returns:
        astrometry.Solution | None: solver result, or None on timeout or missing indices.
    """
    idx = index_files if index_files is not None else _index_files()
    if not idx:
        logger.error("No index files found at %s", settings.astrometry.indices_path)
        return None

    try:
        with astrometry.Solver(idx) as solver, _solve_time_limit(settings.astrometry.cpulimit_seconds):
            return solver.solve(
                stars=stars,
                size_hint=size_hint,
                position_hint=position_hint,
                solution_parameters=solution_parameters,
            )
    except TimeoutError:
        logger.warning(
            "Astrometry solve timed out after %s s", settings.astrometry.cpulimit_seconds
        )
        return None
    except TypeError:
        return None


def project_world_to_pixels(wcs: WCS, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Project world coordinates to pixels, tolerating a non-convergent SIP inverse.

    ``all_world2pix`` inverts the SIP distortion iteratively. A degenerate WCS (e.g. a weak solve
    whose SIP refinement failed) makes that iteration diverge for far-off-field catalog stars and
    raise astropy's ``NoConvergence``. A single bad frame must not 500 the whole detect request, so on
    divergence we fall back to astropy's best-effort solution: converged points keep their accurate
    positions, divergent off-field points get last-iterate values, and the caller's in-bounds mask
    discards anything that lands outside the image. (Diagnosis: marco.kobayashi, fix/guard-wcs-noconverge.)

    Args:
        wcs (WCS): WCS used to project ra/dec to pixel coordinates.
        ra (np.ndarray): right ascensions in degrees.
        dec (np.ndarray): declinations in degrees.

    Returns:
        np.ndarray: (N, 2) array of [x, y] pixel coordinates with origin 0.
    """
    world = np.column_stack([ra, dec])
    try:
        return wcs.all_world2pix(world, 0)
    except NoConvergence as exc:
        logger.warning(
            "all_world2pix did not converge for %d/%d points projected through the fit WCS "
            "(likely a degenerate solution); using best-effort positions and discarding off-field "
            "points. %s",
            np.size(exc.divergent),
            len(ra),
            exc,
        )
        return np.asarray(exc.best_solution)


def _stars_from_match_index(
    index_path: Path,
    fit_wcs_astropy: WCS,
    image_width: int,
    image_height: int,
) -> list[StarInSpace]:
    """Read catalog stars from a winning index file and project them to pixels.

    Reads extension 13 (ra, dec, mag, ref_cat, ref_id) of the given astrometry.net
    index file, projects each star through the supplied WCS, and keeps those landing
    within the image bounds.

    Args:
        index_path (Path): path to the astrometry.net index FITS file.
        fit_wcs_astropy (WCS): WCS used to project catalog ra/dec to pixel coordinates.
        image_width (int): image width in pixels (for the in-bounds test).
        image_height (int): image height in pixels (for the in-bounds test).

    Returns:
        list[StarInSpace]: in-bounds catalog stars sorted by ascending magnitude.
    """
    with fits.open(index_path, memmap=True) as hdul:
        if not _hdu13_has_radec_tagalong(hdul):
            logger.warning(
                "Index %s has no ra/dec catalog tag-along at HDU 13; returning no "
                "catalog stars (the WCS solution is unaffected). Expected for "
                "4100/Tycho-2 indices, which keep star positions only in the kdtree.",
                index_path.name,
            )
            return []
        ext13 = hdul[13].data
        ra = ext13["ra"].astype(float)
        dec = ext13["dec"].astype(float)
        mag = ext13["mag"].astype(float) if "mag" in ext13.names else np.full(len(ra), np.nan)
        ref_cat = ext13["ref_cat"].astype(str) if "ref_cat" in ext13.names else [None] * len(ra)
        ref_id = ext13["ref_id"].astype(str) if "ref_id" in ext13.names else [None] * len(ra)

    pix = project_world_to_pixels(fit_wcs_astropy, ra, dec)
    x, y = pix[:, 0], pix[:, 1]
    mask = (x >= 0) & (x < image_width) & (y >= 0) & (y < image_height)

    stars = [
        StarInSpace(
            ra=float(ra[i]),
            dec=float(dec[i]),
            magnitude=float(mag[i]) if not np.isnan(mag[i]) else None,
            catalog=str(ref_cat[i]) if ref_cat[i] is not None else None,
            catalog_id=str(ref_id[i]) if ref_id[i] is not None else None,
            x=float(x[i]),
            y=float(y[i]),
        )
        for i in np.where(mask)[0]
    ]
    stars.sort(key=lambda s: s.magnitude if s.magnitude is not None else float("inf"))
    return stars


def _verify_index_logodds(
    index_path: Path,
    wcs: WCS,
    det_xy: np.ndarray,
    image_width: int,
    image_height: int,
    match_radius: float = 3.0,
) -> tuple[int, int, float]:
    """Compute verify-mode logodds for a single index against a WCS.

    Direct Python analog of verify.c:
        p_bg = n_det * pi * r^2 / image_area
        logodds = n_matches * log(1/p_bg) + n_unmatched * log(1/(1-p_bg))

    Favours denser indices (more in-bounds catalog stars → more matches →
    higher logodds), replicating the index competition in --verify --tag-all.

    Returns:
        (n_inbounds, n_matches, logodds)
    """
    with fits.open(index_path, memmap=True) as hdul:
        if not _hdu13_has_radec_tagalong(hdul):
            return 0, 0, float("-inf")
        ext13 = hdul[13].data
        cat_ra = ext13["ra"].astype(float)
        cat_dec = ext13["dec"].astype(float)

    pix = wcs.wcs_world2pix(np.column_stack([cat_ra, cat_dec]), 0)
    in_bounds = (
        (pix[:, 0] >= 0)
        & (pix[:, 0] < image_width)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] < image_height)
    )
    n_inbounds = int(in_bounds.sum())
    if n_inbounds == 0:
        return 0, 0, float("-inf")

    cat_ib = pix[in_bounds]
    det_tree = cKDTree(det_xy)
    dists, _ = det_tree.query(cat_ib, k=1)
    n_matches = int((dists <= match_radius).sum())
    if n_matches == 0:
        return n_inbounds, 0, float("-inf")

    p_bg = min(
        len(det_xy) * np.pi * match_radius**2 / float(image_width * image_height),
        1.0 - 1e-9,
    )
    logodds = n_matches * np.log(1.0 / p_bg) + (n_inbounds - n_matches) * np.log(
        1.0 / (1.0 - p_bg)
    )
    return n_inbounds, n_matches, logodds


def _series_number(path: Path) -> int:
    """Extract series number from an index filename (e.g. 5205 from index-5205-43.fits)."""
    return int(path.stem.split("-")[1])


def _count_catalog_matches(
    index_path: Path,
    wcs: WCS,
    det_xy: np.ndarray,
    image_width: int,
    image_height: int,
    match_radius: float = 3.0,
) -> tuple[int, int]:
    """Count in-bounds catalog stars within match_radius pixels of any detection.

    Thin wrapper around _verify_index_logodds that discards the logodds score.
    Called after _refit_wcs_to_all_sources so the WCS is accurate enough for
    reliable pixel-space matching.

    Returns:
        (n_inbounds, n_matches)
    """
    n_inbounds, n_matches, _ = _verify_index_logodds(
        index_path, wcs, det_xy, image_width, image_height, match_radius=match_radius
    )
    return n_inbounds, n_matches


def _build_starfield_from_match(
    match: astrometry.Match, detections: list[StarInImage], fits_img: ProcessedFitsImage
) -> StarField:
    """Construct a StarField from a successful astrometry.net Match.

    Replicates solve-field --verify --tweak-order 1 --tag-all --continue:
    - WCS comes directly from match.wcs_fields (SIP-1, set by _refine_solve).
    - Catalog stars come from match.index_path (the verify-logodds winner) by
      reading ext13 catalog positions and projecting through the WCS — exactly
      what the C `--tag-all` writes to the RDLS file.

    Args:
        match (astrometry.Match): the verify winner returned by _refine_solve.
        detections (list[StarInImage]): source detections to carry forward.
        fits_img (ProcessedFitsImage): the fits image object

    Returns:
        StarField: populated StarField with fit=True and a valid WCS.
    """
    fit_wcs_astropy = WCS(
        fits.Header(fits.Card(key, value[0], value[1]) for key, value in match.wcs_fields.items()),
        relax=True,
    )
    fit_wcs = WCSModel.from_astropy_wcs(
        fit_wcs_astropy, image_shape=(fits_img.metadata.height, fits_img.metadata.width)
    )
    stars_in_space = _stars_from_match_index(
        match.index_path,
        fit_wcs_astropy,
        fits_img.metadata.width,
        fits_img.metadata.height,
    )
    logger.info(
        "astrometric_fit_stars: %d catalog stars from %s",
        len(stars_in_space),
        match.index_path.name,
    )
    return StarField(
        astrometric_fit_stars=stars_in_space,
        detections=detections,
        image_metadata=fits_img.metadata,
        fit=True,
        wcs=fit_wcs,
        wcs_status=WCSStatus.SIDEREAL_FIT_WCS,
        astrometry=ReturnAstrometryConfig.from_settings(),
    )


def _empty_starfield(detections: list[StarInImage], metadata: ImageMetadata) -> StarField:
    """Return a StarField representing a failed solve.

    Args:
        detections (list[StarInImage]): source detections to carry forward into the
            StarField.
        metadata (ImageMetadata): image metadata for the unsolved frame.

    Returns:
        StarField: StarField with fit=False and wcs=None.
    """
    return StarField(
        astrometric_fit_stars=[],
        detections=detections,
        image_metadata=metadata,
        fit=False,
        wcs=None,
        wcs_status=WCSStatus.SIDEREAL_FIT_WCS,
        astrometry=ReturnAstrometryConfig.from_settings(),
    )


def _refine_solve(
    initial_match: astrometry.Match,
    detections: list[StarInImage],
    fits_img: ProcessedFitsImage,
) -> astrometry.Match:
    """Replicate solve-field --verify --tweak-order 1 --tag-all --continue.

    Steps (matching astrometry.net/solver/onefield.c verify_wcs loop):
      1. For each scale-compatible same-cell index, compute verification logodds
         via `_solver_verify_sip_wcs_logodds` (pure-Python port of
         solver_verify_sip_wcs -> real_verify_star_lists in solver/verify.c).
      2. Pick the index with the highest bestlogodds — this is what --tag-all
         uses to write RDLS, and it matches the C behaviour exactly on our test
         collects (3be3ec47, 032c622a, ff36dda3, 80abb2e4, c1842f2a).
      3. Refine that index's WCS with SIP-1 via _solve_with_timeout, constrained
         tightly around the initial WCS (±3% scale, 1° radius) so the same
         index re-wins the quad search.

    Test stars for verify are the FULL image2xy detection list (uncapped) —
    matching the pre-refactor pipeline where Pass 2's --continue reuses the
    sources.axy from Pass 1's --use-source-extractor.

    Falls back to initial_match if no candidate verifies.
    """
    fit_wcs = WCS(
        fits.Header(fits.Card(k, v[0], v[1]) for k, v in initial_match.wcs_fields.items()),
        relax=True,
    )

    center = fit_wcs.wcs_pix2world(
        [[fits_img.metadata.width / 2, fits_img.metadata.height / 2]], 0
    )[0]
    pixel_scale_arcsec = float(np.mean(proj_plane_pixel_scales(fit_wcs))) * 3600

    # image2xy uncapped detections — matches the C verify's source list.
    verify_dets = _extract_stars_image2xy(fits_img, max_sources=10000)
    det_xy = np.array([[d.x, d.y] for d in verify_dets])

    cell = int(initial_match.index_path.stem.split("-")[-1])
    candidates = sorted(initial_match.index_path.parent.glob(f"index-*-{cell:02d}.fits"))

    best_idx: Path | None = None
    best_lo = float("-inf")
    for idx_file in candidates:
        lo = _solver_verify_sip_wcs_logodds(
            idx_file,
            fit_wcs,
            det_xy,
            fits_img.metadata.width,
            fits_img.metadata.height,
            pixel_scale_arcsec=pixel_scale_arcsec,
        )
        logger.debug("Verify logodds for %s: %.1f", idx_file.name, lo)
        if lo > best_lo:
            best_lo = lo
            best_idx = idx_file

    if best_idx is None:
        logger.warning("Sidereal WCS verify found no candidate; using initial match")
        return initial_match

    logger.info("Sidereal WCS verify winner: %s (bestlogodds=%.1f)", best_idx.name, best_lo)

    # Refine the WCS with SIP-1 on the verify winner.
    refined = _solve_with_timeout(
        stars=[[d.x, d.y] for d in detections],
        size_hint=astrometry.SizeHint(
            lower_arcsec_per_pixel=pixel_scale_arcsec * 0.97,
            upper_arcsec_per_pixel=pixel_scale_arcsec * 1.03,
        ),
        position_hint=astrometry.PositionHint(
            ra_deg=float(center[0]),
            dec_deg=float(center[1]),
            radius_deg=1.0,
        ),
        solution_parameters=_solution_parameters(sip_order=1, tune_up_logodds_threshold=14.0),
        index_files=[best_idx],
    )
    if refined is not None and refined.has_match():
        m = refined.best_match()
        logger.info(
            "Sidereal WCS refinement successful (logodds=%.1f) on %s",
            m.logodds,
            m.index_path.name,
        )
        return m

    # SIP-1 refine failed on the winning index — keep the initial WCS but
    # swap the index path so _stars_from_match_index pulls from the verify winner.
    logger.warning(
        "SIP-1 refine failed on %s; using initial WCS with verify winner", best_idx.name
    )
    initial_match.index_path = best_idx
    return initial_match


def solve_field(
    fits_img: ProcessedFitsImage,
    sip_order: int | None = None,
    min_logodds_threshold: float | None = None,
    tune_up_logodds_threshold: float | None = 14.0,
    parity: astrometry.Parity = astrometry.Parity.BOTH,
    positional_noise_pixels: float = 1.0,
    distractor_ratio: float = 0.25,
    code_tolerance_l2_distance: float = 0.01,
    minimum_quad_size_pixels: float | None = None,
    maximum_quads: int = 0,
    maximum_matches: int = 0,
    logodds_callback: Callable[[list[float]], astrometry.Action] | None = None,
    detection_match_radius_px: float = 10.0,
) -> StarField:
    """Plate-solve a preprocessed FITS image using the astrometry Python package.

    Source extraction is performed in-process using the method set by
    settings.astrometry.source_extractor ('sextractor', 'daofind', or 'image2xy').
    Index files are loaded from settings.astrometry.indices_path.

    All solver parameters default to settings values or astrometry.net library defaults
    and can be overridden per-call without touching config.

    Args:
        fits_img (ProcessedFitsImage): preprocessed FITS image to solve.
        sip_order (int | None): SIP distortion polynomial order; 0 disables SIP.
            Defaults to settings.astrometry.sip_order.
        min_logodds_threshold (float | None): minimum log-odds to accept a solution.
            Defaults to settings.astrometry.min_logodds_threshold.
        tune_up_logodds_threshold (float | None): log-odds threshold before SIP fitting
            is applied. Defaults to 14.0.
        parity (astrometry.Parity): axis orientation to search (NORMAL, FLIP, or BOTH).
            Defaults to astrometry.Parity.BOTH.
        positional_noise_pixels (float): expected star position error in pixels.
            Defaults to 1.0.
        distractor_ratio (float): expected fraction of spurious detections.
            Defaults to 0.25.
        code_tolerance_l2_distance (float): hash space tolerance. Defaults to 0.01.
        minimum_quad_size_pixels (float | None): smallest quad size; None = auto.
            Defaults to None.
        maximum_quads (int): quads to attempt; 0 = unlimited. Defaults to 0.
        maximum_matches (int): match attempts; 0 = unlimited. Defaults to 0.
        logodds_callback (Callable | None): early-stopping callback for the solver.
            Defaults to None (uses _default_logodds_callback, stopping at threshold).
        detection_match_radius_px (float): radius in pixels for matching extracted
            sources to catalog stars. Defaults to 10.0.

    Returns:
        StarField: solved StarField with fit=True and a valid WCS, or an empty StarField
            (fit=False) on timeout, extraction failure, or no match found.
    """
    logger.info("Attempting astrometric solution on image")

    try:
        detections = _extract_stars(fits_img)
        if not detections:
            logger.warning("No stars extracted, cannot solve")
            return _empty_starfield([], fits_img.metadata)

        stars = [[d.x, d.y] for d in detections]
        logger.info("Extracted %d stars for plate solving", len(stars))
        size_hint = _size_hint(fits_img.metadata.width)
        position_hint = _position_hint(
            fits_img.metadata.boresight_ra, fits_img.metadata.boresight_dec
        )
        solution_parameters = _solution_parameters(
            sip_order=sip_order,
            min_logodds_threshold=min_logodds_threshold,
            tune_up_logodds_threshold=tune_up_logodds_threshold,
            parity=parity,
            positional_noise_pixels=positional_noise_pixels,
            distractor_ratio=distractor_ratio,
            code_tolerance_l2_distance=code_tolerance_l2_distance,
            minimum_quad_size_pixels=minimum_quad_size_pixels,
            maximum_quads=maximum_quads,
            maximum_matches=maximum_matches,
            logodds_callback=logodds_callback,
        )

        solution = _solve_with_timeout(stars, size_hint, position_hint, solution_parameters)

        if solution is None or not solution.has_match():
            logger.info("No astrometric solution found")
            return _empty_starfield([], fits_img.metadata)

        logger.info("Astrometric solution found (logodds=%.1f)", solution.best_match().logodds)

        refined_match = _refine_solve(solution.best_match(), detections, fits_img)
        return _build_starfield_from_match(refined_match, detections, fits_img)
    finally:
        # The solver's index mmaps are released by now; hand their page cache back to the kernel so it
        # doesn't accumulate toward the full ~52 GiB index across a run (looks like a leak otherwise).
        if settings.astrometry.release_index_cache_after_solve:
            _release_index_page_cache()


def examine_astrometry_install() -> bool:
    """Check that the astrometry package is importable and index files are present.

    Does not raise; returns False and logs the reason on failure.

    Returns:
        bool: True if the package is available and at least one index file exists.
    """
    try:
        import astrometry as _astrometry  # noqa: F401
    except ImportError:  # pragma: no cover
        logger.exception("astrometry package is not installed")
        return False

    index_files = _index_files()
    if not index_files:
        logger.warning("No index files found at %s", settings.astrometry.indices_path)
        return False

    missing_binaries = _missing_astrometry_binaries()
    if missing_binaries:
        logger.warning(
            "Required astrometry binaries missing from PATH: %s (install astrometry.net)",
            ", ".join(missing_binaries),
        )
        return False

    logger.info(
        "astrometry package available, %d index files found at %s",
        len(index_files),
        settings.astrometry.indices_path,
    )
    return True


def require_astrometry_install() -> None:
    """Raise ValueError if the astrometry package or index files are missing.

    Raises:
        ValueError: if examine_astrometry_install() returns False.
    """
    if not examine_astrometry_install():
        raise ValueError(
            "astrometry package not installed or no index files found at "
            f"{settings.astrometry.indices_path}"
        )
