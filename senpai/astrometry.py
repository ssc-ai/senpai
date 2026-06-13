"""Adapter module bridging senpai's data models to the astroeasy plate-solving library.

Provides the same public API that the rest of senpai expects (solve_field,
test_astrometry_install, examine_indices, enforce_indices, require_astrometry_install)
while delegating all actual astrometry work to astroeasy.
"""

import logging
from pathlib import Path

import astroeasy
from astroeasy import AstrometryIndexSeries

from senpai.core.config import get_or_initialize_config
from senpai.engine.models.astrometry import ReturnAstrometryConfig, WCSModel, WCSStatus
from senpai.engine.models.starfield import StarField, StarInSpace, StarListImage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config conversion
# ---------------------------------------------------------------------------

def _build_astroeasy_config() -> astroeasy.AstrometryConfig:
    """Convert senpai's AppConfig.astrometry to an astroeasy AstrometryConfig."""
    cfg = get_or_initialize_config().astrometry
    return astroeasy.AstrometryConfig(
        indices_path=Path(cfg.indices_path),
        indices_series=AstrometryIndexSeries(cfg.indices_series),
        cpulimit_seconds=cfg.cpulimit_seconds,
        min_width_degrees=cfg.min_width_degrees,
        max_width_degrees=cfg.max_width_degrees,
        tweak_order=cfg.tweak_order,
        max_sources=cfg.max_sources,
        min_sources_for_attempt=cfg.min_sources_for_attempt,
        docker_image=cfg.docker_image,
    )


# ---------------------------------------------------------------------------
# Model conversion helpers
# ---------------------------------------------------------------------------

def _sources_to_detections(sources: StarListImage) -> tuple[list[astroeasy.Detection], astroeasy.ImageMetadata]:
    """Convert senpai StarListImage → astroeasy Detection list + ImageMetadata."""
    detections = [
        astroeasy.Detection(x=s.x, y=s.y, flux=s.counts)
        for s in sources.detections
    ]
    metadata = astroeasy.ImageMetadata(
        width=sources.image_metadata.width,
        height=sources.image_metadata.height,
        boresight_ra=sources.image_metadata.boresight_ra,
        boresight_dec=sources.image_metadata.boresight_dec,
    )
    return detections, metadata


def _wcsmodel_to_wcsresult(wcs: WCSModel) -> astroeasy.WCSResult:
    """Convert senpai WCSModel → astroeasy WCSResult via FITS header round-trip."""
    from astropy.io import fits as astropy_fits

    header_dict = {k: v for k, v in wcs.model_dump().items() if v is not None}
    header_dict["IMAGEW"] = header_dict.pop("NAXIS1", wcs.NAXIS1)
    header_dict["IMAGEH"] = header_dict.pop("NAXIS2", wcs.NAXIS2)

    # Build a FITS header so WCSResult.from_fits_header can parse it
    hdr = astropy_fits.Header()
    for key, value in header_dict.items():
        hdr[key] = value

    return astroeasy.WCSResult.from_fits_header(hdr)


def _wcsresult_to_wcsmodel(wcs_result: astroeasy.WCSResult) -> WCSModel:
    """Convert astroeasy WCSResult → senpai WCSModel via raw FITS header."""
    import contextlib

    from astropy.io import fits as astropy_fits

    hdr = astropy_fits.Header()
    for key, value in wcs_result.raw_header.items():
        if key and not key.startswith("COMMENT") and not key.startswith("HISTORY"):
            with contextlib.suppress(ValueError, KeyError):
                hdr[key] = value

    hdu = astropy_fits.PrimaryHDU(header=hdr)
    return WCSModel.from_astrometrydotnet(hdu)


def _solve_result_to_starfield(
    result: astroeasy.SolveResult,
    sources: StarListImage,
) -> StarField:
    """Convert astroeasy SolveResult → senpai StarField."""
    config = get_or_initialize_config()

    fit_wcs = _wcsresult_to_wcsmodel(result.wcs) if result.wcs else None

    astrometric_fit_stars = [
        StarInSpace(
            ra=m.ra,
            dec=m.dec,
            magnitude=m.magnitude,
            catalog=m.catalog,
            catalog_id=m.catalog_id,
            x=m.x,
            y=m.y,
        )
        for m in result.matched_stars
    ] if result.matched_stars else []

    return StarField(
        astrometric_fit_stars=astrometric_fit_stars or None,
        detections=sources.detections,
        image_metadata=sources.image_metadata,
        fit=result.success,
        wcs=fit_wcs,
        wcs_status=WCSStatus.SIDEREAL_FIT_WCS if result.success else WCSStatus.NO_WCS,
        astrometry=ReturnAstrometryConfig.from_app_config(config),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_field(sources: StarListImage, wcs: WCSModel | None = None) -> StarField:
    """Solve astrometry for detected sources.

    Args:
        sources: Detected sources with pixel coordinates and image metadata.
        wcs: Optional existing WCS to verify/refine.

    Returns:
        StarField with WCS solution, matched stars, and fit status.
    """
    logger.info("attempting astrometric solution on %i sources", len(sources.detections))
    config = get_or_initialize_config()

    if len(sources.detections) < config.astrometry.min_sources_for_attempt:
        logger.error(
            f"{len(sources.detections)} [less than {config.astrometry.min_sources_for_attempt}] "
            "detections found, skipping astrometry"
        )
        return StarField(
            wcs=None,
            detections=sources.detections,
            image_metadata=sources.image_metadata,
        )

    # solver_mode dispatch (see astroeasy docs/catalog-native-solving-roadmap.md §0.2):
    # 'dotnet' is the original astrometry.net path; 'tetra3'/'chain' run the
    # catalog-native cascade (with astrometry.net as chain's backstop).
    if config.astrometry.solver_mode != "dotnet":
        return _solve_field_cascade(sources, wcs, config)

    ae_config = _build_astroeasy_config()
    detections, metadata = _sources_to_detections(sources)
    existing_wcs = _wcsmodel_to_wcsresult(wcs) if wcs is not None else None

    result = astroeasy.solve_field(detections, metadata, ae_config, existing_wcs)
    return _solve_result_to_starfield(result, sources)


def _solve_field_cascade(sources: StarListImage, wcs: WCSModel | None, config) -> StarField:
    """Run the astroeasy escalation cascade (solver_mode 'tetra3'/'chain').

    'tetra3' = native tiers only (T0 refine + T1 pattern match, no
    astrometry.net required); 'chain' = native tiers with the existing
    astrometry.net path as the T3 backstop.
    """
    from astroeasy import cascade

    a = config.astrometry
    mode = a.solver_mode
    fast = a.fast_solve

    # The catalog cone source: explicit fast_solve.mirror_dir, else reuse the
    # already-configured local Gaia mirror from star_catalog.
    mirror_dir = fast.mirror_dir
    if mirror_dir is None:
        sc = getattr(config, "star_catalog", None)
        if sc is not None and getattr(sc, "type", None) == "gaia_local":
            mirror_dir = sc.path
    if mirror_dir is None:
        logger.warning(
            "solver_mode=%s but no catalog mirror configured "
            "(astrometry.fast_solve.mirror_dir or star_catalog type gaia_local) — "
            "native tiers will be skipped", mode,
        )

    if fast.sensor_profile:
        profile = cascade.SensorProfile.from_yaml(fast.sensor_profile)
        if fast.tetra3_db_path:
            profile.tetra3_db_path = fast.tetra3_db_path
    else:
        profile = cascade.SensorProfile(
            sensor_id="senpai",
            scale_bounds_degrees=(a.min_width_degrees, a.max_width_degrees),
            sip_order=a.tweak_order,
            tetra3_db_path=fast.tetra3_db_path,
        )

    detections, metadata = _sources_to_detections(sources)
    prior_wcs = _wcsmodel_to_wcsresult(wcs) if wcs is not None else None
    tiers = ("T0", "T1") if mode == "tetra3" else ("T0", "T1", "T3")
    dotnet_config = _build_astroeasy_config() if mode == "chain" else None

    result = cascade.solve(
        detections, metadata,
        profile=profile, mirror_dir=mirror_dir,
        dotnet_config=dotnet_config, prior_wcs=prior_wcs, tiers=tiers,
    )
    logger.info(
        "cascade (%s): %s — attempts: %s", mode,
        f"solved at {result.tier}" if result.tier else "all tiers failed",
        [(t.tier, t.status, f"{t.duration_ms:.0f}ms") for t in result.attempts],
    )
    return _solve_result_to_starfield(result.solve, sources)


def test_astrometry_install() -> bool:
    """Test if astrometry.net is properly installed."""
    config = get_or_initialize_config()
    return astroeasy.test_install(docker_image=config.astrometry.docker_image)


def require_astrometry_install():
    """Raise if astrometry.net is not installed."""
    if not test_astrometry_install():
        raise ValueError("Astrometry.net is not installed or not in PATH")


def examine_indices() -> bool:
    """Check that configured index files are complete and valid."""
    config = get_or_initialize_config()
    return astroeasy.examine_indices(
        indices_path=config.astrometry.indices_path,
        series=AstrometryIndexSeries(config.astrometry.indices_series),
    )


def enforce_indices():
    """Raise if configured index files are missing or incomplete."""
    config = get_or_initialize_config()
    if not examine_indices():
        raise RuntimeError(
            f"Astrometry indices for {config.astrometry.indices_series} are missing or incomplete"
        )
