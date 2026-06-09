import logging
import math
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from astropy.wcs import WCS

import senpai.catalog.sstr7 as sstr7
import senpai.catalog.sdss as sdss
import senpai.catalog.gaia as gaia
from senpai.catalog.constants import CatalogType
from senpai.core.config import get_config, get_or_initialize_config
from senpai.engine.models.astrometry import WCSModel
from senpai.engine.models.starfield import ImageMetadata, StarInSpace, StarListSpace

logger = logging.getLogger(__name__)


def _validate_catalog_coverage(
    stars_from_catalog: List[Dict[str, Any]],
    star_list: List[StarInSpace],
    pixel_width: float,
    pixel_height: float,
    catalog_type: str,
    min_ra: float,
    max_ra: float,
    min_dec: float,
    max_dec: float,
) -> None:
    """Validate catalog coverage and warn if sparse or empty.

    Args:
        stars_from_catalog: Raw stars returned from catalog query
        star_list: Stars within image bounds after WCS transformation
        pixel_width: Image width in pixels
        pixel_height: Image height in pixels
        catalog_type: Name of catalog being used
        min_ra: Minimum RA of query bounds (degrees)
        max_ra: Maximum RA of query bounds (degrees)
        min_dec: Minimum DEC of query bounds (degrees)
        max_dec: Maximum DEC of query bounds (degrees)
    """
    # Check if catalog query returned no stars
    if len(stars_from_catalog) == 0:
        logger.error(
            f"{catalog_type} catalog query returned NO stars for "
            f"RA=[{min_ra:.3f}, {max_ra:.3f}]°, DEC=[{min_dec:.3f}, {max_dec:.3f}]°. "
            f"This may indicate: (1) catalog has no coverage in this region, "
            f"(2) magnitude limits are too restrictive, or (3) query bounds are invalid."
        )
        return

    # Check if very few stars returned relative to field size.
    # Handle RA wraparound (e.g., RA range 359°-1° should be 2°, not 358°)
    ra_span = (max_ra - min_ra) % 360
    if ra_span > 180:
        ra_span = 360 - ra_span
    # Apply cos(dec) correction for RA → true angular span
    mean_dec = (min_dec + max_dec) / 2
    ra_span_corrected = ra_span * np.cos(np.radians(mean_dec))
    dec_span = abs(max_dec - min_dec)
    field_area_deg2 = abs(ra_span_corrected * dec_span)
    stars_per_deg2 = len(stars_from_catalog) / max(field_area_deg2, 1e-6)

    # Typical star densities: ~1000-10000 stars/deg² at magnitude ~20
    # Warn if density is very low (< 10 stars/deg²)
    if (
        stars_per_deg2 < 10 and field_area_deg2 > 0.01
    ):  # Only warn for fields > 0.01 deg²
        logger.warning(
            f"{catalog_type} catalog returned very sparse coverage: "
            f"{len(stars_from_catalog)} stars ({stars_per_deg2:.1f} stars/deg²) "
            f"for field {field_area_deg2:.4f} deg². "
            f"This may indicate incomplete catalog coverage in this region."
        )

    # Check if stars are clustered in only part of the field
    if len(star_list) > 0:
        # Divide field into a grid and check coverage
        grid_size = 4  # 4x4 grid = 16 cells
        x_bins = np.linspace(0, pixel_width, grid_size + 1)
        y_bins = np.linspace(0, pixel_height, grid_size + 1)

        # Count stars in each grid cell
        grid_coverage = np.zeros((grid_size, grid_size), dtype=int)
        for star in star_list:
            if star.x is not None and star.y is not None:
                x_idx = np.digitize(star.x, x_bins) - 1
                y_idx = np.digitize(star.y, y_bins) - 1
                x_idx = np.clip(x_idx, 0, grid_size - 1)
                y_idx = np.clip(y_idx, 0, grid_size - 1)
                grid_coverage[y_idx, x_idx] += 1

        # Check how many grid cells have stars
        cells_with_stars = np.sum(grid_coverage > 0)
        total_cells = grid_size * grid_size
        coverage_fraction = cells_with_stars / total_cells

        # Warn if stars are clustered in less than 25% of the field
        if coverage_fraction < 0.25 and len(star_list) > 5:
            logger.warning(
                f"{catalog_type} catalog stars are clustered in only "
                f"{cells_with_stars}/{total_cells} grid cells ({coverage_fraction*100:.0f}% coverage). "
                f"This may indicate incomplete catalog coverage or query bounds issues."
            )

    # Check if many catalog stars were filtered out (outside image bounds)
    if len(stars_from_catalog) > 0:
        in_bounds_fraction = len(star_list) / len(stars_from_catalog)
        if in_bounds_fraction < 0.1 and len(stars_from_catalog) > 10:
            logger.warning(
                f"Only {len(star_list)}/{len(stars_from_catalog)} ({in_bounds_fraction*100:.0f}%) "
                f"{catalog_type} catalog stars are within image bounds. "
                f"This may indicate WCS transformation issues or query bounds are too large."
            )


def _make_wcs_hashable(wcs: WCSModel) -> tuple:
    """Convert WCS model to a hashable tuple for caching purposes."""
    astropy_wcs = wcs.to_astropy_wcs()
    header = astropy_wcs.to_header()

    # Extract key parameters that uniquely identify the WCS
    hashable_components = [
        header.get("CRVAL1", 0),
        header.get("CRVAL2", 0),
        header.get("CRPIX1", 0),
        header.get("CRPIX2", 0),
        header.get("PC1_1", header.get("CD1_1", 1)),
        header.get("PC1_2", header.get("CD1_2", 0)),
        header.get("PC2_1", header.get("CD2_1", 0)),
        header.get("PC2_2", header.get("CD2_2", 1)),
        header.get("CDELT1", 1.0),
        header.get("CDELT2", 1.0),
        wcs.NAXIS1,
        wcs.NAXIS2,
    ]
    return tuple(hashable_components)


@lru_cache(maxsize=10000)  # Cache up to 10000 distinct queries
def _query_catalog_sstr7_cached(
    wcs_tuple: tuple,
    catalog_path: str,
    faint_lim: int | None,
    bright_lim: int | None,
    proper_motion_date_timestamp: float | None,
    max_stars: int | None,
) -> tuple:
    """Cached version of query_catalog_sstr7 that takes hashable arguments."""
    # Reconstruct WCS from tuple components
    header = {
        "WCSAXES": 2,
        "CRVAL1": wcs_tuple[0],
        "CRVAL2": wcs_tuple[1],
        "CRPIX1": wcs_tuple[2],
        "CRPIX2": wcs_tuple[3],
        "PC1_1": wcs_tuple[4],
        "PC1_2": wcs_tuple[5],
        "PC2_1": wcs_tuple[6],
        "PC2_2": wcs_tuple[7],
        "CDELT1": wcs_tuple[8],
        "CDELT2": wcs_tuple[9],
        "NAXIS1": wcs_tuple[10],
        "NAXIS2": wcs_tuple[11],
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
    }

    astropy_wcs = WCS(header)
    wcs = WCSModel.from_astropy_wcs(astropy_wcs)

    proper_motion_date = (
        datetime.fromtimestamp(proper_motion_date_timestamp)
        if proper_motion_date_timestamp
        else None
    )

    fov_width, fov_height, pixel_width, pixel_height = wcs.get_fov_and_dimensions()

    # Get center coordinates
    header = astropy_wcs.to_header()
    center_ra, center_dec = astropy_wcs.wcs_pix2world(
        [[header["CRPIX1"], header["CRPIX2"]]], 0
    )[0]

    # Extract rotation from WCS
    rotation = 0.0
    if "PC1_1" in header and "PC1_2" in header:
        # Calculate rotation from PC matrix
        rotation = np.degrees(np.arctan2(header["PC1_2"], header["PC1_1"]))
    elif "CROTA2" in header:
        rotation = header["CROTA2"]

    # Use the new function that handles rotation
    stars_from_catalog = sstr7.query_by_los_radec_with_rotation(
        fov_height,
        fov_width,
        center_ra,
        center_dec,
        rotation=rotation,
        rootPath=catalog_path,
        faint_lim=faint_lim,
        bright_lim=bright_lim,
        safety_margin=0.2,  # Add 20% safety margin to ensure complete coverage
    )

    if max_stars is not None and len(stars_from_catalog) > max_stars:
        # Sort stars from brightest to dimmest (lowest to highest magnitude)
        stars_from_catalog = sorted(stars_from_catalog, key=lambda star: star["mv"])
        stars_from_catalog = stars_from_catalog[:max_stars]

    if proper_motion_date is not None:
        logger.info(f"Applying proper motion for {proper_motion_date}")
        # Calculate seconds elapsed since J2000 (2000-01-01)
        j2000 = datetime(2000, 1, 1)
        seconds_elapsed = (proper_motion_date - j2000).total_seconds()
        for star in stars_from_catalog:
            star["ra"] += star["ra_pm"] * seconds_elapsed
            star["dec"] += star["dec_pm"] * seconds_elapsed

    # Vectorize the coordinate transformation
    ra_deg = np.rad2deg([star["ra"] for star in stars_from_catalog])
    dec_deg = np.rad2deg([star["dec"] for star in stars_from_catalog])
    coords = np.column_stack((ra_deg, dec_deg))
    pixel_coords = astropy_wcs.wcs_world2pix(coords, 0)

    star_list = []
    for i, star in enumerate(stars_from_catalog):
        xf, yf = pixel_coords[i]
        ra = ra_deg[i]
        dec = dec_deg[i]

        # Only check if star is within image bounds (magnitude filtering done at catalog level)
        if xf > 0 and xf < pixel_width and yf > 0 and yf < pixel_height:
            # Ensure magnitudes dict is always populated if magnitude exists
            # The catalog should populate magnitudes, but ensure it's never None/empty if magnitude exists
            magnitudes = star.get("magnitudes")
            if magnitudes is None:
                magnitudes = {}

            # If magnitude exists but magnitudes dict is empty, populate it with the primary magnitude
            # This ensures magnitudes dict is never empty when magnitude is set
            if star["mv"] is not None and star["mv"] < 32 and len(magnitudes) == 0:
                # Use a generic name since we don't know which filter was used
                magnitudes["Primary"] = float(star["mv"])

            star_list.append(
                StarInSpace(
                    ra=ra,
                    dec=dec,
                    x=xf,
                    y=yf,
                    magnitude=star["mv"],
                    magnitudes=magnitudes if len(magnitudes) > 0 else None,
                    catalog=star["catalog"],
                )
            )

    return star_list, ImageMetadata(
        wcs=wcs,
        width=pixel_width,
        height=pixel_height,
        boresight_dec=center_dec,
        boresight_ra=center_ra,
    )


def query_catalog_sstr7(
    wcs: WCSModel,
    catalog_path: str | Path,
    faint_lim: int | None = None,
    bright_lim: int | None = None,
    proper_motion_date: datetime | None = None,
    max_stars: int | None = None,
) -> StarListSpace:
    """Query the SSTR7 star catalog with caching support."""
    # Convert inputs to hashable types
    wcs_tuple = _make_wcs_hashable(wcs)
    catalog_path_str = str(catalog_path)
    proper_motion_timestamp = (
        proper_motion_date.timestamp() if proper_motion_date else None
    )

    logger.info("building SENPAI catalog")
    start_time = time.time()

    # Call cached function (uses simplified WCS for FOV calculation and catalog querying)
    # Note: Pixel coordinates from the cached function use the linear WCS without SIP
    # distortion (this keeps the cache key SIP-independent and the query fast). Callers
    # that need distortion-correct pixel positions should use
    # query_catalog(..., apply_sip=True), which re-projects the returned stars.
    star_list, image_metadata = _query_catalog_sstr7_cached(
        wcs_tuple,
        catalog_path_str,
        faint_lim,
        bright_lim,
        proper_motion_timestamp,
        max_stars,
    )

    logger.info(
        f"Found {len(star_list)} stars in catalog in {time.time() - start_time:.2f} seconds"
    )

    # Calculate bounds for validation
    astropy_wcs = wcs.to_astropy_wcs()
    fov_width, fov_height, pixel_width, pixel_height = wcs.get_fov_and_dimensions()
    corners_pix = np.array(
        [
            [0, 0],
            [pixel_width, 0],
            [pixel_width, pixel_height],
            [0, pixel_height],
        ]
    )
    corners_world = astropy_wcs.wcs_pix2world(corners_pix, 0)
    ra_corners = corners_world[:, 0]
    dec_corners = corners_world[:, 1]
    safety_margin = 0.2
    ra_range = np.max(ra_corners) - np.min(ra_corners)
    dec_range = np.max(dec_corners) - np.min(dec_corners)
    ra_expansion = ra_range * safety_margin / 2
    dec_expansion = dec_range * safety_margin / 2
    min_ra = np.min(ra_corners) - ra_expansion
    max_ra = np.max(ra_corners) + ra_expansion
    min_dec = np.min(dec_corners) - dec_expansion
    max_dec = np.max(dec_corners) + dec_expansion

    # Get raw stars from catalog for validation (need to reconstruct from cached function)
    # For SSTRC7, we'll validate based on final star_list since raw stars aren't easily accessible
    # Create a dummy list for validation (validation will still work on star_list)
    _validate_catalog_coverage(
        stars_from_catalog=star_list,  # Use star_list as proxy for SSTRC7
        star_list=star_list,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        catalog_type="SSTRC7",
        min_ra=min_ra,
        max_ra=max_ra,
        min_dec=min_dec,
        max_dec=max_dec,
    )

    return StarListSpace(stars=star_list, image_metadata=image_metadata)


def query_catalog_sdss(
    wcs: WCSModel,
    faint_lim: int | None = None,
    bright_lim: int | None = None,
    proper_motion_date: datetime | None = None,
    max_stars: int | None = None,
) -> StarListSpace:

    astropy_wcs = wcs.to_astropy_wcs()

    fov_width, fov_height, pixel_width, pixel_height = wcs.get_fov_and_dimensions()

    # Get center coordinates
    header = astropy_wcs.to_header()
    center_ra, center_dec = astropy_wcs.wcs_pix2world(
        [[header["CRPIX1"], header["CRPIX2"]]], 0
    )[0]

    logger.info("Querying SDSS catalog online")
    start_time = time.time()

    # Use WCS to transform image corners to world coordinates
    # This properly accounts for all WCS transformations including rotation, distortion, etc.
    corners_pix = np.array(
        [
            [0, 0],  # bottom-left
            [pixel_width, 0],  # bottom-right
            [pixel_width, pixel_height],  # top-right
            [0, pixel_height],  # top-left
        ]
    )

    # Transform corners to world coordinates (RA/DEC in degrees)
    corners_world = astropy_wcs.wcs_pix2world(corners_pix, 0)
    ra_corners = corners_world[:, 0]
    dec_corners = corners_world[:, 1]

    # Apply safety margin by expanding the bounding box
    safety_margin = 0.2
    ra_range = np.max(ra_corners) - np.min(ra_corners)
    dec_range = np.max(dec_corners) - np.min(dec_corners)
    ra_expansion = ra_range * safety_margin / 2
    dec_expansion = dec_range * safety_margin / 2

    min_ra = np.min(ra_corners) - ra_expansion
    max_ra = np.max(ra_corners) + ra_expansion
    min_dec = np.min(dec_corners) - dec_expansion
    max_dec = np.max(dec_corners) + dec_expansion

    # Query SDSS using the actual WCS bounds
    stars_from_catalog = sdss.query_by_ra_dec_bounds(
        min_ra=min_ra,
        max_ra=max_ra,
        min_dec=min_dec,
        max_dec=max_dec,
        faint_lim=faint_lim,
        bright_lim=bright_lim,
    )

    # Limit number of stars if requested
    if max_stars is not None and len(stars_from_catalog) > max_stars:
        # Sort stars from brightest to dimmest (lowest to highest magnitude)
        stars_from_catalog = sorted(stars_from_catalog, key=lambda star: star["mv"])
        stars_from_catalog = stars_from_catalog[:max_stars]

    # Apply proper motion if date is provided
    if proper_motion_date is not None:
        logger.info(f"Applying proper motion for {proper_motion_date}")
        # Calculate seconds elapsed since J2000 (2000-01-01)
        j2000 = datetime(2000, 1, 1)
        seconds_elapsed = (proper_motion_date - j2000).total_seconds()
        for star in stars_from_catalog:
            star["ra"] += star["ra_pm"] * seconds_elapsed
            star["dec"] += star["dec_pm"] * seconds_elapsed

    # Vectorize the coordinate transformation
    ra_deg = np.rad2deg([star["ra"] for star in stars_from_catalog])
    dec_deg = np.rad2deg([star["dec"] for star in stars_from_catalog])
    coords = np.column_stack((ra_deg, dec_deg))
    pixel_coords = astropy_wcs.wcs_world2pix(coords, 0)

    star_list = []
    for i, star in enumerate(stars_from_catalog):
        xf, yf = pixel_coords[i]
        ra = ra_deg[i]
        dec = dec_deg[i]

        # Only include stars within image bounds
        if xf > 0 and xf < pixel_width and yf > 0 and yf < pixel_height:
            # Ensure magnitudes dict is always populated if magnitude exists
            magnitudes = star.get("magnitudes")
            if magnitudes is None:
                magnitudes = {}
            # If magnitude exists but magnitudes dict is empty, populate it with the primary magnitude
            if star["mv"] is not None and star["mv"] < 32 and len(magnitudes) == 0:
                magnitudes["Primary"] = float(star["mv"])

            star_list.append(
                StarInSpace(
                    ra=ra,
                    dec=dec,
                    x=xf,
                    y=yf,
                    magnitude=star["mv"],
                    magnitudes=magnitudes if len(magnitudes) > 0 else None,
                    catalog=star["catalog"],
                    catalog_id=star.get("objid"),
                )
            )

    logger.info(
        f"Found {len(star_list)} stars in SDSS catalog in {time.time() - start_time:.2f} seconds"
    )

    # Validate catalog coverage
    _validate_catalog_coverage(
        stars_from_catalog=stars_from_catalog,
        star_list=star_list,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        catalog_type="SDSS",
        min_ra=min_ra,
        max_ra=max_ra,
        min_dec=min_dec,
        max_dec=max_dec,
    )

    return StarListSpace(
        stars=star_list,
        image_metadata=ImageMetadata(
            wcs=wcs,
            width=pixel_width,
            height=pixel_height,
            boresight_dec=center_dec,
            boresight_ra=center_ra,
        ),
    )


# Sky-region cache for the expensive online Gaia query. The per-WCS lru_cache
# below never hits within a batch (every frame shift / WCS-refinement pass nudges
# CRVAL/CRPIX, so the WCS tuple — and thus the key — changes), which meant 2-3
# full-field online queries per batch over almost-identical sky. This caches the
# raw sky stars (RA/Dec/mag) per (faint,bright) as growing-coverage regions:
#   * a box already inside a region's coverage  -> reuse, no online query;
#   * a box that overlaps a region              -> fetch ONLY the new sliver(s)
#     (the box-difference of the grown bbox and the existing coverage — an L of
#     up to 2 rects for a diagonal RA+Dec shift), merge, and grow the coverage;
#   * a box overlapping nothing                 -> fresh query, new region.
# Each fetch is padded a touch so the sub-degree intra-batch jitter (frame shift +
# WCS-refinement passes) lands inside coverage and costs no query at all, while
# real pointing changes only pull the genuinely-new strip. Pixel projection still
# happens per actual WCS downstream, so positions stay exact; results are filtered
# back to the requested box so callers see exactly the box they asked for. RA-wrap
# fields bypass the cache.
_GAIA_SKY_CACHE: list[dict] = []
_GAIA_SKY_PAD_DEG = 0.1  # absorb intra-batch jitter (shift+refine) into coverage
_GAIA_SKY_CACHE_MAX = 64  # regions
# Total-star bound across all regions. Region *coverage* grows without limit
# as a night's pointings accumulate (a worker that has seen hundreds of
# batches can hold multi-GB of star dicts — observed as an OOM kill at 8.7 GB
# RSS on a full-night run). Evict least-recently-used whole regions once the
# total crosses this; ~1M dicts ≈ 300 MB per worker.
_GAIA_SKY_CACHE_MAX_STARS = 1_000_000


def _trim_sky_cache() -> None:
    """Evict least-recently-used regions until the total star count fits.
    Callers move the active region to the end of the list first (LRU touch)."""
    total = sum(len(r["stars"]) for r in _GAIA_SKY_CACHE)
    while len(_GAIA_SKY_CACHE) > 1 and (
        total > _GAIA_SKY_CACHE_MAX_STARS or len(_GAIA_SKY_CACHE) > _GAIA_SKY_CACHE_MAX
    ):
        dropped = _GAIA_SKY_CACHE.pop(0)
        total -= len(dropped["stars"])
        logger.info(
            "Gaia sky-cache EVICT: region %.0f deg² / %d stars (total now %d)",
            (dropped["box"][1] - dropped["box"][0])
            * (dropped["box"][3] - dropped["box"][2]),
            len(dropped["stars"]), total,
        )


def _box_overlap(a, b) -> bool:
    return not (b[0] >= a[1] or b[1] <= a[0] or b[2] >= a[3] or b[3] <= a[2])


def _box_contains(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] >= inner[1]
            and outer[2] <= inner[2] and outer[3] >= inner[3])


def _box_difference_strips(C, U):
    """Rectangles tiling U \\ C, given U ⊇ C (disjoint; ≤4, typically 1-2 for a
    shifted box): left/right full-height strips + top/bottom strips over C's RA."""
    rmn, rmx, dmn, dmx = U
    crmn, crmx, cdmn, cdmx = C
    eps = 1e-9
    strips = []
    if crmn - rmn > eps:
        strips.append((rmn, crmn, dmn, dmx))
    if rmx - crmx > eps:
        strips.append((crmx, rmx, dmn, dmx))
    if cdmn - dmn > eps:
        strips.append((crmn, crmx, dmn, cdmn))
    if dmx - cdmx > eps:
        strips.append((crmn, crmx, cdmx, dmx))
    return strips


def _sky_dedup_key(s):
    sid = s.get("source_id")
    return sid if sid is not None else (round(s["ra"], 8), round(s["dec"], 8))


def _query_gaia_sky(
    min_ra: float, max_ra: float, min_dec: float, max_dec: float,
    faint_lim: float | None, bright_lim: float | None,
) -> list[dict[str, Any]]:
    """Online Gaia query with a growing sky-region cache (see note above).

    Only the sky area not already cached is fetched online; the result is the raw
    star dicts within the requested box. The dicts are never mutated downstream."""
    key_fb = (faint_lim, bright_lim)
    B = (min_ra, max_ra, min_dec, max_dec)

    def _online(box):
        # Swap online TAP for the local mirror when configured (gaia_local). The
        # sliver cache above still wraps this, so local reads get deduped too.
        sc = get_or_initialize_config().star_catalog
        if getattr(sc, "type", "gaia") == "gaia_local":
            from senpai.catalog import gaia_local

            return gaia_local.query_by_ra_dec_bounds(
                box[0], box[1], box[2], box[3],
                faint_lim=faint_lim, bright_lim=bright_lim, mirror_dir=sc.path,
            )
        return gaia.query_by_ra_dec_bounds(
            min_ra=box[0], max_ra=box[1], min_dec=box[2], max_dec=box[3],
            faint_lim=faint_lim, bright_lim=bright_lim,
        )

    def _within_B(stars):
        # Trim to the requested box so callers get exactly what a fresh query of B
        # would return (radians stored on each star).
        lo_ra, hi_ra = np.deg2rad(min_ra), np.deg2rad(max_ra)
        lo_dec, hi_dec = np.deg2rad(min_dec), np.deg2rad(max_dec)
        return [s for s in stars
                if lo_ra <= s["ra"] <= hi_ra and lo_dec <= s["dec"] <= hi_dec]

    # RA-wrap fields: skip the cache rather than handle 0/360 seam geometry.
    if min_ra < 0.0 or max_ra > 360.0 or min_ra >= max_ra:
        return _online(B)

    pad = _GAIA_SKY_PAD_DEG
    Bp = (min_ra - pad, max_ra + pad,
          max(-90.0, min_dec - pad), min(90.0, max_dec + pad))

    for region in _GAIA_SKY_CACHE:
        if region["fb"] != key_fb or not _box_overlap(region["box"], B):
            continue
        # LRU touch: keep the active region at the end so eviction starts
        # from regions that haven't been used recently.
        _GAIA_SKY_CACHE.remove(region)
        _GAIA_SKY_CACHE.append(region)
        C = region["box"]
        if _box_contains(C, B):
            logger.info("Gaia sky-cache HIT (within coverage); no online query")
            return _within_B(region["stars"])
        # Partial overlap: grow coverage to include the padded box, fetch only the
        # new sliver(s), merge (dedup), keep the overlap.
        U = (min(C[0], Bp[0]), max(C[1], Bp[1]), min(C[2], Bp[2]), max(C[3], Bp[3]))
        strips = _box_difference_strips(C, U)
        seen = {_sky_dedup_key(s) for s in region["stars"]}
        added = 0
        for st in strips:
            for s in _online(st):
                k = _sky_dedup_key(s)
                if k not in seen:
                    seen.add(k)
                    region["stars"].append(s)
                    added += 1
        region["box"] = U
        logger.info(
            "Gaia sky-cache PARTIAL: %d sliver(s), +%d stars (reused overlap); "
            "coverage now [%.3f,%.3f]x[%.3f,%.3f]",
            len(strips), added, U[0], U[1], U[2], U[3],
        )
        _trim_sky_cache()
        return _within_B(region["stars"])

    # No overlapping region: fresh (padded) query, start a new region.
    stars = _online(Bp)
    _GAIA_SKY_CACHE.append({"fb": key_fb, "box": Bp, "stars": list(stars)})
    _trim_sky_cache()
    return _within_B(stars)


# Small maxsize ON PURPOSE: the WCS-tuple key changes on every frame shift /
# refinement nudge, so entries almost never re-hit across batches — at 50000
# this was a pure accumulator pinning each query's full star list (tens of MB
# per entry, and it kept evicted sky-cache regions alive through shared dict
# refs) — a worker-process memory leak that OOM-killed full-night runs. A
# handful of entries covers any genuine same-WCS re-query within a frame.
@lru_cache(maxsize=8)
def _query_catalog_gaia_cached(
    wcs_tuple: tuple,
    faint_lim: int | None,
    bright_lim: int | None,
    proper_motion_date_timestamp: float | None,
    max_stars: int | None,
) -> tuple[list[dict[str, Any]], ImageMetadata]:
    """Cached Gaia query using a hashable WCS representation."""

    # Reconstruct WCS from tuple components
    header = {
        "WCSAXES": 2,
        "CRVAL1": wcs_tuple[0],
        "CRVAL2": wcs_tuple[1],
        "CRPIX1": wcs_tuple[2],
        "CRPIX2": wcs_tuple[3],
        "PC1_1": wcs_tuple[4],
        "PC1_2": wcs_tuple[5],
        "PC2_1": wcs_tuple[6],
        "PC2_2": wcs_tuple[7],
        "CDELT1": wcs_tuple[8],
        "CDELT2": wcs_tuple[9],
        "NAXIS1": wcs_tuple[10],
        "NAXIS2": wcs_tuple[11],
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
    }

    astropy_wcs = WCS(header)
    wcs = WCSModel.from_astropy_wcs(astropy_wcs)

    proper_motion_date = (
        datetime.fromtimestamp(proper_motion_date_timestamp)
        if proper_motion_date_timestamp
        else None
    )

    fov_width, fov_height, pixel_width, pixel_height = wcs.get_fov_and_dimensions()

    # Get center coordinates
    header = astropy_wcs.to_header()
    center_ra, center_dec = astropy_wcs.wcs_pix2world(
        [[header["CRPIX1"], header["CRPIX2"]]], 0
    )[0]

    # Use WCS to transform image corners to world coordinates
    corners_pix = np.array(
        [
            [0, 0],
            [pixel_width, 0],
            [pixel_width, pixel_height],
            [0, pixel_height],
        ]
    )
    corners_world = astropy_wcs.wcs_pix2world(corners_pix, 0)
    ra_corners = corners_world[:, 0]
    dec_corners = corners_world[:, 1]

    # Expand the corner bounding box by a tiny pad. The box already
    # circumscribes the frame corners, so only a small margin is needed to cover
    # WCS/SIP edge distortion; the old 20% pulled a box ~2.5x the FoV (mostly
    # off-frame stars that get filtered out anyway), making the online query
    # much larger than necessary.
    safety_margin = 0.03
    ra_range = np.max(ra_corners) - np.min(ra_corners)
    dec_range = np.max(dec_corners) - np.min(dec_corners)
    ra_expansion = ra_range * safety_margin / 2
    dec_expansion = dec_range * safety_margin / 2

    min_ra = np.min(ra_corners) - ra_expansion
    max_ra = np.max(ra_corners) + ra_expansion
    min_dec = np.min(dec_corners) - dec_expansion
    max_dec = np.max(dec_corners) + dec_expansion

    # Query Gaia via the sky-region cache: a shifted/re-refined frame at (nearly)
    # the same pointing reuses the prior online fetch instead of re-querying the
    # overlapping field. Returns a superset of the box; the in-frame pixel filter
    # below trims it.
    stars_from_catalog = _query_gaia_sky(
        min_ra, max_ra, min_dec, max_dec, faint_lim, bright_lim
    )

    # Limit number of stars if requested
    if max_stars is not None and len(stars_from_catalog) > max_stars:
        stars_from_catalog = sorted(stars_from_catalog, key=lambda star: star["mv"])
        stars_from_catalog = stars_from_catalog[:max_stars]

    # Proper motion + radian->deg on LOCAL arrays — never mutate the cached star
    # dicts in place, or a later cache reuse would double-apply proper motion.
    ra_rad = np.array([star["ra"] for star in stars_from_catalog], dtype=float)
    dec_rad = np.array([star["dec"] for star in stars_from_catalog], dtype=float)
    if proper_motion_date is not None:
        logger.info(f"Applying proper motion for {proper_motion_date}")
        j2000 = datetime(2000, 1, 1)
        seconds_elapsed = (proper_motion_date - j2000).total_seconds()
        ra_rad = ra_rad + np.array(
            [s["ra_pm"] for s in stars_from_catalog], dtype=float
        ) * seconds_elapsed
        dec_rad = dec_rad + np.array(
            [s["dec_pm"] for s in stars_from_catalog], dtype=float
        ) * seconds_elapsed

    # Vectorize the coordinate transformation
    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.rad2deg(dec_rad)
    coords = np.column_stack((ra_deg, dec_deg))
    pixel_coords = astropy_wcs.wcs_world2pix(coords, 0)

    # Build in-bounds star list
    star_list: list[dict[str, Any]] = []
    for i, star in enumerate(stars_from_catalog):
        xf, yf = pixel_coords[i]
        ra = ra_deg[i]
        dec = dec_deg[i]

        if xf > 0 and xf < pixel_width and yf > 0 and yf < pixel_height:
            star_list.append(
                {
                    **star,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "x_pix": float(xf),
                    "y_pix": float(yf),
                }
            )

    image_metadata = ImageMetadata(
        wcs=wcs,
        width=pixel_width,
        height=pixel_height,
        boresight_dec=center_dec,
        boresight_ra=center_ra,
    )

    return star_list, image_metadata


def query_catalog_gaia(
    wcs: WCSModel,
    faint_lim: int | None = None,
    bright_lim: int | None = None,
    proper_motion_date: datetime | None = None,
    max_stars: int | None = None,
) -> StarListSpace:

    cfg = get_config()

    # Apply default faint limit from config if not provided
    if faint_lim is None:
        faint_conf = getattr(cfg.star_catalog, "faint_limit", None)
        faint_lim = int(faint_conf) if faint_conf is not None else None

    logger.info("Querying Gaia catalog online")
    start_time = time.time()

    # Convert WCS to hashable tuple and timestamp for caching
    wcs_tuple = _make_wcs_hashable(wcs)
    proper_motion_timestamp = (
        proper_motion_date.timestamp() if proper_motion_date else None
    )

    stars_from_catalog, image_metadata = _query_catalog_gaia_cached(
        wcs_tuple, faint_lim, bright_lim, proper_motion_timestamp, max_stars
    )

    pixel_width = image_metadata.width
    pixel_height = image_metadata.height
    center_ra = image_metadata.boresight_ra
    center_dec = image_metadata.boresight_dec

    # Recompute RA/DEC bounds for validation from the cached WCS
    astropy_wcs = wcs.to_astropy_wcs()
    corners_pix = np.array(
        [
            [0, 0],
            [pixel_width, 0],
            [pixel_width, pixel_height],
            [0, pixel_height],
        ]
    )
    corners_world = astropy_wcs.wcs_pix2world(corners_pix, 0)
    ra_corners = corners_world[:, 0]
    dec_corners = corners_world[:, 1]
    safety_margin = 0.2
    ra_range = np.max(ra_corners) - np.min(ra_corners)
    dec_range = np.max(dec_corners) - np.min(dec_corners)
    ra_expansion = ra_range * safety_margin / 2
    dec_expansion = dec_range * safety_margin / 2
    min_ra = float(np.min(ra_corners) - ra_expansion)
    max_ra = float(np.max(ra_corners) + ra_expansion)
    min_dec = float(np.min(dec_corners) - dec_expansion)
    max_dec = float(np.max(dec_corners) + dec_expansion)

    # Convert cached result into StarInSpace list
    star_list: list[StarInSpace] = []
    for star in stars_from_catalog:
        xf = star["x_pix"]
        yf = star["y_pix"]
        ra = star["ra_deg"]
        dec = star["dec_deg"]

        # Ensure magnitudes dict is always populated if magnitude exists
        magnitudes = star.get("magnitudes")
        if magnitudes is None:
            magnitudes = {}
        # If magnitude exists but magnitudes dict is empty, populate it with the primary magnitude
        if star["mv"] is not None and star["mv"] < 32 and len(magnitudes) == 0:
            magnitudes["Primary"] = float(star["mv"])

        star_list.append(
            StarInSpace(
                ra=ra,
                dec=dec,
                x=xf,
                y=yf,
                magnitude=star["mv"],
                magnitudes=magnitudes if len(magnitudes) > 0 else None,
                catalog=star["catalog"],
                catalog_id=star.get("source_id"),
            )
        )

    logger.info(
        f"Found {len(star_list)} stars in Gaia catalog in {time.time() - start_time:.2f} seconds"
    )

    # Validate catalog coverage
    _validate_catalog_coverage(
        stars_from_catalog=stars_from_catalog,
        star_list=star_list,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        catalog_type="Gaia",
        min_ra=min_ra,
        max_ra=max_ra,
        min_dec=min_dec,
        max_dec=max_dec,
    )

    return StarListSpace(
        stars=star_list,
        image_metadata=ImageMetadata(
            wcs=wcs,
            width=pixel_width,
            height=pixel_height,
            boresight_dec=center_dec,
            boresight_ra=center_ra,
        ),
    )


def examine_catalog():
    """Examine the configured catalog and return True if valid."""
    config = get_or_initialize_config()
    catalog_type = config.star_catalog.type
    catalog_path = config.star_catalog.path

    try:
        catalog_enum = CatalogType(catalog_type)
    except ValueError:
        logger.error(f"Unknown catalog type: {catalog_type}")
        return False

    if catalog_enum == CatalogType.SSTRC7:
        return _examine_sstrc7_catalog(catalog_path)
    elif catalog_enum == CatalogType.SDSS:
        return _examine_sdss_catalog()
    elif catalog_enum == CatalogType.GAIA:
        return _examine_gaia_catalog()
    elif catalog_enum == CatalogType.GAIA_LOCAL:
        return _examine_gaia_local_catalog(catalog_path)
    else:
        logger.error(f"Unsupported catalog type: {catalog_enum}")
        return False


def _examine_gaia_local_catalog(catalog_path: str) -> bool:
    """Validate the local Gaia mirror: index.json present and every tile it
    references exists on disk (catches a partial / in-progress download)."""
    import json
    import os

    if not catalog_path:
        logger.error("gaia_local catalog path not configured")
        return False
    index_path = os.path.join(catalog_path, "index.json")
    if not os.path.isfile(index_path):
        logger.error("gaia_local mirror index missing: %s", index_path)
        return False
    try:
        with open(index_path) as fh:
            tiles = (json.load(fh).get("tiles") or {})
    except Exception as e:
        logger.error("gaia_local index unreadable: %s", e)
        return False
    if not tiles:
        logger.error("gaia_local index has no tiles")
        return False
    missing = [m["file"] for m in tiles.values()
               if not os.path.isfile(os.path.join(catalog_path, m["file"]))]
    if missing:
        logger.error("gaia_local mirror incomplete: %d/%d tiles missing (e.g. %s)",
                     len(missing), len(tiles), missing[0])
        return False
    logger.info("gaia_local mirror OK: %d tiles at %s", len(tiles), catalog_path)
    return True


def _examine_sstrc7_catalog(catalog_path: str) -> bool:
    """Validate SSTRC7 catalog by checking path and key files."""
    from senpai.catalog.sstrc7_management import examine_sstrc7_by_path_and_structure
    from senpai.catalog.constants import SSTR7_EXPECTED_FILES

    if not catalog_path:
        logger.error("SSTRC7 catalog path not configured")
        return False

    return examine_sstrc7_by_path_and_structure(catalog_path, SSTR7_EXPECTED_FILES)


def _examine_sdss_catalog() -> bool:
    """Validate SDSS catalog by testing connectivity."""
    try:
        # Try a minimal test query to check SDSS connectivity
        from astroquery.sdss import SDSS

        # Use a very small region to test connectivity without downloading much data
        test_result = SDSS.query_sql("SELECT TOP 1 objid FROM PhotoPrimary")
        if test_result is None or len(test_result) == 0:
            logger.error("SDSS service test query failed")
            return False

        logger.info("SDSS catalog connectivity confirmed")
        return True
    except Exception as e:
        logger.error(f"SDSS catalog connectivity test failed: {e}")
        return False


def _examine_gaia_catalog() -> bool:
    """Validate Gaia catalog by testing connectivity."""
    try:
        # Try a minimal test query to check Gaia connectivity
        from astroquery.gaia import Gaia

        # Use a very small test query
        adql = "SELECT TOP 1 source_id FROM gaiadr3.gaia_source"
        test_result = Gaia.launch_job(adql).get_results()
        if test_result is None or len(test_result) == 0:
            logger.error("Gaia service test query failed")
            return False

        logger.info("Gaia catalog connectivity confirmed")
        return True
    except Exception as e:
        logger.error(f"Gaia catalog connectivity test failed: {e}")
        return False


def enforce_catalog():
    """Enforce catalog validation - raises RuntimeError if catalog is invalid."""
    if not examine_catalog():
        config = get_or_initialize_config()
        catalog_type = config.star_catalog.type
        catalog_path = config.star_catalog.path

        if catalog_type == "sstrc7" and catalog_path:
            raise RuntimeError(
                f"Catalog {catalog_type} is missing or incomplete. "
                f"Run: python -m senpai.catalog.sstrc7_management download --catalog_path {catalog_path}"
            )
        else:
            raise RuntimeError(f"Catalog {catalog_type} is missing or incomplete")


def query_catalog(
    wcs: WCSModel,
    faint_lim: int | None = None,
    bright_lim: int | None = None,
    max_stars: int | None = None,
    proper_motion_date: datetime | None = None,
    apply_sip: bool = False,
) -> StarListSpace:
    """Query the configured star catalog and project stars to pixel coordinates.

    By default the returned pixel coordinates come from the *linear* WCS (no SIP
    distortion). This is intentional and load-bearing: the underlying queries cache
    on a SIP-independent WCS key, so near-identical solutions share cache entries and
    the projection stays cheap. On wide / distorted fields this under-places catalog
    stars at the corners by tens of pixels (a pincushion in any catalog overlay).

    Set ``apply_sip=True`` to re-project the returned stars through the full WCS
    (including SIP distortion) before returning. This leaves the cached query and its
    speed untouched — it only updates the pixel coordinates of the returned stars — so
    callers that render overlays or match catalog stars to detections get
    distortion-correct positions. Sidereal refinement already re-projects with SIP via
    ``existing_stars_from_wcs``; this flag is the equivalent for one-shot callers.
    """
    config = get_config()
    catalog = config.star_catalog.type
    catalog_path = config.star_catalog.path

    if catalog == "sstrc7":
        result = query_catalog_sstr7(
            wcs, catalog_path, faint_lim, bright_lim, proper_motion_date, max_stars
        )
    elif catalog == "sdss":
        result = query_catalog_sdss(
            wcs, faint_lim, bright_lim, proper_motion_date, max_stars
        )
    elif catalog in ("gaia", "gaia_local"):
        # gaia_local reuses the whole gaia path (projection, SIP, sliver cache);
        # only the underlying fetch swaps online TAP for the local mirror (see
        # _query_gaia_sky / gaia_local).
        result = query_catalog_gaia(
            wcs, faint_lim, bright_lim, proper_motion_date, max_stars
        )
    else:
        raise ValueError(
            f"Catalog type {catalog} not supported, choose from: sstrc7, sdss, gaia, gaia_local"
        )

    # Bound dense fields for full-catalog callers (max_stars=None): a galactic-
    # plane frame can return 70k+ stars, and every downstream structure (pydantic
    # star copies per WCS update, isolation trees, photometry scaffolding) scales
    # with it — observed ~30 GB/worker on one batch, enough to OOM a j8 pool.
    # Stratified by magnitude so the faint bins survive for completeness; callers
    # passing an explicit max_stars keep the brightest-N semantics above.
    cap = getattr(config.star_catalog, "max_stars_per_frame", None)
    if max_stars is None and cap and len(result.stars) > cap:
        n_before = len(result.stars)
        result.stars = _stratified_mag_cap(result.stars, cap)
        logger.info(
            "Capped catalog %d -> %d stars (magnitude-stratified; "
            "star_catalog.max_stars_per_frame=%d)", n_before, len(result.stars), cap,
        )

    if apply_sip and result.stars:
        # Re-project the (cached, linear) catalog positions through the full WCS so
        # SIP distortion is applied. Returns fresh StarInSpace objects, so the cached
        # linear positions are never mutated. Lazy import avoids a circular import
        # (wcs_ops imports query_catalog at module load).
        from senpai.engine.utils.wcs_ops import existing_stars_from_wcs

        result.stars = existing_stars_from_wcs(wcs, result.stars)

    return result


def _stratified_mag_cap(stars: list, cap: int) -> list:
    """Subsample to ``cap`` stars, stratified over 0.5-mag bins.

    Waterfilling from the sparsest bin up: small (bright) bins are kept whole,
    over-full (faint) bins are evenly subsampled by magnitude rank, so per-bin
    statistics (completeness fractions, per-bin SNR medians) stay unbiased.
    Deterministic — no RNG. Stars without a magnitude share one bin.
    """
    bins: dict[float, list] = {}
    for s in stars:
        m = getattr(s, "magnitude", None)
        key = math.floor(m * 2.0) / 2.0 if m is not None else math.inf
        bins.setdefault(key, []).append(s)

    out: list = []
    remaining = cap
    # Sparsest bins first so their full contents fit before quotas tighten.
    order = sorted(bins.values(), key=len)
    for i, members in enumerate(order):
        quota = remaining // (len(order) - i)
        if len(members) <= quota:
            out.extend(members)
            remaining -= len(members)
        else:
            members = sorted(
                members,
                key=lambda s: s.magnitude if s.magnitude is not None else math.inf,
            )
            step = len(members) / quota
            out.extend(members[int(j * step)] for j in range(quota))
            remaining -= quota
    return out
