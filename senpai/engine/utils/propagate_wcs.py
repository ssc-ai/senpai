"""Backward-compatible facade — re-exports public API from split submodules."""

from senpai.engine.utils.wcs_ops import (
    existing_stars_from_wcs,
    filter_catalog_stars_by_radius,
    shift_wcs_by_pixel_shift,
)
from senpai.engine.utils.wcs_refinement import (
    get_global_shift_from_astrometric_stars,
    refine_sidereal_frame,
    refine_wcs_by_kernel_convolution,
)

__all__ = [
    "existing_stars_from_wcs",
    "filter_catalog_stars_by_radius",
    "get_global_shift_from_astrometric_stars",
    "refine_sidereal_frame",
    "refine_wcs_by_kernel_convolution",
    "shift_wcs_by_pixel_shift",
]
