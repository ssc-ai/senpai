"""Backward-compatible facade over the EOS engine WCS refinement.

The refinement implementations live in :mod:`senpai.engine.utils.propagate_wcs`;
this module re-exports the public names for callers of the previous split layout.
"""

from senpai.engine.utils.propagate_wcs import (  # re-exported API
    get_global_shift_from_astrometric_stars,
    refine_sidereal_frame,
    refine_sidereal_with_catalog_stars,
    refine_wcs_by_kernel_convolution,
    refine_wcs_with_catalog_stars,
)

__all__ = [
    "get_global_shift_from_astrometric_stars",
    "refine_sidereal_frame",
    "refine_sidereal_with_catalog_stars",
    "refine_wcs_by_kernel_convolution",
    "refine_wcs_with_catalog_stars",
]
