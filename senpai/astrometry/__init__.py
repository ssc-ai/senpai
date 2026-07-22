"""Astrometry support: plate solving, index-series management, and solver backends.

Two solve backends live here, selected by ``astrometry.solver_mode``:

- ``dotnet`` / ``tetra3`` / ``chain`` — the astroeasy-based paths
  (:mod:`senpai.astrometry.astroeasy_backend`): real ``solve-field`` (locally or in
  Docker) or the catalog-native fast-solve cascade.
- ``senpai`` — the in-process EOS solver (:mod:`senpai.astrometry.runner`), built on
  the ``astrometry`` PyPI package: blind solve, same-cell index verify, and SIP-1
  refinement replicating ``solve-field --verify --tweak-order 1``.

The public API of the previous ``senpai.astrometry`` module is re-exported here, so
existing imports keep working.
"""

from senpai.astrometry.astroeasy_backend import (  # re-exported API
    enforce_indices,
    examine_indices,
    require_astrometry_install,
    solve_field,
    solve_field_fits,
    test_astrometry_install,
)

__all__ = [
    "enforce_indices",
    "examine_indices",
    "require_astrometry_install",
    "solve_field",
    "solve_field_fits",
    "test_astrometry_install",
]
