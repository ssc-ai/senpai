# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [2.7.0] — unreleased

Upstreams an MDP-validated detection/WCS engine onto the open-source base while
preserving the existing feature set. Dependencies remain PyPI-only.

### Added

- In-process astrometry.net solver as `astrometry.solver_mode: senpai` (via the `astrometry`
  PyPI package), alongside the existing `dotnet` / `tetra3` / `chain` astroeasy modes.
- pydantic-settings config framework: `__`-delimited environment overrides for every field
  (env beats YAML), `SENPAI_CONFIG_PATH` discovery, and flat-or-`app:`-nested YAML.
- Degenerate-input / environment crash guards: streak-kernel grid cap
  (`MAX_KERNEL_FINE_ELEMENTS`), rate→rate zero-elapsed-time route-around, string-`EXPTIME`
  coercion, and a typed `MissingDependencyError` with an `image2xy`-on-PATH check.
- Automated PyPI release workflow (Trusted Publishing on `dev` → `main`); CI now also runs on
  `dev` without publishing.
- New dependencies (all PyPI): `sep`, `astrometry`, `scikit-optimize`, `scipy`,
  `pydantic-settings`.
- README sections documenting the configuration framework and expected FITS header fields.

### Changed

- Core registration flow (sidereal solve → shift chain → refinement → detection) is now the ported
  implementation; the previous implementations are preserved as wired `*_extra` siblings on the
  rate / photometry / API-astrometry paths.
- Point-source detection uses a sep-based detector with a flux-concentration gate.
- `sstr7.get_star_mv` restored to open-band-first magnitude priority; catalog query rebuilt from
  the full WCS (previously reduced to a CRPIX-centered cache key).
- float64 restored on the shared registration hot path.
- All tests folded into a top-level `tests/` tree; repo-wide docstrings + type annotations added;
  `ruff check senpai tests` is clean under the full project select.

### Validation

- Metrics-identical to the reference on the 134-collect MDP benchmark (true positives 1061,
  0 uncorrelated tracks, RMS 1.244″); bit-identical on 134/134 collects under the reference
  numeric stack. Stock v2.6.0 scores 647 true positives on the same benchmark for comparison.
