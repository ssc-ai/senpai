# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.9.0] - 2026-10-29

### Added

- **Half-flux radius for focus sweeps (opt-in).** `measure_half_flux_radius()`
  reports the median half-flux radius of a frame's unsaturated sources as
  `DetectionMetadata.pixel_hfr` / `n_hfr_measurements`. Unlike the
  detection-stage FWHM it needs no centroid peak, so it keeps growing through
  donut defocus: on a Huntsman focus sweep the V-curve fit went from R² = 0.17
  (FWHM) to R² = 0.999 (2 × HFR). It costs one extra segmentation pass
  (~0.15–0.5 s per 2744×1836 frame), so it runs only in `pipeline_mode: detect`
  or when `detection.measure_hfr: true` (default `false`).

### Changed

- **Licence: AGPL-3.0-or-later** (from 2.9.0 on).

## [2.8.2] - 2026-08-31

### Fixed

- **The rate-shift solvers no longer crash on a starfield that has no WCS.**
  `solve_rate_from_sidereal` skipped a shift when the sidereal anchor had no
  starfield or no detection metadata, using "a starfield exists" as a stand-in
  for "a WCS exists". 2.8.0's `pipeline_mode` work (ec04ddc) made `collect.py`
  retain the starfield of a frame that *failed* to solve, so its detection-stage
  FWHM would still reach the results for sparse fields and autofocus. That
  created a state the guard does not cover — starfield present, WCS absent — in
  which both of its conditions are false and
  `starfield.wcs_metadata.x_ifov_arcsec` raises `AttributeError`. The exception
  escapes the solver, so a partially recoverable observation became a total
  loss. Measured on the calsat-cert set against 2.8.1: 12 observations lost
  across two sensors, spanning easy/medium/hard/degrading, every one of them
  scored `solved=True` under 2.7.0 — 11 solved frames and 10 target detections
  at ~1.8" median residual, none of them gross. The guard now tests
  `wcs_metadata`, the attribute the function actually dereferences (`StarField`
  derives it from `wcs` in a model validator, so this is equivalently "no WCS").
- **`solve_rate_from_rate` guards `catalog_stars` for the same reason.** It hands
  them to `validate_proposed_shift`, which sorts them, so `None` surfaces as a
  `TypeError` from inside a helper rather than a skipped shift. Not reachable
  today — a rate frame's starfield comes from WCS propagation — but it is the
  same latent shape the sidereal guard had before an unrelated feature made its
  equivalent state reachable.

## [2.8.1] - 2026-08-31

### Fixed

- **Hot pixels are no longer reported as detections.** `filter_point_sources`'
  brightest-pixel concentration check (and the 3x3 median prefilter) only ever
  ran on the rate path, so the sidereal non-catalog promotion in
  `collect.py` emitted single hot pixels as unknown objects — 76 of them on a
  GEO frame, every one hot at a fixed detector pixel in all 7 frames.
  `validate_point_detection` now applies the same 0.35 concentration gate.
  Measured on that frame: real stars peak at 26.3% of flux in one pixel,
  hot pixels start at 57.4%.
- **Point-source width is measured with the estimator it is compared against.**
  The "PSF too wide" gate in `filter_point_sources` fitted a single Gaussian
  and compared it to `pixel_seeing`, which comes from `_radial_profile_fwhm`.
  The Gaussian widens to absorb PSF wings and reads ~80% high, putting the
  median *real* star at the 1.5x limit and rejecting a bright tracked target in
  every rate frame (fit 5.5px, radial profile 3.70px, limit 5.28px). It now
  measures with `_radial_profile_fwhm`; the ratio-based roundness and aspect
  checks still use the fit, where the bias cancels.
- **The catalog-star trail veto no longer drops compact sources.** It compared
  peak brightness only, so a trail passing within a fifth of a bright target's
  peak vetoed it — losing the target on 2 of 6 rate frames, one by a single
  count. A trailed star continues along the trail axis through the detection
  and a tracked point source does not, so the veto now exempts detections with
  no along-axis continuation (target 0.005-0.007 against 0.35-0.93 for trailed
  stars). Sampling runs through the detection, not the catalog position, so it
  does not inherit catalog or shift position error.
- **Hot pixels no longer become streaks.** The sidereal streak width floor sat
  at 0.3xFWHM (1.06px at a 3.52px PSF), which single hot pixels cleared by as
  little as 0.014px; it is now 0.8xFWHM, on the grounds that a streak is the
  PSF smeared along one axis and cannot be narrower than the PSF. Candidates
  whose flux is concentrated in one sample along their own axis are rejected,
  and the edge margin now covers the refined streak length, so a streak running
  off the frame — whose position is not measurable — is dropped.
- **`extract_streak_dims_robust` is no longer defeated by hot pixels.** The
  matched filter smears a hot pixel into a kernel-shaped blob with a large
  response, so all 15 peaks the search is allowed to examine were hot pixels;
  it returned `None` on 2 of 6 statistically identical rate frames and its
  width oscillated between 7.0 and 13.0px. Peak-finding and cutouts now use a
  3x3 median-filtered copy (the rate point path has done this before detection
  all along). Across a GEO set the measurement went from
  `52/50/None/52/50/None` to 49.0px on all six frames.
- **A failed streak extraction no longer discards a validated shift.** When the
  pixel extraction returns `None` but the shift passed star validation, the
  shift-derived streak is used instead of invalidating the link.
- **The sidereal-to-rate shift search is bounded when extraction fails.**
  Without an estimate the cross-correlation was left unbounded and locked onto
  the wrong peak (returning the 6->4 displacement for the 6->5 pair). The
  commanded track rate from the headers now seeds the mask radius as a prior
  only — the shift is still star-validated, so a wrong header fails validation
  and retries as before.

### Added

- **Sidereal streak detection can match the rate-frame trail.** In a
  rate+sidereal collect the tracked object moves at exactly the rate the stars
  trailed at, so a trailed star is the target's signature. The measured trail
  length (median across rate frames) drives a second filter-bank pass; the
  general 5xFWHM bank still runs over all angles and the stronger response per
  pixel is kept, so nothing the default bank found is lost. On a GEO frame the
  target's reported angle moved from 79.4 to 86.0 degrees (independently
  measured at 84.3-85.4), its length from 42 to 46px, and its SNR by 11%.
- Verbose logging for catalog-star veto decisions, which previously reported
  only how many detections were vetoed and never why.

## [2.8.0] - 2026-08-14

### Added

- **`astrometry.pipeline_mode`** trims the sidereal pipeline for callers that
  only need a fast per-frame FWHM (e.g. autofocus sweeps). `detect_solve` runs
  detection + plate solve then stops — the frame keeps its WCS and plate scale
  (via `wcs_metadata`) but reports `fit=False`, skipping WCS refinement, the
  catalog query, catalog FWHM, and photometry. `detect` runs point-source
  detection only, no solve. Both report the detection-stage FWHM in
  `detection_metadata.pixel_fwhm` / `fwhm_stats` and leave `starfield.fit`
  False, which is what keeps every downstream collect pass skipped. Settable in
  config or overridden per call, so one process can interleave reduced-mode
  sweeps with full science batches without mutating global config. `full` (the
  default) is unchanged.
- Rate-track point detections are tagged with `detection_type="point"`.

## [2.7.0] - 2026-08-13

### Fixed

- **Shift-validation control draws are seeded.** `validate_shift_lightweight`
  scores a proposed chain shift against `n_random_trials` random control shifts
  and accepts only if the proposal beats them by `min_correlation_ratio`. Those
  controls came from the global unseeded `np.random`, so every run sampled a
  different null distribution and a marginal shift passed in one run and failed
  in the next; the frame's WCS then flipped solved/unsolved and the chain
  carried the difference downstream. Measured on calsat-cert, 2.6.0 against a
  second run of itself: 73 of 2211 frames (3.3%) and 36 of 230 observations
  (15.7%) changed with no code change at all. The generator is now seeded from
  the frame pair, the trial number and the proposed shift, so distinct shifts
  still draw independent nulls while any given shift always sees the same one.


### Fixed

- **Proper motion was 9.77x too large.** SSTRC7 stores proper motion in units of
  0.32 mas/yr per count, so decoding multiplies by 0.32; the vendored reader
  divided instead (`(1 / 0.32) * mas2rad / year2sec`), overstating every proper
  motion by exactly 1 / 0.32**2 = 9.7656. Right ascension carried a further
  spurious `cos(dec)` applied to a value that is already a coordinate proper
  motion. Only queries passing `proper_motion_date` were affected, but there the
  error was large: a star moving 60 mas/yr, propagated 25 years, was displaced
  14.6 arcsec instead of 1.5. Measured against the catalog at four fields from
  dec -60 to +80; the ratio is 9.7656 at every one.

### Changed

- **SSTRC7 catalog reads now come from the [`sstrc7`](https://pypi.org/project/sstrc7/)
  package** rather than a vendored reader. It reads the same 1801 files from the
  same directory, so existing local catalogs and `catalog.path` settings work
  unchanged. Star records keep senpai's cross-catalog dict shape, and positions,
  magnitudes and provenance strings are identical to 2.6.x -- verified star by
  star against the old reader on real catalog data (positions to 1e-12 rad, all
  18 bands to 1e-9 mag). Magnitudes are rounded to the catalog's integer-millimag
  storage quantum rather than carrying float32 residue.
- **The catalog downloader works again.** `SSTR7_GITHUB_REPO` had been blanked
  pending re-hosting, so `sstrc7_management.py` could not fetch anything; the
  package hosts the catalog at `ssc-ai/sstrc7` and adds partial fetches by
  declination zone, SHA-256 verification, and resumable downloads. Use
  `python -m sstrc7 get --path <dir>`; senpai points at it when the catalog is
  incomplete at startup.

### Removed

- `senpai/catalog/sstr7.py` and `senpai/catalog/sstrc7_management.py`, plus the
  file-size and checksum tables in `senpai/catalog/constants.py` -- about 3400
  lines, now maintained upstream. `SSTR7_EXPECTED_FILES`,
  `SSTR7_EXPECTED_CHECKSUMS`, `SSTR7_GITHUB_REPO` and `SSTR7_RELEASE_TAG` are
  gone with them; `CatalogType`, `SSTRC7Filter` and the other enums are unchanged.

## [2.6.3] - 2026-08-11

### Fixed

- **`create_app()` crashed on every non-editable install**
  ([#6](https://github.com/ssc-ai/senpai/issues/6)). Paths in
  `senpai/core/constants.py` were anchored at the repo root, but a wheel ships
  `senpai/` alone — so on a normal `pip`/`uv` install they resolved into
  `site-packages/`, where neither `resources/` nor `tests/` exists. Two
  independent failures followed: the default config `resources/config/local.yaml`
  was unreadable (leaving `AppConfig` to fail validation on 8 missing fields),
  and the OpenAPI example for `/astrometry/solve/sources` raised
  `FileNotFoundError` loading a repo test fixture at import time. Both are gone;
  CI now builds a wheel, installs it into a clean environment, and constructs
  the app, which is the only way this class of bug is visible.

### Changed

- **`resources/` now lives inside the package**, at `senpai/resources/`, so it
  ships in the wheel and resolves identically in a checkout and when installed.
  `RESOURCES_DIR`, `CONFIG_DIR`, `ASSETS_DIR`, `DATA_DIR`, `APP_DIR` and the
  config-override constants are anchored at the package; `TEST_DATA_DIR` is
  anchored at the new `REPO_ROOT` and is for tests only. `BASE_DIR` remains as
  a deprecated alias of `REPO_ROOT`. In-repo paths change accordingly (e.g.
  `senpai/resources/config/local.yaml`).
- **Cache and log files no longer land in the install tree.** `CACHE_DIR`
  defaults to `$XDG_CACHE_HOME/senpai` (else `~/.cache/senpai`) and logs to
  `CACHE_DIR/logs/app.log`, each overridable with `SENPAI_CACHE_DIR` and
  `SENPAI_LOG_DIR`. Previously both were written under the package —
  `constants.py` even created the log directory as an import side effect, which
  fails outright on a read-only filesystem. That import-time `mkdir` is gone;
  `setup_logging()` still creates the directory it needs.
- The OpenAPI example for `/astrometry/solve/sources` is now a 10-detection
  literal instead of a 100-row fixture read from disk. It is illustrative
  rather than a solvable field, and renders usefully in `/docs`.

## [2.6.2] - 2026-08-11

Release infrastructure only — no library code changed.

2.6.1 was tagged and released on GitHub but never reached PyPI: the publish
workflow below had not been merged yet, so nothing fired. PyPI therefore goes
from 2.6.0 straight to 2.6.2, and this release carries the 2.6.1 stability
fixes listed below to PyPI for the first time.

### Added

- **Automated PyPI publishing** (`.github/workflows/python-publish.yml`) via
  Trusted Publishing (OIDC), so no API token is stored. It triggers only when a
  GitHub Release is published — never on a merge, and never for draft or
  prerelease releases — and pauses for manual approval on the `pypi`
  environment before uploading. Before releasing it verifies the release tag
  matches `pyproject.toml`, that the version is not already on PyPI, and that
  lint, the CI test subset, the artifact filenames, and `twine check` all pass.
  Setup and procedure are documented in `docs/RELEASING.md`.
- **A packaging check in CI** so a broken build or bad metadata surfaces on a
  pull request rather than mid-release, when the GitHub Release is already
  public.

### Changed

- CI also runs on `dev`, not only `main`.

## [2.6.1] - 2026-07-27

### Fixed

- Decompose the streak pixel covariance in `analyze_source_shape_fwhm` with `np.linalg.eigh`
  instead of `np.linalg.eig`. The matrix is `[[xx, xy], [xy, yy]]` — symmetric by construction, so
  its eigenvalues are real by definition. numpy < 2.5 downcast a real `eig` result to `float64`,
  hiding the general solver's complex branch; **numpy 2.5.0 dropped that downcast**
  (`np.linalg.eig(np.eye(2))` is now `complex128`), so every streak length became complex and
  `extract_streak_dims_mapping`'s `round(length / (length_std * 0.5))` raised "type
  numpy.complex128 doesn't define `__round__`", aborting the entire collect — no frame in the
  observation returned a WCS. `eigh` is version-independent and faster, and is already what the
  other six eigendecompositions in the streak code use. This is a pre-existing v2.6.0 bug exposed
  by a dependency upgrade, not a regression on this branch; note `numpy>=2.2.4` is declared with no
  upper bound, so any fresh resolve picks up 2.5.x. Measurements are unchanged — only the dtype.
- Coerce `EXPTIME` to `float` in the rate-to-rate and sidereal-to-rate solvers; a string-valued
  FITS `EXPTIME` header previously crashed the exposure arithmetic.
- Route around a rate-to-rate frame pair that shares a timestamp instead of dividing the estimated
  track rate by a zero inter-frame gap (which crashed streak sizing via `int(inf)`); the pair is
  marked processed-but-invalid, as the missing-starfield guard already does.
- Catch the `ValueError` that `fit_wcs_from_points` raises on degenerate matched-star geometry in
  `fit_and_validate_wcs` and fall back to the provided WCS, rather than failing the whole collect.
- Return the declared 3-tuple `(mag, mag50, mag90)` from every `_estimate_simple_limiting_magnitude`
  fallback path. The `no results` and `< 3 catalog stars` branches previously returned a bare float,
  so `measure_rate_starfield_photometry` / `measure_simple_starfield_photometry` crashed with
  "cannot unpack non-iterable float" and aborted the whole photometry stage on any sparse or
  degenerate frame.

### Changed

- Rewrote `rectangle_pyramoid` to build the rotated streak kernel as the exact per-pixel area
  coverage of the rotated rectangle, evaluated directly on the output grid: no OpenCV in the kernel
  path, output-sized memory only, and a hard `MAX_KERNEL_FINE_ELEMENTS` cap that raises on a
  degenerate streak-length estimate instead of allocating a huge array (the previous
  supersample-rotate-downsample approach could OOM the worker). Reproduces the previous kernel to
  correlation >= 0.98; the call signature is unchanged (`upsample`/`halo_level` accepted as no-ops).
- Reclaim process memory (`gc` + glibc `malloc_trim`) after every collect so long-lived workers do
  not ratchet RSS upward across requests. No-op on non-glibc platforms.

### Dependencies

- Require `astroeasy>=1.2.1`, which releases the plate-solve index page cache after each solve
  (bounds resident memory in long-running solve loops).
