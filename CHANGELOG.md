# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A `settings` singleton for reading configuration.** `from senpai.core.config import
  settings` at import time, then read `settings.detection.snr_threshold` at call time,
  instead of repeating `get_config()` at each use or threading an `AppConfig` through
  signatures that do not otherwise need one. Attribute access resolves against whatever
  `initialize_config` most recently loaded, so a top-level invocation that loads a
  different YAML swaps what every module already holding the import sees. `get_config()`
  is unchanged and remains the accessor everywhere else in the tree for now.
- **Environment-variable overrides for every configuration field.** `AppConfig`
  is now a pydantic-settings `BaseSettings`, so any field can be set with
  `SENPAI_<SECTION>__<FIELD>` — `SENPAI_ASTROMETRY__CPULIMIT_SECONDS=3600`,
  `SENPAI_HEADERS__POINTING__RA_UNITS=degrees` — including nested sections, via
  the `__` delimiter. Environment variables take precedence over the config
  file, so containerized and CI runs can override a single setting without
  templating a YAML. Existing config files, `initialize_config`, `get_config`
  and the immutability of the loaded config are all unchanged.
- **`senpai/resources/config/mdp.yaml`** -- the configuration behind the benchmark
  figures reported across this stack, so a reviewer can reproduce them from the tree. It is
  written out in full, including fields left at their default, so a change to an upstream
  default shows up in review rather than silently altering a run. Covered by
  `test_all_shipped_configs_load`.
- **Optional source masking before 2D background subtraction** —
  `calibrations.background_mask_sources` (default `False`, current behaviour).
  `measure_background` estimates the mesh from every pixel, so a faint target's
  own flux inflates the local mesh and is then subtracted along with it. When
  enabled, pixels above 2 sigma are masked (dilated) before the estimate, so the
  mesh is taken from source-free background only.

### Fixed

- **Point-source re-detection ran on frames whose registration had failed.** The
  end-of-collect catch-all extracted sources from any rate-track frame without
  detections, which includes frames that were shift targets whose shift was
  rejected. The shift loop deliberately leaves those undetected, so re-detecting
  them measured sources against a WCS the pipeline had already discarded, and
  those surface downstream as uncorrelated tracks. Processed shift targets are
  now skipped; every other rate-track frame is unaffected.
- **The SSTRC7 region query was centred on CRVAL rather than on the image.** The
  query is a field-of-view-sized box about a sky position, and that position came
  from `pix2world(CRPIX)` -- which is CRVAL by definition, the WCS reference
  point, not the image centre. Any solution whose CRPIX sits away from the centre
  (a propagated or re-anchored WCS) therefore centred the box off-frame, leaving
  part of the image outside the region that was read and its stars absent from
  the result.
- **`max_stars` capped the padded query region instead of the image.** The region
  carries a 20% safety margin plus rotation padding, so the brightest N of that
  region are largely out of frame: a `max_stars=1000` request returned 61 in-frame
  stars on one benchmark field, starving the catalog FWHM measurement and the WCS
  refit that size themselves from it. Stars are now sorted brightest-first and the
  cap counts in-bounds stars. The sort now runs whether or not a cap was
  requested, so returned stars are magnitude-ordered for every caller.
- **A degenerate FWHM could size aperture masks until the worker was OOM-killed.**
  Aperture radii and streak rectangles derive from the measured FWHM and streak
  fit, which is unbounded; on a noise-flooded frame that fit blows up (a median
  FWHM of 252 px against a normal <=16 px was observed in production) and the
  masks grow quadratically with it. Both starfield photometry paths now bound the
  aggregate mask footprint before allocating any of it. Since both call sites
  already treat a photometry exception as "no photometry for this frame", one
  degenerate frame now degrades to a warning instead of ending the run.

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
