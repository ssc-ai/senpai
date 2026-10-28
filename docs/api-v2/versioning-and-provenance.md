# senpai v2: versioning, pinning and provenance

**Status:** draft for owner review, 2026-10-19. Design only. Numbers marked
**[proposal]** are recommendations awaiting an owner decision (see
[`docs/deployment.md` section 12](../deployment.md#12-open-questions-for-the-owner)).

## 1. Why this matters

senpai results feed long-lived products. A pointing model is refit continuously from
stored observations; a collect's photometry is compared across months. If a result cannot
say exactly what produced it, a refit silently mixes outputs of different algorithms and
data, and a drift in the model cannot be told apart from a change in the software.

senpai 2.x cannot answer this today:

- The API reports YAML `app.version` (`0.1.0`), not the package version.
- Patch releases change numeric results (2.8.2 changed solve outcomes on 12 observations),
  so the package version is not an algorithm version.
- `uv.lock` is gitignored and astrometry.net is cloned at an unpinned HEAD, so "senpai
  2.8.2" does not identify a build.
- No data artifact (indices, tetra3 DBs, Gaia mirror) carries a version or hash that
  results could cite.

## 2. The version axes

Each axis has its own identifier and changes independently. Every result records all of
them (section 7).

| Axis | Identifier (example) | Changes when | Who adopts it | Seen in |
| --- | --- | --- | --- | --- |
| API contract | `2.3`: URL major `/v2` + integer minor | Minor: additive change (section 6.1). Major: breaking change, new URL prefix `/v3`. | Server operator; clients read capabilities | `Senpai-Api-Version` header, `capabilities.api.version`, `provenance.api_version` |
| Server software | `astro-senpai 3.1.4` + `build` (git SHA) + `lock_sha256` (committed `uv.lock`) + `image_digest`, plus `environment_sha256` (the dependency set actually loaded) | Any code release (SemVer for packaging and API stability, not for numerics) | Operator (pull, import) | `capabilities.server`, `provenance.software` |
| Algorithm | `<pipeline>@<int>`: `senpai.frame@5`, `senpai.collect@8`, `senpai.sources-solve@3`; later `starcsp.frame@1`. Informational `components` (`detect: mf-point/3`, `trail: rate-trail/2`, `solve: astroeasy-planner/2`, `photometry: aper/1`) | **Any** change to numeric results: algorithm, default knob, weights, or a dependency upgrade that moves numbers. Enforced by the golden gate (section 3). | Operator chooses the default; clients may pin | `capabilities.algorithms[]`, `provenance.algorithm` |
| Effective config | `config_hash: sha256:...` + echoed `overrides` | Site config or request overrides change a result-affecting knob | Site (YAML) and each request | `provenance.config` |
| Data bundles | `<kind>/<name>@<version>` + `sha256:` digest of its signed manifest; the active set has its own digest (`bundle_set`) | Data rebuild (new Gaia slice, index subset, tetra3 DB, IERS table, weights). Never mutated in place. | Operator activates; never automatic on air-gapped | `capabilities.bundles`, `provenance.data` |
| Release | `channel` + monotonic `seq` (`stable#41`, `lts-2027#3`) in a signed release manifest that lists the digests of everything above | Every published release | Operator | `capabilities.server.release`, `provenance.release` |

"detector_version" in the owner's direction is the **algorithm** axis. It covers
detection, solve planning and photometry, because all of them move results; it is named
per pipeline because a frame analysis and a collect evolve separately.

Rules that keep the axes independent:

1. A software release that does not move results keeps every algorithm version. Security
   patches can therefore be applied without touching a pointing model's inputs.
2. A data rebuild is a new bundle digest, not a software release.
3. A changed default knob is an algorithm change, not a config change. `config_hash` covers
   what the operator or request set explicitly, on top of the algorithm's defaults.
4. One server release contains exactly one version of each pipeline. Serving several
   versions means running several releases side by side (hosted does; section 5).

## 3. Keeping the algorithm version honest: the golden gate

| Item | Rule |
| --- | --- |
| Corpus | `senpai-golden` data bundle, about 30 to 60 real frames from our telescopes plus sdasim frames: sidereal, rate-tracked, trailed, sparse, crowded, binned, RA = 0 and polar fields, multi-frame collects. |
| Run | CI runs every pipeline on the corpus on every PR and compares canonicalised outputs with stored references. |
| Tolerance [proposal; tune on first run] | Positions 1e-4 px; WCS centre 1e-3 arcsec; magnitudes 1e-4 mag; counts, statuses, tiers and rung outcomes exact. |
| On a diff | CI fails unless the PR bumps `PIPELINE_VERSIONS[<pipeline>]` and adds a `### Results` changelog entry naming what moved and by how much. CI attaches a delta report. |
| Dependency upgrades | Same gate. A numpy/astropy/photutils bump that moves outputs is an algorithm bump. |
| Platforms | The gate runs on `linux-x86_64` and `win-amd64`. Cross-platform last-bit differences are not promised away; that is why `platform_class` is part of the reproducibility key. |

## 4. `config_hash`

1. Build the effective config for the pipeline: server config plus request overrides
   (`detection`, `solve`, `photometry`, `streaks`, `expert`, ...), validated by pydantic.
2. Keep the result-affecting subset. Drop deployment-only settings (paths, logging,
   plotting, worker counts); data identity enters through bundle digests, not paths.
   Deployment limits that change results are **not** deployment-only and stay in the hash:
   `star_catalog.max_stars_per_frame` (magnitude-stratified catalog subsampling,
   `senpai/catalog/runner.py:1090-1096`), any server clamp of `detection.max_sources`,
   internal rescaling or binning, and per-tier time budgets that can cut a rung short. The
   list of result-affecting keys is kept in code next to `PIPELINE_VERSIONS` and checked
   by the golden gate.
3. Serialize with RFC 8785 (JCS) canonical JSON and SHA-256 it. Hash the validated model
   dump, never the YAML file (unknown YAML keys are dropped silently today).
4. Echo every result-affecting value that differs from the algorithm's defaults in
   `provenance.config.overrides`, whether it came from the request, the site config or a
   deployment limit, so the hash can be recomputed. Two deployments with different
   `max_stars_per_frame` therefore produce different `config_hash` and
   `reproducibility_key` values.

## 5. Pinning

### 5.1 Per request

`pin` in the request:

| Field | Values | Effect |
| --- | --- | --- |
| `pin.algorithm` | `senpai.frame@5`, `5`, `stable`, `candidate`, or null | Exact version, or channel alias (floats), or the server default. The named pipeline must match the route (`senpai.frame` on `/v2/analyze`, `senpai.sources-solve` on `/v2/solve/sources`, `senpai.collect` for collects); a mismatch is `400 invalid-request`. |
| `pin.bundles` | `{"tetra3-db/fov0.5-1.5_g11": "sha256:..."}` | These exact bundles must be used. |
| `pin.policy` | `exact` (default), `prefer` | `exact`: unavailable pin is `422 algorithm-unavailable` / `bundle-unavailable` with `available[]`. `prefer`: use the default and add warning `PIN_NOT_HONORED`. The server never substitutes silently. |

`provenance.algorithm.pin_requested` and `pin_honored` record the outcome.

Clients that use more than one route pin per pipeline (for example
`{senpai.frame: 5, senpai.sources-solve: 3}`). A split result (local detection, remote
solve) has two producers: `client_asserted_detection_provenance.algorithm.id` and
`provenance.algorithm.id`; a client compares each against its pin for that pipeline.

### 5.2 Server policy

| Profile | What is served |
| --- | --- |
| Hosted | The current `stable` version of each pipeline, every version still within its support window (one release pool per version behind a router), and an opt-in `candidate`. |
| Self-hosted, air-gapped | Whatever the active release provides. Running two releases behind the shipped router is optional and documented. |
| Local / in-process | The installed engine. |

Hosted support window [proposal]: each superseded algorithm version stays pinnable for at
least 12 months after it is superseded, and at least the last 2 versions are always served.
`capabilities.algorithms[].supported_until` states the date.

### 5.3 Promotion on hosted

1. **Golden gate** passes; the release carries the delta report.
2. **Shadow:** a sample of production requests is re-run on the new version. Only
   requests that themselves carry `data_policy.contribute.purposes` containing
   `algorithm_eval`, from a tenant that has agreed to the current contribution
   `terms_version`, are eligible; a tenant-level setting alone is never enough. Such a
   request echoes `data_policy_applied.contribute: true` and `purposes: [algorithm_eval]`.
   The frame bytes are held for the shadow run only: the hold is recorded in the frame
   record (`GET /v2/frames/{pixel_sha256}`, `holds[]`) and in
   `data_policy_applied.frame_retain_until`, with a hard cap of 1 hour [proposal], and the
   bytes are purged right after the shadow run. Shadow results are never returned.
   Compared per stratum (sidereal, rate, scale, tier): solve
   rate, `n_matched`, `rms_arcsec`, centre delta vs stable, recall/precision on matched
   stars, latency. Promotion criteria are written down before the shadow starts
   [proposal: no stratum loses more than 0.5 percentage points of solve rate; median centre
   delta below 0.1 px].
3. **Candidate:** `pin.algorithm: candidate` for opt-in testers; `status: candidate` in
   capabilities.
4. **Stable:** announced at least 14 days ahead [proposal] in the changelog feed and in
   `capabilities.algorithms[].promotes_at`. From that date unpinned and `stable`-pinned
   requests get the new version.

Software-only releases (same algorithm versions) use an ordinary traffic canary watching
errors and latency only, because the gate already proved them result-identical.

### 5.4 Recommended client default

Pin exactly (`senpai.frame@N`) and move the pin deliberately, after reprocessing
stored frames under the new version and comparing results.

## 6. Compatibility and deprecation

### 6.1 API contract

Compatible within `/v2` (a minor bump):

- new optional request fields whose default preserves behaviour;
- new response fields, routes, problem codes, media types;
- new enum values in **responses** (clients treat unknown values as "other"; every
  response enum is marked `x-extensible-enum` in `openapi.yaml`, generated clients map
  unknown values to a raw-string fallback, and the FakeServer can emit unknown values so
  client CI checks this);
- relaxed limits.

Breaking (needs `/v3`): removing or renaming a field; changing units, origin or meaning;
making an optional field required; tightening validation of an existing field; a new
request enum value that changes default behaviour.

Client/server skew: because the server rejects unknown request fields, a newer client must
not send fields an older server lacks. The generated client reads
`capabilities.api.version` and omits newer fields (each field carries an `x-since`
annotation once 2.1 exists); a client that needs a minor sends `Senpai-Api-Min-Minor` and
gets `412 api-minor-unsupported` from an older server.

Every server release implements all v2 minors up to its own. After `/v3` reaches GA, `/v2`
is served for at least 24 months [proposal] and in every LTS line that starts within that
window.

### 6.2 Deprecation sequence (any route or version)

1. Announce in the changelog and `capabilities.api.deprecated`.
2. Send `Deprecation: @<unix-ts>` (RFC 9745) and `Link: <migration doc>; rel="deprecation"`.
3. At least 6 months later, add `Sunset: <HTTP-date>` (RFC 8594).
4. After the sunset date, hosted answers `410 api-version-gone`.
5. Self-hosted software removes the code only in the next major release, and never while a
   supported LTS line still contains it.

Clients should log a warning and note it in the run manifest when they see either header.

### 6.3 v1 timeline [proposal]

| Step | When | What |
| --- | --- | --- |
| senpai 3.0.0 | v2 GA | v1 kept with `Deprecation` headers. Breaking fixes that justify the major: `GET /astrometry/` config dump removed, CORS allow-list, auth required on non-loopback binds, `/astrometry/solve/fits` stops writing PNGs. v1 handlers move behind the v2 auth (tenant and scopes), limits, rate limits, worker pool and tenant default retention; only their bodies stay unchanged. Hosted enables v1 per tenant, default off. |
| GA + 6 months | | `Sunset` header on v1, set to GA + 12 months. |
| GA + 12 months | | Hosted v1 returns `410`. |
| senpai 4.0.0 | at least 12 months after GA, and after the last LTS line containing v1 ends | v1 code removed. |

### 6.4 Algorithm versions

- Any result change bumps the pipeline's integer version (section 3).
- An LTS line freezes algorithm versions for its whole life.
- Hosted availability of old versions: section 5.2.

### 6.5 Data bundles

- Bundles are immutable; a rebuild is a new version and digest.
- Each bundle kind has an integer reader format (`readers: {"tetra3-db": 1}`). A bundle
  declares `requires` (engine and reader versions); the server refuses to activate a
  bundle it cannot read, and a release manifest lists the reader formats it supports.
- Published bundles stay downloadable by digest for at least 5 years [proposal], subject to
  data licences, so old results stay reproducible.
- Every bundle credits its upstream sources and carries their licence/terms text
  (`bundle.json` `license.upstreams`, a `NOTICE` file), and every bundle kind has a pinned
  build recipe with an expected digest. A user can therefore rebuild a cited bundle from
  upstream instead of downloading ours; a self-built bundle with the same digest is the same
  bundle (`docs/deployment.md` section 3.1.1).

### 6.6 Server software

- SemVer. Channels `edge`, `stable`, `lts` (see `docs/deployment.md`).
- Persistent state (job store, retention index, idempotency keys) uses expand/contract
  migrations: release N's schema is readable by N-1, so rollback by one release always
  works. When that cannot hold, the release manifest declares an upgrade stop and the
  updater refuses to skip it.

## 7. The provenance block

Present in every `FrameResult`, `CollectResult` (one per run) and `SourcesSolveResult`,
with the same shape in-process, on localhost and remote. Schema: `Provenance` in
`openapi.yaml`.

In-process builds: a client kit (for example a Windows zip) installs `astro-senpai[engine]` from
the client's own lock, so numpy, astropy, photutils, scipy and astroeasy versions may
differ from the release the golden gate ran on. The engine therefore computes
`software.environment_sha256` at import (`importlib.metadata` over its dependency
closure: name, version, wheel tag) and compares it with the dependency set listed in the
release manifest (`environment_matches_release`). `lock_sha256` is meaningful only when
they match. To keep in-process results inside the golden guarantee, client kits install
senpai's dependencies with senpai's lock as a constraints file, and the client's CI fails
if `environment_matches_release` would be false.

| Field | Content | Why a consumer needs it |
| --- | --- | --- |
| `result_id`, `request_id`, `job_id`, `idempotency_key` | Server identities | Retrieve, replay, audit, backfill late results |
| `received_at`, `completed_at` | Timestamps | Ordering, latency |
| `api_version` | `2.N` | Interpret fields correctly |
| `release` | `{channel, seq}` | Map to a signed release manifest |
| `software` | senpai version, git build, `lock_sha256`, `image_digest`, `environment_sha256`, `environment_matches_release`; astroeasy, astropy, photutils, numpy, scipy, python; `platform_class`; astrometry.net commit and the digest of astroeasy's `astrometry-cli` image | Identify the exact build, including an in-process engine whose dependencies were chosen by a client's lock; diagnose platform effects |
| `algorithm` | `id`, `pipeline`, `version`, `components`, `pin_requested`, `pin_honored` | Group or filter observations by producer; detect silent version changes |
| `config` | `hash`, `overrides` | Same algorithm, different knobs, is a different producer |
| `data` | `bundle_set`; `used[]` (kind, name, version, digest, origin `upstream`/`site`/`path`, `signed`, `attribution`) for data that touched this result, including data configured as a raw path (hashed at load, `origin: path`); catalog name, release, mag limit, epoch applied; astrometry.net index files used, each with the digest of the bundle it came from | Catalog and index changes move solved positions |
| `time_data` | astropy-iers-data version, IERS table age, `iers_stale`, leap-second expiry, digest of the `iers` bundle whose tables were loaded | Air-gapped sites run on aging time data; refits must see it |
| `input` | `transport_sha256`, `pixel_sha256`, `header_sha256` (or `sources_sha256`), `encoding`, `lossy`; for sources-solve `detection_attested: false` | Tie result to exact pixels; dedup; detect lossy input; tell client-asserted detections from server-attested ones |
| `execution` | `mode` (`remote`, `service`, `in_process`), `profile` (`hosted`, `self_hosted`, `airgapped`, `local`), `deployment_id`, `worker` | Know where it ran; local fallback results are distinguishable |
| `cache` | `hit`, `original_result_id` | A cached result is not a new measurement |
| `reproducibility_key` | section 8 | Reprocessing and equality checks |

For the living pointing model specifically:

- **Filter and weight by producer.** A refit can exclude or down-weight observations from a
  superseded algorithm, or from a fallback producer, instead of mixing them silently.
- **Explain model drift.** A step in residuals that coincides with a change in
  `algorithm.id` or `data.bundle_set` is a software or data effect, not the mount.
- **Evaluate before switching.** Stored frames can be reprocessed under a candidate pin and
  compared observation by observation (same `pixel_sha256`, different `algorithm.id`).
- **Detect stale time data.** `iers_stale` flags results computed on old IERS predictions.

## 8. Reproducibility guarantees

```
reproducibility_key = sha256(
    algorithm.id || config.hash || data.bundle_set-used digests ||
    input.pixel_sha256 (or sources_sha256) || JCS(hint, frame meta used) || platform_class ||
    (software.environment_sha256, only when environment_matches_release is false) ||
    (client_asserted_detection_provenance.reproducibility_key, or JCS of the block when it
     has none; sources-solve only))
```

| Guarantee | Scope |
| --- | --- |
| Equal key implies equal results within golden tolerance | Any deployment, any software release that serves that algorithm version, with `environment_matches_release: true`. An in-process engine on a different dependency set gets a different key and no golden guarantee. |
| Same frame, same pin, different software patch | Equal within golden tolerance (the gate enforces it) |
| Different `platform_class` | Not promised bit-identical; expected within golden tolerance, verified by running the gate on both platforms |
| Result cache | May return a stored result for an equal key; marked in `provenance.cache` |
| Software digest | Recorded but excluded from the key; the golden gate is what makes this safe |

## 9. How changes are communicated

| Channel | Carries |
| --- | --- |
| Release notes (four mandatory sections: **API**, **Software**, **Results**, **Data**) | For each pipeline: "Will my results change? yes/no", with the golden delta report |
| Signed release manifest and channel index (`channels/<channel>.json`; keyless Sigstore, [`../deployment.md`](../deployment.md) section 4) | Digests, algorithm versions, API version, upgrade stops, EOL dates, security advisories. The channel index carries `expires` (issued + 14 days [proposal]) and is re-signed on a schedule even without a release, so a stale or frozen mirror is detectable. |
| `GET /v2/capabilities` | What this deployment serves now, including `promotes_at`, `supported_until`, deprecated routes |
| Response headers | `Senpai-Api-Version`; `Deprecation`, `Sunset`, `Link` on deprecated routes |
| Result warnings | `ALGORITHM_DEPRECATED`, `PIN_NOT_HONORED`, `IERS_STALE` |
| Changelog feed | `/changelog.json` and Atom on the project site |
