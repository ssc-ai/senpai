# senpai API v2

**Status:** draft for owner review, 2026-10-19. Design only; nothing here is implemented yet.

| Document | What it covers |
| --- | --- |
| [`openapi.yaml`](openapi.yaml) | The contract (OpenAPI 3.1). Normative. Validates with `openapi-spec-validator`. |
| this README | Purpose, principles, routes, conventions, errors, idempotency, sync vs jobs, auth, limits, v1 migration. |
| [`versioning-and-provenance.md`](versioning-and-provenance.md) | Version axes, pinning, compatibility and deprecation policy, the provenance block. |
| [`../deployment.md`](../deployment.md) | Hosted, self-hosted, air-gapped and local deployment; artifacts, signing, updates, data policy, egress. |

## 1. Purpose and consumers

senpai v2 is the HTTP contract for senpai as a general product: detection of stars and
moving objects in sidereal and rate-tracked frames, plate solving, multi-frame collects
with correlated streaks, photometry and seeing. Frames go to the service, so detection
and solve improvements (new algorithms, data, later ML weights) reach users without
client software updates.

| Consumer | Uses |
| --- | --- |
| Pointing-model clients | `POST /v2/analyze`, one frame per request, in the slew loop. `POST /v2/solve/sources` in low-bandwidth mode. Stores `provenance` with every observation. |
| Collect processing (SDA, photometry) | `POST /v2/jobs` with `kind: collect`, SSE progress, Arrow tables. Targets (satellites) in rate-tracked and sidereal frames come back in `objects` (`ops: objects`), separate from the star list. |
| Autofocus and seeing tools | `POST /v2/analyze` with `ops: [detect, seeing]`. |
| Clients with their own detector | `POST /v2/solve/sources`. |

The same request and result models run three ways: in-process (`senpai.v2.engine`),
against a localhost service, and against a remote server (`senpai-client`). One engine
function, `analyze_frame(image, AnalyzeRequest) -> FrameResult`, is the reference
implementation; the HTTP server is a thin wrapper around it. For equal algorithm version,
config hash and data bundles, local and remote results are the same and carry the same
provenance block.

## 2. Design principles

1. **One contract, every deployment.** No hosted-only routes or fields. Auth, limits,
   retention and jobs are in the open-source server. Billing and accounts live in a gateway
   outside senpai.
2. **Strict input, tolerant output.** Unknown request fields are rejected
   (`400 invalid-request`); senpai 2.x silently ignored unknown config keys and the FoV
   hint. Clients must ignore unknown response fields and unknown enum values. Every enum
   that can appear in a response is marked `x-extensible-enum` in `openapi.yaml`;
   generated clients map unknown values to a raw-string fallback instead of raising, and
   the FakeServer has a mode that emits unknown values so client CI tests this.
3. **"No solution" is an answer, not an error.** A processed frame returns `200` with
   `solution.status`. 4xx/5xx mean the request or server failed.
4. **Every result says what produced it.** `provenance` names API, software, algorithm,
   config hash and every data bundle used.
5. **Full precision, explicit conventions.** No rounding on the wire. Pixel origin, WCS
   convention, angles, epoch and units are fixed for all of v2 (section 4).
6. **Server stays stateless per frame.** Seeing memory, predicted star motion and prior WCS
   come in the request.
7. **OpenAPI is the contract.** `openapi.yaml` is checked in; the server's
   `/v2/openapi.json` must equal it; `senpai-client`, the `senpai.v2` models and a
   schema-valid fake server (`senpai.v2.testing.FakeServer`) are generated from it.

## 3. Routes

All paths are under `/v2`. Auth is `Authorization: Bearer <token>` unless noted. Every
id in a path (`job_id`, `result_id`, `pixel_sha256`, table URLs, SSE streams, `upload:<id>`)
is looked up within the caller's tenant; another tenant's id is `404`, never `403`.

| Method and path | Purpose | Mode | Scope |
| --- | --- | --- | --- |
| `GET /v2/health` | Readiness: `ok`, `degraded`, `unavailable` (`503`). Without a token only `status`; versions and check detail only with a token (or on the `local` profile). | sync | none |
| `GET /v2/health/live` | Liveness. | sync | none |
| `GET /v2/capabilities` | Versions, served algorithms, ops, formats, tiers, bundles (with their upstream credits and licence terms), the caller's effective limits, auth modes, data policies. No paths or secrets. | sync | any valid token (none on `local`) |
| `GET /v2/openapi.json` | The contract as served. | sync | none |
| `POST /v2/analyze` | One frame: detect, and optionally solve, seeing, photometry, streaks, objects. The slew-loop route. | bounded sync; optional promotion to job | `analyze` |
| `POST /v2/solve/sources` | Detection list in, WCS out. Successor of `/astrometry/solve/sources`. | bounded sync | `analyze` |
| `POST /v2/jobs` | Async `kind: frame`, `collect` or `sources`. `202` + `Location`. | async | `jobs` |
| `GET /v2/jobs/{job_id}` | Status: `queued`, `running`, `succeeded`, `failed`, `canceled`; progress; `result_url`. | | `jobs` |
| `GET /v2/jobs/{job_id}/events` | SSE: `state`, `progress`, `frame_done`, `done`. | stream | `jobs` |
| `DELETE /v2/jobs/{job_id}` | Cancel and purge. | | `data:delete` |
| `GET /v2/results/{result_id}` | A retained result (frame, collect, sources-solve). JSON or msgpack. | | `results:read` |
| `DELETE /v2/results/{result_id}` | Delete now (an audit row remains; fields and lifetime in `../deployment.md` section 8). | | `data:delete` |
| `GET /v2/results/{result_id}/tables/{table}` | Bulk tables: `detections`, `matches`, `photometry_stars`, `fwhm_samples`, `streaks`, `objects`. Arrow, msgpack or JSON. | | `results:read` |
| `POST /v2/uploads` | Presigned upload slot for `image.ref` (only where `capabilities.inputs.image_ref.uploads`). | | `analyze` |
| `GET /v2/frames/{pixel_sha256}` | What the server holds for a frame: retention, holds, results, contribution grant. | | `results:read` |
| `DELETE /v2/frames/{pixel_sha256}` | Delete a retained frame and its results; revoke contribution. | | `data:delete` |

There are deliberately no admin or bundle-management routes. Bundles are managed on the
host with `senpai-admin` (see `docs/deployment.md`).

Data credits: `capabilities.bundles.active[].license` lists each active bundle's licence and
every upstream source (name, release, credit, citation, terms URL), and each
`provenance.data.used[]` entry carries that bundle's `attribution` line, so every result
maps a bundle digest to its credit (`../deployment.md` section 3.1.1).

Reserved for a later minor: `POST /v2/catalog/cone` (WCS-free catalog cone).

## 4. Conventions (normative)

| Topic | Rule |
| --- | --- |
| Pixel coordinates | `pixel_origin: 0`. The centre of the first pixel is `(0.0, 0.0)`; pixel centres are at integers. `x` is the column (FITS axis 1), `y` the row (FITS axis 2). Every result states `pixel_convention` once. |
| Pixel grid | Always the frame **as delivered** (binned, not rescaled). If the server bins or auto-scales internally, it maps outputs back and reports `frame.internal_scale_factor` for information only. |
| WCS | A standard FITS keyword dict: `CTYPE`, `CRVAL`, `CRPIX` (**1-based**, per FITS), `CD` or `PC`+`CDELT`, SIP `A_*`/`B_*`/`AP_*`/`BP_*` (all coefficients present whenever CTYPE ends in `-SIP`), `RADESYS=ICRS`, `EQUINOX`, `IMAGEW`/`IMAGEH`. Time keywords: `DATE-OBS`/`MJD-OBS` and `DATE-BEG`/`MJD-BEG` = exposure start; `DATE-AVG`/`MJD-AVG` = mid-exposure (the epoch of every position); `TIMESYS='UTC'`. Must satisfy `astropy.wcs.WCS(header).all_pix2world(x, y, 0)` for the 0-based coordinates in this API. Per-tier CRPIX offsets are fixed inside senpai/astroeasy with a regression test per tier; clients apply no correction. |
| Pointing reference | `solution.derived.center_*` is the sky position of `reference_pixel = [W/2, H/2]` (0-based). CRVAL is not the pointing. |
| Scale, roll, parity | `scale_arcsec_px = sqrt(|det CD|) * 3600`. `roll_deg` = position angle of image +y, north through east, `[0, 360)`. `parity = +1` when `det(CD) < 0` (sky-like), `-1` when mirrored. |
| Pixel-space angles | `angle_deg` is axial, `[0, 180)`, counter-clockwise from +x toward +y in array coordinates. `direction_deg` (directed motion) is `[0, 360)` with the same zero and sense. |
| Sky | ICRS, degrees, RA in `[0, 360)`. Catalog positions are propagated to the frame epoch (proper motion applied). |
| Time | UTC, ISO 8601 with `Z`. Position epoch is **mid-exposure**. Trail and streak midpoints are mid-exposure positions. `t_mid` precedence: explicit `frame.t_mid_utc` > explicit `frame.t_start_utc` + exposure/2 > header mid key (`header_map.t_mid_utc`, then `DATE-AVG`, `MIDPOINT`) > header start key + exposure/2, where `frame.header_time_ref` (`start`/`mid`/`end`) says what the mapped start key really marks (MaxIm with the DL Imaging driver >= 6.24 writes the midpoint into `DATE-OBS`). Header times without a zone (TheSkyX dropped the `Z`) are UTC. Every frame echoes `t_start_utc`, `t_mid_utc` and `t_mid_source` (e.g. `header:DATE-AVG`). |
| Units | In field names: `_deg`, `_arcsec`, `_px`, `_s`, `_ms`, `_adu`, `_e`, `_mag`. Flux is total instrumental counts on the delivered grid, `flux_unit` `adu` (or `e` when gain is known); internal binning is undone. |
| Magnitudes | Every limiting magnitude states its basis: `{value_mag, snr, completeness}`. The v1 names `limiting_mag_5sigma`/`_3sigma` (mislabeled) are gone. |
| Precision | Full IEEE double; no rounding (v1 rounded RA/Dec to 1e-4 deg). `NaN`/`Inf` serialize as `null`. |
| Ordering | `detection.sources` is brightest first (`rank` 0-based). Collect `frames[]` are in request order with `time_index`. |
| Metadata precedence | Explicit `frame` values override FITS headers (`header_policy: fallback`); headers fill gaps using the per-request `header_map`, then server key maps. Each value used is echoed with its `source` (`request`, `header:<KEY>`, `derived`, ...). An RA or Dec of exactly 0 is a value, not "missing". |
| Naming | `snake_case`; enums are lowercase strings. |

Vocabulary: a **trail** is a star smeared by tracking or exposure (`detection.method:
trail`). A **streak** is a moving object (`streaks[]`,
`correlated_streaks[]`).

## 5. Requests

### 5.1 Frames: `POST /v2/analyze`

`multipart/form-data` with two parts:

| Part | Content type | Notes |
| --- | --- | --- |
| `request` | `application/json` | `AnalyzeRequest`, at most 64 KiB, strict. |
| `image` | `application/fits` (or `application/gzip` for a gzip-wrapped file) | Exactly one of this part or `request.image.ref`. |

| Image encoding | `frame.encoding` | Handling |
| --- | --- | --- |
| Tile-compressed FITS, Rice / fpack (`CompImageHDU`) | `fits+rice` | **Preferred.** First HDU with 2-D data is used unless `image.hdu` is set. |
| Tile-compressed FITS, GZIP_1 / GZIP_2 | `fits+tile-gzip` | Accepted; lossless when unquantized (the usual choice for float frames). |
| Plain FITS | `fits` | Accepted. |
| gzip-wrapped FITS file | `fits+gzip` | Accepted. Signalled by the part's `Content-Type: application/gzip` or detected by magic bytes. Per-part `Content-Encoding` is not used (RFC 7578 section 4.8 forbids it in form-data parts). |
| Lossy (quantized float) tile compression | as above, `frame.lossy: true` | `422 lossy-encoding-not-allowed` unless `image.allow_lossy: true`. |

Lossy is defined on the compressed HDU: ZBITPIX < 0 (float) **and** quantized, meaning a
`ZSCALE`/`ZZERO` column or keyword is present, or `ZQUANTIZ` is present and not `NONE`.
Integer ZBITPIX is always lossless. Presence of `ZQUANTIZ` alone is not the test: astropy
writes no `ZQUANTIZ` keyword at all, both for its default (quantized, lossy) float Rice
and GZIP compression and for `quantize_level=0` (lossless); only the `ZSCALE`/`ZZERO`
columns tell them apart (checked with astropy 8.0.2). The golden corpus carries fixtures
for both cases.

Alternatively send `application/json` (the `AnalyzeRequest` alone) with `image.ref` set:
`upload:<id>` from `POST /v2/uploads`, `https://` or `s3://` (with `sha256`, enforced by
the schema), or `file:///` where the operator enabled local paths. Section 5.5 gives the
resolution rules.

`AnalyzeRequest` blocks (full schema in `openapi.yaml`):

| Block | Carries |
| --- | --- |
| `ops` | `detect`, `solve`, `seeing`, `photometry`, `streaks`. Default `[detect, solve]`. |
| `frame` | Exposure, start/mid time, binning, filter, gain, read noise, saturation, track mode, site, `calibrated`, `header_policy`, `header_map` (incl. `t_mid_utc`), `header_time_ref`, `header_time_scale`. |
| `hint` | `phase` (cold/warm/locked), `center` + `radius_deg`, `scale` + `rel_tol`, `roll` + `tol_deg`, `parity`, `prior_wcs`, `star_motion` (pixel `length_px`/`angle_deg` or sky `ra_arcsec_s`/`dec_arcsec_s`), `fwhm_px` (seeing memory). |
| `detection` | `method` auto/point/trail, `threshold_sigma` (matched-filter significance, default 5), `max_sources` (default 100), `routing_fwhm_factor` (1.5), `n_trails`, `refine_centroids`, `edge_margin_fwhm`, `background`. |
| `solve` | `tiers` (default `[T0, T1, T4]`), `planner` auto/explicit, `gate`, `sip_order`, `max_t0_error_fov_frac`, `catalog_mag_limit_g`, `timeout_s`. |
| `photometry`, `streaks`, `objects` | Aperture, SNR, bands; moving-object streak detection and correlation; target (satellite) detection with `max_objects`, `min_snr` and confirmation method, independent of `detection.max_sources`. |
| `products` | Inline detections, matches, `candidate_wcs`, photometry stars, FWHM map, processing history. |
| `pin` | `algorithm` (`senpai.frame@5`, `5`, `stable`, `candidate`), `bundles`, `policy` exact/prefer. The pipeline must match the route (`senpai.frame` here); a mismatch is `400 invalid-request`. |
| `data_policy` | `retain` none/results/frames (classes defined in [`../deployment.md` section 8](../deployment.md#8-data-ownership-retention-and-contribution)), `retain_days`, `contribute`. Can only restrict the tenant policy. |
| `deadline_s`, `expert` | Sync deadline (starts when the body is fully received); whitelisted engine overrides, exact keys from `capabilities.expert_keys`, never resource limits, deadlines, timeouts or paths (hashed into `config_hash`). |

### 5.2 Semantics moved into senpai

These are client-side behaviours today; v2 makes them senpai's (detection) and astroeasy's
(solve planning), so a remote server and the in-process engine behave the same.

| Behaviour | Definition |
| --- | --- |
| `threshold_sigma` | Significance in a zero-mean matched filter with annulus noise, after global median background subtraction. No padding up to `max_sources` with noise peaks. |
| Routing (`method: auto`) | Trail path when predicted (or measured) trail length `>= routing_fwhm_factor x FWHM`; FWHM seed is trail-corrected. The trail path has no silent fallback: it fails with `detection.reason`. |
| Trail path | Star-trail model from `n_trails` trails seeded with the FWHM, centroids refined on the seeing FWHM, edge margin `L/2 + edge_margin_fwhm x FWHM`. |
| Rung planner (`planner: auto`, in astroeasy) | `T0:prior` when `prior_wcs` is given; roll-constrained T0; T0 boresight skipped when `radius_deg > max_t0_error_fov_frac x FoV`; T1 tried in both parities; T3/T4 when enabled. Every rung is reported in `solution.attempts`. |
| Server gate vs client gate | The server gate picks among candidates. With `products.candidate_wcs`, a rejected best candidate and its raw gate stats are still returned (`status: rejected_by_gate`, `wcs.is_candidate: true`), so a client can apply its own policy. |

### 5.3 Collects: `POST /v2/jobs` with `kind: collect`

Multipart: part `request` is a `CollectRequest`; each frame's bytes go in part
`image.<client_frame_id>` (or `frames[i].image.ref`). `defaults` apply to every frame;
`frames[i].frame`/`hint` override per frame. `collect.solve_every_frame: false` (default)
solves the first sidereal frame that solves and propagates the WCS along the shift chain;
each frame's `wcs_status` (`direct_fit`, `pixel_shifted`, `kernel_refined`,
`refined_unvalidated`, `none`) says which happened. Pointing-model clients must use
`/v2/analyze` instead.

`CollectResult`: `run` (completed, error, compute time, pixel rate), `frames[]` in input
order with `input_index`/`time_index`, `frame_shifts[]`, `frame_shifts_failed[]`, typed
`correlated_streaks[]`, `provenance`. SSE `frame_done` streams per-frame results.

### 5.4 Solve from sources: `POST /v2/solve/sources`

JSON body (`SourcesSolveRequest`): `image_width`, `image_height`, `sources[]` (`x`, `y`
0-based on the delivered grid, `flux`; brightest first; at most
`limits.max_input_sources`), plus the same `frame`, `hint`, `solve`, `pin` and
`data_policy` blocks as `/v2/analyze`. The server converts to astroeasy's 1-based
convention internally and honours `hint.scale` (v1 ignored the FoV hint). `pin.algorithm`
names `senpai.sources-solve`; a pin naming another pipeline is `400 invalid-request`.

Clients that detected locally with the `senpai.v2` engine send that engine's provenance
block as `client_asserted_detection_provenance`. The server validates it against the
`Provenance` schema (at most 16 KiB), echoes it under the same name, folds it into
`reproducibility_key`, and sets `provenance.input.detection_attested: false`. It never
copies it into its own `provenance.algorithm`, `software` or `data`. The producer of a
split result (local detection, remote solve) is the pair
(`client_asserted_detection_provenance.algorithm.id`, `provenance.algorithm.id`), and
clients compare both against their pins. This is the low-bandwidth path: a few kB instead
of a frame.

### 5.5 Image references (normative)

| Scheme | Rules |
| --- | --- |
| all | Allow-lists match the **parsed** scheme, host and port (and S3 bucket and key prefix), never the raw string. Bytes are streamed with a cap of `limits.max_request_bytes`. Connect and read timeouts are charged to the request deadline. No client or server credentials are forwarded, except the per-tenant S3 role below. `sha256` is required for `https` and `s3` (schema-enforced) and checked before decoding. |
| `https` | Resolve DNS once and connect to that pinned IP. Refuse loopback, RFC 1918, link-local (incl. 169.254.169.254 and other cloud metadata addresses), CGNAT 100.64/10, ULA fc00::/7, multicast and IPv4-mapped IPv6 forms of any of these. Follow no redirects (a `3xx` is `image-ref-unreachable`). Plain `http` is not accepted. |
| `s3` | Per-tenant bucket/prefix allow-lists held in tenant config, not server-wide. Reads go through a per-tenant assumed role (ExternalId = tenant id); the server's own buckets (uploads, retained data) are never resolvable through `s3:`. On `hosted`, `s3:` may be disabled in favour of presigned `https` URLs. |
| `file` | Refused by the `hosted` profile (the server will not start with `file` enabled). Elsewhere: resolve with `realpath`, open with `O_NOFOLLOW`, and the result must lie under a configured root; otherwise `image-ref-forbidden`. |
| `upload` | The id is bound to (tenant, sha256) at slot creation and expires at the slot's `expires_at`; another tenant's id is `404`. |

Errors: `image-ref-forbidden`, `image-ref-unreachable`, `image-ref-checksum-mismatch`, each
with a free-text `detail`, on self-hosted and local profiles (useful to an operator). On
`hosted` all three collapse into one `image-ref-unusable` with a generic `detail`, so the
error cannot be used to scan internal networks.

### 5.6 Decoding limits (normative)

| Check | Rule |
| --- | --- |
| Image size before decompression | From `ZNAXISn` and `ZBITPIX` for tile-compressed HDUs (their `NAXISn` describe the binary table), from `NAXISn` and `BITPIX` otherwise. Checked against `limits.max_decoded_pixels` **and** `limits.max_decoded_bytes` (pixels x abs(BITPIX)/8) before any pixel is decoded. |
| gzip wrapper | Decompressed as a stream and aborted at `max_decoded_bytes` plus a header allowance (the gzip `ISIZE` field is mod 2^32 and at the end, so it is not trusted). |
| Structure | At most `limits.max_hdus` HDUs and `limits.max_header_cards` header cards, scanned with a streaming reader before the image HDU is chosen. |
| JSON bodies | `limits.max_request_bytes` applies to JSON bodies too; `sources[]` is bounded by `limits.max_input_sources`. |
| `expert` | Can never change resource limits, deadlines, timeouts, worker settings or paths. |
| Exceeded | `413 payload-too-large`, naming the limit. |

## 6. Responses

`FrameResult` top level: `result_type`, `result_id`, `request_id`, `job_id`,
`client_frame_id`, `status` (complete/partial/failed), `pixel_convention`, `frame`
(dimensions, encoding, hashes, times, track-mode decision, metadata with sources),
`warnings[]` (structured, from what is missing), `detection`, `solution` (with `tier` and
the accepted `rung`), `seeing`, `photometry`, `streaks[]`, `objects`, `tables`,
`timings_ms`, `data_policy_applied`, `provenance`. Magnitudes are per band (`mags:
{band: {instrumental, calibrated, err}}`), matching the engine's per-band dicts.

| `solution.status` | Meaning |
| --- | --- |
| `solved` | Accepted WCS in `wcs`, plus `derived`, `quality`, `attempts`, `matches`. |
| `rejected_by_gate` | Best candidate failed the gate; candidate WCS returned if requested. |
| `no_solution` | No rung produced a candidate. |
| `insufficient_sources` | Fewer sources than the solver needs. |
| `not_attempted` | `solve` not in `ops`, or (under `planner: auto`) none of the requested tiers is available here (`reason: tier-unavailable`). |
| `error` | Solver crashed; `reason` carries the code; the rest of the result stands. |

Unavailable tiers (normative): tiers in `solve.tiers` that this server cannot run are
skipped and listed in `solution.attempts[]` with `status: skipped` and `reason:
tier-unavailable:<T>`; the solve proceeds with the available tiers and the response is
`200`. Only `planner: explicit` naming a tier this server does not serve returns `422
tier-unavailable`. Clients use `capabilities.solve.tiers[].ready` to choose a backend per
phase.

Content negotiation: `application/json` (default) or `application/vnd.senpai.v2+msgpack`
(same document). Arrow (`application/vnd.apache.arrow.stream`) is for
`/tables/{table}` only. Small source lists (at most 100) stay JSON.

## 7. Sync vs jobs

| | `POST /v2/analyze`, `POST /v2/solve/sources` | `POST /v2/jobs` |
| --- | --- | --- |
| For | One frame, slew loop, autofocus | Collects, batches, reprocessing, COLD blind solves |
| Deadline | `min(Prefer: wait=N, deadline_s, limits.sync_max_deadline_s)`, counted from the moment the request body has been fully received. Client transport timeout = transfer estimate + deadline + margin. | Job TTL |
| Overrun | `504 deadline-exceeded`, worker killed. With `Prefer: respond-async, wait=N`: `202` + `Location` + `Preference-Applied: respond-async` (if `policies.async_promotion`; default off on `hosted`). Setting `solve.timeout_s` below the deadline turns a slow rung into `attempts[].status: timeout` inside a `200`. | n/a |
| Promotion bounds | A promoted request keeps its sync worker for at most `limits.promoted_max_s` and counts against the caller's per-tenant sync concurrency until it finishes. | n/a |
| Per-tier budget | Optional `limits.sync_tier_budget_s` (e.g. T4 on hosted); an exhausted budget is `attempts[].status: timeout`. COLD blind solves belong on jobs. | none |
| Saturation | `503 overloaded` within `limits.overload_reject_ms` (default 1000), so the client can fall back. Sync never queues behind batch; admission is by tenant fair share (at most ceil(pool / active tenants) per tenant under contention, and never above `limits.per_tenant.sync_concurrency`). | Queued |
| Result | Body | `GET /v2/jobs/{id}` then `result_url`, or SSE |

Execution: every compute call runs in a killable worker process (the engine cannot be
cancelled mid-call), never on the event loop, so `/v2/health` stays responsive. Plotting
and debug artifacts are forced off in server profiles; nothing is written to the working
directory.

## 8. Errors (RFC 9457)

Errors are `application/problem+json` with `type`, `title`, `status`, `detail`,
`instance`, plus `code`, `request_id`, `retryable`, and where relevant `errors[]` (JSON
pointers), `available[]`, `retry_after_s`. `type` is
`https://ssc-ai.github.io/senpai/problems/<code>` (domain is an open question); the pages
also ship in the offline docs.

| HTTP | `code` | Retryable |
| --- | --- | --- |
| 400 | `invalid-request` (schema, unknown field, pin for another pipeline; `errors[]`), `idempotency-key-missing` (only on routes in `policies.idempotency_required`) | no |
| 401 / 403 | `unauthenticated` / `forbidden` (token lacks the operation's scope; tenant policy forbids) | no |
| 404 | `not-found` (also for another tenant's ids) | no |
| 409 | `idempotency-in-flight` (sync routes only) | yes, after `Retry-After` |
| 410 | `api-version-gone` (sunset route on hosted; declared on every v1 operation) | no |
| 412 | `api-minor-unsupported` (server older than `Senpai-Api-Min-Minor`) | no |
| 413 | `payload-too-large` (any limit in section 5.6, or frame count) | no |
| 415 | `unsupported-format` | no |
| 422 | `unreadable-image`, `lossy-encoding-not-allowed`, `image-ref-forbidden`, `image-ref-unreachable`, `image-ref-checksum-mismatch` (hosted: `image-ref-unusable`), `idempotency-key-reused`, `algorithm-unavailable`, `bundle-unavailable`, `tier-unavailable` (`planner: explicit` only), `operation-unavailable` | no (`image-ref-unreachable`: maybe) |
| 429 | `rate-limited` + `Retry-After`, `RateLimit`, `RateLimit-Policy` | yes |
| 500 | `internal` | maybe |
| 503 | `overloaded`, `not-ready` + `Retry-After` | yes |
| 504 | `deadline-exceeded` | yes |

422 is never used for "did not solve" (v1 used it for both validation failures and failed
solves).

## 9. Idempotency and dedup

| Item | Rule |
| --- | --- |
| Header | `Idempotency-Key`: a token of 1 to 255 chars from `[A-Za-z0-9._:-]`; the double-quoted structured-field form of the IETF draft is also accepted (quotes are not part of the key). Optional, except on routes listed in `policies.idempotency_required`. One key per exposure, for example `<run_id>:<target_index>:<attempt>`, allocated before upload. |
| Scope | (tenant, key) |
| Fingerprint | Frames: `sha256(pixel_sha256 || JCS(request without client_frame_id, data_policy and deadline_s))`. Sources-solve: `sha256(sources_sha256 || JCS(same exclusions))`. Collects: the ordered frame pixel hashes plus the request. The `Prefer` header is not included. The pin and all knobs are included, so the same frame with different knobs is a different request. |
| Replay | Same key, same fingerprint, finished: stored result with `Senpai-Replayed: true`. |
| In flight | Sync routes: `409 idempotency-in-flight`; re-send the identical request with the same key after `Retry-After`. `/v2/jobs`: `202` with the existing job's `Location` in every job state. |
| Mismatch | `422 idempotency-key-reused`. |
| Expiry | `capabilities.policies.idempotency_ttl_s` (default 24 h), capped by the result's retention; after the result is purged a replay reprocesses. |
| Pixel hash | `sha256("senpai-px/1\0" || "<H>x<W>" || tag || C-order little-endian bytes)`, where the array is the decoded image **after** integer BZERO/BSCALE is applied (so FITS uint16 hashes as unsigned 16-bit, floats with BSCALE applied), converted to little-endian, and `tag = f"{dtype.kind}{dtype.itemsize}"` (`u2`, `i2`, `f4`, ...; no byte-order character). astropy returns big-endian arrays for plain FITS (`>i2`, `>f4`) and native little-endian for decoded tile-compressed HDUs, so `dtype.str` must not be used. Plain, Rice-lossless, tile-GZIP and gzip-wrapped encodings of one frame then share the hash. `senpai-client` ships the reference `pixel_sha256(array)`; a golden test hashes one frame stored all four ways. |
| Result cache | Optional, per tenant, keyed by `reproducibility_key` plus `ops` and `products`. Honours `retain`: results under `retain: none` are never cached. A hit sets `Senpai-Cache: hit` and `provenance.cache.original_result_id`. Off by default on self-hosted. |

## 10. Auth, tenancy, limits

| Item | Rule |
| --- | --- |
| Auth modes | `static_keys` (hashed key file, `snp_` prefix with checksum so leaks are scannable; self-hosted/air-gapped default). Key record: `{key_id, sha256(secret), tenant_id, scopes, not_after, disabled}`; `senpai-admin keys add\|rotate\|revoke`. `oidc` (JWT from a configured issuer): config must pin `iss`, `aud` and the allowed algorithms; the server refuses to start in `oidc` mode without `aud`. `none`: only on a loopback bind and only with `auth.allow_none_on_loopback: true`; the server refuses to start with `none` on any other bind. |
| Hosted | Uses `oidc` like any deployment. The gateway forwards the caller's JWT, or mints a short-lived JWT (`aud=senpai-<deployment_id>`, `tenant_id`, `scope`) that the standard `oidc` mode verifies. There is no gateway-header auth path, so a request that reaches a senpai pod directly cannot assert a tenant. |
| Scopes | `analyze`, `jobs`, `results:read`, `data:delete`, per operation as in section 3 (each operation's `security` in `openapi.yaml`). A token without the scope gets `403`. Telescope tokens should carry `analyze` (and `jobs`) only. The FakeServer enforces scopes. |
| Tenancy | Principal maps to `tenant_id`. Idempotency keys, jobs, results, tables, SSE streams, uploads, retention, frame records and contribution grants are all keyed by tenant; a foreign id is `404`. |
| Loopback (local profile) | The default is a token file (`senpai-admin init --profile local`, file ACL in `../deployment.md` section 2). The server always checks `Host` against `127.0.0.1:<port>`, `localhost:<port>`, `[::1]:<port>` and rejects any request carrying an `Origin` not on the CORS allow-list (default: reject all), so a web page on the observatory PC cannot drive the service through CSRF or DNS rebinding, even with `none`. |
| Limits | Always in `capabilities.limits`, as the caller's effective values: request bytes, decoded pixels and bytes, HDUs, header cards, input sources, frames per collect, `max_sources`, `max_stars_per_frame` (memory guard; result-affecting, so part of `config_hash`), sync default/max deadline, sync concurrency, per-tier sync budget, promotion bound, overload bound, rate, and `per_tenant` (sync concurrency, compute seconds per hour, queued jobs, requests per minute). Per-tenant limits, fair-share admission and compute metering are in the open-source server, not only in a gateway. |
| CORS | Allow-list, default empty. Never `*` with credentials. |
| Version headers | Every response, including errors and `204`s: `Senpai-Api-Version: 2.N`, `Senpai-Request-Id` (declared on every response in `openapi.yaml`). Client may send `Senpai-Api-Min-Minor`. |

## 11. Migration from v1

v1 routes stay with `Deprecation` (RFC 9745), `Link; rel="deprecation"` and later `Sunset`
(RFC 8594) headers. "Unchanged" refers to request and response bodies only: from 3.0 the
v1 handlers sit behind the same auth (tenant and scopes), `capabilities.limits` (bytes,
decoded size, frames per request, `max_stars_per_frame`), rate limits and killable worker
pool as v2, and use the tenant's default retention (hosted: `none`). On hosted, v1 is
enabled per tenant, default off (there is no hosted v1 install base). Timeline and policy:
[`versioning-and-provenance.md` section 6](versioning-and-provenance.md#6-compatibility-and-deprecation).

| v1 | v2 |
| --- | --- |
| `POST /senpai/detect`, `/detect/upload`, `/sidereal`, `/rate` (base64 JSON list) | `POST /v2/jobs` `kind: collect` (multipart); `POST /v2/analyze` for one frame |
| `GET /senpai/` | `GET /v2/health` + `GET /v2/capabilities` |
| `GET /astrometry/` (config dump) | Removed in 3.0 (security). `/v2/capabilities` exposes no paths. |
| `POST /astrometry/solve/sources` (`StarListImage`) | `POST /v2/solve/sources`; scale hint honoured; `200` + `solution.status` |
| `POST /astrometry/solve/fits` | `POST /v2/analyze` with `ops: [detect, solve]` |
| `AstrometrySummary` (crval, scale, rotation) | `solution.wcs.header` (full, with SIP) + `solution.derived` |
| `correlated_streaks: list[dict]` | typed `CorrelatedStreak[]` |
| `limiting_mag_5sigma` / `_3sigma` | `photometry.limiting_mag[]` with `{value_mag, snr, completeness}` |
| `FrameResult.index` (time-sorted) | `client_frame_id`, `input_index`, `time_index` |
| `senpai_version` dropped, `error_message` logged only | `provenance`, `run.error`, per-block `status`/`reason` |

senpai fixes required before or with 3.0 (from the 2.8.2 code review):

| Fix | Where today |
| --- | --- |
| Loader reads Rice `CompImageHDU` and gzip (both fail now) | `senpai/engine/models/images.py:204-219` |
| `pix2world_0based`/`world2pix_0based` add 1 to already 0-based input | `senpai/engine/models/astrometry.py:371-412` |
| Frame centre at `(N+1)/2` in the 0-based API | `senpai/engine/models/astrometry.py:493-499` |
| 0-based centroids passed to astroeasy's 1-based `Detection` | `senpai/astrometry.py:47-50` |
| RA/Dec rounded to 1e-4 deg, x/y to 0.01 px | `senpai/engine/models/starfield.py:53-57,90-94` |
| `fov_*` hint ignored; scale bounds from global config | `senpai/astrometry.py:25-56` |
| Per-request knobs hit a frozen global `AppConfig` | `senpai/core/config.py:628-670` |
| `if ra and dec:` treats 0.0 as missing (also sends RA/Dec = 0 frames down the Alt/Az path) | `senpai/engine/utils/fits_io.py:294` |
| Alt/Az-only headers run `SkyCoord(altaz).icrs`, which needs IERS and can download or fail offline; the engine must set `iers.conf.auto_download=False`, `utils.data.conf.allow_internet=False`, `iers_degraded_accuracy='warn'` at import | `senpai/engine/utils/fits_io.py:337-350` (called from `processing/sidereal.py`, `rate.py`, `collect.py`, `models/metadata.py`) |
| Handlers run CPU work on the event loop; process pool unused | `senpai/api/main.py:37-59` |
| API reports YAML `version: 0.1.0`, not the package version | `senpai/api/main.py:81-89` |
| Startup checks' results ignored | `senpai/api/main.py:47-48` |
| PNG written to CWD on every `/solve/fits` | `senpai/api/routes/astrometry.py:113-120` |
| CORS `*` with credentials; YAML `cors` ignored | `senpai/api/main.py:96-102` |
