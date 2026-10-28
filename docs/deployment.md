# Deploying and updating senpai

**Status:** draft for owner review, 2026-10-19. Design only; nothing here is implemented.
Items marked **[proposal]** are recommendations; **[unverified]** marks facts not confirmed
yet. Owner decisions are collected in [section 12](#12-open-questions-for-the-owner).

Related: [`api-v2/README.md`](api-v2/README.md) (the HTTP contract),
[`api-v2/versioning-and-provenance.md`](api-v2/versioning-and-provenance.md) (version axes
and provenance), [`RELEASING.md`](RELEASING.md) (today's PyPI release process).

## 0. Summary

1. **One set of artifacts, four profiles.** Every release publishes the same signed wheels,
   images, data bundles and release manifest. Hosted, self-hosted, air-gapped and local
   differ only in where artifacts come from (our registry, a mirror, a USB medium) and which
   config `profile` is active. The hosted service has no private build.
2. **Code, algorithm and data update separately** and are named separately (API `2.N`,
   software SemVer + digest, algorithm `senpai.frame@N`, config hash, bundle digests). Every
   result records all of them.
3. **The operator owns the clock.** Nothing on a self-hosted, air-gapped or local install
   changes without an explicit `activate`; algorithm versions never change automatically.
4. **No outbound network by default** outside hosted: no telemetry, no update check unless
   opted in, no online catalogs, no IERS downloads, no Docker pulls, no CDN docs assets.
5. **Verification needs no network.** Every release is signed keyless with Sigstore by
   its GitHub Actions release workflow; we hold no long-lived signing keys (owner decision
   2026-10-19). Kits carry the Sigstore bundles and a copy of Sigstore's
   `trusted_root.json`, so air-gapped sites verify offline against a signer identity and
   root digest confirmed out of band. Rollback protection is a monotonic sequence number
   per (product, channel). The first install is verified with an independently obtained
   `cosign`, not with code from the medium being verified.
6. **Rollback is a pointer flip.** Images are pinned by digest with the previous pins kept;
   data bundles are content-addressed and side by side; the state DB uses expand/contract
   migrations.
7. **Frames belong to the sender.** Hosted keeps no frame bytes by default; contribution
   back to us needs a tenant agreement and a per-request opt-in; self-hosted and air-gapped
   contribute only by manual export.
8. **astrometry.net belongs to astroeasy** (owner decision 2026-10-19). senpai and astroeasy
   keep their own licences (senpai: after v2.8.2, AGPL-3.0; astroeasy MIT).
   astrometry.net (GPL) ships only in astroeasy's `astrometry-cli` image and is
   only ever run as a separate program; senpai never installs or builds it. Server
   containers reach it through an astroeasy sidecar backend, never the Docker socket
   (section 3.5).
9. **Data bundles credit their sources, and can be rebuilt** (owner direction 2026-10-19).
   Every bundle and kit carries its upstream credits and licence text; every bundle kind has
   a pinned build recipe, so "pull from our mirror" and "build it yourself" are both
   supported (section 3.1.1).
10. **Fully local everywhere, Windows included.** All tiers run on the observatory PC; on
    Windows, blind tiers are heavy and we suggest a server for them when one exists
    (section 6.4).

## 1. Principles

| # | Principle |
| --- | --- |
| P1 | The operator owns the data and the clock. No silent behaviour change. |
| P2 | Code, algorithm and data change independently and are named independently. |
| P3 | Every result says exactly what produced it. |
| P4 | Verification never needs the network: a USB medium plus the signer identity and `trusted_root.json` digest we publish is enough. |
| P5 | The hosted service has no special powers. Auth, limits, retention and jobs are in the open-source server. |
| P6 | Every outbound call has an off switch, is off by default outside hosted, and fails quietly when off. |

## 2. Deployment profiles

| | (a) Hosted (ours) | (b) Self-hosted, connected | (c) Air-gapped | (d) Local (observatory PC) |
| --- | --- | --- | --- | --- |
| Who runs it | us | the user's org, on its server | the user's org, no internet ever | the user, next to TheSkyX / the capture app |
| Form | images on our cluster | compose (Helm later) or wheels + `senpai serve` | same images or wheels, loaded from an offline kit | in-process library (`senpai.v2.engine`) or localhost service |
| Data held by | us, per retention policy | the user | the user | the user |
| Bind | 0.0.0.0 behind TLS gateway | operator | operator | **127.0.0.1 only** |
| Auth | `oidc` (gateway forwards or mints the JWT) | `static_keys` (or `oidc`) | `static_keys` | token file by default; `none` only with `auth.allow_none_on_loopback: true`; `Host`/`Origin` checks always on |
| TLS | gateway | `init` takes `--tls-cert/--tls-key` or generates a site CA; compose includes a TLS proxy | same as (b) | loopback only, plain HTTP |
| Online catalogs (`gaia`, `sdss`) | forbidden | only with the `[online]` extra and an explicit flag | **refuse to start** | forbidden |
| astropy `allow_internet`, IERS `auto_download` | false | false | **false, enforced before astropy loads** | false |
| Update check | n/a | opt-in | **no code path** | opt-in |
| Docs UI (`/docs`) | off or behind auth | on, assets served from the package | on, local assets | on, local assets |
| Plotting / debug artifacts | forced off | off unless set | off unless set | off unless set |
| astrometry.net tier (astroeasy's `astrometry-cli`, section 3.5) | sidecar container | sidecar container (optional) | sidecar container (optional) | optional. Linux: astroeasy's local or docker backend. Windows: `astrometry-cli` under Docker Desktop or WSL2 (heavy; a server is suggested for blind solves when available, section 6.4) |
| Retention default | `none` | operator | operator | operator (the client stores its own frames) |
| Contribution | per-request opt-in + tenant agreement | export only | export only | export only |
| Channel | continuous `stable` | `stable` or `lts` | `lts` | the senpai line the client kit was built for: client kits are published per senpai line (e.g. `<client>-<ver>+senpai-lts-2027-win64.zip`), so a site's local fallback can serve the same frozen algorithm versions as its server |

`profile:` is a top-level key of the server config. It sets these defaults and **refuses to
start** on incompatible settings (for example `profile: airgapped` with
`star_catalog.type: gaia`, `auth: none` on a non-loopback bind, `image_ref` scheme `file`
on `hosted`, `oidc` without `aud`, or a non-loopback bind without TLS unless
`--insecure-http` is passed).

Loopback service on the observatory PC (profile d): any web page open in a browser on that
PC can send a multipart POST to `127.0.0.1` without a CORS preflight, and DNS rebinding can
read responses. The server therefore always validates `Host` (`127.0.0.1:<port>`,
`localhost:<port>`, `[::1]:<port>`) and rejects any request with an `Origin` not on the
CORS allow-list (default: reject all), even in `none` mode. The token file is the default:
`senpai-admin init --profile local` creates it with inheritance removed and access only for
the service account or interactive user, SYSTEM and Administrators (`icacls
/inheritance:r /grant`); on Linux mode 0600. Clients refuse a token file readable by
`BUILTIN\Users` (Windows) or by group or others (POSIX). DPAPI or Windows Credential
Manager are acceptable alternatives on Windows.

TLS (profiles b and c): `senpai-admin init --profile self_hosted|airgapped` takes
`--tls-cert/--tls-key`, or generates a site CA and server certificate and exports
`site-ca.pem` for clients. The compose files include a
TLS-terminating proxy. Bearer tokens and frames never travel over plain HTTP off the
loopback interface.

`senpai-admin doctor` prints all version axes, the active bundles, enabled tiers, and the
egress posture. On Linux it can run a no-network self-test (`unshare -rn`).

## 3. What we publish

All of these are listed by digest in one signed release manifest (section 3.3).

| Artifact | Form | Where | Notes |
| --- | --- | --- | --- |
| `astro-senpai` | wheel + sdist. Extras: `[engine]` (numerics only: numpy, astropy, photutils, scipy, `astroeasy[cascade]`), `[server]` (fastapi, uvicorn, multipart), `[admin]` (bundle and update tooling), `[plot]` (matplotlib, seaborn), `[online]` (astroquery) | PyPI, Trusted Publishing + PEP 740 attestations | 3.0 moves FastAPI and plotting out of core. Scripts: `senpai` (`serve`, CLI), `senpai-admin`. |
| `senpai-client` | pure-Python wheel (httpx, pydantic), generated from `openapi.yaml`, ships `FakeServer` | PyPI | Hosted-only users install only this. |
| Lock | `uv.lock` **committed**; CI and images build with `--frozen` (astroeasy's model) | repo | `lock_sha256` is recorded in provenance. |
| Wheelhouses | `senpai-<ver>-wheelhouse-<platform>-py313.tar` for `linux_x86_64`, `linux_aarch64` [decide], `win_amd64` | release assets + our HTTPS mirror | Built in CI from the lock with hashes; `pip install --no-index --require-hashes`. |
| `senpai` image | OCI, multi-arch, non-root, read-only rootfs, `HEALTHCHECK`, OCI labels (version, revision), base Debian 13-slim or Ubuntu 24.04 [decide] | `ghcr.io/ssc-ai/senpai@sha256:...` [decide registry] | Engine + server, native tiers T0/T1. **senpai code only (AGPL-3.0-or-later): no astrometry.net, no data baked in.** |
| `astrometry-cli` image (**astroeasy's, referenced, not built by senpai**) | `solve-field` only, built from `astroeasy/dotnet/Dockerfile` at a **pinned** astrometry.net commit | published by the astroeasy repo and its release process, e.g. `ghcr.io/ssc-ai/astrometry-cli@sha256:...` [decide registry] | GPL-3.0-or-later; carries the GPL source offer (source tarball as an OCI referrer and in offline kits). Run only as a separate program by astroeasy's docker backend (plain hosts, Docker Desktop, WSL2) or sidecar backend (server profiles), section 3.5. senpai's release manifest pins the digest it was tested with. Replaces today's locally built, unpinned image (astroeasy `make build-docker`). |
| Data bundles | OCI artifacts (custom media types, ORAS) **and** `.tar.zst`, same signed manifest | `ghcr.io/ssc-ai/senpai-data/<kind>:<name>-<version>` + HTTPS mirror | Section 3.1. Each carries upstream credits, licence text and a rebuild recipe (section 3.1.1). |
| Release manifest | `release-<ver>.json` + `.sigstore.json` (keyless Sigstore bundle) | release assets; channel index `channels/<channel>.json`, signed the same way | Section 3.3. |
| Deploy kit | `senpai-deploy-<ver>.tar.gz`: compose files pinned by digest (incl. the TLS proxy and the optional `astrometry-cli` sidecar), example configs per profile (incl. air-gapped), systemd unit, release manifest + its Sigstore bundle, `senpai-admin.pyz`, `SHA256SUMS` + `SHA256SUMS.sigstore.json`, `trusted_root.json` | release assets; also inside every offline kit | The files we use for hosted. Helm later. |
| Offline kits | built by `senpai-admin export` (section 6.3); code-only kits per platform as release assets | release assets / on demand | |
| SBOM | SPDX per image and wheelhouse | OCI referrers + release assets | |
| Contract | `docs/api-v2/openapi.yaml` | repo + release asset | |

### 3.1 Data bundles: one format for every kind

Modeled on sstrc7's manifest (per-file size and SHA-256), which is the only artifact with
integrity checks today.

```
<bundle>/
  bundle.json            manifest (below)
  bundle.sigstore.json   keyless Sigstore bundle over bundle.json (our bundles)
  bundle.json.sig        site-key signature (optional; site-built bundles only)
  NOTICE                 upstream credits, citations and licence/terms text (section 3.1.1)
  recipe/                build recipe: script, pinned upstream inputs, expected digest
  files/...              payload
```

```json
{
  "schema": "senpai.bundle/1",
  "kind": "tetra3-db",
  "name": "fov0.5-1.5_g11",
  "version": "2026.10.0",
  "created": "2026-10-12T00:00:00Z",
  "builder": {"tool": "senpai-admin 3.1.0", "build": "g1a2b3c"},
  "inputs": [{"kind": "gaia-mirror", "digest": "sha256:..."}],
  "params": {"max_fov_deg": 1.5, "min_fov_deg": 0.5, "mag_limit_g": 11.0},
  "files": [{"path": "files/db.npz", "size": 312004512, "sha256": "..."}],
  "coverage": {"dec_deg": [-30, 90], "fov_deg": [0.5, 1.5]},
  "license": {
    "spdx": "LicenseRef-CC-BY-NC-3.0-IGO",
    "attribution": "ESA/Gaia/DPAC",
    "notice": "NOTICE",
    "upstreams": [{
      "name": "Gaia DR3",
      "release": "DR3, gaiadr3.gaia_source",
      "credit": "ESA/Gaia/DPAC",
      "citation": "Gaia Collaboration, Prusti et al. 2016, A&A 595, A1; Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1",
      "license": "CC BY-NC 3.0 IGO; ESA ESDC terms and conditions",
      "terms_url": "https://www.cosmos.esa.int/web/gaia-users/license",
      "url": "https://gea.esac.esa.int/archive/"
    }]
  },
  "recipe": {
    "script": "recipe/build.py",
    "upstream_inputs": [{"url": "...", "sha256": "..."}],
    "expected_digest": "sha256:...",
    "reproducibility": "byte-identical"
  },
  "requires": {"senpai": ">=3.0,<4", "readers": {"tetra3-db": 1}},
  "origin": "upstream"
}
```

Bundle digest = SHA-256 of the canonical (JCS) `bundle.json`; it covers every file hash,
the `license` block and the recipe. Results cite this digest.

| Kind | Contents | Notes | Build it yourself (recipe) |
| --- | --- | --- | --- |
| `anet-index` | astrometry.net index files by series and scale subset (e.g. `5200-lite-s04-06`, `4100-s07-09`) | Mirrored from upstream (NERSC `temp/` path is fragile). | Download the pinned upstream files from data.astrometry.net / NERSC and check each SHA-256: byte-identical. |
| `tetra3-db` | `.npz` per FoV range and magnitude limit | Records the input mirror digest. Built with astroeasy's vendored tetra3 (cedar-solve, Apache-2.0) from a `gaia-mirror`. | astroeasy `build-tetra3-db` from the recorded `gaia-mirror` digest and params. Byte-identical only if the `.npz` writer is deterministic [verify]; otherwise the recipe documents what differs and checks a digest over the pattern arrays. |
| `gaia-mirror` | HEALPix tiles by depth (`dr3-g16`, `dr3-g18`, `dr3-g20`) and optional declination band | Needs a new manifest (release, G limit, row count, tile hashes) and the depth chop (designed, not built). | Pinned ADQL query against the ESA Gaia archive (DR3 is a fixed release), rows sorted before writing tiles: expected byte-identical [verify]. |
| `sstrc7` | wraps the existing sstrc7 manifest; optional declination band | Gaia-derived (G < 18). | `sstrc7 get` at the pinned release, checked against its manifest: byte-identical. Rebuilding sstrc7 itself from Gaia is sstrc7's own process [verify]. |
| `iers` | Raw tables, not a wheel: `finals2000A.all` (IERS-A) and `Leap_Second.dat` | Refreshed in every LTS patch. Loaded explicitly at start by the engine and by clients (section 3.2); installing a wheel into a hash-pinned environment would break `--require-hashes` and make the environment digest false. | Download from the recorded IERS/USNO URLs. Upstream files change daily/weekly, so a later download is a new version; the recipe records URL, fetch time and hash and verifies a saved copy, not the live file. |
| `sensor-profile` | astroeasy `SensorProfile` YAML | Usually site-built. | astroeasy characterisation on the site's frames. |
| `weights` | starcsp checkpoints, **safetensors only** (never pickle) | Future. | Training recipe; not byte-identical (GPU nondeterminism), documented. |
| `golden` | regression corpus | CI and `--canary`. | Our frames; no upstream rebuild. |

### 3.1.1 Redistribution, attribution and build-it-yourself

Owner direction (2026-10-19), **decided in principle**; the per-source terms below still need
verification (rows marked [verify]). This is not legal advice.

| Rule | Design |
| --- | --- |
| Redistribution | We may redistribute derived data (astrometry.net index files, tetra3 databases, Gaia-derived catalog slices, sstrc7 data, IERS tables) **provided every bundle fully and visibly credits its upstream sources and carries their licence/terms text.** |
| Where credits appear | (1) `bundle.json` `license`: `spdx`, a one-line `attribution`, and `upstreams[]` listing every upstream with name, exact release used, credit, citation, licence/terms and URL. (2) A `NOTICE` file inside every bundle, and a combined `NOTICE`/`ATTRIBUTION` file in every kit and deploy kit. (3) `GET /v2/capabilities` lists each active bundle's licence and upstreams (`bundles.active[].license`); every `provenance.data.used[]` entry carries the bundle's `attribution`, so a result maps bundle digest to credit. |
| Two paths, always | "Pull from our mirror / bundle" **or** "build it yourself". Every bundle kind has a documented, reproducible recipe (script + pinned upstream inputs + expected output digest; table above) so users can rebuild from upstream (ESA Gaia archive, data.astrometry.net, IERS, ...) and get a byte-identical or digest-verifiable result where feasible; where not, the recipe documents what differs. `senpai-admin bundle build <kind> --recipe` runs it on a staging machine. |
| Self-built bundles | First-class: `origin: site`, signed by the site key or accepted unsigned with the explicit `--allow-unsigned-site-bundle` flag (section 3.1). A self-built bundle that matches our expected digest is the same bundle and is cited the same way. |
| Commercial use | Gaia-derived data are CC BY-NC 3.0 IGO and ESA requires prior authorisation for commercial use (table below). Whether our hosted service or any paid kit is commercial use, and the ESA authorisation if so, is a per-source verification item [verify]. |

Upstream terms as found on 2026-10-19:

| Upstream | In bundle kinds | Terms found | Credit / citation to carry | Sources |
| --- | --- | --- | --- | --- |
| Gaia DR3 | `gaia-mirror`, `tetra3-db` (built from the mirror), `sstrc7` (Gaia-derived) | "Gaia data are distributed under the CC BY-NC 3.0 IGO license" (Gaia user licence page). DR3 docs: "open and free to use, provided credit is given to 'ESA/Gaia/DPAC'". ESA ESDC terms: commercial use (direct or indirect financial gain) needs prior request to data.licences@esa.int; no implied ESA endorsement. Redistribution of derived products is not addressed explicitly [verify]. | "ESA/Gaia/DPAC"; DR3 acknowledgement text; Gaia Collaboration, Prusti et al. 2016, A&A 595, A1; Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 | [licence](https://www.cosmos.esa.int/web/gaia-users/license), [DR3 credit](https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/), [ESDC terms](https://www.cosmos.esa.int/web/esdc/terms-and-conditions) |
| Gaia DR2 | `anet-index` 5200 series | Same ESA terms as DR3 [verify DR2-specific wording]. | "ESA/Gaia/DPAC"; Gaia Collaboration, Brown et al. 2018, A&A 616, A1 | as above |
| astrometry.net index files, 4100 series | `anet-index` | Built from Tycho-2. data.astrometry.net publishes no licence statement [verify with the astrometry.net maintainers]; Debian packages them (`astrometry-data-tycho2`) as BSD-2-Clause, based on Høg's informal OK for Tycho-2. | Lang et al. 2010, AJ 139, 1782; Tycho-2 credit (below) | [data.astrometry.net](https://data.astrometry.net/), [Debian copyright](https://sources.debian.org/src/tycho2/2-4/debian/copyright) |
| astrometry.net index files, 5200 series (LITE, HEAVY) | `anet-index` | Built from Tycho-2 + Gaia DR2 (HEAVY adds DR2 magnitudes, proper motions, parallaxes), so Gaia terms apply. No licence statement on data.astrometry.net [verify]; Fedora labels `astrometry-data-52xx` GPL-2.0-or-later [verify what that label covers]. | Lang et al. 2010; Gaia DR2 and Tycho-2 credits | [data.astrometry.net](https://data.astrometry.net/), [Fedora package](https://packages.fedoraproject.org/pkgs/astrometry/astrometry-data-5205) |
| Tycho-2 | via 4100/5200 indices | The distribution has no licence note (Debian copyright file); the author agreed by e-mail to a BSD-style clause for Debian. CDS: each dataset's own licence applies, commercial use depends on origin, cite the dataset and CDS [verify]. | Høg et al. 2000, A&A 355, L27; CDS catalogue I/259 | [CDS I/259](https://cdsarc.cds.unistra.fr/viz-bin/cat/I/259), [CDS terms](https://cds.unistra.fr/legals/) |
| IERS `finals2000A.all`, `Leap_Second.dat` | `iers` | No published licence or terms page found [verify with IERS]. IERS reportedly allows copying with the source credited (tz mailing list, 2017, second-hand); redistributed inside astropy-iers-data (BSD-3-Clause package, no separate data terms). | IERS Rapid Service/Prediction Centre (USNO) for finals2000A; IERS Earth Orientation Centre (Paris Observatory) for Leap_Second.dat | [tz list 2017](https://mm.icann.org/pipermail/tz/2017-April/024967.html), [tz list 2023](https://mm.icann.org/pipermail/tz/2023-December/058331.html), [astropy-iers-data](https://github.com/astropy/astropy-iers-data) |
| tetra3 databases | `tetra3-db` | Our DBs are built from the `gaia-mirror` (Gaia DR3 terms). The tetra3 code (ESA tetra3, cedar-solve fork vendored in astroeasy) is Apache-2.0; ESA tetra3 also supports BSC5, Hipparcos and Tycho as sources. | Gaia DR3 credits; tetra3 / cedar-solve NOTICE | [ESA tetra3](https://github.com/esa/tetra3), [cedar-solve](https://github.com/smroid/cedar-solve) |
| sstrc7 data | `sstrc7` | sstrc7 code is MIT; the data are Gaia-derived (G < 18), so Gaia terms apply; sstrc7's own upstream terms [verify]. | Gaia DR3 credits; sstrc7 | sstrc7 README |

`origin: site`: bundles built locally record their inputs by digest. Typical uses are a
tetra3 DB from the site's mirror, or the site's own astrometry.net index files:

```
senpai-admin keys add-site site-key.pub                  # register the site's signing key
senpai-admin bundle build anet-index --from /data/my-indexes --name site-5200-sub \
    --sign-with site-key                                  # or omit --sign-with
senpai-admin bundle install site-5200-sub.tar.zst        # stage
senpai-admin bundle activate --set <set name>            # takes effect after restart
```

`import` and `bundle install` accept a site bundle only if it is signed by a key in the
trust store (`keys add-site`), or with the explicit `--allow-unsigned-site-bundle` flag,
which is logged and recorded in provenance (`origin: site`, `signed: false`). Upstream
bundles always need our keyless signature (section 4). The site key is the site's own
business: we do not issue, hold or rotate it.

### 3.2 On-disk store (every profile)

Linux `/var/lib/senpai` (a volume in containers); Windows `%ProgramData%\senpai`.

```
store/
  bundles/<kind>/<digest>/...       immutable, side by side
  sets/<set-digest>.json            a bundle set (section 3.4)
  active.json                       {"product": "senpai", "channel": "lts-2027", "release": 3,
                                     "bundle_set": "sha256:...", "previous": {...}}
  active/anet-index/astrometry.cfg  generated by activate: one add_path per active anet-index bundle
  deploy/compose.yaml               current pins   (written only by activate)
  deploy/compose.next.yaml          staged pins    (written by upgrade)
  deploy/compose.prev.yaml          previous pins  (kept by activate, restored by rollback)
  releases/<product>/<channel>/<seq>-<ver>/release.json(+sigs)   imported manifests
  trust/trusted_root.json           Sigstore trusted root in use (section 4)
  trust/identities.json             pinned signer identities; trust/site/ for site keys
  state/verified.json               per-file verification stamps
```

`senpai-admin init --store <path>` creates the store; every later command (`import`,
`bundle install`, `activate`) writes into it, so `init` comes first.

The server reads `active.json` at start and reports it in `GET /v2/capabilities`. Switching
is `senpai-admin activate` followed by a restart (hot reload later). Server containers
mount the store read-only.

How the engine finds data: config refers to store bundles, not paths, for example
`star_catalog: {type: gaia_local, bundle: gaia-mirror/dr3-g16}` and
`tetra3: {bundle: tetra3-db/fov0.5-1.5_g11}`. At start the server resolves each reference
through `active.json` to `bundles/<kind>/<digest>/files/...`, so a data update never
requires editing a path. Today `star_catalog.path` is required for local catalogs
(`senpai/core/config.py:158`); 3.0 makes `bundle` an alternative to `path`. A raw `path` is
still accepted (transition, development): the engine then hashes the files at load and
records them in `provenance.data.used` with `origin: path` and their combined SHA-256.
Clients that pin bundles (`pin.bundles`) are refused (`422 bundle-unavailable`) by a server
or engine that serves that data from a raw path.

astrometry.net index files: `solve-field` reads one data directory or a config file, so
`activate` generates `store/active/anet-index/astrometry.cfg` with one `add_path` per active
`anet-index` bundle (or a read-only symlink farm for astroeasy's Docker backend, which
bind-mounts a single directory). The `astrometry-cli` container (astroeasy's docker backend
or sidecar, section 3.5) mounts that read-only. `provenance.data.anet_indices_used` maps each index file solve-field reports
back to the digest of the bundle it came from.

Integrity at run time: file hashes are checked at `import` and `bundle install`, and again
at `activate` and at every server start. Large stores are verified once per file and
cached in `state/verified.json` keyed by (inode, size, mtime, bundle digest); a changed
stamp triggers a re-hash. Capabilities report `verified` and `verified_at` per bundle. A
bundle that fails or has not been verified is not served: the tiers that need it report
`ready: false` (or, with `--serve-unverified` for diagnosis, results carry warning
`UNVERIFIED_BUNDLE`).

### 3.3 Release manifest

```json
{
  "schema": "senpai.release/1",
  "product": "senpai",
  "version": "3.1.0",
  "channel": "stable",
  "seq": 41,
  "released": "2026-11-15",
  "eol": null,
  "api": {"version": "2.3", "min_supported_minor": 0, "deprecated": ["v1"], "v1_sunset": "2027-12-01"},
  "algorithms": {"senpai.frame": 5, "senpai.collect": 8, "senpai.sources-solve": 3},
  "algorithm_changes": {"senpai.frame": {"from": 4, "summary": "...", "golden_diff": "..."}},
  "software": {"lock_sha256": "...", "git_sha": "..."},
  "artifacts": {
    "images": {"senpai": "sha256:..."},
    "external_images": {"astrometry-cli": {"digest": "sha256:...", "from": "astroeasy 1.4.0", "anet_commit": "..."}},
    "wheels": {"astro_senpai-3.1.0-py3-none-any.whl": "sha256:..."},
    "wheelhouses": {"win_amd64-py313": "sha256:...", "linux_x86_64-py313": "sha256:..."}
  },
  "recommended_bundles": {"iers/iers": "sha256:...", "tetra3-db/fov0.5-1.5_g11": "sha256:..."},
  "optional_bundles": {"golden/senpai-golden": "sha256:..."},
  "compat": {
    "bundle_readers": {"tetra3-db": [1], "gaia-mirror": [1]},
    "state_db": {"schema": 3, "rollback_to": "3.0.x"}
  },
  "upgrade_stops": [],
  "security": {"advisories": []}
}
```

The release manifest also lists the engine's dependency set (name, version, wheel tag) so
that `environment_matches_release` can be computed (versioning doc, section 7).

`external_images` are referenced, not built: `astrometry-cli` is astroeasy's artifact, signed
keyless by astroeasy's release workflow (its own pinned identity, section 4); senpai's manifest pins the digest that senpai's release was
tested with, so offline kits can carry it.

### 3.4 Bundle sets and the bundle verbs

A bundle set is a named list of bundle references. One schema everywhere:

```json
{
  "schema": "senpai.bundle-set/1",
  "name": "narrowfield-win",
  "bundles": {
    "tetra3-db/fov0.5-1.5_g11": "sha256:...",
    "gaia-mirror/dr3-g16": "sha256:...",
    "iers/iers": "sha256:..."
  }
}
```

Published set names (also used in section 10): `narrowfield-win`, `narrowfield-win+anet`,
`full-server`, `golden`. A site can write its own set file (`sets/site-c8.json`). The
store keeps imported sets as `sets/<set-digest>.json`.

| Verb | Effect |
| --- | --- |
| `senpai-admin bundle install <file\|set name\|kind/name@version>` | Stage only: fetch (from the named file, or the configured source when online), verify, copy into the store. Never activates. |
| `senpai-admin bundle activate --set <name\|file>` | Make that set active (written to `active.json`; generates the anet config); takes effect on restart. |
| `senpai-admin bundle list` | Installed and active bundles, with verification state. |

`bundle sync` and `bundle pull` from earlier drafts are folded into `bundle install`. Every
install sequence in this document ends with `bundle activate`, because nothing activates by
itself: a Windows operator who only runs `bundle install` has no active tetra3 DB, and
a client's COLD preflight fails with that reason.

### 3.5 astrometry.net: owned by astroeasy, run as a separate program

Owner decision 2026-10-19. astrometry.net is GPL-3.0-or-later (it bundles GPL code, so its
binaries are GPL). It is integrated by astroeasy, not senpai, so **astroeasy stays MIT** (senpai: after v2.8.2,
AGPL-3.0).

| Who | Licence | Role |
| --- | --- | --- |
| senpai | AGPL-3.0-or-later | Never installs, builds, links or vendors astrometry.net. Calls astroeasy, which picks a backend. |
| astroeasy | MIT | Runs `solve-field` only as a separate program (subprocess or container). Builds and publishes the `astrometry-cli` image. |
| `astrometry-cli` image | GPL-3.0-or-later | Built from `astroeasy/dotnet/Dockerfile` at a pinned astrometry.net commit, published by astroeasy's release process with the source tarball (OCI referrer, offline kits). |

astroeasy execution backends:

| Backend | Where | How |
| --- | --- | --- |
| `local` (exists) | plain Linux/macOS host, `solve-field` on PATH | subprocess; not usable natively on Windows (POSIX `timeout`). |
| `docker` (exists) | plain host with Docker; Windows via Docker Desktop or WSL2 | `docker run --rm` per solve, by digest, with `--pull=never --network=none` (section 11). |
| `sidecar` (new) | server profiles (hosted, self-hosted, air-gapped) where senpai itself runs in a container | The `astrometry-cli` container runs next to the `senpai` container in the same compose project (or pod). Shared scratch volume (read-write for both) for job inputs and outputs; the active `anet-index` set mounted **read-only** in the sidecar; the sidecar has no network egress. astroeasy writes a job (sources and `solve-field` arguments) to the scratch volume and the sidecar's runner executes it and writes the results back. Exact protocol (spool directory or a small local endpoint, timeouts, cancellation, concurrency) [decide in astroeasy]. |

The Docker socket is **never** mounted into the senpai container. On a plain host (local
profile, Linux), astroeasy's existing local or docker backends apply unchanged. senpai's
config selects the backend (`astrometry.backend: local|docker|sidecar` [proposal], plus
image digest or sidecar scratch path); provenance records the astrometry.net commit and the
`astrometry-cli` digest.

Work item: senpai's repo `Dockerfile` builds astrometry.net into the senpai image today; it
is retired from there, so the published `senpai` image carries only senpai's own code (section 11).

## 4. Signing and verification

| Item | Design |
| --- | --- |
| Model | **Keyless Sigstore only** (owner decision 2026-10-19). Wheels, the `senpai` image, astroeasy's `astrometry-cli` image, data bundles, release manifests, channel indexes and the `SHA256SUMS` of every kit are signed in GitHub Actions with the workflow's OIDC identity: Fulcio issues a short-lived certificate and Rekor logs the signature. We hold no long-lived signing keys, so there is nothing to keep in custody, rotate or expire. Each signature ships as a Sigstore bundle (`.sigstore.json`: certificate, signature and transparency-log inclusion proof). |
| Pinned identities | Verification pins the issuer `https://token.actions.githubusercontent.com`, the source repository **id** (not only its name, so a rename or re-registration cannot satisfy it), the workflow path and the ref. Stable and LTS: `.github/workflows/release.yml` at a `refs/tags/v*` ref; `v*` tags are protected by a ruleset only release maintainers can bypass, and the signing job runs in a protected environment with required reviewers. Each product signs from its own repository, so another repository's identity cannot sign senpai releases; `astrometry-cli` is pinned to astroeasy's identity. |
| Edge | Signed by `.github/workflows/nightly.yml` on `refs/heads/main`. That identity is accepted only for the `edge` channel, so an edge manifest cannot verify for a stable or LTS site. |
| Channel index | `channels/<channel>.json` is re-signed on a schedule by `.github/workflows/channel-index.yml` on `refs/heads/main`, pinned as its own identity. |
| PyPI | PEP 740 attestations via Trusted Publishing (already used). pip/uv do not enforce them, so the release manifest also lists wheel hashes. |
| Images | Keyless cosign signatures (same identities) + SBOM as OCI referrers. Compose and Helm pin by digest. |
| Online verification (connected) | `cosign verify-blob --bundle <file>.sigstore.json --certificate-oidc-issuer https://token.actions.githubusercontent.com --certificate-identity-regexp '^https://github\.com/ssc-ai/senpai/\.github/workflows/release\.yml@refs/tags/v' --certificate-github-workflow-repository ssc-ai/senpai <file>` (cosign fetches Sigstore's trusted root through TUF), or sigstore-python `sigstore verify github`, plus a repository-id check in `senpai-admin verify` [verify how: cosign has no flag for the repository-id certificate extension]. Requires cosign 3.0.4+ or 2.6.2+ (fix for CVE-2026-22703). |
| Offline verification (air-gapped) | The same command plus `--trusted-root trusted_root.json` (the copy installed in `store/trust/`). With an explicit trusted root and a bundle that carries the inclusion proof, cosign checks the certificate chain, the signature, the inclusion proof and that the entry was logged while the certificate was valid, with no network call [verify: the current `verify-blob` reference lists no `--offline` flag; confirm with the no-network test]. sigstore-python equivalent: `sigstore --trust-config <file> verify github --offline --bundle ...` [verify: `--trust-config` takes a client trust config, not a bare `trusted_root.json`]. |
| First install (trust bootstrap) | Before running anything from the medium: obtain cosign from its official release and check it against its published checksum (or use a copy site IT already trusts); compare the digest of the kit's `trusted_root.json` (`sha256sum`, or `Get-FileHash` on Windows) with the one we publish out of band (docs site, README, release page); run the offline command on `SHA256SUMS`, then check the file hashes. Only then run the verified `senpai-admin.pyz`, which records the trusted root and the pinned identities in `store/trust/`. |
| Which verifier runs | **Updates are verified by the `senpai-admin` already installed and activated from the previous verified release, with the installed `trusted_root.json`**, never by copies on the new medium. A newer verifier or `trusted_root.json` on the medium is just a verified artifact of the new release: the trusted root is replaced only after the kit carrying it has verified against the installed one. |
| Verifier (`senpai-admin verify`) | Runs the installed cosign for the Sigstore check (pinned identity for the product and channel, installed trusted root) and does the rest with the standard library: repository id, manifest `channel`, every file hash, and `seq` above the last imported for that (product, channel). The `.pyz` stays pure Python and portable; cosign is a static binary, carried per platform in kits. `verify.ps1` on Windows automates the hash checks and calls cosign; it is not a substitute for comparing the published digests. |
| Rollback protection | `import` refuses a manifest whose `seq` is below the highest imported for that (product, channel), or whose `channel` differs from the site's subscribed channel in `active.json` unless `--switch-channel <name>` is given (logged). `activate` also refuses a release with an older `released` date than the active one without `--allow-downgrade` (logged). |
| Freshness | The signed channel index carries `expires` (issued + 14 days [proposal]) and is re-signed on a schedule even without a release. `senpai-admin release check` warns, or fails with `--strict`, when the index is expired or older than the last one seen. Mirrors re-serve the signed file byte for byte and never re-sign it. |
| Stated limitations | Air-gapped sites get authenticity, not freshness: nothing tells them a newer release or advisory exists until a kit arrives, and rollback protection is only the `seq` and downgrade checks above. A misused release workflow (for example a compromised maintainer pushing a tag) yields validly signed releases; the transparency log makes that publicly visible, but offline sites cannot consult it. If Sigstore rotates keys in its trusted root, an air-gapped site needs a kit carrying the new `trusted_root.json` before it can verify releases signed under them; regular kits pick it up ahead of time because Sigstore publishes new keys before using them [verify], and a site that missed a rotation re-confirms the new digest out of band, as on first install. Mirror-fed connected sites get freshness only up to the channel index's `expires`. Signed security advisories are in the channel index and in every kit. |
| GPL | Only astroeasy's `astrometry-cli` image contains astrometry.net; it carries the matching source tarball (referrer and in offline kits). senpai's images and wheels contain no GPL code; a kit that includes `astrometry-cli` carries it as a separate image with its source. |
| Data credits | Every bundle's `license` block and `NOTICE` are covered by its signature (section 3.1.1); kits carry a combined `NOTICE`. |

### 4.1 Later, if needed (not in force)

Deferred by the owner on 2026-10-19. None of this is part of the current flows, manifests,
kits or work items; each is added only when its trigger occurs.

| Addition | Trigger |
| --- | --- |
| An offline release key (ed25519) on a hardware token, signing stable/LTS manifests and kit `SHA256SUMS` in a manual signing ceremony, verifiable with stock `ssh-keygen -Y verify`. | A site whose policy forbids Sigstore or transparency logs, or a regulatory need for a key we hold. |
| A TUF-style `root.json`: a 2-of-3 threshold of hardware-token root keys, per-product and per-channel key roles, key expiry (with an `--accept-expired` escape for sites past due). | Many outside sites, or a commercial product. |

## 5. Release channels

| Channel | Cadence | Contents | Support | For |
| --- | --- | --- | --- | --- |
| `edge` | nightly from `main` | anything; signed by the nightly workflow identity, accepted only for `edge` (section 4) | none | developers |
| `stable` | about monthly [proposal] | features, algorithm bumps allowed | until next stable + 30 days | hosted, connected self-hosted, local |
| `lts-YYYY` | about yearly [proposal] | branched from a stable; security, crash and data-refresh fixes only; **algorithm versions frozen** | about 18 months, so 6 months overlap with the next LTS [proposal]; EOL notice at least 3 months ahead in the channel index | air-gapped, conservative self-hosted |

## 6. Update flows

Shared rule: **stage, verify, activate, (roll back)**. Staging never changes behaviour:
it writes only `compose.next.yaml` and staged bundles, so a reboot or a `docker compose up`
after staging still runs the current pins. `activate` does change behaviour: it prints
which algorithm versions and bundles change, runs the DB expand migrations, atomically
renames `compose.yaml` to `compose.prev.yaml` and `compose.next.yaml` to `compose.yaml`,
updates `active.json`, and restarts. `rollback` swaps `compose.prev.yaml` back the same
way.

### 6.1 (a) Hosted

| Step | What |
| --- | --- |
| Release | Publish (section 3); deploy to staging by digest. |
| Check | Golden corpus + shadow evaluation on requests that carry their own `algorithm_eval` consent (per-request purpose plus tenant agreement; [versioning doc 5.3](api-v2/versioning-and-provenance.md#53-promotion-on-hosted)). |
| Software-only release | Traffic canary 5% / 50% / 100% on errors and latency. |
| Algorithm change | Shadow, candidate, stable promotion with at least 14 days notice ([versioning doc 5.3](api-v2/versioning-and-provenance.md#53-promotion-on-hosted)). Old versions keep serving pinned clients through their support window. |
| Users see | `GET /v2/capabilities`, `Deprecation`/`Sunset` headers, changelog feed, provenance. Clients do nothing. |

Topology: a gateway (TLS, billing metering, coarse DDoS limits) in front of a router that
maps the pinned algorithm to a release pool (one `senpai` digest per pool, a reserved sync
pool and a batch pool per release), an object store (uploads, retained data; per-tenant
prefixes with lifecycle rules) and a small metadata DB (jobs, idempotency keys, retention
index, contribution grants). Tenant limits, fair-share admission to the sync pool, compute
metering in worker-seconds and size caps are in the open-source server
(`capabilities.limits.per_tenant`), so self-hosters get the same controls (P5). The gateway
authenticates callers and forwards their JWT, or mints a short-lived JWT for the
standard `oidc` mode; there is no gateway-header auth. Hosted defaults: `async_promotion`
off until promotion bounds are implemented; T4 on the sync route limited by
`limits.sync_tier_budget_s` (blind solves belong on jobs).

Sync SLI and SLO [proposal; benchmark first]:

| Item | Definition |
| --- | --- |
| SLI | Fraction of well-formed `/v2/analyze` requests within the tenant's quota that return `200` within the requested deadline, measured from the last request byte received. `503`, `504`, other `5xx` and promotions to `202` count as failures. `4xx` (including over-quota `429`) are excluded. |
| Availability target | 99.5% monthly. |
| Latency targets | T0/T1 class (WARM or LOCKED hint, 8 MP): server-side p95 at most 3 s. T3/T4 class: separate target set after benchmarking; not covered by the 3 s figure. |
| Fast-fail | `limits.overload_reject_ms` (1000 on hosted) bounds the `503 overloaded` response clients rely on for fallback. |

### 6.2 (b) Self-hosted, connected (own data, receives updates)

Install (Linux, containers):

```
# 1. download and verify before running anything from the kit (cosign 3.0.4+, section 4)
curl -LO https://github.com/ssc-ai/senpai/releases/download/v3.1.0/senpai-deploy-3.1.0.tar.gz
tar xf senpai-deploy-3.1.0.tar.gz && cd senpai-deploy
cosign verify-blob --bundle SHA256SUMS.sigstore.json \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --certificate-identity-regexp '^https://github\.com/ssc-ai/senpai/\.github/workflows/release\.yml@refs/tags/v' \
    --certificate-github-workflow-repository ssc-ai/senpai SHA256SUMS
sha256sum -c SHA256SUMS
# 2. from here on, run the verified tool
python senpai-admin.pyz verify release-3.1.0.json        # identity, repository id, hashes, seq
python senpai-admin.pyz init --profile self_hosted --store /srv/senpai \
    --tls-cert /etc/ssl/senpai.pem --tls-key /etc/ssl/senpai.key   # or --generate-site-ca
python senpai-admin.pyz import release-3.1.0.json        # into store/releases/; sets the seq baseline
python senpai-admin.pyz images pull --release 3.1.0      # by digest, cosign-verified
python senpai-admin.pyz bundle install --set narrowfield-win
python senpai-admin.pyz bundle activate --set narrowfield-win
python senpai-admin.pyz activate --release 3.1.0         # writes compose.yaml and active.json
docker compose -f /srv/senpai/deploy/compose.yaml up -d  # images by digest, --pull=never
```

`init` creates the store, the config, the first API key and (with `--generate-site-ca`) a
site CA whose `site-ca.pem` clients use. `senpai-admin` itself makes no network call unless
the command names a remote source (`images pull`, `bundle install` of a set or reference,
`upgrade`, `mirror sync`, `release check`); those honour `updates.source`.

Without containers: `pip install --no-index --require-hashes -r
senpai-3.1.0-requirements-linux_x86_64.txt --find-links wheelhouse/` (installs
`astro-senpai[server,admin]`), the same `init`, `import`, `bundle install`, `bundle
activate`, `activate`, then `senpai serve --config /etc/senpai/senpai.yaml` from the
shipped systemd unit; the astrometry.net tier then uses astroeasy's local backend
(`solve-field` on PATH, installed by the operator) or its docker backend (the `astrometry-cli`
image). senpai installs neither.

Updates:

| Step | Command | Effect |
| --- | --- | --- |
| 1 | `senpai-admin release check` | Opt-in (`updates.check: true`). One GET of the signed `channels/<channel>.json` from our endpoint or the operator's mirror, `User-Agent: senpai-admin/<ver>` (suppressible), no install id, no frames. Warns on an expired index. Prints the plan: software version, API minor, **which algorithm versions change** (with golden diff link), which bundles change, upgrade stops. |
| 2 | `senpai-admin upgrade --to 3.2.0` | Verified by the **installed** `senpai-admin`. Pulls images, wheels and bundles by digest; verifies; writes `compose.next.yaml` and the staged bundle set only. **Does not activate**; a restart or reboot still runs the current pins. |
| 3 | `senpai-admin activate --release 3.2.0 [--canary]` | Runs DB expand migrations, renames next to current and current to prev, updates `active.json`, restarts. `--canary` first runs the golden corpus on the new stack and fails with "golden bundle not installed" if it is absent (never skips silently). |
| 4 | `senpai-admin rollback` | Swaps `compose.prev.yaml` and the previous bundle set back (allowed when `compat.state_db.rollback_to` covers it). |
| Data only | `senpai-admin bundle install tetra3-db/fov0.5-1.5_g11@2026.12.0` then `bundle activate --set <set>` | Independent of software. |

Auto-update policy `updates.auto`: `none` (default); `download` (stage only); `security`
(activate security-only releases that keep every algorithm version). Algorithm versions
never change without an explicit `activate`.

Mirrors: `senpai-admin mirror sync --channel lts --to /srv/mirror`, or `oras cp` into
Harbor, Artifactory or `registry:2`; sites set `updates.source` to the mirror. Identity is
the digest and signatures are carried byte for byte, so a mirror never changes provenance.
Test OCI 1.1 referrer support per registry (ECR has rejected referrer manifests with
`oras cp -r`).

### 6.3 (c) Air-gapped (no internet, ever)

Model: an **inventory round-trip**. The site exports what it has; a connected staging
machine builds a kit with only what is missing; the kit crosses on approved media.

Initial install:

```
# connected staging machine (any OS with Python 3.11+)
senpai-admin export --release lts-2027@latest --platform linux_x86_64 \
    --bundles sets/site-c8.json --with-anet [--with-golden] --out /media/usb/

# air-gapped host
mkdir /tmp/kit && tar xf /media/usb/senpai-offline-3.0.7-linux_x86_64-<set>.tar -C /tmp/kit
cd /tmp/kit
# cosign: brought in and checked against its published checksum (section 4)
sha256sum trusted_root.json                 # must equal the digest published out of band
cosign verify-blob --bundle SHA256SUMS.sigstore.json --trusted-root trusted_root.json \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --certificate-identity-regexp '^https://github\.com/ssc-ai/senpai/\.github/workflows/release\.yml@refs/tags/v' \
    --certificate-github-workflow-repository ssc-ai/senpai SHA256SUMS   # no network
sha256sum -c SHA256SUMS
python3 senpai-admin.pyz verify release.json            # identity, repository id, hashes, seq
python3 senpai-admin.pyz init --profile airgapped --store /srv/senpai --generate-site-ca
python3 senpai-admin.pyz import .                       # docker/podman load, bundles, manifest
python3 senpai-admin.pyz bundle activate --set site-c8
python3 senpai-admin.pyz activate --release 3.0.7
docker compose -f /srv/senpai/deploy/compose.yaml up -d  # or: systemctl enable --now senpai
```

After the first install, `senpai-admin` is installed from the kit's wheelhouse (or keep
invoking it as `python3 senpai-admin.pyz` from the activated release directory); later
kits are verified by this installed copy.

An offline kit contains: the deploy kit (compose files pinned by digest, TLS proxy, example
air-gapped config, systemd unit); OCI image layout (`senpai`, optionally astroeasy's
`astrometry-cli` for the sidecar);
wheelhouse with hashes; selected bundles (`.tar.zst` with manifests); optionally the
`golden` bundle (`--with-golden`, 1 to 2 GB) for `activate --canary`; release manifest and
its Sigstore bundle; Sigstore's `trusted_root.json`; cosign binaries per platform (checked
by hash on first install, section 4); SBOMs; with `astrometry-cli`, its astrometry.net
source tarball; the combined `NOTICE`/`ATTRIBUTION` file and each bundle's recipe;
`SHA256SUMS` and `SHA256SUMS.sigstore.json`; `senpai-admin.pyz`; the static docs
site (incl. problem-type pages). `import` runs `docker load`/`podman load` and copies
bundles into the store.

Updates:

| Where | Step |
| --- | --- |
| inside | `senpai-admin inventory > site-inventory.json`: product, channel, release seqs, image digests, bundle digests and the active set, platform. No frames, results or secrets. Carry it out. |
| outside | `senpai-admin export --have site-inventory.json --release lts-2027@latest [--bundles <set>] [--with-golden]`: a delta kit. Without `--bundles` it includes the target release's `recommended_bundles` plus newer versions of the site's current set (e.g. `iers`, a rebuilt tetra3 DB) that the site lacks. The Gaia mirror and index sets are tens of GB and rarely change, so they are included only when their digest changed. |
| inside | Verify with the **installed** `senpai-admin verify`, then `import`, `bundle activate` if the set changed, `activate`; `rollback` if needed. |

Trust bootstrap: the first install confirms the `trusted_root.json` digest and the signer
identity out of band (docs site, README, release page) and records them in `store/trust/`;
nothing on the medium is trusted before that check. Later kits are verified with the
installed trusted root, and a kit's newer `trusted_root.json` is installed only after the
kit verifies (section 4). Expect one or two crossings per year plus security
patches.

### 6.4 (d) Local on the observatory PC (incl. Windows next to TheSkyX)

Two modes from one install:

| Mode | How | When |
| --- | --- | --- |
| In-process (default) | `senpai.v2.engine.analyze_frame(...)` with `astro-senpai[engine]`; calls run in a 1-worker subprocess so a deadline can kill a stuck solve | One client on the PC; offline and low-bandwidth fallback |
| Localhost service | `senpai serve --profile local` on `127.0.0.1:8765`, as a Windows service (WinSW, NSSM or pywin32 [decide]) or a client-spawned subprocess | Several tools share one warm engine (catalog and tetra3 DB loaded); isolates senpai's config |

Windows install, in order:

1. Extract the kit. Check `cosign.exe` and `trusted_root.json` with `Get-FileHash`
   against the digests published out of band, run the offline `cosign verify-blob`
   (section 6.3) on `SHA256SUMS`, then check the file hashes with `Get-FileHash`.
   `verify.ps1` automates these steps but is not a substitute for comparing the published
   digests.
2. Install senpai into its own versioned environment,
   `%ProgramData%\senpai\versions\<ver>` (python-build-standalone 3.13 plus the senpai
   `win_amd64` wheelhouse: `astro-senpai[engine,server,admin]`, `uv pip install --no-index
   --require-hashes`). A client kit may carry the same wheelhouse.
3. `senpai-admin init --profile local --store %ProgramData%\senpai`: creates the store and
   the token file with its ACL (section 2).
4. `senpai-admin bundle install narrowfield-win.tar.zst` (from media; online, the set
   name), then `senpai-admin bundle activate --set narrowfield-win`.
5. Optional, for T3/T4 on this PC (supported; heavy on Windows, see below): `docker load -i
   astrometry-cli-<digest>.oci.tar` (astroeasy's image) into Docker Desktop or WSL2 from the
   kit (or `senpai-admin import --images-only <kit>`), install an `anet-index` bundle,
   activate a set that includes it.
6. Localhost service only: `senpai-admin service install` registers the service to run
   from `%ProgramData%\senpai\versions\<active ver>`, managed by `senpai-admin activate`
   and `rollback`. It never runs from a client's `current` junction, so updating the client
   never silently changes the service's senpai version.
7. Check: `senpai-admin doctor`, and the client's own preflight.

Windows specifics:

- Tiers: fully local operation is supported on Windows for every tier. T0/T1 run natively
  (Gaia mirror slice + tetra3 DB, no Docker). T3/T4 run astroeasy's `astrometry-cli` image
  under Docker Desktop or WSL2 (step 5); astroeasy's local backend cannot run `solve-field`
  natively on Windows (it wraps it in POSIX `timeout`).
- Caveat, blind solves: T3/T4 are especially heavy on Windows: the Docker Desktop / WSL2
  VM's memory and I/O overhead and a container start per solve [unverified: benchmark],
  index sets of several GB (about 2.4 GB for a narrow-field subset, 36 GB for 5200-LITE),
  memory for blind search, and Docker Desktop's licence terms (a paid subscription for
  organisations over 250 employees or $10M revenue). When a self-hosted or hosted server is
  available, we suggest pointing COLD/blind solves at it, with local T3/T4 as the fallback.
- `GET /v2/capabilities` (or `engine.capabilities()`) says which tiers are ready.
- Updates: online, `senpai-admin upgrade` stages a new `versions\<ver>` and `activate`
  switches it; offline, the same from USB, verified by the installed `senpai-admin`.
- In-process engines installed by a client kit report `environment_sha256` and
  `environment_matches_release` (versioning doc, section 7); client kits install senpai's
  dependencies with senpai's lock as a constraints file so the two match.
- Prerequisite: Windows + Python 3.13 CI for senpai and astroeasy, installing the `solve`
  extra. Neither exists today.

Linux observatory box: profile (b) on one machine.

## 7. Rollback

| What | How |
| --- | --- |
| Images / wheels | Previous digests kept (`compose.prev.yaml`, Windows `versions\`). `senpai-admin rollback` swaps them back atomically. |
| Bundles | Side by side in the store; `active.json.previous`. `senpai-admin gc --keep 2` prunes. |
| State DB (jobs, idempotency keys, retention index) | SQLite (self-hosted, local) or Postgres (hosted). Expand/contract migrations; release N's schema readable by N-1. Otherwise a declared upgrade stop that `upgrade` refuses to skip. |
| Algorithm | Clients pin (`senpai.frame@N`); hosted keeps old versions in their window; self-hosted rolls back the release. |
| Hosted | Redeploy the previous digest. |

## 8. Data ownership, retention and contribution

| Item | Rule |
| --- | --- |
| Retention classes (normative; all profiles) | `retain: none`: frame bytes are never persisted beyond the worker. The result is held, encrypted, only for min(`idempotency_ttl_s`, `results_ttl_h`), solely for idempotent replay and for fetching a job or promoted result, then purged; it never enters the result cache; its purge time is echoed as `data_policy_applied.retain_until`. `retain: results`: results kept until `retain_days`, no frame bytes. `retain: frames`: frames and results kept until `retain_days`. |
| Hosted defaults | `retain: none`; `results_ttl_h` 24 h [proposal]; a tenant may raise retention to `retention_max_days` (30 [proposal]). |
| Consented holds | The only way frame bytes outlive a `retain: none` request is a hold with its own consent basis, e.g. a shadow evaluation for a request that itself carries `contribute.purposes: [algorithm_eval]` under a tenant agreement (versioning doc 5.3; hard cap 1 h [proposal], purged right after the run). Holds are shown in `GET /v2/frames/{pixel_sha256}` (`holds[]`) and in `data_policy_applied.frame_retain_until`. |
| Uploads | An object from `POST /v2/uploads` that is never referenced is deleted at the slot's `expires_at`; once referenced it follows the referencing request's class. |
| Audit rows | After any purge or `DELETE`, an audit row keeps: result, request and job ids; tenant id; timestamps; API, software, algorithm, config-hash and bundle-set identifiers; `pixel_sha256`; retention class. No hint, WCS, detections, matches or pixels. Kept `policies.audit_retention_days` (395 days [proposal]). |
| Precedence | Tenant policy is the ceiling; `data_policy` in a request can only shorten it. `data_policy_applied` is echoed. |
| Data subject operations | `GET /v2/frames/{pixel_sha256}` shows what is held; `DELETE` removes frame and results and revokes contribution. `DELETE /v2/results/{id}`, `DELETE /v2/jobs/{id}`. |
| Contribution (hosted) | Needs all of: tenant agreement to a versioned contribution terms text; per-request `data_policy.contribute` with `purposes` (`algorithm_eval`, `training`, `publication`), licence and `public` flag (nova.astrometry.net's per-upload model); profile `hosted`. Contributed frames go to a separate access-controlled corpus with the grant recorded. Revocation removes them from future sets; trained models are not retrained (stated in the terms). |
| Contribution (self-hosted, air-gapped, local) | Export only: `senpai-admin contribute pack --results <ids> --out contrib.tar` (frames, grant records, provenance; signed with the site key). The site reviews it and sends it by its own means. **The open-source server has no automatic outbound contribution path.** |
| Telemetry | None, in any profile. |

## 9. Network egress inventory

Verified on 2026-10-19: with local catalogs and data present, importing senpai and
astroeasy and the v1 detect/solve request path make no network calls (tested in a
no-network namespace). That test used frames with RA/Dec headers; it did not cover the
Alt/Az header path below, which can reach IERS servers. The no-network test and the golden
corpus gain an Alt/Az-only header frame and an RA = 0 frame. The table lists every path that does reach out.

| Endpoint | Trigger | Offline behaviour today | How it is disabled |
| --- | --- | --- | --- |
| `gea.esac.esa.int` (Gaia TAP) | `star_catalog.type: gaia`; importing `astroquery.gaia` (fetches status on import); catalog probes; Gaia mirror build | Prints "Status messages could not be retrieved"; queries fail, exceptions swallowed to `[]` | `[online]` extra only; `airgapped` refuses `gaia`; use `gaia_local` or `sstrc7`. Mirror builds happen on the staging machine. |
| `skyserver.sdss.org` | `star_catalog.type: sdss` | Fails | Same as Gaia. |
| `datacenter.iers.org`, `maia.usno.navy.mil` (IERS-A); `hpiers.obspm.fr`, `data.iana.org` (leap seconds) | astropy `IERS_Auto` when the bundled table is over 30 days old or the leap-second file has expired. Reached (1) from senpai calibration/burr CLIs (`engine/observability/calibration.py:200,231-237,467`) and (2) **on the frame and collect path**: `extract_boresight_from_header` (`engine/utils/fits_io.py:337-350`) does `SkyCoord(altaz).icrs` for headers with Alt/Az but no RA/Dec, and also for RA or Dec exactly 0 because of `if ra and dec:` (`fits_io.py:294`); it is called from `processing/sidereal.py`, `rate.py`, `collect.py` and `models/metadata.py` | **Hard failure reproduced:** `ValueError: interpolating from IERS_Auto using predictive values that are more than 30.0 days old`; 10 s stall per attempt on drop-firewalls | The engine (not only the server) sets `iers.conf.auto_download=False`, `utils.data.conf.allow_internet=False`, `iers_degraded_accuracy='warn'` **at import of `senpai.engine`**, before any astropy time use, so in-process use is covered too. Fix `fits_io.py:294` first. Ship the `iers` bundle and load its tables explicitly; record `time_data` in provenance; warn `IERS_STALE` (never fail). |
| pyerfa built-in leap-second table | Pure-erfa time calls before astropy.time refreshes it (table expires 2017-06-30) | `ErfaWarning: dubious year`; value still right (no leap second since 2017); a future one would be missed | At start the engine loads the active `iers` bundle explicitly: `iers.earth_orientation_table.set(iers.IERS_A.open(<bundle>/finals2000A.all))` and `erfa.leap_seconds.update(iers.LeapSeconds.from_iers_leap_seconds(<bundle>/Leap_Second.dat))`, with downloads already disabled. `time_data.iers_bundle` and `leap_second_expiry` in provenance. Without an `iers` bundle it uses the installed astropy-iers-data and says so. |
| `portal.nersc.gov/.../temp/dstn/index-5200`, `data.astrometry.net/4100`, `/4200` | `astroeasy indices download` only (scrapes listings; HEAD even for present files) | n/a at runtime | Ship `anet-index` bundles from our mirror with hashes, or rebuild them with the recipe on a staging machine (section 3.1.1). |
| `github.com/ssc-ai/sstrc7/releases` | `sstrc7 get` only | n/a at runtime | `sstrc7` bundle. |
| Docker Hub, ghcr.io | astroeasy builds its own `docker run --rm ... <image>` (`astroeasy/dotnet/docker.py:33-41`) and `docker run --rm <image> solve-field --help` (`:126`) with no way to pass flags; senpai calls `astroeasy.test_install` at startup (`senpai/astrometry.py:232`), so a missing or mistyped image is pulled at server start and on every T3/T4 solve | Pull, or error offline | Needs an astroeasy change (section 11): always pass `--pull=never --network=none`, accept `name@sha256:` references, and make `test_install` fail clearly when the image is missing. senpai's `astrometry.docker_image` takes digest references. Images come from the kit (`docker load`). Documentation alone cannot enforce this. In server profiles the sidecar backend makes no Docker calls from senpai at all. |
| `cdn.jsdelivr.net`, `fonts.googleapis.com`, `fastapi.tiangolo.com` | Browser opening `/docs` or `/redoc` | Blank pages | Serve Swagger/ReDoc assets from the package. |
| Build time: apt, astrometry.net git clone (unpinned HEAD, in senpai's and astroeasy's Dockerfiles), PyPI | Image builds | n/a | Build in CI with pinned base and `--frozen`; astrometry.net leaves senpai's Dockerfile, and astroeasy builds `astrometry-cli` at a pinned commit; ship images and wheelhouses. |
| `download.pytorch.org` | starcsp model build (ResNet ImageNet weights, every build) | Fails | Before starcsp ships: load `weights=None` then our state dict from a `weights` bundle (safetensors). |
| Update check | `senpai-admin release check` | n/a | Opt-in; no code path in `airgapped`. |
| `tuf-repo-cdn.sigstore.dev` | Online verification on connected sites (cosign or sigstore-python refreshing Sigstore's trusted root) during `verify`, `images pull`, `upgrade` | n/a | Never in `airgapped`, which verifies offline with the installed `trusted_root.json` and makes no call [verify in the no-network test]. Rekor is not queried: bundles carry the inclusion proof. |
| `ghcr.io`, our HTTPS mirror, `updates.source` | `images pull`, `bundle install <set name or kind/name@version>`, `upgrade`, `mirror sync` | Fail | Only when the operator runs a command that names a remote source; honours `updates.source`. `senpai-admin` makes no network call otherwise. On Windows, `bundle install` from a `.tar.zst` file is offline. |
| Client to server | client to senpai v2 | — | TLS with custom CA, bearer token, fallback to local engine. |
| Telemetry | none found (no sentry, otel, prometheus push, HTTP log handlers) | — | Keep it that way. |

Writable paths for read-only containers: `SENPAI_CACHE_DIR`, `SENPAI_LOG_DIR`,
`MPLCONFIGDIR` on a volume.

## 10. Sizing (rough, from fact-finding)

| Bundle set | Contents | Size |
| --- | --- | --- |
| `narrowfield-win` (C8-class, T0/T1; also the in-process fallback pack for hosted users) | tetra3 DB for the FoV (0.03 to 0.35 GB), Gaia mirror G<=16 regional (about 3 GB, **depth chop not built**), `iers` | about 3.5 GB |
| `narrowfield-win+anet` | above + index subset 5204-5206 and 4107-4109 (about 2.4 GB) + astroeasy's `astrometry-cli` image (1.6 GB today) | about 7 GB |
| `full-server` | Gaia mirror G<=20 (about 50 GB), 5200-LITE (36 GB), SSTRC7 (17.6 GB), 4100 (0.4 GB), `iers` | about 105 GB (130 GB with G<=21) |
| `golden` (bundle `golden/senpai-golden`) | regression corpus; optional kit content for `activate --canary` | 1 to 2 GB [unverified] |

Server memory: dense fields reached about 30 GB per worker without `max_stars_per_frame`;
that cap is mandatory in server profiles and reported in `capabilities.limits`.

## 11. Prerequisites and order of work

1. **Licence:** senpai: after v2.8.2, AGPL-3.0 (question 1). Release hygiene: commit `uv.lock` and build `--frozen`; consistent `v*`
   tags (`1.2.1`, `2.6.0` are untagged variants) protected by a ruleset; remove the manual
   `make publish` that bypasses review; move off ubuntu:20.04; fix the Docker README mount
   paths.
   **Retire astrometry.net from senpai's repo `Dockerfile`** (it builds astrometry.net into
   the senpai image today), so the published `senpai` image carries only senpai's own code; T3/T4 in containers
   then go through astroeasy's sidecar backend (section 3.5).
2. Golden corpus and the algorithm-version CI gate (before any numerical change),
   including an Alt/Az-only header frame, an RA = 0 frame, and one frame stored plain, Rice,
   tile-GZIP and gzip-wrapped (pixel-hash test vector). The no-network test runs the engine
   on these frames. Fix `fits_io.py:294` (`if ra and dec:`) and move the astropy
   offline settings to engine import before this gate is cut.
3. **astroeasy changes** (prerequisite for any offline T3/T4): the Docker backend always
   passes `--pull=never --network=none`, accepts `name@sha256:` image references, and
   `test_install` fails with a clear "image not present" error instead of pulling;
   astroeasy builds and publishes `astrometry-cli` in CI at a pinned astrometry.net commit
   with its source tarball; a new `sidecar` backend for containerised servers (section 3.5).
   Windows + Docker Desktop/WSL2 CI for the docker backend.
4. Engine refactor (per-call config, loader, pixel conventions, precision), port of client-side
   detection logic into senpai and the rung planner into astroeasy, `/v2/analyze`,
   `/v2/solve/sources`, health, capabilities, auth, process pool. Publish `3.0.0rc`.
5. Bundle format and set files, store references in config, `senpai-admin` (`verify`
   wrapping the installed cosign with pinned identities, `init`, `import`, `images pull`,
   `bundle install|activate`, `export`, `inventory`, `activate`, `rollback`, `keys
   add-site`), keyless signing of manifests, bundles, channel index and kit `SHA256SUMS`
   in the release, nightly and channel-index workflows, `trusted_root.json` and cosign in
   every kit, published identities and trusted-root digest, a no-network test of offline
   verification, Gaia mirror manifest and depth chop. Data credits and
   recipes: `license.upstreams` in every manifest, `NOTICE` per bundle and per kit,
   capabilities and provenance attribution, a recipe per bundle kind with pinned inputs and
   expected digest (section 3.1.1). **Per-source terms verification** (owner direction
   2026-10-19): resolve every [verify] row in section 3.1.1, including whether hosted or
   paid use of Gaia-derived bundles needs ESA authorisation (data.licences@esa.int).
6. Jobs, SSE, collects, results/tables, idempotency store, uploads.
7. Windows CI, wheelhouses, localhost service wrapper.
8. Hosted: gateway, router, retention enforcement, shadow evaluation. First `lts` line.

## 12. Open questions for the owner

| # | Question | Recommendation |
| --- | --- | --- |
| 1 | **Licence.** senpai: after v2.8.2, AGPL-3.0. Still open: a separate licence for model weights. | Decided. For weights, consider a data licence (e.g. CC-BY-4.0) separate from the code. |
| 2 | **astrometry.net (GPL-3.0-or-later).** **Decided (owner, 2026-10-19):** astrometry.net lives only in astroeasy's `astrometry-cli` image, as a separate program; astroeasy stays MIT. Server containers use astroeasy's sidecar backend (section 3.5). | Decided. Work item: retire astrometry.net from senpai's `Dockerfile` (section 11). |
| 3 | **Data redistribution.** **Decided in principle (owner direction, 2026-10-19):** we redistribute derived data provided every bundle and kit fully and visibly credits its upstream sources and carries their licence/terms text, and every bundle kind also has a build-it-yourself recipe (section 3.1.1). | Remaining: per-source verification of the [verify] rows in section 3.1.1, notably ESA authorisation for any commercial use of Gaia-derived data. |
| 4 | **Support windows:** stable cadence, LTS cadence and length, hosted retention of old algorithm versions, `/v2` lifetime after `/v3`, v1 sunset date. | Monthly stable; yearly LTS with 18 months support; old algorithms 12 months / at least 2; v2 24 months after v3; v1 per the versioning doc. |
| 5 | **Release signing.** **Decided (owner, 2026-10-19):** keyless Sigstore only for now; offline key / TUF root deferred with triggers listed (section 4.1). Release signing proves that artifacts and data bundles came from us and were not altered; it is separate from API access, which uses API keys or OIDC tokens. | Decided. Publish the signer identities and the `trusted_root.json` digest on the docs site and README. |
| 6 | **Hosted data policy:** retention classes as defined in section 8 (`none` holds the result only for the replay window), results 24 h default / 30 days max, audit rows 395 days, shadow holds capped at 1 h, contribution terms and licence options, whether `algorithm_eval` consent is separate from `training`. | As proposed; keep the two consents separate; shadow evaluation only on per-request `algorithm_eval` consent. |
| 7 | **Update check:** opt-in or opt-out for self-hosted and local; is a version-bearing User-Agent acceptable? | Opt-in; User-Agent suppressible. |
| 8 | **Tenancy for users of our telescopes:** per organisation, telescope or person; quotas or pricing; pre-provisioned tokens per telescope. | Tenant per organisation, keys per telescope. |
| 9 | **Names:** registry (`ghcr.io/ssc-ai`), problem-type URI domain, update endpoint host, hosted API host, PyPI `senpai-client`. | `ghcr.io/ssc-ai`, `ssc-ai.github.io/senpai/problems/`; claim names now. |
| 10 | **Windows packaging:** zip kit first then MSI? Service wrapper (WinSW, NSSM, pywin32)? Authenticode signing for observatory IT? | Zip + `setup` first; WinSW; sign when an MSI exists. |
| 11 | **T3/T4 on Windows.** **Decided (owner, 2026-10-19):** fully local operation is supported everywhere, Windows included: T0/T1 natively, T3/T4 via astroeasy's `astrometry-cli` under Docker Desktop or WSL2. | Decided. Docs carry the caveats (VM overhead, multi-GB index files, memory, Docker Desktop licence terms) and suggest pointing COLD/blind solves at a self-hosted or hosted server when one is available, with local fallback (section 6.4). |
| 12 | **Platforms:** `linux_aarch64` (Pi/Jetson at the telescope)? macOS? | aarch64 yes for wheels and images; macOS best effort. |
| 13 | **Calibration on hosted:** per-tenant dark/flat libraries, or require pre-calibrated frames (`frame.calibrated: true`)? | Pre-calibrated for 2.0. |
| 14 | **Hosted SLO** (the SLI in section 6.1, availability, p95 per tier class, max sync deadline, T4 on the sync route, `async_promotion` on hosted). | 99.5%; T0/T1 p95 3 s; T3/T4 target after benchmarking; max 30 s; T4 on sync only within `sync_tier_budget_s`, otherwise jobs; promotion off until bounded. |
| 15 | **Multi-tenant self-hosting** via OIDC in 3.0, or later? (Per-tenant limits and tenant-scoped lookups are in the contract either way.) | Contract supports it now; ship hosted first. |
| 16 | **starcsp:** move from `zgazak/` to `ssc-ai/`; drop the private Nexus dependency; fix the unconditional torchvision download and `torch.load(weights_only=False)` before shipping weights. | Required before any `weights` bundle. |
| 17 | **Air-gapped freshness:** accept "authentic but not fresh" as the documented posture (rollback limited by the `seq` and downgrade checks; nothing fresher than the last kit); IERS staleness warns, never fails. | Accept. |
| 18 | **Sync route shape:** separate `/v2/analyze` with optional `Prefer: respond-async` promotion (this draft), or one `Prefer`-driven route? API minors numeric `2.N` (this draft) or dated? | As drafted. |
| 19 | **Hosted v1:** enable v1 per tenant (default off), since hosted has no v1 install base? | Yes. |
| 20 | **Image references on hosted:** allow `s3:` through per-tenant assumed roles, or accept only presigned `https` URLs? | Presigned `https` only at launch. |
