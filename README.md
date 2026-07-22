# SENPAI

[![CI](https://github.com/ssc-ai/senpai/actions/workflows/ci.yml/badge.svg)](https://github.com/ssc-ai/senpai/actions/workflows/ci.yml) ![Tests](https://raw.githubusercontent.com/ssc-ai/senpai/main/tests.svg) ![Coverage](https://raw.githubusercontent.com/ssc-ai/senpai/main/coverage.svg)

A classic star detector and astrometry tool.

<img src="https://raw.githubusercontent.com/ssc-ai/senpai/main/resources/senpai_logo.png" alt="senpai" width="600"/>

SENPAI is built off of the algorithm descriptions in [Gazak et al. 2026, PASP, 138, 014502](https://iopscience.iop.org/article/10.1088/1538-3873/ae2b35/meta) — see [Citation](#citation) below if you use this software.


Is SENPAI the tool I'm looking for? SENPAI processes FITS imagery from ground-based telescopes and provides:

- ✔️ **Astrometric fitting (WCS)** — from a sidereal image, a list of extracted star positions, a sidereal + rate-track series, or a single or series of rate-track images with no prior WCS
- ✔️ **Point source detection** — stars in sidereal frames, satellites in rate-track frames
- ✔️ **Streak detection** — satellite/debris trail detection and measurement, with multi-frame confirmation
- ✔️ **Photometry** — aperture photometry with star catalog cross-matching (Gaia, SDSS, SSTRC7)
- ✔️ **Batch processing** — process whole directories of imagery via CLI or REST API
- ✔️ **ML dataset export** — export detections as COCO-format datasets for model training

## Dependencies

- [astroeasy](https://github.com/zgazak/astroeasy) - Handles all Astrometry.net considerations (installation, index files). See its README for setup and getting your config right
- Star catalogs - See [senpai/catalog/README.md](senpai/catalog/README.md) for setup and usage


## Install

Install from PyPI (package name `astro-senpai`, imports as `senpai`):

```sh
pip install astro-senpai
```

### Development install

This repo uses [uv](https://docs.astral.sh/uv/) to manage python dependencies.  First, install uv.  Then,

```sh
make sync
```
or 
```sh
uv sync --all-extras
```

then

```sh
source .venv/bin/activate
```

## Run SENPAI

When you run the SENPAI api, it loads a config file, which you can specify on command line (or use default resources/config/local.yaml)


## Run SENPAI CLI

you can always provide your own config.yaml with --config <your_config.yaml> flag.

I want to:

1. fit + detect on one or more frames (auto-routes single/multi, sidereal/rate):

```sh
python -m senpai.cli.detect -f <your_fits_file(s)_or_dir> -o <your_output_directory> -D
```

`-D` enables non-star object detection (point sources in rate frames, streaks in sidereal/rate);
omit it to fit WCS + stars only. See `python -m senpai.cli.detect --help` for all options, and
`senpai.cli.batch` for whole-directory batch runs.





## Run SENPAI API

When you run the SENPAI api, it loads a config file, which you can specify on command line (or use default resources/config/local.yaml)

- the default config is resources/config/local.yaml
- on startup, SENPAI will check for downloaded indices files


### local SENPAI API

```sh
make run
```

or 

```sh
uv run python -m senpai.api.main --config resources/config/local.yaml
```


### containerized SENPAI API

#### Build container

```sh
docker build -t senpai .
```

Or, if you have a custom base image:
```sh
docker build --build-arg BASE_IMAGE=<your-custom-base-image> -t senpai .
```

#### run container

- **config** this container builds with resources/local/containerize.yaml
- **port** runs on 8000 in container

Run like this, noting that **target** is the path to your indices in the container, and must match your config file (containerized.yaml by default).

```sh
docker run -p 8000:8000 --mount type=bind,source=/path/to/indices/5000/5200,target=/home/starman/indices/5000/5200 senpai:latest
```

If you want to use a different config file (to specify different indices or other settings), you can do so like this:

```sh
docker run -p 8000:8000 \
    --mount type=bind,source=/path/to/indices/5000/5200,target=/home/starman/indices/5000/5200 \
    --mount type=bind,source=/path/to/your/config.yaml,target=/app/resources/config/containerized.yaml \
    senpai:latest
```

This will mount your custom config file in place of the default containerize.yaml. Make sure your custom config file follows the same format as the default configuration.

http://localhost:8000/docs

## Configuration

SENPAI is configured by a YAML file (validated by `senpai.core.config.AppConfig`), selected with
`--config <file>`, the `SENPAI_CONFIG_PATH` environment variable, or the default
`resources/config/local.yaml`. Every field can also be overridden by an environment variable using
the nested delimiter `__` (case-insensitive); env vars take precedence over the YAML:

```sh
# override astrometry.solver_mode and detection.snr_threshold without editing the YAML
ASTROMETRY__SOLVER_MODE=tetra3 DETECTION__SNR_THRESHOLD=5.0 \
  python -m senpai.cli.detect -f <your_fits_file>
```

Top-level config sections:

| Section | Purpose |
| --- | --- |
| `astrometry` | Plate solving: `solver_mode` (`dotnet` / `tetra3` / `chain` / `senpai`), index series + path, scale + search hints, SIP order. |
| `star_catalog` | Star catalog `type` (`gaia` / `gaia_local` / `sdss` / `sstrc7`) and on-disk path. |
| `detection` | Point-source + streak detection thresholds, sub-pixel centroid guard, WCS-refinement gating. |
| `streak` | Streak-extraction parameters (max FWHM, masking). |
| `photometry` | Aperture photometry, zero-point, limiting magnitude. |
| `calibrations` | Master bias / dark / flat handling. |
| `headers` / `observations` | FITS header-key mapping (exposure / time / site / pointing / tracking) for non-standard sensors. |
| `validation` / `wcs_validation` / `chain_gate` | Detection / WCS quality gates and chain-consistency checks. |
| `plotting` / `logging` / `runtime` | Debug/review plots, log level, output dir + run id. |

See `resources/config/local.yaml` for a complete annotated example.

## FITS header fields

SENPAI reads the following keywords from each input frame's FITS header. Only a couple are
strictly required; the rest improve accuracy, speed, or reproducibility when present. WCS
keywords (`CRVAL*`, `CD*`/`PC*`, `CTYPE*`) are **not** read from input frames — SENPAI plate-solves
the WCS itself, so they are outputs, not inputs. The keys below are what SENPAI reads directly;
track-mode classification and some calibration/observability header handling are additionally
configurable via the `headers` and `observations` config sections.

| Field | Status | Format / example | Used for | If missing |
| --- | --- | --- | --- | --- |
| `DATE-OBS` | **Required** | ISO-8601 UTC — `2024-07-07T11:20:37.375` | Time-orders frames within a collect; sets the observation timestamp. | The collect cannot be processed (invalid-input error). |
| `NAXIS1`, `NAXIS2` | **Required** | integer pixels — `2048` | Image dimensions. | Frame cannot be loaded (mandatory in any valid FITS image). |
| `TRKMODE` | **Strongly recommended** | `rate` or `sidereal` | Classifies each frame (drives rate-track vs sidereal solving). | Defaults to `sidereal`; rate frames are misclassified and rate-track detection is skipped. |
| `EXPTIME` | **Strongly recommended** | seconds, float — `1.0` | Streak-length and rate time-base in the shift solves. | Defaults to `1.0 s`; rate/streak scaling is wrong when the true exposure ≠ 1 s. |
| `TELTKRA`, `TELTKDEC` | **Strongly recommended** (rate tracks) | arcsec/sec, float — `19.96` / `-30.23` | Mount track rate; seeds the rate→rate search window with the object's true pixel rate. | Falls back to the measured streak length (a less reliable first-pair seed). |
| `OBJCTRA`/`OBJCTDEC` (or `RA`/`DEC`, `TELRA`/`TELDEC`, `CRVAL1`/`CRVAL2`) | **Strongly recommended** | sexagesimal or decimal degrees — `06 00 00` / `+12 00 00` | Boresight estimate used as a plate-solve position hint (speeds the solve, improves success). | Falls back to a blind plate solve — slower and more failure-prone. |
| `IMGSETID` (or `IMAGESETID`, `IMAGEID`) | Observability | string / UUID | Image-set id logged once per collect for lookup / reproducibility. | Logged as `unknown`; core detection unaffected. |
| `SENID`, `SITELAT`, `SITELONG`, `OBJECT` | Observability | string / sexagesimal — `OBS-01` | Logged once per collect (sensor, site, tracked object). | Logged as `unknown` / `?`; core detection unaffected. |

Notes:

- The plate-solve **scale** hint comes from config (`astrometry.min_width_degrees` /
  `astrometry.max_width_degrees`), not a header field.
- `TELTKRA`/`TELTKDEC` are the mount's commanded track rates, meaningful only on rate-tracked
  frames. Only the first rate→rate pair uses this seed — every later pair re-measures — so a
  missing or approximate value degrades gracefully.
- SENPAI reports celestial (RA/Dec) positions and does not use an observer location;
  `SITELAT`/`SITELONG` are read only for the per-collect log line.

## Citation

SENPAI implements the algorithms described in:

> Gazak, J. Z., Fisher, L., Phelps, M., Swindle, R., Baruela, L., & Fletcher, J. 2026, "SENPAI: Sidereal Enriched Rate-track Astrometry in Deep Imagery of Solar System Bodies", PASP, 138, 014502. [doi:10.1088/1538-3873/ae2b35](https://doi.org/10.1088/1538-3873/ae2b35)

If you use SENPAI in your research, please cite this paper:

```bibtex
@article{Gazak2026SENPAI,
  title   = {SENPAI: Sidereal Enriched Rate-track Astrometry in Deep Imagery of Solar System Bodies},
  author  = {Gazak, J. Zachary and Fisher, Lauren and Phelps, Matthew and Swindle, Ryan and Baruela, Leonard and Fletcher, Justin},
  journal = {Publications of the Astronomical Society of the Pacific},
  volume  = {138},
  number  = {1},
  pages   = {014502},
  year    = {2026},
  doi     = {10.1088/1538-3873/ae2b35}
}
```