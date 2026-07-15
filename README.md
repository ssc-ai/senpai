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

1. fit a single sidereal image:

```sh
python -m senpai.cli.single --image <your_fits_file> --output_dir <your_output_directory> --plot
```





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