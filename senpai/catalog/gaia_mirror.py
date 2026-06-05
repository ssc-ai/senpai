"""Build a trimmed local Gaia DR3 mirror (G <= mag_limit) for offline catalog
queries — the data-engineering side of [[project-local-gaia-mirror]].

Online Gaia TAP fetches dominate the burr per-batch runtime (~120-170 s/batch).
A trimmed all-sky mirror turns each fetch into a sub-second local read. Only the
columns senpai actually uses are kept, with a magnitude cut, so all-sky G<=20 is
~50 GB (not the ~10-15 TB full catalog).

Dep-free: numpy + astroquery (download), numpy only (ingest/query). HEALPix tiles
are derived for FREE from source_id (Gaia encodes HEALPix level 12 in the top
bits), so no healpy is needed; a small JSON bbox index makes box->tiles a cheap
overlap scan at query time (see senpai.catalog.gaia_local).

Usage:
    python -m senpai.catalog.gaia_mirror download --out /path/to/gaia_chunks --mag-limit 20
    python -m senpai.catalog.gaia_mirror ingest  --chunks /path/to/gaia_chunks --mirror /path/to/gaia_g20
Then set in the config:  star_catalog: {type: gaia_local, path: /path/to/gaia_g20}
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Gaia DR3 source_id encodes HEALPix level 12 in the high bits:
#   healpix_L = source_id >> (35 + 2*(12 - L))
# Tile the mirror at level 4 -> 12*4^4 = 3072 equal-area tiles (~3.7 deg each),
# so a ~3 deg field overlaps only a handful of tiles.
HPX_LEVEL = 4
HPX_SHIFT = 35 + 2 * (12 - HPX_LEVEL)  # = 51
GAIA_DR3_NSOURCES = 1_811_709_771  # DR3 gaia_source row count (random_index range)

# Stored columns. ra/dec MUST be f8 (f4 loses arcsec precision at RA~300 deg).
MIRROR_DTYPE = np.dtype([
    ("source_id", "i8"),
    ("ra", "f8"), ("dec", "f8"),      # degrees
    ("g", "f4"), ("bp", "f4"), ("rp", "f4"),
    ("pmra", "f4"), ("pmdec", "f4"),  # mas/yr
])
_GAIA_COLS = [
    "source_id", "ra", "dec",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "pmra", "pmdec",
]
_FIELD_FROM_COL = {
    "phot_g_mean_mag": "g", "phot_bp_mean_mag": "bp", "phot_rp_mean_mag": "rp",
    "pmra": "pmra", "pmdec": "pmdec",
}


def download_chunks(
    out_dir: str, mag_limit: float = 20.0, step: int = 5_000_000,
    login_user: str | None = None, max_retries: int = 4, retry_wait: float = 30.0,
    pause: float = 3.0, max_consecutive_failures: int = 6, job_timeout: float = 300.0,
) -> None:
    """Download G<=mag_limit Gaia DR3 in random_index-tiled async chunks.

    Tiling by ``random_index`` (a uniform shuffle) gives equal-size chunks and
    beats the per-job row cap; transfers only the kept columns/rows (~50 GB for
    G<=20, ~80 GB for G<=21, vs ~600 GB for the full CSV dump). Resumable: existing
    chunk files are skipped, so a killed run just re-invokes. The Gaia archive is
    flaky (DR4-evolution warning), so each chunk is retried with backoff and, if it
    still fails, skipped (logged) rather than crashing the whole multi-hour run —
    a later re-run picks up the gaps.
    """
    import time

    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    if login_user:
        Gaia.login(user=login_user)  # higher async limits/priority than anonymous

    os.makedirs(out_dir, exist_ok=True)
    nchunks = GAIA_DR3_NSOURCES // step + 1
    cols = ", ".join(_GAIA_COLS)
    done = failed = consec_fail = 0
    for i in range(nchunks):
        path = os.path.join(out_dir, f"chunk_{i:04d}.npy")
        if os.path.exists(path):
            done += 1
            continue
        # Circuit-breaker: if the archive is blocking/down, stop rather than hammer
        # it all night. Transient single-chunk failures reset on the next success.
        if consec_fail >= max_consecutive_failures:
            logger.critical(
                "ABORTING: %d consecutive chunk failures — archive likely "
                "throttling/down. %d/%d done. Re-run later to resume.",
                consec_fail, done, nchunks,
            )
            break
        lo, hi = i * step, (i + 1) * step
        adql = (
            f"SELECT {cols} FROM gaiadr3.gaia_source "
            f"WHERE phot_g_mean_mag <= {mag_limit} "
            f"AND random_index >= {lo} AND random_index < {hi}"
        )
        for attempt in range(1, max_retries + 1):
            try:
                # Run the (blocking) async job under a hard wall-clock timeout so a
                # job stuck "EXECUTING" server-side can't silently block forever —
                # a timeout becomes a normal failure (retry -> circuit-breaker).
                import concurrent.futures as _cf

                _ex = _cf.ThreadPoolExecutor(max_workers=1)
                try:
                    tbl = _ex.submit(
                        lambda: Gaia.launch_job_async(adql).get_results()
                    ).result(timeout=job_timeout)
                except _cf.TimeoutError:
                    raise TimeoutError(f"job exceeded {job_timeout:.0f}s")
                finally:
                    _ex.shutdown(wait=False)  # abandon a stuck worker thread
                arr = np.empty(len(tbl), dtype=MIRROR_DTYPE)
                arr["source_id"] = np.asarray(tbl["source_id"], dtype="i8")
                arr["ra"] = np.asarray(tbl["ra"], dtype="f8")
                arr["dec"] = np.asarray(tbl["dec"], dtype="f8")
                for col, field in _FIELD_FROM_COL.items():
                    # masked/NaN (missing BP/RP/pm) survive as NaN in f4
                    arr[field] = np.asarray(np.ma.filled(tbl[col], np.nan), dtype="f4")
                # write atomically (tmp + rename) so a kill mid-write can't leave a
                # truncated chunk that the resume logic would wrongly skip. np.save
                # appends ".npy" to bare paths, so write via a file handle to keep
                # the tmp name exact for the rename.
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    np.save(f, arr)
                os.replace(tmp, path)
                done += 1
                consec_fail = 0
                logger.info(
                    "chunk %d/%d: %d stars (random_index %d-%d) -> %s",
                    i + 1, nchunks, len(arr), lo, hi, os.path.basename(path),
                )
                if pause > 0:
                    time.sleep(pause)  # be a polite citizen between jobs
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        "chunk %d/%d attempt %d/%d failed (%s); retrying in %.0fs",
                        i + 1, nchunks, attempt, max_retries, e, retry_wait,
                    )
                    time.sleep(retry_wait)
                else:
                    failed += 1
                    consec_fail += 1
                    logger.error(
                        "chunk %d/%d FAILED after %d attempts (%s); skipping — "
                        "re-run later to fill the gap", i + 1, nchunks, max_retries, e,
                    )
    logger.info(
        "download pass complete: %d/%d chunks present, %d still failing in %s",
        done, nchunks, failed, out_dir,
    )


def ingest(chunk_dir: str, mirror_dir: str) -> None:
    """Regroup random_index chunks into per-HEALPix-tile files + a bbox index.

    Streams chunk-by-chunk and appends each chunk's stars to per-tile raw .bin
    files (so peak memory is one chunk, not the whole ~48 GB catalog). Then reads
    each tile once to record its RA/Dec bounding box in index.json — that index
    lets the query layer pick overlapping tiles without any HEALPix geometry.
    """
    os.makedirs(mirror_dir, exist_ok=True)
    # Fresh ingest: clear any stale tile .bin (append mode would double them).
    for stale in glob.glob(os.path.join(mirror_dir, "tile_*.bin")):
        os.remove(stale)

    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npy")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npy in {chunk_dir}")
    for ci, cf in enumerate(chunk_files):
        a = np.load(cf)
        tid = (a["source_id"] >> HPX_SHIFT).astype("i8")
        order = np.argsort(tid, kind="stable")
        a, tid = a[order], tid[order]
        uniq, starts = np.unique(tid, return_index=True)
        starts = list(starts) + [len(a)]
        for j, t in enumerate(uniq):
            part = a[starts[j]:starts[j + 1]]
            with open(os.path.join(mirror_dir, f"tile_{int(t):05d}.bin"), "ab") as fh:
                part.tofile(fh)
        logger.info("ingested chunk %d/%d (%s)", ci + 1, len(chunk_files), os.path.basename(cf))

    index = {"hpx_level": HPX_LEVEL, "shift": HPX_SHIFT,
             "dtype": MIRROR_DTYPE.descr, "tiles": {}}
    total = 0
    for tf in sorted(glob.glob(os.path.join(mirror_dir, "tile_*.bin"))):
        arr = np.fromfile(tf, dtype=MIRROR_DTYPE)
        if len(arr) == 0:
            continue
        t = int(os.path.basename(tf)[5:-4])
        index["tiles"][str(t)] = {
            "file": os.path.basename(tf),
            "ra_min": float(arr["ra"].min()), "ra_max": float(arr["ra"].max()),
            "dec_min": float(arr["dec"].min()), "dec_max": float(arr["dec"].max()),
            "n": int(len(arr)),
        }
        total += len(arr)
    with open(os.path.join(mirror_dir, "index.json"), "w") as fh:
        json.dump(index, fh)
    logger.info("ingest complete: %d tiles, %d stars -> %s", len(index["tiles"]), total, mirror_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build/ingest a local Gaia DR3 mirror.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pd = sub.add_parser("download", help="Download G<=mag_limit chunks via Gaia TAP.")
    pd.add_argument("--out", required=True)
    pd.add_argument("--mag-limit", type=float, default=20.0)
    pd.add_argument("--step", type=int, default=5_000_000)
    pd.add_argument("--login-user", default=None)
    pd.add_argument("--job-timeout", type=float, default=300.0)
    pi = sub.add_parser("ingest", help="Regroup chunks into HEALPix-tile mirror.")
    pi.add_argument("--chunks", required=True)
    pi.add_argument("--mirror", required=True)
    a = p.parse_args(argv)
    if a.cmd == "download":
        download_chunks(a.out, a.mag_limit, a.step, a.login_user, job_timeout=a.job_timeout)
    else:
        ingest(a.chunks, a.mirror)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
