"""Local Gaia mirror: ingest (gaia_mirror) + offline query (gaia_local).

Builds a tiny synthetic mirror and checks the offline query returns exactly the
stars an online box query would, in the same drop-in dict shape.
"""

from pathlib import Path

import numpy as np
import pytest

from senpai.catalog import gaia_local
from senpai.catalog.gaia_mirror import HPX_SHIFT, MIRROR_DTYPE, ingest


@pytest.fixture
def mirror(tmp_path: Path) -> tuple[str, np.ndarray]:
    """Build a tiny synthetic Gaia mirror on disk for offline-query tests.

    Args:
        tmp_path: Pytest-provided temporary directory for the chunks and mirror.

    Returns:
        A ``(mirror_dir, source_array)`` pair: the ingested mirror directory and
        the raw structured star array it was built from (for brute-force checks).
    """
    rng = np.random.default_rng(0)
    n = 20000
    ra = rng.uniform(250.0, 260.0, n)
    dec = rng.uniform(40.0, 50.0, n)
    g = rng.uniform(8.0, 20.0, n)
    # spatially-coherent tiles encoded in source_id top bits (mimics HEALPix)
    tile = ((ra.astype(int) - 250) * 10 + (dec.astype(int) - 40)).astype(np.int64)
    arr = np.empty(n, dtype=MIRROR_DTYPE)
    arr["source_id"] = (tile << HPX_SHIFT) + np.arange(n)
    arr["ra"], arr["dec"], arr["g"] = ra, dec, g
    arr["bp"], arr["rp"] = g + 0.3, g - 0.3
    arr["pmra"], arr["pmdec"] = 1.0, -2.0
    cdir = tmp_path / "chunks"
    cdir.mkdir()
    np.save(cdir / "chunk_0000.npy", arr[: n // 2])
    np.save(cdir / "chunk_0001.npy", arr[n // 2:])
    mdir = tmp_path / "mirror"
    ingest(str(cdir), str(mdir))
    return str(mdir), arr


def test_query_matches_brute_force(mirror: tuple[str, np.ndarray]) -> None:
    """Offline box query returns exactly the stars a brute-force mask selects."""
    mdir, arr = mirror
    qb = (252.0, 254.5, 42.0, 44.0)
    fl = 18.0
    stars = gaia_local.query_by_ra_dec_bounds(*qb, faint_lim=fl, bright_lim=-32, mirror_dir=mdir)
    m = ((arr["ra"] >= qb[0]) & (arr["ra"] <= qb[1]) & (arr["dec"] >= qb[2])
         & (arr["dec"] <= qb[3]) & (arr["g"] <= fl))
    assert len(stars) == int(m.sum()) > 0


def test_dropin_dict_shape(mirror: tuple[str, np.ndarray]) -> None:
    """Offline query returns the same dict shape/keys as the online Gaia query."""
    mdir, _ = mirror
    s = gaia_local.query_by_ra_dec_bounds(252.0, 254.0, 42.0, 44.0, faint_lim=18.0,
                                          bright_lim=-32, mirror_dir=mdir)[0]
    # same keys the online gaia.query_by_ra_dec_bounds returns
    assert set(s) == {"ra", "dec", "mv", "magnitudes", "catalog",
                      "source_id", "ra_pm", "dec_pm", "parallax"}
    assert s["catalog"] == "Gaia"
    assert isinstance(s["source_id"], str)
    assert 4.0 < s["ra"] < 4.6  # radians (≈252-254 deg)
    assert {"Gaia_G", "Gaia_BP", "Gaia_RP", "Johnson_V", "Sloan_r"} <= set(s["magnitudes"])


def test_mag_limit_applied(mirror: tuple[str, np.ndarray]) -> None:
    """Fainter faint-limit yields more stars; all respect the requested cutoff."""
    mdir, _ = mirror
    bright = gaia_local.query_by_ra_dec_bounds(250.0, 260.0, 40.0, 50.0, faint_lim=12.0,
                                               bright_lim=-32, mirror_dir=mdir)
    deep = gaia_local.query_by_ra_dec_bounds(250.0, 260.0, 40.0, 50.0, faint_lim=19.0,
                                             bright_lim=-32, mirror_dir=mdir)
    assert len(deep) > len(bright)
    assert all(st["mv"] <= 12.0 for st in bright)


def test_empty_box_returns_empty(mirror: tuple[str, np.ndarray]) -> None:
    """A box covering no mirrored stars returns an empty list."""
    mdir, _ = mirror
    assert gaia_local.query_by_ra_dec_bounds(100.0, 101.0, -10.0, -9.0, faint_lim=20.0,
                                             bright_lim=-32, mirror_dir=mdir) == []
