"""Unit tests for the engine data models.

Covers ``SenpaiRun`` analysis-chain bookkeeping methods (valid-path construction, chain
logging, frame lookup) and the slimmed serialized-output helper
``_starfield_for_output`` / ``MAX_SERIALIZED_CATALOG_STARS`` (serialized StarFields keep
only the brightest catalog stars while the live frame retains the full list).

All tests use synthetic, deterministic data -- no network, astrometry, or catalog access.
"""

from datetime import datetime

import numpy as np
from astropy.io import fits

from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.metadata import CollectionMetadata, ImageMetadata
from senpai.engine.models.senpai import (
    MAX_SERIALIZED_CATALOG_STARS,
    RateTrackFrame,
    SenpaiRun,
    SiderealFrame,
    _starfield_for_output,
)
from senpai.engine.models.starfield import StarField, StarInSpace


def _make_sidereal_frame(index: int, data: np.ndarray) -> SiderealFrame:
    """Build a minimal sidereal frame from synthetic image data.

    Args:
        index: Frame index within the run.
        data: Backing image array.

    Returns:
        A ``SiderealFrame`` wrapping the data with a minimal header.
    """
    header = fits.Header()
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    frame = ProcessedFitsImage(
        data=data,
        header=header,
        data_type=data.dtype,
        metadata=ImageMetadata(width=data.shape[1], height=data.shape[0]),
    )
    return SiderealFrame(frame=frame, index=index, timestamp=datetime(2024, 1, 1, 0, 0, index))


def _low_noise_array(rng: np.random.Generator, shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Build a low-noise uint16 image.

    Args:
        rng: Seeded random generator.
        shape: Output image shape (height, width).

    Returns:
        A uint16 array of low-amplitude noise.
    """
    return rng.integers(100, 500, size=shape, dtype=np.uint16)


class TestSenpaiRunMethods:
    """Analysis-chain bookkeeping methods on ``SenpaiRun``."""

    def _make_run(
        self,
        sidereal_frames: list[SiderealFrame] | None = None,
        rate_track_frames: list[RateTrackFrame] | None = None,
    ) -> SenpaiRun:
        """Build a run with the given frames and no collect metadata of interest.

        Args:
            sidereal_frames: Sidereal frames for the run.
            rate_track_frames: Rate-track frames for the run.

        Returns:
            A ``SenpaiRun`` populated with the supplied frames.
        """
        return SenpaiRun(
            id="test",
            num_frames=0,
            collect_metadata=CollectionMetadata(),
            sidereal_frames=sidereal_frames or [],
            rate_track_frames=rate_track_frames or [],
        )

    def test_create_valid_path_no_frames_does_not_raise(self) -> None:
        """Building the valid path on an empty run yields no frame shifts."""
        run = self._make_run()
        run.create_valid_path()
        assert run.frame_shifts == []

    def test_create_valid_path_single_frame_creates_no_shifts(self) -> None:
        """A single-frame run has no adjacent pairs, so no shifts are created."""
        rng = np.random.default_rng(0)
        f0 = _make_sidereal_frame(0, _low_noise_array(rng))
        run = SenpaiRun(
            id="test",
            num_frames=1,
            collect_metadata=CollectionMetadata(),
            sidereal_frames=[f0],
            rate_track_frames=[],
        )
        run.create_valid_path()
        assert len(run.frame_shifts) == 0

    def test_log_analysis_chain_empty_shifts_does_not_raise(self) -> None:
        """Logging the analysis chain with no shifts completes without error."""
        run = self._make_run()
        run.log_analysis_chain()

    def test_get_frame_by_index_missing_returns_none(self) -> None:
        """Looking up a frame index that is not present returns None."""
        rng = np.random.default_rng(1)
        f = _make_sidereal_frame(0, _low_noise_array(rng))
        run = SenpaiRun(
            id="test",
            num_frames=1,
            collect_metadata=CollectionMetadata(),
            sidereal_frames=[f],
            rate_track_frames=[],
        )
        assert run.get_frame_by_index(999) is None


# --------------------------------------------------------------------------- #
# Slim serialized outputs (_starfield_for_output / MAX_SERIALIZED_CATALOG_STARS)
# --------------------------------------------------------------------------- #
def _catalog_stars(n: int, magnitudes: np.ndarray) -> list[StarInSpace]:
    """Build ``n`` catalog stars with the given magnitudes.

    Args:
        n: Number of stars to build.
        magnitudes: Per-star magnitudes (length >= ``n``).

    Returns:
        A list of ``StarInSpace`` with linked catalog ids.
    """
    return [
        StarInSpace(
            ra=10.0 + i * 0.001, dec=20.0, magnitude=float(magnitudes[i]),
            catalog="gaia", catalog_id=f"gaia-{i}", x=float(i), y=1.0,
        )
        for i in range(n)
    ]


def _starfield_with_catalog(stars: list[StarInSpace]) -> StarField:
    """Build an unfit starfield carrying the given catalog stars.

    Args:
        stars: Catalog stars to attach.

    Returns:
        A ``StarField`` with the catalog stars and no WCS.
    """
    return StarField(
        catalog_stars=stars,
        detections=[],
        image_metadata=ImageMetadata(width=100, height=100),
        wcs=None,
    )


def test_starfield_for_output_keeps_only_brightest_500() -> None:
    """Serialization keeps exactly the 500 brightest catalog stars (smallest magnitudes)."""
    rng = np.random.default_rng(0)
    mags = rng.uniform(5.0, 20.0, size=600)
    sf = _starfield_with_catalog(_catalog_stars(600, mags))

    out = _starfield_for_output(sf)

    assert MAX_SERIALIZED_CATALOG_STARS == 500
    assert len(out.catalog_stars) == 500
    # Exactly the 500 smallest magnitudes (brightest) are kept.
    kept = sorted(s.magnitude for s in out.catalog_stars)
    assert kept == sorted(mags)[:500]


def test_starfield_for_output_preserves_id_linkage() -> None:
    """The kept stars retain their catalog linkage and id<->magnitude pairing."""
    rng = np.random.default_rng(1)
    mags = rng.uniform(5.0, 20.0, size=600)
    sf = _starfield_with_catalog(_catalog_stars(600, mags))

    out = _starfield_for_output(sf)

    assert all(s.catalog_id is not None for s in out.catalog_stars)
    assert all(s.catalog == "gaia" for s in out.catalog_stars)
    # The kept stars' ids match their original id<->magnitude pairing.
    mag_by_id = {s.catalog_id: s.magnitude for s in sf.catalog_stars}
    for star in out.catalog_stars:
        assert mag_by_id[star.catalog_id] == star.magnitude


def test_starfield_for_output_leaves_live_starfield_untouched() -> None:
    """Serialization does not mutate the in-memory starfield's full catalog list."""
    mags = np.linspace(5.0, 20.0, 600)
    sf = _starfield_with_catalog(_catalog_stars(600, mags))
    _starfield_for_output(sf)
    # The in-memory frame keeps the full catalog list.
    assert len(sf.catalog_stars) == 600


def test_starfield_for_output_returns_same_object_at_or_below_cap() -> None:
    """A starfield already at or below the cap is returned unchanged (same object)."""
    mags = np.linspace(5.0, 20.0, 10)
    sf = _starfield_with_catalog(_catalog_stars(10, mags))
    # Under the cap: returned unchanged (same object, no copy).
    assert _starfield_for_output(sf) is sf


def test_starfield_for_output_handles_none() -> None:
    """A None starfield passes through as None."""
    assert _starfield_for_output(None) is None
