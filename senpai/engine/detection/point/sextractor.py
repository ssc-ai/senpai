"""SExtractor-style source extraction for plate solving.

Reproduces astrometry.net's ``--use-source-extractor`` invocation in-process via
``sep`` (no external binary): the same background mesh, detection threshold,
convolution filter, and deblend/clean parameters the SExtractor binary uses when
astrometry.net drives it. This is the source list the Bayesian engine
feeds to the plate solve for sidereal frames (``astrometry.source_extractor:
sextractor``); its 1.5-sigma threshold recovers far more sources on
background-limited fields than the point-source detector, so marginal fields
that would otherwise fall below the astrometry source floor still solve.
"""

import numpy as np
import sep
from astropy.table import Table

from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.starfield import StarInImage, StarListImage

# 5x5 Gaussian convolution filter for FWHM=2.0 px — the exact kernel
# astrometry.net writes to disk for SExtractor (default gauss_2.0_5x5.conv).
_ASTROMETRY_SEXTRACTOR_FILTER = np.array(
    [
        [0.006319, 0.040599, 0.075183, 0.040599, 0.006319],
        [0.040599, 0.260856, 0.483068, 0.260856, 0.040599],
        [0.075183, 0.483068, 0.894573, 0.483068, 0.075183],
        [0.040599, 0.260856, 0.483068, 0.260856, 0.040599],
        [0.006319, 0.040599, 0.075183, 0.040599, 0.006319],
    ],
    dtype=np.float32,
)


def _detect_sources_sextractor(
    image: np.ndarray,
    max_sources: int = 100,
    threshold_sigma: float = 1.5,
) -> Table:
    """Extract sources via ``sep``, matching astrometry.net's SExtractor invocation.

    Parameter choices (astrometry.net/solver/augment-xylist.c + SExtractor's
    default.sex): DETECT_THRESH=1.5 sigma, BACK_SIZE=64, BACK_FILTERSIZE=3, the
    5x5 FWHM=2.0 Gaussian filter, minarea=5, DEBLEND_NTHRESH=32,
    DEBLEND_MINCONT=0.005, CLEAN=Y, CLEAN_PARAM=1.0.

    Args:
        image (np.ndarray): 2D image array.
        max_sources (int): cap on returned sources (brightest first).
        threshold_sigma (float): detection threshold in units of background RMS.

    Returns:
        Table: detected sources (xcentroid, ycentroid, flux), flux-descending,
            truncated to ``max_sources``; empty if none found.

    """
    data = np.ascontiguousarray(image, dtype=np.float64)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    objects = sep.extract(
        data - bkg,
        threshold_sigma * bkg.globalrms,
        minarea=5,
        filter_kernel=_ASTROMETRY_SEXTRACTOR_FILTER,
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        clean_param=1.0,
    )
    if len(objects) == 0:
        return Table()
    t = Table(objects[["x", "y", "flux"]])
    t.rename_column("x", "xcentroid")
    t.rename_column("y", "ycentroid")
    t.sort("flux", reverse=True)
    return t[:max_sources]


def extract_sextractor_sources(image: ProcessedFitsImage, max_detections: int = 100) -> StarListImage:
    """SExtractor-based extraction returning a senpai StarListImage.

    Args:
        image (ProcessedFitsImage): frame to extract from (``.data`` used).
        max_detections (int): maximum number of sources to return.

    Returns:
        StarListImage: detected sources with pixel coordinates and fluxes,
            carrying the frame's image metadata.

    """
    table = _detect_sources_sextractor(image.data, max_sources=max_detections)
    stars = [StarInImage(x=float(r["xcentroid"]), y=float(r["ycentroid"]), counts=float(r["flux"])) for r in table]
    return StarListImage(detections=stars, image_metadata=image.metadata)
