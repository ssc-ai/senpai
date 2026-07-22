"""Shared pytest fixtures for the SENPAI test suite."""

from __future__ import annotations

import numpy as np
import pytest

from senpai.core.constants import TEST_DATA_DIR
from senpai.engine.models.starfield import ImageMetadata, StarInImage, StarListImage


@pytest.fixture
def xyls_data() -> StarListImage:
    """Load the bundled x/y/counts detection list as a StarListImage.

    Returns:
        A StarListImage populated from the ``x_y_counts_1024_1024.txt`` test
        data file, sized 1024x1024.
    """
    data = np.loadtxt(TEST_DATA_DIR / "x_y_counts_1024_1024.txt", delimiter="\t", dtype=float)

    return StarListImage(
        detections=[StarInImage(x=row[0], y=row[1], counts=row[2]) for row in data],
        image_metadata=ImageMetadata(image_id="x_y_counts", width=1024, height=1024),
    )
