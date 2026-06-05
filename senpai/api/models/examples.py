from typing import Any, Dict

import numpy as np
from pydantic import BaseModel

from senpai.core.constants import TEST_DATA_DIR
from senpai.engine.models.starfield import ImageMetadata, StarInImage, StarListImage


class StarListImageExample(BaseModel):
    def __init__(self):
        super().__init__()
        self._value: StarListImage | None = None

    @property
    def summary(self) -> str:
        return "A list of stars in an image with image metadata"

    @property
    def value(self) -> StarListImage:
        """Get example StarListImage value"""
        if self._value is None:
            data = np.loadtxt(TEST_DATA_DIR / "x_y_counts_1024_1024.txt", delimiter="\t", dtype=float)
            self._value = StarListImage(
                detections=[StarInImage(x=row[0], y=row[1], counts=row[2]) for row in data],
                image_metadata=ImageMetadata(
                    image_id="x_y_counts", width=1024, height=1024, boresight_ra=245.45, boresight_dec=41.8
                ),
            )
        return self._value

    def get_openapi_examples(self) -> Dict[str, Dict[str, Any]]:
        """Convert to OpenAPI examples format"""
        return [self.value]
