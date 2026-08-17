from typing import Any

from pydantic import BaseModel

from senpai.engine.models.starfield import ImageMetadata, StarInImage, StarListImage

# A small (x, y, counts) sample of a real 1024x1024 detection list, spanning the
# brightness range of the field. This is an illustrative payload for /docs, not
# a solvable star field -- a full field would swamp the Swagger UI.
#
# It is a literal on purpose. FastAPI evaluates ``Body(examples=...)`` at import
# time, so anything read from disk here runs on every ``create_app()``; reading
# a repo test fixture there broke non-editable installs entirely (issue #6).
_EXAMPLE_DETECTIONS: list[tuple[float, float, float]] = [
    (125.432, 568.183, 917909.8),
    (291.171, 210.435, 10242.4),
    (611.514, 645.360, 6810.8),
    (501.782, 543.489, 4385.3),
    (545.078, 117.519, 3354.3),
    (158.725, 198.313, 2544.5),
    (205.254, 510.905, 1744.2),
    (372.452, 621.724, 1586.4),
    (415.934, 294.103, 1235.9),
    (78.394, 195.271, 1004.6),
]


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
            self._value = StarListImage(
                detections=[StarInImage(x=x, y=y, counts=counts) for x, y, counts in _EXAMPLE_DETECTIONS],
                image_metadata=ImageMetadata(
                    image_id="x_y_counts", width=1024, height=1024, boresight_ra=245.45, boresight_dec=41.8
                ),
            )
        return self._value

    def get_openapi_examples(self) -> dict[str, dict[str, Any]]:
        """Convert to OpenAPI examples format"""
        return [self.value]
