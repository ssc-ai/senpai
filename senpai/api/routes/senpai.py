"""SENPAI API routes — detection, astrometry, photometry.

Endpoints:
    GET  /senpai/                Health check
    POST /senpai/detect          Full pipeline (auto sidereal+rate organise)
    POST /senpai/detect/upload   Alias for /detect (backward compat)
    POST /senpai/sidereal        Single-frame sidereal processing (via collect)
    POST /senpai/rate            Single-frame rate-track processing (via collect)
"""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from senpai.api.models.returns import (
    DetectResponse,
    frame_result_from_rate,
    frame_result_from_sidereal,
)
from senpai.core.config import get_config
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.processing.collect import process_senpai_collect
from senpai.engine.utils.file_io import load_base64_files

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class FilePayloadItem(BaseModel):
    """One base64-encoded FITS file plus its position in the upload sequence."""

    file: str = Field(..., description="Base64-encoded FITS file bytes")
    sequence_id: int | None = Field(None, description="Sequence ID for ordering")
    sequence_count: int | None = Field(None, description="Total sequence count")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_collect_pipeline(
    payload: list[FilePayloadItem],
    request_path: str,
) -> DetectResponse:
    """Run the collect pipeline and build a DetectResponse."""
    encoded_files = [item.file for item in payload]
    file_list: list[ProcessedFitsImage] = load_base64_files(encoded_files)

    senpai_run = process_senpai_collect(file_list)

    frames = []
    for frame in senpai_run.sidereal_frames:
        frames.append(frame_result_from_sidereal(frame))
    for frame in senpai_run.rate_track_frames:
        frames.append(frame_result_from_rate(frame))
    frames.sort(key=lambda f: f.index)

    correlated = [
        cs.model_dump(mode="json") for cs in senpai_run.correlated_streaks
    ]

    logger.info(
        "POST %s — done: %d frames, %d correlated streaks, %s",
        request_path,
        len(frames),
        len(correlated),
        "completed" if senpai_run.completed else f"error: {senpai_run.error_message}",
    )
    return DetectResponse(frames=frames, correlated_streaks=correlated)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/")
async def index(request: Request) -> dict[str, str]:
    """Health check returning the API base URL and version.

    Args:
        request: The incoming FastAPI request.

    Returns:
        A mapping with the API base URL and the current SENPAI version.
    """
    config = get_config()
    return {"api": str(request.base_url), "version": config.version}


# ---------------------------------------------------------------------------
# /detect — full multi-frame pipeline (sidereal + rate auto-organise)
# ---------------------------------------------------------------------------


@router.post("/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    payload: list[FilePayloadItem],
) -> DetectResponse:
    """Process frames through the full SENPAI collect pipeline.

    Automatically organises frames into sidereal and rate-track based on
    FITS header metadata, solves astrometry, performs photometry, and
    detects sources.

    **Request**::

        [{"file": "<base64 FITS>", "sequence_id": 0, "sequence_count": 1}]

    **Response**: ``DetectResponse`` with per-frame astrometry, photometry,
    seeing, and detections (each with RA/Dec and calibrated magnitude).
    """
    logger.info("POST %s — %d frame(s)", request.url.path, len(payload))
    return _run_collect_pipeline(payload, request.url.path)


@router.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(
    request: Request,
    payload: list[FilePayloadItem],
) -> DetectResponse:
    """Alias for ``/detect`` — backward compatibility."""
    return await detect(request, payload)


# ---------------------------------------------------------------------------
# /sidereal — sidereal processing (via collect pipeline)
# ---------------------------------------------------------------------------


@router.post("/sidereal", response_model=DetectResponse)
async def process_sidereal(
    request: Request,
    payload: list[FilePayloadItem],
) -> DetectResponse:
    """Process sidereal frame(s) through the collect pipeline.

    Uses the same full pipeline as /detect — astrometry, catalog matching,
    photometry, and optional streak detection.
    """
    logger.info("POST %s — %d sidereal frame(s)", request.url.path, len(payload))
    return _run_collect_pipeline(payload, request.url.path)


# ---------------------------------------------------------------------------
# /rate — rate-track processing (via collect pipeline)
# ---------------------------------------------------------------------------


@router.post("/rate", response_model=DetectResponse)
async def process_rate(
    request: Request,
    payload: list[FilePayloadItem],
) -> DetectResponse:
    """Process rate-track frame(s) through the collect pipeline.

    Uses the same full pipeline as /detect — streak measurement, WCS from
    streak centroids, catalog matching, photometry, and detection.
    """
    logger.info("POST %s — %d rate-track frame(s)", request.url.path, len(payload))
    return _run_collect_pipeline(payload, request.url.path)
