"""FastAPI routes for the astrometry solving endpoints."""

import logging
import time

from fastapi import APIRouter, Body, File, Request, UploadFile
from fastapi.responses import JSONResponse

from senpai.api.models.examples import StarListImageExample
from senpai.astrometry import solve_field
from senpai.core.config import get_config
from senpai.engine.detection.point.sidereal_extra import extract_point_sources
from senpai.engine.models.images import ProcessedFitsImage
from senpai.engine.models.starfield import StarField, StarListImage
from senpai.engine.plotting.images import plot_single_frame
from senpai.engine.utils.preprocessing import remove_column_and_row_medians

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/")
async def index(request: Request) -> JSONResponse:
    """Return API info and the current configuration.

    Args:
        request: The incoming FastAPI request.

    Returns:
        A JSON response with the API base URL and the serialized config.
    """
    logger.info("GET / - Fetching API info")
    config = get_config()
    return JSONResponse(
        {
            "api": request.base_url.__str__(),  # This gets the full URL path
            "config": config.model_dump(),  # This converts the config to a dictionary with all fields
        }
    )


@router.post("/solve/sources")
async def solve_sources(
    request: Request,
    sources: StarListImage = Body(description="sources list", examples=StarListImageExample().get_openapi_examples()),  # noqa: B008  # FastAPI dependency-injection default, required by framework
) -> StarField:
    """Solve astrometry for a supplied source list.

    Args:
        request (Request): FastAPI request object
        sources (StarListImage): A StarListImage object containing the stars to solve for and some image metadata

    Returns:
        StarField: A StarField object, which enriches the input StarListImage with astrometry information when solved.
    """
    logger.info("POST %s - Solving astrometry for image_id: %s", request.url.path, sources.image_metadata.image_id)
    start_time = time.time()
    wcs_starfield = solve_field(sources)
    time_taken = time.time() - start_time
    # Set response status:
    # 200 (OK) if astrometry solve was successful (wcs_field.fit = True)
    # 422 (Unprocessable Entity) if solve failed (wcs_field.fit = False)
    status_code = 200 if wcs_starfield.fit else 422

    logger.info(
        "POST returning %i in %.1f seconds %s - %s astrometry for image_id: %s",
        status_code,
        time_taken,
        request.url.path,
        "Solved" if wcs_starfield.fit else "Failed to solve",
        sources.image_metadata.image_id,
    )

    return JSONResponse(content=wcs_starfield.model_dump(), status_code=status_code)


@router.post("/solve/fits")
async def solve_fits(
    request: Request,
    fits_file: UploadFile = File(..., description="FITS image file"),  # noqa: B008  # FastAPI dependency-injection default, required by framework
) -> StarField:
    """Solve astrometry for an uploaded FITS file.

    Args:
        request (Request): FastAPI request object
        fits_file (UploadFile): The uploaded FITS file

    Returns:
        StarField: A StarField object containing the astrometry solution
    """
    logger.info("POST %s - Solving astrometry for FITS file: %s", request.url.path, fits_file.filename)

    # TODO: Add your FITS processing logic here
    # You'll likely want to use astropy.io.fits to handle the FITS data
    # and extract the necessary information for solve_field()

    start_time = time.time()
    fits_content = await fits_file.read()

    fits_file = ProcessedFitsImage.from_file_bytes(fits_content)

    # plot_single_frame(fits_file.data, output_file="sidereal_original.png", scale=True)

    fits_file = remove_column_and_row_medians(fits_file)
    # plot_single_frame(fits_file.data, output_file="sidereal_subtracted.png", scale=True)

    sources, fwhm = extract_point_sources(fits_file, max_detections=100)

    # plot_single_frame(
    #     fits_file.data,
    #     starlist=sources,
    #     output_file="sidereal_detected.png",
    #     scale=True,
    #     markersize=1.5 * fwhm,
    #     centercross=False,
    # )

    wcs_field = solve_field(sources)

    time_taken = time.time() - start_time

    plot_single_frame(
        fits_file.data,
        starfield=wcs_field,
        output_file="sidereal_solved.png",
        scale=True,
        markersize=2 * fwhm,
        centercross=False,
    )

    # Set response status:
    # 200 (OK) if astrometry solve was successful (wcs_field.fit = True)
    # 422 (Unprocessable Entity) if solve failed (wcs_field.fit = False)
    status_code = 200 if wcs_field.fit else 422

    logger.info(
        "POST returning %i in %.1f seconds %s - %s astrometry for image_id: %s",
        status_code,
        time_taken,
        request.url.path,
        "Solved" if wcs_field.fit else "Failed to solve",
        sources.image_metadata.image_id,
    )

    return JSONResponse(content=wcs_field.model_dump(), status_code=status_code)
