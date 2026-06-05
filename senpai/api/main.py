"""
Main entry point for API server.
"""

import concurrent.futures
import contextlib
import logging
import os
from pathlib import Path

import click
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from senpai.astrometry import examine_indices, test_astrometry_install
from senpai.core.config import AppConfig, initialize_config
from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from senpai.core.logging import get_local_config, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_EXECUTOR_MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def setup_routes(app: FastAPI) -> None:
    """Configure application routes"""
    logger.info("Setting up routes...")
    # Import routes here to avoid circular imports
    from senpai.api.routes import astrometry, senpai

    app.include_router(senpai.router, tags=["SENPAI"], prefix="/senpai")
    app.include_router(astrometry.router, tags=["Astrometry"], prefix="/astrometry")

    logger.info("Routes setup completed")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    max_workers_env = os.getenv("SENPAI_EXECUTOR_WORKERS")
    max_workers = int(max_workers_env) if max_workers_env else DEFAULT_EXECUTOR_MAX_WORKERS
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_pool_worker)
    logger.info(f"Using ProcessPoolExecutor with max_workers={max_workers}")

    app.state.executor = executor

    try:
        test_astrometry_install()
        examine_indices()
        yield
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _init_pool_worker():
    from senpai.core.config import initialize_config

    cfg_path = os.getenv("SENPAI_CONFIG")
    if cfg_path:
        initialize_config(Path(cfg_path))


def create_app(config: AppConfig | str | Path | None = None) -> FastAPI:
    """Application factory supporting both local and Lambda environments"""
    # Suppress specific warnings
    # Setup logging, will check if already configured
    # Example usage
    setup_logging(
        level="INFO",
        disabled_loggers=["matplotlib", "astropy.io.fits", "scipy.optimize"],
    )

    if config is None:
        config = Path(os.environ.get("SENPAI_CONFIG", LOCAL_APP_CONFIG_OVERRIDE))

    if isinstance(config, (str, Path)):
        logger.info(f"Loading config from path: {config}")
        config = initialize_config(config)

    # Create FastAPI app with conditional docs
    logger.info(f"Creating app with config:\n{config.model_dump_json(indent=2)}")
    app = FastAPI(
        title="SENPAI API",
        version=config.version,
        debug=config.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store uvicorn logging config
    app.state.log_config = get_local_config()
    app.state.log_level = config.logging.level.lower()

    # Add CORS middleware configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup routes
    setup_routes(app)

    logger.info("App created successfully")

    return app


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--workers", default=1, show_default=True, envvar="WORKERS")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=LOCAL_APP_CONFIG_OVERRIDE,  # CLI use defaults to local config. Lambda does not use this.
    show_default=True,
)
def run_server(host: str, port: int, workers: int | None, config: Path | None = None) -> None:
    """Run the API server locally"""
    import uvicorn

    # Load config first
    config_path = config or LOCAL_APP_CONFIG_OVERRIDE
    app_config = initialize_config(config_path)

    logger.info("Starting uvicorn server...")

    try:
        workers = int(workers)
    except ValueError:
        workers = 1

    if workers > 1:
        # Set config path in the env so zero-arg factory can pick it up
        os.environ.setdefault("SENPAI_CONFIG", str(config_path))

        # Using zero-arg factory so workers can spawn
        uvicorn.run(
            "senpai.api.main:create_app",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            log_config=get_local_config(),
            log_level=app_config.logging.level.lower(),
        )
    else:
        # Create and run app
        app = create_app(app_config)
        # Single process - use in-memory app
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_config=app.state.log_config,
            log_level=app.state.log_level,
        )


if __name__ == "__main__":
    run_server()
