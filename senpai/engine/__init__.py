"""SENPAI processing engine: detection, astrometry/WCS, photometry, and pipeline.

Importing this package initializes engine-wide components (currently logging) via
:func:`initialize_engine`.
"""

import logging

logger = logging.getLogger(__name__)


def initialize_engine() -> None:
    """Initialize all required engine components."""
    logger.info("Initializing engine...")

    # Set up matplotlib first (affects matplotlib imports)
    # setup_matplotlib()

    # Set up Astropy configuration
    # setup_astropy()

    logger.info("Engine initialized.")


# Initialize everything when engine is imported
initialize_engine()
