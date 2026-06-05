import logging

# from engine.utils.astropy import setup_astropy
# from engine.utils.mpl import setup_matplotlib

logger = logging.getLogger(__name__)


def initialize_engine():
    """Initialize all required engine components."""

    logger.info("Initializing engine...")

    # Set up matplotlib first (affects matplotlib imports)
    # setup_matplotlib()

    # Set up Astropy configuration
    # setup_astropy()

    logger.info("Engine initialized.")


# Initialize everything when engine is imported
initialize_engine()
