"""SENPAI: physics-based satellite detection and WCS/astrometry engine.

Distributed on PyPI as ``astro-senpai`` and imported as ``senpai``. Importing the
package configures logging; ``__version__`` reports the installed distribution version.
"""

from importlib.metadata import PackageNotFoundError, version

from senpai.core.logging import setup_logging

setup_logging()


try:
    # Distribution is named astro-senpai (import package stays senpai)
    __version__ = version("astro-senpai")
except PackageNotFoundError:
    try:
        __version__ = version("senpai")  # pre-rename installs
    except PackageNotFoundError:
        __version__ = "unknown"  # Fallback if the package is not installed
