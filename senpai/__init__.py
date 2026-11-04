from importlib.metadata import PackageNotFoundError, version

from senpai.core.logging import setup_logging

setup_logging()


try:
    __version__ = version("senpai")
except PackageNotFoundError:
    try:
        # astro-senpai <= 2.9.0 shipped the code under its own name
        __version__ = version("astro-senpai")
    except PackageNotFoundError:
        __version__ = "unknown"  # Fallback if the package is not installed
