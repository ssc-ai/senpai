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
