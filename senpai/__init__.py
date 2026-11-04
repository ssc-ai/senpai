import warnings
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


def _warn_if_installed_via_astro_senpai():
    # astro-senpai is now an empty package that depends on senpai. Wheels cannot
    # print at install time, so tell people at import to depend on senpai.
    try:
        version("senpai")
        version("astro-senpai")
    except PackageNotFoundError:
        return
    warnings.warn(
        "senpai was installed through astro-senpai, its old PyPI name. Depend on "
        "senpai instead: pip install senpai, or uv add senpai && uv remove "
        "astro-senpai.",
        FutureWarning,
        stacklevel=3,
    )


_warn_if_installed_via_astro_senpai()
