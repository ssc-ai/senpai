"""Where SENPAI keeps its own files, honouring the platform's conventions.

Cache and config locations follow the usual environment variables when they are set, so a
containerised or multi-user install can redirect them without code changes.
"""

import os
from pathlib import Path

# Package root: `senpai/`, the only tree a wheel ships. Everything that has to
# resolve after a normal (non-editable) install is anchored here, so paths are
# identical in a source checkout and in site-packages.
PACKAGE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PACKAGE_DIR / "resources"

# Repo root. Valid only in a source checkout -- a wheel installs `senpai/`
# alone, so this points into site-packages once installed. Never anchor
# shipped code here; test fixtures and dev tooling only.
REPO_ROOT = PACKAGE_DIR.parent
BASE_DIR = REPO_ROOT  # Deprecated alias, kept for out-of-tree imports.

TEST_DATA_DIR = REPO_ROOT / "tests" / "data"

# Resource directories (read-only, shipped with the package)
ASSETS_DIR = RESOURCES_DIR / "assets"
DATA_DIR = RESOURCES_DIR / "data"
CONFIG_DIR = RESOURCES_DIR / "config"

# Config overrides
LOCAL_APP_CONFIG_OVERRIDE = CONFIG_DIR / "local.yaml"
LOCAL_APP_LOCAL_ASTROMETRY_CONFIG_OVERRIDE = CONFIG_DIR / "local-localastrometry.yaml"
DEV_APP_CONFIG_OVERRIDE = CONFIG_DIR / "dev.yaml"
PROD_APP_CONFIG_OVERRIDE = CONFIG_DIR / "prod.yaml"
CI_PIPELINE_CONFIG_PATH = CONFIG_DIR / "ci_pipeline_config.yaml"

# App-specific paths
APP_DIR = PACKAGE_DIR / "api"
APP_CONFIG_PATH = CONFIG_DIR / "application.yaml"


def _user_dir(env_var: str, default: Path) -> Path:
    """Resolve a writable directory, preferring an explicit environment override."""
    value = os.getenv(env_var)
    return Path(value).expanduser() if value else default


# Writable directories. These deliberately live outside the install tree: an
# installed package may sit on a read-only filesystem (containers, system-wide
# installs), and writing cache or log files into site-packages is wrong even
# when it happens to be permitted. Directories are created by the code that
# writes to them, not at import time.
def _default_cache_dir() -> Path:
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    root = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    return root / "senpai"


CACHE_DIR = _user_dir("SENPAI_CACHE_DIR", _default_cache_dir())
LOG_DIR = _user_dir("SENPAI_LOG_DIR", CACHE_DIR / "logs")
LOG_PATH = LOG_DIR / "app.log"
