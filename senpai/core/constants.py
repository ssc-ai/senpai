from pathlib import Path

# Base paths (read-only)
BASE_DIR = base_dir = Path(__file__).parent.parent.parent  # repo root
RESOURCES_DIR = BASE_DIR / "resources"

TEST_DATA_DIR = BASE_DIR  / "tests" / "data"

# Resource directories (read-only)
ASSETS_DIR = RESOURCES_DIR / "assets"
DATA_DIR = RESOURCES_DIR / "data"
CONFIG_DIR = RESOURCES_DIR / "config"

# Cache directory (writable)
CACHE_DIR = BASE_DIR / "cache"

# Config overrides
LOCAL_APP_CONFIG_OVERRIDE = CONFIG_DIR / "local.yaml"
LOCAL_APP_LOCAL_ASTROMETRY_CONFIG_OVERRIDE = CONFIG_DIR / "local-localastrometry.yaml"
DEV_APP_CONFIG_OVERRIDE = CONFIG_DIR / "dev.yaml"
PROD_APP_CONFIG_OVERRIDE = CONFIG_DIR / "prod.yaml"
CI_PIPELINE_CONFIG_PATH = CONFIG_DIR / "ci_pipeline_config.yaml"
# App-specific paths
APP_DIR = BASE_DIR / "senpai" / "api"
APP_CONFIG_PATH = CONFIG_DIR / "application.yaml"
LOG_PATH = APP_DIR / "logs" / "app.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
