import logging
import logging.config
import logging.handlers
from enum import Enum
from typing import Any, Dict, Literal

from senpai.core.constants import LOG_PATH

logger = logging.getLogger(__name__)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class MultiprocessSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler tolerant of concurrent rollovers.

    The burr night pipeline runs many worker processes that all log to the
    same file. With size-based rotation they race in doRollover: once one
    process renames app.log -> app.log.1, the others' os.rename of the
    now-missing app.log raises FileNotFoundError, which logging prints as a
    spurious traceback (the record is dropped but processing is unaffected).
    Swallow that specific race and just reopen the (recreated) base file so
    the loser keeps logging to the fresh app.log.
    """

    def rotate(self, source: str, dest: str) -> None:
        try:
            super().rotate(source, dest)
        except FileNotFoundError:
            # Another process already rotated this file out from under us.
            pass


class LogMode(str, Enum):
    LOCAL = "local"


def get_local_config() -> Dict[str, Any]:
    """Configuration for local development with timestamps and color"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "log_colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            },
            "file": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"},
            "uvicorn": {
                "()": "uvicorn.logging.DefaultFormatter",
                "format": "%(levelprefix)s %(name)s - %(message)s",
                "use_colors": True,
            },
            "uvicorn.access": {
                "()": "uvicorn.logging.AccessFormatter",
                "format": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "use_colors": True,
            },
        },
        "handlers": {
            "console": {"class": "colorlog.StreamHandler", "formatter": "colored", "stream": "ext://sys.stdout"},
            "file": {
                "class": "senpai.core.logging.MultiprocessSafeRotatingFileHandler",
                "formatter": "file",
                "filename": str(LOG_PATH),
                "maxBytes": 10485760,
                "backupCount": 5,
            },
            "uvicorn": {"class": "logging.StreamHandler", "formatter": "uvicorn", "stream": "ext://sys.stdout"},
            "uvicorn.access": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn.access",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn": {"handlers": ["uvicorn", "file"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["uvicorn.access", "file"], "level": "INFO", "propagate": False},
        },
    }


def setup_logging(level: str = "INFO", disabled_loggers: list[str] = None) -> None:
    """Configure logging based on level and context

    Args:
        level: The default logging level for all loggers
        disabled_loggers: List of logger names to disable or set to a higher level
    """
    # Check if logging is already configured with our settings
    root_logger = logging.getLogger()
    if root_logger.handlers and hasattr(root_logger, "senpai_logging_configured"):
        logger.info("Logging already configured, skipping setup")
        return

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = get_local_config()

    # Update log levels
    for logger_ in config["loggers"].values():
        logger_["level"] = level.upper()

    # Disable specific loggers if requested
    if disabled_loggers:
        for logger_name in disabled_loggers:
            # If the logger is already in the config, set it to WARNING or higher
            if logger_name in config["loggers"]:
                config["loggers"][logger_name]["level"] = "WARNING"
            # Otherwise add a new logger config
            else:
                config["loggers"][logger_name] = {
                    "handlers": ["console", "file"],
                    "level": "WARNING",
                    "propagate": False,
                }

            # Also directly set the level for any already-created loggers
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.config.dictConfig(config)

    # Mark logging as configured
    root_logger.senpai_logging_configured = True
    logger.info(f"Completed logging setup with level `{level}`")


def set_log_level(level: LogLevel) -> None:
    """Change the log level of all configured loggers.

    Args:
        level: The new logging level to set
    """
    root_logger = logging.getLogger()
    if not hasattr(root_logger, "senpai_logging_configured"):
        logger.warning("Logging not configured with setup_logging(), setting level may have limited effect")

    # Convert string to uppercase for consistency
    level_upper = level.upper()

    # Set level for root logger
    root_logger.setLevel(level_upper)

    # Set level for all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level_upper)

    logger.info(f"Changed logging level to `{level_upper}`")
