"""Logging configuration for the application.

Provides the local logging dictConfig, a multiprocess-safe rotating file handler,
and helpers to set up and adjust log levels at runtime.
"""

import contextlib
import logging
import logging.config
import logging.handlers
from enum import StrEnum
from typing import Any, Literal

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
        """Rotate the log file, tolerating a concurrent rollover by another process.

        Args:
            source: Path of the current log file being rotated out.
            dest: Path the current log file is renamed to.
        """
        # Another process already rotating this file out from under us is fine.
        with contextlib.suppress(FileNotFoundError):
            super().rotate(source, dest)


class LogMode(StrEnum):
    """Supported logging configuration modes."""

    LOCAL = "local"


def get_local_config() -> dict[str, Any]:
    """Build the logging dictConfig for local development with timestamps and color.

    Returns:
        A logging configuration dictionary suitable for
        :func:`logging.config.dictConfig`.
    """
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


def setup_logging(level: str = "INFO", disabled_loggers: list[str] | None = None) -> None:
    """Configure logging based on level and context.

    Args:
        level: The default logging level for all loggers.
        disabled_loggers: List of logger names to disable or set to a higher level.
    """
    # Check if logging is already configured with our settings
    root_logger = logging.getLogger()
    if root_logger.handlers and hasattr(root_logger, "senpai_logging_configured"):
        logger.info("Logging already configured, skipping setup")
        return

    config = get_local_config()

    # Create the log directory lazily here (not at import). If the location is
    # unwritable — an installed wheel on a read-only root, or a locked-down
    # site-packages — degrade to console-only rather than failing `import senpai`.
    # The file handler is dropped from the config and from every logger's handler
    # list so dictConfig doesn't try to open a file we can't create.
    default_handlers = ["console", "file"]
    log_dir_error: str | None = None
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log_dir_error = str(exc)
        default_handlers = ["console"]
        config["handlers"].pop("file", None)
        for logger_ in config["loggers"].values():
            logger_["handlers"] = [h for h in logger_["handlers"] if h != "file"]

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
                    "handlers": list(default_handlers),
                    "level": "WARNING",
                    "propagate": False,
                }

            # Also directly set the level for any already-created loggers
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.config.dictConfig(config)

    # Mark logging as configured
    root_logger.senpai_logging_configured = True
    logger.info(f"Completed logging setup with level `{level}`")
    if log_dir_error is not None:
        # Emitted after dictConfig so it reaches the (console) handler now in place.
        logger.warning(
            "File logging disabled: could not create log directory %s (%s). "
            "Logging to console only.",
            LOG_PATH.parent,
            log_dir_error,
        )


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
