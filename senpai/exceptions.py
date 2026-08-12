"""Typed exceptions for explicit, unrecoverable SENPAI failures.

Recoverable conditions (a single mid-chain shift that can be routed around) are logged as
warnings and recorded as frame-shift status, never raised. Genuinely unrecoverable failures
raise one of the exceptions below, which propagate uncaught to the outermost boundary (the API
exception handlers or the CLI), where they are logged with a stack trace and mapped to a
response. Each class carries the HTTP ``status_code`` the API boundary should use.
"""


class SenpaiError(Exception):
    """Base for explicit, unrecoverable SENPAI failures.

    Attributes:
        status_code: HTTP status the API boundary maps this failure to.
    """

    status_code: int = 500


class InvalidInputError(SenpaiError):
    """The submitted frames are malformed or missing required data.

    Examples: an unreadable FITS file, or a frame with no usable observation-time header.
    """

    status_code = 422


class SiderealSolveError(SenpaiError):
    """No valid sidereal WCS solution could be found, so the collect cannot be anchored."""

    status_code = 500


class WcsPropagationError(SenpaiError):
    """The sidereal WCS could not be propagated to the rate-tracked frames.

    Raised when the sidereal->rate anchor shift cannot be measured (so there is nothing to
    propagate), or when, after the analysis chain, no rate-tracked frame ended up registered to
    a WCS solution.
    """

    status_code = 500


class ConfigError(SenpaiError):
    """Configuration is missing or invalid (e.g. the config was never initialized)."""

    status_code = 500


class MissingDependencyError(SenpaiError):
    """A required external binary (e.g. astrometry.net's ``image2xy``) is not installed.

    Raised when SENPAI shells out to a command-line tool that is absent from ``PATH``. Without it
    the affected step (e.g. astrometry source extraction / WCS refinement) cannot run at all, so
    every solve would fail; surfacing it explicitly turns an opaque ``FileNotFoundError`` into an
    actionable "install this dependency" message instead of a silent 0% solve rate.
    """

    status_code = 500


class ProcessingTimeoutError(SenpaiError):
    """Detection exceeded the configured per-request timeout."""

    status_code = 504


class ExternalServiceError(SenpaiError):
    """A required external service returned an error response."""

    status_code = 502
