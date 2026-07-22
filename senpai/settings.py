"""Facade over :mod:`senpai.core.config` exposing the lazy ``settings`` proxy.

The configuration state (models, singleton, ``initialize_config``) lives in
``senpai.core.config``; this module provides the ``from senpai.settings import settings``
access style used by the ported engine modules. Both entry points share the same
process-global instance.
"""

from typing import TYPE_CHECKING, Any

from senpai.core import config as _config_module
from senpai.core.config import (  # noqa: F401  # re-exported public API
    AppConfig,
    AstrometryConfig,
    DetectionConfig,
    LoggingConfig,
    ObservationsConfig,
    PlottingConfig,
    StarCatalogConfig,
    StreakDetectionConfig,
    get_config,
    get_or_initialize_config,
    initialize_config,
)
from senpai.exceptions import ConfigError

AppSettings = AppConfig


class _SettingsProxy:
    """Lazy proxy that forwards attribute access to the global settings instance.

    Allows modules to import ``settings`` at import time while deferring access to
    the underlying :class:`AppConfig` until ``initialize_config`` has populated it.
    """

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401  # proxies arbitrary AppConfig attributes
        """Return the named attribute from the initialized settings instance.

        Args:
            name (str): attribute name to look up on the underlying settings.

        Returns:
            Any: the corresponding attribute value from the global :class:`AppConfig`.

        Raises:
            ConfigError: if the global config has not been initialized yet.
        """
        instance = _config_module._config_instance
        if instance is None:
            raise ConfigError("Config not initialized. Call initialize_config() first.")
        return getattr(instance, name)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401  # proxies arbitrary AppConfig attributes
        """Set the named attribute on the initialized settings instance.

        Args:
            name (str): attribute name to set on the underlying settings.
            value (Any): value to assign.

        Raises:
            ConfigError: if the global config has not been initialized yet.
        """
        instance = _config_module._config_instance
        if instance is None:
            raise ConfigError("Config not initialized. Call initialize_config() first.")
        setattr(instance, name, value)


if TYPE_CHECKING:  # pragma: no cover
    settings: AppConfig
else:
    settings = _SettingsProxy()
