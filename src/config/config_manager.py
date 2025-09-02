"""
Configuration management system.

This module provides centralized configuration management.
"""

import threading
from typing import Any, Optional

from .providers.base import ConfigProvider
from .providers.dotenv_provider import DotEnvProvider
from .providers.env_provider import EnvVarProvider
from .providers.railway_provider import RailwayProvider


class ConfigManager:
    """
    Manages configuration from multiple sources with fallback support.

    Default priority order:
    1. Railway environment variables (if available)
    2. Environment variables
    3. .env file
    """

    def __init__(self, providers: Optional[list[ConfigProvider]] = None):
        """
        Initialize ConfigManager with providers.

        Args:
            providers: List of configuration providers in priority order.
                      If None, uses default providers.
        """
        if providers is None:
            # Default provider chain - Railway provider first for Railway deployments
            self.providers = [RailwayProvider(), EnvVarProvider(), DotEnvProvider()]
        else:
            self.providers = providers

        # Log available providers
        available_providers = [p for p in self.providers if p.is_available()]
        if available_providers:
            print(
                f"Configuration providers available: {[p.provider_name for p in available_providers]}"
            )
        else:
            print("Warning: No configuration providers available!")

    def get(self, key: str, default: Optional[Any] = None) -> Optional[str]:
        """
        Get a configuration value from the first available provider.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found in any provider

        Returns:
            Configuration value or default
        """
        for provider in self.providers:
            if provider.is_available():
                value = provider.get(key)
                if value is not None:
                    return value

        return default

    def get_required(self, key: str) -> str:
        """
        Get a required configuration value. Raises exception if not found.

        Args:
            key: Configuration key to retrieve

        Returns:
            Configuration value

        Raises:
            ValueError: If configuration key is not found
        """
        value = self.get(key)
        if value is None:
            available = [p.provider_name for p in self.providers if p.is_available()]
            raise ValueError(
                f"Required configuration '{key}' not found. Checked providers: {available}"
            )
        return value

    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration value as integer"""
        value = self.get(key, str(default))
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration value as float"""
        value = self.get(key, str(default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration value as boolean"""
        value = self.get(key)
        if value is None:
            return default

        # Handle common boolean representations
        return value.lower() in ("true", "1", "yes", "on", "enabled")

    def get_list(
        self, key: str, delimiter: str = ",", default: Optional[list[str]] = None
    ) -> list[str]:
        """Get configuration value as list"""
        value = self.get(key)
        if value is None:
            return default or []

        return [item.strip() for item in value.split(delimiter) if item.strip()]

    def get_all(self) -> dict[str, Any]:
        """
        Get all configuration values from all providers.
        Later providers override earlier ones.
        """
        all_config = {}
        for provider in reversed(self.providers):
            if provider.is_available():
                all_config.update(provider.get_all())
        return all_config

    def refresh(self) -> None:
        """Refresh all providers that support refreshing"""
        for provider in self.providers:
            if provider.is_available():
                provider.refresh()

    def add_provider(self, provider: ConfigProvider, priority: int = 0) -> None:
        """
        Add a new provider at the specified priority.

        Args:
            provider: Configuration provider to add
            priority: Position in provider list (0 = highest priority)
        """
        self.providers.insert(priority, provider)

    def remove_provider(self, provider_name: str) -> bool:
        """
        Remove a provider by name.

        Args:
            provider_name: Name of the provider to remove

        Returns:
            True if provider was removed, False if not found
        """
        for i, provider in enumerate(self.providers):
            if provider.provider_name == provider_name:
                self.providers.pop(i)
                return True
        return False

    def get_all_config(self) -> dict[str, Any]:
        """Get all configuration values."""
        config = {}
        for key in self._get_all_keys():
            config[key] = self.get(key)
        return config

    def _get_all_keys(self) -> list[str]:
        """Get all configuration keys."""
        keys = set()
        for provider in self.providers:
            if hasattr(provider, "get_all_keys"):
                keys.update(provider.get_all_keys())
        return list(keys)

    def get_config_sources(self) -> list[str]:
        """Get list of configuration sources."""
        return [provider.__class__.__name__ for provider in self.providers]

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary."""
        return {
            "sources": self.get_config_sources(),
            "total_keys": len(self._get_all_keys()),
            "values": self.get_all_config(),
        }


# Global configuration instance
_config_instance: Optional[ConfigManager] = None
_config_lock: Optional[threading.Lock] = None


def get_config() -> ConfigManager:
    """
    Get the global configuration instance.

    Returns:
        Global ConfigManager instance
    """
    global _config_instance, _config_lock

    # Initialize lock if not already done (thread-safe)
    if _config_lock is None:
        _config_lock = threading.Lock()

    # Thread-safe singleton creation
    if _config_instance is None:
        with _config_lock:
            # Double-check pattern to avoid race condition
            if _config_instance is None:
                _config_instance = ConfigManager()

    return _config_instance


def set_config(config: ConfigManager) -> None:
    """
    Set the global configuration instance.

    Args:
        config: ConfigManager instance to use globally
    """
    global _config_instance
    _config_instance = config
