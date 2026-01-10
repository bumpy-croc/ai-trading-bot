"""
Base configuration provider.

This module defines the base interface for configuration providers.
"""

from abc import ABC, abstractmethod
from typing import Any


class ConfigProvider(ABC):
    """Abstract base class for configuration providers"""

    @abstractmethod
    def get(self, key: str, default: Any | None = None) -> Any | None:
        """Get configuration value."""
        pass

    def get_all(self) -> dict[str, Any]:
        """Get all configuration values."""
        return {}

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available/configured"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider"""
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Refresh cached values (optional implementation)"""
        raise NotImplementedError
