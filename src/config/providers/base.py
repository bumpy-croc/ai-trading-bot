"""
Base configuration provider interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ConfigProvider(ABC):
    """Abstract base class for configuration providers"""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get a configuration value by key"""
        pass

    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        pass

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
