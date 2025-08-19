"""
Environment variable configuration provider.

This module provides configuration from environment variables.
"""

import os
from typing import Any, Optional

from .base import ConfigProvider


class EnvVarProvider(ConfigProvider):
    """Provider that reads configuration from environment variables"""

    def __init__(self):
        self._prefix = ""  # Optional prefix for env vars

    def get(self, key: str, default: Optional[Any] = None) -> Optional[str]:
        """Get configuration value from environment variables."""
        return os.getenv(key, default)

    def get_all(self) -> dict[str, Any]:
        """Get all environment variables."""
        return dict(os.environ)

    def is_available(self) -> bool:
        """Environment variables are always available"""
        return True

    @property
    def provider_name(self) -> str:
        return "Environment Variables"

    def refresh(self) -> None:
        """No-op for environment variables; values are read directly from os.environ"""
        return
