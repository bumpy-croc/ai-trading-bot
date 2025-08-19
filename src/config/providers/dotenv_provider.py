"""
Dotenv configuration provider.

This module provides configuration from .env files.
"""

import os
from pathlib import Path
from typing import Any, Optional

from .base import ConfigProvider


class DotEnvProvider(ConfigProvider):
    """Provider that reads configuration from .env files"""

    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self._cache: dict[str, str] = {}
        self._loaded = False
        self._load_env_file()

    def _load_env_file(self) -> None:
        """Load and parse .env file"""
        if not self.env_file.exists():
            return

        try:
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Remove quotes if present
                        value = value.strip().strip('"').strip("'")
                        self._cache[key.strip()] = value
            self._loaded = True
        except Exception as e:
            print(f"Warning: Failed to load {self.env_file}: {e}")

    def get(self, key: str, default: Optional[Any] = None) -> Optional[str]:
        """Get configuration value from .env file."""
        return self._cache.get(key, default)

    def get_all(self) -> dict[str, Any]:
        """Get all environment variables."""
        return dict(os.environ)

    def is_available(self) -> bool:
        """Check if .env file exists and was loaded"""
        return self._loaded and bool(self._cache)

    @property
    def provider_name(self) -> str:
        return f".env file ({self.env_file})"

    def refresh(self) -> None:
        """Reload the .env file"""
        self._cache.clear()
        self._loaded = False
        self._load_env_file()
