"""
.env file configuration provider
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import ConfigProvider


class DotEnvProvider(ConfigProvider):
    """Provider that reads configuration from .env files"""

    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self._cache: Dict[str, str] = {}
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

    def get(self, key: str) -> Optional[str]:
        """Get a configuration value from .env file"""
        return self._cache.get(key)

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values from .env file"""
        return self._cache.copy()

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
