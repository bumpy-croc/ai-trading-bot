"""
Railway configuration provider.

This module provides configuration from Railway environment variables.
"""

import os
from typing import Any

from .base import ConfigProvider


class RailwayProvider(ConfigProvider):
    """
    Configuration provider for Railway deployments.

    Railway automatically injects environment variables like:
    - RAILWAY_ENVIRONMENT_NAME (staging/production)
    - RAILWAY_PROJECT_NAME
    - RAILWAY_SERVICE_NAME
    - Plus any custom variables set in Railway dashboard
    """

    def __init__(self):
        """Initialize Railway provider"""
        self._env_vars = {}
        self._load_env_vars()

    @property
    def provider_name(self) -> str:
        """Return the name of this provider"""
        return "Railway"

    def _load_env_vars(self) -> None:
        """Load all environment variables"""
        self._env_vars = dict(os.environ)

    def is_available(self) -> bool:
        """
        Check if running in Railway environment.

        Returns:
            True if Railway environment variables are detected
        """
        # Railway sets these environment variables automatically
        railway_indicators = [
            "RAILWAY_DEPLOYMENT_ID",
            "RAILWAY_PROJECT_ID",
            "RAILWAY_SERVICE_ID",
            "RAILWAY_ENVIRONMENT_ID",
        ]

        return any(key in self._env_vars for key in railway_indicators)

    def get(self, key: str, default: Any | None = None) -> str | None:
        """Get configuration value from Railway environment variables."""
        # Railway-specific environment variable
        railway_key = f"RAILWAY_{key.upper()}"
        value = os.getenv(railway_key)
        if value is not None:
            return value

        # Fallback to regular environment variable
        return os.getenv(key, default)

    def get_all(self) -> dict[str, Any]:
        """Get all Railway environment variables."""
        railway_vars = {}
        for key, value in os.environ.items():
            if key.startswith("RAILWAY_"):
                # Remove RAILWAY_ prefix for consistency
                clean_key = key[8:].lower()
                railway_vars[clean_key] = value
        return railway_vars

    def refresh(self) -> None:
        """Refresh environment variables"""
        self._load_env_vars()

    def get_railway_info(self) -> dict[str, str | None]:
        """
        Get Railway-specific deployment information.

        Returns:
            Dictionary with Railway deployment info
        """
        return {
            "project_id": os.getenv("RAILWAY_PROJECT_ID"),
            "service_id": os.getenv("RAILWAY_SERVICE_ID"),
            "environment": os.getenv("RAILWAY_ENVIRONMENT_NAME"),
            "domain": os.getenv("RAILWAY_DOMAIN"),
        }
