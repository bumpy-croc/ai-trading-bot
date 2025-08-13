"""Database Manager Web Application Package

Provides a simple Flask-based admin interface for interacting with the trading bot database.
"""

from .app import create_app  # noqa: F401

__all__ = ["create_app"]
