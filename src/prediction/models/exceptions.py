"""Exceptions for model loading and registry operations."""

from __future__ import annotations


class ModelLoadError(Exception):
    """Raised when a model or its metadata fails to load or validate."""


class ModelNotAvailableError(Exception):
    """Raised when a requested model bundle is not available for selection."""


