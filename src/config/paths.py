"""
Path utilities for the AI Trading Bot project.

This module provides utilities to resolve paths relative to the project structure,
regardless of whether the code is running from the project root or from within
subdirectories.
"""

from pathlib import Path

from src.utils.project_paths import get_project_root as get_project_root_utils


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path: The project root directory
    """
    return get_project_root_utils()


def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path: The data directory path
    """
    return get_project_root() / "src" / "data"


def get_cache_dir() -> Path:
    """
    Get the cache directory path.

    Returns:
        Path: The cache directory path
    """
    return get_project_root() / "cache" / "market_data"


def get_database_path(*_args, **_kwargs):  # type: ignore[override]
    """DEPRECATED: SQLite is no longer supported.

    This helper previously returned an SQLite connection string.  The project
    now requires PostgreSQL exclusively, so this function raises a
    ``RuntimeError`` if called.
    """
    raise RuntimeError(
        "SQLite has been removed from the project.  Replace calls to"
        " `get_database_path()` with environment-based PostgreSQL URLs."
    )


def get_sentiment_data_path() -> Path:
    """
    Get the sentiment data CSV path.

    Returns:
        Path: The sentiment data CSV path
    """
    return get_data_dir() / "senticrypt_sentiment_data.csv"


def resolve_data_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the data directory.

    Args:
        relative_path: Path relative to the data directory

    Returns:
        Path: Absolute path to the file
    """
    return get_data_dir() / relative_path


def ensure_dir_exists(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path: The directory path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
