"""Project path utilities for finding project root and managing paths."""

from pathlib import Path


def find_project_root() -> Path:
    """Find the project root by looking for alembic.ini or pyproject.toml.

    This function handles both development and production environments:
    - In development: Uses the traditional parents[2] approach
    - In production (installed as package): Walks up directory tree to find project markers

    Returns:
        Path: The project root directory
    """
    current = Path(__file__).resolve()

    # When installed as package, walk up from site-packages to find project root
    for parent in current.parents:
        if (parent / "alembic.ini").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback to the original method for development
    return current.parents[3]  # src/utils -> src -> project_root


# Cache the project root to avoid repeated filesystem operations
_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Get the cached project root, computing it if necessary."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = find_project_root()
    return _PROJECT_ROOT
