"""Project path utilities for finding project root and managing paths."""

import logging
import os
from functools import lru_cache
from pathlib import Path

# Stdlib logging is safe before logging is configured (records are simply dropped
# or routed to the last-resort handler), so a module logger is fine this early.
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Resolve the project root directory robustly.

    Resolution order:
    1) Environment variable ``ATB_PROJECT_ROOT`` if set and valid
    2) Walk up from current working directory for known markers
    3) Walk up from this module's path for known markers (site-packages installs)
    4) Fallback to current working directory
    """
    # * 1) Explicit override via environment variable
    env_root = os.getenv("ATB_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if candidate.exists():
            return candidate

    # * Common deployment path on Railway/Docker images
    app_dir = Path("/app")
    if app_dir.exists():
        return app_dir.resolve()

    # * Helper: look for common project markers while walking up
    def _search_up(start: Path) -> Path | None:
        markers = ("alembic.ini", "pyproject.toml", "migrations")
        for parent in [start, *start.parents]:
            try:
                if any((parent / m).exists() for m in markers):
                    return parent
            except Exception as exc:
                # ! Filesystem edge cases (permissions, etc.) — startup-only
                # path, so a WARNING cannot flood and surfaces FS problems.
                logger.warning("Skipping %s during project-root search: %s", parent, exc)
                continue
        return None

    # * 2) Try from current working directory
    cwd = Path.cwd().resolve()
    found = _search_up(cwd)
    if found is not None:
        return found

    # * 3) Try from this module's file location (site-packages scenario)
    current = Path(__file__).resolve()
    found = _search_up(current.parent)
    if found is not None:
        return found

    # * 4) Last resort: use CWD
    return cwd


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Get the cached project root, computing it if necessary.

    Uses lru_cache for thread-safe lazy initialization.
    """
    return find_project_root()


def get_model_registry_root() -> Path:
    """Resolve the model registry root directory.

    Honors the ``MODEL_REGISTRY_PATH`` environment variable when set -- the
    same key ``PredictionConfig.model_registry_path`` reads (via
    ConfigManager) for the READ side (``PredictionModelRegistry``/backtest
    bundle lookups). Reusing it here for the WRITE side
    (``TrainingPaths.default()``'s ``models_dir``,
    ``promote_model_version``'s default ``registry_root``) lets a single env
    var redirect both sides consistently -- e.g. for acceptance-test
    hermeticity, so trained/promoted artifacts never touch the real
    ``src/ml/models/`` registry.

    An absolute override replaces the path outright; a relative one
    resolves against ``get_project_root()``. Unset (the common case)
    preserves the existing default: ``{project_root}/src/ml/models``.

    Deliberately NOT cached (unlike ``get_project_root``) -- reading a plain
    env var is cheap, and caching would make it impossible for a
    within-process caller (e.g. a test using ``monkeypatch.setenv``) to
    change the override after the first call.
    """
    override = os.getenv("MODEL_REGISTRY_PATH")
    root = get_project_root()
    if not override:
        return root / "src" / "ml" / "models"
    override_path = Path(override)
    return override_path if override_path.is_absolute() else root / override_path
