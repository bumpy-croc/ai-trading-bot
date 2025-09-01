"""Database package.

Provides backward-compatibility aliases so tests that patch `database.*` still
affect the `src.database.*` modules.
"""

# * Back-compat module aliases for tests that patch `database.*`
try:  # pragma: no cover - defensive shim
    import sys as _sys
    from importlib import import_module as _import_module

    _src_db_pkg = _import_module("src.database")
    _sys.modules.setdefault("database", _src_db_pkg)
    _sys.modules.setdefault("database.manager", _import_module("src.database.manager"))
    _sys.modules.setdefault("database.models", _import_module("src.database.models"))
except Exception:  # pragma: no cover - best-effort aliasing
    pass

__all__ = [
    # Re-exports for convenience
]
