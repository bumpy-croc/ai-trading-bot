"""Interpreter-level shim that pins ``src``/``cli`` imports to the invoking checkout.

Installed into the venv's ``site-packages`` (see ``tools/install_worktree_shim.py``) and
executed on every interpreter start via a ``.pth`` file.

Why this exists (GH #1070): ``pip install -e .`` generates a ``_EditableFinder`` on
``sys.meta_path`` whose MAPPING hardcodes the *absolute path of the checkout the install was
run from*. ``sys.meta_path`` is consulted after ``PathFinder``, so the mapping only loses when
some ``sys.path`` entry already contains ``src``/``cli``. That is true for ``python -c`` and
``python <repo-root>/x.py`` (cwd or script dir is the repo root) but FALSE for console scripts
(``sys.path[0]`` is ``.venv/bin``) and for ``python experiments/x.py`` (``sys.path[0]`` is
``experiments/``). Those invocations silently import the install-time checkout even when run
from a different git worktree on a different branch.

This shim front-runs that mapping: it resolves the repo root containing the *current working
directory* and installs a meta-path finder ahead of everything else that binds top-level
``src`` and ``cli`` to that root. Descendant modules follow automatically via the parent
package's ``__path__``.

Design constraints:
- stdlib only, no imports from the repo (it must run before any repo code exists on the path);
- never raise at interpreter start — a broken shim must not brick every python invocation;
- a no-op when the cwd is not inside an ai-trading-bot checkout, or when the resolved root
  already matches what the editable install would pick (the normal single-checkout case).

Escape hatch: set ``ATB_DISABLE_WORKTREE_SHIM=1`` to skip installation entirely.
"""

from __future__ import annotations

import os
import sys
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path

__all__ = ["find_repo_root", "install", "REDIRECTED_PACKAGES"]

REDIRECTED_PACKAGES = ("src", "cli")

_PROJECT_MARKER = 'name = "ai-trading-bot"'
_DISABLE_ENV = "ATB_DISABLE_WORKTREE_SHIM"

# Set by install() so downstream guards can report what the shim decided.
ATB_SHIM_ROOT_ENV = "ATB_SHIM_REPO_ROOT"


def _looks_like_repo_root(candidate: Path) -> bool:
    """True when ``candidate`` is an ai-trading-bot checkout root.

    Requires both the project marker and the importable packages so that an unrelated
    directory tree can never capture ``src``/``cli``.
    """
    pyproject = candidate / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        if _PROJECT_MARKER not in pyproject.read_text(encoding="utf-8", errors="replace"):
            return False
    except OSError:
        return False
    return all((candidate / pkg / "__init__.py").is_file() for pkg in REDIRECTED_PACKAGES)


def find_repo_root(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd) to the enclosing ai-trading-bot checkout root."""
    try:
        current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    except OSError:
        return None
    for candidate in (current, *current.parents):
        if _looks_like_repo_root(candidate):
            return candidate
    return None


class _InvocationRootFinder:
    """Meta-path finder binding top-level ``src``/``cli`` to a fixed repo root.

    Only top-level names are handled: once ``src`` is bound, ``src.__path__`` points into the
    correct root and importlib resolves every descendant from there.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def find_spec(self, fullname: str, path=None, target=None) -> ModuleSpec | None:
        if fullname not in REDIRECTED_PACKAGES:
            return None
        return PathFinder.find_spec(fullname, path=[str(self.root)])

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} root={self.root}>"


def _already_installed() -> bool:
    return any(isinstance(finder, _InvocationRootFinder) for finder in sys.meta_path)


def install(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Install the finder for the checkout enclosing ``start``. Returns the root, or None."""
    if os.environ.get(_DISABLE_ENV) == "1":
        return None
    if _already_installed():
        return None
    root = find_repo_root(start)
    if root is None:
        return None
    sys.meta_path.insert(0, _InvocationRootFinder(root))
    # Exported so subprocesses and the atb guard can see which root the shim chose.
    os.environ[ATB_SHIM_ROOT_ENV] = str(root)
    return root


def install_quietly() -> None:
    """Entry point invoked by the ``.pth`` file. Swallows everything by design."""
    try:
        install()
    except Exception:  # pragma: no cover - defensive; a shim must never break the interpreter
        pass
