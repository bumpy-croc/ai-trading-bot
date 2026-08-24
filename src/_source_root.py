"""Detect and reject "imported one checkout, invoked from another" runs.

Second line of defence behind ``tools/atb_worktree_shim.py`` (GH #1070). The shim makes the
right thing happen automatically; this module makes the wrong thing *loud* when the shim is
absent, disabled, or defeated — because the failure it guards is a silent wrong answer on a
backtest, which is worse than no answer at all.

``src/__init__.py`` calls :func:`verify_source_root` on import, so every entry point that can
run repo code — the ``atb`` console script, ``pytest``, ``python experiments/x.py``, an ad-hoc
``python -c`` — is covered without opting in. Nothing else needs to call it.

This module lives at the top level of ``src`` and imports nothing from the repo, so the guard
runs before any other repo module executes. Putting it under ``src.utils`` would drag that
package's ``__init__`` (and its siblings) in first — code from the very checkout whose identity
is still in question.

The root-finding logic is intentionally duplicated from the shim rather than shared: the shim
must run before any repo code is importable. ``tests/unit/test_source_root_guard.py`` asserts
the two implementations agree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "SourceRootMismatchError",
    "find_repo_root",
    "invocation_root",
    "source_root",
    "verify_source_root",
]

PROJECT_MARKER = 'name = "ai-trading-bot"'
REQUIRED_PACKAGES = ("src", "cli")
ALLOW_MISMATCH_ENV = "ATB_ALLOW_SOURCE_ROOT_MISMATCH"
SHIM_ROOT_ENV = "ATB_SHIM_REPO_ROOT"


class SourceRootMismatchError(RuntimeError):
    """Raised when imported source and invocation directory come from different checkouts."""


def _looks_like_repo_root(candidate: Path) -> bool:
    pyproject = candidate / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        if PROJECT_MARKER not in pyproject.read_text(encoding="utf-8", errors="replace"):
            return False
    except OSError:
        return False
    return all((candidate / pkg / "__init__.py").is_file() for pkg in REQUIRED_PACKAGES)


def find_repo_root(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd) to the enclosing ai-trading-bot checkout root.

    The walk stops at the first directory holding a ``.git`` entry even when that directory
    does not qualify as a checkout root. Worktrees live *inside* the primary checkout
    (``<primary>/.claude/worktrees/<name>``), so a worktree that momentarily fails the
    structural test — mid-rebase, a missing ``cli/__init__.py`` — would otherwise let the walk
    climb straight into the primary checkout and silently reproduce the very bug this guards.
    """
    try:
        current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    except OSError:
        return None
    for candidate in (current, *current.parents):
        if _looks_like_repo_root(candidate):
            return candidate
        if (candidate / ".git").exists():
            return None
    return None


def source_root() -> Path:
    """Repo root the currently-imported ``src`` package was loaded from."""
    return Path(__file__).resolve().parents[1]


def invocation_root() -> Path | None:
    """Repo root enclosing the current working directory, if any."""
    return find_repo_root()


def _shim_status() -> str:
    chosen = os.environ.get(SHIM_ROOT_ENV)
    if chosen:
        return f"  worktree shim   : active, chose {chosen}\n"
    return (
        "  worktree shim   : NOT ACTIVE — this is almost certainly why you are seeing this.\n"
        "                    It lives in site-packages, which no venv rebuild restores.\n"
    )


def _mismatch_message(imported: Path, invoked: Path) -> str:
    return (
        "\n"
        "==================== ATB SOURCE ROOT MISMATCH (GH #1070) ====================\n"
        f"  imported code from : {imported}\n"
        f"  invoked from       : {invoked}\n"
        f"{_shim_status()}"
        "\n"
        "Python resolved `src`/`cli` to a DIFFERENT checkout than the one you are\n"
        "working in. This happens when the shared venv's editable install still points\n"
        "at the checkout it was created from. Any result produced this way is invalid:\n"
        "it reflects the other checkout's branch, not yours.\n"
        "\n"
        "Fix (installs the import shim into this venv; instant, no rebuild):\n"
        "    python tools/install_worktree_shim.py\n"
        "\n"
        "One-off escape hatch for a single command:\n"
        '    PYTHONPATH="$(pwd)" <your command>\n'
        "\n"
        f"To proceed anyway (you almost never want this): {ALLOW_MISMATCH_ENV}=1\n"
        "============================================================================\n"
    )


def verify_source_root() -> Path:
    """Raise when the imported source root differs from the invoking checkout.

    Returns the imported source root on success. A no-op when there is nothing meaningful to
    compare: the working directory is outside any checkout, or ``src`` came from a regular
    (non-editable) site-packages install, which is how production containers run.

    Set ``ATB_ALLOW_SOURCE_ROOT_MISMATCH=1`` to downgrade the failure to a stderr warning.
    """
    imported = source_root()
    invoked = invocation_root()
    if invoked is None or invoked == imported:
        return imported
    if not _looks_like_repo_root(imported):
        # `src` was installed into site-packages rather than linked to a checkout; the
        # source-vs-cwd comparison is meaningless (and would break prod containers).
        return imported

    message = _mismatch_message(imported, invoked)
    if os.environ.get(ALLOW_MISMATCH_ENV) == "1":
        print(message, file=sys.stderr)
        return imported
    raise SourceRootMismatchError(message)
