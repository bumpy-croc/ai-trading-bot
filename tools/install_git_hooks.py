"""Install this repo's tracked git hooks (``.githooks/``) into the active hooks directory.

The hooks used to live only in ``.git/hooks``: untracked, unreviewable, and silently different
on every machine — which is how the inert pre-push hook of GH #1077 survived unnoticed. The
hook source is now version-controlled; this installer puts it where git will run it.

Two properties matter more than elegance here, both because getting them wrong recreates
#1077 — a safeguard that silently stops safeguarding:

* **Hooks are copied, never symlinked.** ``$GIT_COMMON_DIR/hooks`` is shared by every linked
  worktree, but ``make install`` runs inside ephemeral agent worktrees. A symlink into one of
  those dangles the moment the worktree is pruned or its branch stops shipping ``.githooks/``,
  and git skips a dangling hook *silently*, exiting 0. A copy cannot dangle.
* **The source is read from the primary checkout** (the parent of the git common dir) when it
  has one, so a feature branch in a worktree cannot seed the repo-wide hook.

Drift is the price of copying, so ``--check`` compares content and is wired to ``make
hooks-check``; ``make install`` re-copies on every run.

Usage:
    python tools/install_git_hooks.py           # install / repair
    python tools/install_git_hooks.py --check   # report drift, exit 1 if out of date
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SOURCE_DIR_NAME = ".githooks"
HOOK_MODE = 0o755


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repo_root(start: Path | None = None) -> Path:
    """Root of the checkout being invoked from (a linked worktree, possibly)."""
    return Path(_git("rev-parse", "--show-toplevel", cwd=start)).resolve()


def _common_dir(root: Path) -> Path:
    common = Path(_git("rev-parse", "--git-common-dir", cwd=root))
    if not common.is_absolute():
        common = root / common
    return common.resolve()


def primary_root(root: Path) -> Path:
    """Root of the checkout that owns the shared git dir — the same rule the hook uses."""
    return _common_dir(root).parent


def hooks_dir(root: Path) -> Path:
    """Directory git actually runs hooks from: ``core.hooksPath`` if set, else the common dir."""
    try:
        configured = _git("config", "--get", "core.hooksPath", cwd=root)
    except subprocess.CalledProcessError:
        configured = ""
    if configured:
        path = Path(os.path.expanduser(configured))
        return path if path.is_absolute() else (root / path)
    return _common_dir(root) / "hooks"


def source_dir(root: Path) -> tuple[Path | None, Path]:
    """Return the ``.githooks`` to install from, and the primary root it was judged against.

    Prefers the primary checkout's copy: the hooks directory is shared, so seeding it from
    whatever branch a worktree happens to be on would leak a feature branch's hook to every
    other worktree. Falls back to the invoking checkout — safe only because we copy.
    """
    primary = primary_root(root)
    for candidate in (primary / SOURCE_DIR_NAME, root / SOURCE_DIR_NAME):
        if candidate.is_dir():
            return candidate, primary
    return None, primary


def install(root: Path, *, check_only: bool = False) -> int:
    source, primary = source_dir(root)
    if source is None:
        print(
            f"error: no {SOURCE_DIR_NAME}/ found in {primary} or {root}",
            file=sys.stderr,
        )
        return 1
    if source.parent != primary:
        print(
            f"note: {primary} has no {SOURCE_DIR_NAME}/ (branch predates tracked hooks); "
            f"installing from {source} instead. Hooks are copied, so pruning this worktree "
            f"cannot disable them.",
        )

    target_dir = hooks_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    problems = 0

    for hook in sorted(p for p in source.iterdir() if p.is_file()):
        target = target_dir / hook.name
        wanted = hook.read_bytes()
        current = target.read_bytes() if target.is_file() else None

        if current == wanted and not target.is_symlink():
            print(f"ok      {target}")
            continue

        if check_only:
            state = "<missing>" if current is None else "differs"
            print(f"DRIFT   {target}: {state} (want a copy of {hook})", file=sys.stderr)
            problems += 1
            continue

        if current is not None or target.is_symlink():
            backup = target.with_name(target.name + ".bak")
            target.replace(backup)
            print(f"backup  {target} -> {backup}")
        target.write_bytes(wanted)
        target.chmod(HOOK_MODE)
        print(f"copied  {hook} -> {target}")

    if check_only and problems:
        print(
            "\nHooks are out of date. Run: python tools/install_git_hooks.py",
            file=sys.stderr,
        )
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the repo's tracked git hooks.")
    parser.add_argument(
        "--check", action="store_true", help="report drift without changing anything"
    )
    args = parser.parse_args(argv)
    return install(repo_root(), check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
