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
* **The source is the integration branch, not a working tree.** ``origin/develop`` is read
  first. The primary checkout is held on ``main``, which trails ``develop`` between promotes,
  so installing from it (the old rule) *downgraded* a hook that ``develop`` had already fixed,
  and ``--check`` reported DRIFT against an install byte-identical to ``develop``'s. A feature
  branch in a worktree still cannot seed the repo-wide hook. Only when there is no such ref (a
  fresh clone, a test repo) does the primary's, then the invoking checkout's, ``.githooks/``
  stand in.

Drift is the price of copying, so ``--check`` compares content **and the executable bit** —
git skips a non-executable hook silently — and is wired to ``make hooks-check``; ``make
install`` re-copies on every run.

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
SOURCE_REF = "origin/develop"


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


def _hooks_from_ref(root: Path) -> dict[str, bytes] | None:
    """The tracked hooks as committed on ``SOURCE_REF``, or None when that ref has none."""
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", SOURCE_REF, "--", SOURCE_DIR_NAME],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0 or not listing.stdout.strip():
        return None
    hooks: dict[str, bytes] = {}
    for path in listing.stdout.splitlines():
        blob = subprocess.run(
            ["git", "show", f"{SOURCE_REF}:{path}"], cwd=str(root), capture_output=True
        )
        if blob.returncode != 0:
            return None
        hooks[Path(path).name] = blob.stdout
    return hooks


def source_hooks(root: Path) -> tuple[dict[str, bytes] | None, str]:
    """Return ``({hook name: content}, where it came from)``; the dict is None if none found.

    Order: ``SOURCE_REF``, the primary checkout's ``.githooks/``, the invoking checkout's.
    The fallbacks are safe only because hooks are copied, so pruning a worktree cannot
    disable an installed hook.
    """
    from_ref = _hooks_from_ref(root)
    if from_ref is not None:
        return from_ref, SOURCE_REF
    for candidate in (primary_root(root) / SOURCE_DIR_NAME, root / SOURCE_DIR_NAME):
        if candidate.is_dir():
            hooks = {p.name: p.read_bytes() for p in sorted(candidate.iterdir()) if p.is_file()}
            return hooks, str(candidate)
    return None, SOURCE_REF


def install(root: Path, *, check_only: bool = False) -> int:
    hooks, origin = source_hooks(root)
    if hooks is None:
        print(
            f"error: no {SOURCE_DIR_NAME}/ found on {SOURCE_REF}, in {primary_root(root)} or "
            f"{root}",
            file=sys.stderr,
        )
        return 1
    if origin != SOURCE_REF:
        print(
            f"note: {SOURCE_REF} has no {SOURCE_DIR_NAME}/; using {origin} instead. Hooks are "
            f"copied, so pruning a worktree cannot disable them.",
        )

    target_dir = hooks_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    problems = 0

    for name, wanted in sorted(hooks.items()):
        target = target_dir / name
        hook = f"{SOURCE_DIR_NAME}/{name}"
        current = target.read_bytes() if target.is_file() else None
        # The mode bit is as load-bearing as the content: git skips a non-executable hook
        # SILENTLY and the push succeeds. Content-only drift detection would print "ok" over
        # exactly the always-pass this tool exists to prevent.
        executable = target.is_file() and os.access(target, os.X_OK)
        content_ok = current == wanted and not target.is_symlink()

        if content_ok and executable:
            print(f"ok      {target}")
            continue

        if check_only:
            if current is None:
                state = "<missing>"
            elif not content_ok:
                state = "is a symlink" if target.is_symlink() else "differs"
            else:
                state = "not executable — git would skip it silently"
            print(f"DRIFT   {target}: {state} (want an executable copy of {hook})", file=sys.stderr)
            problems += 1
            continue

        if content_ok:
            target.chmod(HOOK_MODE)
            print(f"chmod   {target} (restored the executable bit)")
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
