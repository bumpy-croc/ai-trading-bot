"""Install this repo's tracked git hooks (``.githooks/``) into the active hooks directory.

The hooks used to live only in ``.git/hooks``: untracked, unreviewable, and silently different
on every machine — which is how the inert pre-push hook of GH #1077 survived unnoticed. The
hook source is now version-controlled; this installer links it into place.

Symlinks (rather than copies) are used so a merged fix reaches every checkout without a
re-install. Existing hooks that this repo does not own are left alone.

Usage:
    python tools/install_git_hooks.py           # install / repair
    python tools/install_git_hooks.py --check    # report drift, exit 1 if not installed
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SOURCE_DIR_NAME = ".githooks"


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repo_root(start: Path | None = None) -> Path:
    return Path(_git("rev-parse", "--show-toplevel", cwd=start)).resolve()


def hooks_dir(root: Path) -> Path:
    """Directory git actually runs hooks from: ``core.hooksPath`` if set, else the common dir."""
    try:
        configured = _git("config", "--get", "core.hooksPath", cwd=root)
    except subprocess.CalledProcessError:
        configured = ""
    if configured:
        path = Path(os.path.expanduser(configured))
        return path if path.is_absolute() else (root / path)
    common = Path(_git("rev-parse", "--git-common-dir", cwd=root))
    if not common.is_absolute():
        common = root / common
    return common.resolve() / "hooks"


def install(root: Path, *, check_only: bool = False) -> int:
    source_dir = root / SOURCE_DIR_NAME
    if not source_dir.is_dir():
        print(f"error: {source_dir} does not exist", file=sys.stderr)
        return 1

    target_dir = hooks_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    problems = 0

    for source in sorted(p for p in source_dir.iterdir() if p.is_file()):
        target = target_dir / source.name
        wanted = source.resolve()
        current = target.resolve() if target.exists() or target.is_symlink() else None

        if current == wanted:
            print(f"ok      {target} -> {wanted}")
            continue

        if check_only:
            actual = current if current is not None else "<missing>"
            print(f"DRIFT   {target}: {actual} (want {wanted})", file=sys.stderr)
            problems += 1
            continue

        if target.exists() or target.is_symlink():
            backup = target.with_suffix(target.suffix + ".bak")
            target.replace(backup)
            print(f"backup  {target} -> {backup}")
        target.symlink_to(wanted)
        print(f"linked  {target} -> {wanted}")

    if check_only and problems:
        print(
            "\nHooks are not installed. Run: python tools/install_git_hooks.py",
            file=sys.stderr,
        )
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report drift without changing anything"
    )
    args = parser.parse_args(argv)
    return install(repo_root(), check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
