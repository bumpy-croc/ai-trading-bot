"""Register this repo's custom git merge drivers in the local git config.

A merge driver has two halves. The half that says *which* files use it lives in
`.gitattributes` and is version-controlled. The half that says *what to run* is a
``merge.<name>.driver`` config key, and git deliberately refuses to take that from a tracked
file — so it must be installed per clone. When it is missing, git silently ignores the
`.gitattributes` entry and falls back to an ordinary conflict.

That silence is the whole problem: an unregistered driver looks exactly like a driver that
had nothing to do (GH #1077's class — a mechanism whose absence is invisible). So this runs
from ``make install`` alongside the git-hook and worktree-shim installers, and ``--check``
reports an unregistered or stale driver as drift.

Usage:
    python tools/install_merge_drivers.py           # register / repair
    python tools/install_merge_drivers.py --check   # report drift, exit 1 if not registered
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Registered as a *relative* command on purpose: git runs a merge driver from the top of the
# work tree doing the merge, so a relative path follows the worktree while an absolute one
# would pin every linked worktree to whichever checkout happened to run `make install`
# (git config lives in the shared common dir). Stdlib-only, so bare `python3` suffices.
DRIVERS: dict[str, dict[str, str]] = {
    "append-only": {
        "name": "Union merge for append-only records (keeps both sides, chronological)",
        "driver": "python3 tools/merge_append_only.py %O %A %B %L %P",
    },
}


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )


def repo_root(start: Path | None = None) -> Path:
    proc = _git("rev-parse", "--show-toplevel", cwd=start)
    proc.check_returncode()
    return Path(proc.stdout.strip()).resolve()


def configured(root: Path, key: str) -> str | None:
    proc = _git("config", "--get", key, cwd=root)
    return proc.stdout.strip() if proc.returncode == 0 else None


def install(root: Path, *, check_only: bool = False) -> int:
    problems = 0
    for driver, settings in DRIVERS.items():
        for suffix, wanted in settings.items():
            key = f"merge.{driver}.{suffix}"
            current = configured(root, key)
            if current == wanted:
                print(f"ok      {key}")
                continue
            if check_only:
                state = "<unregistered>" if current is None else f"{current!r}"
                print(f"DRIFT   {key}: {state} (want {wanted!r})", file=sys.stderr)
                problems += 1
                continue
            _git("config", "--local", key, wanted, cwd=root).check_returncode()
            print(f"set     {key} = {wanted}")

    if check_only and problems:
        print(
            "\nMerge drivers are not registered, so .gitattributes entries for append-only "
            "files are being IGNORED and those files will conflict by hand.\n"
            "Run: python tools/install_merge_drivers.py  (or: make merge-drivers)",
            file=sys.stderr,
        )
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="report drift without changing anything"
    )
    args = parser.parse_args(argv)
    return install(repo_root(), check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
