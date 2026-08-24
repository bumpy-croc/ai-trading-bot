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

``--check`` inspects *this machine's* git config, so — like ``make hooks-check`` — it is a
workstation check and not a CI gate: a fresh clone would fail it every run, and running it
after ``make install`` would pass tautologically. What CI can usefully assert is repository
content, which ``tests/unit/test_append_only_merge_driver.py`` does: `.gitattributes` and this
tool's allowlist must name the same files.

Usage:
    python tools/install_merge_drivers.py           # register / repair
    python tools/install_merge_drivers.py --check   # report drift, exit 1 if not registered
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Registered as a *relative* command on purpose, and deliberately unlike the hook installer
# (tools/install_git_hooks.py, GH #1077), which copies hook *sources* from the primary
# checkout. The difference is that a hook is a file git copies once, while this is a command
# git re-runs at merge time:
#
# * git runs a merge driver from the top of the work tree doing the merge, so a relative path
#   resolves inside that worktree and always matches the code being merged.
# * git config lives in the shared common dir, so an absolute path would pin every linked
#   worktree to one checkout. Pinning to the primary checkout — the hook installer's rule —
#   would be actively wrong here: the primary is held on the production branch, which does not
#   carry this tool at all until this change ships to prod.
# * If a worktree is on a branch that predates the driver, the command simply is not there,
#   git treats the failure as "unresolved" and leaves ordinary conflict markers. Every failure
#   mode degrades to the status quo rather than to a bad merge.
#
# Stdlib-only, so bare `python3` suffices (and, unlike `python`, it exists on macOS).
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
