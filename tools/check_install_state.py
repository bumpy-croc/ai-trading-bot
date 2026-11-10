"""Run every install-required ``--check`` and report them together (GH #1147).

Three artifacts cannot be version-controlled and drift silently when nobody re-installs them:
the worktree import shim (site-packages), the git hooks (``.git/hooks``) and the append-only
merge driver (git config). Each has had a ``--check`` for weeks; none was ever *run*, so the
pre-push hook of GH #1077 stayed inert for 21 days after its fix merged. A check nothing
invokes is documentation — this is the single entry point a scheduled caller invokes: the
pre-push hook (every push, human and agent), ``make install-check``, a standup.

Runs the checks with the interpreter running this script, because the shim check inspects
*that* interpreter's site-packages.

Exit status: 0 when all are current, 1 when any has drifted or could not be run. A check that
cannot run counts as drift: "unverifiable" is not "ok".

Usage:
    python tools/check_install_state.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent

CHECKS: tuple[tuple[str, str, str], ...] = (
    ("worktree import shim", "install_worktree_shim.py", "make shim"),
    ("git hooks", "install_git_hooks.py", "make hooks"),
    ("append-only merge driver", "install_merge_drivers.py", "make merge-drivers"),
)

CHECK_TIMEOUT_SECONDS = 30


def run_check(script: Path, python: str = sys.executable) -> tuple[bool, str]:
    """Run one installer's ``--check``. Returns (healthy, combined output)."""
    try:
        result = subprocess.run(
            [python, str(script), "--check"],
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run {script.name}: {exc!r}"
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def run_all(
    checks: tuple[tuple[str, str, str], ...] = CHECKS,
    tools_dir: Path = TOOLS_DIR,
    python: str = sys.executable,
) -> list[tuple[str, bool, str, str]]:
    """Run every check. Returns ``(name, healthy, output, remedy)`` rows."""
    rows = []
    for name, script, remedy in checks:
        healthy, output = run_check(tools_dir / script, python)
        rows.append((name, healthy, output, remedy))
    return rows


def main() -> int:
    rows = run_all()
    failed = [row for row in rows if not row[1]]
    for name, healthy, output, remedy in rows:
        print(f"{'ok   ' if healthy else 'DRIFT'}  {name}")
        if not healthy:
            for line in output.splitlines():
                print(f"        {line}")
            print(f"        fix: {remedy}")
    if failed:
        print(
            f"\n{len(failed)} install-required artifact(s) out of date. Merged is not the same "
            "as in effect.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
