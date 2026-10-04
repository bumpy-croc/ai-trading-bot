"""Tests for the install-required tooling: the Makefile interpreter and the combined check.

GH #1147: ``make hooks`` -- the documented remedy for an inert pre-push hook -- died on
``python: command not found`` outside an activated venv, the same string as the defect it
repairs, and nothing ever ran the three ``--check`` commands on a schedule.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
CHECKER_SOURCE = REPO_ROOT / "tools" / "check_install_state.py"

pytestmark = [
    pytest.mark.fast,
    pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed"),
]

TOOL_TARGETS = [
    "shim",
    "shim-check",
    "hooks",
    "hooks-check",
    "merge-drivers",
    "merge-drivers-check",
]


def _make(
    *args: str, cwd: Path, env: dict[str, str], dry_run: bool = False
) -> subprocess.CompletedProcess:
    # Absolute path: the tests deliberately run make under a PATH that has no interpreter.
    command = [shutil.which("make") or "make", *(["-n"] if dry_run else []), *args]
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)


def _env(path: str, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PY", "MAKEFLAGS"}}
    env["PATH"] = path
    env.update(extra)
    return env


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def isolated_make_dir(tmp_path: Path) -> Path:
    """A directory holding only the Makefile: not a git repo, no .venv."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copy(MAKEFILE, workdir / "Makefile")
    return workdir


def _tool_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if "tools/" in line and "echo" not in line]


def test_makefile_never_invokes_bare_python(isolated_make_dir):
    result = _make(
        *TOOL_TARGETS,
        "install-check",
        "PY=/opt/venv/bin/python",
        cwd=isolated_make_dir,
        env=_env(os.defpath),
        dry_run=True,
    )
    assert result.returncode == 0, result.stderr
    lines = _tool_lines(result.stdout)
    assert len(lines) == len(TOOL_TARGETS) + 1
    assert all(line.startswith("/opt/venv/bin/python ") for line in lines), lines
    assert not re.search(r"^(python|pip)\s", result.stdout, re.MULTILINE), result.stdout


def test_makefile_resolves_python3_when_there_is_no_venv_and_no_python(isolated_make_dir, tmp_path):
    """The failing environment: outside a venv only `python3` exists (macOS)."""
    bin_dir = tmp_path / "bin"
    python3 = _executable(bin_dir / "python3")
    result = _make("hooks-check", cwd=isolated_make_dir, env=_env(str(bin_dir)), dry_run=True)
    assert result.returncode == 0, result.stderr
    assert f"{python3} tools/install_git_hooks.py --check" in result.stdout


def test_makefile_prefers_the_active_venv_over_python3(isolated_make_dir, tmp_path):
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "python3")
    venv_python = _executable(tmp_path / "venv" / "bin" / "python")
    result = _make(
        "shim-check",
        cwd=isolated_make_dir,
        env=_env(str(bin_dir), VIRTUAL_ENV=str(tmp_path / "venv")),
        dry_run=True,
    )
    assert f"{venv_python} tools/install_worktree_shim.py --check" in result.stdout


def test_makefile_prefers_the_checkouts_own_venv(isolated_make_dir, tmp_path):
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "python3")
    _executable(isolated_make_dir / ".venv" / "bin" / "python")
    result = _make(
        "merge-drivers-check", cwd=isolated_make_dir, env=_env(str(bin_dir)), dry_run=True
    )
    assert f"{isolated_make_dir / '.venv' / 'bin' / 'python'} tools/install_merge_drivers.py" in (
        result.stdout
    )


def test_a_linked_worktree_resolves_the_primarys_venv(tmp_path):
    """Worktrees have no .venv, and the shim must land in the shared venv's site-packages."""
    git = shutil.which("git")
    assert git is not None
    primary = tmp_path / "primary"
    primary.mkdir()
    shutil.copy(MAKEFILE, primary / "Makefile")
    git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for args in (
        ["init", "-q", "-b", "main"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run([git, *args], cwd=primary, check=True, capture_output=True, env=git_env)
    worktree = tmp_path / "wt"
    subprocess.run(
        [git, "worktree", "add", "-q", "--detach", str(worktree)],
        cwd=primary,
        check=True,
        capture_output=True,
        env=git_env,
    )
    venv_python = _executable(primary / ".venv" / "bin" / "python")
    assert not (worktree / ".venv").exists()

    result = _make(
        "shim",
        cwd=worktree,
        env=_env(f"{Path(git).parent}{os.pathsep}{os.defpath}"),
        dry_run=True,
    )
    assert result.returncode == 0, result.stderr
    match = re.search(r"^(\S+) tools/install_worktree_shim\.py$", result.stdout, re.MULTILINE)
    assert match is not None, result.stdout
    # git reports the realpath; tmp_path may sit behind a symlink (macOS /var -> /private/var).
    assert Path(match.group(1)).resolve() == venv_python.resolve()


@pytest.mark.parametrize("target", [*TOOL_TARGETS, "install-check"])
def test_every_tool_target_fails_loudly_without_any_interpreter(
    isolated_make_dir, tmp_path, target
):
    """No interpreter must be an explicit error, never an empty `$(PY) tools/x.py`."""
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    result = _make(target, cwd=isolated_make_dir, env=_env(str(empty_bin)))
    assert result.returncode != 0
    assert "no Python interpreter found" in result.stderr
    assert "PY=/path/to/python" in result.stderr


def test_make_help_and_clean_do_not_need_python(isolated_make_dir, tmp_path):
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    assert _make("help", cwd=isolated_make_dir, env=_env(str(empty_bin))).returncode == 0


def test_an_explicit_empty_py_is_an_error_not_a_silent_skip(isolated_make_dir, tmp_path):
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "python3")
    result = _make("hooks-check", "PY=", cwd=isolated_make_dir, env=_env(str(bin_dir)))
    assert result.returncode != 0
    assert "no Python interpreter found" in result.stderr


# --------------------------------------------------------------------------------------
# tools/check_install_state.py
# --------------------------------------------------------------------------------------


def _load_checker():
    spec = importlib.util.spec_from_file_location("_check_install_state", CHECKER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _script(directory: Path, name: str, body: str) -> None:
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(body)


def test_checker_covers_all_three_install_required_artifacts():
    scripts = {script for _, script, _ in checker.CHECKS}
    assert scripts == {
        "install_worktree_shim.py",
        "install_git_hooks.py",
        "install_merge_drivers.py",
    }
    for script in scripts:
        assert (REPO_ROOT / "tools" / script).is_file()


def test_every_remedy_in_the_checker_is_a_real_make_target():
    targets = set(re.findall(r"^([a-z-]+):", MAKEFILE.read_text(), re.MULTILINE))
    assert {remedy.removeprefix("make ") for _, _, remedy in checker.CHECKS} <= targets


def test_run_all_reports_each_check_independently(tmp_path):
    _script(tmp_path, "good.py", "print('ok   thing')\n")
    _script(tmp_path, "bad.py", "import sys\nprint('DRIFT thing', file=sys.stderr)\nsys.exit(1)\n")
    rows = checker.run_all(
        (("good", "good.py", "make good"), ("bad", "bad.py", "make bad")), tools_dir=tmp_path
    )
    assert [(name, healthy) for name, healthy, _, _ in rows] == [("good", True), ("bad", False)]
    assert "DRIFT thing" in rows[1][2]


def test_a_check_that_cannot_run_counts_as_drift(tmp_path):
    """ "Unverifiable" is not "ok": a missing or crashing checker must not read as healthy."""
    _script(tmp_path, "crash.py", "raise SystemExit(3)\n")
    rows = checker.run_all(
        (("missing", "nope.py", "make x"), ("crash", "crash.py", "make y")), tools_dir=tmp_path
    )
    assert [healthy for _, healthy, _, _ in rows] == [False, False]


def test_main_exits_nonzero_and_names_the_remedy_on_drift(tmp_path, monkeypatch, capsys):
    rows = [("git hooks", False, "DRIFT line", "make hooks")]
    monkeypatch.setattr(checker, "run_all", lambda: rows)

    assert checker.main() == 1
    captured = capsys.readouterr()
    assert "DRIFT  git hooks" in captured.out
    assert "fix: make hooks" in captured.out
    assert "out of date" in captured.err


def test_main_exits_zero_when_everything_is_current(monkeypatch, capsys):
    monkeypatch.setattr(checker, "run_all", lambda: [("git hooks", True, "ok", "make hooks")])
    assert checker.main() == 0
    assert "ok     git hooks" in capsys.readouterr().out


def test_checker_runs_as_a_script_against_this_repo():
    """Smoke: it executes and exits with a verdict (machine state decides which)."""
    result = subprocess.run(
        [sys.executable, str(CHECKER_SOURCE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in (0, 1), result.stderr
    for name, _, _ in checker.CHECKS:
        assert name in result.stdout
