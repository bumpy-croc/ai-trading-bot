"""Behavioural tests for the tracked ``.githooks/pre-push`` hook (GH #1077).

The bug these guard against was invisible precisely because nobody ever proved the hook could
FAIL: it piped pytest into ``tail`` and read the pipeline's status, so it exited 0 whatever the
tests did. So the load-bearing test here is ``test_hook_fails_on_broken_test`` -- "it passes on
a clean tree" is worthless evidence on its own.

Each test builds a throwaway git checkout with a two-line test suite and runs the real hook
script against it, so the assertions are about the shipped file, not a reimplementation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".githooks" / "pre-push"

PASSING_TEST = "import pytest\n\n@pytest.mark.fast\ndef test_ok():\n    assert True\n"
FAILING_TEST = "import pytest\n\n@pytest.mark.fast\ndef test_broken():\n    assert False, 'deliberately broken'\n"
# Mirrors the real pytest.ini closely enough to catch hook flags that fight the project
# config (e.g. `-p no:randomly` vs the inherited `--randomly-seed`).
PYTEST_INI = (
    "[pytest]\naddopts = --randomly-seed=1 --maxfail=5\nmarkers =\n    fast: Fast running tests\n"
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_checkout(path: Path, test_body: str) -> Path:
    """A minimal git repo containing one fast-marked test and a .venv pointing at this python."""
    (path / "tests" / "unit").mkdir(parents=True)
    (path / "tests" / "unit" / "test_sample.py").write_text(test_body)
    (path / "pytest.ini").write_text(PYTEST_INI)

    venv_bin = path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)

    _git("init", "-q", "-b", "main", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    _git("add", "-A", "--", "tests", "pytest.ini", cwd=path)
    _git("commit", "-q", "-m", "initial", cwd=path)
    return path


def _run_hook(
    cwd: Path, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    env = {
        k: v for k, v in os.environ.items() if k not in {"PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST"}
    }
    # Do not let the outer run's venv silently rescue a checkout that has none.
    env.pop("VIRTUAL_ENV", None)
    env.pop("ATB_PREPUSH_PYTHON", None)
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(HOOK)], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


@pytest.fixture
def checkout(tmp_path: Path):
    def _build(test_body: str = PASSING_TEST, name: str = "repo") -> Path:
        return _make_checkout(tmp_path / name, test_body)

    return _build


@pytest.mark.fast
def test_hook_is_executable_and_tracked():
    assert HOOK.is_file(), "the hook must be version-controlled, not only in .git/hooks"
    assert os.access(HOOK, os.X_OK)
    # git skips a non-executable hook silently, so the committed mode is load-bearing too.
    entry = subprocess.run(
        ["git", "ls-files", "-s", "--", ".githooks/pre-push"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert entry.startswith("100755 "), f"hook must be committed executable, got: {entry!r}"


@pytest.mark.slow
def test_hook_fails_on_broken_test(checkout):
    """THE test: a deliberately failing unit test must abort the push."""
    result = _run_hook(checkout(FAILING_TEST))
    assert result.returncode != 0, (
        "hook exited 0 despite a failing test -- this is exactly the GH #1077 defect:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "fast unit tests failed" in result.stderr
    assert "--no-verify" in result.stderr


@pytest.mark.slow
def test_hook_passes_on_clean_tree(checkout):
    result = _run_hook(checkout())
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "fast unit tests passed" in result.stdout


@pytest.mark.slow
def test_hook_fails_when_no_interpreter_is_available(checkout):
    """A hook that cannot verify must not report success."""
    repo = checkout()
    shutil.rmtree(repo / ".venv")
    scrubbed = repo / "empty-bin"
    scrubbed.mkdir()
    for tool in ("git", "bash", "dirname", "uname"):
        found = shutil.which(tool)
        if found:
            (scrubbed / tool).symlink_to(found)

    result = _run_hook(repo, {"PATH": str(scrubbed)})
    assert result.returncode != 0
    assert "no Python interpreter able to run this suite" in result.stderr


@pytest.mark.slow
def test_hook_fails_from_subdirectory(checkout):
    """cwd must not decide anything: run from a subdir, still find the venv and still fail."""
    repo = checkout(FAILING_TEST)
    subdir = repo / "tests" / "unit"
    result = _run_hook(subdir)
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    assert "fast unit tests failed" in result.stderr


@pytest.mark.slow
def test_hook_uses_primary_venv_from_linked_worktree(checkout, tmp_path):
    """Worktrees have no .venv; the hook must reach the primary checkout's interpreter."""
    repo = checkout(FAILING_TEST)
    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "feature", str(worktree), cwd=repo)
    assert not (worktree / ".venv").exists()

    result = _run_hook(worktree)
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    assert "fast unit tests failed" in result.stderr, f"{result.stdout}\n{result.stderr}"
    assert ".venv/bin/python" in result.stdout, "should have used the primary checkout's venv"


def _run_installer(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "install_git_hooks.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.fast
def test_installer_copies_hooks_and_check_reports_drift(checkout):
    """A correct hook nobody installed is the same failure class as an inert one."""
    repo = checkout()
    shutil.copytree(REPO_ROOT / ".githooks", repo / ".githooks")

    before = _run_installer(repo, "--check")
    assert before.returncode == 1, "--check must fail before installation"
    assert "DRIFT" in before.stderr

    assert _run_installer(repo).returncode == 0
    installed = repo / ".git" / "hooks" / "pre-push"
    assert installed.is_file() and not installed.is_symlink()
    assert installed.read_bytes() == (repo / ".githooks" / "pre-push").read_bytes()
    assert os.access(installed, os.X_OK)

    assert _run_installer(repo, "--check").returncode == 0


@pytest.mark.fast
def test_installer_preserves_foreign_hooks(checkout):
    """Hooks this repo does not ship (e.g. a local pre-commit) must survive installation."""
    repo = checkout()
    shutil.copytree(REPO_ROOT / ".githooks", repo / ".githooks")
    foreign = repo / ".git" / "hooks" / "pre-commit"
    foreign.write_text("#!/bin/sh\nexit 0\n")

    assert _run_installer(repo).returncode == 0
    assert foreign.read_text() == "#!/bin/sh\nexit 0\n"


@pytest.mark.slow
def test_installed_hook_survives_pruning_the_worktree_it_was_installed_from(checkout, tmp_path):
    """The regression that a symlinking installer would have shipped.

    ``$GIT_COMMON_DIR/hooks`` is shared by every worktree, but ``make install`` runs inside
    ephemeral agent worktrees. If the installer linked the shared hook into a worktree, pruning
    that worktree would leave a dangling symlink -- and git skips a dangling hook *silently*,
    exiting 0. That is #1077 again, arriving via its own remedy.
    """
    repo = checkout()
    shutil.copytree(REPO_ROOT / ".githooks", repo / ".githooks")
    _git("add", "-A", "--", ".githooks", cwd=repo)
    _git("commit", "-q", "-m", "hooks", cwd=repo)

    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "feature", str(worktree), cwd=repo)
    assert _run_installer(worktree).returncode == 0

    installed = repo / ".git" / "hooks" / "pre-push"
    assert not installed.is_symlink(), "hooks must be copied, never linked into a worktree"

    shutil.rmtree(worktree)
    _git("worktree", "prune", cwd=repo)
    assert installed.is_file(), "installed hook must outlive the worktree it came from"

    # And it must still be able to block a push.
    (repo / "tests" / "unit" / "test_sample.py").write_text(FAILING_TEST)
    result = subprocess.run(
        ["bash", str(installed)], cwd=repo, capture_output=True, text=True, timeout=300
    )
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    assert "fast unit tests failed" in result.stderr


@pytest.mark.slow
def test_hook_reports_environment_failure_distinctly_from_test_failure(checkout):
    """Exit 5 (nothing collected) must not be reported as 'the tests failed'."""
    repo = checkout(PASSING_TEST.replace("@pytest.mark.fast\n", ""))
    result = _run_hook(repo)
    assert result.returncode != 0
    assert "collected no tests" in result.stderr
    assert "fast unit tests failed" not in result.stderr


@pytest.mark.slow
def test_hook_rejects_an_interpreter_that_cannot_run_the_suite(checkout):
    """An interpreter with pytest but without the repo's deps must be skipped, not used."""
    repo = checkout()
    stub = repo / "stub-python"
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)

    result = _run_hook(repo, {"ATB_PREPUSH_PYTHON": str(stub)})
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert str(stub) not in result.stdout, "the unusable interpreter must not have been used"
    assert ".venv/bin/python" in result.stdout


@pytest.mark.fast
def test_check_reports_a_non_executable_hook_as_drift(checkout):
    """A content-identical hook with no +x is skipped by git silently — that is drift."""
    repo = checkout()
    shutil.copytree(REPO_ROOT / ".githooks", repo / ".githooks")
    assert _run_installer(repo).returncode == 0

    installed = repo / ".git" / "hooks" / "pre-push"
    installed.chmod(0o644)

    drifted = _run_installer(repo, "--check")
    assert drifted.returncode == 1, "--check called a hook git would ignore healthy"
    assert "not executable" in drifted.stderr

    assert _run_installer(repo).returncode == 0
    assert os.access(installed, os.X_OK), "install() must restore the executable bit"
    assert installed.read_bytes() == (repo / ".githooks" / "pre-push").read_bytes()
    assert not (repo / ".git" / "hooks" / "pre-push.bak").exists(), "no backup churn for a chmod"
    assert _run_installer(repo, "--check").returncode == 0
