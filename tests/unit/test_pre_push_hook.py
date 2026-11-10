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
import signal
import subprocess
import sys
import time
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
    # Never share the machine-wide lock with real pushes or with the other tests in this file.
    env["ATB_PREPUSH_LOCK_DIR"] = str(cwd.parent / f"{cwd.name}.prepush-lock")
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


GIT_SHELLOUT_TEST = (
    "import subprocess\nimport pytest\n\n"
    "@pytest.mark.fast\n"
    "def test_shells_out_to_git(tmp_path):\n"
    "    nested = tmp_path / 'nested'\n"
    "    nested.mkdir()\n"
    "    subprocess.run(['git', 'init', '-q', str(nested)], check=True)\n"
    "    subprocess.run(['git', 'config', 'user.email', 't@t'], cwd=nested, check=True)\n"
    "    subprocess.run(['git', 'config', 'user.name', 'T'], cwd=nested, check=True)\n"
    "    (nested / 'f.txt').write_text('x')\n"
    "    subprocess.run(['git', 'add', 'f.txt'], cwd=nested, check=True)\n"
    "    subprocess.run(['git', 'commit', '-q', '-m', 'x'], cwd=nested, check=True)\n"
)


@pytest.mark.slow
def test_hook_passes_when_a_test_shells_out_to_git_under_a_real_pushs_env(checkout):
    """GH #1148: a real `git push` exports GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE/GIT_PREFIX into
    the hook process. Any test that does its own `git init`/`git commit` in a throwaway
    directory must not inherit those and silently corrupt/redirect against the checkout being
    pushed. `_run_hook`'s default env strips PYTEST_ADDOPTS etc but never simulated git's own
    injection -- which is exactly why this regression shipped invisibly for 21 days."""
    repo = checkout(GIT_SHELLOUT_TEST)
    git_dir = repo / ".git"
    result = _run_hook(
        repo,
        {
            "GIT_DIR": str(git_dir),
            "GIT_WORK_TREE": str(repo),
            "GIT_PREFIX": "",
            "GIT_INDEX_FILE": str(git_dir / "index"),
        },
    )
    assert result.returncode == 0, (
        "hook failed with git's own hook-injected env present -- GIT_DIR leaked into the "
        f"pytest subprocess and corrupted its git fixture (GH #1148):\n{result.stdout}\n{result.stderr}"
    )
    assert "fast unit tests passed" in result.stdout


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
        ["bash", str(installed)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "ATB_PREPUSH_LOCK_DIR": str(tmp_path / "installed.prepush-lock")},
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


NEWER_HOOK = "#!/usr/bin/env bash\n# the version develop already fixed\nexit 0\n"
STALE_HOOK = "#!/usr/bin/env bash\n# what main still carries\nexit 0\n"


def _checkout_with_main_behind_develop(checkout) -> Path:
    """Primary working tree on a stale ``main``; ``origin/develop`` already has the fix."""
    repo = checkout()
    (repo / ".githooks").mkdir()
    (repo / ".githooks" / "pre-push").write_text(STALE_HOOK)
    _git("add", "-A", "--", ".githooks", cwd=repo)
    _git("commit", "-q", "-m", "stale hook on main", cwd=repo)
    _git("checkout", "-q", "-b", "develop", cwd=repo)
    (repo / ".githooks" / "pre-push").write_text(NEWER_HOOK)
    _git("commit", "-q", "-am", "fixed hook on develop", cwd=repo)
    _git("update-ref", "refs/remotes/origin/develop", "develop", cwd=repo)
    _git("checkout", "-q", "main", cwd=repo)
    assert (repo / ".githooks" / "pre-push").read_text() == STALE_HOOK
    return repo


@pytest.mark.fast
def test_installer_sources_from_origin_develop_not_a_stale_primary(checkout):
    """GH #1138/#1260: installing from the primary (held on main) downgraded a fixed hook."""
    repo = _checkout_with_main_behind_develop(checkout)

    assert _run_installer(repo).returncode == 0
    installed = repo / ".git" / "hooks" / "pre-push"
    assert installed.read_text() == NEWER_HOOK
    assert os.access(installed, os.X_OK)
    assert _run_installer(repo, "--check").returncode == 0


@pytest.mark.fast
def test_check_does_not_report_drift_for_an_install_that_matches_develop(checkout):
    """The false DRIFT that made a scheduled check worthless until the next promote."""
    repo = _checkout_with_main_behind_develop(checkout)
    installed = repo / ".git" / "hooks" / "pre-push"
    installed.write_text(NEWER_HOOK)
    installed.chmod(0o755)

    assert _run_installer(repo, "--check").returncode == 0


@pytest.mark.fast
def test_check_reports_drift_when_the_install_is_older_than_develop(checkout):
    repo = _checkout_with_main_behind_develop(checkout)
    installed = repo / ".git" / "hooks" / "pre-push"
    installed.write_text(STALE_HOOK)
    installed.chmod(0o755)

    drifted = _run_installer(repo, "--check")
    assert drifted.returncode == 1
    assert "DRIFT" in drifted.stderr


@pytest.mark.fast
def test_installer_check_names_the_tracked_path_it_wants(checkout):
    repo = checkout()
    shutil.copytree(REPO_ROOT / ".githooks", repo / ".githooks")
    drifted = _run_installer(repo, "--check")
    assert "want an executable copy of .githooks/pre-push" in drifted.stderr


def _install_checker_stub(repo: Path, body: str) -> None:
    (repo / "tools").mkdir(exist_ok=True)
    (repo / "tools" / "check_install_state.py").write_text(body)


@pytest.mark.slow
def test_hook_warns_but_does_not_block_when_an_install_has_drifted(checkout):
    """GH #1147: every push is the scheduled assertion; stale installs must be loud, not fatal."""
    repo = checkout()
    _install_checker_stub(repo, "import sys\nprint('DRIFT git hooks')\nsys.exit(1)\n")
    result = _run_hook(repo)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "install-required artifacts are out of date" in result.stderr
    assert "DRIFT git hooks" in result.stderr
    assert "fast unit tests passed" in result.stdout


@pytest.mark.slow
def test_hook_is_silent_about_installs_when_they_are_current(checkout):
    repo = checkout()
    _install_checker_stub(repo, "print('ok')\n")
    result = _run_hook(repo)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "out of date" not in result.stderr


@pytest.mark.slow
def test_a_crashing_install_checker_cannot_block_or_pass_the_push(checkout):
    """The checker is advisory in both directions: a crash must not decide the push."""
    repo = checkout(FAILING_TEST)
    _install_checker_stub(repo, "raise SystemExit('checker exploded')\n")
    result = _run_hook(repo)
    assert result.returncode != 0, "a broken checker must not rescue a failing suite"
    assert "fast unit tests failed" in result.stderr

    passing = checkout(PASSING_TEST, name="repo2")
    _install_checker_stub(passing, "raise SystemExit('checker exploded')\n")
    assert _run_hook(passing).returncode == 0


# --------------------------------------------------------------------------------------
# Serialisation and cleanup: one run at a time, and never outliving the push that started it
# --------------------------------------------------------------------------------------


def _hold_lock(lock_dir: Path, pid: int, age_seconds: int = 0) -> None:
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(str(pid))
    (lock_dir / "started").write_text(str(int(time.time()) - age_seconds))


@pytest.fixture
def live_process():
    """A process that stays alive for the test, standing in for another hook run."""
    proc = subprocess.Popen(["sleep", "300"])
    yield proc
    proc.kill()
    proc.wait()


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


@pytest.mark.slow
def test_hook_waits_for_a_live_lock_holder_then_fails_without_running_tests(
    checkout, tmp_path, live_process
):
    """If the suite cannot be run because another run holds the machine, that is a failure."""
    repo = checkout()
    lock = tmp_path / "lock"
    _hold_lock(lock, live_process.pid)

    result = _run_hook(
        repo, {"ATB_PREPUSH_LOCK_DIR": str(lock), "ATB_PREPUSH_LOCK_WAIT_SECONDS": "5"}
    )
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    assert "waiting for another pre-push run" in result.stderr
    assert "The tests were NOT run" in result.stderr
    assert (lock / "pid").read_text() == str(live_process.pid), "must not steal a live lock"


@pytest.mark.slow
def test_hook_takes_over_a_lock_whose_owner_is_dead(checkout, tmp_path):
    repo = checkout()
    lock = tmp_path / "lock"
    _hold_lock(lock, _dead_pid())

    result = _run_hook(repo, {"ATB_PREPUSH_LOCK_DIR": str(lock)})
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "taking over the lock left by dead run" in result.stderr
    assert not lock.exists(), "the lock must be released after the run"


@pytest.mark.slow
def test_hook_takes_over_a_stale_lock_even_if_the_pid_is_alive(checkout, tmp_path, live_process):
    """A reused pid must not wedge every push forever."""
    repo = checkout()
    lock = tmp_path / "lock"
    _hold_lock(lock, live_process.pid, age_seconds=5000)

    result = _run_hook(
        repo, {"ATB_PREPUSH_LOCK_DIR": str(lock), "ATB_PREPUSH_LOCK_STALE_SECONDS": "60"}
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "stale after 60s" in result.stderr
    assert not lock.exists()


@pytest.mark.slow
def test_lock_is_released_when_the_tests_fail(checkout, tmp_path):
    repo = checkout(FAILING_TEST)
    lock = tmp_path / "lock"
    result = _run_hook(repo, {"ATB_PREPUSH_LOCK_DIR": str(lock)})
    assert result.returncode != 0
    assert "fast unit tests failed" in result.stderr
    assert not lock.exists()


HANGING_TEST = (
    "import os, time\nimport pytest\n\n"
    "@pytest.mark.fast\n"
    "def test_hangs():\n"
    "    with open(os.environ['ATB_TEST_PIDFILE'], 'w') as handle:\n"
    "        handle.write(str(os.getpid()))\n"
    "    time.sleep(25)\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(condition, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.25)
    return condition()


@pytest.mark.slow
def test_tests_and_lock_do_not_outlive_the_process_that_started_the_hook(checkout, tmp_path):
    """The failure that wedged the machine: a timed-out `git push` left pytest running."""
    repo = checkout(HANGING_TEST)
    lock = tmp_path / "lock"
    pidfile = tmp_path / "worker.pid"
    env = {k: v for k, v in os.environ.items() if k not in {"PYTEST_ADDOPTS", "VIRTUAL_ENV"}}
    env.pop("ATB_PREPUSH_PYTHON", None)
    env.update({"ATB_PREPUSH_LOCK_DIR": str(lock), "ATB_TEST_PIDFILE": str(pidfile)})
    # The trailing `; true` stops bash exec-ing the hook directly, so the hook has a parent.
    parent = subprocess.Popen(
        ["bash", "-c", f"bash {HOOK}; true"],
        cwd=repo,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker_pid = None
    try:
        assert _wait_for(pidfile.exists, 90), "the test never started"
        assert lock.exists(), "the run should hold the lock while the tests execute"
        worker_pid = int(pidfile.read_text())
        parent.kill()  # SIGKILL: no chance to forward a signal, like a killed `git push`
        parent.wait()

        assert _wait_for(lambda: not _alive(worker_pid), 20), "pytest outlived its push"
        assert _wait_for(lambda: not lock.exists(), 20), "the lock leaked"
    finally:
        if worker_pid is not None and _alive(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)
