"""Regression tests for the primary-checkout write guard (GH #1082).

The defect: an agent shell's cwd is silently reset to the primary checkout mid-task, after
which every relative path resolves against the primary checkout instead of the agent's
worktree — a stale read, or a write into the tree that must stay pinned to ``main``.

These tests build a synthetic primary checkout plus a linked worktree under ``tmp_path`` (real
``git worktree add``, so ``.git``-file resolution is exercised for real) and drive the guard as
a subprocess over the actual hook stdin/exit-code protocol.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_SOURCE = REPO_ROOT / "tools" / "primary_checkout_guard.py"

pytestmark = pytest.mark.fast


def _load_guard():
    spec = importlib.util.spec_from_file_location("_primary_checkout_guard", GUARD_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Per-test HOME so session pins and the override sentinel never leak between tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    yield home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    )


def _make_checkout(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "cli").mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "cli" / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text('[project]\nname = "ai-trading-bot"\n')
    (root / "CLAUDE.md").write_text("primary copy\n")
    (root / ".gitignore").write_text(".venv\nlogs/\n.claude/worktrees/\n.agent-active\n")


@pytest.fixture
def checkouts(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic primary checkout and a linked worktree inside ``.claude/worktrees``."""
    primary = tmp_path / "ai-trading-bot"
    primary.mkdir()
    _make_checkout(primary)
    _git(primary, "init", "-q", "-b", "main")
    _git(primary, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(primary, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")

    worktree = primary / ".claude" / "worktrees" / "agent-1"
    _git(primary, "worktree", "add", "-q", "--detach", str(worktree))
    (worktree / ".agent-active").touch()
    (primary / ".venv").mkdir()
    (primary / "logs").mkdir()
    return primary, worktree


def _payload(tool_name: str, tool_input: dict, cwd: Path, session_id: str = "s1") -> dict:
    return {
        "session_id": session_id,
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "cwd": str(cwd),
    }


def _run_hook(payload: dict, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != guard.ALLOW_ENV}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(GUARD_SOURCE)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )


# --------------------------------------------------------------------------------------
# an agent-context write to the primary is refused
# --------------------------------------------------------------------------------------


def test_edit_tool_write_into_primary_is_refused(checkouts):
    primary, _ = checkouts
    result = _run_hook(_payload("Edit", {"file_path": str(primary / "CLAUDE.md")}, cwd=primary))
    assert result.returncode == 2
    assert "ATB PRIMARY CHECKOUT WRITE REFUSED (GH #1082)" in result.stderr
    assert str(primary / "CLAUDE.md") in result.stderr
    assert guard.ALLOW_ENV in result.stderr


def test_relative_edit_after_cwd_reset_is_refused(checkouts):
    """The exact #1082 shape: a relative path resolving against the reset cwd."""
    primary, _ = checkouts
    result = _run_hook(_payload("Edit", {"file_path": "CLAUDE.md"}, cwd=primary))
    assert result.returncode == 2
    assert str(primary / "CLAUDE.md") in result.stderr


def test_sed_in_place_into_primary_is_refused(checkouts):
    primary, _ = checkouts
    result = _run_hook(_payload("Bash", {"command": "sed -i '' 's/a/b/' CLAUDE.md"}, cwd=primary))
    assert result.returncode == 2
    assert "WRITE REFUSED" in result.stderr


def test_redirect_into_primary_is_refused(checkouts):
    primary, _ = checkouts
    result = _run_hook(_payload("Bash", {"command": "echo hi > CLAUDE.md"}, cwd=primary))
    assert result.returncode == 2


def test_absolute_write_into_primary_from_a_worktree_is_refused(checkouts):
    """Not just a cwd accident — an explicit cross-checkout write is refused too."""
    primary, worktree = checkouts
    result = _run_hook(_payload("Bash", {"command": f"rm -f {primary}/CLAUDE.md"}, cwd=worktree))
    assert result.returncode == 2


def test_git_working_tree_mutation_in_primary_is_refused(checkouts):
    primary, _ = checkouts
    result = _run_hook(_payload("Bash", {"command": "git checkout develop"}, cwd=primary))
    assert result.returncode == 2


def test_cd_into_primary_then_write_is_refused(checkouts):
    primary, worktree = checkouts
    result = _run_hook(
        _payload("Bash", {"command": f"cd {primary} && touch scratch.txt"}, cwd=worktree)
    )
    assert result.returncode == 2


# --------------------------------------------------------------------------------------
# a worktree write is unaffected
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "sed -i '' 's/a/b/' CLAUDE.md",
        "echo hi > CLAUDE.md",
        "git checkout -b feature/x",
        "touch .agent-active",
        "rm -f scratch.txt",
    ],
)
def test_worktree_writes_are_allowed(checkouts, command):
    _, worktree = checkouts
    result = _run_hook(_payload("Bash", {"command": command}, cwd=worktree))
    assert result.returncode == 0, result.stderr


def test_edit_inside_worktree_is_allowed(checkouts):
    _, worktree = checkouts
    result = _run_hook(_payload("Edit", {"file_path": str(worktree / "CLAUDE.md")}, cwd=worktree))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "git worktree add .claude/worktrees/agent-2 --detach",
        "git worktree list",
        "git fetch origin develop",
        "git status --porcelain",
        "cat CLAUDE.md",
        'python3 -c "print(1 > 2)"',
        'grep -rn "a|b" src/',
    ],
)
def test_non_mutating_primary_operations_are_allowed(checkouts, command):
    """Creating worktrees and reading are how agent work starts; blocking them is not the goal."""
    primary, _ = checkouts
    result = _run_hook(_payload("Bash", {"command": command}, cwd=primary))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("relative", [".venv/pyvenv.cfg", "logs/bot.log"])
def test_git_ignored_paths_in_primary_are_allowed(checkouts, relative):
    """The shared venv and log dirs live in the primary checkout and are written all the time."""
    primary, _ = checkouts
    result = _run_hook(_payload("Bash", {"command": f"touch {relative}"}, cwd=primary))
    assert result.returncode == 0, result.stderr


def test_writes_outside_any_checkout_are_allowed(checkouts, tmp_path):
    primary, _ = checkouts
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = _run_hook(_payload("Bash", {"command": f"touch {scratch}/note.txt"}, cwd=primary))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------------------
# a human/normal write still works
# --------------------------------------------------------------------------------------


def test_env_override_allows_the_write(checkouts):
    primary, _ = checkouts
    result = _run_hook(
        _payload("Edit", {"file_path": str(primary / "CLAUDE.md")}, cwd=primary),
        env_extra={guard.ALLOW_ENV: "1"},
    )
    assert result.returncode == 0
    assert "override active" in result.stderr


def test_guard_is_inert_outside_claude_code(checkouts):
    """No hook payload, no enforcement: the human's own editor and terminal are untouched."""
    primary, _ = checkouts
    target = primary / "CLAUDE.md"
    target.write_text("edited by a human\n")
    assert target.read_text() == "edited by a human\n"
    result = _run_hook({})
    assert result.returncode == 0


def test_malformed_payload_fails_open(checkouts):
    result = _run_hook({"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert result.returncode == 0


# --------------------------------------------------------------------------------------
# cwd-reset detection (covers READS)
# --------------------------------------------------------------------------------------


def test_relative_read_from_primary_after_worktree_pin_is_refused(checkouts):
    """The 247-vs-578 truncation: a relative read that silently hits the primary copy."""
    primary, worktree = checkouts

    # First call from the worktree pins the session.
    first = _run_hook(_payload("Bash", {"command": "ls"}, cwd=worktree, session_id="s-pin"))
    assert first.returncode == 0, first.stderr

    # cwd is then reset to the primary checkout; a relative read is now stale.
    second = _run_hook(
        _payload("Bash", {"command": "wc -l CLAUDE.md"}, cwd=primary, session_id="s-pin")
    )
    assert second.returncode == 2
    assert "WORKING-DIRECTORY RESET DETECTED (GH #1082)" in second.stderr
    assert str(worktree) in second.stderr


def test_absolute_read_from_primary_after_pin_is_allowed(checkouts):
    primary, worktree = checkouts
    _run_hook(_payload("Bash", {"command": "ls"}, cwd=worktree, session_id="s-abs"))
    result = _run_hook(
        _payload(
            "Bash", {"command": f"wc -l {worktree}/CLAUDE.md"}, cwd=primary, session_id="s-abs"
        )
    )
    assert result.returncode == 0, result.stderr


def test_unpinned_session_may_read_the_primary_relatively(checkouts):
    """The PM session works in the primary checkout and must keep reading it."""
    primary, _ = checkouts
    result = _run_hook(
        _payload("Bash", {"command": "wc -l CLAUDE.md"}, cwd=primary, session_id="s-pm")
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------------------
# unit-level checks on the pieces
# --------------------------------------------------------------------------------------


def test_find_primary_checkout_from_worktree(checkouts):
    primary, worktree = checkouts
    assert guard.find_primary_checkout(worktree) == primary
    assert guard.find_primary_checkout(primary) == primary


def test_git_dir_and_worktrees_dir_are_not_protected(checkouts):
    primary, worktree = checkouts
    assert not guard.is_protected(primary / ".git" / "HEAD", primary)
    assert not guard.is_protected(worktree / "CLAUDE.md", primary)
    assert guard.is_protected(primary / "CLAUDE.md", primary)


def test_guard_reuses_the_shim_root_finder():
    """Requirement of GH #1082: reuse #1070's mechanism, do not duplicate it."""
    shim = guard._load_shim_helpers()
    assert shim.find_repo_root is not None
    assert GUARD_SOURCE.read_text().count("def find_repo_root") == 1


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("echo a > out.txt", ["out.txt"]),
        ("echo a >> out.txt", ["out.txt"]),
        ('python3 -c "print(1 > 2)"', []),
        ("grep -c '>' file", []),
        ("cmd 2>&1", []),
    ],
)
def test_redirect_detection_is_quote_aware(command, expected):
    assert guard._redirect_targets(command) == expected
