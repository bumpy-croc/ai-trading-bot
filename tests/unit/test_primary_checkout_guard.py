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
import time
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


@pytest.fixture
def single_clone(tmp_path: Path) -> Path:
    """An ordinary clone with no worktrees — what every contributor and CI runner has."""
    clone = tmp_path / "fresh-clone"
    clone.mkdir()
    _make_checkout(clone)
    _git(clone, "init", "-q", "-b", "main")
    _git(clone, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(clone, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return clone


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


def _pin(worktree: Path, session_id: str = "s1") -> str:
    """Put the session in agent context: one tool call whose cwd is inside the worktree.

    This is what a dispatched agent does on its first call, and it is the signal the guard
    keys on — an unpinned session (fresh clone, PM daemon in the primary) is never guarded.
    """
    result = _run_hook(_payload("Bash", {"command": "ls"}, cwd=worktree, session_id=session_id))
    assert result.returncode == 0, result.stderr
    return session_id


# --------------------------------------------------------------------------------------
# an agent-context write to the primary is refused
# --------------------------------------------------------------------------------------


def test_edit_tool_write_into_primary_is_refused(checkouts):
    primary, worktree = checkouts
    _pin(worktree)
    result = _run_hook(_payload("Edit", {"file_path": str(primary / "CLAUDE.md")}, cwd=primary))
    assert result.returncode == 2
    assert "ATB PRIMARY CHECKOUT WRITE REFUSED (GH #1082)" in result.stderr
    assert str(primary / "CLAUDE.md") in result.stderr
    assert guard.ALLOW_ENV in result.stderr


def test_relative_edit_after_cwd_reset_is_refused(checkouts):
    """The exact #1082 shape: a relative path resolving against the reset cwd."""
    primary, worktree = checkouts
    _pin(worktree)
    result = _run_hook(_payload("Edit", {"file_path": "CLAUDE.md"}, cwd=primary))
    assert result.returncode == 2
    assert str(primary / "CLAUDE.md") in result.stderr


def test_sed_in_place_into_primary_is_refused(checkouts):
    primary, worktree = checkouts
    _pin(worktree)
    result = _run_hook(_payload("Bash", {"command": "sed -i '' 's/a/b/' CLAUDE.md"}, cwd=primary))
    assert result.returncode == 2
    assert "WRITE REFUSED" in result.stderr


def test_redirect_into_primary_is_refused(checkouts):
    primary, worktree = checkouts
    _pin(worktree)
    result = _run_hook(_payload("Bash", {"command": "echo hi > CLAUDE.md"}, cwd=primary))
    assert result.returncode == 2


def test_absolute_write_into_primary_from_a_worktree_is_refused(checkouts):
    """Not just a cwd accident — an explicit cross-checkout write is refused too."""
    primary, worktree = checkouts
    result = _run_hook(_payload("Bash", {"command": f"rm -f {primary}/CLAUDE.md"}, cwd=worktree))
    assert result.returncode == 2


def test_git_working_tree_mutation_in_primary_is_refused(checkouts):
    primary, worktree = checkouts
    _pin(worktree)
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
    primary, worktree = checkouts
    _pin(worktree)
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
# P0 regressions: who must NEVER be guarded
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Edit", {"file_path": "CLAUDE.md"}),
        ("Write", {"file_path": "src/new.py"}),
        ("Bash", {"command": "touch newfile"}),
        ("Bash", {"command": "git commit -am wip"}),
        ("Bash", {"command": "sed -i '' 's/a/b/' CLAUDE.md"}),
    ],
)
def test_single_clone_with_no_worktrees_is_unguarded(single_clone, tool, tool_input):
    """A fresh clone is structurally 'the primary' but must not write-lock itself.

    `.claude/settings.json` is checked in, so guarding writes unconditionally would reach
    every contributor and every Claude Code Web session — which clones into a single
    checkout with no worktrees, and would get a read-only repo plus a banner naming a path
    that does not exist on their machine.
    """
    result = _run_hook(_payload(tool, tool_input, cwd=single_clone, session_id="contributor"))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Bash", {"command": "echo '- entry' >> .claude/state/log.md"}),
        ("Write", {"file_path": ".claude/state/incidents/x.md"}),
        ("Edit", {"file_path": ".claude/state/log.md"}),
    ],
)
def test_pm_daemon_writes_its_own_state_in_the_primary(checkouts, tool, tool_input):
    """The daemon runs in the primary by design and must append to the tracked state record."""
    primary, _ = checkouts
    (primary / ".claude" / "state" / "incidents").mkdir(parents=True)
    (primary / ".claude" / "state" / "log.md").write_text("# log\n")
    result = _run_hook(_payload(tool, tool_input, cwd=primary, session_id="pm-daemon"))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------------------
# false-positive regressions (session in agent context)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "chmod +x /tmp/foo.sh",
        "chmod 755 /tmp/foo.sh",
        "OUT=/tmp/o.txt; echo hi > $OUT",
        'echo hi > "$LOG"',
        "echo hi > ${DIR}/x",
        'echo x > "logs/x.log"',
        "cat <<'EOF'\nif a > b then\nEOF",
        'gh pr create --body "$(cat <<EOF\n- a > b\nEOF\n)"',
        "cd $SOME_WORKTREE && rm -f CLAUDE.md",
    ],
)
def test_non_path_and_unresolvable_operands_do_not_block(checkouts, command):
    """A token we cannot resolve must never be guessed as primary-relative."""
    primary, worktree = checkouts
    _pin(worktree)
    result = _run_hook(_payload("Bash", {"command": command}, cwd=primary))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------------------
# false-negative regressions
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "sed -i -e s/a/b/ CLAUDE.md",
        "sed -i '' -e 's/a/b/' CLAUDE.md",
        "sed --in-place=.bak 's/a/b/' CLAUDE.md",
        "rm -- -weird-file",
    ],
)
def test_write_shapes_that_used_to_slip_through(checkouts, command):
    primary, worktree = checkouts
    _pin(worktree)
    (primary / "-weird-file").touch()
    result = _run_hook(_payload("Bash", {"command": command}, cwd=primary))
    assert result.returncode == 2, command


def test_bsd_sed_banner_names_the_file_not_the_script(checkouts):
    """The banner's whole job is legibility; naming `<primary>/s/a/b` defeats it."""
    primary, worktree = checkouts
    _pin(worktree)
    result = _run_hook(_payload("Bash", {"command": "sed -i '' 's/a/b/' CLAUDE.md"}, cwd=primary))
    assert result.returncode == 2
    target_row = next(line for line in result.stderr.splitlines() if "would write to" in line)
    assert target_row.endswith(str(primary / "CLAUDE.md"))
    assert "s/a/b" not in target_row


# --------------------------------------------------------------------------------------
# pin and override lifecycle
# --------------------------------------------------------------------------------------


def test_stale_pin_naming_a_deleted_worktree_is_discarded(checkouts, _isolated_home):
    """The nightly pruner deletes worktrees; a pin outliving one must not block reads."""
    primary, worktree = checkouts
    _pin(worktree, "s-stale")
    pin_file = _isolated_home / ".cache" / "atb-primary-guard" / "s-stale.worktree"
    pin_file.write_text(str(primary / ".claude" / "worktrees" / "deleted-one"))
    result = _run_hook(
        _payload("Bash", {"command": "wc -l CLAUDE.md"}, cwd=primary, session_id="s-stale")
    )
    assert result.returncode == 0, result.stderr
    assert not pin_file.exists()


def test_expired_pin_is_discarded(checkouts, _isolated_home):
    primary, worktree = checkouts
    _pin(worktree, "s-old")
    pin_file = _isolated_home / ".cache" / "atb-primary-guard" / "s-old.worktree"
    old = time.time() - (guard.PIN_MAX_AGE_SECONDS + 60)
    os.utime(pin_file, (old, old))
    result = _run_hook(
        _payload("Edit", {"file_path": "CLAUDE.md"}, cwd=primary, session_id="s-old")
    )
    assert result.returncode == 0, result.stderr


def test_fresh_sentinel_overrides_and_expired_one_does_not(checkouts, _isolated_home):
    primary, worktree = checkouts
    _pin(worktree)
    sentinel = _isolated_home / ".claude" / "atb-allow-primary-write"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    payload = _payload("Edit", {"file_path": str(primary / "CLAUDE.md")}, cwd=primary)
    assert _run_hook(payload).returncode == 0

    stale = time.time() - (guard.SENTINEL_MAX_AGE_SECONDS + 60)
    os.utime(sentinel, (stale, stale))
    assert _run_hook(payload).returncode == 2


def test_git_ignore_probe_fails_open(tmp_path):
    """Every other failure path allows the call; this one used to be the exception."""
    missing = tmp_path / "not-a-repo"
    assert guard._git_ignores(missing, missing / "x") is True


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
