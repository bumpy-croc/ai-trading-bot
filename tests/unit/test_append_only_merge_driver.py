"""Behavioural tests for the append-only merge driver (GH #1079).

These drive real ``git merge`` runs in throwaway repositories rather than calling the driver's
functions directly: the defect being fixed is that git silently ignores an unregistered driver,
so a test that bypasses git's own dispatch would prove nothing about the case that matters.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
DRIVER = TOOLS / "merge_append_only.py"
INSTALLER = TOOLS / "install_merge_drivers.py"

LOG_PATH = ".claude/state/log.md"
PREAMBLE = "# Log\n\nAppend-only. Newest last.\n\n"
BASE_ENTRY = "## 2026-08-01 09:00 · note · base\nThe ancestor entry.\n\n"


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


def write(repo: Path, rel: str, text: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "Test")
    write(repo, ".gitattributes", f"{LOG_PATH} merge=append-only\n")
    return repo


def register(repo: Path) -> None:
    """Register the driver the way `make merge-drivers` does, but with an absolute path.

    The shipped registration is relative (it must follow the worktree); these throwaway repos
    have no `tools/`, so point at the real script.
    """
    git(
        repo,
        "config",
        "merge.append-only.driver",
        f"{sys.executable} {DRIVER} %O %A %B %L %P",
    )
    git(repo, "config", "merge.append-only.name", "test")


def seed(repo: Path, path: str = LOG_PATH, body: str = PREAMBLE + BASE_ENTRY) -> None:
    write(repo, path, body)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "seed")


def branch_commit(repo: Path, branch: str, path: str, body: str, *, from_ref: str = "main") -> None:
    git(repo, "checkout", "-q", "-b", branch, from_ref)
    write(repo, path, body)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", branch)


OURS_ENTRY = "## 2026-08-10 12:00 · note · ours\nOurs appended this.\n\n"
THEIRS_ENTRY = "## 2026-08-05 08:00 · note · theirs\nTheirs appended this.\n\n"


def _two_appends(repo: Path) -> subprocess.CompletedProcess[str]:
    seed(repo)
    branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + OURS_ENTRY)
    branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + THEIRS_ENTRY)
    git(repo, "checkout", "-q", "ours")
    return git(repo, "merge", "theirs", "-m", "merge", check=False)


def test_concurrent_appends_merge_without_conflict_and_keep_both(repo: Path) -> None:
    register(repo)
    merged = _two_appends(repo)
    assert merged.returncode == 0, merged.stdout + merged.stderr

    text = (repo / LOG_PATH).read_text()
    assert "Ours appended this." in text
    assert "Theirs appended this." in text
    assert "<<<<<<<" not in text


def test_ancestor_content_is_not_duplicated(repo: Path) -> None:
    register(repo)
    _two_appends(repo)
    text = (repo / LOG_PATH).read_text()
    assert text.count("The ancestor entry.") == 1
    assert text.count("# Log") == 1


def test_appended_entries_land_in_timestamp_order(repo: Path) -> None:
    register(repo)
    _two_appends(repo)
    text = (repo / LOG_PATH).read_text()
    # Theirs is dated 08-05, ours 08-10: chronological order, not merge-side order.
    assert text.index("Theirs appended this.") < text.index("Ours appended this.")
    assert text.index("The ancestor entry.") < text.index("Theirs appended this.")


def test_identical_entry_on_both_sides_appears_once(repo: Path) -> None:
    register(repo)
    seed(repo)
    same = "## 2026-08-09 07:00 · note · both\nSame text on both sides.\n\n"
    branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + same)
    branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + same)
    git(repo, "checkout", "-q", "ours")
    assert git(repo, "merge", "theirs", "-m", "m", check=False).returncode == 0
    assert (repo / LOG_PATH).read_text().count("Same text on both sides.") == 1


def test_editing_the_same_existing_line_still_conflicts(repo: Path) -> None:
    """A union driver would ship both halves of an edit. This one must not."""
    register(repo)
    seed(repo)
    branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY.replace("ancestor", "OURS-EDIT"))
    branch_commit(
        repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY.replace("ancestor", "THEIRS-EDIT")
    )
    git(repo, "checkout", "-q", "ours")
    merged = git(repo, "merge", "theirs", "-m", "m", check=False)

    assert merged.returncode != 0, "an edit/edit collision must surface, not union-merge"
    assert "<<<<<<<" in (repo / LOG_PATH).read_text()


def test_edit_racing_a_deletion_still_conflicts(repo: Path) -> None:
    register(repo)
    seed(repo)
    branch_commit(repo, "ours", LOG_PATH, PREAMBLE)  # entry removed
    branch_commit(
        repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY.replace("ancestor", "THEIRS-EDIT")
    )
    git(repo, "checkout", "-q", "ours")
    assert git(repo, "merge", "theirs", "-m", "m", check=False).returncode != 0


def test_deletion_is_honoured_and_a_concurrent_append_survives_it(repo: Path) -> None:
    """The weekly-retro `AGENDA.md` clear (GH #1090): the wipe wins, the new item lives."""
    register(repo)
    seed(repo)
    branch_commit(repo, "clear", LOG_PATH, PREAMBLE)
    branch_commit(repo, "append", LOG_PATH, PREAMBLE + BASE_ENTRY + OURS_ENTRY)
    git(repo, "checkout", "-q", "clear")
    merged = git(repo, "merge", "append", "-m", "m", check=False)

    assert merged.returncode == 0, merged.stdout + merged.stderr
    text = (repo / LOG_PATH).read_text()
    assert "The ancestor entry." not in text, "the deliberate clear must not be undone"
    assert "Ours appended this." in text, "the concurrent item must not be dropped"


def test_unregistered_driver_leaves_a_real_conflict(repo: Path) -> None:
    """The defect this whole change guards against: silence, not an error."""
    merged = _two_appends(repo)  # note: register() deliberately not called
    assert merged.returncode != 0
    assert "<<<<<<<" in (repo / LOG_PATH).read_text()


def test_unlisted_path_is_not_union_merged(repo: Path) -> None:
    """.gitattributes alone must not be enough to union-merge a file."""
    register(repo)
    other = "docs/notes.md"
    write(repo, ".gitattributes", f"{other} merge=append-only\n")
    seed(repo, other, PREAMBLE + BASE_ENTRY)
    branch_commit(repo, "ours", other, PREAMBLE + BASE_ENTRY + OURS_ENTRY)
    branch_commit(repo, "theirs", other, PREAMBLE + BASE_ENTRY + THEIRS_ENTRY)
    git(repo, "checkout", "-q", "ours")
    merged = git(repo, "merge", "theirs", "-m", "m", check=False)

    assert merged.returncode != 0, "an unlisted path must fall back to a normal conflict"
    assert "<<<<<<<" in (repo / other).read_text()


class TestInstaller:
    def _run(self, repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(INSTALLER), *args],
            cwd=repo,
            capture_output=True,
            text=True,
        )

    def test_check_reports_an_unregistered_driver_as_drift(self, repo: Path) -> None:
        seed(repo)
        proc = self._run(repo, "--check")
        assert proc.returncode == 1
        assert "DRIFT" in proc.stderr
        assert "merge.append-only.driver" in proc.stderr

    def test_install_then_check_is_clean_and_idempotent(self, repo: Path) -> None:
        seed(repo)
        assert self._run(repo).returncode == 0
        assert self._run(repo, "--check").returncode == 0
        assert self._run(repo).returncode == 0
        assert self._run(repo, "--check").returncode == 0

    def test_check_reports_a_stale_registration_as_drift(self, repo: Path) -> None:
        seed(repo)
        assert self._run(repo).returncode == 0
        git(repo, "config", "merge.append-only.driver", "cat %A")
        proc = self._run(repo, "--check")
        assert proc.returncode == 1
        assert "DRIFT" in proc.stderr

    def test_registration_matches_what_gitattributes_asks_for(self, repo: Path) -> None:
        """Both halves must name the same driver, or the entry is inert."""
        attributes = (Path(__file__).resolve().parents[2] / ".gitattributes").read_text()
        assert "merge=append-only" in attributes
        seed(repo)
        assert self._run(repo).returncode == 0
        value = git(repo, "config", "--get", "merge.append-only.driver").stdout
        assert "tools/merge_append_only.py" in value


def test_same_entry_header_with_different_bodies_conflicts(repo: Path) -> None:
    """Two appends that claim to be the same entry are a disagreement, not two entries."""
    register(repo)
    seed(repo)
    header = "## 2026-08-09 07:00 · note · both\n"
    branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + header + "Ours body.\n\n")
    branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + header + "Theirs body.\n\n")
    git(repo, "checkout", "-q", "ours")
    assert git(repo, "merge", "theirs", "-m", "m", check=False).returncode != 0


def test_preamble_edits_still_conflict(repo: Path) -> None:
    """Content before the first entry is merged as one block, not union-merged."""
    register(repo)
    seed(repo)
    branch_commit(repo, "ours", LOG_PATH, PREAMBLE.replace("Newest last", "OURS") + BASE_ENTRY)
    branch_commit(repo, "theirs", LOG_PATH, PREAMBLE.replace("Newest last", "THEIRS") + BASE_ENTRY)
    git(repo, "checkout", "-q", "ours")
    assert git(repo, "merge", "theirs", "-m", "m", check=False).returncode != 0


def test_real_log_md_tail_merges_cleanly(repo: Path) -> None:
    """Exercise the true file: two agents appending to the repository's own log.md."""
    real = (Path(__file__).resolve().parents[2] / LOG_PATH).read_text(encoding="utf-8")
    register(repo)
    seed(repo, LOG_PATH, real)
    branch_commit(repo, "ours", LOG_PATH, real + "\n" + OURS_ENTRY)
    branch_commit(repo, "theirs", LOG_PATH, real + "\n" + THEIRS_ENTRY)
    git(repo, "checkout", "-q", "ours")
    merged = git(repo, "merge", "theirs", "-m", "m", check=False)

    assert merged.returncode == 0, merged.stdout + merged.stderr
    text = (repo / LOG_PATH).read_text()
    assert "Ours appended this." in text and "Theirs appended this." in text
    # Every ancestor line survives, exactly once each in aggregate.
    assert len(text) >= len(real)
    for line in real.splitlines():
        if line.strip():
            assert line in text


AGENDA_PATH = ".claude/skills/weekly-retro/AGENDA.md"
AGENDA_HEADER = "# Weekly Retro — Agenda\n\n## Items\n\n"
AGENDA_ITEM = (
    "- **2026-08-13 (#1036)** — an item nobody actioned.\n  Continued on a second line.\n\n"
)


def test_agenda_clear_keeps_a_concurrently_appended_item(repo: Path) -> None:
    """GH #1090: the retro clears the agenda while a branch is still adding to it."""
    register(repo)
    write(repo, ".gitattributes", f"{AGENDA_PATH} merge=append-only\n")
    seed(repo, AGENDA_PATH, AGENDA_HEADER + AGENDA_ITEM)
    new_item = "- **2026-08-20 (#1084)** — noticed while the retro was running.\n\n"
    branch_commit(repo, "retro", AGENDA_PATH, AGENDA_HEADER)
    branch_commit(repo, "agent", AGENDA_PATH, AGENDA_HEADER + AGENDA_ITEM + new_item)
    git(repo, "checkout", "-q", "retro")
    merged = git(repo, "merge", "agent", "-m", "m", check=False)

    assert merged.returncode == 0, merged.stdout + merged.stderr
    text = (repo / AGENDA_PATH).read_text()
    assert "#1084" in text, "an item added during the retro must not be lost"
    assert "#1036" not in text, "the retro's deliberate clear must stand"


def test_reordered_entries_stay_separated_by_a_blank_line(repo: Path) -> None:
    """An entry appended without a trailing blank line must not butt against its new neighbour."""
    register(repo)
    seed(repo)
    late = "## 2026-08-10 12:00 · note · ours\nOurs appended this.\n"  # no trailing blank
    early = "## 2026-08-05 08:00 · note · theirs\nTheirs appended this.\n"
    branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + late)
    branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + early)
    git(repo, "checkout", "-q", "ours")
    assert git(repo, "merge", "theirs", "-m", "m", check=False).returncode == 0
    assert "\n\n## 2026-08-10" in (repo / LOG_PATH).read_text()
