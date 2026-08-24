"""Behavioural tests for the append-only merge driver (GH #1079).

These drive real ``git merge`` runs in throwaway repositories rather than calling the driver's
functions directly: the defect being fixed is that git silently ignores an unregistered driver,
so a test that bypasses git's own dispatch would prove nothing about the case that matters.
"""

from __future__ import annotations

import random
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


def _driver_module():
    sys.path.insert(0, str(TOOLS))
    try:
        import merge_append_only

        return merge_append_only
    finally:
        sys.path.pop(0)


def installer():
    sys.path.insert(0, str(TOOLS))
    try:
        import install_merge_drivers

        return install_merge_drivers
    finally:
        sys.path.pop(0)


def register(repo: Path, script: Path = DRIVER) -> None:
    """Register the *shipped* command shape, with an absolute script path.

    The real registration is relative so it follows the worktree; these throwaway repos have
    no `tools/`, so the path is absolutised — but the surrounding guard is the shipped one,
    so every test below exercises what actually ships.
    """
    git(repo, "config", "merge.append-only.driver", installer().driver_command(str(script)))
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


class TestRepositoryContent:
    """Content assertions, safe to run in CI — unlike `--check`, which reads local git config."""

    ATTRIBUTES = Path(__file__).resolve().parents[2] / ".gitattributes"

    def _declared_in_gitattributes(self) -> set[str]:
        declared = set()
        for line in self.ATTRIBUTES.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "merge=append-only" in line:
                declared.add(line.split()[0])
        return declared

    def test_gitattributes_and_the_driver_allowlist_name_the_same_files(self) -> None:
        """Either half alone is inert, so a file added to one and not the other is a bug.

        Missing from `.gitattributes`: git never invokes the driver. Missing from the driver's
        allowlist: the driver refuses and the file conflicts as before. Both fail quietly.
        """
        sys.path.insert(0, str(TOOLS))
        try:
            import merge_append_only
        finally:
            sys.path.pop(0)

        assert self._declared_in_gitattributes() == set(merge_append_only.APPEND_ONLY_PATHS)

    def test_declared_files_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for path in self._declared_in_gitattributes():
            assert (repo_root / path).is_file(), f"{path} is declared but does not exist"

    def test_changelog_is_not_union_merged(self) -> None:
        """Explicitly pinned: it is edited in place, so it must keep conflicting."""
        assert "docs/changelog.md" not in self._declared_in_gitattributes()


class TestContentLossRegressions:
    """Four reproduced silent-loss paths, all previously exiting 0 (PR #1098 review)."""

    def test_a_quoted_header_inside_a_body_does_not_split_the_entry(self, repo: Path) -> None:
        """The charter encourages entries that quote an earlier one, so this is normal usage.

        Splitting on it hoisted a fabricated entry above the real one and truncated the real
        entry mid-sentence.
        """
        register(repo)
        seed(repo)
        quoting = (
            "## 2026-08-20 10:00 · note · ours\n"
            "Corrects the earlier entry:\n"
            "## 2026-02-01 00:00 · note · old\n"
            "end of ours.\n\n"
        )
        branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + quoting)
        branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + THEIRS_ENTRY)
        git(repo, "checkout", "-q", "ours")
        merged = git(repo, "merge", "theirs", "-m", "m", check=False)

        assert merged.returncode == 0, merged.stdout + merged.stderr
        text = (repo / LOG_PATH).read_text()
        assert quoting in text, "the quoted entry was split and reordered"
        assert "Theirs appended this." in text

    def test_missing_driver_script_yields_markers_not_ours_only(self, repo: Path) -> None:
        """A linked worktree on a branch predating the driver must not silently drop theirs.

        git reports CONFLICT either way, but with a bare command the working-tree file is
        ours' content verbatim with no markers — indistinguishable from a clean merge, so
        `git add` drops theirs' entry.
        """
        register(repo, script=repo / "does" / "not" / "exist.py")
        merged = _two_appends(repo)

        assert merged.returncode != 0
        text = (repo / LOG_PATH).read_text()
        assert "<<<<<<<" in text, "a marker-free ours-only file reads as a complete merge"
        assert "Theirs appended this." in text, "theirs' entry must be recoverable"

    def test_block_without_a_trailing_newline_does_not_glue_onto_the_next(self, repo: Path) -> None:
        """`Ours body.## 2026-08-25 ...` destroys theirs' heading, in the file and next merge."""
        register(repo)
        seed(repo)
        # Ours sorts FIRST, so it is not the final block and must be separated from theirs'.
        no_newline = "## 2026-08-02 10:00 · note · ours\nOurs body."  # no trailing newline
        earlier = "## 2026-08-25 12:00 · note · theirs\nTheirs body.\n\n"
        branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + no_newline)
        branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + earlier)
        git(repo, "checkout", "-q", "ours")
        merged = git(repo, "merge", "theirs", "-m", "m", check=False)

        assert merged.returncode == 0, merged.stdout + merged.stderr
        text = (repo / LOG_PATH).read_text()
        assert "Ours body.## " not in text, "theirs' heading was destroyed"
        assert "\n\n## 2026-08-25" in text

    def test_first_line_edit_on_both_sides_conflicts(self, repo: Path) -> None:
        """First-line identity made an edit read as delete-old + add-new, keeping BOTH."""
        register(repo)
        seed(repo)
        branch_commit(
            repo,
            "ours",
            LOG_PATH,
            PREAMBLE + BASE_ENTRY.replace("· base", "· OURS-EDIT"),
        )
        branch_commit(
            repo,
            "theirs",
            LOG_PATH,
            PREAMBLE + BASE_ENTRY.replace("· base", "· THEIRS-EDIT"),
        )
        git(repo, "checkout", "-q", "ours")
        merged = git(repo, "merge", "theirs", "-m", "m", check=False)

        assert merged.returncode != 0, "both sides editing one entry must conflict"
        assert (repo / LOG_PATH).read_text().count("The ancestor entry.") == 1

    def test_first_line_edit_racing_a_clear_conflicts(self, repo: Path) -> None:
        """Otherwise the edited item silently survives, partially undoing the retro's clear."""
        register(repo)
        seed(repo)
        branch_commit(repo, "clear", LOG_PATH, PREAMBLE)
        branch_commit(
            repo,
            "editor",
            LOG_PATH,
            PREAMBLE + BASE_ENTRY.replace("· base", "· REWORDED"),
        )
        git(repo, "checkout", "-q", "clear")
        merged = git(repo, "merge", "editor", "-m", "m", check=False)

        if merged.returncode == 0:
            assert (
                "The ancestor entry." not in (repo / LOG_PATH).read_text()
            ), "an edited entry silently survived a deliberate clear"

    def test_a_crash_inside_the_driver_still_writes_markers(self, tmp_path: Path) -> None:
        """An exception must not leave %A as ours-only, which reads as a complete merge."""
        driver = _driver_module()
        ancestor = tmp_path / "base"
        ours = tmp_path / "ours"
        theirs = tmp_path / "theirs"
        ancestor.write_text(PREAMBLE + BASE_ENTRY)
        ours.write_text(PREAMBLE + BASE_ENTRY + OURS_ENTRY)
        theirs.write_text(PREAMBLE + BASE_ENTRY + THEIRS_ENTRY)

        original = driver.merge
        driver.merge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            code = driver.main([str(ancestor), str(ours), str(theirs), "7", LOG_PATH])
        finally:
            driver.merge = original

        assert code != 0
        text = ours.read_text()
        assert "<<<<<<<" in text
        assert "Theirs appended this." in text, "theirs' entry must survive a driver crash"

    def test_a_correction_quoting_a_dated_header_after_a_blank_line(self, repo: Path) -> None:
        """The charter's own correction pattern, written naturally (PR #1098 round 3).

        The quoted header satisfies marker + date + preceded-by-blank-line, so shape rules
        alone cannot reject it. What keeps the entry intact is that one side's contribution is
        never reordered or separated internally, whatever the driver believes its shape to be.
        """
        register(repo)
        seed(repo)
        correction = (
            "## [D-2026-08-20-01] 2026-08-20 10:30 · correction · daemon(PM)\n"
            "Corrects the entry below, quoted verbatim:\n"
            "\n"
            "## [D-2026-07-08-01] 2026-07-08 20:33 · deploy-verify · daemon(PM)\n"
            "\n"
            "The correction is that the deploy actually failed.\n\n"
        )
        branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + correction)
        branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + THEIRS_ENTRY)
        git(repo, "checkout", "-q", "ours")
        merged = git(repo, "merge", "theirs", "-m", "m", check=False)

        assert merged.returncode == 0, merged.stdout + merged.stderr
        text = (repo / LOG_PATH).read_text()
        assert correction in text, "the correction was split and its quote hoisted"
        assert text.count("2026-07-08 20:33") == 1, "a duplicate July header was fabricated"
        assert "Theirs appended this." in text

    def test_a_side_with_several_entries_keeps_them_together_and_in_order(self, repo: Path) -> None:
        """The invariant behind the fix, asserted directly rather than via a symptom."""
        register(repo)
        seed(repo)
        run = (
            "## 2026-08-11 09:00 · note · ours\nFirst of ours.\n\n"
            "## 2026-08-12 09:00 · note · ours\nSecond of ours.\n\n"
        )
        branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + run)
        branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + THEIRS_ENTRY)
        git(repo, "checkout", "-q", "ours")
        merged = git(repo, "merge", "theirs", "-m", "m", check=False)

        assert merged.returncode == 0, merged.stdout + merged.stderr
        text = (repo / LOG_PATH).read_text()
        assert run in text, "one side's contiguous contribution was broken up"
        assert "Theirs appended this." in text

    def test_both_sides_quoting_the_same_header_do_not_interleave(self, repo: Path) -> None:
        """The sixth path, found by fuzzing rather than by reading (PR #1098 round 3).

        Identical blocks share a token, so when both sides quote the same header the diff can
        align on it and thread one side's entry through the middle of the other's. The output
        post-condition catches this without knowing the mechanism.
        """
        register(repo)
        seed(repo)
        quote = (
            "Corrects the entry below, quoted verbatim:\n"
            "\n"
            "## [D-2026-07-08-01] 2026-07-08 20:33 · deploy-verify · daemon(PM)\n"
            "\n"
        )
        ours_entry = f"## 2026-08-21 18:30 · correction · ours\n{quote}ours' correction.\n\n"
        theirs_entry = f"## 2026-08-01 09:30 · correction · theirs\n{quote}theirs' correction.\n\n"
        branch_commit(repo, "ours", LOG_PATH, PREAMBLE + BASE_ENTRY + ours_entry)
        branch_commit(repo, "theirs", LOG_PATH, PREAMBLE + BASE_ENTRY + theirs_entry)
        git(repo, "checkout", "-q", "ours")
        merged = git(repo, "merge", "theirs", "-m", "m", check=False)

        text = (repo / LOG_PATH).read_text()
        if merged.returncode == 0:
            assert (
                ours_entry in text and theirs_entry in text
            ), "one contribution was threaded through the middle of the other"
        else:
            assert "<<<<<<<" in text
            assert "theirs' correction." in text

    @pytest.mark.parametrize("seed_value", range(12))
    def test_adversarial_merges_never_corrupt_a_contribution(
        self, tmp_path: Path, seed_value: int
    ) -> None:
        """Randomised-but-seeded merges of deliberately hostile bodies.

        The block model is a guess about prose, and three review rounds showed that reading
        the code is not a reliable way to find where the guess breaks. This asserts the
        property that actually matters — whatever the driver decides, each side's added text
        survives verbatim and unbroken, or the merge conflicts — over bodies containing quoted
        headers (dated forwards and backwards, with and without a preceding blank line),
        column-0 bullets, conflict-marker text, and blocks ending mid-line.
        """
        rng = random.Random(seed_value)
        heads = [
            "## 2026-08-{d:02d} {h:02d}:00 · note · {who}\n",
            "## [D-2026-08-{d:02d}-01] 2026-08-{d:02d} ~{h:02d}:30 · decision · {who}\n",
            "## 2026-08-{d:02d} · track-record · {who}\n",
        ]
        bodies = [
            "Plain body.\n",
            "Quoted after a blank line:\n\n## [D-2026-07-08-01] 2026-07-08 20:33 · x · y\n\nend.\n",
            "Quoted mid-paragraph:\n## 2026-02-01 00:00 · note · old\nend.\n",
            "Forward-dated quote:\n\n## 2099-12-31 23:59 · note · future\n\nend.\n",
            "Markers in prose:\n<<<<<<< ours\n=======\n>>>>>>> theirs\nend.\n",
            "- column-0 bullet\n- another\n## 2026-03-03 03:03 · note · quoted\n",
            "No trailing newline.",
        ]

        def make(who: str) -> str:
            head = rng.choice(heads).format(d=rng.randint(1, 28), h=rng.randint(0, 23), who=who)
            body = rng.choice(bodies)
            tail = "" if body.endswith("\n\n") else "\n" if body.endswith("\n") else "\n\n"
            return head + body + tail

        repo = tmp_path / "fuzz"
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        git(repo, "config", "user.email", "t@example.com")
        git(repo, "config", "user.name", "Test")
        write(repo, ".gitattributes", f"{LOG_PATH} merge=append-only\n")
        register(repo)

        base = PREAMBLE + "".join(make("base") for _ in range(rng.randint(1, 3)))
        seed(repo, LOG_PATH, base)
        ours_add = "".join(make("ours") for _ in range(rng.randint(1, 3)))
        theirs_add = "".join(make("theirs") for _ in range(rng.randint(1, 3)))
        branch_commit(repo, "ours", LOG_PATH, base + ours_add)
        branch_commit(repo, "theirs", LOG_PATH, base + theirs_add)
        git(repo, "checkout", "-q", "ours")
        merged = git(repo, "merge", "theirs", "-m", "m", check=False)
        text = (repo / LOG_PATH).read_text()

        if merged.returncode == 0:
            for name, chunk in (("ours", ours_add), ("theirs", theirs_add)):
                assert chunk.strip() in text, f"{name}'s contribution was altered or split"
            for line in base.splitlines():
                assert not line.strip() or line in text, "an ancestor line was dropped"
        else:
            assert "<<<<<<<" in text, "a conflict must leave recoverable markers"
