"""Tests for tools/check_promotion_log.py (GH #1135/#1242)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_promotion_log.py"
_spec = importlib.util.spec_from_file_location("check_promotion_log", _TOOL)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

NO_CHANGE_DIFF = "+## 2026-10-04 — ETHUSDT/basic — NO CHANGE (incumbent retained)\n+body\n"
PROMOTION_DIFF = "+## 2026-10-04 — ETHUSDT/basic\n+- **New version**: `v2`\n"
LATEST = "src/ml/models/ETHUSDT/price/latest"


class TestViolations:
    def test_no_change_entry_with_new_latest_is_flagged(self):
        assert check.violations(NO_CHANGE_DIFF, [LATEST, "docs/x.md"]) == [
            f"{LATEST} changed in a PR that records a NO CHANGE promotion"
        ]

    def test_no_change_entry_without_latest_change_passes(self):
        assert (
            check.violations(NO_CHANGE_DIFF, ["src/ml/models/ETHUSDT/price/v2/metadata.json"]) == []
        )

    def test_other_symbol_may_be_promoted_next_to_a_no_change_entry(self):
        assert check.violations(NO_CHANGE_DIFF, ["src/ml/models/BTCUSDT/basic/latest"]) == []

    def test_model_type_is_not_matched(self):
        """#1130 shipped a price/latest under an entry about basic."""
        assert check.violations(NO_CHANGE_DIFF, [LATEST]) != []

    def test_heading_without_a_symbol_covers_every_symbol(self):
        diff = "+## 2026-10-04 — weekly retrain — NO CHANGE\n"
        assert check.violations(diff, ["src/ml/models/BTCUSDT/basic/latest"]) != []

    def test_real_promotion_may_move_latest(self):
        assert check.violations(PROMOTION_DIFF, [LATEST]) == []

    def test_unchanged_log_may_move_latest(self):
        assert check.violations("", [LATEST]) == []

    def test_quoting_an_old_no_change_heading_is_not_an_addition(self):
        assert check.violations(" ## old — NO CHANGE\n-## x\n", [LATEST]) == []

    @pytest.mark.parametrize(
        "path", ["src/ml/models/BTCUSDT/basic/latest", "models/ETHUSDT/sentiment/latest"]
    )
    def test_latest_pointer_paths_match(self, path):
        assert check.moved_latest_pointers([path]) == [path]

    @pytest.mark.parametrize(
        "path", ["src/ml/models/BTCUSDT/basic/latest/metadata.json", "docs/latest", "latest.md"]
    )
    def test_other_paths_do_not_match(self, path):
        assert check.moved_latest_pointers([path]) == []


class TestMainAgainstARealRepo:
    @staticmethod
    def _git(repo: Path, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
            check=True,
            capture_output=True,
            env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        )

    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        # check.main shells out to git too: scrub hook-injected GIT_* (GH #1148)
        for key in [k for k in os.environ if k.startswith("GIT_")]:
            monkeypatch.delenv(key)
        self._git(tmp_path, "init", "-q", "-b", "base")
        (tmp_path / "docs/research").mkdir(parents=True)
        (tmp_path / "docs/research/model-promotions.md").write_text("# Model promotions\n")
        self._git(tmp_path, "add", "-A")
        self._git(tmp_path, "commit", "-q", "-m", "base")
        self._git(tmp_path, "checkout", "-q", "-b", "pr")
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def _commit_pr(self, repo: Path, *, with_latest: bool) -> None:
        log = repo / "docs/research/model-promotions.md"
        log.write_text(log.read_text() + "\n## 2026-10-04 — ETHUSDT/basic — NO CHANGE\n")
        if with_latest:
            type_dir = repo / "src/ml/models/ETHUSDT/price"
            (type_dir / "v2").mkdir(parents=True)
            (type_dir / "latest").symlink_to("v2")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", "pr")

    def test_fails_when_pr_adds_latest_next_to_no_change_entry(self, repo, capsys):
        self._commit_pr(repo, with_latest=True)

        assert check.main(["--base", "base"]) == 1
        assert LATEST in capsys.readouterr().err

    def test_passes_when_pr_only_documents_the_decision(self, repo):
        self._commit_pr(repo, with_latest=False)

        assert check.main(["--base", "base"]) == 0
