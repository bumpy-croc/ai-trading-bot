"""Unit tests for the shared atomic 'latest' symlink helper."""

from pathlib import Path

import pytest

from src.ml.cloud.artifacts.latest_link import update_latest_symlink


@pytest.mark.fast
class TestUpdateLatestSymlink:
    """Tests for update_latest_symlink atomicity semantics."""

    def test_creates_link_when_absent(self, tmp_path: Path) -> None:
        (tmp_path / "v1").mkdir()

        update_latest_symlink(tmp_path, "v1")

        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == (tmp_path / "v1").resolve()

    def test_replaces_existing_link(self, tmp_path: Path) -> None:
        (tmp_path / "v1").mkdir()
        (tmp_path / "v2").mkdir()
        (tmp_path / "latest").symlink_to("v1")

        update_latest_symlink(tmp_path, "v2")

        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == (tmp_path / "v2").resolve()

    def test_no_temp_link_left_behind(self, tmp_path: Path) -> None:
        (tmp_path / "v1").mkdir()

        update_latest_symlink(tmp_path, "v1")

        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".latest.")]
        assert leftovers == []

    def test_cleans_stale_temp_link(self, tmp_path: Path) -> None:
        (tmp_path / "v1").mkdir()
        # Stale temp link from an interrupted earlier update
        (tmp_path / ".latest.v1.tmp").symlink_to("v1")

        update_latest_symlink(tmp_path, "v1")

        assert (tmp_path / "latest").is_symlink()
        assert not (tmp_path / ".latest.v1.tmp").is_symlink()

    def test_failure_cleans_temp_and_raises(self, tmp_path: Path) -> None:
        # Missing type_dir makes symlink creation fail
        missing_dir = tmp_path / "does-not-exist"

        with pytest.raises(OSError):
            update_latest_symlink(missing_dir, "v1")
