"""Tests for the model artifact retention policy."""

from pathlib import Path

import pytest

from src.ml.model_retention import (
    RegistryStateError,
    apply_plan,
    plan_cleanup,
    version_sort_key,
)

pytestmark = pytest.mark.fast


def make_version(registry: Path, symbol: str, mtype: str, version: str, payload: int = 100) -> Path:
    vdir = registry / symbol / mtype / version
    vdir.mkdir(parents=True)
    (vdir / "model.onnx").write_bytes(b"x" * payload)
    return vdir


def set_latest(registry: Path, symbol: str, mtype: str, version: str) -> None:
    (registry / symbol / mtype / "latest").symlink_to(version)


def candidate_keys(plan) -> set[str]:
    return {v.key for v in plan.candidates}


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    return root


class TestVersionSortKey:
    def test_orders_chronologically_across_formats(self):
        ids = [
            "2025-09-16_legacy",
            "2025-09-17_1h_v1",
            "2025-12-23_20h_v1",
            "2025-12-23_20h_v2",
            "2026-07-05_10h30m00s_v1",
        ]
        keys = [version_sort_key(i) for i in ids]
        assert keys == sorted(keys)

    def test_undated_name_has_no_key(self):
        assert version_sort_key("experimental") is None


class TestPlanCleanup:
    def test_never_selects_latest_target_even_when_oldest(self, registry):
        for v in ("2025-01-01_10h_v1", "2025-02-01_10h_v1", "2025-03-01_10h_v1"):
            make_version(registry, "BTCUSDT", "basic", v)
        set_latest(registry, "BTCUSDT", "basic", "2025-01-01_10h_v1")

        for keep in (0, 1, 2):
            plan = plan_cleanup(registry, keep_newest=keep)
            assert "BTCUSDT/basic/2025-01-01_10h_v1" not in candidate_keys(plan)
            assert "target of 'latest' symlink" in plan.protected["BTCUSDT/basic/2025-01-01_10h_v1"]

    def test_keeps_newest_n_per_symbol_and_type(self, registry):
        for v in ("2025-01-01_10h_v1", "2025-02-01_10h_v1", "2025-03-01_10h_v1"):
            make_version(registry, "BTCUSDT", "basic", v)
        make_version(registry, "BTCUSDT", "sentiment", "2025-01-01_10h_v1")
        make_version(registry, "ETHUSDT", "basic", "2025-01-01_10h_v1")

        plan = plan_cleanup(registry, keep_newest=2)

        assert candidate_keys(plan) == {"BTCUSDT/basic/2025-01-01_10h_v1"}

    def test_same_day_versions_order_by_sequence(self, registry):
        for v in ("2025-12-23_20h_v1", "2025-12-23_20h_v2", "2025-12-23_20h_v3"):
            make_version(registry, "BTCUSDT", "sentiment", v)

        plan = plan_cleanup(registry, keep_newest=1)

        assert candidate_keys(plan) == {
            "BTCUSDT/sentiment/2025-12-23_20h_v1",
            "BTCUSDT/sentiment/2025-12-23_20h_v2",
        }

    def test_referenced_version_is_protected(self, registry, tmp_path):
        make_version(registry, "ETHUSDT", "basic", "2026-07-04_22h_v1")
        make_version(registry, "ETHUSDT", "basic", "2026-08-01_10h_v1")
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "model-promotions.md").write_text("Deployed `2026-07-04_22h_v1` on 07-04.")

        plan = plan_cleanup(registry, keep_newest=0, reference_paths=[notes])

        assert candidate_keys(plan) == {"ETHUSDT/basic/2026-08-01_10h_v1"}
        assert any(
            r.startswith("referenced in") for r in plan.protected["ETHUSDT/basic/2026-07-04_22h_v1"]
        )

    def test_reference_match_respects_version_boundaries(self, registry, tmp_path):
        make_version(registry, "BTCUSDT", "basic", "2025-01-01_10h_v1")
        make_version(registry, "BTCUSDT", "basic", "2025-01-01_10h_v12")
        cfg = tmp_path / "config.json"
        cfg.write_text('{"model_version": "2025-01-01_10h_v12"}')

        plan = plan_cleanup(registry, keep_newest=0, reference_paths=[cfg])

        assert candidate_keys(plan) == {"BTCUSDT/basic/2025-01-01_10h_v1"}

    def test_undated_version_is_protected(self, registry):
        make_version(registry, "BTCUSDT", "basic", "hand-built")

        plan = plan_cleanup(registry, keep_newest=0)

        assert plan.candidates == []
        assert "BTCUSDT/basic/hand-built" in plan.protected

    def test_bytes_freed_sums_candidates(self, registry):
        make_version(registry, "BTCUSDT", "basic", "2025-01-01_10h_v1", payload=300)
        make_version(registry, "BTCUSDT", "basic", "2025-02-01_10h_v1", payload=500)
        make_version(registry, "BTCUSDT", "basic", "2025-03-01_10h_v1", payload=900)

        plan = plan_cleanup(registry, keep_newest=1)

        assert plan.bytes_freed == 800

    def test_dangling_latest_aborts(self, registry):
        make_version(registry, "BTCUSDT", "basic", "2025-01-01_10h_v1")
        set_latest(registry, "BTCUSDT", "basic", "2099-gone")

        with pytest.raises(RegistryStateError, match="Dangling"):
            plan_cleanup(registry, keep_newest=0)

    def test_latest_directory_aborts(self, registry):
        make_version(registry, "BTCUSDT", "basic", "latest")

        with pytest.raises(RegistryStateError, match="not a symlink"):
            plan_cleanup(registry, keep_newest=0)

    def test_latest_pointing_outside_type_dir_aborts(self, registry):
        make_version(registry, "BTCUSDT", "sentiment", "2025-01-01_10h_v1")
        make_version(registry, "BTCUSDT", "basic", "2025-01-01_10h_v1")
        (registry / "BTCUSDT" / "basic" / "latest").symlink_to("../sentiment/2025-01-01_10h_v1")

        with pytest.raises(RegistryStateError, match="outside"):
            plan_cleanup(registry, keep_newest=0)

    def test_type_without_latest_still_keeps_newest(self, registry):
        make_version(registry, "ETHUSDT", "price", "2026-09-06_07h21m47s_v1")

        plan = plan_cleanup(registry, keep_newest=1)

        assert plan.candidates == []

    def test_negative_keep_rejected(self, registry):
        with pytest.raises(ValueError):
            plan_cleanup(registry, keep_newest=-1)

    def test_missing_registry_aborts(self, tmp_path):
        with pytest.raises(RegistryStateError, match="does not exist"):
            plan_cleanup(tmp_path / "nope")

    def test_planning_deletes_nothing(self, registry):
        for v in ("2025-01-01_10h_v1", "2025-02-01_10h_v1"):
            make_version(registry, "BTCUSDT", "basic", v)

        plan_cleanup(registry, keep_newest=0)

        assert len(list((registry / "BTCUSDT" / "basic").iterdir())) == 2


class TestApplyPlan:
    def test_removes_candidates_and_leaves_latest_intact(self, registry):
        for v in ("2025-01-01_10h_v1", "2025-02-01_10h_v1", "2025-03-01_10h_v1"):
            make_version(registry, "BTCUSDT", "basic", v)
        set_latest(registry, "BTCUSDT", "basic", "2025-02-01_10h_v1")
        plan = plan_cleanup(registry, keep_newest=1)

        removed = apply_plan(registry, plan)

        type_dir = registry / "BTCUSDT" / "basic"
        assert [v.version for v in removed] == ["2025-01-01_10h_v1"]
        assert sorted(p.name for p in type_dir.iterdir()) == [
            "2025-02-01_10h_v1",
            "2025-03-01_10h_v1",
            "latest",
        ]
        assert (type_dir / "latest").is_symlink()
        assert (type_dir / "latest").resolve().name == "2025-02-01_10h_v1"

    def test_refuses_candidate_that_became_a_symlink(self, registry, tmp_path):
        for v in ("2025-01-01_10h_v1", "2025-02-01_10h_v1"):
            make_version(registry, "BTCUSDT", "basic", v)
        plan = plan_cleanup(registry, keep_newest=1)
        victim = plan.candidates[0].path
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("precious")
        for child in victim.iterdir():
            child.unlink()
        victim.rmdir()
        victim.symlink_to(outside)

        with pytest.raises(RegistryStateError, match="not a plain version directory"):
            apply_plan(registry, plan)

        assert (outside / "keep.txt").exists()
