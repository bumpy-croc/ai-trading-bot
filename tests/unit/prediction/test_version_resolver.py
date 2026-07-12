"""Tests for point-in-time model version resolution (GH #988).

The resolver answers "which version WAS latest at date X?" from bundle
metadata timestamps so a backtest spanning a model-promotion boundary can
pin the model that was actually live, instead of silently comparing
against whatever `latest` resolves to today.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.prediction.models.exceptions import ModelNotAvailableError
from src.prediction.models.version_resolver import (
    VersionRecord,
    list_version_records,
    promotion_segments,
    resolve_version_as_of,
)

SYMBOL = "TESTUSDT"
MODEL_TYPE = "basic"

V1 = "2026-01-01_1h_v1"
V2 = "2026-03-01_1h_v1"
V3 = "2026-06-01_1h_v1"

T1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
T2 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
T3 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _write_bundle(
    registry: Path,
    version_id: str,
    created_at: str | None,
    *,
    symbol: str = SYMBOL,
    model_type: str = MODEL_TYPE,
) -> Path:
    bundle_dir = registry / symbol / model_type / version_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.onnx").write_bytes(b"dummy")
    metadata: dict[str, object] = {
        "symbol": symbol,
        "model_type": model_type,
        "timeframe": "1h",
        "version_id": version_id,
    }
    if created_at is not None:
        metadata["created_at"] = created_at
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata))
    return bundle_dir


@pytest.fixture
def three_version_registry(tmp_path: Path) -> Path:
    """Synthetic registry with three versions promoted at T1 < T2 < T3."""
    registry = tmp_path / "models"
    _write_bundle(registry, V1, T1.isoformat())
    _write_bundle(registry, V2, T2.isoformat())
    v3_dir = _write_bundle(registry, V3, T3.isoformat())
    latest = registry / SYMBOL / MODEL_TYPE / "latest"
    latest.symlink_to(v3_dir, target_is_directory=True)
    return registry


@pytest.mark.fast
class TestListVersionRecords:
    def test_lists_all_versions_sorted_by_effective_at(self, three_version_registry: Path):
        records = list_version_records(three_version_registry, SYMBOL, MODEL_TYPE)

        assert [r.version_id for r in records] == [V1, V2, V3]
        assert [r.effective_at for r in records] == [T1, T2, T3]

    def test_ignores_latest_symlink_and_stray_files(self, three_version_registry: Path):
        (three_version_registry / SYMBOL / MODEL_TYPE / "notes.txt").write_text("stray")

        records = list_version_records(three_version_registry, SYMBOL, MODEL_TYPE)

        assert [r.version_id for r in records] == [V1, V2, V3]

    def test_missing_created_at_falls_back_to_version_id_date(self, tmp_path: Path):
        registry = tmp_path / "models"
        _write_bundle(registry, "2026-02-15_1h_v1", None)

        records = list_version_records(registry, SYMBOL, MODEL_TYPE)

        assert len(records) == 1
        assert records[0].effective_at == datetime(2026, 2, 15, tzinfo=UTC)
        assert records[0].source == "version_id"

    def test_naive_created_at_treated_as_utc(self, tmp_path: Path):
        registry = tmp_path / "models"
        _write_bundle(registry, "2026-02-15_1h_v1", "2026-02-15T09:30:00")

        records = list_version_records(registry, SYMBOL, MODEL_TYPE)

        assert records[0].effective_at == datetime(2026, 2, 15, 9, 30, tzinfo=UTC)
        assert records[0].source == "metadata"

    def test_undateable_version_dir_is_skipped(self, tmp_path: Path):
        registry = tmp_path / "models"
        _write_bundle(registry, "2026-02-15_1h_v1", None)
        undateable = registry / SYMBOL / MODEL_TYPE / "experimental"
        undateable.mkdir(parents=True)
        (undateable / "metadata.json").write_text(json.dumps({"symbol": SYMBOL}))

        records = list_version_records(registry, SYMBOL, MODEL_TYPE)

        assert [r.version_id for r in records] == ["2026-02-15_1h_v1"]

    def test_missing_model_type_dir_returns_empty(self, tmp_path: Path):
        registry = tmp_path / "models"
        registry.mkdir()

        assert list_version_records(registry, SYMBOL, MODEL_TYPE) == []

    def test_tie_on_effective_at_orders_by_version_id(self, tmp_path: Path):
        registry = tmp_path / "models"
        same_ts = "2026-02-15T00:00:00+00:00"
        _write_bundle(registry, "2026-02-15_1h_v2", same_ts)
        _write_bundle(registry, "2026-02-15_1h_v1", same_ts)

        records = list_version_records(registry, SYMBOL, MODEL_TYPE)

        assert [r.version_id for r in records] == ["2026-02-15_1h_v1", "2026-02-15_1h_v2"]


@pytest.mark.fast
class TestResolveVersionAsOf:
    def test_as_of_before_first_version_raises(self, three_version_registry: Path):
        with pytest.raises(ModelNotAvailableError) as excinfo:
            resolve_version_as_of(
                three_version_registry, SYMBOL, MODEL_TYPE, datetime(2025, 12, 1, tzinfo=UTC)
            )

        assert V1 in str(excinfo.value)

    def test_as_of_between_versions_resolves_earlier(self, three_version_registry: Path):
        record = resolve_version_as_of(
            three_version_registry, SYMBOL, MODEL_TYPE, datetime(2026, 2, 15, tzinfo=UTC)
        )
        assert record.version_id == V1

        record = resolve_version_as_of(
            three_version_registry, SYMBOL, MODEL_TYPE, datetime(2026, 4, 1, tzinfo=UTC)
        )
        assert record.version_id == V2

    def test_as_of_after_last_version_resolves_latest(self, three_version_registry: Path):
        record = resolve_version_as_of(
            three_version_registry, SYMBOL, MODEL_TYPE, datetime(2026, 7, 1, tzinfo=UTC)
        )
        assert record.version_id == V3

    def test_as_of_exactly_at_promotion_is_inclusive(self, three_version_registry: Path):
        record = resolve_version_as_of(three_version_registry, SYMBOL, MODEL_TYPE, T2)
        assert record.version_id == V2

    def test_naive_as_of_treated_as_utc(self, three_version_registry: Path):
        record = resolve_version_as_of(
            three_version_registry, SYMBOL, MODEL_TYPE, datetime(2026, 2, 15)
        )
        assert record.version_id == V1

    def test_empty_registry_raises_with_clear_message(self, tmp_path: Path):
        registry = tmp_path / "models"
        registry.mkdir()

        with pytest.raises(ModelNotAvailableError) as excinfo:
            resolve_version_as_of(
                registry, SYMBOL, MODEL_TYPE, datetime(2026, 2, 15, tzinfo=UTC)
            )

        assert SYMBOL in str(excinfo.value)
        assert MODEL_TYPE in str(excinfo.value)


@pytest.mark.fast
class TestPromotionSegments:
    def _records(self) -> list[VersionRecord]:
        return [
            VersionRecord(version_id=V1, effective_at=T1, source="metadata"),
            VersionRecord(version_id=V2, effective_at=T2, source="metadata"),
            VersionRecord(version_id=V3, effective_at=T3, source="metadata"),
        ]

    def test_window_within_single_version_yields_one_segment(self):
        start = datetime(2026, 1, 10, tzinfo=UTC)
        end = datetime(2026, 2, 10, tzinfo=UTC)

        segments = promotion_segments(self._records(), start, end)

        assert len(segments) == 1
        assert segments[0].version_id == V1
        assert segments[0].start == start
        assert segments[0].end == end

    def test_window_spanning_one_boundary_yields_two_segments(self):
        start = datetime(2026, 2, 1, tzinfo=UTC)
        end = datetime(2026, 4, 1, tzinfo=UTC)

        segments = promotion_segments(self._records(), start, end)

        assert [(s.version_id, s.start, s.end) for s in segments] == [
            (V1, start, T2),
            (V2, T2, end),
        ]

    def test_window_spanning_all_boundaries_yields_all_segments(self):
        start = datetime(2025, 12, 1, tzinfo=UTC)
        end = datetime(2026, 7, 1, tzinfo=UTC)

        segments = promotion_segments(self._records(), start, end)

        assert [s.version_id for s in segments] == [None, V1, V2, V3]
        assert segments[0].start == start
        assert segments[0].end == T1
        assert segments[-1].end == end

    def test_window_before_any_version_yields_none_segment(self):
        start = datetime(2025, 10, 1, tzinfo=UTC)
        end = datetime(2025, 11, 1, tzinfo=UTC)

        segments = promotion_segments(self._records(), start, end)

        assert len(segments) == 1
        assert segments[0].version_id is None

    def test_promotion_exactly_at_window_end_does_not_split(self):
        start = datetime(2026, 2, 1, tzinfo=UTC)

        segments = promotion_segments(self._records(), start, T2)

        assert len(segments) == 1
        assert segments[0].version_id == V1

    def test_no_records_yields_single_none_segment(self):
        start = datetime(2026, 2, 1, tzinfo=UTC)
        end = datetime(2026, 3, 1, tzinfo=UTC)

        segments = promotion_segments([], start, end)

        assert len(segments) == 1
        assert segments[0].version_id is None
