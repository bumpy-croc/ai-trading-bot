"""Unit tests for the `atb train cloud*` CLI helpers."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.commands.train_cloud import (
    DEFAULT_TRAINING_DAYS,
    _handle_cloud_list,
    _handle_cloud_promote,
    _handle_cloud_status,
    _parse_job_name,
    _resolve_date_range,
    _sync_job_artifacts,
)
from src.ml.cloud.exceptions import ModelPromotionError
from src.ml.cloud.providers.base import TrainingJobStatus


@pytest.mark.fast
class TestResolveDateRange:
    """Tests for _resolve_date_range (date-window resolution for cloud training)."""

    def test_days_only_ends_now(self) -> None:
        before = datetime.now(UTC)
        start, end = _resolve_date_range(days=30, start_date_str=None, end_date_str=None)
        after = datetime.now(UTC)

        assert before <= end <= after
        assert (end - start).days == 30

    def test_defaults_to_365_days(self) -> None:
        start, end = _resolve_date_range(days=None, start_date_str=None, end_date_str=None)

        assert (end - start).days == DEFAULT_TRAINING_DAYS

    def test_explicit_start_and_end(self) -> None:
        start, end = _resolve_date_range(
            days=None, start_date_str="2026-05-01", end_date_str="2026-06-01"
        )

        assert start == datetime(2026, 5, 1, tzinfo=UTC)
        assert end == datetime(2026, 6, 1, tzinfo=UTC)

    def test_start_date_only_ends_now(self) -> None:
        before = datetime.now(UTC)
        start, end = _resolve_date_range(days=None, start_date_str="2026-05-01", end_date_str=None)
        after = datetime.now(UTC)

        assert start == datetime(2026, 5, 1, tzinfo=UTC)
        assert before <= end <= after

    def test_end_date_with_days_gives_fixed_cutoff_window(self) -> None:
        start, end = _resolve_date_range(days=90, start_date_str=None, end_date_str="2026-06-01")

        assert end == datetime(2026, 6, 1, tzinfo=UTC)
        assert (end - start).days == 90

    def test_days_and_start_date_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_date_range(days=30, start_date_str="2026-05-01", end_date_str=None)

    def test_start_after_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="before"):
            _resolve_date_range(days=None, start_date_str="2026-06-01", end_date_str="2026-05-01")

    def test_start_equal_to_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="before"):
            _resolve_date_range(days=None, start_date_str="2026-05-01", end_date_str="2026-05-01")

    def test_invalid_date_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid date"):
            _resolve_date_range(days=None, start_date_str="not-a-date", end_date_str=None)

    def test_timezone_aware_iso_input_preserved(self) -> None:
        start, end = _resolve_date_range(
            days=None,
            start_date_str="2026-05-01T00:00:00+00:00",
            end_date_str="2026-06-01T12:00:00+00:00",
        )

        assert start == datetime(2026, 5, 1, tzinfo=UTC)
        assert end == datetime(2026, 6, 1, 12, tzinfo=UTC)


@pytest.mark.fast
class TestParseJobName:
    """Tests for _parse_job_name (symbol/timeframe recovery from job names)."""

    def test_parses_standard_job_name(self) -> None:
        assert _parse_job_name("atb-btcusdt-1h-20260704-215649") == ("BTCUSDT", "1h")

    def test_parses_eth_job_name(self) -> None:
        assert _parse_job_name("atb-ethusdt-4h-20260120-005532") == ("ETHUSDT", "4h")

    def test_rejects_non_atb_job_name(self) -> None:
        assert _parse_job_name("custom-job-name") is None

    def test_rejects_short_job_name(self) -> None:
        assert _parse_job_name("atb-btcusdt") is None

    def test_parses_full_sagemaker_arn(self) -> None:
        arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:training-job/"
            "atb-btcusdt-1h-20260704-215649"
        )
        assert _parse_job_name(arn) == ("BTCUSDT", "1h")


def _ns(args: list[str]) -> argparse.Namespace:
    """Build the REMAINDER-style namespace the handlers receive."""
    return argparse.Namespace(args=args)


@pytest.mark.fast
class TestHandleCloudPromote:
    """Tests for the cloud-promote CLI handler."""

    def test_promotes_with_defaults(self, tmp_path: Path) -> None:
        with patch(
            "src.ml.cloud.promotion.promote_model_version",
            return_value=tmp_path / "BTCUSDT" / "basic" / "v1",
        ) as mock_promote:
            rc = _handle_cloud_promote(_ns(["BTCUSDT", "2026-07-05_10h00m00s_v1"]))

        assert rc == 0
        mock_promote.assert_called_once_with(
            symbol="BTCUSDT",
            version_id="2026-07-05_10h00m00s_v1",
            source_type="price",
            target_type="basic",
            set_latest=False,
        )

    def test_set_latest_flag_is_threaded(self, tmp_path: Path) -> None:
        with patch(
            "src.ml.cloud.promotion.promote_model_version",
            return_value=tmp_path / "BTCUSDT" / "basic" / "v1",
        ) as mock_promote:
            rc = _handle_cloud_promote(_ns(["BTCUSDT", "v1", "--set-latest"]))

        assert rc == 0
        assert mock_promote.call_args.kwargs["set_latest"] is True

    def test_custom_source_and_target(self, tmp_path: Path) -> None:
        with patch(
            "src.ml.cloud.promotion.promote_model_version",
            return_value=tmp_path / "BTCUSDT" / "scratch" / "v1",
        ) as mock_promote:
            rc = _handle_cloud_promote(_ns(["BTCUSDT", "v1", "--from", "price", "--to", "scratch"]))

        assert rc == 0
        assert mock_promote.call_args.kwargs["source_type"] == "price"
        assert mock_promote.call_args.kwargs["target_type"] == "scratch"

    def test_promotion_error_returns_1(self) -> None:
        with patch(
            "src.ml.cloud.promotion.promote_model_version",
            side_effect=ModelPromotionError("Target version already exists"),
        ):
            rc = _handle_cloud_promote(_ns(["BTCUSDT", "v1"]))

        assert rc == 1


@pytest.mark.fast
class TestSyncJobArtifacts:
    """Tests for the cloud-status --sync round-trip helper."""

    def test_syncs_using_symbol_from_job_name(self, tmp_path: Path) -> None:
        provider = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.sync_artifacts.return_value = tmp_path / "synced"

        with (
            patch.dict("os.environ", {"SAGEMAKER_S3_BUCKET": "test-bucket"}),
            patch(
                "src.ml.cloud.orchestrator.CloudTrainingOrchestrator",
                return_value=mock_orchestrator,
            ) as mock_cls,
        ):
            rc = _sync_job_artifacts(
                "atb-btcusdt-1h-20260705-181756",
                "atb-btcusdt-1h-20260705-181756",
                None,
                provider,
            )

        assert rc == 0
        mock_orchestrator.sync_artifacts.assert_called_once_with("atb-btcusdt-1h-20260705-181756")
        cloud_config = mock_cls.call_args.args[0]
        assert cloud_config.training_config.symbol == "BTCUSDT"
        assert cloud_config.training_config.timeframe == "1h"

    def test_symbol_override_used_for_custom_job_names(self, tmp_path: Path) -> None:
        provider = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.sync_artifacts.return_value = tmp_path / "synced"

        with (
            patch.dict("os.environ", {"SAGEMAKER_S3_BUCKET": "test-bucket"}),
            patch(
                "src.ml.cloud.orchestrator.CloudTrainingOrchestrator",
                return_value=mock_orchestrator,
            ) as mock_cls,
        ):
            rc = _sync_job_artifacts("custom-job", "custom-job", "ethusdt", provider)

        assert rc == 0
        assert mock_cls.call_args.args[0].training_config.symbol == "ETHUSDT"

    def test_underivable_symbol_returns_1(self) -> None:
        rc = _sync_job_artifacts("custom-job", "custom-job", None, MagicMock())

        assert rc == 1

    def test_sync_failure_returns_1(self) -> None:
        provider = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.sync_artifacts.side_effect = RuntimeError("download failed")

        with (
            patch.dict("os.environ", {"SAGEMAKER_S3_BUCKET": "test-bucket"}),
            patch(
                "src.ml.cloud.orchestrator.CloudTrainingOrchestrator",
                return_value=mock_orchestrator,
            ),
        ):
            rc = _sync_job_artifacts(
                "atb-btcusdt-1h-20260705-181756",
                "atb-btcusdt-1h-20260705-181756",
                None,
                provider,
            )

        assert rc == 1


@pytest.mark.fast
class TestHandleCloudStatusSync:
    """Tests for --sync wiring in the cloud-status handler."""

    def _provider_with_status(self, status: TrainingJobStatus) -> MagicMock:
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get_job_status.return_value = status
        return provider

    def test_sync_dispatches_for_completed_job(self) -> None:
        status = TrainingJobStatus(
            job_name="atb-btcusdt-1h-20260705-181756",
            status="Completed",
            output_s3_path="s3://bucket/out/model.tar.gz",
        )
        provider = self._provider_with_status(status)

        with (
            patch("src.ml.cloud.providers.get_provider", return_value=provider),
            patch("cli.commands.train_cloud._sync_job_artifacts", return_value=0) as mock_sync,
        ):
            rc = _handle_cloud_status(_ns(["atb-btcusdt-1h-20260705-181756", "--sync"]))

        assert rc == 0
        mock_sync.assert_called_once_with(
            "atb-btcusdt-1h-20260705-181756",
            "atb-btcusdt-1h-20260705-181756",
            None,
            provider,
        )

    def test_sync_refused_for_incomplete_job(self) -> None:
        status = TrainingJobStatus(job_name="job-1", status="InProgress")
        provider = self._provider_with_status(status)

        with (
            patch("src.ml.cloud.providers.get_provider", return_value=provider),
            patch("cli.commands.train_cloud._sync_job_artifacts") as mock_sync,
        ):
            rc = _handle_cloud_status(_ns(["job-1", "--sync"]))

        assert rc == 1
        mock_sync.assert_not_called()


@pytest.mark.fast
class TestHandleCloudList:
    """Tests for the cloud-list handler."""

    def test_lists_jobs_from_s3(self) -> None:
        mock_manager = MagicMock()
        mock_manager.list_training_jobs.return_value = ["atb-btcusdt-1h-20260704-215649"]

        with (
            patch.dict("os.environ", {"SAGEMAKER_S3_BUCKET": "test-bucket"}),
            patch(
                "src.ml.cloud.artifacts.s3_manager.S3ArtifactManager",
                return_value=mock_manager,
            ),
        ):
            rc = _handle_cloud_list(_ns(["BTCUSDT"]))

        assert rc == 0
        mock_manager.list_training_jobs.assert_called_once_with(symbol="BTCUSDT")

    def test_missing_bucket_env_returns_1(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            rc = _handle_cloud_list(_ns([]))

        assert rc == 1
