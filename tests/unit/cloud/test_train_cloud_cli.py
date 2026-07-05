"""Unit tests for the `atb train cloud*` CLI helpers."""

from datetime import UTC, datetime

import pytest

from cli.commands.train_cloud import (
    DEFAULT_TRAINING_DAYS,
    _parse_job_name,
    _resolve_date_range,
)


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
