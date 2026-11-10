"""Cache load integrity + provenance pinning for data downloads (#982)."""

import argparse
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cli.commands.data import _audit_cache, _download, _prefill
from src.data_providers.cached_data_provider import CachedDataProvider, count_index_defects


def _frame(timestamps, closes):
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True), name="timestamp")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx
    )


DUP_UNSORTED = _frame(
    ["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00"],
    [3.0, 1.0, 2.0, 2.5],
)


@pytest.mark.fast
class TestCacheLoadIntegrity:
    def test_load_dedups_keeping_last_and_sorts(self, tmp_path):
        path = tmp_path / "x.parquet"
        DUP_UNSORTED.to_parquet(path)
        provider = CachedDataProvider(Mock(), cache_dir=str(tmp_path))

        with patch("src.data_providers.cached_data_provider.logger") as log:
            loaded = provider._load_from_cache(str(path))

        assert "integrity check" in log.warning.call_args.args[0]
        assert loaded.index.is_unique and loaded.index.is_monotonic_increasing
        assert list(loaded["close"]) == [1.0, 2.5, 3.0]
        # repair is persisted, so a fresh read is already clean
        assert count_index_defects(pd.read_parquet(path)) == (0, False)

    def test_clean_file_is_untouched(self, tmp_path):
        path = tmp_path / "ok.parquet"
        clean = _frame(["2024-01-01 00:00", "2024-01-01 01:00"], [1.0, 2.0])
        clean.to_parquet(path)
        provider = CachedDataProvider(Mock(), cache_dir=str(tmp_path))

        with patch("src.data_providers.cached_data_provider.logger") as log:
            loaded = provider._load_from_cache(str(path))

        assert list(loaded["close"]) == [1.0, 2.0]
        log.warning.assert_not_called()

    def test_count_index_defects(self):
        assert count_index_defects(DUP_UNSORTED) == (1, True)


@pytest.mark.fast
class TestCacheAudit:
    def test_flags_defective_file_and_passes_clean_dir(self, tmp_path, capsys):
        _frame(["2024-01-01 00:00", "2024-01-01 01:00"], [1.0, 2.0]).to_parquet(
            tmp_path / "ok.parquet"
        )
        assert _audit_cache(str(tmp_path)) == 0

        DUP_UNSORTED.to_parquet(tmp_path / "bad.parquet")
        assert _audit_cache(str(tmp_path)) == 1
        assert "DEFECT bad.parquet" in capsys.readouterr().out

    def test_flags_non_datetime_index(self, tmp_path, capsys):
        pd.DataFrame({"close": [1.0, 2.0]}).to_parquet(tmp_path / "flat.parquet")
        assert _audit_cache(str(tmp_path)) == 1
        assert "not datetime" in capsys.readouterr().out


@pytest.mark.fast
class TestDownloadsPinBinance:
    def test_download_fails_loudly_on_binance_error_without_fallback(self):
        args = argparse.Namespace(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir="data",
            format="csv",
        )
        provider = Mock(spec=["get_historical_data"])  # BinanceProvider has no close()
        provider.get_historical_data.side_effect = ConnectionError("binance down")
        with patch(
            "src.data_providers.provider_factory.create_data_provider", return_value=provider
        ) as create:
            assert _download(args) == 1
        create.assert_called_once_with(provider_type="binance")

    def test_prefill_pins_binance(self, tmp_path):
        args = argparse.Namespace(
            symbols=["BTCUSDT"],
            timeframes=["1h"],
            years=1,
            start=None,
            end="2024-01-02",
            cache_dir=str(tmp_path),
            cache_ttl_hours=24,
        )
        with patch("src.data_providers.provider_factory.create_data_provider") as create:
            create.return_value.get_historical_data.return_value = pd.DataFrame()
            _prefill(args)
        create.assert_called_once_with(provider_type="binance")
