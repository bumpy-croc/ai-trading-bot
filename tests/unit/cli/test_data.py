"""Tests for atb data commands."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cli.commands.data import _download, _prefill


class TestDataDownload:
    """Tests for the data download command."""

    def test_downloads_data_successfully(self):
        """Test that data download succeeds."""
        # Arrange
        args = argparse.Namespace(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir="data",
            format="csv",
        )

        mock_ohlcv = [
            [1704067200000, 40000, 40100, 39900, 40050, 100],
            [1704070800000, 40050, 40150, 39950, 40100, 150],
        ]

        # Act
        with (
            patch("cli.commands.data.ccxt") as mock_ccxt,
            patch("cli.commands.data.Path") as mock_path,
            patch("cli.commands.data.SymbolFactory.to_exchange_symbol") as mock_symbol_factory,
        ):

            mock_symbol_factory.return_value = "BTCUSDT"

            mock_binance = Mock()
            mock_binance.fetch_ohlcv.return_value = mock_ohlcv
            mock_ccxt.binance.return_value = mock_binance

            mock_path_instance = Mock()
            mock_path_instance.mkdir.return_value = None
            mock_path_instance.__truediv__ = Mock(return_value=Path("/tmp/test.csv"))
            mock_path.return_value = mock_path_instance

            with patch("pandas.DataFrame.to_csv"):
                result = _download(args)

            # Assert
            assert result == 0

    def test_returns_error_when_no_data_fetched(self):
        """Test that error is returned when no data is fetched."""
        # Arrange
        args = argparse.Namespace(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir="data",
            format="csv",
        )

        # Act
        with patch("cli.commands.data.ccxt") as mock_ccxt:
            mock_binance = Mock()
            mock_binance.fetch_ohlcv.return_value = []
            mock_ccxt.binance.return_value = mock_binance

            result = _download(args)

            # Assert
            assert result == 1

    def test_supports_feather_format(self):
        """Test that feather format is supported."""
        # Arrange
        args = argparse.Namespace(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir="data",
            format="feather",
        )

        mock_ohlcv = [
            [1704067200000, 40000, 40100, 39900, 40050, 100],
        ]

        # Act
        with (
            patch("cli.commands.data.ccxt") as mock_ccxt,
            patch("cli.commands.data.Path") as mock_path,
            patch("cli.commands.data.SymbolFactory.to_exchange_symbol") as mock_symbol_factory,
        ):

            mock_symbol_factory.return_value = "BTCUSDT"

            mock_binance = Mock()
            mock_binance.fetch_ohlcv.return_value = mock_ohlcv
            mock_ccxt.binance.return_value = mock_binance

            mock_path_instance = Mock()
            mock_path_instance.mkdir.return_value = None
            mock_path_instance.__truediv__ = Mock(return_value=Path("/tmp/test.feather"))
            mock_path.return_value = mock_path_instance

            with patch("pandas.DataFrame.to_feather"):
                result = _download(args)

            # Assert
            assert result == 0


class TestDataPrefill:
    """Tests for the data prefill-cache command."""

    def test_prefills_cache_successfully(self):
        """Test that cache prefill succeeds."""
        # Arrange
        args = argparse.Namespace(
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes=["1h", "4h"],
            years=2,
            start=None,
            end=None,
            cache_dir="/tmp/cache",
            cache_ttl_hours=24,
        )

        # Act
        with (
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
        ):

            mock_provider_instance = Mock()
            mock_provider.return_value = mock_provider_instance

            mock_cached = Mock()
            mock_cached.get_historical_data.return_value = pd.DataFrame(
                {"close": [50000]}, index=pd.DatetimeIndex([datetime.now(UTC)])
            )
            mock_cached_provider.return_value = mock_cached

            result = _prefill(args)

            # Assert
            assert result == 0
