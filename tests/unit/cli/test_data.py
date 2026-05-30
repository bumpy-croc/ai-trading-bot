"""Tests for atb data commands."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from cli.commands.data import _download, _prefill, _preload_offline


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

        # Create mock data with proper index
        mock_df = pd.DataFrame(
            {
                "open": [40000, 40050],
                "high": [40100, 40150],
                "low": [39900, 39950],
                "close": [40050, 40100],
                "volume": [100, 150],
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
                ],
                name="timestamp",
            ),
        )

        # Act
        with (
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch("cli.commands.data.Path") as mock_path,
        ):

            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = mock_df
            mock_provider.close.return_value = None
            mock_create_provider.return_value = mock_provider

            mock_path_instance = Mock()
            mock_path_instance.mkdir.return_value = None
            mock_path_instance.__truediv__ = Mock(return_value=Path("/tmp/test.csv"))
            mock_path.return_value = mock_path_instance

            with patch("pandas.DataFrame.to_csv"):
                result = _download(args)

            # Assert
            assert result == 0
            mock_create_provider.assert_called_once_with(provider_type="auto")
            mock_provider.get_historical_data.assert_called_once()
            mock_provider.close.assert_called_once()

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

        # Act - Mock provider to return None (no data)
        with patch(
            "src.data_providers.provider_factory.create_data_provider"
        ) as mock_create_provider:
            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = None
            mock_provider.close.return_value = None
            mock_create_provider.return_value = mock_provider

            result = _download(args)

            # Assert
            assert result == 1
            mock_provider.close.assert_called_once()

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

        # Create mock data with proper index
        mock_df = pd.DataFrame(
            {
                "open": [40000],
                "high": [40100],
                "low": [39900],
                "close": [40050],
                "volume": [100],
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2024, 1, 1, tzinfo=UTC),
                ],
                name="timestamp",
            ),
        )

        # Act
        with (
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch("cli.commands.data.Path") as mock_path,
        ):

            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = mock_df
            mock_provider.close.return_value = None
            mock_create_provider.return_value = mock_provider

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


class TestDataPreloadOffline:
    """Tests for the data preload-offline command."""

    def test_preload_offline_uses_utc_aware_bounds(self, tmp_path):
        """Test that preload uses UTC-aware datetime ranges."""
        # Arrange
        args = argparse.Namespace(
            symbols=["BTCUSDT"],
            timeframes=["1h"],
            years_back=1,
            cache_dir=str(tmp_path),
            force_refresh=True,
            test_offline=False,
        )

        class DummyCachedProvider:
            def __init__(self, provider, cache_dir=None, cache_ttl_hours=None):
                self.provider = provider

            def get_historical_data(self, symbol, timeframe, start, end):
                assert start.tzinfo is UTC
                assert end.tzinfo is UTC
                return pd.DataFrame({"close": [50000]}, index=pd.DatetimeIndex([start]))

        class DummyTqdm:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, count):
                return None

        # Act
        with (
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider", DummyCachedProvider
            ),
            patch("tqdm.tqdm", DummyTqdm),
        ):
            mock_provider.return_value = Mock()
            result = _preload_offline(args)

        # Assert
        assert result == 0


class TestSafePickleLoad:
    """Regression tests for the restricted unpickler used by cache-manager."""

    @pytest.mark.fast
    @pytest.mark.parametrize("protocol", [4, 5])
    def test_round_trips_realistic_cached_frames(self, protocol):
        """Legitimate cached DataFrames/Series must load unchanged.

        Covers a datetime/tz index (the OHLCV cache shape), naive index,
        RangeIndex, integer index, mixed dtypes, and a Series.
        """
        import io
        import pickle

        from cli.commands.data import _safe_pickle_load

        idx = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.arange(5.0),
                "close": np.arange(5.0),
                "volume": np.arange(5),
            },
            index=idx,
        )
        cases = [
            df,
            df.tz_localize(None),
            df.reset_index(drop=True),
            df["close"],
            pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30])),
            pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5], "c": ["p", "q"], "d": [True, False]}),
        ]
        for obj in cases:
            loaded = _safe_pickle_load(io.BytesIO(pickle.dumps(obj, protocol=protocol)))
            if isinstance(obj, pd.Series):
                pd.testing.assert_series_equal(loaded, obj)
            else:
                pd.testing.assert_frame_equal(loaded, obj)

    @pytest.mark.fast
    def test_blocks_os_system_gadget(self):
        import io
        import pickle

        from cli.commands.data import _safe_pickle_load

        class _Evil:
            def __reduce__(self):
                import os

                return (os.system, ("echo pwned",))

        with pytest.raises(pickle.UnpicklingError):
            _safe_pickle_load(io.BytesIO(pickle.dumps(_Evil())))

    @pytest.mark.fast
    def test_blocks_builtins_eval_gadget(self):
        import builtins
        import io
        import pickle

        from cli.commands.data import _safe_pickle_load

        class _Evil:
            def __reduce__(self):
                return (builtins.eval, ("__import__('os').system('id')",))

        with pytest.raises(pickle.UnpicklingError):
            _safe_pickle_load(io.BytesIO(pickle.dumps(_Evil())))
