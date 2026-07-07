"""Unit tests for ML training pipeline data ingestion module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ml.training_pipeline.config import TrainingConfig, TrainingContext, TrainingPaths
from src.ml.training_pipeline.ingestion import (
    _load_price_data_file,
    download_price_data,
    load_sentiment_data,
    load_training_corpus,
)


def _make_ctx(tmp_path, start=None, end=None, timeframe="1h", symbol="BTCUSDT"):
    """Build a TrainingContext rooted in tmp_path."""
    config = TrainingConfig(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start or datetime(2024, 1, 1),
        end_date=end or datetime(2024, 1, 31),
    )
    paths = TrainingPaths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )
    return TrainingContext(config=config, paths=paths)


def _hourly_frame(start: str, end: str) -> pd.DataFrame:
    """Build an OHLCV frame with one row per hour over [start, end] (UTC)."""
    index = pd.date_range(start, end, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 1000.0,
        },
        index=index,
    )


def _patch_cache_provider(df_or_exc):
    """Patch the corpus loader's provider stack; returns the CachedDataProvider mock."""
    provider = MagicMock()
    if isinstance(df_or_exc, Exception):
        provider.get_historical_data.side_effect = df_or_exc
    else:
        provider.get_historical_data.return_value = df_or_exc
    return provider


@pytest.mark.fast
class TestLoadTrainingCorpus:
    """Tests for load_training_corpus (cache tier + single-source contract)."""

    def _run(self, ctx, df_or_exc):
        provider = _patch_cache_provider(df_or_exc)
        with (
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider",
                return_value=provider,
            ) as provider_cls,
            patch("src.data_providers.binance_provider.BinanceProvider") as binance_cls,
        ):
            result = load_training_corpus(ctx)
        return result, provider, provider_cls, binance_cls

    def test_full_coverage_returns_corpus(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00")

        # Act
        result, provider, _, _ = self._run(ctx, df)

        # Assert
        assert len(result) == len(df)
        assert result.index.min() == pd.Timestamp("2024-01-01 00:00", tz="UTC")
        assert result.index.max() == pd.Timestamp("2024-01-31 23:00", tz="UTC")
        provider.get_historical_data.assert_called_once()

    def test_wraps_binance_without_third_party_fallback(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00")

        # Act
        _, _, provider_cls, binance_cls = self._run(ctx, df)

        # Assert - cache wraps BinanceProvider directly (no FallbackProvider)
        binance_cls.assert_called_once_with()
        assert provider_cls.call_args.args[0] is binance_cls.return_value

    def test_queries_full_end_day(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00")

        # Act
        _, provider, _, _ = self._run(ctx, df)

        # Assert - end boundary is end-of-day, not midnight (matches ctx.end_iso)
        call = provider.get_historical_data.call_args
        assert call.args[2] == datetime(2024, 1, 1, tzinfo=UTC)
        assert call.args[3] == datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC)

    def test_intraday_listing_offset_on_start_day_accepted(self, tmp_path):
        # Arrange - ETHUSDT-style listing: first candle opens 04:00, not midnight
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 04:00", "2024-01-31 23:00")

        # Act
        result, _, _, _ = self._run(ctx, df)

        # Assert
        assert result.index.min() == pd.Timestamp("2024-01-01 04:00", tz="UTC")

    def test_start_a_calendar_day_late_rejected(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-02 00:00", "2024-01-31 23:00")

        # Act & Assert
        with pytest.raises(RuntimeError, match="does not cover"):
            self._run(ctx, df)

    def test_end_one_bar_short_accepted(self, tmp_path):
        # Arrange - candle index is open time: last bar of the day opens at 23:00
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00")

        # Act
        result, _, _, _ = self._run(ctx, df)

        # Assert
        assert result.index.max() == pd.Timestamp("2024-01-31 23:00", tz="UTC")

    def test_end_more_than_one_bar_short_rejected(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 21:00")

        # Act & Assert
        with pytest.raises(RuntimeError, match="does not cover"):
            self._run(ctx, df)

    def test_gap_in_middle_rejected(self, tmp_path):
        # Arrange - drop a full day of bars inside the range (real cache gap)
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00")
        gap_mask = (df.index < "2024-01-10") | (df.index >= "2024-01-12")
        df = df[gap_mask]

        # Act & Assert
        with pytest.raises(RuntimeError, match="expected bars"):
            self._run(ctx, df)

    def test_empty_result_fails_loudly_with_guidance(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)

        # Act & Assert
        with pytest.raises(RuntimeError, match="prefill-cache"):
            self._run(ctx, pd.DataFrame())

    def test_provider_error_fails_loudly_instead_of_fallback(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)

        # Act & Assert
        with pytest.raises(RuntimeError, match="prefill-cache"):
            self._run(ctx, ConnectionError("Binance unreachable"))

    def test_window_extending_to_now_clamps_end_boundary(self, tmp_path):
        # Arrange - end date is today: only bars up to the last closed candle exist
        now = datetime.now(UTC)
        start = (now - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
        ctx = _make_ctx(tmp_path, start=start, end=now)
        last_closed = pd.Timestamp(now).floor("1h")
        df = _hourly_frame(
            start.strftime("%Y-%m-%d %H:%M"),
            last_closed.tz_localize(None).strftime("%Y-%m-%d %H:%M"),
        )

        # Act
        result, _, _, _ = self._run(ctx, df)

        # Assert
        assert result.index.max() == last_closed

    def test_naive_index_from_provider_normalized_to_utc(self, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00").tz_localize(None)

        # Act
        result, _, _, _ = self._run(ctx, df)

        # Assert
        assert result.index.tz is not None
        assert len(result) == len(df)


@pytest.mark.fast
class TestDownloadPriceData:
    """Tests for download_price_data tier ordering."""

    @patch("src.ml.training_pipeline.ingestion.load_training_corpus")
    def test_uses_training_corpus_tier(self, mock_corpus, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        df = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00")
        mock_corpus.return_value = df

        # Act
        result = download_price_data(ctx)

        # Assert
        assert result is df
        mock_corpus.assert_called_once_with(ctx)

    @patch("src.ml.training_pipeline.ingestion.load_training_corpus")
    @patch("src.ml.training_pipeline.ingestion._load_price_data_file")
    @patch("src.ml.training_pipeline.ingestion._download_from_s3")
    def test_s3_uri_takes_priority_over_corpus_tier(
        self, mock_s3, mock_load, mock_corpus, tmp_path
    ):
        # Arrange
        ctx = _make_ctx(tmp_path)
        mock_s3.return_value = tmp_path / "data.csv"
        expected = _hourly_frame("2024-01-01 00:00", "2024-01-31 23:00")
        mock_load.return_value = expected

        # Act
        result = download_price_data(ctx, s3_data_uri="s3://bucket/data.csv")

        # Assert
        assert result is expected
        mock_corpus.assert_not_called()

    @patch("src.ml.training_pipeline.ingestion.load_training_corpus")
    def test_corpus_failure_propagates(self, mock_corpus, tmp_path):
        # Arrange
        ctx = _make_ctx(tmp_path)
        mock_corpus.side_effect = RuntimeError("Training corpus for BTCUSDT could not be loaded")

        # Act & Assert
        with pytest.raises(RuntimeError, match="could not be loaded"):
            download_price_data(ctx)


@pytest.mark.fast
class TestLoadPriceDataFile:
    """Tests for _load_price_data_file parsing behavior."""

    def test_parses_mixed_timestamp_formats(self, tmp_path):
        # Arrange - Binance's earliest kline archives mix on-the-hour and
        # sub-second-offset timestamps within one file (e.g. ETHUSDT 2018-02)
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2018-02-04 00:00:00,100,101,99,100,1000\n"
            "2018-02-04 01:28:14.8,100,101,99,100,1000\n"
            "2018-02-04 02:00:00,100,101,99,100,1000\n"
        )

        # Act
        result = _load_price_data_file(csv_file)

        # Assert
        assert len(result) == 3
        assert result.index[1] == pd.Timestamp("2018-02-04 01:28:14.8", tz="UTC")

    def test_naive_timestamps_become_utc(self, tmp_path):
        # Arrange
        csv_file = tmp_path / "naive.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n2024-01-01 00:00:00,100,101,99,100,1000\n"
        )

        # Act
        result = _load_price_data_file(csv_file)

        # Assert
        assert result.index.tz is not None
        assert result.index[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    def test_sorts_by_index(self, tmp_path):
        # Arrange
        csv_file = tmp_path / "unsorted.csv"
        csv_file.write_text(
            "timestamp,open,close\n"
            "2024-01-01 02:00:00,102,105\n"
            "2024-01-01 00:00:00,100,103\n"
            "2024-01-01 01:00:00,101,104\n"
        )

        # Act
        result = _load_price_data_file(csv_file)

        # Assert
        assert result.index[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        assert result.index[1] == pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
        assert result.index[2] == pd.Timestamp("2024-01-01 02:00:00", tz="UTC")

    def test_loads_feather_file(self, tmp_path):
        # Arrange
        feather_file = tmp_path / "data.feather"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
                "open": [100.0, 101.0, 102.0],
                "close": [103.0, 104.0, 105.0],
            }
        )
        df.to_feather(feather_file)

        # Act
        result = _load_price_data_file(feather_file)

        # Assert
        assert len(result) == 3
        assert result.index.name == "timestamp"


@pytest.mark.fast
class TestLoadSentimentData:
    """Test load_sentiment_data function."""

    @patch("src.ml.training_pipeline.ingestion.FearGreedProvider")
    def test_load_sentiment_data_success(self, mock_provider_class):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        ctx = TrainingContext(config=config)

        sentiment_df = pd.DataFrame(
            {
                "sentiment_score": [0.5, 0.6, 0.7],
                "sentiment_volume": [100, 110, 120],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )

        mock_provider = MagicMock()
        mock_provider.get_historical_sentiment.return_value = sentiment_df
        mock_provider_class.return_value = mock_provider

        # Act
        result = load_sentiment_data(ctx)

        # Assert
        assert result is not None
        assert len(result) == 3
        assert "sentiment_score" in result.columns
        mock_provider.get_historical_sentiment.assert_called_once_with("BTCUSDT", start, end)

    def test_load_sentiment_data_force_price_only(self):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            force_price_only=True,
        )
        ctx = TrainingContext(config=config)

        # Act
        result = load_sentiment_data(ctx)

        # Assert
        assert result is None

    @patch("src.ml.training_pipeline.ingestion.FearGreedProvider")
    def test_load_sentiment_data_provider_exception(self, mock_provider_class):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        ctx = TrainingContext(config=config)

        mock_provider = MagicMock()
        mock_provider.get_historical_sentiment.side_effect = ValueError("API error")
        mock_provider_class.return_value = mock_provider

        # Act
        result = load_sentiment_data(ctx)

        # Assert - should return None on error, not raise
        assert result is None

    @patch("src.ml.training_pipeline.ingestion.FearGreedProvider")
    def test_load_sentiment_data_network_exception(self, mock_provider_class):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        ctx = TrainingContext(config=config)

        mock_provider = MagicMock()
        mock_provider.get_historical_sentiment.side_effect = ConnectionError("Network error")
        mock_provider_class.return_value = mock_provider

        # Act
        result = load_sentiment_data(ctx)

        # Assert - should return None on error, not raise
        assert result is None
