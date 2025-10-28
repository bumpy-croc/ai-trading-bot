"""Unit tests for ML training pipeline data ingestion module."""

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.ml.training_pipeline.config import TrainingConfig, TrainingContext, TrainingPaths
from src.ml.training_pipeline.ingestion import (
    _resolve_latest_file,
    download_price_data,
    load_sentiment_data,
)


@pytest.mark.fast
class TestResolveLatestFile:
    """Test _resolve_latest_file helper function."""

    def test_resolve_latest_file_single_match(self, tmp_path):
        # Arrange
        file1 = tmp_path / "data_2024-01-01.csv"
        file1.touch()

        # Act
        result = _resolve_latest_file("data_*.csv", tmp_path)

        # Assert
        assert result == file1

    def test_resolve_latest_file_multiple_matches_returns_newest(self, tmp_path):
        # Arrange
        file1 = tmp_path / "data_2024-01-01.csv"
        file2 = tmp_path / "data_2024-01-02.csv"
        file1.touch()
        file2.touch()

        # Ensure file2 has newer modification time
        import time

        time.sleep(0.01)
        file2.touch()

        # Act
        result = _resolve_latest_file("data_*.csv", tmp_path)

        # Assert
        assert result == file2

    def test_resolve_latest_file_no_matches_raises_error(self, tmp_path):
        # Arrange - no files created

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="No files matched pattern"):
            _resolve_latest_file("data_*.csv", tmp_path)


@pytest.mark.fast
class TestDownloadPriceData:
    """Test download_price_data function."""

    @patch("src.ml.training_pipeline.ingestion.data_commands._download")
    @patch("src.ml.training_pipeline.ingestion._resolve_latest_file")
    def test_download_price_data_csv_format(self, mock_resolve, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        csv_file = tmp_path / "data" / "test.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [103.0, 104.0, 105.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )
        df.to_csv(csv_file, index=False)

        mock_download.return_value = 0
        mock_resolve.return_value = csv_file

        # Act
        result = download_price_data(ctx)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "open" in result.columns
        assert result.index.name == "timestamp"
        assert isinstance(result.index[0], pd.Timestamp)
        mock_download.assert_called_once()

    @patch("src.ml.training_pipeline.ingestion.data_commands._download")
    @patch("src.ml.training_pipeline.ingestion._resolve_latest_file")
    def test_download_price_data_feather_format(self, mock_resolve, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        feather_file = tmp_path / "data" / "test.feather"
        feather_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [103.0, 104.0, 105.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        )
        df.to_feather(feather_file)

        mock_download.return_value = 0
        mock_resolve.return_value = feather_file

        # Act
        result = download_price_data(ctx)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "open" in result.columns
        assert result.index.name == "timestamp"
        mock_download.assert_called_once()

    @patch("src.ml.training_pipeline.ingestion.data_commands._download")
    def test_download_price_data_download_failure(self, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        mock_download.return_value = 1  # Failure status

        # Act & Assert
        with pytest.raises(RuntimeError, match="Price data download failed"):
            download_price_data(ctx)

    @patch("src.ml.training_pipeline.ingestion.data_commands._download")
    @patch("src.ml.training_pipeline.ingestion._resolve_latest_file")
    def test_download_price_data_sorts_by_index(self, mock_resolve, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        csv_file = tmp_path / "data" / "test.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        # Create unsorted data
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 02:00:00"),
                    pd.Timestamp("2024-01-01 00:00:00"),
                    pd.Timestamp("2024-01-01 01:00:00"),
                ],
                "open": [102.0, 100.0, 101.0],
                "close": [105.0, 103.0, 104.0],
            }
        )
        df.to_csv(csv_file, index=False)

        mock_download.return_value = 0
        mock_resolve.return_value = csv_file

        # Act
        result = download_price_data(ctx)

        # Assert
        assert result.index[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert result.index[1] == pd.Timestamp("2024-01-01 01:00:00")
        assert result.index[2] == pd.Timestamp("2024-01-01 02:00:00")


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
        mock_provider.get_historical_sentiment.assert_called_once_with(
            "BTCUSDT", start, end
        )

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
