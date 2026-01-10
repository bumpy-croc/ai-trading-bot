"""Unit tests for ML training pipeline configuration module."""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.ml.training_pipeline.config import (
    DiagnosticsOptions,
    TrainingConfig,
    TrainingContext,
    TrainingPaths,
)


@pytest.mark.fast
class TestDiagnosticsOptions:
    """Test DiagnosticsOptions dataclass."""

    def test_default_values(self):
        # Arrange & Act
        options = DiagnosticsOptions()

        # Assert
        assert options.generate_plots is True
        assert options.evaluate_robustness is True
        assert options.convert_to_onnx is True

    def test_custom_values(self):
        # Arrange & Act
        options = DiagnosticsOptions(
            generate_plots=False,
            evaluate_robustness=False,
            convert_to_onnx=False,
        )

        # Assert
        assert options.generate_plots is False
        assert options.evaluate_robustness is False
        assert options.convert_to_onnx is False


@pytest.mark.fast
class TestTrainingPaths:
    """Test TrainingPaths dataclass."""

    def test_initialization(self, tmp_path):
        # Arrange
        data_dir = tmp_path / "data"
        models_dir = tmp_path / "models"

        # Act
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=data_dir,
            models_dir=models_dir,
        )

        # Assert
        assert paths.project_root == tmp_path
        assert paths.data_dir == data_dir
        assert paths.models_dir == models_dir

    @patch("src.ml.training_pipeline.config.get_project_root")
    def test_default_factory(self, mock_get_root, tmp_path):
        # Arrange
        mock_get_root.return_value = tmp_path

        # Act
        paths = TrainingPaths.default()

        # Assert
        assert paths.project_root == tmp_path
        assert paths.data_dir == tmp_path / "data"
        assert paths.models_dir == tmp_path / "src" / "ml" / "models"
        assert paths.data_dir.exists()
        assert paths.models_dir.exists()


@pytest.mark.fast
class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        # Act
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )

        # Assert
        assert config.symbol == "BTCUSDT"
        assert config.timeframe == "1h"
        assert config.start_date == start
        assert config.end_date == end
        assert config.epochs == 300
        assert config.batch_size == 32
        assert config.sequence_length == 120
        assert config.force_sentiment is False
        assert config.force_price_only is False
        assert config.mixed_precision is True
        assert isinstance(config.diagnostics, DiagnosticsOptions)

    def test_custom_values(self):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)
        diagnostics = DiagnosticsOptions(generate_plots=False)

        # Act
        config = TrainingConfig(
            symbol="ETHUSDT",
            timeframe="4h",
            start_date=start,
            end_date=end,
            epochs=100,
            batch_size=64,
            sequence_length=60,
            force_sentiment=True,
            force_price_only=False,
            mixed_precision=False,
            diagnostics=diagnostics,
        )

        # Assert
        assert config.symbol == "ETHUSDT"
        assert config.timeframe == "4h"
        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.sequence_length == 60
        assert config.force_sentiment is True
        assert config.force_price_only is False
        assert config.mixed_precision is False
        assert config.diagnostics.generate_plots is False

    def test_days_requested(self):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )

        # Act
        days = config.days_requested()

        # Assert
        assert days == 30


@pytest.mark.fast
class TestTrainingContext:
    """Test TrainingContext dataclass."""

    def test_initialization(self, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
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

        # Act
        ctx = TrainingContext(config=config, paths=paths)

        # Assert
        assert ctx.config == config
        assert ctx.paths == paths

    @patch("src.ml.training_pipeline.config.get_project_root")
    def test_default_paths(self, mock_get_root, tmp_path):
        # Arrange
        mock_get_root.return_value = tmp_path
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )

        # Act
        ctx = TrainingContext(config=config)

        # Assert
        assert ctx.config == config
        assert ctx.paths.project_root == tmp_path

    @patch("src.trading.symbols.factory.SymbolFactory.to_exchange_symbol")
    def test_symbol_exchange_property(self, mock_factory):
        # Arrange
        mock_factory.return_value = "BTCUSDT"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        config = TrainingConfig(
            symbol="BTC-USD",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        ctx = TrainingContext(config=config)

        # Act
        exchange_symbol = ctx.symbol_exchange

        # Assert
        assert exchange_symbol == "BTCUSDT"
        mock_factory.assert_called_once_with("BTC-USD", "binance")

    def test_start_iso_property(self):
        # Arrange
        start = datetime(2024, 1, 15, 10, 30, 45)
        end = datetime(2024, 12, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        ctx = TrainingContext(config=config)

        # Act
        start_iso = ctx.start_iso

        # Assert
        assert start_iso == "2024-01-15T00:00:00Z"

    def test_end_iso_property(self):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31, 10, 30, 45)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        ctx = TrainingContext(config=config)

        # Act
        end_iso = ctx.end_iso

        # Assert
        assert end_iso == "2024-12-31T23:59:59Z"

    @patch("src.trading.symbols.factory.SymbolFactory.to_exchange_symbol")
    def test_price_data_glob_property(self, mock_factory):
        # Arrange
        mock_factory.return_value = "BTCUSDT"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        ctx = TrainingContext(config=config)

        # Act
        glob_pattern = ctx.price_data_glob

        # Assert
        assert glob_pattern == "BTCUSDT_1h_2024-01-01T00:00:00Z_2024-01-31T23:59:59Z.*"
