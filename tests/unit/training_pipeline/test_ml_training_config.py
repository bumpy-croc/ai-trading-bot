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
        # New fields (TARGET-REDESIGN scaffolding): default preserves the
        # incumbent next-bar price-regression behavior exactly.
        assert config.target_type == "regression"
        assert config.target_horizon == 1

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

    def test_accepts_tft_model_type(self):
        # Arrange & Act
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            model_type="tft",
        )

        # Assert
        assert config.model_type == "tft"

    def test_accepts_tft_ternary_model_type(self):
        # Arrange & Act
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            model_type="tft_ternary",
        )

        # Assert
        assert config.model_type == "tft_ternary"

    def test_rejects_unknown_model_type(self):
        # Act & Assert
        with pytest.raises(ValueError, match="model_type"):
            TrainingConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                model_type="transformer",
            )

    def test_rejects_unknown_model_variant(self):
        # Act & Assert
        with pytest.raises(ValueError, match="model_variant"):
            TrainingConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                model_variant="huge",
            )

    def test_accepts_target_type_and_horizon(self):
        # Arrange & Act
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            target_type="triple_barrier",
            target_horizon=6,
        )

        # Assert
        assert config.target_type == "triple_barrier"
        assert config.target_horizon == 6

    def test_rejects_unknown_target_type(self):
        # Act & Assert
        with pytest.raises(ValueError, match="target_type"):
            TrainingConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                target_type="not_a_real_target",
            )

    def test_rejects_non_positive_target_horizon(self):
        # Act & Assert
        with pytest.raises(ValueError, match="target_horizon"):
            TrainingConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                target_horizon=0,
            )

    def test_config_construction_does_not_cross_validate_model_vs_target(self):
        """TrainingConfig construction only validates each field is a known
        value -- it does NOT cross-validate model_type against target_type.
        That cross-check (the #947 guard) runs at run_training_pipeline()
        entry instead, so config objects remain freely constructible/mutable
        before a run, and CLI callers that don't yet expose --target-type
        (e.g. `atb train cloud --model-type tft`) are unaffected."""
        # tft (binary_classification head) + default target_type=regression
        # would fail the cross-check, but must NOT fail here.
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            model_type="tft",
        )
        assert config.model_type == "tft"
        assert config.target_type == "regression"

    def test_accepts_lightgbm_model_type(self):
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            model_type="lightgbm",
        )
        assert config.model_type == "lightgbm"

    def test_accepts_meta_label_target_type_with_primary_model_type(self):
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            target_type="meta_label",
            primary_model_type="basic",
        )
        assert config.target_type == "meta_label"
        assert config.primary_model_type == "basic"

    def test_meta_label_without_primary_model_type_raises(self):
        """meta_label needs a primary signal to run forward first --
        primary_model_type names its registry model_type. Required, per
        train.py's --primary-model-type help text."""
        with pytest.raises(ValueError, match="primary_model_type"):
            TrainingConfig(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                target_type="meta_label",
            )

    def test_primary_model_type_defaults_to_none(self):
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        assert config.primary_model_type is None

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
