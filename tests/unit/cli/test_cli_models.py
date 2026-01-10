"""Tests for atb models commands."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from cli.commands.models import (
    _handle_compare,
    _handle_list,
    _handle_promote,
    _handle_validate,
)


class TestModelsListCommand:
    """Tests for the models list command."""

    def test_lists_available_models(self):
        """Test that available models are listed."""
        # Arrange
        args = argparse.Namespace(models_cmd="list")

        mock_bundle = Mock()
        mock_bundle.symbol = "BTCUSDT"
        mock_bundle.timeframe = "1h"
        mock_bundle.model_type = "basic"
        mock_bundle.version_id = "2024-10-01_v1"

        # Act
        with (
            patch("cli.commands.models.PredictionConfig") as mock_config_class,
            patch("cli.commands.models.PredictionModelRegistry") as mock_registry_class,
            patch("cli.commands.models.Path") as mock_path,
        ):

            mock_config = Mock()
            mock_config.model_registry_path = "/tmp/models"
            mock_config_class.from_config_manager.return_value = mock_config

            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            mock_registry = Mock()
            mock_registry.list_bundles.return_value = [mock_bundle]
            mock_registry_class.return_value = mock_registry

            result = _handle_list(args)

            # Assert
            assert result == 0
            mock_registry.list_bundles.assert_called_once()

    def test_handles_no_models_directory(self):
        """Test that missing models directory is handled gracefully."""
        # Arrange
        args = argparse.Namespace(models_cmd="list")

        # Act
        with (
            patch("cli.commands.models.PredictionConfig") as mock_config_class,
            patch("cli.commands.models.PredictionModelRegistry") as mock_registry_class,
            patch("cli.commands.models.Path") as mock_path,
        ):

            mock_config = Mock()
            mock_config.model_registry_path = "/nonexistent/models"
            mock_config_class.from_config_manager.return_value = mock_config

            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance

            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            result = _handle_list(args)

            # Assert
            assert result == 0


class TestModelsCompareCommand:
    """Tests for the models compare command."""

    def test_compares_models_successfully(self):
        """Test that model comparison succeeds."""
        # Arrange
        args = argparse.Namespace(
            models_cmd="compare",
            symbol="BTCUSDT",
            timeframe="1h",
            model_type="basic",
        )

        mock_bundle = Mock()
        mock_bundle.metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
        }

        # Act
        with (
            patch("cli.commands.models.PredictionConfig") as mock_config_class,
            patch("cli.commands.models.PredictionModelRegistry") as mock_registry_class,
        ):

            mock_config = Mock()
            mock_config_class.from_config_manager.return_value = mock_config

            mock_registry = Mock()
            mock_registry.select_bundle.return_value = mock_bundle
            mock_registry_class.return_value = mock_registry

            result = _handle_compare(args)

            # Assert
            assert result == 0
            mock_registry.select_bundle.assert_called_once_with(
                symbol="BTCUSDT", model_type="basic", timeframe="1h"
            )

    def test_returns_error_when_model_not_found(self):
        """Test that error is returned when model is not found."""
        # Arrange
        args = argparse.Namespace(
            models_cmd="compare",
            symbol="INVALID",
            timeframe="1h",
            model_type="basic",
        )

        # Act
        with (
            patch("cli.commands.models.PredictionConfig") as mock_config_class,
            patch("cli.commands.models.PredictionModelRegistry") as mock_registry_class,
        ):

            mock_config = Mock()
            mock_config_class.from_config_manager.return_value = mock_config

            mock_registry = Mock()
            mock_registry.select_bundle.side_effect = Exception("Model not found")
            mock_registry_class.return_value = mock_registry

            result = _handle_compare(args)

            # Assert
            assert result == 1


class TestModelsValidateCommand:
    """Tests for the models validate command."""

    def test_validates_models_successfully(self):
        """Test that model validation succeeds."""
        # Arrange
        args = argparse.Namespace(models_cmd="validate")

        # Act
        with (
            patch("cli.commands.models.PredictionConfig") as mock_config_class,
            patch("cli.commands.models.PredictionModelRegistry") as mock_registry_class,
        ):

            mock_config = Mock()
            mock_config_class.from_config_manager.return_value = mock_config

            mock_registry = Mock()
            mock_registry.reload_models.return_value = None
            mock_registry_class.return_value = mock_registry

            result = _handle_validate(args)

            # Assert
            assert result == 0
            mock_registry.reload_models.assert_called_once()

    def test_returns_error_when_validation_fails(self):
        """Test that error is returned when validation fails."""
        # Arrange
        args = argparse.Namespace(models_cmd="validate")

        # Act
        with (
            patch("cli.commands.models.PredictionConfig") as mock_config_class,
            patch("cli.commands.models.PredictionModelRegistry") as mock_registry_class,
        ):

            mock_config = Mock()
            mock_config_class.from_config_manager.return_value = mock_config

            mock_registry = Mock()
            mock_registry.reload_models.side_effect = Exception("Invalid model format")
            mock_registry_class.return_value = mock_registry

            result = _handle_validate(args)

            # Assert
            assert result == 1


class TestModelsPromoteCommand:
    """Tests for the models promote command."""

    def test_promotes_model_version_successfully(self):
        """Test that model promotion succeeds."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            model_dir = base_path / "BTCUSDT" / "basic"
            model_dir.mkdir(parents=True)

            version_dir = model_dir / "2024-10-01_v1"
            version_dir.mkdir()

            args = argparse.Namespace(
                models_cmd="promote",
                symbol="BTCUSDT",
                model_type="basic",
                version="2024-10-01_v1",
            )

            # Act
            with (
                patch("cli.commands.models.PredictionConfig") as mock_config_class,
                patch("cli.commands.models.Path") as mock_path_class,
            ):

                mock_config = Mock()
                mock_config.model_registry_path = str(base_path)
                mock_config_class.from_config_manager.return_value = mock_config

                # Mock Path to return our temp directory structure
                def path_side_effect(path_str):
                    if path_str == str(base_path):
                        return base_path
                    return Path(path_str)

                mock_path_class.side_effect = path_side_effect

                result = _handle_promote(args)

                # Assert
                assert result == 0
                latest_link = model_dir / "latest"
                assert latest_link.exists() or latest_link.is_symlink()

    def test_returns_error_when_version_not_found(self):
        """Test that error is returned when version doesn't exist."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            args = argparse.Namespace(
                models_cmd="promote",
                symbol="BTCUSDT",
                model_type="basic",
                version="nonexistent",
            )

            # Act
            with (
                patch("cli.commands.models.PredictionConfig") as mock_config_class,
                patch("cli.commands.models.Path") as mock_path_class,
            ):

                mock_config = Mock()
                mock_config.model_registry_path = str(base_path)
                mock_config_class.from_config_manager.return_value = mock_config

                def path_side_effect(path_str):
                    if path_str == str(base_path):
                        return base_path
                    return Path(path_str)

                mock_path_class.side_effect = path_side_effect

                result = _handle_promote(args)

                # Assert
                assert result == 1
