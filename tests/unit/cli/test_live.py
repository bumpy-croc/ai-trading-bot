"""Tests for atb live and live-control commands."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

import pytest

from cli.commands.live import (
    _handle,
    _control,
    _date_range,
    _list_registry,
    _resolve_version_path,
    _repoint_latest,
)


class TestDateRange:
    """Tests for the _date_range function."""

    def test_returns_date_range_for_given_days(self):
        """Test that date range is calculated correctly for given days."""
        # Arrange & Act
        with patch("cli.commands.live.datetime") as mock_datetime:
            from datetime import datetime, timedelta

            mock_now = datetime(2024, 10, 29)
            mock_datetime.utcnow.return_value = mock_now

            start, end = _date_range(7)

            # Assert
            assert end == "2024-10-29"
            assert start == "2024-10-22"


class TestListRegistry:
    """Tests for the _list_registry function."""

    def test_returns_empty_dict_when_registry_does_not_exist(self):
        """Test that empty dict is returned when registry doesn't exist."""
        # Arrange & Act
        with patch("cli.commands.live.MODEL_REGISTRY", Path("/nonexistent")):
            result = _list_registry()

            # Assert
            assert result == {}

    def test_lists_models_in_registry(self):
        """Test that models are listed correctly from registry."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir)

            # Create mock registry structure
            btc_basic = registry / "BTCUSDT" / "basic"
            btc_basic.mkdir(parents=True)
            (btc_basic / "2024-10-01_v1").mkdir()
            (btc_basic / "2024-10-15_v2").mkdir()

            eth_sentiment = registry / "ETHUSDT" / "sentiment"
            eth_sentiment.mkdir(parents=True)
            (eth_sentiment / "2024-10-20_v1").mkdir()

            # Act
            with patch("cli.commands.live.MODEL_REGISTRY", registry):
                result = _list_registry()

                # Assert
                assert "BTCUSDT" in result
                assert "basic" in result["BTCUSDT"]
                assert "2024-10-01_v1" in result["BTCUSDT"]["basic"]["versions"]
                assert "2024-10-15_v2" in result["BTCUSDT"]["basic"]["versions"]

                assert "ETHUSDT" in result
                assert "sentiment" in result["ETHUSDT"]
                assert "2024-10-20_v1" in result["ETHUSDT"]["sentiment"]["versions"]


class TestResolveVersionPath:
    """Tests for the _resolve_version_path function."""

    def test_resolves_relative_path(self):
        """Test that relative path is resolved correctly."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir)
            version_dir = registry / "BTCUSDT" / "basic" / "2024-10-01_v1"
            version_dir.mkdir(parents=True)

            # Act
            with patch("cli.commands.live.MODEL_REGISTRY", registry):
                result = _resolve_version_path("BTCUSDT/basic/2024-10-01_v1")

                # Assert
                assert result.exists()
                assert result.name == "2024-10-01_v1"

    def test_resolves_absolute_path(self):
        """Test that absolute path is resolved correctly."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir)
            version_dir = registry / "BTCUSDT" / "basic" / "2024-10-01_v1"
            version_dir.mkdir(parents=True)

            # Act
            with patch("cli.commands.live.MODEL_REGISTRY", registry):
                result = _resolve_version_path(str(version_dir))

                # Assert
                assert result.exists()
                assert result.name == "2024-10-01_v1"

    def test_raises_error_for_nonexistent_path(self):
        """Test that error is raised for nonexistent path."""
        # Arrange & Act & Assert
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir)

            with patch("cli.commands.live.MODEL_REGISTRY", registry):
                with pytest.raises(FileNotFoundError, match="does not exist"):
                    _resolve_version_path("BTCUSDT/basic/nonexistent")

    def test_raises_error_for_path_outside_registry(self):
        """Test that error is raised for path outside registry."""
        # Arrange & Act & Assert
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir)

            with patch("cli.commands.live.MODEL_REGISTRY", registry):
                with pytest.raises(ValueError, match="must be inside the registry"):
                    _resolve_version_path("/etc/passwd")


class TestRepointLatest:
    """Tests for the _repoint_latest function."""

    def test_creates_latest_symlink(self):
        """Test that latest symlink is created successfully."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            model_type_dir = Path(tmpdir) / "BTCUSDT" / "basic"
            model_type_dir.mkdir(parents=True)

            version_dir = model_type_dir / "2024-10-01_v1"
            version_dir.mkdir()

            # Act
            _repoint_latest(version_dir)

            # Assert
            latest_link = model_type_dir / "latest"
            assert latest_link.exists() or latest_link.is_symlink()
            assert latest_link.resolve() == version_dir.resolve()

    def test_updates_existing_latest_symlink(self):
        """Test that existing latest symlink is updated."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            model_type_dir = Path(tmpdir) / "BTCUSDT" / "basic"
            model_type_dir.mkdir(parents=True)

            old_version = model_type_dir / "2024-10-01_v1"
            old_version.mkdir()

            new_version = model_type_dir / "2024-10-15_v2"
            new_version.mkdir()

            # Create initial symlink
            latest_link = model_type_dir / "latest"
            latest_link.symlink_to(old_version.name)

            # Act
            _repoint_latest(new_version)

            # Assert
            assert latest_link.resolve() == new_version.resolve()


class TestHandleLive:
    """Tests for the _handle function."""

    def test_forwards_to_live_runner(self):
        """Test that live command forwards to runner module."""
        # Arrange
        args = argparse.Namespace(args=["ml_basic", "--paper-trading"])

        # Act
        with patch("cli.commands.live.forward_to_module_main") as mock_forward:
            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            mock_forward.assert_called_once_with("src.engines.live.runner", ["ml_basic", "--paper-trading"])

    def test_handles_empty_args(self):
        """Test that live command handles empty args."""
        # Arrange
        args = argparse.Namespace(args=None)

        # Act
        with patch("cli.commands.live.forward_to_module_main") as mock_forward:
            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            mock_forward.assert_called_once_with("src.engines.live.runner", [])


class TestControlTrain:
    """Tests for the train subcommand of live-control."""

    def test_trains_basic_model_successfully(self):
        """Test that basic model training executes successfully."""
        # Arrange
        args = argparse.Namespace(
            control_cmd="train",
            symbol="BTCUSDT",
            sentiment=False,
            days=365,
            epochs=50,
            auto_deploy=False,
        )

        metadata = {
            "version_id": "2024-10-01_v1",
            "symbol": "BTCUSDT",
            "model_type": "basic",
        }

        # Act
        with (
            patch("cli.commands.live.train_price_model_main") as mock_train,
            patch("cli.commands.live._latest_metadata") as mock_metadata_path,
            patch("builtins.open", mock_open(read_data=json.dumps(metadata))),
        ):

            mock_train.return_value = 0
            mock_metadata_path.return_value = Path("/tmp/metadata.json")

            result = _control(args)

            # Assert
            assert result == 0
            mock_train.assert_called_once()

    def test_trains_sentiment_model_successfully(self):
        """Test that sentiment model training executes successfully."""
        # Arrange
        args = argparse.Namespace(
            control_cmd="train",
            symbol="BTCUSDT",
            sentiment=True,
            days=365,
            epochs=50,
            auto_deploy=False,
        )

        metadata = {
            "version_id": "2024-10-01_v1",
            "symbol": "BTCUSDT",
            "model_type": "sentiment",
        }

        # Act
        with (
            patch("cli.commands.live.train_model_main") as mock_train,
            patch("cli.commands.live._latest_metadata") as mock_metadata_path,
            patch("builtins.open", mock_open(read_data=json.dumps(metadata))),
        ):

            mock_train.return_value = 0
            mock_metadata_path.return_value = Path("/tmp/metadata.json")

            result = _control(args)

            # Assert
            assert result == 0
            mock_train.assert_called_once()

    def test_returns_error_when_training_fails(self):
        """Test that error is returned when training fails."""
        # Arrange
        args = argparse.Namespace(
            control_cmd="train",
            symbol="BTCUSDT",
            sentiment=False,
            days=365,
            epochs=50,
            auto_deploy=False,
        )

        # Act
        with patch("cli.commands.live.train_price_model_main") as mock_train:
            mock_train.return_value = 1

            result = _control(args)

            # Assert
            assert result == 1


class TestControlDeployModel:
    """Tests for the deploy-model subcommand of live-control."""

    def test_deploys_model_successfully(self):
        """Test that model deployment executes successfully."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir)
            version_dir = registry / "BTCUSDT" / "basic" / "2024-10-01_v1"
            version_dir.mkdir(parents=True)

            args = argparse.Namespace(
                control_cmd="deploy-model",
                model_path=str(version_dir),
                close_positions=False,
            )

            # Act
            with patch("cli.commands.live.MODEL_REGISTRY", registry):
                result = _control(args)

                # Assert
                assert result == 0

    def test_returns_error_for_invalid_model_path(self):
        """Test that error is returned for invalid model path."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = Path(tmpdir)

            args = argparse.Namespace(
                control_cmd="deploy-model",
                model_path="nonexistent/path",
                close_positions=False,
            )

            # Act
            with patch("cli.commands.live.MODEL_REGISTRY", registry):
                result = _control(args)

                # Assert
                assert result == 1


class TestControlListModels:
    """Tests for the list-models subcommand of live-control."""

    def test_lists_models_successfully(self):
        """Test that model listing executes successfully."""
        # Arrange
        args = argparse.Namespace(control_cmd="list-models")

        # Act
        with patch("cli.commands.live._list_registry") as mock_list:
            mock_list.return_value = {
                "BTCUSDT": {
                    "basic": {
                        "latest": "2024-10-01_v1",
                        "versions": ["2024-10-01_v1"],
                    }
                }
            }

            result = _control(args)

            # Assert
            assert result == 0
            mock_list.assert_called_once()


class TestControlStatus:
    """Tests for the status subcommand of live-control."""

    def test_returns_status_successfully(self):
        """Test that status command returns successfully."""
        # Arrange
        args = argparse.Namespace(control_cmd="status")

        # Act
        result = _control(args)

        # Assert
        assert result == 0


class TestControlEmergencyStop:
    """Tests for the emergency-stop subcommand of live-control."""

    def test_executes_emergency_stop(self):
        """Test that emergency stop executes successfully."""
        # Arrange
        args = argparse.Namespace(control_cmd="emergency-stop")

        # Act
        result = _control(args)

        # Assert
        assert result == 0


class TestControlSwapStrategy:
    """Tests for the swap-strategy subcommand of live-control."""

    def test_swaps_strategy_successfully(self):
        """Test that strategy swap executes successfully."""
        # Arrange
        args = argparse.Namespace(
            control_cmd="swap-strategy",
            strategy="ml_adaptive",
            close_positions=True,
        )

        # Act
        result = _control(args)

        # Assert
        assert result == 0
