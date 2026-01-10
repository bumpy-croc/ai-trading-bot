"""Tests for atb regime commands."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cli.commands.regime import _apply_regime_detection, _fetch_price_data


class TestFetchPriceData:
    """Tests for the _fetch_price_data function."""

    def test_fetches_price_data_successfully(self):
        """Test that price data is fetched successfully."""
        # Arrange
        mock_df = pd.DataFrame(
            {
                "open": [40000, 40100],
                "high": [40100, 40200],
                "low": [39900, 40000],
                "close": [40050, 40150],
                "volume": [100, 150],
            },
            index=pd.date_range(start="2024-01-01", periods=2, freq="1h"),
        )

        # Act
        with patch("cli.commands.regime.BinanceProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = mock_df
            mock_provider_class.return_value = mock_provider

            result = _fetch_price_data("BTCUSDT", "1h", 7)

            # Assert
            assert not result.empty
            assert len(result) == 2
            mock_provider.get_historical_data.assert_called_once()

    def test_raises_error_when_no_data_returned(self):
        """Test that error is raised when no data is returned."""
        # Arrange
        empty_df = pd.DataFrame()

        # Act & Assert
        with patch("cli.commands.regime.BinanceProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = empty_df
            mock_provider_class.return_value = mock_provider

            with pytest.raises(ValueError, match="No data returned"):
                _fetch_price_data("BTCUSDT", "1h", 7)


class TestApplyRegimeDetection:
    """Tests for the _apply_regime_detection function."""

    def test_applies_regime_detection_successfully(self):
        """Test that regime detection is applied successfully."""
        # Arrange
        mock_df = pd.DataFrame(
            {
                "open": [40000, 40100],
                "high": [40100, 40200],
                "low": [39900, 40000],
                "close": [40050, 40150],
                "volume": [100, 150],
            },
            index=pd.date_range(start="2024-01-01", periods=2, freq="1h"),
        )

        mock_annotated_df = mock_df.copy()
        mock_annotated_df["regime"] = ["trend_up", "trend_up"]

        # Act
        with patch("cli.commands.regime.RegimeDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.annotate.return_value = mock_annotated_df
            mock_detector_class.return_value = mock_detector

            result = _apply_regime_detection(mock_df)

            # Assert
            assert "regime" in result.columns
            mock_detector.annotate.assert_called_once()
