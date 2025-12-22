"""Integration tests for atb data commands."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli.commands.data import _download


@pytest.mark.integration
class TestDataIntegration:
    """Integration tests for data commands."""

    def test_download_creates_cache_files(self):
        """Test that download command creates cache files."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                symbol="BTCUSDT",
                timeframe="1h",
                start_date="2024-10-01",
                end_date="2024-10-02",
                output_dir=tmpdir,
                format="csv",
            )

            mock_ohlcv = [
                [1704067200000, 40000, 40100, 39900, 40050, 100],
            ]

            # Act
            with patch("cli.commands.data.ccxt") as mock_ccxt:
                mock_binance = Mock()
                mock_binance.fetch_ohlcv.return_value = mock_ohlcv
                mock_ccxt.binance.return_value = mock_binance

                result = _download(args)

                # Assert
                assert result == 0
                output_dir = Path(tmpdir)
                csv_files = list(output_dir.glob("*.csv"))
                assert len(csv_files) > 0
