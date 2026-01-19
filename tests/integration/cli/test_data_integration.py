"""Integration tests for atb data commands."""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
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

            # Create mock data
            mock_df = pd.DataFrame(
                {
                    "open": [40000.0],
                    "high": [40100.0],
                    "low": [39900.0],
                    "close": [40050.0],
                    "volume": [100.0],
                },
                index=pd.DatetimeIndex([datetime(2024, 10, 1)], name="timestamp"),
            )

            # Act - Mock the provider factory to return a provider that returns our mock data
            with patch("src.data_providers.provider_factory.create_data_provider") as mock_create:
                mock_provider = Mock()
                mock_provider.get_historical_data.return_value = mock_df
                mock_provider.close.return_value = None
                mock_create.return_value = mock_provider

                result = _download(args)

                # Assert
                assert result == 0
                output_dir = Path(tmpdir)
                csv_files = list(output_dir.glob("*.csv"))
                assert len(csv_files) > 0
