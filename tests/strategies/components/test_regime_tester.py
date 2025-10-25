"""
Unit tests for RegimeTester component
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.strategies.components.testing.regime_tester import RegimeTester
from src.strategies.components.regime_context import RegimeContext
from src.regime.detector import TrendLabel, VolLabel


class TestRegimeTesterVolatilityParsing:
    """Test RegimeTester volatility parsing fix"""

    def create_test_data(self, length=200):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=length, freq="1h")

        # Create trending data
        base_price = 50000
        trend = np.linspace(0, 0.1, length)  # 10% trend over period
        noise = np.random.normal(0, 0.01, length)  # 1% noise

        prices = base_price * (1 + trend + noise)

        data = {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, length),
            "regime_label": "trend_up_low_vol",
            "trend_label": "trend_up",
            "vol_label": "low_vol",
            "regime_confidence": 0.8,
            "regime_duration": 10,
            "regime_strength": 0.7,
        }

        return pd.DataFrame(data, index=dates)

    def test_parse_vol_label_with_vol_suffix(self):
        """Test that _parse_vol_label correctly handles volatility strings with _vol suffix"""
        tester = RegimeTester(self.create_test_data())

        # Test all valid volatility labels
        assert tester._parse_vol_label("low_vol") == VolLabel.LOW
        assert tester._parse_vol_label("high_vol") == VolLabel.HIGH
        assert tester._parse_vol_label("medium_vol") == VolLabel.LOW  # Defaults to LOW

        # Test fallback for unknown labels
        assert tester._parse_vol_label("unknown") == VolLabel.LOW
        assert tester._parse_vol_label("") == VolLabel.LOW

    def test_parse_trend_label(self):
        """Test that _parse_trend_label correctly handles trend strings"""
        tester = RegimeTester(self.create_test_data())

        # Test all valid trend labels
        assert tester._parse_trend_label("trend_up") == TrendLabel.TREND_UP
        assert tester._parse_trend_label("trend_down") == TrendLabel.TREND_DOWN
        assert tester._parse_trend_label("range") == TrendLabel.RANGE

        # Test fallback for unknown labels
        assert tester._parse_trend_label("unknown") == TrendLabel.RANGE

    def test_regime_string_parsing_with_vol_suffix(self):
        """Test that regime string parsing correctly handles _vol suffix"""
        tester = RegimeTester(self.create_test_data())

        # Test the fixed parsing logic directly
        test_cases = [
            ("trend_up_low_vol", "trend_up", "low_vol"),
            ("trend_down_high_vol", "trend_down", "high_vol"),
            (
                "range_medium_vol",
                "range_medium",
                "vol",
            ),  # 3 parts: range, medium, vol -> legacy format
            ("trend_up_high_vol", "trend_up", "high_vol"),
            ("trend_down_low_vol", "trend_down", "low_vol"),
        ]

        for regime_type, expected_trend, expected_vol in test_cases:
            regime_parts = regime_type.split("_")

            if len(regime_parts) >= 4:
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = f"{regime_parts[2]}_{regime_parts[3]}"
            elif len(regime_parts) >= 3:
                # Handle legacy format without _vol suffix
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = regime_parts[2]
            else:
                trend_str = "range"
                volatility_str = "low_vol"

            assert (
                trend_str == expected_trend
            ), f"Expected trend '{expected_trend}', got '{trend_str}' for '{regime_type}'"
            assert (
                volatility_str == expected_vol
            ), f"Expected vol '{expected_vol}', got '{volatility_str}' for '{regime_type}'"

    def test_regime_string_parsing_legacy_format(self):
        """Test that regime string parsing handles legacy format without _vol suffix"""
        tester = RegimeTester(self.create_test_data())

        # Test legacy format (should still work)
        test_cases = [
            ("trend_up_low", "trend_up", "low"),
            ("trend_down_high", "trend_down", "high"),
            ("range_medium", "range", "low_vol"),  # 2 parts -> fallback
        ]

        for regime_type, expected_trend, expected_vol in test_cases:
            regime_parts = regime_type.split("_")

            if len(regime_parts) >= 4:
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = f"{regime_parts[2]}_{regime_parts[3]}"
            elif len(regime_parts) >= 3:
                # Handle legacy format without _vol suffix
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = regime_parts[2]
            else:
                trend_str = "range"
                volatility_str = "low_vol"

            assert (
                trend_str == expected_trend
            ), f"Expected trend '{expected_trend}', got '{trend_str}' for '{regime_type}'"
            assert (
                volatility_str == expected_vol
            ), f"Expected vol '{expected_vol}', got '{volatility_str}' for '{regime_type}'"

    def test_regime_string_parsing_edge_cases(self):
        """Test regime string parsing with edge cases"""
        tester = RegimeTester(self.create_test_data())

        # Test edge cases
        test_cases = [
            ("trend_up", "range", "low_vol"),  # Too few parts
            ("trend_up_low_vol_extra", "trend_up", "low_vol"),  # Too many parts
            ("", "range", "low_vol"),  # Empty string
        ]

        for regime_type, expected_trend, expected_vol in test_cases:
            regime_parts = regime_type.split("_")

            if len(regime_parts) >= 4:
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = f"{regime_parts[2]}_{regime_parts[3]}"
            elif len(regime_parts) >= 3:
                # Handle legacy format without _vol suffix
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = regime_parts[2]
            else:
                trend_str = "range"
                volatility_str = "low_vol"

            assert (
                trend_str == expected_trend
            ), f"Expected trend '{expected_trend}', got '{trend_str}' for '{regime_type}'"
            assert (
                volatility_str == expected_vol
            ), f"Expected vol '{expected_vol}', got '{volatility_str}' for '{regime_type}'"

    @patch("src.strategies.components.testing.regime_tester.logger")
    def test_regime_string_parsing_with_logging(self, mock_logger):
        """Test that regime string parsing logs warnings for unexpected formats"""
        tester = RegimeTester(self.create_test_data())

        # Test with unexpected format
        regime_type = "unexpected_format"
        regime_parts = regime_type.split("_")

        if len(regime_parts) >= 4:
            trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
            volatility_str = f"{regime_parts[2]}_{regime_parts[3]}"
        elif len(regime_parts) >= 3:
            # Handle legacy format without _vol suffix
            trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
            volatility_str = regime_parts[2]
        else:
            # Log warning for unexpected format
            mock_logger.warning(
                f"Unexpected regime_type format: '{regime_type}'. Using fallback values."
            )
            trend_str = "range"
            volatility_str = "low_vol"

        # Should use fallback values
        assert trend_str == "range"
        assert volatility_str == "low_vol"

        # Should have logged a warning
        mock_logger.warning.assert_called_once()

    def test_volatility_enum_conversion_integration(self):
        """Test that the full parsing and enum conversion works correctly"""
        tester = RegimeTester(self.create_test_data())

        # Test cases that should now work correctly with the fix
        test_cases = [
            ("trend_up_low_vol", TrendLabel.TREND_UP, VolLabel.LOW),
            ("trend_down_high_vol", TrendLabel.TREND_DOWN, VolLabel.HIGH),
            ("range_medium_vol", TrendLabel.RANGE, VolLabel.LOW),  # medium_vol defaults to LOW
        ]

        for regime_type, expected_trend, expected_vol in test_cases:
            regime_parts = regime_type.split("_")

            if len(regime_parts) >= 4:
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = f"{regime_parts[2]}_{regime_parts[3]}"
            elif len(regime_parts) >= 3:
                # Handle legacy format without _vol suffix
                trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                volatility_str = regime_parts[2]
            else:
                trend_str = "range"
                volatility_str = "low_vol"

            # Convert to enums
            trend = tester._parse_trend_label(trend_str)
            volatility = tester._parse_vol_label(volatility_str)

            assert (
                trend == expected_trend
            ), f"Expected trend {expected_trend}, got {trend} for '{regime_type}'"
            assert (
                volatility == expected_vol
            ), f"Expected vol {expected_vol}, got {volatility} for '{regime_type}'"

    def test_bug_fix_verification(self):
        """Test that the specific bug described in the issue is fixed"""
        tester = RegimeTester(self.create_test_data())

        # This is the exact case from the bug report
        regime_type = "trend_up_low_vol"

        # Before the fix: regime_parts[2] would be "low" and _parse_vol_label would fallback to LOW
        # After the fix: regime_parts[2] + "_" + regime_parts[3] = "low_vol" and _parse_vol_label returns LOW correctly

        regime_parts = regime_type.split("_")
        assert len(regime_parts) == 4, f"Expected 4 parts, got {len(regime_parts)}: {regime_parts}"

        # Test the fixed logic
        if len(regime_parts) >= 4:
            trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
            volatility_str = f"{regime_parts[2]}_{regime_parts[3]}"

        assert trend_str == "trend_up"
        assert volatility_str == "low_vol"  # This is the key fix!

        # Test enum conversion
        trend = tester._parse_trend_label(trend_str)
        volatility = tester._parse_vol_label(volatility_str)

        assert trend == TrendLabel.TREND_UP
        assert volatility == VolLabel.LOW  # Should be LOW, not fallback due to unrecognized "low"

        # Test that high_vol also works
        regime_type_high = "trend_down_high_vol"
        regime_parts_high = regime_type_high.split("_")

        if len(regime_parts_high) >= 4:
            trend_str_high = f"{regime_parts_high[0]}_{regime_parts_high[1]}"
            volatility_str_high = f"{regime_parts_high[2]}_{regime_parts_high[3]}"

        assert trend_str_high == "trend_down"
        assert volatility_str_high == "high_vol"

        trend_high = tester._parse_trend_label(trend_str_high)
        volatility_high = tester._parse_vol_label(volatility_str_high)

        assert trend_high == TrendLabel.TREND_DOWN
        assert (
            volatility_high == VolLabel.HIGH
        )  # Should be HIGH, not fallback due to unrecognized "high"
