"""Tests for config.constants module.

Uses pytest parameterization to reduce code duplication while maintaining
comprehensive coverage of all constant categories.
"""

import pytest

from src.config.constants import (
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_PREDICTION_HORIZONS,
    DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_PREDICTION_LATENCY,
    DEFAULT_MODEL_REGISTRY_PATH,
    DEFAULT_ENABLE_SENTIMENT,
    DEFAULT_FEATURE_CACHE_TTL,
    DEFAULT_MODEL_CACHE_TTL,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_RSI_PERIOD,
    DEFAULT_ATR_PERIOD,
    DEFAULT_BOLLINGER_PERIOD,
    DEFAULT_MACD_FAST_PERIOD,
    DEFAULT_MACD_SLOW_PERIOD,
    DEFAULT_MACD_SIGNAL_PERIOD,
    DEFAULT_OPTIMIZATION_ENABLED,
    DEFAULT_PERFORMANCE_LOOKBACK_DAYS,
    DEFAULT_ERROR_COOLDOWN,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_REGIME_ADJUST_POSITION_SIZE,
    DEFAULT_REGIME_HYSTERESIS_K,
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_MIN_CHECK_INTERVAL,
    DEFAULT_MAX_CHECK_INTERVAL,
    DEFAULT_DYNAMIC_RISK_ENABLED,
    DEFAULT_DRAWDOWN_THRESHOLDS,
    DEFAULT_RISK_REDUCTION_FACTORS,
    DEFAULT_PARTIAL_EXIT_TARGETS,
    DEFAULT_PARTIAL_EXIT_SIZES,
    DEFAULT_TRAILING_ACTIVATION_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_CORRELATION_WINDOW_DAYS,
)


# Test data for parameterized tests
POSITIVE_CONSTANTS = [
    ("DEFAULT_INITIAL_BALANCE", DEFAULT_INITIAL_BALANCE, 1000),
    ("DEFAULT_MAX_HOLDING_HOURS", DEFAULT_MAX_HOLDING_HOURS, 336),
    ("DEFAULT_MAX_PREDICTION_LATENCY", DEFAULT_MAX_PREDICTION_LATENCY, 0.1),
    ("DEFAULT_FEATURE_CACHE_TTL", DEFAULT_FEATURE_CACHE_TTL, 3600),
    ("DEFAULT_MODEL_CACHE_TTL", DEFAULT_MODEL_CACHE_TTL, 600),
    ("DEFAULT_SEQUENCE_LENGTH", DEFAULT_SEQUENCE_LENGTH, 120),
    ("DEFAULT_PERFORMANCE_LOOKBACK_DAYS", DEFAULT_PERFORMANCE_LOOKBACK_DAYS, 30),
    ("DEFAULT_ERROR_COOLDOWN", DEFAULT_ERROR_COOLDOWN, 30),
    ("DEFAULT_REGIME_HYSTERESIS_K", DEFAULT_REGIME_HYSTERESIS_K, 3),
    ("DEFAULT_MIN_CHECK_INTERVAL", DEFAULT_MIN_CHECK_INTERVAL, None),
    ("DEFAULT_CHECK_INTERVAL", DEFAULT_CHECK_INTERVAL, None),
    ("DEFAULT_MAX_CHECK_INTERVAL", DEFAULT_MAX_CHECK_INTERVAL, None),
    ("DEFAULT_TRAILING_ACTIVATION_THRESHOLD", DEFAULT_TRAILING_ACTIVATION_THRESHOLD, 0.015),
    ("DEFAULT_CORRELATION_WINDOW_DAYS", DEFAULT_CORRELATION_WINDOW_DAYS, 30),
]

BOOLEAN_CONSTANTS = [
    ("DEFAULT_OPTIMIZATION_ENABLED", DEFAULT_OPTIMIZATION_ENABLED),
    ("DEFAULT_REGIME_ADJUST_POSITION_SIZE", DEFAULT_REGIME_ADJUST_POSITION_SIZE),
    ("DEFAULT_DYNAMIC_RISK_ENABLED", DEFAULT_DYNAMIC_RISK_ENABLED),
    ("DEFAULT_ENABLE_SENTIMENT", DEFAULT_ENABLE_SENTIMENT),
]

RANGE_CONSTANTS = [
    ("DEFAULT_MIN_CONFIDENCE_THRESHOLD", DEFAULT_MIN_CONFIDENCE_THRESHOLD, 0, 1, 0.6),
    ("DEFAULT_CORRELATION_THRESHOLD", DEFAULT_CORRELATION_THRESHOLD, 0, 1, 0.7),
]

INTEGER_CONSTANTS = [
    ("DEFAULT_SEQUENCE_LENGTH", DEFAULT_SEQUENCE_LENGTH),
    ("DEFAULT_RSI_PERIOD", DEFAULT_RSI_PERIOD),
    ("DEFAULT_ATR_PERIOD", DEFAULT_ATR_PERIOD),
    ("DEFAULT_BOLLINGER_PERIOD", DEFAULT_BOLLINGER_PERIOD),
    ("DEFAULT_MACD_FAST_PERIOD", DEFAULT_MACD_FAST_PERIOD),
    ("DEFAULT_MACD_SLOW_PERIOD", DEFAULT_MACD_SLOW_PERIOD),
    ("DEFAULT_MACD_SIGNAL_PERIOD", DEFAULT_MACD_SIGNAL_PERIOD),
    ("DEFAULT_CHECK_INTERVAL", DEFAULT_CHECK_INTERVAL),
    ("DEFAULT_CORRELATION_WINDOW_DAYS", DEFAULT_CORRELATION_WINDOW_DAYS),
]

NUMERIC_CONSTANTS = [
    ("DEFAULT_INITIAL_BALANCE", DEFAULT_INITIAL_BALANCE),
    ("DEFAULT_MIN_CONFIDENCE_THRESHOLD", DEFAULT_MIN_CONFIDENCE_THRESHOLD),
    ("DEFAULT_TRAILING_ACTIVATION_THRESHOLD", DEFAULT_TRAILING_ACTIVATION_THRESHOLD),
    ("DEFAULT_CORRELATION_THRESHOLD", DEFAULT_CORRELATION_THRESHOLD),
]


class TestPositiveConstants:
    """Tests for constants that must be positive."""

    @pytest.mark.parametrize("name,value,expected", POSITIVE_CONSTANTS)
    def test_constant_is_positive(self, name, value, expected):
        """Test that constant is positive and optionally matches expected value."""
        assert value > 0, f"{name} should be positive"
        if expected is not None:
            assert value == expected, f"{name} should equal {expected}"


class TestBooleanConstants:
    """Tests for boolean constants."""

    @pytest.mark.parametrize("name,value", BOOLEAN_CONSTANTS)
    def test_constant_is_boolean(self, name, value):
        """Test that constant is a boolean type."""
        assert isinstance(value, bool), f"{name} should be bool"


class TestRangeConstants:
    """Tests for constants with valid ranges."""

    @pytest.mark.parametrize("name,value,min_val,max_val,expected", RANGE_CONSTANTS)
    def test_constant_in_range(self, name, value, min_val, max_val, expected):
        """Test that constant is within valid range and matches expected."""
        assert min_val < value <= max_val, f"{name} should be in ({min_val}, {max_val}]"
        assert value == expected, f"{name} should equal {expected}"


class TestIntegerConstants:
    """Tests for integer constants."""

    @pytest.mark.parametrize("name,value", INTEGER_CONSTANTS)
    def test_constant_is_integer(self, name, value):
        """Test that constant is an integer type."""
        assert isinstance(value, int), f"{name} should be int"


class TestNumericConstants:
    """Tests for numeric (int or float) constants."""

    @pytest.mark.parametrize("name,value", NUMERIC_CONSTANTS)
    def test_constant_is_numeric(self, name, value):
        """Test that constant is numeric (int or float)."""
        assert isinstance(value, (int, float)), f"{name} should be numeric"


class TestTechnicalIndicatorConstants:
    """Tests for technical indicator period constants."""

    def test_rsi_period_standard(self):
        """Test that RSI period is standard (14)."""
        assert DEFAULT_RSI_PERIOD == 14

    def test_atr_period_standard(self):
        """Test that ATR period is standard (14)."""
        assert DEFAULT_ATR_PERIOD == 14

    def test_bollinger_period_standard(self):
        """Test that Bollinger period is standard (20)."""
        assert DEFAULT_BOLLINGER_PERIOD == 20

    def test_macd_periods_standard(self):
        """Test that MACD periods are standard (12, 26, 9)."""
        assert DEFAULT_MACD_FAST_PERIOD == 12
        assert DEFAULT_MACD_SLOW_PERIOD == 26
        assert DEFAULT_MACD_SIGNAL_PERIOD == 9

    def test_macd_fast_less_than_slow(self):
        """Test that MACD fast period is less than slow period."""
        assert DEFAULT_MACD_FAST_PERIOD < DEFAULT_MACD_SLOW_PERIOD


class TestCheckIntervalConstants:
    """Tests for check interval ordering."""

    def test_check_intervals_ordered(self):
        """Test that check intervals are properly ordered."""
        assert DEFAULT_MIN_CHECK_INTERVAL < DEFAULT_CHECK_INTERVAL
        assert DEFAULT_CHECK_INTERVAL < DEFAULT_MAX_CHECK_INTERVAL


class TestListConstants:
    """Tests for list-based constants."""

    def test_prediction_horizons_is_nonempty_list(self):
        """Test that prediction horizons is a non-empty list."""
        assert isinstance(DEFAULT_PREDICTION_HORIZONS, list)
        assert len(DEFAULT_PREDICTION_HORIZONS) >= 1

    def test_drawdown_thresholds_ascending(self):
        """Test that drawdown thresholds are in ascending order."""
        assert len(DEFAULT_DRAWDOWN_THRESHOLDS) > 0
        for i in range(len(DEFAULT_DRAWDOWN_THRESHOLDS) - 1):
            assert DEFAULT_DRAWDOWN_THRESHOLDS[i] < DEFAULT_DRAWDOWN_THRESHOLDS[i + 1]

    def test_risk_reduction_factors_descending(self):
        """Test that risk reduction factors are in descending order."""
        assert len(DEFAULT_RISK_REDUCTION_FACTORS) > 0
        for i in range(len(DEFAULT_RISK_REDUCTION_FACTORS) - 1):
            assert DEFAULT_RISK_REDUCTION_FACTORS[i] > DEFAULT_RISK_REDUCTION_FACTORS[i + 1]

    def test_partial_exit_targets_ascending(self):
        """Test that partial exit targets are in ascending order."""
        assert len(DEFAULT_PARTIAL_EXIT_TARGETS) > 0
        for i in range(len(DEFAULT_PARTIAL_EXIT_TARGETS) - 1):
            assert DEFAULT_PARTIAL_EXIT_TARGETS[i] < DEFAULT_PARTIAL_EXIT_TARGETS[i + 1]


class TestMatchingListLengths:
    """Tests for list constants that must have matching lengths."""

    def test_drawdown_and_risk_factors_match(self):
        """Test that drawdown thresholds and risk factors have same length."""
        assert len(DEFAULT_DRAWDOWN_THRESHOLDS) == len(DEFAULT_RISK_REDUCTION_FACTORS)

    def test_partial_exit_targets_and_sizes_match(self):
        """Test that exit targets and sizes have same length."""
        assert len(DEFAULT_PARTIAL_EXIT_TARGETS) == len(DEFAULT_PARTIAL_EXIT_SIZES)

    def test_partial_exit_sizes_sum_to_one(self):
        """Test that partial exit sizes sum to approximately 1."""
        total = sum(DEFAULT_PARTIAL_EXIT_SIZES)
        assert 0.99 <= total <= 1.01


class TestStringConstants:
    """Tests for string constants."""

    def test_model_registry_path_is_valid(self):
        """Test that model registry path is a valid string containing 'models'."""
        assert isinstance(DEFAULT_MODEL_REGISTRY_PATH, str)
        assert "models" in DEFAULT_MODEL_REGISTRY_PATH


class TestSentimentConstants:
    """Tests for sentiment-related constants."""

    def test_sentiment_disabled_by_default(self):
        """Test that sentiment is disabled by default."""
        assert DEFAULT_ENABLE_SENTIMENT is False


@pytest.mark.fast
class TestConstantsIntegrity:
    """Critical integrity tests for constants."""

    CRITICAL_CONSTANTS = [
        DEFAULT_INITIAL_BALANCE,
        DEFAULT_PREDICTION_HORIZONS,
        DEFAULT_MIN_CONFIDENCE_THRESHOLD,
        DEFAULT_SEQUENCE_LENGTH,
        DEFAULT_RSI_PERIOD,
        DEFAULT_ATR_PERIOD,
        DEFAULT_CHECK_INTERVAL,
    ]

    def test_no_none_values(self):
        """Test that critical constants are not None."""
        for const in self.CRITICAL_CONSTANTS:
            assert const is not None
