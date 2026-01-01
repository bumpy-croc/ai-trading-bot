"""Tests for config.constants module."""

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


class TestTradingConstants:
    """Tests for trading-related constants."""

    def test_initial_balance_is_positive(self):
        """Test that default initial balance is positive."""
        assert DEFAULT_INITIAL_BALANCE > 0
        assert DEFAULT_INITIAL_BALANCE == 1000

    def test_max_holding_hours_is_reasonable(self):
        """Test that max holding hours is reasonable (14 days)."""
        assert DEFAULT_MAX_HOLDING_HOURS == 336  # 14 * 24
        assert DEFAULT_MAX_HOLDING_HOURS > 0


class TestPredictionConstants:
    """Tests for prediction-related constants."""

    def test_prediction_horizons_is_list(self):
        """Test that prediction horizons is a list."""
        assert isinstance(DEFAULT_PREDICTION_HORIZONS, list)
        assert len(DEFAULT_PREDICTION_HORIZONS) >= 1

    def test_confidence_threshold_in_valid_range(self):
        """Test that confidence threshold is between 0 and 1."""
        assert 0 < DEFAULT_MIN_CONFIDENCE_THRESHOLD <= 1
        assert DEFAULT_MIN_CONFIDENCE_THRESHOLD == 0.6

    def test_prediction_latency_is_positive(self):
        """Test that max prediction latency is positive."""
        assert DEFAULT_MAX_PREDICTION_LATENCY > 0
        assert DEFAULT_MAX_PREDICTION_LATENCY == 0.1

    def test_model_registry_path_is_string(self):
        """Test that model registry path is a valid string."""
        assert isinstance(DEFAULT_MODEL_REGISTRY_PATH, str)
        assert "models" in DEFAULT_MODEL_REGISTRY_PATH


class TestFeatureEngineeringConstants:
    """Tests for feature engineering constants."""

    def test_sequence_length_is_positive(self):
        """Test that sequence length is positive."""
        assert DEFAULT_SEQUENCE_LENGTH > 0
        assert DEFAULT_SEQUENCE_LENGTH == 120

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
        assert DEFAULT_MACD_FAST_PERIOD < DEFAULT_MACD_SLOW_PERIOD


class TestCacheConstants:
    """Tests for cache-related constants."""

    def test_feature_cache_ttl_is_positive(self):
        """Test that feature cache TTL is positive."""
        assert DEFAULT_FEATURE_CACHE_TTL > 0
        assert DEFAULT_FEATURE_CACHE_TTL == 3600  # 1 hour

    def test_model_cache_ttl_is_positive(self):
        """Test that model cache TTL is positive."""
        assert DEFAULT_MODEL_CACHE_TTL > 0
        assert DEFAULT_MODEL_CACHE_TTL == 600  # 10 minutes


class TestOptimizationConstants:
    """Tests for optimization-related constants."""

    def test_optimization_enabled_is_bool(self):
        """Test that optimization enabled is boolean."""
        assert isinstance(DEFAULT_OPTIMIZATION_ENABLED, bool)

    def test_performance_lookback_days_is_positive(self):
        """Test that performance lookback is positive."""
        assert DEFAULT_PERFORMANCE_LOOKBACK_DAYS > 0
        assert DEFAULT_PERFORMANCE_LOOKBACK_DAYS == 30


class TestErrorHandlingConstants:
    """Tests for error handling constants."""

    def test_error_cooldown_is_positive(self):
        """Test that error cooldown is positive."""
        assert DEFAULT_ERROR_COOLDOWN > 0
        assert DEFAULT_ERROR_COOLDOWN == 30


class TestRegimeConstants:
    """Tests for regime detection constants."""

    def test_regime_adjust_position_size_is_bool(self):
        """Test that regime adjust position size is boolean."""
        assert isinstance(DEFAULT_REGIME_ADJUST_POSITION_SIZE, bool)

    def test_regime_hysteresis_is_positive(self):
        """Test that regime hysteresis K is positive."""
        assert DEFAULT_REGIME_HYSTERESIS_K > 0
        assert DEFAULT_REGIME_HYSTERESIS_K == 3


class TestCheckIntervalConstants:
    """Tests for check interval constants."""

    def test_check_intervals_ordered(self):
        """Test that check intervals are properly ordered."""
        assert DEFAULT_MIN_CHECK_INTERVAL < DEFAULT_CHECK_INTERVAL
        assert DEFAULT_CHECK_INTERVAL < DEFAULT_MAX_CHECK_INTERVAL

    def test_check_intervals_positive(self):
        """Test that all check intervals are positive."""
        assert DEFAULT_MIN_CHECK_INTERVAL > 0
        assert DEFAULT_CHECK_INTERVAL > 0
        assert DEFAULT_MAX_CHECK_INTERVAL > 0


class TestDynamicRiskConstants:
    """Tests for dynamic risk management constants."""

    def test_dynamic_risk_enabled_is_bool(self):
        """Test that dynamic risk enabled is boolean."""
        assert isinstance(DEFAULT_DYNAMIC_RISK_ENABLED, bool)

    def test_drawdown_thresholds_ordered(self):
        """Test that drawdown thresholds are in ascending order."""
        assert len(DEFAULT_DRAWDOWN_THRESHOLDS) > 0
        for i in range(len(DEFAULT_DRAWDOWN_THRESHOLDS) - 1):
            assert DEFAULT_DRAWDOWN_THRESHOLDS[i] < DEFAULT_DRAWDOWN_THRESHOLDS[i + 1]

    def test_risk_reduction_factors_descending(self):
        """Test that risk reduction factors are in descending order."""
        assert len(DEFAULT_RISK_REDUCTION_FACTORS) > 0
        for i in range(len(DEFAULT_RISK_REDUCTION_FACTORS) - 1):
            assert DEFAULT_RISK_REDUCTION_FACTORS[i] > DEFAULT_RISK_REDUCTION_FACTORS[i + 1]

    def test_matching_threshold_and_factor_lengths(self):
        """Test that drawdown thresholds and risk factors have same length."""
        assert len(DEFAULT_DRAWDOWN_THRESHOLDS) == len(DEFAULT_RISK_REDUCTION_FACTORS)


class TestPartialOperationsConstants:
    """Tests for partial operations (exits/scale-ins) constants."""

    def test_partial_exit_targets_ordered(self):
        """Test that partial exit targets are in ascending order."""
        assert len(DEFAULT_PARTIAL_EXIT_TARGETS) > 0
        for i in range(len(DEFAULT_PARTIAL_EXIT_TARGETS) - 1):
            assert DEFAULT_PARTIAL_EXIT_TARGETS[i] < DEFAULT_PARTIAL_EXIT_TARGETS[i + 1]

    def test_partial_exit_sizes_sum_to_one(self):
        """Test that partial exit sizes sum to approximately 1."""
        total = sum(DEFAULT_PARTIAL_EXIT_SIZES)
        assert 0.99 <= total <= 1.01

    def test_matching_target_and_size_lengths(self):
        """Test that exit targets and sizes have same length."""
        assert len(DEFAULT_PARTIAL_EXIT_TARGETS) == len(DEFAULT_PARTIAL_EXIT_SIZES)


class TestTrailingStopConstants:
    """Tests for trailing stop constants."""

    def test_trailing_activation_threshold_positive(self):
        """Test that trailing activation threshold is positive."""
        assert DEFAULT_TRAILING_ACTIVATION_THRESHOLD > 0
        assert DEFAULT_TRAILING_ACTIVATION_THRESHOLD == 0.015  # 1.5%


class TestCorrelationConstants:
    """Tests for correlation control constants."""

    def test_correlation_threshold_in_valid_range(self):
        """Test that correlation threshold is between 0 and 1."""
        assert 0 < DEFAULT_CORRELATION_THRESHOLD <= 1
        assert DEFAULT_CORRELATION_THRESHOLD == 0.7

    def test_correlation_window_days_positive(self):
        """Test that correlation window is positive."""
        assert DEFAULT_CORRELATION_WINDOW_DAYS > 0
        assert DEFAULT_CORRELATION_WINDOW_DAYS == 30


class TestSentimentConstants:
    """Tests for sentiment-related constants."""

    def test_sentiment_disabled_by_default(self):
        """Test that sentiment is disabled by default."""
        assert DEFAULT_ENABLE_SENTIMENT is False


@pytest.mark.fast
class TestConstantsIntegrity:
    """Tests for overall constants integrity."""

    def test_no_none_values(self):
        """Test that important constants are not None."""
        constants = [
            DEFAULT_INITIAL_BALANCE,
            DEFAULT_PREDICTION_HORIZONS,
            DEFAULT_MIN_CONFIDENCE_THRESHOLD,
            DEFAULT_SEQUENCE_LENGTH,
            DEFAULT_RSI_PERIOD,
            DEFAULT_ATR_PERIOD,
            DEFAULT_CHECK_INTERVAL,
        ]
        for const in constants:
            assert const is not None

    def test_numeric_constants_types(self):
        """Test that numeric constants have correct types."""
        int_constants = [
            DEFAULT_SEQUENCE_LENGTH,
            DEFAULT_RSI_PERIOD,
            DEFAULT_ATR_PERIOD,
            DEFAULT_BOLLINGER_PERIOD,
            DEFAULT_MACD_FAST_PERIOD,
            DEFAULT_MACD_SLOW_PERIOD,
            DEFAULT_MACD_SIGNAL_PERIOD,
            DEFAULT_CHECK_INTERVAL,
            DEFAULT_CORRELATION_WINDOW_DAYS,
        ]
        for const in int_constants:
            assert isinstance(const, int), f"{const} should be int"

        float_constants = [
            DEFAULT_INITIAL_BALANCE,
            DEFAULT_MIN_CONFIDENCE_THRESHOLD,
            DEFAULT_TRAILING_ACTIVATION_THRESHOLD,
            DEFAULT_CORRELATION_THRESHOLD,
        ]
        for const in float_constants:
            assert isinstance(const, (int, float)), f"{const} should be numeric"
