"""Unit tests for the ChaosTest strategy and ChaosSignalGenerator."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.chaos_test import ChaosSignalGenerator, create_chaos_test_strategy
from src.strategies.components.signal_generator import SignalDirection


def _make_ohlcv(n: int = 50, start_close: float = 100.0) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = start_close + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.random.randint(100, 1000, n).astype(float),
        }
    )


def _make_extreme_rsi_df(direction: str, n: int = 30) -> pd.DataFrame:
    """Create a DataFrame that produces extreme RSI values.

    Args:
        direction: 'oversold' for monotonically decreasing prices (RSI < 35),
                   'overbought' for monotonically increasing prices (RSI > 65).
    """
    if direction == "oversold":
        close = np.linspace(110, 90, n)
    else:
        close = np.linspace(90, 110, n)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 500.0),
        }
    )


class TestChaosSignalGenerator:
    """Tests for ChaosSignalGenerator."""

    def test_warmup_returns_hold(self):
        """During warmup period, generator returns HOLD."""
        gen = ChaosSignalGenerator(rsi_period=14)
        df = _make_ohlcv(30)

        for i in range(gen.warmup_period):
            signal = gen.generate_signal(df, i)
            assert signal.direction == SignalDirection.HOLD
            assert signal.metadata["trigger"] == "warmup"

    def test_warmup_period_matches_rsi_period(self):
        """Warmup period equals rsi_period + 1."""
        gen = ChaosSignalGenerator(rsi_period=10)
        assert gen.warmup_period == 11

    def test_rsi_oversold_generates_buy(self):
        """When RSI < buy threshold, generate BUY."""
        gen = ChaosSignalGenerator(rsi_buy_threshold=35.0, rsi_sell_threshold=65.0)
        df = _make_extreme_rsi_df("oversold", n=30)
        index = 25  # Well past warmup

        signal = gen.generate_signal(df, index)
        assert signal.direction == SignalDirection.BUY
        assert signal.metadata["trigger"] == "rsi_oversold"
        assert signal.confidence == 0.9

    def test_rsi_overbought_generates_sell(self):
        """When RSI > sell threshold, generate SELL."""
        gen = ChaosSignalGenerator(rsi_buy_threshold=35.0, rsi_sell_threshold=65.0)
        df = _make_extreme_rsi_df("overbought", n=30)
        index = 25

        signal = gen.generate_signal(df, index)
        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["trigger"] == "rsi_overbought"
        assert signal.metadata["enter_short"] is True
        assert signal.confidence == 0.9

    def test_forced_alternation_after_max_hold(self):
        """After max_hold_candles, direction flips regardless of RSI."""
        gen = ChaosSignalGenerator(max_hold_candles=3, rsi_period=14)
        df = _make_ohlcv(50)

        # Force an initial BUY
        gen._last_direction = SignalDirection.BUY
        gen._candles_in_position = 3  # At max hold

        signal = gen.generate_signal(df, 20)
        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["trigger"] == "forced_alternation"
        assert signal.metadata["enter_short"] is True

    def test_forced_alternation_flips_sell_to_buy(self):
        """Forced alternation flips SELL to BUY."""
        gen = ChaosSignalGenerator(max_hold_candles=2, rsi_period=14)
        df = _make_ohlcv(50)

        gen._last_direction = SignalDirection.SELL
        gen._candles_in_position = 2

        signal = gen.generate_signal(df, 20)
        assert signal.direction == SignalDirection.BUY
        assert signal.metadata["trigger"] == "forced_alternation"

    def test_hold_increments_candle_counter(self):
        """HOLD signals increment the candles_in_position counter."""
        gen = ChaosSignalGenerator(max_hold_candles=10, rsi_period=14)
        df = _make_ohlcv(50)

        # Set up as if in a position
        gen._last_direction = SignalDirection.BUY
        gen._candles_in_position = 1

        # Neutral RSI -- should HOLD and increment counter
        signal = gen.generate_signal(df, 20)
        if signal.direction == SignalDirection.HOLD:
            assert gen._candles_in_position == 2

    def test_same_direction_rsi_increments_counter(self):
        """Sustained RSI in same direction increments counter toward forced flip."""
        gen = ChaosSignalGenerator(max_hold_candles=3, rsi_period=14)
        df = _make_ohlcv(50)
        df["rsi"] = 20.0  # Force sustained oversold (BUY direction)

        # First BUY signal sets direction
        signal1 = gen.generate_signal(df, 20)
        assert signal1.direction == SignalDirection.BUY
        assert gen._candles_in_position == 1

        # Second BUY signal increments (same direction)
        signal2 = gen.generate_signal(df, 21)
        assert signal2.direction == SignalDirection.BUY
        assert gen._candles_in_position == 2

        # Third BUY signal increments to 3 (equals max_hold_candles)
        signal3 = gen.generate_signal(df, 22)
        assert signal3.direction == SignalDirection.BUY
        assert gen._candles_in_position == 3

        # Fourth candle: counter >= max_hold_candles, forced SELL
        signal4 = gen.generate_signal(df, 23)
        assert signal4.direction == SignalDirection.SELL
        assert signal4.metadata["trigger"] == "forced_alternation"

    def test_high_confidence_output(self):
        """All non-HOLD signals have 0.9 confidence."""
        gen = ChaosSignalGenerator(max_hold_candles=2, rsi_period=14)
        df = _make_ohlcv(50)

        gen._last_direction = SignalDirection.BUY
        gen._candles_in_position = 2

        signal = gen.generate_signal(df, 20)
        assert signal.confidence == 0.9

    def test_get_confidence_returns_fixed(self):
        """get_confidence() always returns 0.9."""
        gen = ChaosSignalGenerator()
        df = _make_ohlcv(30)
        assert gen.get_confidence(df, 20) == 0.9

    def test_invalid_thresholds_rejected(self):
        """buy threshold must be less than sell threshold."""
        with pytest.raises(ValueError, match="rsi_buy_threshold"):
            ChaosSignalGenerator(rsi_buy_threshold=70, rsi_sell_threshold=30)

    def test_invalid_max_hold_rejected(self):
        """max_hold_candles must be >= 1."""
        with pytest.raises(ValueError, match="max_hold_candles"):
            ChaosSignalGenerator(max_hold_candles=0)

    def test_invalid_rsi_period_rejected(self):
        """rsi_period must be >= 2."""
        with pytest.raises(ValueError, match="rsi_period"):
            ChaosSignalGenerator(rsi_period=1)

    def test_get_parameters(self):
        """get_parameters() includes all configuration."""
        gen = ChaosSignalGenerator(rsi_buy_threshold=30, rsi_sell_threshold=70, max_hold_candles=5)
        params = gen.get_parameters()

        assert params["rsi_buy_threshold"] == 30
        assert params["rsi_sell_threshold"] == 70
        assert params["max_hold_candles"] == 5
        assert params["last_direction"] == "hold"

    def test_precomputed_rsi_column_used(self):
        """If DataFrame has an 'rsi' column, use it instead of manual calculation."""
        gen = ChaosSignalGenerator(rsi_buy_threshold=40, rsi_period=14)
        df = _make_ohlcv(30)
        df["rsi"] = 20.0  # Force oversold

        signal = gen.generate_signal(df, 20)
        assert signal.direction == SignalDirection.BUY
        assert signal.metadata["rsi"] == 20.0


class TestCreateChaosTestStrategy:
    """Tests for the create_chaos_test_strategy factory function."""

    def test_default_creation(self):
        """Factory creates a valid strategy with default parameters."""
        strategy = create_chaos_test_strategy()

        assert strategy.name == "ChaosTest"
        assert hasattr(strategy, "signal_generator")
        assert hasattr(strategy, "risk_manager")
        assert hasattr(strategy, "position_sizer")

    def test_custom_name(self):
        """Factory accepts a custom strategy name."""
        strategy = create_chaos_test_strategy(name="MyChaos")
        assert strategy.name == "MyChaos"

    def test_risk_overrides_configured(self):
        """Risk overrides include partial operations and trailing stop."""
        strategy = create_chaos_test_strategy()
        overrides = strategy._risk_overrides

        assert overrides["position_sizer"] == "fixed_fraction"
        assert overrides["base_fraction"] == 0.02
        assert overrides["stop_loss_pct"] == 0.01
        assert overrides["take_profit_pct"] == 0.02

        # Partial operations (flat keys matching engine hydration)
        partial = overrides["partial_operations"]
        assert partial["exit_targets"] == [0.005, 0.01]
        assert partial["exit_sizes"] == [0.3, 0.3]
        assert partial["scale_in_thresholds"] == [-0.003]
        assert partial["scale_in_sizes"] == [0.5]
        assert partial["max_scale_ins"] == 1

        # Trailing stop (engine-expected keys)
        trailing = overrides["trailing_stop"]
        assert trailing["activation_threshold"] == 0.005
        assert trailing["trailing_distance_pct"] == 0.003

    def test_ignore_signal_reversal_metadata(self):
        """Strategy has ignore_signal_reversal flag set."""
        strategy = create_chaos_test_strategy()
        assert strategy._extra_metadata.get("ignore_signal_reversal") is True

    def test_signal_generator_is_chaos(self):
        """Factory uses ChaosSignalGenerator."""
        strategy = create_chaos_test_strategy()
        assert isinstance(strategy.signal_generator, ChaosSignalGenerator)

    def test_custom_max_hold_candles(self):
        """max_hold_candles parameter propagates to signal generator."""
        strategy = create_chaos_test_strategy(max_hold_candles=5)
        assert strategy.signal_generator.max_hold_candles == 5
