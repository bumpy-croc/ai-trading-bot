"""Tests for leveraged_regime strategy factory."""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.components.leverage_manager import LeverageManager
from src.strategies.components.position_sizer import LeveragedPositionSizer
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel
from src.strategies.components.signal_generator import Signal, SignalDirection
from src.strategies.leveraged_regime import create_leveraged_regime_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a realistic OHLCV dataframe for strategy testing."""
    periods = 100
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    # Simulate an uptrend with some noise
    base = 40000.0
    closes = [base + i * 10 + (i % 5) * 5 for i in range(periods)]
    return pd.DataFrame(
        {
            "open": [c - 5 for c in closes],
            "high": [c + 15 for c in closes],
            "low": [c - 15 for c in closes],
            "close": closes,
            "volume": [100 + i for i in range(periods)],
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


class TestCreateLeveragedRegimeStrategy:
    """Test the strategy factory function."""

    def test_default_creation(self) -> None:
        strategy = create_leveraged_regime_strategy()
        assert strategy.name == "LeveragedRegime"
        assert hasattr(strategy, "leverage_manager")
        assert isinstance(strategy.leverage_manager, LeverageManager)
        assert isinstance(strategy.position_sizer, LeveragedPositionSizer)

    def test_custom_name(self) -> None:
        strategy = create_leveraged_regime_strategy(name="MyLevStrategy")
        assert strategy.name == "MyLevStrategy"

    def test_momentum_signal_source(self) -> None:
        strategy = create_leveraged_regime_strategy(signal_source="momentum")
        assert "LeveragedRegime_signals" in strategy.signal_generator.name

    def test_ml_signal_source(self) -> None:
        strategy = create_leveraged_regime_strategy(signal_source="ml")
        assert "LeveragedRegime_signals" in strategy.signal_generator.name

    def test_custom_leverage_params(self) -> None:
        strategy = create_leveraged_regime_strategy(
            max_leverage=2.0,
            leverage_decay_rate=0.3,
            min_regime_bars=10,
        )
        lm = strategy.leverage_manager
        assert lm.max_leverage == 2.0
        assert lm.decay_rate == 0.3
        assert lm.min_regime_bars == 10

    def test_risk_overrides_include_leverage(self) -> None:
        strategy = create_leveraged_regime_strategy(max_leverage=2.5)
        overrides = strategy.get_risk_overrides()
        assert overrides["leverage"]["enabled"] is True
        assert overrides["leverage"]["max_leverage"] == 2.5

    def test_risk_overrides_include_trailing_stop(self) -> None:
        strategy = create_leveraged_regime_strategy()
        overrides = strategy.get_risk_overrides()
        assert "trailing_stop" in overrides
        assert overrides["trailing_stop"]["activation_threshold"] > 0

    def test_risk_overrides_include_dynamic_risk(self) -> None:
        strategy = create_leveraged_regime_strategy()
        overrides = strategy.get_risk_overrides()
        assert overrides["dynamic_risk"]["enabled"] is True

    def test_base_position_size_exposed(self) -> None:
        strategy = create_leveraged_regime_strategy(base_fraction=0.15)
        assert strategy.base_position_size == 0.15

    def test_take_profit_exposed(self) -> None:
        strategy = create_leveraged_regime_strategy(take_profit_pct=0.08)
        assert strategy.take_profit_pct == 0.08

    def test_invalid_signal_source_raises(self) -> None:
        with pytest.raises(ValueError, match="signal_source must be one of"):
            create_leveraged_regime_strategy(signal_source="invalid")

    def test_max_fraction_capped_at_050(self) -> None:
        """max_fraction in risk_overrides should not exceed 0.50."""
        strategy = create_leveraged_regime_strategy(base_fraction=0.20, max_leverage=3.0)
        overrides = strategy.get_risk_overrides()
        assert overrides["max_fraction"] <= 0.50

    def test_position_sizer_is_leveraged(self) -> None:
        """Strategy should use LeveragedPositionSizer, not bare sizer."""
        strategy = create_leveraged_regime_strategy()
        assert isinstance(strategy.position_sizer, LeveragedPositionSizer)
        assert strategy.position_sizer.name == "leveraged_position_sizer"


# ---------------------------------------------------------------------------
# Integration: process_candle
# ---------------------------------------------------------------------------


class TestLeveragedRegimeProcessCandle:
    """Test that the strategy can process candles without errors."""

    def test_process_candle_returns_decision(self, sample_dataframe: pd.DataFrame) -> None:
        """Strategy should produce a valid TradingDecision."""
        strategy = create_leveraged_regime_strategy()
        idx = len(sample_dataframe) - 1
        decision = strategy.process_candle(sample_dataframe, idx, 10000.0)
        assert decision is not None
        assert decision.signal is not None
        assert decision.position_size >= 0

    def test_process_candle_with_ml_source(self, sample_dataframe: pd.DataFrame) -> None:
        """ML signal source should also produce decisions."""
        strategy = create_leveraged_regime_strategy(signal_source="ml")
        idx = len(sample_dataframe) - 1
        decision = strategy.process_candle(sample_dataframe, idx, 10000.0)
        assert decision is not None

    def test_multiple_candles(self, sample_dataframe: pd.DataFrame) -> None:
        """Processing multiple candles sequentially should not error."""
        strategy = create_leveraged_regime_strategy()
        decisions = []
        for idx in range(50, len(sample_dataframe)):
            d = strategy.process_candle(sample_dataframe, idx, 10000.0)
            decisions.append(d)
        assert len(decisions) == 50
        # All decisions should have non-negative position sizes
        assert all(d.position_size >= 0 for d in decisions)


# ---------------------------------------------------------------------------
# Leverage Manager Attachment
# ---------------------------------------------------------------------------


class TestLeverageManagerAttachment:
    """Test that leverage manager is properly attached to strategy."""

    def test_leverage_manager_is_accessible(self) -> None:
        strategy = create_leveraged_regime_strategy()
        assert hasattr(strategy, "leverage_manager")
        lm = strategy.leverage_manager
        assert isinstance(lm, LeverageManager)

    def test_leverage_manager_params_match(self) -> None:
        strategy = create_leveraged_regime_strategy(
            max_leverage=2.0,
            leverage_decay_rate=0.25,
            min_regime_bars=7,
        )
        lm = strategy.leverage_manager
        params = lm.get_parameters()
        assert params["max_leverage"] == 2.0
        assert params["decay_rate"] == 0.25
        assert params["min_regime_bars"] == 7


# ---------------------------------------------------------------------------
# Leverage Actually Scales Positions (P0 verification)
# ---------------------------------------------------------------------------


class TestLeverageActuallyScalesPositions:
    """Verify that leverage is wired into the pipeline and affects position sizes."""

    def test_bull_regime_amplifies_vs_bear(self) -> None:
        """Bull regime should produce larger positions than bear regime."""
        strategy = create_leveraged_regime_strategy(
            max_leverage=2.5,
            leverage_decay_rate=1.0,  # instant transitions for test clarity
            min_regime_bars=0,
        )

        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            confidence=0.8,
            metadata={},
        )
        balance = 10000.0
        risk_amount = 500.0

        bull_regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.9,
            duration=30,
            strength=0.8,
        )
        bear_regime = RegimeContext(
            trend=TrendLabel.TREND_DOWN,
            volatility=VolLabel.HIGH,
            confidence=0.9,
            duration=30,
            strength=0.8,
        )

        bull_size = strategy.position_sizer.calculate_size(
            signal, balance, risk_amount, bull_regime
        )
        bear_size = strategy.position_sizer.calculate_size(
            signal, balance, risk_amount, bear_regime
        )

        # Bull should be meaningfully larger than bear
        assert bull_size > bear_size
        # Bull should be amplified above base size (leverage > 1)
        assert bull_size > 0
        # Bear should be reduced (leverage near 0)
        # Bear with high vol target is 0.0, so leveraged size should be ~0
        assert bear_size < bull_size * 0.5

    def test_leveraged_sizer_produces_larger_than_base(self) -> None:
        """LeveragedPositionSizer should produce larger sizes than bare sizer in bull."""
        strategy = create_leveraged_regime_strategy(
            max_leverage=2.5,
            leverage_decay_rate=1.0,
            min_regime_bars=0,
        )

        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            confidence=0.8,
            metadata={},
        )
        balance = 10000.0
        risk_amount = 500.0
        bull_regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.9,
            duration=30,
            strength=0.8,
        )

        leveraged_sizer = strategy.position_sizer
        base_sizer = leveraged_sizer.base_sizer

        base_size = base_sizer.calculate_size(signal, balance, risk_amount, bull_regime)
        leveraged_size = leveraged_sizer.calculate_size(signal, balance, risk_amount, bull_regime)

        # Leveraged size should be larger due to bull leverage > 1.0
        assert leveraged_size > base_size
