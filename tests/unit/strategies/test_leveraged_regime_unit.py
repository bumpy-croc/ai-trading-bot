"""Tests for leveraged_regime strategy factory."""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.components.leverage_manager import LeverageManager
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
