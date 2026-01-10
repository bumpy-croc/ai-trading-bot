"""
Unit tests for ensemble strategies - Component-based implementation
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.components import (
    ConfidenceWeightedSizer,
    MLBasicSignalGenerator,
    MLSignalGenerator,
    SignalDirection,
    Strategy,
    WeightedVotingSignalGenerator,
)
from src.strategies.ensemble_weighted import (
    BASE_POSITION_SIZE,
    MAX_POSITION_SIZE_RATIO,
    MIN_POSITION_SIZE_RATIO,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    create_ensemble_weighted_strategy,
)

pytestmark = pytest.mark.unit


class TestEnsembleWeighted:
    """Validate weighted ensemble strategy behaviour with component API."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="1h")
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price path with mild drift
        base_price = 30_000
        returns = np.random.normal(0, 0.01, len(dates))  # 1% hourly volatility
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "close": prices,
                "volume": np.random.uniform(100, 1000, len(dates)),
            },
            index=dates,
        )

        # Ensure candle integrity
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df

    def test_ensemble_weighted_initialization(self):
        """Ensure the factory returns a fully configured Strategy."""
        strategy = create_ensemble_weighted_strategy()

        assert isinstance(strategy, Strategy)
        assert strategy.name == "EnsembleWeighted"
        assert strategy.trading_pair == "BTCUSDT"
        assert isinstance(strategy.signal_generator, WeightedVotingSignalGenerator)

        generators = strategy.signal_generator.generators
        assert len(generators) >= 2
        assert any(isinstance(gen, MLBasicSignalGenerator) for gen in generators)
        assert any(isinstance(gen, MLSignalGenerator) for gen in generators)

        total_weight = sum(generators.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_ensemble_weighted_decision_interface(self, sample_data):
        """Strategy should produce TradingDecision objects via process_candle."""
        strategy = create_ensemble_weighted_strategy()
        balance = 10_000.0

        for index in [130, 150, 180]:
            decision = strategy.process_candle(sample_data, index, balance)

            assert decision.signal.direction in {
                SignalDirection.BUY,
                SignalDirection.SELL,
                SignalDirection.HOLD,
            }
            assert 0.0 <= decision.signal.confidence <= 1.0
            assert decision.position_size >= 0.0
            assert abs(decision.signal.strength) <= 1.0

    def test_ensemble_weighted_signal_metadata(self, sample_data):
        """Weighted voting metadata should reflect generator participation."""
        strategy = create_ensemble_weighted_strategy()
        balance = 10_000.0

        decision = strategy.process_candle(sample_data, 150, balance)
        metadata = decision.signal.metadata

        assert metadata["total_generators"] == len(strategy.signal_generator.generators)
        assert metadata["valid_signals"] <= metadata["total_generators"]
        assert "consensus_threshold" in metadata

    def test_ensemble_weighted_position_sizing(self, sample_data):
        """Position sizing should respect configured bounds."""
        strategy = create_ensemble_weighted_strategy()
        balance = 10_000.0

        decision = strategy.process_candle(sample_data, 150, balance)

        assert decision.position_size <= balance * MAX_POSITION_SIZE_RATIO
        if decision.signal.direction != SignalDirection.HOLD:
            assert decision.position_size > 0

    def test_ensemble_weighted_parameters(self):
        """Factory should expose component parameters via Strategy.get_parameters()."""
        strategy = create_ensemble_weighted_strategy()
        params = strategy.get_parameters()

        assert params["name"] == "EnsembleWeighted"
        assert params["stop_loss_pct"] == STOP_LOSS_PCT
        assert params["take_profit_pct"] == TAKE_PROFIT_PCT
        assert params["components"]["signal_generator"]["type"] == "WeightedVotingSignalGenerator"


class TestEnsembleOptimized:
    """Validate the aggressive ensemble configuration and risk controls."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing optimized features."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="1h")
        np.random.seed(42)

        base_price = 30_000
        returns = np.random.normal(0, 0.01, len(dates))
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "close": prices,
                "volume": np.random.uniform(100, 1000, len(dates)),
            },
            index=dates,
        )

        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df

    def test_optimized_position_sizing_configuration(self):
        """Position sizer should reflect aggressive allocations."""
        strategy = create_ensemble_weighted_strategy()

        assert isinstance(strategy.position_sizer, ConfidenceWeightedSizer)
        assert strategy.position_sizer.base_fraction == BASE_POSITION_SIZE
        assert strategy.min_position_size_ratio == MIN_POSITION_SIZE_RATIO
        assert strategy.max_position_size_ratio == MAX_POSITION_SIZE_RATIO

    def test_optimized_position_sizing_outputs(self, sample_data):
        """Calculated sizes should stay within configured bounds when trading."""
        strategy = create_ensemble_weighted_strategy()
        balance = 10_000.0

        decision = strategy.process_candle(sample_data, 150, balance)

        assert decision.position_size <= balance * MAX_POSITION_SIZE_RATIO
        if decision.signal.direction != SignalDirection.HOLD:
            assert decision.position_size > 0

    def test_optimized_risk_parameters(self):
        """Risk parameters should expose wider stops and trailing configuration."""
        strategy = create_ensemble_weighted_strategy()

        assert strategy.stop_loss_pct == STOP_LOSS_PCT
        assert strategy.take_profit_pct == TAKE_PROFIT_PCT

        risk_overrides = strategy.get_risk_overrides()
        assert risk_overrides is not None
        assert "trailing_stop" in risk_overrides
        assert risk_overrides["trailing_stop"]["activation_threshold"] == 0.04

    def test_enhanced_strategy_components(self):
        """Weighted voting ensemble should include ML-based generators."""
        strategy = create_ensemble_weighted_strategy()
        generators = strategy.signal_generator.generators

        assert any(isinstance(gen, MLBasicSignalGenerator) for gen in generators)
        assert any(isinstance(gen, MLSignalGenerator) for gen in generators)
        assert len(generators) >= 2
