"""
Test per-strategy dynamic risk override functionality.
"""

from unittest.mock import Mock

import pandas as pd

from src.engines.backtest.engine import Backtester
from src.engines.live.trading_engine import LiveTradingEngine
from src.position_management.dynamic_risk import DynamicRiskConfig
from src.strategies.components import (
    Strategy,
    MLBasicSignalGenerator,
    FixedRiskManager,
    ConfidenceWeightedSizer,
)


def create_mock_strategy_with_overrides(name="test_strategy", overrides=None):
    """Create a mock strategy with risk overrides for testing"""
    signal_generator = MLBasicSignalGenerator()
    risk_manager = FixedRiskManager()
    position_sizer = ConfidenceWeightedSizer()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
    )

    # Attach risk overrides
    strategy._risk_overrides = overrides or {}

    # Add get_risk_overrides method
    def get_risk_overrides():
        return strategy._risk_overrides

    strategy.get_risk_overrides = get_risk_overrides

    return strategy


class TestStrategyOverrides:
    """Test per-strategy dynamic risk override functionality"""

    def test_strategy_dynamic_risk_overrides(self):
        """Test that strategy can override dynamic risk settings"""
        # Create strategy with custom dynamic risk settings
        overrides = {
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.03, 0.08, 0.15],  # More aggressive
                "risk_reduction_factors": [0.9, 0.7, 0.5],
                "recovery_thresholds": [0.01, 0.03],  # Lower recovery thresholds
                "volatility_adjustment_enabled": False,  # Disable volatility adjustments
                "performance_window_days": 14,  # Shorter window
            }
        }

        strategy = create_mock_strategy_with_overrides("aggressive_strategy", overrides)

        # Test the override method
        risk_overrides = strategy.get_risk_overrides()
        assert "dynamic_risk" in risk_overrides

        dynamic_config = risk_overrides["dynamic_risk"]
        assert dynamic_config["drawdown_thresholds"] == [0.03, 0.08, 0.15]
        assert dynamic_config["risk_reduction_factors"] == [0.9, 0.7, 0.5]
        assert dynamic_config["volatility_adjustment_enabled"] is False

    def test_live_engine_merge_strategy_overrides(self):
        """Test that live trading engine merges strategy overrides correctly"""
        # Create strategy with overrides
        overrides = {
            "dynamic_risk": {
                "drawdown_thresholds": [0.04, 0.09, 0.16],
                "risk_reduction_factors": [0.85, 0.65, 0.45],
                "performance_window_days": 20,
            }
        }
        strategy = create_mock_strategy_with_overrides("test_strategy", overrides)

        # Create base config
        base_config = DynamicRiskConfig(
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4],
            performance_window_days=30,
            volatility_adjustment_enabled=True,
        )

        # Mock LiveTradingEngine to test merge functionality
        engine = Mock()
        engine.strategy = strategy

        # Use the merge method from LiveTradingEngine
        merged_config = LiveTradingEngine._merge_dynamic_risk_config(engine, base_config)

        # Verify that strategy overrides were applied
        assert merged_config.drawdown_thresholds == [0.04, 0.09, 0.16]
        assert merged_config.risk_reduction_factors == [0.85, 0.65, 0.45]
        assert merged_config.performance_window_days == 20
        # Non-overridden values should remain from base config
        assert merged_config.volatility_adjustment_enabled is True

    def test_backtesting_engine_merge_strategy_overrides(self):
        """Test that backtesting engine merges strategy overrides correctly"""
        # Create strategy with partial overrides
        overrides = {
            "dynamic_risk": {
                "enabled": True,
                "volatility_adjustment_enabled": False,
                "high_volatility_threshold": 0.04,  # Higher threshold
            }
        }
        strategy = create_mock_strategy_with_overrides("conservative_strategy", overrides)

        # Create base config
        base_config = DynamicRiskConfig(
            enabled=False,  # Strategy should override this
            volatility_adjustment_enabled=True,  # Strategy should override this
            high_volatility_threshold=0.03,  # Strategy should override this
            low_volatility_threshold=0.01,  # Strategy should NOT override this
        )

        # Mock Backtester to test merge functionality
        backtester = Mock()

        # Use the merge method from Backtester
        merged_config = Backtester._merge_dynamic_risk_config(backtester, base_config, strategy)

        # Verify that strategy overrides were applied
        assert merged_config.enabled is True
        assert merged_config.volatility_adjustment_enabled is False
        assert merged_config.high_volatility_threshold == 0.04
        # Non-overridden values should remain from base config
        assert merged_config.low_volatility_threshold == 0.01

    def test_strategy_without_dynamic_risk_overrides(self):
        """Test strategy without dynamic risk overrides uses defaults"""
        # Create strategy without dynamic risk overrides
        overrides = {"position_sizer": "fixed_fraction", "base_fraction": 0.03}
        strategy = create_mock_strategy_with_overrides("normal_strategy", overrides)

        base_config = DynamicRiskConfig()

        # Mock engine
        engine = Mock()
        engine.strategy = strategy

        merged_config = LiveTradingEngine._merge_dynamic_risk_config(engine, base_config)

        # Should return the base config unchanged
        assert merged_config.drawdown_thresholds == base_config.drawdown_thresholds
        assert merged_config.risk_reduction_factors == base_config.risk_reduction_factors
        assert (
            merged_config.volatility_adjustment_enabled == base_config.volatility_adjustment_enabled
        )

    def test_strategy_empty_risk_overrides(self):
        """Test strategy with empty risk overrides"""
        strategy = create_mock_strategy_with_overrides("empty_strategy", {})

        base_config = DynamicRiskConfig(
            drawdown_thresholds=[0.05, 0.10], risk_reduction_factors=[0.8, 0.6]
        )

        # Mock engine
        engine = Mock()
        engine.strategy = strategy

        merged_config = LiveTradingEngine._merge_dynamic_risk_config(engine, base_config)

        # Should return the base config unchanged
        assert merged_config.drawdown_thresholds == [0.05, 0.10]
        assert merged_config.risk_reduction_factors == [0.8, 0.6]

    def test_invalid_strategy_overrides_graceful_handling(self):
        """Test that invalid strategy overrides are handled gracefully"""
        # Create strategy with invalid overrides
        overrides = {
            "dynamic_risk": {
                "drawdown_thresholds": "invalid",  # Should be list
                "risk_reduction_factors": [0.8, 0.6, 0.4, 0.2],  # Wrong length
                "invalid_field": True,  # Unknown field
            }
        }
        strategy = create_mock_strategy_with_overrides("invalid_strategy", overrides)

        base_config = DynamicRiskConfig()

        # Mock engine
        engine = Mock()
        engine.strategy = strategy

        # Should not raise exception, should fall back to base config
        merged_config = LiveTradingEngine._merge_dynamic_risk_config(engine, base_config)

        # Should have base config values due to error handling
        assert merged_config.drawdown_thresholds == base_config.drawdown_thresholds
        assert merged_config.risk_reduction_factors == base_config.risk_reduction_factors

    def test_strategy_override_validation(self):
        """Test that strategy overrides are validated properly"""
        # Create a DynamicRiskManager with strategy overrides
        overrides = {
            "dynamic_risk": {
                "drawdown_thresholds": [0.03, 0.08],
                "risk_reduction_factors": [0.9, 0.7],  # Matching length
                "recovery_thresholds": [0.02],
            }
        }
        create_mock_strategy_with_overrides("validated_strategy", overrides)

        # Extract the dynamic risk config
        dynamic_overrides = overrides["dynamic_risk"]

        # Create config with overrides (this should work)
        config = DynamicRiskConfig(
            drawdown_thresholds=dynamic_overrides["drawdown_thresholds"],
            risk_reduction_factors=dynamic_overrides["risk_reduction_factors"],
            recovery_thresholds=dynamic_overrides["recovery_thresholds"],
        )

        # Should not raise validation errors
        assert config.drawdown_thresholds == [0.03, 0.08]
        assert config.risk_reduction_factors == [0.9, 0.7]

    def test_complete_strategy_override_example(self):
        """Test a complete real-world example of strategy overrides"""
        # Create a strategy that wants more aggressive dynamic risk settings
        overrides = {
            "position_sizer": "confidence_weighted",
            "base_fraction": 0.025,
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.02, 0.05, 0.10],  # Earlier intervention
                "risk_reduction_factors": [0.9, 0.7, 0.5],  # Less aggressive reduction
                "recovery_thresholds": [0.01, 0.025],  # Quicker recovery
                "volatility_adjustment_enabled": True,
                "volatility_window_days": 14,  # Shorter volatility window
                "high_volatility_threshold": 0.025,  # More sensitive to volatility
                "low_volatility_threshold": 0.008,
                "performance_window_days": 21,  # 3-week performance window
            },
        }

        strategy = create_mock_strategy_with_overrides("advanced_strategy", overrides)

        # Verify all overrides are accessible
        risk_config = strategy.get_risk_overrides()["dynamic_risk"]

        assert risk_config["enabled"] is True
        assert risk_config["drawdown_thresholds"] == [0.02, 0.05, 0.10]
        assert risk_config["risk_reduction_factors"] == [0.9, 0.7, 0.5]
        assert risk_config["recovery_thresholds"] == [0.01, 0.025]
        assert risk_config["volatility_window_days"] == 14
        assert risk_config["performance_window_days"] == 21

    def test_strategy_can_disable_dynamic_risk(self):
        """Test that strategy can completely disable dynamic risk"""
        overrides = {"dynamic_risk": {"enabled": False}}
        strategy = create_mock_strategy_with_overrides("no_dynamic_risk_strategy", overrides)

        base_config = DynamicRiskConfig(enabled=True)  # Base has it enabled

        # Mock engine
        engine = Mock()
        engine.strategy = strategy

        merged_config = LiveTradingEngine._merge_dynamic_risk_config(engine, base_config)

        # Strategy should have disabled dynamic risk
        assert merged_config.enabled is False
