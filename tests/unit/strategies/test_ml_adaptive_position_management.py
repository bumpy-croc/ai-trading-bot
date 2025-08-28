"""
Unit tests for MlAdaptive strategy position management integration.
"""

from src.strategies.ml_adaptive import MlAdaptive


class TestMlAdaptivePositionManagement:
    """Test position management features integration in MlAdaptive strategy."""

    def test_risk_overrides_include_all_position_management_features(self):
        """Test that MlAdaptive includes all position management features in risk overrides."""
        strategy = MlAdaptive()
        overrides = strategy.get_risk_overrides()
        
        # Verify all position management features are present
        assert overrides is not None
        assert "dynamic_risk" in overrides
        assert "partial_operations" in overrides
        assert "trailing_stop" in overrides
        assert "time_exits" in overrides

    def test_dynamic_risk_configuration(self):
        """Test dynamic risk configuration is properly set."""
        strategy = MlAdaptive()
        overrides = strategy.get_risk_overrides()
        dynamic_risk = overrides["dynamic_risk"]
        
        # Verify dynamic risk is enabled
        assert dynamic_risk["enabled"] is True
        
        # Verify drawdown thresholds
        assert dynamic_risk["drawdown_thresholds"] == [0.05, 0.10, 0.15]
        assert dynamic_risk["risk_reduction_factors"] == [0.8, 0.6, 0.4]
        
        # Verify recovery thresholds
        assert dynamic_risk["recovery_thresholds"] == [0.02, 0.05]
        
        # Verify volatility settings
        assert dynamic_risk["volatility_adjustment_enabled"] is True
        assert dynamic_risk["high_volatility_threshold"] == 0.03
        assert dynamic_risk["low_volatility_threshold"] == 0.01
        assert dynamic_risk["volatility_risk_multipliers"] == (0.7, 1.3)

    def test_partial_operations_configuration(self):
        """Test partial operations configuration is properly set."""
        strategy = MlAdaptive()
        overrides = strategy.get_risk_overrides()
        partial_ops = overrides["partial_operations"]
        
        # Verify exit targets and sizes
        assert partial_ops["exit_targets"] == [0.03, 0.06, 0.10]
        assert partial_ops["exit_sizes"] == [0.25, 0.25, 0.50]
        
        # Verify scale-in configuration
        assert partial_ops["scale_in_thresholds"] == [0.02, 0.05]
        assert partial_ops["scale_in_sizes"] == [0.25, 0.25]
        assert partial_ops["max_scale_ins"] == 2

    def test_trailing_stop_configuration(self):
        """Test trailing stop configuration is properly set."""
        strategy = MlAdaptive()
        overrides = strategy.get_risk_overrides()
        trailing_stop = overrides["trailing_stop"]
        
        # Verify trailing stop settings
        assert trailing_stop["activation_threshold"] == 0.015
        assert trailing_stop["trailing_distance_pct"] == 0.005
        assert trailing_stop["breakeven_threshold"] == 0.02
        assert trailing_stop["breakeven_buffer"] == 0.001

    def test_time_exits_configuration(self):
        """Test time-based exits configuration is properly set for crypto trading."""
        strategy = MlAdaptive()
        overrides = strategy.get_risk_overrides()
        time_exits = overrides["time_exits"]
        
        # Verify crypto-appropriate time exit settings
        assert time_exits["max_holding_hours"] == 24
        assert time_exits["end_of_day_flat"] is False
        assert time_exits["weekend_flat"] is False
        assert time_exits["time_restrictions"]["no_overnight"] is False
        assert time_exits["time_restrictions"]["trading_hours_only"] is False
        assert time_exits["market_timezone"] == "UTC"

    def test_strategy_initialization_with_position_management(self):
        """Test that MlAdaptive initializes correctly with position management features."""
        strategy = MlAdaptive()
        
        # Verify basic strategy properties
        assert strategy.name == "MlAdaptive"
        assert strategy.trading_pair == "BTCUSDT"
        
        # Verify position management doesn't interfere with core functionality
        assert hasattr(strategy, "calculate_indicators")
        assert hasattr(strategy, "check_entry_conditions")
        assert hasattr(strategy, "check_exit_conditions")
        assert hasattr(strategy, "calculate_position_size")
        assert hasattr(strategy, "get_risk_overrides")