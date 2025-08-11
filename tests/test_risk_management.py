"""
Tests for the risk management system.

Risk management is critical for capital preservation. Tests cover:
- Position sizing calculations
- Stop loss calculations
- Drawdown monitoring
- Daily risk limits
- Portfolio exposure limits
- Risk parameter validation
"""

import pytest

from risk.risk_manager import RiskManager, RiskParameters


class TestRiskParameters:
    """Test risk parameter validation and defaults"""

    def test_default_risk_parameters(self):
        """Test default risk parameters are reasonable"""
        params = RiskParameters()

        assert params.base_risk_per_trade == 0.02  # 2%
        assert params.max_risk_per_trade == 0.03  # 3%
        assert params.max_position_size == 0.25  # 25%
        assert params.max_daily_risk == 0.06  # 6%
        assert params.max_drawdown == 0.20  # 20%
        assert params.position_size_atr_multiplier == 1.0

    def test_custom_risk_parameters(self):
        """Test custom risk parameters"""
        params = RiskParameters(
            base_risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_position_size=0.10,
            max_daily_risk=0.04,
            max_drawdown=0.15,
        )

        assert params.base_risk_per_trade == 0.01
        assert params.max_risk_per_trade == 0.02
        assert params.max_position_size == 0.10
        assert params.max_daily_risk == 0.04
        assert params.max_drawdown == 0.15

    def test_risk_parameter_validation(self):
        """Test that risk parameters are logically consistent"""
        params = RiskParameters()

        # Base risk should be <= max risk
        assert params.base_risk_per_trade <= params.max_risk_per_trade

        # Daily risk should be reasonable multiple of per-trade risk
        assert params.max_daily_risk >= params.max_risk_per_trade

        # Position size should be reasonable
        assert 0 < params.max_position_size <= 1.0

        # Drawdown should be reasonable
        assert 0 < params.max_drawdown <= 1.0


class TestRiskManager:
    """Test the core RiskManager functionality"""

    def test_risk_manager_initialization(self, risk_parameters):
        """Test risk manager initialization"""
        risk_manager = RiskManager(risk_parameters)

        assert risk_manager.params == risk_parameters
        assert risk_manager.daily_risk_used == 0.0
        assert len(risk_manager.positions) == 0

    def test_risk_manager_with_default_parameters(self):
        """Test risk manager with default parameters"""
        risk_manager = RiskManager()

        assert risk_manager.params is not None
        assert isinstance(risk_manager.params, RiskParameters)

    @pytest.mark.risk_management
    def test_position_size_calculation_normal_regime(self, risk_parameters):
        """Test position size calculation in normal market regime"""
        risk_manager = RiskManager(risk_parameters)

        # Test parameters
        price = 50000
        atr = 1000  # 2% ATR
        balance = 10000

        position_size = risk_manager.calculate_position_size(
            price=price, atr=atr, balance=balance, regime="normal"
        )

        # Check position size is reasonable
        assert position_size > 0
        assert position_size <= balance * risk_parameters.max_position_size / price

        # Calculate expected position size
        risk_amount = balance * risk_parameters.base_risk_per_trade
        expected_size = risk_amount / atr
        max_size = balance * risk_parameters.max_position_size / price
        expected_size = min(expected_size, max_size)

        assert abs(position_size - expected_size) < 0.0001

    @pytest.mark.risk_management
    def test_position_size_calculation_trending_regime(self, risk_parameters):
        """Test position size calculation in trending market regime"""
        risk_manager = RiskManager(risk_parameters)

        # Use much larger ATR to avoid hitting position size caps
        position_size_trending = risk_manager.calculate_position_size(
            price=50000, atr=5000, balance=10000, regime="trending"
        )

        position_size_normal = risk_manager.calculate_position_size(
            price=50000, atr=5000, balance=10000, regime="normal"
        )

        # Trending regime should allow larger positions
        assert position_size_trending > position_size_normal

    @pytest.mark.risk_management
    def test_position_size_calculation_volatile_regime(self, risk_parameters):
        """Test position size calculation in volatile market regime"""
        risk_manager = RiskManager(risk_parameters)

        # Use much larger ATR to avoid hitting position size caps
        position_size_volatile = risk_manager.calculate_position_size(
            price=50000, atr=5000, balance=10000, regime="volatile"
        )

        position_size_normal = risk_manager.calculate_position_size(
            price=50000, atr=5000, balance=10000, regime="normal"
        )

        # Volatile regime should reduce position sizes
        assert position_size_volatile < position_size_normal

    @pytest.mark.risk_management
    def test_stop_loss_calculation(self, risk_parameters):
        """Test stop loss calculation"""
        risk_manager = RiskManager(risk_parameters)

        entry_price = 50000
        atr = 1000

        # Test long position stop loss
        stop_loss_long = risk_manager.calculate_stop_loss(
            entry_price=entry_price, atr=atr, side="long"
        )

        expected_stop_long = entry_price - (atr * risk_parameters.position_size_atr_multiplier)
        assert stop_loss_long == expected_stop_long
        assert stop_loss_long < entry_price

        # Test short position stop loss
        stop_loss_short = risk_manager.calculate_stop_loss(
            entry_price=entry_price, atr=atr, side="short"
        )

        expected_stop_short = entry_price + (atr * risk_parameters.position_size_atr_multiplier)
        assert stop_loss_short == expected_stop_short
        assert stop_loss_short > entry_price

    @pytest.mark.risk_management
    def test_drawdown_monitoring(self, risk_parameters):
        """Test drawdown monitoring"""
        risk_manager = RiskManager(risk_parameters)

        # Test no drawdown
        assert not risk_manager.check_drawdown(10000, 10000)

        # Test acceptable drawdown
        assert not risk_manager.check_drawdown(9000, 10000)  # 10% drawdown

        # Test excessive drawdown
        assert risk_manager.check_drawdown(7500, 10000)  # 25% drawdown (exceeds 20% limit)

        # Test edge case
        peak_balance = 10000
        max_allowed_drawdown = peak_balance * (1 - risk_parameters.max_drawdown)
        assert not risk_manager.check_drawdown(max_allowed_drawdown + 1, peak_balance)
        assert risk_manager.check_drawdown(max_allowed_drawdown - 1, peak_balance)

    @pytest.mark.risk_management
    def test_position_tracking(self, risk_parameters):
        """Test position tracking functionality"""
        risk_manager = RiskManager(risk_parameters)

        # Add positions
        risk_manager.update_position("BTCUSDT", "long", 0.1, 50000)
        risk_manager.update_position("ETHUSDT", "long", 0.2, 3000)

        assert len(risk_manager.positions) == 2
        assert "BTCUSDT" in risk_manager.positions
        assert "ETHUSDT" in risk_manager.positions

        # Check position details
        btc_position = risk_manager.positions["BTCUSDT"]
        assert btc_position["side"] == "long"
        assert btc_position["size"] == 0.1
        assert btc_position["entry_price"] == 50000

        # Close position
        risk_manager.close_position("BTCUSDT")
        assert len(risk_manager.positions) == 1
        assert "BTCUSDT" not in risk_manager.positions

    @pytest.mark.risk_management
    def test_total_exposure_calculation(self, risk_parameters):
        """Test total portfolio exposure calculation"""
        risk_manager = RiskManager(risk_parameters)

        # Add positions
        risk_manager.update_position("BTCUSDT", "long", 0.1, 50000)  # $5000 exposure
        risk_manager.update_position("ETHUSDT", "long", 0.2, 3000)  # $600 exposure

        total_exposure = risk_manager.get_total_exposure()
        expected_exposure = (0.1 * 50000) + (0.2 * 3000)  # 5000 + 600 = 5600

        assert total_exposure == expected_exposure

    @pytest.mark.risk_management
    def test_daily_risk_tracking(self, risk_parameters):
        """Test daily risk usage tracking"""
        risk_manager = RiskManager(risk_parameters)

        initial_daily_risk = risk_manager.daily_risk_used
        assert initial_daily_risk == 0.0

        # Add position which should increase daily risk usage
        risk_manager.update_position("BTCUSDT", "long", 0.1, 50000)

        # Daily risk should increase
        assert risk_manager.daily_risk_used > initial_daily_risk

        # Test daily risk reset
        risk_manager.reset_daily_risk()
        assert risk_manager.daily_risk_used == 0.0

    @pytest.mark.risk_management
    def test_maximum_position_size_enforcement(self, risk_parameters):
        """Test that maximum position size is enforced"""
        risk_manager = RiskManager(risk_parameters)

        # Try to calculate position size that would exceed maximum
        large_balance = 1000000  # $1M balance
        small_atr = 10  # Very small ATR to create large position

        position_size = risk_manager.calculate_position_size(
            price=50000, atr=small_atr, balance=large_balance, regime="normal"
        )

        max_allowed_position_value = large_balance * risk_parameters.max_position_size
        actual_position_value = position_size * 50000

        # Position value should not exceed maximum
        assert (
            actual_position_value <= max_allowed_position_value + 0.01
        )  # Small tolerance for rounding


class TestRiskScenarios:
    """Test risk management in various market scenarios"""

    @pytest.mark.risk_management
    def test_high_volatility_scenario(self, risk_parameters):
        """Test risk management during high volatility"""
        risk_manager = RiskManager(risk_parameters)

        # High volatility scenario (large ATR)
        high_atr = 5000  # 10% ATR - very volatile
        normal_atr = 1000  # 2% ATR - normal

        position_high_vol = risk_manager.calculate_position_size(
            price=50000, atr=high_atr, balance=10000, regime="volatile"
        )

        position_normal_vol = risk_manager.calculate_position_size(
            price=50000, atr=normal_atr, balance=10000, regime="normal"
        )

        # High volatility should result in smaller positions
        assert position_high_vol < position_normal_vol

    @pytest.mark.risk_management
    def test_trending_market_scenario(self, risk_parameters):
        """Test risk management during trending markets"""
        risk_manager = RiskManager(risk_parameters)

        # Trending markets should allow larger positions but still respect limits
        position_size = risk_manager.calculate_position_size(
            price=50000, atr=1000, balance=10000, regime="trending"
        )

        max_position_value = 10000 * risk_parameters.max_position_size
        actual_position_value = position_size * 50000

        # Should not exceed maximum even in trending markets
        assert actual_position_value <= max_position_value

    @pytest.mark.risk_management
    def test_multiple_correlated_positions(self, risk_parameters):
        """Test risk management with multiple correlated positions"""
        risk_manager = RiskManager(risk_parameters)

        # Add multiple BTC-related positions (simulating correlation)
        btc_symbols = ["BTCUSDT", "BTCETH", "BTCBUSD"]

        for symbol in btc_symbols:
            risk_manager.update_position(symbol, "long", 0.1, 50000)

        # Calculate correlation risk
        correlation_risk = risk_manager.get_position_correlation_risk(btc_symbols)
        expected_risk = 3 * 0.1 * 50000  # 3 positions of $5000 each

        # Use approximate comparison due to floating point precision
        assert abs(correlation_risk - expected_risk) < 0.01

    @pytest.mark.risk_management
    def test_daily_risk_limit_enforcement(self, risk_parameters):
        """Test that daily risk limits are enforced"""
        risk_manager = RiskManager(risk_parameters)

        # Simulate using most of daily risk allocation
        risk_manager.daily_risk_used = risk_parameters.max_daily_risk * 0.9  # 90% used

        # Try to calculate new position
        position_size = risk_manager.calculate_position_size(
            price=50000, atr=1000, balance=10000, regime="normal"
        )

        # Position should be smaller due to limited remaining daily risk
        remaining_daily_risk = risk_parameters.max_daily_risk - risk_manager.daily_risk_used
        max_risk_amount = 10000 * remaining_daily_risk
        max_position_size = max_risk_amount / 1000  # ATR-based calculation

        assert position_size <= max_position_size


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.risk_management
    def test_zero_balance(self, risk_parameters):
        """Test behavior with zero balance"""
        risk_manager = RiskManager(risk_parameters)

        position_size = risk_manager.calculate_position_size(
            price=50000, atr=1000, balance=0, regime="normal"
        )

        assert position_size == 0

    @pytest.mark.risk_management
    def test_zero_atr(self, risk_parameters):
        """Test behavior with zero ATR"""
        risk_manager = RiskManager(risk_parameters)

        # Zero ATR should not cause division by zero
        try:
            position_size = risk_manager.calculate_position_size(
                price=50000, atr=0, balance=10000, regime="normal"
            )
            # If it doesn't crash, position should be 0 or very small
            assert position_size >= 0
        except ZeroDivisionError:
            pytest.fail("Zero ATR caused division by zero error")

    @pytest.mark.risk_management
    def test_negative_values(self, risk_parameters):
        """Test behavior with negative input values"""
        risk_manager = RiskManager(risk_parameters)

        # Negative price
        position_size = risk_manager.calculate_position_size(
            price=-50000, atr=1000, balance=10000, regime="normal"
        )
        assert position_size >= 0  # Should handle gracefully

        # Negative ATR
        position_size = risk_manager.calculate_position_size(
            price=50000, atr=-1000, balance=10000, regime="normal"
        )
        assert position_size >= 0  # Should handle gracefully

    @pytest.mark.risk_management
    def test_very_large_values(self, risk_parameters):
        """Test behavior with very large input values"""
        risk_manager = RiskManager(risk_parameters)

        # Very large balance
        position_size = risk_manager.calculate_position_size(
            price=50000,
            atr=1000,
            balance=1e12,
            regime="normal",  # $1 trillion
        )

        # Should still respect percentage limits
        max_position_value = 1e12 * risk_parameters.max_position_size
        actual_position_value = position_size * 50000
        assert actual_position_value <= max_position_value * 1.01  # Small tolerance

    @pytest.mark.risk_management
    def test_unknown_regime(self, risk_parameters):
        """Test behavior with unknown market regime"""
        risk_manager = RiskManager(risk_parameters)

        # Unknown regime should default to normal behavior
        position_size_unknown = risk_manager.calculate_position_size(
            price=50000, atr=1000, balance=10000, regime="unknown_regime"
        )

        position_size_normal = risk_manager.calculate_position_size(
            price=50000, atr=1000, balance=10000, regime="normal"
        )

        # Should behave like normal regime
        assert position_size_unknown == position_size_normal
