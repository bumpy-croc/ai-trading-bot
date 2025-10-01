"""
Unit tests for RiskManager components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.components.risk_manager import (
    RiskManager, Position, MarketData, FixedRiskManager
)
from src.strategies.components.signal_generator import Signal, SignalDirection


class TestPosition:
    """Test Position dataclass"""
    
    def test_position_creation_valid(self):
        """Test creating a valid position"""
        entry_time = datetime.now()
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=1.5,
            entry_price=50000.0,
            current_price=51000.0,
            entry_time=entry_time
        )
        
        assert position.symbol == "BTCUSDT"
        assert position.side == "long"
        assert position.size == 1.5
        assert position.entry_price == 50000.0
        assert position.current_price == 51000.0
        assert position.entry_time == entry_time
        assert position.unrealized_pnl == 0.0
        assert position.realized_pnl == 0.0
    
    def test_position_validation_symbol(self):
        """Test position symbol validation"""
        with pytest.raises(ValueError, match="symbol must be a non-empty string"):
            Position(
                symbol="",
                side="long",
                size=1.0,
                entry_price=100.0,
                current_price=100.0,
                entry_time=datetime.now()
            )
    
    def test_position_validation_side(self):
        """Test position side validation"""
        with pytest.raises(ValueError, match="side must be 'long' or 'short'"):
            Position(
                symbol="BTCUSDT",
                side="invalid",
                size=1.0,
                entry_price=100.0,
                current_price=100.0,
                entry_time=datetime.now()
            )
    
    def test_position_validation_size(self):
        """Test position size validation"""
        with pytest.raises(ValueError, match="size must be positive"):
            Position(
                symbol="BTCUSDT",
                side="long",
                size=-1.0,
                entry_price=100.0,
                current_price=100.0,
                entry_time=datetime.now()
            )
    
    def test_position_validation_entry_price(self):
        """Test position entry price validation"""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            Position(
                symbol="BTCUSDT",
                side="long",
                size=1.0,
                entry_price=-100.0,
                current_price=100.0,
                entry_time=datetime.now()
            )
    
    def test_position_validation_current_price(self):
        """Test position current price validation"""
        with pytest.raises(ValueError, match="current_price must be positive"):
            Position(
                symbol="BTCUSDT",
                side="long",
                size=1.0,
                entry_price=100.0,
                current_price=-100.0,
                entry_time=datetime.now()
            )
    
    def test_update_current_price_long(self):
        """Test updating current price for long position"""
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
            entry_price=50000.0,
            current_price=50000.0,
            entry_time=datetime.now()
        )
        
        # Price goes up - profit
        position.update_current_price(51000.0)
        assert position.current_price == 51000.0
        assert position.unrealized_pnl == 1000.0
        
        # Price goes down - loss
        position.update_current_price(49000.0)
        assert position.current_price == 49000.0
        assert position.unrealized_pnl == -1000.0
    
    def test_update_current_price_short(self):
        """Test updating current price for short position"""
        position = Position(
            symbol="BTCUSDT",
            side="short",
            size=1.0,
            entry_price=50000.0,
            current_price=50000.0,
            entry_time=datetime.now()
        )
        
        # Price goes down - profit for short
        position.update_current_price(49000.0)
        assert position.current_price == 49000.0
        assert position.unrealized_pnl == 1000.0
        
        # Price goes up - loss for short
        position.update_current_price(51000.0)
        assert position.current_price == 51000.0
        assert position.unrealized_pnl == -1000.0
    
    def test_update_current_price_invalid(self):
        """Test updating current price with invalid value"""
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
            entry_price=50000.0,
            current_price=50000.0,
            entry_time=datetime.now()
        )
        
        with pytest.raises(ValueError, match="price must be positive"):
            position.update_current_price(-100.0)
    
    def test_get_total_pnl(self):
        """Test total PnL calculation"""
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
            entry_price=50000.0,
            current_price=51000.0,
            entry_time=datetime.now(),
            realized_pnl=500.0
        )
        
        position.update_current_price(52000.0)  # 2000 unrealized
        
        assert position.get_total_pnl() == 2500.0  # 500 realized + 2000 unrealized
    
    def test_get_pnl_percentage(self):
        """Test PnL percentage calculation"""
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=2.0,
            entry_price=50000.0,
            current_price=50000.0,
            entry_time=datetime.now()
        )
        
        position.update_current_price(55000.0)  # 10000 profit on 100000 entry value
        
        assert position.get_pnl_percentage() == 10.0


class TestMarketData:
    """Test MarketData dataclass"""
    
    def test_market_data_creation_valid(self):
        """Test creating valid market data"""
        timestamp = datetime.now()
        market_data = MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1000.0,
            bid=49950.0,
            ask=50050.0,
            timestamp=timestamp,
            volatility=0.02
        )
        
        assert market_data.symbol == "BTCUSDT"
        assert market_data.price == 50000.0
        assert market_data.volume == 1000.0
        assert market_data.bid == 49950.0
        assert market_data.ask == 50050.0
        assert market_data.timestamp == timestamp
        assert market_data.volatility == 0.02
    
    def test_market_data_validation_symbol(self):
        """Test market data symbol validation"""
        with pytest.raises(ValueError, match="symbol must be a non-empty string"):
            MarketData(symbol="", price=100.0, volume=1000.0)
    
    def test_market_data_validation_price(self):
        """Test market data price validation"""
        with pytest.raises(ValueError, match="price must be positive"):
            MarketData(symbol="BTCUSDT", price=-100.0, volume=1000.0)
    
    def test_market_data_validation_volume(self):
        """Test market data volume validation"""
        with pytest.raises(ValueError, match="volume must be non-negative"):
            MarketData(symbol="BTCUSDT", price=100.0, volume=-1000.0)
    
    def test_market_data_validation_bid(self):
        """Test market data bid validation"""
        with pytest.raises(ValueError, match="bid must be positive when provided"):
            MarketData(symbol="BTCUSDT", price=100.0, volume=1000.0, bid=-50.0)
    
    def test_market_data_validation_ask(self):
        """Test market data ask validation"""
        with pytest.raises(ValueError, match="ask must be positive when provided"):
            MarketData(symbol="BTCUSDT", price=100.0, volume=1000.0, ask=-50.0)
    
    def test_market_data_validation_volatility(self):
        """Test market data volatility validation"""
        with pytest.raises(ValueError, match="volatility must be non-negative when provided"):
            MarketData(symbol="BTCUSDT", price=100.0, volume=1000.0, volatility=-0.1)
    
    def test_get_spread(self):
        """Test bid-ask spread calculation"""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1000.0,
            bid=49950.0,
            ask=50050.0
        )
        
        assert market_data.get_spread() == 100.0
    
    def test_get_spread_missing_data(self):
        """Test spread calculation with missing bid/ask"""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1000.0
        )
        
        assert market_data.get_spread() is None
    
    def test_get_spread_percentage(self):
        """Test bid-ask spread percentage calculation"""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1000.0,
            bid=49950.0,
            ask=50050.0
        )
        
        # Spread is 100, mid price is 50000, so percentage is 0.2%
        assert abs(market_data.get_spread_percentage() - 0.2) < 0.001


class MockRiskManager(RiskManager):
    """Mock risk manager for testing abstract base class"""
    
    def calculate_position_size(self, signal, balance, regime=None):
        return balance * 0.02
    
    def should_exit(self, position, current_data, regime=None):
        return False
    
    def get_stop_loss(self, entry_price, signal, regime=None):
        return entry_price * 0.95


class TestRiskManager:
    """Test RiskManager abstract base class"""
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        manager = MockRiskManager("test_manager")
        assert manager.name == "test_manager"
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs"""
        manager = MockRiskManager("test")
        
        # Should not raise any exception
        manager.validate_inputs(1000.0)
    
    def test_validate_inputs_invalid_balance(self):
        """Test input validation with invalid balance"""
        manager = MockRiskManager("test")
        
        with pytest.raises(ValueError, match="balance must be positive"):
            manager.validate_inputs(-1000.0)
        
        with pytest.raises(ValueError, match="balance must be positive"):
            manager.validate_inputs(0.0)
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        manager = MockRiskManager("test_manager")
        params = manager.get_parameters()
        
        assert params['name'] == "test_manager"
        assert params['type'] == "MockRiskManager"


class TestFixedRiskManager:
    """Test FixedRiskManager implementation"""
    
    def create_test_signal(self, direction=SignalDirection.BUY, strength=0.8, confidence=0.9):
        """Create test signal"""
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={}
        )
    
    def test_fixed_risk_manager_initialization_default(self):
        """Test FixedRiskManager initialization with defaults"""
        manager = FixedRiskManager()
        assert manager.name == "fixed_risk_manager"
        assert manager.risk_per_trade == 0.02
        assert manager.stop_loss_pct == 0.05
    
    def test_fixed_risk_manager_initialization_custom(self):
        """Test FixedRiskManager initialization with custom parameters"""
        manager = FixedRiskManager(risk_per_trade=0.03, stop_loss_pct=0.08)
        assert manager.risk_per_trade == 0.03
        assert manager.stop_loss_pct == 0.08
    
    def test_fixed_risk_manager_validation_risk_per_trade(self):
        """Test risk_per_trade validation"""
        with pytest.raises(ValueError, match="risk_per_trade must be between 0.001 and 0.1"):
            FixedRiskManager(risk_per_trade=0.0005)
        
        with pytest.raises(ValueError, match="risk_per_trade must be between 0.001 and 0.1"):
            FixedRiskManager(risk_per_trade=0.15)
    
    def test_fixed_risk_manager_validation_stop_loss_pct(self):
        """Test stop_loss_pct validation"""
        with pytest.raises(ValueError, match="stop_loss_pct must be between 0.01 and 0.5"):
            FixedRiskManager(stop_loss_pct=0.005)
        
        with pytest.raises(ValueError, match="stop_loss_pct must be between 0.01 and 0.5"):
            FixedRiskManager(stop_loss_pct=0.6)
    
    def test_calculate_position_size_buy_signal(self):
        """Test position size calculation for BUY signal"""
        manager = FixedRiskManager(risk_per_trade=0.02)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        
        position_size = manager.calculate_position_size(signal, balance)
        
        # Base: 10000 * 0.02 = 200
        # Confidence adj: 200 * 0.9 = 180
        # Strength adj: 180 * 0.8 = 144
        # Should be between min (10) and max (1000)
        assert 10.0 <= position_size <= 1000.0
        assert position_size == 144.0
    
    def test_calculate_position_size_hold_signal(self):
        """Test position size calculation for HOLD signal"""
        manager = FixedRiskManager()
        signal = self.create_test_signal(SignalDirection.HOLD, 0.0, 1.0)
        balance = 10000.0
        
        position_size = manager.calculate_position_size(signal, balance)
        
        assert position_size == 0.0
    
    def test_calculate_position_size_low_confidence(self):
        """Test position size calculation with low confidence"""
        manager = FixedRiskManager(risk_per_trade=0.02)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.05)  # Very low confidence
        balance = 10000.0
        
        position_size = manager.calculate_position_size(signal, balance)
        
        # Should use minimum confidence multiplier (0.1)
        # Base: 10000 * 0.02 = 200
        # Confidence adj: 200 * 0.1 = 20
        # Strength adj: 20 * 0.8 = 16
        assert position_size == 16.0
    
    def test_calculate_position_size_with_regime(self):
        """Test position size calculation with regime context"""
        from src.regime.detector import TrendLabel, VolLabel
        from src.strategies.components.regime_context import RegimeContext
        
        manager = FixedRiskManager(risk_per_trade=0.02)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        
        # High volatility regime should reduce position size
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.HIGH,
            confidence=0.8,
            duration=10,
            strength=0.7
        )
        
        position_size = manager.calculate_position_size(signal, balance, regime)
        
        # Should be reduced due to high volatility
        base_size = 144.0  # From previous test
        expected_size = base_size * 0.7  # High vol multiplier
        assert abs(position_size - expected_size) < 0.1
    
    def test_should_exit_no_exit(self):
        """Test should_exit when position is profitable"""
        manager = FixedRiskManager(stop_loss_pct=0.05)
        
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
            entry_price=50000.0,
            current_price=52000.0,  # Profitable
            entry_time=datetime.now()
        )
        position.update_current_price(52000.0)
        
        market_data = MarketData(
            symbol="BTCUSDT",
            price=52000.0,
            volume=1000.0
        )
        
        assert not manager.should_exit(position, market_data)
    
    def test_should_exit_stop_loss_triggered(self):
        """Test should_exit when stop loss is triggered"""
        manager = FixedRiskManager(stop_loss_pct=0.05)
        
        position = Position(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
            entry_price=50000.0,
            current_price=47000.0,  # 6% loss > 5% stop loss
            entry_time=datetime.now()
        )
        position.update_current_price(47000.0)
        
        market_data = MarketData(
            symbol="BTCUSDT",
            price=47000.0,
            volume=1000.0
        )
        
        assert manager.should_exit(position, market_data)
    
    def test_get_stop_loss_buy_signal(self):
        """Test stop loss calculation for BUY signal"""
        manager = FixedRiskManager(stop_loss_pct=0.05)
        signal = self.create_test_signal(SignalDirection.BUY)
        
        stop_loss = manager.get_stop_loss(50000.0, signal)
        
        expected_stop_loss = 50000.0 * 0.95  # 5% below entry
        assert stop_loss == expected_stop_loss
    
    def test_get_stop_loss_sell_signal(self):
        """Test stop loss calculation for SELL signal"""
        manager = FixedRiskManager(stop_loss_pct=0.05)
        signal = self.create_test_signal(SignalDirection.SELL)
        
        stop_loss = manager.get_stop_loss(50000.0, signal)
        
        expected_stop_loss = 50000.0 * 1.05  # 5% above entry for short
        assert stop_loss == expected_stop_loss
    
    def test_get_stop_loss_hold_signal(self):
        """Test stop loss calculation for HOLD signal"""
        manager = FixedRiskManager(stop_loss_pct=0.05)
        signal = self.create_test_signal(SignalDirection.HOLD)
        
        stop_loss = manager.get_stop_loss(50000.0, signal)
        
        assert stop_loss == 50000.0  # No stop loss for HOLD
    
    def test_get_stop_loss_invalid_entry_price(self):
        """Test stop loss calculation with invalid entry price"""
        manager = FixedRiskManager()
        signal = self.create_test_signal(SignalDirection.BUY)
        
        with pytest.raises(ValueError, match="entry_price must be positive"):
            manager.get_stop_loss(-50000.0, signal)
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        manager = FixedRiskManager(risk_per_trade=0.03, stop_loss_pct=0.08)
        params = manager.get_parameters()
        
        assert params['name'] == "fixed_risk_manager"
        assert params['type'] == "FixedRiskManager"
        assert params['risk_per_trade'] == 0.03
        assert params['stop_loss_pct'] == 0.08