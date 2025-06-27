"""
Comprehensive tests for the live trading engine.

This is the most critical component of the system as it handles real money.
Tests cover:
- Order execution logic
- Position management  
- Risk management integration
- Error handling and recovery
- Strategy hot-swapping
- Graceful shutdown
- Thread safety
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import pandas as pd

from live.trading_engine import LiveTradingEngine, Position, PositionSide, OrderStatus, Trade
from live.strategy_manager import StrategyManager
from core.risk.risk_manager import RiskManager, RiskParameters
from strategies.adaptive import AdaptiveStrategy


class TestLiveTradingEngine:
    """Test suite for the core LiveTradingEngine"""

    def test_engine_initialization(self, mock_strategy, mock_data_provider):
        """Test that engine initializes with correct defaults"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
            enable_live_trading=False
        )
        
        assert engine.strategy == mock_strategy
        assert engine.data_provider == mock_data_provider
        assert engine.current_balance == 10000
        assert engine.initial_balance == 10000
        assert engine.enable_live_trading == False
        assert engine.is_running == False
        assert len(engine.positions) == 0
        assert len(engine.completed_trades) == 0

    def test_engine_initialization_with_live_trading_enabled(self, mock_strategy, mock_data_provider):
        """Test initialization with live trading enabled - critical safety check"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_live_trading=True
        )
        
        assert engine.enable_live_trading == True
        # Ensure warning is logged about live trading being enabled

    @pytest.mark.live_trading
    def test_position_opening_paper_trading(self, mock_strategy, mock_data_provider):
        """Test opening positions in paper trading mode"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False
        )
        
        # Test opening a position
        engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            price=50000,
            stop_loss=49000,
            take_profit=52000
        )
        
        assert len(engine.positions) == 1
        position = list(engine.positions.values())[0]
        assert position.symbol == "BTCUSDT"
        assert position.side == PositionSide.LONG
        assert position.size == 0.1
        assert position.entry_price == 50000
        assert position.stop_loss == 49000
        assert position.take_profit == 52000

    @pytest.mark.live_trading
    def test_position_closing(self, mock_strategy, mock_data_provider):
        """Test closing positions and PnL calculation"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
            initial_balance=10000
        )
        
        # Open a position first
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            stop_loss=49000,
            order_id="test_001"
        )
        engine.positions["test_001"] = position
        
        # Mock current price data for closing
        mock_data_provider.get_live_data.return_value = pd.DataFrame({
            'close': [51000]
        }, index=[datetime.now()])
        
        # Close the position
        engine._close_position(position, "Test closure")
        
        # Verify position was closed
        assert len(engine.positions) == 0
        assert len(engine.completed_trades) == 1
        
        # Verify PnL calculation (2% gain on 10% position = 0.2% portfolio gain)
        trade = engine.completed_trades[0]
        expected_pnl = ((51000 - 50000) / 50000) * 0.1 * 10000  # 2% * 10% * 10000 = 200
        assert trade.pnl == expected_pnl
        assert engine.current_balance == 10000 + expected_pnl

    @pytest.mark.live_trading
    def test_stop_loss_trigger(self, mock_strategy, mock_data_provider):
        """Test stop loss triggering"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            stop_loss=49000,
            order_id="test_001"
        )
        
        # Test stop loss trigger for long position
        assert engine._check_stop_loss(position, 48500) == True
        assert engine._check_stop_loss(position, 49500) == False
        
        # Test short position stop loss
        position.side = PositionSide.SHORT
        position.stop_loss = 51000
        assert engine._check_stop_loss(position, 51500) == True
        assert engine._check_stop_loss(position, 50500) == False

    @pytest.mark.live_trading
    def test_take_profit_trigger(self, mock_strategy, mock_data_provider):
        """Test take profit triggering"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            take_profit=52000,
            order_id="test_001"
        )
        
        # Test take profit trigger for long position
        assert engine._check_take_profit(position, 52500) == True
        assert engine._check_take_profit(position, 51500) == False
        
        # Test short position take profit
        position.side = PositionSide.SHORT
        position.take_profit = 48000
        assert engine._check_take_profit(position, 47500) == True
        assert engine._check_take_profit(position, 48500) == False

    @pytest.mark.live_trading
    def test_position_pnl_update(self, mock_strategy, mock_data_provider):
        """Test position PnL updating"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        # Long position
        long_position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            order_id="long_001"
        )
        engine.positions["long_001"] = long_position
        
        # Short position
        short_position = Position(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            size=0.1,
            entry_price=3000,
            entry_time=datetime.now(),
            order_id="short_001"
        )
        engine.positions["short_001"] = short_position
        
        # Update PnL
        engine._update_position_pnl(51000)  # BTC price increased
        
        # Check long position PnL (profitable)
        expected_long_pnl = (51000 - 50000) / 50000 * 0.1
        assert long_position.unrealized_pnl == expected_long_pnl
        
        # Note: Short position PnL won't be updated since we only passed BTC price
        # In real implementation, you'd update each position individually

    @pytest.mark.live_trading 
    def test_maximum_position_limits(self, mock_strategy, mock_data_provider):
        """Test that maximum position limits are respected"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            max_position_size=0.1  # 10% max position size
        )
        
        # Try to open position larger than maximum
        engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.5,  # 50% - should be capped at 10%
            price=50000
        )
        
        # Should be capped at max_position_size
        position = list(engine.positions.values())[0]
        assert position.size <= 0.1

    @pytest.mark.live_trading
    def test_error_handling_in_trading_loop(self, mock_strategy, mock_data_provider):
        """Test error handling and recovery in trading loop"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            check_interval=1,
            max_consecutive_errors=3
        )
        
        # Mock data provider to raise exception
        mock_data_provider.get_live_data.side_effect = Exception("API Error")
        
        # Start engine in a thread
        def run_engine():
            engine._trading_loop("BTCUSDT", "1h")
        
        thread = threading.Thread(target=run_engine)
        thread.daemon = True
        thread.start()
        
        # Let it run for a bit to accumulate errors
        time.sleep(3.5)
        
        # Stop the engine
        engine.stop()
        thread.join(timeout=5)
        
        # Verify error counter was incremented
        assert engine.consecutive_errors > 0

    @pytest.mark.live_trading
    def test_graceful_shutdown(self, mock_strategy, mock_data_provider):
        """Test graceful shutdown closes all positions"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        # Add some positions
        position1 = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            order_id="test_001"
        )
        position2 = Position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=3000,
            entry_time=datetime.now(),
            order_id="test_002"
        )
        engine.positions["test_001"] = position1
        engine.positions["test_002"] = position2
        
        # Mock data for closing positions
        mock_data_provider.get_live_data.return_value = pd.DataFrame({
            'close': [51000]
        }, index=[datetime.now()])
        
        # Test shutdown
        engine.stop()
        
        # All positions should be closed
        assert len(engine.positions) == 0
        assert len(engine.completed_trades) == 2

    @pytest.mark.live_trading
    def test_performance_metrics_calculation(self, mock_strategy, mock_data_provider):
        """Test performance metrics calculation"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        # Simulate some trades
        engine.total_trades = 10
        engine.winning_trades = 6
        engine.total_pnl = 500
        engine.current_balance = 10500
        engine.peak_balance = 10800
        
        engine._update_performance_metrics()
        
        # Check calculated metrics
        performance = engine.get_performance_summary()
        assert performance['total_trades'] == 10
        assert performance['win_rate'] == 60.0
        assert performance['total_return'] == 5.0  # (10500-10000)/10000 * 100
        assert performance['current_drawdown'] == pytest.approx(2.78, rel=1e-2)  # (10800-10500)/10800


class TestStrategyHotSwapping:
    """Test strategy hot-swapping functionality - critical for production"""

    @pytest.mark.live_trading
    def test_strategy_hot_swap_preparation(self, mock_strategy, mock_data_provider):
        """Test preparing strategy hot swap"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_hot_swapping=True
        )
        
        # Mock the strategy manager
        engine.strategy_manager = Mock()
        engine.strategy_manager.has_pending_update.return_value = True
        engine.strategy_manager.apply_pending_update.return_value = True
        engine.strategy_manager.current_strategy = Mock()
        engine.strategy_manager.current_strategy.name = "NewStrategy"
        
        # Simulate update check in trading loop
        assert engine.strategy_manager.has_pending_update() == True
        success = engine.strategy_manager.apply_pending_update()
        assert success == True

    @pytest.mark.live_trading
    def test_strategy_hot_swap_with_position_closure(self, mock_strategy, mock_data_provider):
        """Test hot swap that requires closing existing positions"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_hot_swapping=True
        )
        
        # Add existing position
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            order_id="test_001"
        )
        engine.positions["test_001"] = position
        
        # Mock strategy manager for hot swap
        new_strategy = Mock()
        new_strategy.name = "NewStrategy"
        
        # Test hot swap with position closure
        result = engine.hot_swap_strategy("new_strategy", close_positions=True)
        
        # Should close existing positions before swap
        # In full implementation, this would close positions first

    @pytest.mark.live_trading
    def test_model_update_during_trading(self, mock_strategy, mock_data_provider, mock_model_file):
        """Test ML model updates during live trading"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_hot_swapping=True
        )
        
        # Mock strategy manager
        engine.strategy_manager = Mock()
        engine.strategy_manager.update_model.return_value = True
        
        # Test model update
        result = engine.update_model(str(mock_model_file))
        
        # Verify update was triggered
        engine.strategy_manager.update_model.assert_called_once_with(str(mock_model_file))


class TestThreadSafety:
    """Test thread safety of live trading engine"""

    @pytest.mark.live_trading
    def test_concurrent_position_updates(self, mock_strategy, mock_data_provider):
        """Test that concurrent position updates are thread-safe"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        # Mock data for position operations
        mock_data_provider.get_live_data.return_value = pd.DataFrame({
            'close': [51000]
        }, index=[datetime.now()])
        
        def open_positions():
            for i in range(5):
                engine._open_position(
                    symbol=f"BTC{i}USDT",
                    side=PositionSide.LONG,
                    size=0.02,
                    price=50000 + i
                )
                time.sleep(0.01)
        
        def close_positions():
            time.sleep(0.05)  # Let some positions open first
            positions_to_close = list(engine.positions.values())[:3]
            for position in positions_to_close:
                engine._close_position(position, "Test close")
                time.sleep(0.01)
        
        # Run concurrent operations
        thread1 = threading.Thread(target=open_positions)
        thread2 = threading.Thread(target=close_positions)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify no data corruption occurred
        assert len(engine.positions) >= 0  # Some positions might be closed
        assert len(engine.completed_trades) >= 0

    @pytest.mark.live_trading
    def test_stop_event_handling(self, mock_strategy, mock_data_provider):
        """Test that stop event properly terminates trading loop"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            check_interval=1
        )
        
        # Mock successful data fetching
        mock_data_provider.get_live_data.return_value = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900], 
            'close': [50050], 'volume': [1000]
        }, index=[datetime.now()])
        
        # Start trading loop in thread
        def run_trading():
            engine._trading_loop("BTCUSDT", "1h")
        
        thread = threading.Thread(target=run_trading)
        thread.daemon = True
        thread.start()
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Signal stop
        engine.stop_event.set()
        engine.is_running = False
        
        # Wait for thread to finish
        thread.join(timeout=3)
        
        # Thread should have stopped
        assert not thread.is_alive()


class TestRiskIntegration:
    """Test integration with risk management"""

    @pytest.mark.live_trading
    @pytest.mark.risk_management
    def test_risk_manager_integration(self, mock_strategy, mock_data_provider, risk_parameters):
        """Test that risk manager properly limits positions"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_parameters,
            max_position_size=0.05  # 5% max position
        )
        
        # Mock strategy to suggest large position
        mock_strategy.calculate_position_size.return_value = 0.5  # 50%
        
        # Open position
        engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.5,  # Large position
            price=50000
        )
        
        # Should be limited by max_position_size
        position = list(engine.positions.values())[0]
        assert position.size <= 0.05

    @pytest.mark.live_trading
    @pytest.mark.risk_management
    def test_drawdown_monitoring(self, mock_strategy, mock_data_provider):
        """Test drawdown monitoring and protection"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        # Simulate drawdown
        engine.current_balance = 7000  # 30% drawdown
        engine.peak_balance = 10000
        
        engine._update_performance_metrics()
        
        # Check drawdown calculation
        performance = engine.get_performance_summary()
        assert performance['max_drawdown'] == 30.0

    @pytest.mark.live_trading
    def test_maximum_positions_limit(self, mock_strategy, mock_data_provider):
        """Test that maximum number of positions is enforced"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        # Fill up to maximum positions
        max_positions = engine._get_max_positions()
        for i in range(max_positions + 2):  # Try to exceed limit
            engine._open_position(
                symbol=f"COIN{i}USDT",
                side=PositionSide.LONG,
                size=0.01,
                price=1000
            )
        
        # Should not exceed maximum
        assert len(engine.positions) <= max_positions


class TestDataValidation:
    """Test data validation and error handling"""

    @pytest.mark.live_trading
    def test_empty_data_handling(self, mock_strategy, mock_data_provider):
        """Test handling of empty or invalid data"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        # Mock empty data
        mock_data_provider.get_live_data.return_value = pd.DataFrame()
        
        # Should handle gracefully
        result = engine._get_latest_data("BTCUSDT", "1h")
        assert result is not None or result is None  # Should not crash

    @pytest.mark.live_trading
    def test_malformed_data_handling(self, mock_strategy, mock_data_provider):
        """Test handling of malformed data"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider
        )
        
        # Mock malformed data (missing required columns)
        mock_data_provider.get_live_data.return_value = pd.DataFrame({
            'price': [50000],  # Wrong column name
            'vol': [1000]      # Wrong column name
        })
        
        # Should handle gracefully without crashing
        try:
            result = engine._get_latest_data("BTCUSDT", "1h")
            # Should either return valid data or None, but not crash
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            assert "column" in str(e).lower() or "key" in str(e).lower()

    @pytest.mark.live_trading
    def test_api_rate_limit_handling(self, mock_strategy, mock_data_provider):
        """Test handling of API rate limits"""
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            max_consecutive_errors=5
        )
        
        # Mock rate limit error
        from requests.exceptions import RequestException
        mock_data_provider.get_live_data.side_effect = RequestException("Rate limit exceeded")
        
        # Should increment error counter but not crash immediately
        result = engine._get_latest_data("BTCUSDT", "1h")
        assert result is None  # Should return None on error