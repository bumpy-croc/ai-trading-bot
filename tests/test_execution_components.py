"""
Test suite for unified components: TradeExecutor, SignalGenerator, and TradingDataRepository

Following pytest best practices with AAA pattern, fixtures, and parameterization.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import pandas as pd
import subprocess
import sys
from unittest.mock import patch
import time
import asyncio

from execution.trade_executor import TradeExecutor, TradeRequest, CloseRequest, ExecutionMode, OrderResult
from execution.signal_generator import SignalGenerator, Signal, MarketContext
from data.repository import TradingDataRepository
from performance.metrics import Side
from strategies.base import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing SignalGenerator"""
    
    def __init__(self):
        super().__init__("test_strategy")
    
    def calculate_indicators(self, df):
        """Calculate mock indicators"""
        df['rsi'] = 50.0  # Mock RSI
        df['trend_strength'] = 0.02  # Mock trend
        return df
    
    def check_entry_conditions(self, df, index):
        return True  # Always signal entry for testing
    
    def check_exit_conditions(self, df, index, entry_price):
        return False  # Never signal exit for testing
    
    def calculate_position_size(self, df, index, balance):
        return balance * 0.1  # 10% position size
    
    def calculate_stop_loss(self, df, index, current_price, side):
        return current_price * 0.95  # 5% stop loss
    
    def get_parameters(self):
        """Return mock strategy parameters"""
        return {
            "name": "test_strategy",
            "position_size": 0.1,
            "stop_loss_pct": 0.05
        }


class MockOrderExecutor:
    """Mock order executor for testing TradeExecutor"""
    
    def __init__(self):
        self.orders = []
        self.current_prices = {"BTCUSDT": 50000.0}
    
    def execute_buy_order(self, symbol, quantity, price):
        order_id = f"buy_{len(self.orders)}"
        self.orders.append({"id": order_id, "type": "buy", "symbol": symbol, "quantity": quantity, "price": price})
        return OrderResult(success=True, order_id=order_id, executed_price=price, executed_quantity=quantity)
    
    def execute_sell_order(self, symbol, quantity, price):
        order_id = f"sell_{len(self.orders)}"
        self.orders.append({"id": order_id, "type": "sell", "symbol": symbol, "quantity": quantity, "price": price})
        return OrderResult(success=True, order_id=order_id, executed_price=price, executed_quantity=quantity)
    
    def get_current_price(self, symbol):
        return self.current_prices.get(symbol, 50000.0)


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    db = Mock()
    db.log_position.return_value = 1
    db.log_trade.return_value = 1
    db.update_balance.return_value = None
    db.execute_query.return_value = []
    return db


@pytest.fixture
def mock_data_provider():
    """Mock data provider"""
    provider = Mock()
    
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 1000.0
    })
    
    provider.get_historical_data.return_value = df
    provider.get_current_price.return_value = 50500.0
    return provider


@pytest.fixture
def trade_executor(mock_db_manager):
    """TradeExecutor fixture"""
    order_executor = MockOrderExecutor()
    return TradeExecutor(
        mode=ExecutionMode.BACKTEST,
        order_executor=order_executor,
        db_manager=mock_db_manager,
        session_id=1,
        initial_balance=10000.0
    )


@pytest.fixture
def signal_generator():
    """SignalGenerator fixture"""
    strategy = MockStrategy()
    return SignalGenerator(strategy)


@pytest.fixture
def trading_repository(mock_db_manager, mock_data_provider):
    """TradingDataRepository fixture"""
    return TradingDataRepository(mock_db_manager, mock_data_provider)


class TestTradeExecutor:
    """Test cases for TradeExecutor"""
    
    def test_open_position_success(self, trade_executor):
        """Test successful position opening"""
        # Arrange
        request = TradeRequest(
            symbol="BTCUSDT",
            side=Side.LONG,
            size=0.1,  # 10% of balance
            price=50000.0,
            strategy_name="test_strategy"
        )
        
        # Act
        result = trade_executor.open_position(request)
        
        # Assert
        assert result.success is True
        assert result.trade_id is not None
        assert result.executed_price == 50000.0
        assert result.executed_size == 0.1
        assert len(trade_executor.active_positions) == 1
    
    def test_open_position_invalid_size(self, trade_executor):
        """Test position opening with invalid size"""
        # Arrange
        request = TradeRequest(
            symbol="BTCUSDT",
            side=Side.LONG,
            size=1.5,  # Invalid: > 100%
            price=50000.0
        )
        
        # Act
        result = trade_executor.open_position(request)
        
        # Assert
        assert result.success is False
        assert "Invalid position size" in result.error_message
        assert len(trade_executor.active_positions) == 0
    
    def test_close_position_success(self, trade_executor):
        """Test successful position closing"""
        # Arrange - First open a position
        open_request = TradeRequest(
            symbol="BTCUSDT",
            side=Side.LONG,
            size=0.1,
            price=50000.0
        )
        open_result = trade_executor.open_position(open_request)
        
        # Update price for profit
        trade_executor.order_executor.current_prices["BTCUSDT"] = 52000.0
        
        close_request = CloseRequest(
            position_id=open_result.trade_id,
            reason="take_profit",
            price=52000.0
        )
        
        # Act
        result = trade_executor.close_position(close_request)
        
        # Assert
        assert result.success is True
        assert result.executed_price == 52000.0
        assert len(trade_executor.active_positions) == 0
        assert trade_executor.current_balance > 10000.0  # Should have profit
    
    def test_close_nonexistent_position(self, trade_executor):
        """Test closing a position that doesn't exist"""
        # Arrange
        request = CloseRequest(
            position_id="nonexistent",
            reason="test"
        )
        
        # Act
        result = trade_executor.close_position(request)
        
        # Assert
        assert result.success is False
        assert "not found" in result.error_message
    
    @pytest.mark.parametrize("mode", [ExecutionMode.BACKTEST, ExecutionMode.PAPER, ExecutionMode.LIVE])
    def test_execution_modes(self, mock_db_manager, mode):
        """Test TradeExecutor works with different execution modes"""
        # Arrange
        order_executor = MockOrderExecutor()
        executor = TradeExecutor(mode, order_executor, mock_db_manager)
        
        # Act & Assert
        assert executor.mode == mode
        assert executor.current_balance == 10000.0  # Default balance


class TestSignalGenerator:
    """Test cases for SignalGenerator"""
    
    def test_generate_entry_signal(self, signal_generator):
        """Test generation of entry signal"""
        # Arrange
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1000.0
        })
        
        context = MarketContext(
            symbol="BTCUSDT",
            current_price=50500.0,
            timestamp=datetime.now(),
            timeframe="1h",
            data=df,
            index=len(df) - 1
        )
        
        # Act
        signal = signal_generator.generate_signal(context, 10000.0)
        
        # Assert
        assert signal.action == "enter"
        assert signal.side == Side.LONG
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence > 0
        assert signal.position_size is not None
        assert signal.reasons is not None
    
    def test_generate_hold_signal_insufficient_data(self, signal_generator):
        """Test hold signal when conditions aren't met"""
        # Arrange - Empty DataFrame
        df = pd.DataFrame()
        context = MarketContext(
            symbol="BTCUSDT",
            current_price=50500.0,
            timestamp=datetime.now(),
            timeframe="1h",
            data=df,
            index=0
        )
        
        # Act
        signal = signal_generator.generate_signal(context, 10000.0)
        
        # Assert
        assert signal.action == "hold"
        assert signal.confidence == 0.0
        assert "error" in signal.metadata
    
    def test_signal_history_tracking(self, signal_generator):
        """Test that signal history is properly tracked"""
        # Arrange
        df = pd.DataFrame({
            'close': [50000.0] * 50,
            'volume': [1000.0] * 50
        })
        context = MarketContext("BTCUSDT", 50500.0, datetime.now(), "1h", df, 49)
        
        # Act - Generate multiple signals
        signal1 = signal_generator.generate_signal(context, 10000.0)
        signal2 = signal_generator.generate_signal(context, 10000.0)
        
        # Assert
        assert len(signal_generator.signal_history) == 2
        assert signal_generator.last_signal == signal2
        
        history = signal_generator.get_signal_history(limit=1)
        assert len(history) == 1
        assert history[0] == signal2


class TestTradingDataRepository:
    """Test cases for TradingDataRepository"""
    
    def test_get_market_data_with_indicators(self, trading_repository):
        """Test market data retrieval with indicators"""
        # Arrange
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        # Act
        df = trading_repository.get_market_data(
            "BTCUSDT", "1h", start_date, end_date, include_indicators=True
        )
        
        # Assert
        assert not df.empty
        assert 'close' in df.columns
        # Indicators should be added
        expected_indicators = ['rsi', 'ema_9', 'ema_21', 'ema_50', 'atr', 'trend_strength']
        for indicator in expected_indicators:
            assert indicator in df.columns
    
    def test_get_market_data_without_indicators(self, trading_repository):
        """Test market data retrieval without indicators"""
        # Arrange
        start_date = datetime(2024, 1, 1)
        
        # Act
        df = trading_repository.get_market_data(
            "BTCUSDT", "1h", start_date, include_indicators=False
        )
        
        # Assert
        assert not df.empty
        assert 'close' in df.columns
        # Indicators should not be added
        assert 'rsi' not in df.columns
    
    def test_get_current_price(self, trading_repository):
        """Test current price retrieval"""
        # Act
        price = trading_repository.get_current_price("BTCUSDT")
        
        # Assert
        assert price == 50500.0
    
    def test_get_trades_with_filters(self, trading_repository):
        """Test trade retrieval with various filters"""
        # Arrange
        trading_repository.db.execute_query.return_value = [
            {
                'id': 1,
                'symbol': 'BTCUSDT',
                'strategy_name': 'test_strategy',
                'pnl': 100.0,
                'entry_time': datetime.now()
            }
        ]
        
        # Act
        trades = trading_repository.get_trades(
            session_id=1,
            strategy_name="test_strategy",
            symbol="BTCUSDT",
            limit=10
        )
        
        # Assert
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'BTCUSDT'
        
        # Verify the query was called with correct parameters
        trading_repository.db.execute_query.assert_called_once()
    
    def test_get_trade_performance_metrics(self, trading_repository):
        """Test trade performance calculation"""
        # Arrange
        trading_repository.db.execute_query.return_value = [{
            'total_trades': 10,
            'winning_trades': 6,
            'total_pnl': 500.0,
            'avg_pnl': 50.0,
            'max_win': 200.0,
            'max_loss': -100.0,
            'pnl_std': 75.0
        }]
        
        # Act
        performance = trading_repository.get_trade_performance(session_id=1)
        
        # Assert
        assert performance['total_trades'] == 10
        assert performance['win_rate'] == 60.0
        assert performance['total_pnl'] == 500.0
        assert performance['profit_factor'] == 2.0  # 200 / 100
    
    def test_empty_trade_performance(self, trading_repository):
        """Test trade performance with no trades"""
        # Arrange
        trading_repository.db.execute_query.return_value = []
        
        # Act
        performance = trading_repository.get_trade_performance(session_id=1)
        
        # Assert
        assert performance == {}
    
    @pytest.mark.parametrize("session_id,expected_calls", [
        (None, 1),  # No session filter
        (1, 1),     # With session filter
        (999, 1)    # Different session
    ])
    def test_get_active_positions_filtering(self, trading_repository, session_id, expected_calls):
        """Test active position retrieval with different session filters"""
        # Arrange
        trading_repository.db.execute_query.return_value = []
        
        # Act
        trading_repository.get_active_positions(session_id=session_id)
        
        # Assert
        assert trading_repository.db.execute_query.call_count == expected_calls


class TestIntegration:
    """Integration tests for unified components"""

    def test_complete_trading_workflow(self, mock_db_manager, mock_data_provider):
        """Test complete workflow from signal generation to trade execution"""
        # Arrange
        strategy = MockStrategy()
        signal_generator = SignalGenerator(strategy)
        order_executor = MockOrderExecutor()
        trade_executor = TradeExecutor(
            ExecutionMode.BACKTEST,
            order_executor,
            mock_db_manager,
            initial_balance=10000.0
        )
        
        # Create market context
        df = pd.DataFrame({
            'close': [50000.0] * 50,
            'volume': [1000.0] * 50
        })
        context = MarketContext("BTCUSDT", 50500.0, datetime.now(), "1h", df, 49)
        
        # Act - Generate signal
        signal = signal_generator.generate_signal(context, trade_executor.current_balance)
        
        # Execute trade based on signal
        if signal.action == "enter":
            trade_request = TradeRequest(
                symbol=signal.symbol,
                side=signal.side,
                size=signal.position_size,
                price=signal.price,
                stop_loss=signal.stop_loss,
                strategy_name="test_strategy"
            )
            result = trade_executor.open_position(trade_request)
        
        # Assert
        assert signal.action == "enter"
        assert result.success is True
        assert len(trade_executor.active_positions) == 1
        
        # Verify database interactions
        mock_db_manager.log_position.assert_called_once()
        
    def test_repository_with_real_data_flow(self, mock_db_manager, mock_data_provider):
        """Test repository with realistic data flow"""
        # Arrange
        repository = TradingDataRepository(mock_db_manager, mock_data_provider)
        
        # Mock different return values for different queries
        def mock_execute_query(query, params=None):
            if "balance" in query and "account_history" in query:
                # Balance history query
                return [
                    {'timestamp': datetime.now() - timedelta(days=30), 'balance': 10000.0},
                    {'timestamp': datetime.now() - timedelta(days=15), 'balance': 11000.0},
                    {'timestamp': datetime.now(), 'balance': 12000.0}
                ]
            elif "trades" in query and "COUNT" in query:
                # Trade performance query
                return [{
                    'total_trades': 5,
                    'winning_trades': 3,
                    'total_pnl': 500.0,
                    'avg_pnl': 100.0,
                    'max_win': 300.0,
                    'max_loss': -100.0,
                    'pnl_std': 150.0
                }]
            else:
                return []
        
        mock_db_manager.execute_query.side_effect = mock_execute_query
        
        # Act
        metrics = repository.calculate_session_metrics(session_id=1, initial_balance=10000.0)
        
        # Assert
        assert metrics['initial_balance'] == 10000.0
        assert metrics['final_balance'] == 12000.0
        assert metrics['total_return_pct'] > 0  # Should show profit
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
        assert metrics['total_trades'] == 5
        assert metrics['win_rate'] == 60.0


def test_run_backtest_script():
    """Test that the run_backtest.py script runs without errors."""
    
    # Define the command to run the script
    command = [
        sys.executable,  # Use the same python interpreter
        "scripts/run_backtest.py",
        "adaptive",
        "--days", "1",
        "--no-db"
    ]
    
    # Run the script as a subprocess
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    # Check for errors
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
    assert "BACKTEST RESULTS" in result.stdout, "Backtest results not found in output"
    assert "error" not in result.stderr.lower(), "Error message found in stderr"


@pytest.mark.asyncio
async def test_run_live_trading_script():
    """Test that the run_live_trading.py script starts without errors."""
    
    # Mock the engine and its dependencies
    with patch('scripts.run_live_trading.LiveTradingEngine') as MockEngine:
        mock_engine_instance = MockEngine.return_value
        mock_engine_instance.start = AsyncMock()
        mock_engine_instance.stop = AsyncMock()

        # Get the current event loop
        loop = asyncio.get_event_loop()

        # Create the app with the test's event loop
        from scripts.run_live_trading import TradingApp
        app = TradingApp(mock_engine_instance, loop=loop)
        
        # Run the app for a short time
        run_task = loop.create_task(app.run())
        await asyncio.sleep(0.1)
        
        # Trigger shutdown
        app.shutdown_event.set()
        await run_task
        
        # Assert that the engine was started and stopped
        mock_engine_instance.start.assert_called_once()
        mock_engine_instance.stop.assert_called_once()


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs) 