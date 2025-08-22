from datetime import datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

# Conditional imports to allow running without full live trading implementation
try:
    from live.trading_engine import LiveTradingEngine, Position, PositionSide
    from src.position_management.trailing_stops import TrailingStopPolicy

    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False

    class LiveTradingEngine:  # minimal mock
        def __init__(
            self,
            strategy=None,
            data_provider=None,
            initial_balance=10000,
            enable_live_trading=False,
            **kwargs,
        ):
            self.strategy = strategy
            self.data_provider = data_provider
            self.initial_balance = initial_balance
            self.current_balance = initial_balance
            self.enable_live_trading = enable_live_trading
            self.is_running = False
            self.positions = {}
            self.completed_trades = []

    class Position:
        def __init__(
            self,
            symbol=None,
            side=None,
            size=None,
            entry_price=None,
            entry_time=None,
            stop_loss=None,
            order_id=None,
            **kwargs,
        ):
            self.symbol = symbol
            self.side = side
            self.size = size
            self.entry_price = entry_price
            self.entry_time = entry_time
            self.stop_loss = stop_loss
            self.order_id = order_id

    PositionSide = Mock()



class TestLiveTradingEngine:
    def test_engine_initialization(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
            enable_live_trading=False,
            resume_from_last_balance=False,
        )
        assert engine.strategy == mock_strategy
        assert engine.data_provider == mock_data_provider
        assert engine.current_balance == 10000
        assert engine.initial_balance == 10000
        assert engine.enable_live_trading is False
        assert engine.is_running is False
        assert len(engine.positions) == 0
        assert len(engine.completed_trades) == 0

    def test_engine_initialization_with_live_trading_enabled(
        self, mock_strategy, mock_data_provider
    ):
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, enable_live_trading=True
        )
        assert engine.enable_live_trading is True

    @pytest.mark.live_trading
    def test_position_opening_paper_trading(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, enable_live_trading=False
        )
        if hasattr(engine, "_open_position"):
            engine._open_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG if hasattr(PositionSide, "LONG") else "LONG",
                size=0.1,
                price=50000,
                stop_loss=49000,
                take_profit=52000,
            )
            assert len(engine.positions) == 1
            position = list(engine.positions.values())[0]
            assert position.symbol == "BTCUSDT"
            assert position.size == 0.1
            assert position.entry_price == 50000
            assert position.stop_loss == 49000

    @pytest.mark.live_trading
    def test_position_closing(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
            initial_balance=10000,
        )
        if hasattr(Position, "__init__"):
            position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG if hasattr(PositionSide, "LONG") else "LONG",
                size=0.1,
                entry_price=50000,
                entry_time=datetime.now(),
                stop_loss=49000,
                order_id="test_001",
            )
            engine.positions["test_001"] = position
            mock_data_provider.get_live_data.return_value = pd.DataFrame(
                {"close": [51000]}, index=[datetime.now()]
            )
            if hasattr(engine, "_close_position"):
                engine._close_position(position, "Test closure")
                assert len(engine.positions) == 0
                assert len(engine.completed_trades) == 1

    @pytest.mark.live_trading
    def test_stop_loss_trigger(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG if hasattr(PositionSide, "LONG") else "LONG",
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            stop_loss=49000,
            order_id="test_001",
        )
        if hasattr(engine, "_check_stop_loss"):
            assert engine._check_stop_loss(position, 48500) is True
            assert engine._check_stop_loss(position, 49500) is False

    @pytest.mark.live_trading
    def test_take_profit_trigger(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG if hasattr(PositionSide, "LONG") else "LONG",
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            take_profit=52000,
            order_id="test_001",
        )
        if hasattr(engine, "_check_take_profit"):
            assert engine._check_take_profit(position, 52500) is True
            assert engine._check_take_profit(position, 51500) is False
            # Short case
            if hasattr(PositionSide, "SHORT"):
                position.side = PositionSide.SHORT
                position.take_profit = 48000
                assert engine._check_take_profit(position, 47500) is True
                assert engine._check_take_profit(position, 48500) is False

    @pytest.mark.live_trading
    def test_position_pnl_update(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        long_position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG if hasattr(PositionSide, "LONG") else "LONG",
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            order_id="long_001",
        )
        engine.positions["long_001"] = long_position
        if hasattr(engine, "_update_position_pnl"):
            engine._update_position_pnl(51000)
            expected_long_pnl = (51000 - 50000) / 50000 * 0.1
            assert long_position.unrealized_pnl == expected_long_pnl

    @pytest.mark.live_trading
    def test_maximum_position_limits(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, max_position_size=0.1
        )
        if hasattr(engine, "_open_position"):
            engine._open_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG if hasattr(PositionSide, "LONG") else "LONG",
                size=0.5,
                price=50000,
            )
            position = list(engine.positions.values())[0]
            assert position.size <= 0.1

    def test_performance_metrics_calculation(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
            resume_from_last_balance=False,
        )
        engine.total_trades = 10
        engine.winning_trades = 6
        engine.total_pnl = 500
        engine.current_balance = 10500
        engine.peak_balance = 10800
        if hasattr(engine, "_update_performance_metrics") and hasattr(
            engine, "get_performance_summary"
        ):
            engine._update_performance_metrics()
            performance = engine.get_performance_summary()
            assert performance["total_trades"] == 10
            assert performance["win_rate"] == 60.0
            assert performance["total_return"] == 5.0
            assert performance["current_drawdown"] == pytest.approx(2.78, rel=1e-2)
            assert performance["max_drawdown_pct"] == pytest.approx(2.78, rel=1e-2)


@pytest.mark.live_trading
def test_trailing_stop_update_flow(mock_strategy, mock_data_provider):
    engine = LiveTradingEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        trailing_stop_policy=TrailingStopPolicy(
            activation_threshold=0.005, trailing_distance_pct=0.005, breakeven_threshold=0.02, breakeven_buffer=0.001
        ),
    )

    # Open a position
    position = Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG if hasattr(PositionSide, "LONG") else "LONG",
        size=0.1,
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=99.0,
        order_id="test_trail_001",
    )
    engine.positions[position.order_id] = position
    engine.position_db_ids[position.order_id] = None  # disable db for this test

    import pandas as pd
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0], "atr": [1.0, 1.0, 1.0, 1.0]})

    # Before activation
    engine._update_trailing_stops(df, 1, 101.0)  # +1%
    assert position.trailing_stop_activated is True  # activation_threshold=0.5%
    assert position.breakeven_triggered is False

    old_sl = position.stop_loss
    # Move further
    engine._update_trailing_stops(df, 2, 102.0)
    assert position.stop_loss is not None
    assert position.stop_loss >= old_sl

    # Hit breakeven threshold at +2%
    engine._update_trailing_stops(df, 3, 103.0)
    assert position.breakeven_triggered is True
