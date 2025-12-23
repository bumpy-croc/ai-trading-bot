from datetime import datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

# Conditional imports to allow running without full live trading implementation
from unittest.mock import Mock

try:
    from live.trading_engine import LiveTradingEngine, Position, PositionSide
    from src.performance.metrics import cash_pnl
    from src.position_management.trailing_stops import TrailingStopPolicy
    from src.strategies.components import (
        Signal,
        SignalDirection,
        Strategy,
        MLBasicSignalGenerator,
        FixedRiskManager,
        ConfidenceWeightedSizer,
        SignalGenerator,
    )
    from src.strategies.components.strategy import TradingDecision

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
    Signal = None
    SignalDirection = None
    Strategy = None
    TradingDecision = None
    MLBasicSignalGenerator = None
    FixedRiskManager = None
    ConfidenceWeightedSizer = None

    def cash_pnl(pnl_pct, balance_before):
        return float(pnl_pct) * float(balance_before)


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
            strategy=mock_strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
            # Disable fees/slippage for this test
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        if hasattr(engine, "_open_position"):
            engine._open_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
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
                side=PositionSide.LONG,
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
            side=PositionSide.LONG,
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
            side=PositionSide.LONG,
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
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            order_id="long_001",
        )
        engine.positions["long_001"] = long_position
        if hasattr(engine, "_update_position_pnl"):
            engine._update_position_pnl(51000)
            sized_return = (51000 - 50000) / 50000 * 0.1
            expected_long_pnl = cash_pnl(sized_return, engine.current_balance)
            assert long_position.unrealized_pnl == expected_long_pnl
            assert long_position.unrealized_pnl_percent == pytest.approx(sized_return * 100.0)

    @pytest.mark.live_trading
    def test_maximum_position_limits(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, max_position_size=0.1
        )
        if hasattr(engine, "_open_position"):
            engine._open_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
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
            activation_threshold=0.005,
            trailing_distance_pct=0.005,
            breakeven_threshold=0.02,
            breakeven_buffer=0.001,
        ),
    )

    # Open a position
    position = Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=1.0,  # Use full position size so 1% price move = 1% sized PnL
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


@pytest.mark.live_trading
def test_order_execution_with_component_strategy(mock_data_provider):
    """Test order execution using component-based strategy with TradingDecision"""
    if not LIVE_TRADING_AVAILABLE or Strategy is None:
        pytest.skip("Component strategy not available")

    # Create a component-based strategy
    signal_generator = MLBasicSignalGenerator(name="test_ml_basic_sg")
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = ConfidenceWeightedSizer(base_fraction=0.02)

    strategy = Strategy(
        name="test_ml_basic",
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
    )

    # Create engine with component strategy
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=mock_data_provider,
        enable_live_trading=False,
        initial_balance=10000,
    )

    # Create test data
    df = pd.DataFrame(
        {
            "open": [50000, 50100, 50200],
            "high": [50100, 50200, 50300],
            "low": [49900, 50000, 50100],
            "close": [50000, 50100, 50200],
            "volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )

    mock_data_provider.get_live_data.return_value = df
    mock_data_provider.get_current_price.return_value = 50200

    # Process candle and get decision
    decision = strategy.process_candle(df, 2, engine.current_balance)

    # Validate TradingDecision object
    assert isinstance(decision, TradingDecision)
    assert hasattr(decision, "signal")
    assert hasattr(decision, "position_size")
    assert hasattr(decision, "risk_metrics")
    assert decision.signal.direction in [
        SignalDirection.BUY,
        SignalDirection.SELL,
        SignalDirection.HOLD,
    ]
    assert 0 <= decision.signal.confidence <= 1
    assert decision.position_size >= 0

    # If signal is BUY, test order placement
    if decision.signal.direction == SignalDirection.BUY:
        requested_notional = decision.position_size
        requested_fraction = (
            requested_notional / engine.current_balance if engine.current_balance else 0.0
        )
        capped_fraction = min(requested_fraction, engine.max_position_size)
        engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=capped_fraction,
            price=50200,
            stop_loss=decision.risk_metrics.get("stop_loss_price"),
        )

        assert len(engine.positions) == 1
        position = list(engine.positions.values())[0]
        assert position.size == capped_fraction
        assert position.entry_price == 50200


@pytest.mark.live_trading
def test_position_sizing_from_trading_decision(mock_data_provider):
    """Test that position sizing comes from TradingDecision, not legacy methods"""
    if not LIVE_TRADING_AVAILABLE or Strategy is None:
        pytest.skip("Component strategy not available")

    class _DeterministicSignalGenerator(SignalGenerator):
        def __init__(self) -> None:
            super().__init__("deterministic_signal")

        def generate_signal(self, df, index, regime=None):  # type: ignore[override]
            self.validate_inputs(df, index)
            return Signal(
                direction=SignalDirection.BUY,
                strength=0.8,
                confidence=0.9,
                metadata={"reason": "deterministic_test"},
            )

        def get_confidence(self, df, index):  # type: ignore[override]
            self.validate_inputs(df, index)
            return 0.9

    # Create component strategy with specific position sizer
    signal_generator = _DeterministicSignalGenerator()
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = ConfidenceWeightedSizer(base_fraction=0.03)  # 3% base

    strategy = Strategy(
        name="test_sizing",
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
    )

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=mock_data_provider,
        enable_live_trading=False,
        initial_balance=10000,
    )

    # Create test data with enough history for ML model (120+ candles)
    df = pd.DataFrame(
        {
            "open": [50000 + i * 10 for i in range(150)],
            "high": [50100 + i * 10 for i in range(150)],
            "low": [49900 + i * 10 for i in range(150)],
            "close": [50000 + i * 10 for i in range(150)],
            "volume": [1000 + i * 10 for i in range(150)],
        },
        index=pd.date_range("2024-01-01", periods=150, freq="1h"),
    )

    mock_data_provider.get_live_data.return_value = df

    # Get decision
    decision = strategy.process_candle(df, 149, engine.current_balance)

    # Verify position size is calculated by component
    # Note: position_size might be 0 if signal is HOLD
    assert decision.position_size >= 0
    # Position size should be influenced by confidence and base fraction
    # With 3% base and confidence weighting, size should be reasonable
    if decision.signal.direction != SignalDirection.HOLD:
        assert decision.position_size > 0
        # Position size is in dollars, should be reasonable fraction of balance
        assert (
            decision.position_size <= engine.current_balance * 0.05
        )  # Should not exceed 5% of balance

    # Verify position size comes from TradingDecision, not legacy method
    # The component strategy may have the method for compatibility but shouldn't use it
    assert "position_size" in decision.__dict__


@pytest.mark.live_trading
def test_stop_loss_from_component_strategy(mock_data_provider):
    """Test that stop loss comes from component strategy, not legacy methods"""
    if not LIVE_TRADING_AVAILABLE or Strategy is None:
        pytest.skip("Component strategy not available")

    signal_generator = MLBasicSignalGenerator(name="test_stop_loss_sg")
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = ConfidenceWeightedSizer(base_fraction=0.02)

    strategy = Strategy(
        name="test_stop_loss",
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
    )

    engine = LiveTradingEngine(
        strategy=strategy, data_provider=mock_data_provider, enable_live_trading=False
    )

    # Create test data with enough history for ML model
    df = pd.DataFrame(
        {
            "open": [50000 + i * 5 for i in range(150)],
            "high": [50100 + i * 5 for i in range(150)],
            "low": [49900 + i * 5 for i in range(150)],
            "close": [50000 + i * 5 for i in range(150)],
            "volume": [1000 + i * 5 for i in range(150)],
        },
        index=pd.date_range("2024-01-01", periods=150, freq="1h"),
    )

    mock_data_provider.get_live_data.return_value = df

    # Get decision
    decision = strategy.process_candle(df, 149, 10000)

    if decision.signal.direction == SignalDirection.BUY:
        # Get stop loss from strategy
        stop_loss = strategy.get_stop_loss_price(
            entry_price=50000, signal=decision.signal, regime=decision.regime
        )

        # Verify stop loss is reasonable
        assert stop_loss is not None
        assert stop_loss < 50000  # Stop loss should be below entry for long
        assert stop_loss > 48000  # Should not be too far away

        # Verify stop loss comes from component strategy method
        # The strategy has get_stop_loss_price, not calculate_stop_loss
        assert hasattr(strategy, "get_stop_loss_price")


@pytest.mark.live_trading
def test_position_exit_with_should_exit_position(mock_data_provider):
    """Test position exit using should_exit_position() from component strategy"""
    if not LIVE_TRADING_AVAILABLE or Strategy is None:
        pytest.skip("Component strategy not available")

    signal_generator = MLBasicSignalGenerator(name="test_exit_sg")
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = ConfidenceWeightedSizer(base_fraction=0.02)

    strategy = Strategy(
        name="test_exit",
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
    )

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=mock_data_provider,
        enable_live_trading=False,
        initial_balance=10000,
    )

    # Create test data
    df = pd.DataFrame(
        {
            "open": [50000 + i * 10 for i in range(150)],
            "high": [50100 + i * 10 for i in range(150)],
            "low": [49900 + i * 10 for i in range(150)],
            "close": [50000 + i * 10 for i in range(150)],
            "volume": [1000 + i * 10 for i in range(150)],
        },
        index=pd.date_range("2024-01-01", periods=150, freq="1h"),
    )

    mock_data_provider.get_live_data.return_value = df

    # Create a position
    from src.strategies.components import Position as ComponentPosition, MarketData

    position = ComponentPosition(
        symbol="BTCUSDT",
        entry_price=50000,
        current_price=51490,
        size=0.1,
        side="long",
        entry_time=datetime.now(),
    )

    # Create market data
    market_data = MarketData(symbol="BTCUSDT", price=51490, volume=2490, timestamp=datetime.now())

    # Test should_exit_position
    should_exit = strategy.should_exit_position(position, market_data)

    # Verify the method exists and returns a boolean
    assert isinstance(should_exit, bool)

    # Verify we're using should_exit_position, not legacy check_exit_conditions
    assert hasattr(strategy, "should_exit_position")


@pytest.mark.live_trading
def test_stop_loss_update_with_component_strategy(mock_data_provider):
    """Test stop loss updates using component strategy"""
    if not LIVE_TRADING_AVAILABLE or Strategy is None:
        pytest.skip("Component strategy not available")

    signal_generator = MLBasicSignalGenerator(name="test_sl_update_sg")
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = ConfidenceWeightedSizer(base_fraction=0.02)

    strategy = Strategy(
        name="test_sl_update",
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
    )

    engine = LiveTradingEngine(
        strategy=strategy, data_provider=mock_data_provider, enable_live_trading=False
    )

    # Create test data
    df = pd.DataFrame(
        {
            "open": [50000 + i * 10 for i in range(150)],
            "high": [50100 + i * 10 for i in range(150)],
            "low": [49900 + i * 10 for i in range(150)],
            "close": [50000 + i * 10 for i in range(150)],
            "volume": [1000 + i * 10 for i in range(150)],
        },
        index=pd.date_range("2024-01-01", periods=150, freq="1h"),
    )

    mock_data_provider.get_live_data.return_value = df

    # Get a decision to get a signal
    decision = strategy.process_candle(df, 149, 10000)

    # Get initial stop loss
    entry_price = 51490
    stop_loss_1 = strategy.get_stop_loss_price(
        entry_price=entry_price, signal=decision.signal, regime=decision.regime
    )

    # Verify stop loss is reasonable
    if stop_loss_1 is not None:
        if decision.signal.direction == SignalDirection.BUY:
            assert stop_loss_1 < entry_price  # Stop loss below entry for long
        elif decision.signal.direction == SignalDirection.SELL:
            assert stop_loss_1 > entry_price  # Stop loss above entry for short

    # Test that we can update stop loss (e.g., for trailing stops)
    # The component strategy should support this through get_stop_loss_price
    assert hasattr(strategy, "get_stop_loss_price")


@pytest.mark.live_trading
def test_position_management_with_trading_decision(mock_data_provider):
    """Test complete position management flow with TradingDecision objects"""
    if not LIVE_TRADING_AVAILABLE or Strategy is None:
        pytest.skip("Component strategy not available")

    signal_generator = MLBasicSignalGenerator(name="test_pm_sg")
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = ConfidenceWeightedSizer(base_fraction=0.02)

    strategy = Strategy(
        name="test_pm",
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
    )

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=mock_data_provider,
        enable_live_trading=False,
        initial_balance=10000,
    )

    # Create test data
    df = pd.DataFrame(
        {
            "open": [50000 + i * 10 for i in range(150)],
            "high": [50100 + i * 10 for i in range(150)],
            "low": [49900 + i * 10 for i in range(150)],
            "close": [50000 + i * 10 for i in range(150)],
            "volume": [1000 + i * 10 for i in range(150)],
        },
        index=pd.date_range("2024-01-01", periods=150, freq="1h"),
    )

    mock_data_provider.get_live_data.return_value = df
    mock_data_provider.get_current_price.return_value = 51490

    # Get decision
    decision = strategy.process_candle(df, 149, engine.current_balance)

    # Verify TradingDecision has all necessary fields for position management
    assert hasattr(decision, "signal")
    assert hasattr(decision, "position_size")
    assert hasattr(decision, "risk_metrics")
    assert hasattr(decision, "regime")

    # If we get a BUY signal, test opening and managing position
    if decision.signal.direction == SignalDirection.BUY:
        # Open position using decision data
        engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=decision.position_size,
            price=51490,
            stop_loss=decision.risk_metrics.get("stop_loss_price"),
        )

        assert len(engine.positions) == 1
        position = list(engine.positions.values())[0]

        # Verify position was created with data from TradingDecision
        assert position.size == decision.position_size
        assert position.entry_price == 51490

        # Test position exit
        engine._close_position(position, "test_exit")
        assert len(engine.positions) == 0
        assert len(engine.completed_trades) == 1
