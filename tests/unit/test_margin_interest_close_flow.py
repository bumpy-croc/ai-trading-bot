"""Tests for margin interest deduction during position close flow."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide
from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    HoldSignalGenerator,
    Strategy,
    StrategyRuntime,
)
from tests.mocks import MockDatabaseManager


@pytest.fixture(autouse=True)
def mock_database_manager(monkeypatch):
    """Mock the DatabaseManager for all tests in this module."""
    original_init = MockDatabaseManager.__init__

    def patched_init(self, database_url=None):
        original_init(self, database_url)
        self._fallback_balance = 1_000.0

    monkeypatch.setattr(MockDatabaseManager, "__init__", patched_init)
    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)


def _make_engine(exchange_interface=None):
    """Create a LiveTradingEngine with zero fees for clean PnL testing."""
    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=1_000.0,
        enable_live_trading=False,
        log_trades=False,
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    if exchange_interface is not None:
        engine.exchange_interface = exchange_interface
    return engine


def _make_position(side=PositionSide.SHORT, symbol="ETHUSDT", entry_price=100.0, size=0.25):
    """Create a test position."""
    return Position(
        symbol=symbol,
        side=side,
        size=size,
        entry_price=entry_price,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        order_id="order-1",
        original_size=size,
        current_size=size,
    )


class TestMarginInterestDeduction:
    """Tests for margin interest being deducted from realized PnL on close."""

    def test_interest_deducted_from_pnl_for_short_in_margin_mode(self):
        """Short positions in margin mode should have interest deducted from PnL."""
        exchange = Mock()
        exchange.is_margin_mode = True
        exchange.get_margin_interest_history.return_value = [
            {"interest": "1.50"},
            {"interest": "0.75"},
        ]

        engine = _make_engine(exchange_interface=exchange)
        position = _make_position(side=PositionSide.SHORT, entry_price=100.0)
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        initial_balance = engine.current_balance
        exit_price = 90.0  # Profitable short: entry 100 -> exit 90

        # Mock record_trade to capture call args
        engine.performance_tracker.record_trade = Mock()

        engine._execute_exit(
            position=position,
            reason="test-close",
            limit_price=None,
            current_price=exit_price,
            candle_high=None,
            candle_low=None,
            candle=None,
            skip_live_close=True,
        )

        # Interest cost = 1.50 + 0.75 = 2.25
        # Gross PnL for short: (100 - 90) / 100 * 0.25 * 1000 = 25.0
        # Net PnL = 25.0 - 2.25 = 22.75
        expected_balance = initial_balance + 25.0 - 2.25
        assert engine.current_balance == pytest.approx(expected_balance, abs=0.01)

        # Verify performance_tracker.record_trade includes interest in fee
        assert engine.performance_tracker.record_trade.call_count == 1
        call_kwargs = engine.performance_tracker.record_trade.call_args[1]
        # With fee_rate=0.0, entry_fee=0 and exit_fee=0, so fee should equal interest_cost
        assert call_kwargs["fee"] == pytest.approx(2.25, abs=0.01)

    def test_no_interest_deduction_for_long_in_margin_mode(self):
        """Long positions should NOT have interest deducted even in margin mode."""
        exchange = Mock()
        exchange.is_margin_mode = True

        engine = _make_engine(exchange_interface=exchange)
        position = _make_position(side=PositionSide.LONG, entry_price=100.0)
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        exit_price = 110.0

        engine._execute_exit(
            position=position,
            reason="test-close",
            limit_price=None,
            current_price=exit_price,
            candle_high=None,
            candle_low=None,
            candle=None,
            skip_live_close=True,
        )

        # No interest query should be made for long positions
        exchange.get_margin_interest_history.assert_not_called()

    def test_no_interest_deduction_when_not_margin_mode(self):
        """When not in margin mode, no interest should be deducted."""
        exchange = Mock()
        exchange.is_margin_mode = False

        engine = _make_engine(exchange_interface=exchange)
        position = _make_position(side=PositionSide.SHORT, entry_price=100.0)
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        exit_price = 90.0

        engine._execute_exit(
            position=position,
            reason="test-close",
            limit_price=None,
            current_price=exit_price,
            candle_high=None,
            candle_low=None,
            candle=None,
            skip_live_close=True,
        )

        exchange.get_margin_interest_history.assert_not_called()

    def test_no_interest_deduction_when_no_exchange_interface(self):
        """When exchange_interface is None (paper trading), no interest deduction."""
        engine = _make_engine(exchange_interface=None)
        position = _make_position(side=PositionSide.SHORT, entry_price=100.0)
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        initial_balance = engine.current_balance
        exit_price = 90.0

        engine._execute_exit(
            position=position,
            reason="test-close",
            limit_price=None,
            current_price=exit_price,
            candle_high=None,
            candle_low=None,
            candle=None,
            skip_live_close=True,
        )

        # Gross PnL only, no interest deduction
        expected_balance = initial_balance + 25.0  # (100-90)/100 * 0.25 * 1000
        assert engine.current_balance == pytest.approx(expected_balance, abs=0.01)


class TestMarginInterestLogTrade:
    """Tests that margin_interest_cost is passed to log_trade."""

    def test_interest_cost_passed_to_log_trade(self):
        """Verify margin_interest_cost is passed to db_manager.log_trade()."""
        exchange = Mock()
        exchange.is_margin_mode = True
        exchange.get_margin_interest_history.return_value = [
            {"interest": "3.00"},
        ]

        engine = _make_engine(exchange_interface=exchange)
        # Set up a trading session so log_trade is called
        engine.trading_session_id = 1
        position = _make_position(side=PositionSide.SHORT, entry_price=100.0)
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        engine._execute_exit(
            position=position,
            reason="test-close",
            limit_price=None,
            current_price=90.0,
            candle_high=None,
            candle_low=None,
            candle=None,
            skip_live_close=True,
        )

        # Check that log_trade was called with margin_interest_cost
        logged_trade = engine.db_manager._trades
        assert len(logged_trade) == 1
        trade_data = list(logged_trade.values())[0]
        assert trade_data.get("margin_interest_cost") == pytest.approx(3.0)

    def test_zero_interest_cost_for_non_margin(self):
        """Verify margin_interest_cost=0.0 is passed for non-margin trades."""
        engine = _make_engine(exchange_interface=None)
        engine.trading_session_id = 1
        position = _make_position(side=PositionSide.LONG, entry_price=100.0)
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        engine._execute_exit(
            position=position,
            reason="test-close",
            limit_price=None,
            current_price=110.0,
            candle_high=None,
            candle_low=None,
            candle=None,
            skip_live_close=True,
        )

        logged_trade = engine.db_manager._trades
        assert len(logged_trade) == 1
        trade_data = list(logged_trade.values())[0]
        assert trade_data.get("margin_interest_cost") == pytest.approx(0.0)


class TestLogTradePersistence:
    """Tests that DatabaseManager.log_trade persists margin_interest_cost."""

    def test_log_trade_accepts_margin_interest_cost_param(self):
        """Verify log_trade() accepts margin_interest_cost and passes it to Trade."""
        from src.database.manager import DatabaseManager

        # Capture kwargs passed to Trade constructor
        captured_kwargs = {}

        with patch.object(DatabaseManager, "__init__", lambda self, *a, **kw: None):
            manager = DatabaseManager.__new__(DatabaseManager)
            manager._current_session_id = None

        mock_session = Mock()
        mock_trade = Mock()
        mock_trade.id = 1
        mock_trade.symbol = "ETHUSDT"
        mock_trade.side = Mock(value="SHORT")
        mock_trade.pnl = 25.0
        mock_trade.pnl_percent = 10.0

        with patch("src.database.manager.Trade") as MockTrade, \
             patch.object(DatabaseManager, "get_session_with_timeout") as mock_ctx, \
             patch.object(DatabaseManager, "_update_performance_metrics"):
            MockTrade.return_value = mock_trade
            mock_ctx.return_value.__enter__ = Mock(return_value=mock_session)
            mock_ctx.return_value.__exit__ = Mock(return_value=False)

            manager.log_trade(
                symbol="ETHUSDT",
                side="SHORT",
                entry_price=100.0,
                exit_price=90.0,
                size=0.25,
                entry_time=datetime(2025, 1, 1, tzinfo=UTC),
                exit_time=datetime(2025, 1, 2, tzinfo=UTC),
                pnl=25.0,
                exit_reason="test",
                strategy_name="test_strategy",
                margin_interest_cost=3.0,
            )

            _, call_kwargs = MockTrade.call_args
            assert call_kwargs["margin_interest_cost"] == 3.0

    def test_log_trade_defaults_margin_interest_cost_to_zero(self):
        """Verify log_trade() defaults margin_interest_cost to 0.0 when None."""
        from src.database.manager import DatabaseManager

        with patch.object(DatabaseManager, "__init__", lambda self, *a, **kw: None):
            manager = DatabaseManager.__new__(DatabaseManager)
            manager._current_session_id = None

        mock_session = Mock()
        mock_trade = Mock()
        mock_trade.id = 1
        mock_trade.symbol = "ETHUSDT"
        mock_trade.side = Mock(value="SHORT")
        mock_trade.pnl = 25.0
        mock_trade.pnl_percent = 10.0

        with patch("src.database.manager.Trade") as MockTrade, \
             patch.object(DatabaseManager, "get_session_with_timeout") as mock_ctx, \
             patch.object(DatabaseManager, "_update_performance_metrics"):
            MockTrade.return_value = mock_trade
            mock_ctx.return_value.__enter__ = Mock(return_value=mock_session)
            mock_ctx.return_value.__exit__ = Mock(return_value=False)

            manager.log_trade(
                symbol="ETHUSDT",
                side="SHORT",
                entry_price=100.0,
                exit_price=90.0,
                size=0.25,
                entry_time=datetime(2025, 1, 1, tzinfo=UTC),
                exit_time=datetime(2025, 1, 2, tzinfo=UTC),
                pnl=25.0,
                exit_reason="test",
                strategy_name="test_strategy",
                # No margin_interest_cost passed - should default to 0.0
            )

            _, call_kwargs = MockTrade.call_args
            assert call_kwargs["margin_interest_cost"] == 0.0
