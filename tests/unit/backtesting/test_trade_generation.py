"""Tests validating Trade lifecycle behaviour for backtesting."""

from datetime import UTC, datetime, timedelta

from src.engines.live.trading_engine import PositionSide, Trade


class TestTradeGeneration:
    """Ensure Trade objects behave as expected."""

    def test_trade_creation(self):
        """Trade instances should preserve constructor arguments."""

        trade = Trade(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            exit_price=50000,
            entry_time=datetime.now(UTC),
            exit_time=datetime.now(UTC),
            pnl=0.0,
            exit_reason="init",
        )

        assert trade.symbol == "BTCUSDT"
        assert trade.side == PositionSide.LONG
        assert trade.entry_price == 50000
        assert trade.size == 0.1

    def test_trade_pnl_calculation(self):
        """Profit and loss should match side-specific expectations."""

        trade_long_profit = Trade(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            exit_price=55000,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            size=0.1,
            pnl=500,
            exit_reason="test",
        )

        assert trade_long_profit.pnl == 500

        trade_short_profit = Trade(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=55000,
            exit_price=50000,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            size=0.1,
            pnl=500,
            exit_reason="test",
        )

        assert trade_short_profit.pnl == 500

    def test_trade_duration_calculation(self):
        """Duration should equal exit minus entry timestamps."""

        entry_time = datetime(2024, 1, 1, 10, 0)
        exit_time = datetime(2024, 1, 1, 12, 0)

        trade = Trade(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            exit_price=55000,
            entry_time=entry_time,
            exit_time=exit_time,
            size=0.1,
            pnl=500,
            exit_reason="test",
        )

        assert trade.exit_time - trade.entry_time == timedelta(hours=2)
