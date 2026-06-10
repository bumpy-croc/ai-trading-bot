"""Regression tests for #758: log_trade misclassified foreign-enum sides.

The engines pass their own ``PositionSide`` enum (``engines/shared``), while
``log_trade`` compared against the **database** ``PositionSide`` — cross-enum
equality is always ``False``, so every long backtest trade was persisted with
the short ``pnl_percent`` formula. ``log_trade`` now normalizes any Enum input
by value before classification.
"""

from datetime import UTC, datetime

import pytest

from src.database.manager import DatabaseManager
from src.database.models import PositionSide as DbPositionSide
from src.engines.shared.models import PositionSide as SharedPositionSide

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


def _log_trade(db: DatabaseManager, side, entry_price: float, exit_price: float) -> int:
    return db.log_trade(
        symbol="BTCUSDT",
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        size=0.1,
        entry_time=datetime(2024, 1, 1, 10, tzinfo=UTC),
        exit_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
        pnl=10.0,
        exit_reason="test",
        strategy_name="test",
        source="BACKTEST",
    )


def _fetch_trade(db: DatabaseManager, trade_id: int):
    from src.database.models import Trade

    with db.get_session() as session:
        trade = session.query(Trade).filter(Trade.id == trade_id).one()
        session.expunge(trade)
        return trade


class TestLogTradeSideNormalization:
    def test_shared_enum_long_uses_long_formula(self, db):
        """Regression: a profitable long logged with the SHARED enum must
        persist a positive pnl_percent (the old cross-enum comparison fell
        through to the short formula and stored it negative)."""
        trade_id = _log_trade(db, SharedPositionSide.LONG, entry_price=100.0, exit_price=110.0)

        trade = _fetch_trade(db, trade_id)
        assert trade.side == DbPositionSide.LONG
        assert float(trade.pnl_percent) == pytest.approx(10.0)

    def test_shared_enum_short_uses_short_formula(self, db):
        trade_id = _log_trade(db, SharedPositionSide.SHORT, entry_price=100.0, exit_price=90.0)

        trade = _fetch_trade(db, trade_id)
        assert trade.side == DbPositionSide.SHORT
        assert float(trade.pnl_percent) == pytest.approx(10.0)

    def test_string_sides_still_work(self, db):
        """Existing string-side callers are unchanged (both cases)."""
        long_id = _log_trade(db, "long", entry_price=100.0, exit_price=105.0)
        short_id = _log_trade(db, "SHORT", entry_price=100.0, exit_price=95.0)

        assert float(_fetch_trade(db, long_id).pnl_percent) == pytest.approx(5.0)
        assert float(_fetch_trade(db, short_id).pnl_percent) == pytest.approx(5.0)

    def test_database_enum_still_works(self, db):
        """Callers already passing the database enum are unchanged."""
        trade_id = _log_trade(db, DbPositionSide.LONG, entry_price=100.0, exit_price=120.0)

        trade = _fetch_trade(db, trade_id)
        assert trade.side == DbPositionSide.LONG
        assert float(trade.pnl_percent) == pytest.approx(20.0)
