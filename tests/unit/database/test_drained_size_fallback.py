"""A fully drained position (current_size == 0.0) must not read back as full size.

``apply_partial_exit_update`` / ``apply_scale_in_update`` used
``position.current_size or position.size``: a Decimal 0 is falsy, so a position
drained to 0.0 by partial exits was treated as holding its full ``size`` on the
next update. Only a NULL ``current_size`` falls back to ``size``.
"""

from __future__ import annotations

import pytest

from src.database.manager import DatabaseManager
from src.database.models import Position, TradeSource

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


def _log_position(db: DatabaseManager) -> int:
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="ETHUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )
    return db.log_position(
        symbol="ETHUSDT",
        side="long",
        entry_price=100.0,
        size=0.10,
        strategy_name="TestStrategy",
        entry_order_id="order-1",
        quantity=1.0,
        entry_balance=1000.0,
        session_id=session_id,
    )


def _set_current_size(db: DatabaseManager, position_id: int, value) -> None:
    with db.get_session() as session:
        row = session.query(Position).filter(Position.id == position_id).first()
        row.current_size = value
        session.commit()


def _current_size(db: DatabaseManager, position_id: int):
    with db.get_session() as session:
        row = session.query(Position).filter(Position.id == position_id).first()
        return None if row.current_size is None else float(row.current_size)


class TestPartialExitUpdate:
    def test_drained_position_stays_at_zero(self, db: DatabaseManager) -> None:
        position_id = _log_position(db)
        db.apply_partial_exit_update(position_id, 0.10, price=110.0, target_level=0)
        assert _current_size(db, position_id) == 0.0

        db.apply_partial_exit_update(position_id, 0.02, price=111.0, target_level=1)

        assert _current_size(db, position_id) == 0.0

    def test_null_current_size_falls_back_to_size(self, db: DatabaseManager) -> None:
        position_id = _log_position(db)
        _set_current_size(db, position_id, None)

        db.apply_partial_exit_update(position_id, 0.02, price=110.0, target_level=0)

        assert _current_size(db, position_id) == pytest.approx(0.08)

    def test_partial_exit_reduces_remaining_size(self, db: DatabaseManager) -> None:
        position_id = _log_position(db)
        _set_current_size(db, position_id, 0.06)

        db.apply_partial_exit_update(position_id, 0.02, price=110.0, target_level=0)

        assert _current_size(db, position_id) == pytest.approx(0.04)


class TestScaleInUpdate:
    def test_scale_in_on_drained_position_adds_to_zero(self, db: DatabaseManager) -> None:
        position_id = _log_position(db)
        _set_current_size(db, position_id, 0.0)

        db.apply_scale_in_update(position_id, 0.03, price=103.0, threshold_level=0)

        assert _current_size(db, position_id) == pytest.approx(0.03)

    def test_null_current_size_falls_back_to_size(self, db: DatabaseManager) -> None:
        position_id = _log_position(db)
        _set_current_size(db, position_id, None)

        db.apply_scale_in_update(position_id, 0.03, price=103.0, threshold_level=0)

        assert _current_size(db, position_id) == pytest.approx(0.13)

    def test_scale_in_adds_to_remaining_size(self, db: DatabaseManager) -> None:
        position_id = _log_position(db)
        _set_current_size(db, position_id, 0.06)

        db.apply_scale_in_update(position_id, 0.03, price=103.0, threshold_level=0)

        assert _current_size(db, position_id) == pytest.approx(0.09)
