"""DB persistence of partial exits/scale-ins must mirror the runtime tracker.

Regression for the P1 units divergence: the live tracker previously passed
the policy's fraction-of-ORIGINAL to ``apply_partial_exit_update`` /
``apply_scale_in_update``, which mutate the balance-fraction
``Position.current_size`` by that raw value. Example: a 25%-of-original
partial on a 10%-of-balance position left runtime ``current_size`` at 0.075
while the DB recorded ``max(0, 0.10 - 0.25) = 0.0`` — a phantom-closed row.
Crash recovery re-registers ``daily_risk_used`` from the DB ``current_size``
(``src/engines/live/recovery.py``), so the divergence corrupts recovery state
once ``live_partial_operations`` is enabled.

Uses a real in-memory SQLite ``DatabaseManager`` (same pattern as
``tests/unit/database/test_position_close_atomic.py``) so the persistence
path is exercised end to end rather than mocked.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.database.manager import DatabaseManager
from src.database.models import PartialOperationType, PartialTrade, Position, TradeSource
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)

pytestmark = pytest.mark.unit

ENTRY_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _make_db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


def _track_position(
    db: DatabaseManager, tracker: LivePositionTracker, side: PositionSide
) -> LivePosition:
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="ETHUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )
    db_id = db.log_position(
        symbol="ETHUSDT",
        side="long" if side == PositionSide.LONG else "short",
        entry_price=100.0,
        size=0.10,
        strategy_name="TestStrategy",
        entry_order_id="order-1",
        quantity=1.0,
        entry_balance=1000.0,
        session_id=session_id,
    )
    position = LivePosition(
        symbol="ETHUSDT",
        side=side,
        entry_price=100.0,
        entry_time=ENTRY_TIME,
        size=0.10,
        entry_balance=1000.0,
        order_id="order-1",
        db_position_id=db_id,
    )
    tracker.track_recovered_position(position, db_id=db_id)
    return position


def _db_current_size(db: DatabaseManager, db_id: int) -> float:
    with db.get_session() as session:
        row = session.query(Position).filter(Position.id == db_id).first()
        assert row is not None
        return float(row.current_size)


@pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT])
class TestPartialOpsDbPersistenceMirrorsRuntime:
    def test_partial_exit_persists_runtime_delta(self, side) -> None:
        db = _make_db()
        tracker = LivePositionTracker(db_manager=db)
        position = _track_position(db, tracker, side)
        exit_price = 105.0 if side == PositionSide.LONG else 95.0

        # 25%-of-original on a 0.10 position = 0.025 balance-fraction slice.
        result = tracker.apply_partial_exit(
            order_id="order-1",
            delta_fraction=0.025,
            price=exit_price,
            target_level=0,
            basis_balance=1000.0,
        )

        assert result is not None
        assert result.new_current_size == pytest.approx(0.075)
        assert position.current_size == pytest.approx(0.075)
        # DB must mirror runtime exactly — NOT max(0, 0.10 - 0.25) = 0.0.
        db_id = tracker._position_db_ids["order-1"]
        assert db_id is not None
        assert _db_current_size(db, db_id) == pytest.approx(position.current_size)

        # The partial-trade ledger row is in the same balance-fraction units.
        with db.get_session() as session:
            pt = (
                session.query(PartialTrade)
                .filter(
                    PartialTrade.position_id == db_id,
                    PartialTrade.operation_type == PartialOperationType.PARTIAL_EXIT,
                )
                .one()
            )
            assert float(pt.size) == pytest.approx(0.025)

    def test_scale_in_persists_runtime_delta(self, side) -> None:
        db = _make_db()
        tracker = LivePositionTracker(db_manager=db)
        position = _track_position(db, tracker, side)

        # 30%-of-original on a 0.10 position = 0.03 balance-fraction add.
        result = tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=0.03,
            price=102.0,
            threshold_level=0,
            max_position_size=0.20,
        )

        assert result is not None
        assert result.new_current_size == pytest.approx(0.13)
        assert position.current_size == pytest.approx(0.13)
        # DB must mirror runtime exactly — NOT min(1.0, 0.10 + 0.30) = 0.40.
        db_id = tracker._position_db_ids["order-1"]
        assert db_id is not None
        assert _db_current_size(db, db_id) == pytest.approx(position.current_size)

    def test_capped_scale_in_persists_applied_delta_only(self, side) -> None:
        """When the tracker's max-position cap trims the add, the DB gets the
        delta actually applied, keeping DB current_size == runtime."""
        db = _make_db()
        tracker = LivePositionTracker(db_manager=db)
        position = _track_position(db, tracker, side)

        result = tracker.apply_scale_in(
            order_id="order-1",
            delta_fraction=0.05,
            price=102.0,
            threshold_level=0,
            max_position_size=0.12,  # only 0.02 headroom
        )

        assert result is not None
        assert position.current_size == pytest.approx(0.12)
        db_id = tracker._position_db_ids["order-1"]
        assert db_id is not None
        assert _db_current_size(db, db_id) == pytest.approx(position.current_size)
