"""A position loaded from a real database never carries a Decimal (#675).

SQLAlchemy hands ``Numeric`` columns back as ``Decimal``. Mixing one with a
float in engine arithmetic raises ``TypeError``, so the load boundary
(``DatabaseManager.get_active_positions``) coerces once. These tests use a real
in-memory database so a newly added ``Numeric`` column that skips the coercion
fails here instead of in a production cycle.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from src.database.manager import DatabaseManager
from src.database.models import TradeSource
from src.engines.live.execution.position_tracker import LivePositionTracker

pytestmark = pytest.mark.fast


@pytest.fixture
def db_with_open_position() -> tuple[DatabaseManager, int]:
    db = DatabaseManager("sqlite:///:memory:")
    session_id = db.create_trading_session(
        strategy_name="HyperGrowth",
        symbol="ETHUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )
    position_id = db.log_position(
        symbol="ETHUSDT",
        side="long",
        entry_price=2000.12345678,
        size=0.1,
        strategy_name="HyperGrowth",
        entry_order_id="entry-1",
        stop_loss=1800.0,
        take_profit=2600.0,
        quantity=0.05,
        entry_balance=1000.0,
        session_id=session_id,
        trailing_stop_price=1900.0,
        mfe_price=2100.0,
        mae_price=1950.0,
        stop_loss_order_id="sl-1",
    )
    db.update_position(
        position_id,
        current_price=2050.0,
        unrealized_pnl=2.5,
        unrealized_pnl_percent=0.25,
        original_size=0.1,
        current_size=0.05,
        last_partial_exit_price=2200.0,
        last_scale_in_price=1990.0,
    )
    return db, session_id


def _decimals(value: Any, path: str = "") -> list[str]:
    if isinstance(value, Decimal):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _decimals(v, f"{path}.{k}")]
    if isinstance(value, list | tuple):
        return [p for i, v in enumerate(value) for p in _decimals(v, f"{path}[{i}]")]
    return []


def test_get_active_positions_returns_no_decimal_from_a_real_database(db_with_open_position):
    db, session_id = db_with_open_position

    rows = db.get_active_positions(session_id)

    assert len(rows) == 1
    assert _decimals(rows) == []
    assert isinstance(rows[0]["entry_price"], float)
    assert isinstance(rows[0]["stop_loss"], float)
    assert isinstance(rows[0]["quantity"], float)


def test_recovered_live_position_numeric_fields_are_float(db_with_open_position):
    db, session_id = db_with_open_position
    tracker = LivePositionTracker(db_manager=db)

    recovered = tracker.recover_positions(session_id)

    assert len(recovered) == 1
    position = recovered[0]
    for name in (
        "size",
        "entry_price",
        "entry_balance",
        "quantity",
        "stop_loss",
        "take_profit",
        "original_size",
        "current_size",
        "trailing_stop_price",
    ):
        value = getattr(position, name)
        assert isinstance(value, float), f"{name} is {type(value).__name__}, not float"
    # The arithmetic the bug class used to break.
    assert position.entry_price * 0.5 - position.stop_loss == pytest.approx(-799.93827161)
