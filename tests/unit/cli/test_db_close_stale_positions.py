"""``atb db close-stale-positions``: dry run by default, --apply closes balance-neutrally (#1044)."""

import argparse
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from cli.commands.db import _close_stale_positions
from src.database.manager import DatabaseManager
from src.database.models import (
    Position,
    PositionStatus,
    ReconciliationAuditEvent,
    TradeSource,
    TradingSession,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


def _ended_session_with_open_position(
    db: DatabaseManager, mode: TradeSource = TradeSource.PAPER, order_id: str = "o-1"
) -> tuple[int, int]:
    session_id = db.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=mode,
        initial_balance=1000.0,
    )
    position_id = db.log_position(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        size=0.1,
        strategy_name="TestStrategy",
        entry_order_id=order_id,
        quantity=1.0,
        entry_balance=1000.0,
        session_id=session_id,
    )
    db.end_trading_session(session_id, carry_forward_open_positions=True)
    # Beyond the 7-day recovery window, so no restart will re-adopt the row.
    with db.get_session() as session:
        trading_session = session.query(TradingSession).filter_by(id=session_id).one()
        trading_session.start_time = datetime.now(UTC) - timedelta(hours=200)
        session.commit()
    return session_id, position_id


def _status(db: DatabaseManager, position_id: int) -> PositionStatus:
    with db.get_session() as session:
        return session.query(Position).filter(Position.id == position_id).one().status


def _audit_count(db: DatabaseManager) -> int:
    with db.get_session() as session:
        return session.query(ReconciliationAuditEvent).count()


def _run(db: DatabaseManager, apply: bool = False, session_id: int | None = None) -> int:
    ns = argparse.Namespace(apply=apply, session_id=session_id, env=None)
    with (
        patch("src.database.manager.DatabaseManager", return_value=db),
        patch("cli.commands.db._get_database_url_for_env", return_value="sqlite:///:memory:"),
    ):
        return _close_stale_positions(ns)


def test_dry_run_lists_rows_and_changes_nothing(db, capsys):
    _, position_id = _ended_session_with_open_position(db)

    exit_code = _run(db)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"position #{position_id}" in out
    assert "Dry run" in out
    assert _status(db, position_id) == PositionStatus.OPEN
    assert _audit_count(db) == 0


def test_apply_closes_rows_and_writes_audit(db, capsys):
    _, position_id = _ended_session_with_open_position(db)

    exit_code = _run(db, apply=True)

    assert exit_code == 0
    assert "Closed 1 row(s)" in capsys.readouterr().out
    assert _status(db, position_id) == PositionStatus.CLOSED
    assert _audit_count(db) == 1


def test_apply_never_closes_live_rows(db, capsys):
    _, live_position = _ended_session_with_open_position(db, TradeSource.LIVE, "live-1")
    _, paper_position = _ended_session_with_open_position(db, TradeSource.PAPER, "paper-1")

    exit_code = _run(db, apply=True)

    assert exit_code == 0
    assert "never closed" in capsys.readouterr().out
    assert _status(db, live_position) == PositionStatus.OPEN
    assert _status(db, paper_position) == PositionStatus.CLOSED


def test_session_id_filter_limits_apply(db):
    first_session, first_position = _ended_session_with_open_position(db, order_id="a")
    _, second_position = _ended_session_with_open_position(db, order_id="b")

    _run(db, apply=True, session_id=first_session)

    assert _status(db, first_position) == PositionStatus.CLOSED
    assert _status(db, second_position) == PositionStatus.OPEN


def test_reports_clean_state_when_nothing_is_stale(db, capsys):
    exit_code = _run(db)

    assert exit_code == 0
    assert "None found" in capsys.readouterr().out


def test_returns_error_when_database_cannot_be_read(capsys):
    ns = argparse.Namespace(apply=False, session_id=None, env=None)
    with patch("cli.commands.db._get_database_url_for_env", side_effect=RuntimeError("no url")):
        exit_code = _close_stale_positions(ns)

    assert exit_code == 1
    assert "no url" in capsys.readouterr().out


def test_apply_failure_returns_error(db, capsys):
    _ended_session_with_open_position(db)
    ns = argparse.Namespace(apply=True, session_id=None, env=None)
    with (
        patch("src.database.manager.DatabaseManager", return_value=db),
        patch("cli.commands.db._get_database_url_for_env", return_value="sqlite:///:memory:"),
        patch.object(
            db, "close_open_positions_in_ended_sessions", side_effect=RuntimeError("boom")
        ),
    ):
        exit_code = _close_stale_positions(ns)

    assert exit_code == 1
    assert "boom" in capsys.readouterr().out
