"""Engine-level tests for exchange-verified position adoption on restart (#683).

``LiveTradingEngine.start()`` runs against a real in-memory database, the engine's real
position tracker, order tracker and (for phantom closes) the real PositionReconciler, with
a hand-written exchange stub. Only the runtime I/O (trading loop, websockets, account sync,
startup exchange reconciliation, stop()) is neutralised so ``start()`` returns promptly.
State is captured when the (mocked) trading loop is entered — after carry-forward, before
shutdown drains anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.data_providers.exchange_interface import OrderStatus
from src.database.manager import DatabaseManager
from src.database.models import (
    Position,
    PositionStatus,
    ReconciliationAuditEvent,
    Trade,
    TradeSource,
)
from src.engines.live.position_adoption import AdoptionDecision, AdoptionVerdict
from src.engines.live.startup import ADOPTION_MESSAGE_MAX_POSITIONS, LiveStartupSequencer
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy
from tests.unit.engines.live.adoption_fakes import StubMarginExchange, make_stop_order

pytestmark = pytest.mark.fast

STRATEGY_NAME = "MlBasic"
SYMBOL = "BTCUSDT"
ENTRY_PRICE = 100.0
ADOPTION_ALERT_CODES = {
    "PHANTOM_POSITION_CLOSED",
    "POSITION_ADOPTED_REDUCED",
    "ADOPTION_UNVERIFIED",
}


@dataclass
class Snapshot:
    """Engine state at the moment the trading loop starts."""

    tracked: dict[str, Any] = field(default_factory=dict)
    close_only: bool = False
    balance: float = 0.0
    events: list[Any] = field(default_factory=list)


def _seed_session(
    db: DatabaseManager,
    *,
    mode: TradeSource = TradeSource.LIVE,
    active: bool = False,
    balance: float = 1000.0,
) -> int:
    session_id = db.create_trading_session(
        strategy_name=STRATEGY_NAME,
        symbol=SYMBOL,
        timeframe="1h",
        mode=mode,
        initial_balance=balance,
    )
    db.update_balance(balance, "session_start", "system", session_id)
    if not active:
        db.end_trading_session(session_id=session_id, final_balance=balance)
    return session_id


def _seed_position(
    db: DatabaseManager,
    session_id: int,
    *,
    quantity: float = 1.0,
    suffix: str = "a",
    side: str = "long",
    strategy: str = STRATEGY_NAME,
    stop_loss_order_id: str | None = None,
) -> int:
    return db.log_position(
        symbol=SYMBOL,
        side=side,
        entry_price=ENTRY_PRICE,
        size=0.1,
        strategy_name=strategy,
        entry_order_id=f"entry-{suffix}",
        quantity=quantity,
        entry_balance=1000.0,
        session_id=session_id,
        stop_loss_order_id=stop_loss_order_id,
    )


def _make_engine(
    db: DatabaseManager, exchange: StubMarginExchange, *, live: bool = True
) -> LiveTradingEngine:
    strategy = create_ml_basic_strategy()
    data_provider = MagicMock()
    data_provider.get_current_price.return_value = ENTRY_PRICE
    with (
        patch("src.engines.live.trading_engine.DatabaseManager"),
        patch("src.engines.live.trading_engine.get_config", return_value={}),
        patch(
            "src.engines.live.trading_engine._create_exchange_provider",
            return_value=(exchange, "stub"),
        ),
    ):
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=1000.0,
            enable_live_trading=live,
            resume_from_last_balance=True,
        )
    engine.db_manager = db
    return engine


def _start(engine: LiveTradingEngine) -> Snapshot:
    """Run start() with runtime I/O neutralised; return the state seen at loop entry."""
    snapshot = Snapshot()
    engine._record_event = MagicMock(wraps=engine._record_event)

    def _capture(*_args: Any, **_kwargs: Any) -> None:
        snapshot.tracked = dict(engine.live_position_tracker.positions)
        snapshot.close_only = engine._close_only_mode
        snapshot.balance = engine.current_balance

    engine._run_trading_loop = MagicMock(side_effect=_capture)
    engine._start_websocket_streams = MagicMock()
    engine._exit_if_loop_crashed = MagicMock()
    engine._print_final_stats = MagicMock()
    engine._reconcile_positions_with_exchange = MagicMock()
    engine.stop = MagicMock()
    sync_result = MagicMock()
    sync_result.success = True
    sync_result.data = {"balance_sync": {"corrected": False}}
    if engine.account_synchronizer is not None:
        engine.account_synchronizer.sync_account_data = MagicMock(return_value=sync_result)
    with patch("src.engines.live.reconciliation.PeriodicReconciler"):
        engine.start(symbol=SYMBOL, timeframe="1h", max_steps=0)
    if engine.order_tracker is not None:
        engine.order_tracker.stop()
    snapshot.events = list(engine._record_event.call_args_list)
    return snapshot


def _row(db: DatabaseManager, position_id: int) -> Position:
    with db.get_session() as s:
        position = s.query(Position).filter(Position.id == position_id).one()
        s.expunge(position)
        return position


def _event_codes(snapshot: Snapshot) -> list[str | None]:
    return [call.kwargs.get("error_code") for call in snapshot.events]


def _paged(snapshot: Snapshot, code: str) -> bool:
    return any(
        call.kwargs.get("error_code") == code and call.kwargs.get("alert") is True
        for call in snapshot.events
    )


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager("sqlite:///:memory:")


class TestCleanRestartAdoptsRealPosition:
    def test_real_position_is_adopted_and_tracked_with_its_stop(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old, stop_loss_order_id="sl-1")
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        snapshot = _start(engine)

        assert engine.trading_session_id != old
        assert _row(db, position_id).session_id == engine.trading_session_id
        assert _row(db, position_id).status == PositionStatus.OPEN
        [tracked] = snapshot.tracked.values()
        assert tracked.db_position_id == position_id
        assert tracked.stop_loss_order_id == "sl-1"
        assert "sl-1" in engine.order_tracker._pending_orders
        assert snapshot.close_only is False
        assert not set(_event_codes(snapshot)) & ADOPTION_ALERT_CODES

    def test_real_position_stranded_under_an_older_session_is_adopted(self, db):
        older = _seed_session(db)
        position_id = _seed_position(db, older)
        _seed_session(db)  # a later flat restart; its session owns no position
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        snapshot = _start(engine)

        assert _row(db, position_id).session_id == engine.trading_session_id
        assert [p.db_position_id for p in snapshot.tracked.values()] == [position_id]

    def test_adoption_is_recorded_in_the_audit_trail(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old)
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        _start(engine)

        with db.get_session() as s:
            audits = (
                s.query(ReconciliationAuditEvent)
                .filter(
                    ReconciliationAuditEvent.entity_id == position_id,
                    ReconciliationAuditEvent.field == "session_id",
                )
                .all()
            )
        assert [(a.old_value, a.new_value) for a in audits] == [
            (str(old), str(engine.trading_session_id))
        ]


class TestPhantomIsClosedNotAdopted:
    def test_phantom_is_not_tracked_and_is_closed_in_the_db(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old, stop_loss_order_id="sl-gone")
        exchange = StubMarginExchange({"BTC": 0.0})
        engine = _make_engine(db, exchange)

        snapshot = _start(engine)

        assert snapshot.tracked == {}
        assert _row(db, position_id).status == PositionStatus.CLOSED
        assert ("sl-gone", SYMBOL) in exchange.cancelled_orders

    def test_phantom_close_is_paged(self, db):
        old = _seed_session(db)
        _seed_position(db, old)
        engine = _make_engine(db, StubMarginExchange({"BTC": 0.0}))

        snapshot = _start(engine)

        assert _paged(snapshot, "PHANTOM_POSITION_CLOSED")

    def test_phantom_close_is_balance_neutral(self, db):
        old = _seed_session(db)
        _seed_position(db, old)
        engine = _make_engine(db, StubMarginExchange({"BTC": 0.0}))

        snapshot = _start(engine)

        assert snapshot.balance == pytest.approx(1000.0)
        assert db.get_current_balance(engine.trading_session_id) == pytest.approx(1000.0)

    def test_phantom_of_an_earlier_session_writes_no_trade(self, db):
        """A stale entry marked to today's price would invent P&L for the sizing metrics."""
        old = _seed_session(db)
        position_id = _seed_position(db, old)
        engine = _make_engine(db, StubMarginExchange({"BTC": 0.0}))

        _start(engine)

        with db.get_session() as s:
            assert s.query(Trade).filter(Trade.position_id == position_id).count() == 0

    def test_phantom_of_the_reused_session_leaves_a_balance_neutral_audit_trade(self, db):
        active = _seed_session(db, active=True)
        position_id = _seed_position(db, active)
        engine = _make_engine(db, StubMarginExchange({"BTC": 0.0}))

        _start(engine)

        with db.get_session() as s:
            trades = s.query(Trade).filter(Trade.position_id == position_id).all()
        assert [t.order_id for t in trades] == [f"reconcile_ext_{position_id}"]
        assert db.get_current_balance(active) == pytest.approx(1000.0)

    def test_phantom_whose_db_close_fails_is_not_reported_closed_and_latches_close_only(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old)
        db.close_position = MagicMock(return_value=False)
        engine = _make_engine(db, StubMarginExchange({"BTC": 0.0}))

        snapshot = _start(engine)

        assert snapshot.tracked == {}
        assert _row(db, position_id).status == PositionStatus.OPEN
        assert not _paged(snapshot, "PHANTOM_POSITION_CLOSED")
        assert _paged(snapshot, "ADOPTION_UNVERIFIED")
        assert snapshot.close_only is True

    def test_phantom_next_to_a_real_position_only_the_real_one_is_adopted(self, db):
        """#683: both rows used to pass because the aggregate balance exceeded each half-qty."""
        old = _seed_session(db)
        phantom_id = _seed_position(db, old, quantity=0.9, suffix="phantom")
        real_id = _seed_position(db, old, quantity=1.0, suffix="real")
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        snapshot = _start(engine)

        assert [p.db_position_id for p in snapshot.tracked.values()] == [real_id]
        assert _row(db, real_id).status == PositionStatus.OPEN
        assert _row(db, phantom_id).status == PositionStatus.CLOSED

    def test_dust_below_min_notional_is_closed_not_adopted(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old, quantity=0.01)
        exchange = StubMarginExchange(
            {"BTC": 0.01}, symbol_info={"min_qty": 0.00001, "min_notional": 5.0}
        )
        engine = _make_engine(db, exchange)

        snapshot = _start(engine)

        # 0.01 BTC at $100 = $1 < $5.
        assert snapshot.tracked == {}
        assert _row(db, position_id).status == PositionStatus.CLOSED
        assert _paged(snapshot, "PHANTOM_POSITION_CLOSED")


class TestUnbookedExitsAreLeftToTheReconciler:
    def test_row_whose_stop_filled_offline_is_adopted_unchanged_not_closed_as_phantom(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old, stop_loss_order_id="sl-filled")
        stop = make_stop_order("sl-filled", SYMBOL, OrderStatus.FILLED, filled_quantity=1.0)
        exchange = StubMarginExchange({"BTC": 0.0}, orders={"sl-filled": stop})
        engine = _make_engine(db, exchange)

        snapshot = _start(engine)

        assert exchange.cancelled_orders == []
        assert not _paged(snapshot, "PHANTOM_POSITION_CLOSED")
        assert _row(db, position_id).session_id == engine.trading_session_id
        assert [p.db_position_id for p in snapshot.tracked.values()] == [position_id]

    def test_unresolved_partial_exit_order_blocks_the_resize(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old, quantity=1.0)
        db.create_order(
            position_id=position_id,
            order_type="PARTIAL_EXIT",
            symbol=SYMBOL,
            side="long",
            quantity=0.5,
            strategy_name=STRATEGY_NAME,
            session_id=old,
            exchange_order_id="exit-1",
        )
        engine = _make_engine(db, StubMarginExchange({"BTC": 0.5}))

        snapshot = _start(engine)

        row = _row(db, position_id)
        assert row.session_id == engine.trading_session_id
        assert row.current_size is None
        assert not _paged(snapshot, "POSITION_ADOPTED_REDUCED")


class TestPartialInventory:
    def test_position_is_adopted_at_the_size_actually_held(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old, quantity=1.0)
        engine = _make_engine(db, StubMarginExchange({"BTC": 0.4}))

        snapshot = _start(engine)

        [tracked] = snapshot.tracked.values()
        assert tracked.db_position_id == position_id
        assert float(tracked.current_size) == pytest.approx(0.04)
        assert float(tracked.original_size) == pytest.approx(0.1)
        row = _row(db, position_id)
        assert row.session_id == engine.trading_session_id
        assert float(row.current_size) == pytest.approx(0.04)
        assert _paged(snapshot, "POSITION_ADOPTED_REDUCED")


class TestLookupFailureIsFailSafe:
    def test_nothing_is_adopted_or_closed_and_close_only_latches(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old)
        exchange = StubMarginExchange({"BTC": 1.0})
        exchange.fail_lookups = True
        engine = _make_engine(db, exchange)

        snapshot = _start(engine)

        assert snapshot.tracked == {}
        row = _row(db, position_id)
        assert row.status == PositionStatus.OPEN
        assert row.session_id == old
        assert snapshot.close_only is True
        assert _paged(snapshot, "ADOPTION_UNVERIFIED")

    def test_unreadable_db_candidates_latch_close_only(self, db):
        old = _seed_session(db)
        _seed_position(db, old)
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))
        db.get_open_positions_across_sessions = MagicMock(side_effect=RuntimeError("db down"))

        snapshot = _start(engine)

        assert snapshot.close_only is True
        assert _paged(snapshot, "ADOPTION_UNVERIFIED")


class TestCrashRestartReusesActiveSession:
    def test_owned_real_position_is_loaded_and_owned_phantom_is_closed(self, db):
        active = _seed_session(db, active=True)
        phantom_id = _seed_position(db, active, quantity=0.9, suffix="phantom")
        real_id = _seed_position(db, active, quantity=1.0, suffix="real")
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        snapshot = _start(engine)

        assert engine.trading_session_id == active
        assert [p.db_position_id for p in snapshot.tracked.values()] == [real_id]
        assert _row(db, phantom_id).status == PositionStatus.CLOSED

    def test_stray_real_position_in_an_older_session_joins_the_active_session(self, db):
        stray_session = _seed_session(db)
        stray_id = _seed_position(db, stray_session, suffix="stray")
        active = _seed_session(db, active=True)
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        snapshot = _start(engine)

        assert engine.trading_session_id == active
        assert _row(db, stray_id).session_id == active
        assert [p.db_position_id for p in snapshot.tracked.values()] == [stray_id]

    def test_owned_position_is_still_loaded_when_the_exchange_lookup_fails(self, db):
        active = _seed_session(db, active=True)
        owned_id = _seed_position(db, active)
        exchange = StubMarginExchange({"BTC": 1.0})
        exchange.fail_lookups = True
        engine = _make_engine(db, exchange)

        snapshot = _start(engine)

        assert [p.db_position_id for p in snapshot.tracked.values()] == [owned_id]
        assert snapshot.close_only is True


class TestScopeOfCandidates:
    def test_other_strategy_and_other_mode_rows_are_left_alone(self, db):
        paper_session = _seed_session(db, mode=TradeSource.PAPER)
        paper_id = _seed_position(db, paper_session, suffix="paper")
        live_session = _seed_session(db)
        other_strategy_id = _seed_position(db, live_session, suffix="other", strategy="Other")
        engine = _make_engine(db, StubMarginExchange({"BTC": 5.0}))

        snapshot = _start(engine)

        assert snapshot.tracked == {}
        for position_id, session_id in (
            (paper_id, paper_session),
            (other_strategy_id, live_session),
        ):
            row = _row(db, position_id)
            assert row.status == PositionStatus.OPEN
            assert row.session_id == session_id

    def test_row_of_another_still_active_session_is_not_taken(self, db):
        concurrent = _seed_session(db, active=True)
        concurrent_id = _seed_position(db, concurrent, suffix="concurrent")
        active = _seed_session(db, active=True)
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        snapshot = _start(engine)

        assert engine.trading_session_id == active
        assert snapshot.tracked == {}
        assert _row(db, concurrent_id).session_id == concurrent

    def test_stale_open_row_with_a_terminal_trade_is_not_adopted(self, db):
        old = _seed_session(db)
        position_id = _seed_position(db, old)
        db.log_trade(
            symbol=SYMBOL,
            side="long",
            entry_price=ENTRY_PRICE,
            exit_price=ENTRY_PRICE,
            size=0.1,
            entry_time=_row(db, position_id).entry_time,
            exit_time=_row(db, position_id).entry_time,
            pnl=0.0,
            exit_reason="stop_loss",
            strategy_name=STRATEGY_NAME,
            source=TradeSource.LIVE,
            session_id=old,
            position_id=position_id,
            exit_order_id="exit-1",
        )
        engine = _make_engine(db, StubMarginExchange({"BTC": 1.0}))

        snapshot = _start(engine)

        assert snapshot.tracked == {}
        assert _row(db, position_id).session_id == old

    def test_no_open_rows_means_no_exchange_lookup(self, db):
        _seed_session(db)
        exchange = StubMarginExchange({"BTC": 1.0})
        exchange.get_margin_account_asset = MagicMock(side_effect=AssertionError("looked up"))
        engine = _make_engine(db, exchange)

        snapshot = _start(engine)

        assert snapshot.tracked == {}
        assert snapshot.close_only is False


class TestPaperModeKeepsPreviousSessionCarryForward:
    def test_previous_session_position_is_carried_forward_without_exchange_checks(self, db):
        old = _seed_session(db, mode=TradeSource.PAPER)
        position_id = _seed_position(db, old)
        engine = _make_engine(db, StubMarginExchange({}), live=False)

        snapshot = _start(engine)

        assert _row(db, position_id).session_id == engine.trading_session_id
        assert [p.db_position_id for p in snapshot.tracked.values()] == [position_id]


class TestAlertMessageIsBounded:
    def test_long_lists_are_truncated_with_a_count(self):
        decisions = [
            AdoptionDecision({"id": i, "session_id": 1}, AdoptionVerdict.PHANTOM, "no inventory")
            for i in range(ADOPTION_MESSAGE_MAX_POSITIONS + 3)
        ]

        text = LiveStartupSequencer._describe(decisions)

        assert text.count("no inventory") == ADOPTION_MESSAGE_MAX_POSITIONS
        assert text.endswith("and 3 more")
