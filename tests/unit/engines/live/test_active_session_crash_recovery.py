"""Engine-level tests for active-session crash recovery when balance is
unrecoverable (#743).

Sibling to ``test_clean_restart_readoption.py`` (#668), which fixed the
INACTIVE-session path: a clean restart remembers the old session id so its
OPEN positions carry forward into the new session INDEPENDENT of whether a
positive balance was recovered. The ACTIVE-session (crash-recovery) path never
got the equivalent fix — reuse of the existing session was gated behind
``if recovered_balance and recovered_balance > 0:``. When the balance read
came back ``None``/non-positive (a genuine near-total-loss balance, or a
trades-fallback calculation summing to <= 0) while a position was genuinely
OPEN under that still-``is_active`` session, the engine:

1. treated the lookup as "no session found" and created a BRAND NEW session,
2. never reloaded the OPEN position into the live tracker (it stays bound to
   the old session id), so both reconcilers see the bot as flat,
3. could re-enter the same symbol -> real double exposure,
4. left the old session ``is_active`` forever, repeating on every restart.

These tests drive ``LiveTradingEngine.start()`` with a REAL in-memory database
(so the reuse/positivity logic runs for real), mirroring
``test_clean_restart_readoption.py``'s harness.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import Position, PositionStatus, TradeSource
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast

# Must match create_ml_basic_strategy().name (the value _strategy_name() returns).
STRATEGY_NAME = "MlBasic"
SYMBOL = "BTCUSDT"


def _seed_active_session_with_open_position(
    db: DatabaseManager,
    *,
    symbol: str = SYMBOL,
    strategy: str = STRATEGY_NAME,
    stop_loss_order_id: str | None = "sl-exch-1",
    entry_order_id: str = "entry-exch-1",
) -> tuple[int, int]:
    """Create a session that is STILL is_active (simulating a crash — no
    clean shutdown ever called ``end_trading_session``) and owns an OPEN
    position. Returns (session_id, position_id).
    """
    session_id = db.create_trading_session(
        strategy_name=strategy,
        symbol=symbol,
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=1000.0,
    )
    position_id = db.log_position(
        symbol=symbol,
        side="long",
        entry_price=100.0,
        size=0.1,
        strategy_name=strategy,
        entry_order_id=entry_order_id,
        quantity=1.0,
        entry_balance=1000.0,
        session_id=session_id,
        stop_loss_order_id=stop_loss_order_id,
    )
    # Deliberately do NOT call end_trading_session — the session stays
    # is_active, exactly like an engine that crashed mid-trade.
    return session_id, position_id


def _make_live_engine_with_real_db(db: DatabaseManager) -> LiveTradingEngine:
    strategy = create_ml_basic_strategy()
    mock_data_provider = MagicMock()

    with (
        patch("src.engines.live.trading_engine.DatabaseManager"),
        patch("src.engines.live.trading_engine.get_config", return_value={}),
        patch(
            "src.engines.live.trading_engine._create_exchange_provider",
            return_value=(MagicMock(), "mock"),
        ),
    ):
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=1000.0,
            enable_live_trading=True,
            resume_from_last_balance=True,
        )

    engine.db_manager = db
    return engine


def _run_start_with_mocked_runtime(
    engine: LiveTradingEngine,
    reconciler_cls: MagicMock,
) -> None:
    """Drive start() to completion with heavy runtime I/O neutralized."""
    engine._run_trading_loop = MagicMock()
    engine._start_websocket_streams = MagicMock()
    engine._exit_if_loop_crashed = MagicMock()
    engine._print_final_stats = MagicMock()

    sync_result = MagicMock()
    sync_result.success = True
    sync_result.data = {"balance_sync": {"corrected": False}}
    engine.account_synchronizer.sync_account_data = MagicMock(return_value=sync_result)

    with patch("src.engines.live.reconciliation.PositionReconciler", reconciler_cls):
        engine.start(symbol=SYMBOL, timeframe="1h", max_steps=0)


@pytest.fixture
def reconciler_cls():
    """A PositionReconciler stand-in that records how it was invoked."""
    captured: dict[str, object] = {}

    def _capture_startup(positions_snapshot):
        captured["startup_snapshot"] = dict(positions_snapshot)
        return []

    instance = MagicMock()
    instance.reconcile_startup.side_effect = _capture_startup
    instance.resolve_pending_orders.return_value = []
    cls = MagicMock(return_value=instance)
    cls.instance = instance
    cls.captured = captured
    return cls


class TestActiveSessionReusedDespiteUnrecoverableBalance:
    """#743: the active/crash session must be reused (and its OPEN position
    recovered) even when the balance read comes back None/non-positive."""

    @pytest.mark.parametrize("bad_balance", [None, 0.0])
    def test_session_is_reused_not_recreated(self, reconciler_cls, bad_balance):
        db = DatabaseManager("sqlite:///:memory:")
        old_session_id, _ = _seed_active_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.recover_last_balance = MagicMock(return_value=bad_balance)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        # The SAME session must be reused, not a freshly created one.
        assert engine.trading_session_id == old_session_id

    def test_open_position_loaded_into_tracker(self, reconciler_cls):
        db = DatabaseManager("sqlite:///:memory:")
        _, position_id = _seed_active_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.recover_last_balance = MagicMock(return_value=None)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        # Not silently orphaned: loaded into the tracker before reconciliation.
        tracked = reconciler_cls.captured["startup_snapshot"]
        assert len(tracked) == 1
        position = next(iter(tracked.values()))
        assert position.symbol == SYMBOL
        assert position.db_position_id == position_id

    def test_position_still_open_under_reused_session_in_db(self, reconciler_cls):
        db = DatabaseManager("sqlite:///:memory:")
        old_session_id, position_id = _seed_active_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.recover_last_balance = MagicMock(return_value=0.0)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        with db.get_session() as s:
            pos = s.query(Position).filter(Position.id == position_id).first()
            assert pos.session_id == old_session_id
            assert pos.status == PositionStatus.OPEN
        active_ids = [p["id"] for p in db.get_active_positions(engine.trading_session_id)]
        assert position_id in active_ids

    def test_reconciler_takes_nonempty_startup_branch(self, reconciler_cls):
        """The tracker is non-empty, so startup reconciliation re-verifies the
        position via reconcile_startup() instead of the empty
        'No local positions to reconcile' path."""
        db = DatabaseManager("sqlite:///:memory:")
        _seed_active_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.recover_last_balance = MagicMock(return_value=None)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        reconciler_cls.instance.reconcile_startup.assert_called_once()
        reconciler_cls.instance.resolve_pending_orders.assert_not_called()

    def test_loud_alert_fires_for_unrecoverable_balance(self, reconciler_cls):
        """Falling through to a fresh session used to be silent. Now a CRITICAL
        alert must fire via the same `_record_event(..., alert=True)`
        mechanism every other CRITICAL condition in this module uses."""
        db = DatabaseManager("sqlite:///:memory:")
        _seed_active_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.recover_last_balance = MagicMock(return_value=None)
        engine._record_event = MagicMock(wraps=engine._record_event)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        alert_calls = [
            call
            for call in engine._record_event.call_args_list
            if call.kwargs.get("error_code") == "ACTIVE_SESSION_BALANCE_UNRECOVERABLE"
        ]
        assert len(alert_calls) == 1
        assert alert_calls[0].kwargs["alert"] is True
        assert alert_calls[0].kwargs["severity"] == "critical"

    def test_no_second_session_created(self, reconciler_cls):
        """Regression guard for the core bug: recovering with a bad balance
        must not call create_trading_session a second time."""
        db = DatabaseManager("sqlite:///:memory:")
        _seed_active_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.recover_last_balance = MagicMock(return_value=None)
        engine.db_manager.create_trading_session = MagicMock(
            wraps=engine.db_manager.create_trading_session
        )

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        engine.db_manager.create_trading_session.assert_not_called()


class TestRecoverExistingSessionActivePathUnitLevel:
    """Unit-level: _recover_existing_session reuses the active session
    unconditionally, independent of the balance-positivity check."""

    def test_active_session_reused_when_balance_none(self):
        db = DatabaseManager("sqlite:///:memory:")
        engine = _make_live_engine_with_real_db(db)
        engine._active_symbol = SYMBOL
        engine.db_manager.get_active_session_id = MagicMock(return_value=55)
        engine.db_manager.recover_last_balance = MagicMock(return_value=None)

        recovered = engine._recover_existing_session()

        assert recovered is None  # balance genuinely unusable
        assert engine.trading_session_id == 55  # but the session IS reused
        assert engine._recovered_inactive_session_id is None

    def test_active_session_reused_when_balance_zero(self):
        db = DatabaseManager("sqlite:///:memory:")
        engine = _make_live_engine_with_real_db(db)
        engine._active_symbol = SYMBOL
        engine.db_manager.get_active_session_id = MagicMock(return_value=56)
        engine.db_manager.recover_last_balance = MagicMock(return_value=0.0)

        recovered = engine._recover_existing_session()

        assert recovered is None
        assert engine.trading_session_id == 56
