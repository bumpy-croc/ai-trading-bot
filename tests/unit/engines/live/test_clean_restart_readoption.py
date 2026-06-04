"""Engine-level tests for clean-restart position re-adoption (#668).

On a graceful restart the engine recovers BALANCE from the most recent inactive
session but creates a NEW session. Before this fix, OPEN positions stayed bound
to the OLD session (``Position.session_id``) and ``get_active_positions`` —
filtered by the new session — returned nothing, orphaning the position (still
OPEN in the DB, invisible to the live tracker / risk manager / reconciler).

These tests drive ``LiveTradingEngine.start()`` with a REAL in-memory database
(so the carry-forward DB transaction runs for real) while the heavy runtime I/O
(trading loop, websocket streams, account sync, periodic reconciler) is mocked
out so ``start()`` returns promptly. They assert the previously-orphaned OPEN
position is carried forward: present in ``live_position_tracker``, registered
with the risk manager, its server-side stop-loss tracked by ``OrderTracker``,
and that startup reconciliation takes the non-empty ``reconcile_startup`` branch
rather than the "No local positions to reconcile" path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import Position, PositionStatus, TradeSource
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast

# Must match create_ml_basic_strategy().name (the value _strategy_name() returns),
# since get_last_session_id() filters the recovery candidate by strategy name.
STRATEGY_NAME = "MlBasic"
SYMBOL = "BTCUSDT"


def _seed_inactive_session_with_open_position(
    db: DatabaseManager,
    *,
    symbol: str = SYMBOL,
    strategy: str = STRATEGY_NAME,
    stop_loss_order_id: str | None = "sl-exch-1",
    entry_order_id: str = "entry-exch-1",
) -> tuple[int, int]:
    """Create an ENDED session that still owns an OPEN position.

    Returns (session_id, position_id).
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
    # Gracefully end the session (mirrors a clean shutdown): is_active -> False.
    db.end_trading_session(session_id=session_id, final_balance=1000.0)
    return session_id, position_id


def _make_live_engine_with_real_db(db: DatabaseManager) -> LiveTradingEngine:
    """Build a live-mode engine wired to a real in-memory DB.

    The exchange provider is mocked (so live mode initializes its account
    synchronizer + order tracker) but the DB is real so the carry-forward
    transaction executes end to end.
    """
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

    # Swap in the real database used to seed the orphan.
    engine.db_manager = db
    return engine


def _run_start_with_mocked_runtime(
    engine: LiveTradingEngine,
    reconciler_cls: MagicMock,
) -> None:
    """Drive start() to completion with heavy runtime I/O neutralized."""
    # Make the trading loop a no-op so the worker thread dies immediately and
    # start()'s keep-alive loop exits and calls stop().
    engine._run_trading_loop = MagicMock()
    engine._start_websocket_streams = MagicMock()
    engine._exit_if_loop_crashed = MagicMock()
    # Display-only; mocked to avoid Decimal/float arithmetic in the final-stats
    # banner when balance is recovered from the DB as a Decimal (orthogonal to
    # the position carry-forward under test).
    engine._print_final_stats = MagicMock()

    # Account sync: report success with no balance correction so start() proceeds
    # to position reconciliation using the real recovered state.
    sync_result = MagicMock()
    sync_result.success = True
    sync_result.data = {"balance_sync": {"corrected": False}}
    engine.account_synchronizer.sync_account_data = MagicMock(return_value=sync_result)

    with patch("src.engines.live.reconciliation.PositionReconciler", reconciler_cls):
        engine.start(symbol=SYMBOL, timeframe="1h", max_steps=0)


@pytest.fixture
def reconciler_cls():
    """A PositionReconciler stand-in that records how it was invoked.

    Crucially, ``reconcile_startup`` runs AFTER recovery has loaded positions
    into the tracker but BEFORE ``stop()`` (which, in live mode, closes and
    drains them). So the snapshot it receives is the right place to observe the
    carried-forward tracker state — capture it for assertions.
    """
    captured: dict[str, object] = {}

    def _capture_startup(positions_snapshot):
        # Copy keys/values so later draining can't mutate what we asserted on.
        captured["startup_snapshot"] = dict(positions_snapshot)
        return []

    instance = MagicMock()
    instance.reconcile_startup.side_effect = _capture_startup
    instance.resolve_pending_orders.return_value = []
    cls = MagicMock(return_value=instance)
    cls.instance = instance
    cls.captured = captured
    return cls


class TestCleanRestartCarriesOpenPositionForward:
    """The OPEN orphan under the old session is carried into the new session and
    loaded into the live tracker on start()."""

    def test_open_position_loaded_into_tracker(self, reconciler_cls):
        db = DatabaseManager("sqlite:///:memory:")
        old_session_id, position_id = _seed_inactive_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        # A brand-new session was created (clean-restart semantics preserved).
        assert engine.trading_session_id is not None
        assert engine.trading_session_id != old_session_id
        # The previously-orphaned position was loaded into the live tracker
        # before reconciliation (snapshot captured at reconcile_startup time,
        # ahead of the live-mode stop() that drains the tracker).
        tracked = reconciler_cls.captured["startup_snapshot"]
        assert len(tracked) == 1
        position = next(iter(tracked.values()))
        assert position.symbol == SYMBOL
        assert position.db_position_id == position_id

    def test_position_carried_to_new_session_in_db(self, reconciler_cls):
        db = DatabaseManager("sqlite:///:memory:")
        old_session_id, position_id = _seed_inactive_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        # The DB row is now bound to the new session and surfaces under it.
        with db.get_session() as s:
            pos = s.query(Position).filter(Position.id == position_id).first()
            assert pos.session_id == engine.trading_session_id
            assert pos.status == PositionStatus.OPEN
        active_ids = [p["id"] for p in db.get_active_positions(engine.trading_session_id)]
        assert position_id in active_ids

    def test_recovered_stop_loss_tracked_by_order_tracker(self, reconciler_cls):
        db = DatabaseManager("sqlite:///:memory:")
        _seed_inactive_session_with_open_position(db, stop_loss_order_id="sl-exch-1")
        engine = _make_live_engine_with_real_db(db)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        assert engine.order_tracker is not None
        # The recovered position's server-side SL order is being monitored.
        assert "sl-exch-1" in engine.order_tracker._pending_orders

    def test_recovered_position_registered_with_risk_manager(self, reconciler_cls):
        db = DatabaseManager("sqlite:///:memory:")
        _seed_inactive_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        # Spy on the real risk manager.
        engine.risk_manager.update_position = MagicMock(wraps=engine.risk_manager.update_position)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        assert engine.risk_manager.update_position.called
        symbols = {
            c.kwargs.get("symbol") for c in engine.risk_manager.update_position.call_args_list
        }
        assert SYMBOL in symbols

    def test_reconciler_takes_nonempty_startup_branch(self, reconciler_cls):
        """With the orphan carried forward, the tracker is non-empty, so startup
        reconciliation re-verifies the position via reconcile_startup() rather
        than falling through to the empty 'No local positions to reconcile' path
        (resolve_pending_orders)."""
        db = DatabaseManager("sqlite:///:memory:")
        _seed_inactive_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        reconciler_cls.instance.reconcile_startup.assert_called_once()
        # The empty-tracker branch must NOT have been taken.
        reconciler_cls.instance.resolve_pending_orders.assert_not_called()

    def test_reassign_called_with_symbol_and_strategy(self, reconciler_cls):
        """The carry-forward is scoped to the active symbol + strategy so a
        co-tenant symbol/strategy in the same DB is not pulled in."""
        db = DatabaseManager("sqlite:///:memory:")
        old_session_id, _ = _seed_inactive_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.reassign_open_positions_to_session = MagicMock(
            wraps=engine.db_manager.reassign_open_positions_to_session
        )

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        engine.db_manager.reassign_open_positions_to_session.assert_called_once()
        kwargs = engine.db_manager.reassign_open_positions_to_session.call_args.kwargs
        assert kwargs["old_session_id"] == old_session_id
        assert kwargs["new_session_id"] == engine.trading_session_id
        assert kwargs["symbol"] == SYMBOL
        assert kwargs["strategy_name"] == STRATEGY_NAME


class TestCleanRestartNoOrphan:
    """When there is no inactive session to recover, the new code is inert and the
    normal boot path is unchanged (regression guard)."""

    def test_no_recovery_does_not_call_reassign(self, reconciler_cls):
        db = DatabaseManager("sqlite:///:memory:")  # empty DB, nothing to recover
        engine = _make_live_engine_with_real_db(db)
        engine.db_manager.reassign_open_positions_to_session = MagicMock(
            wraps=engine.db_manager.reassign_open_positions_to_session
        )

        _run_start_with_mocked_runtime(engine, reconciler_cls)

        assert engine._recovered_inactive_session_id is None
        engine.db_manager.reassign_open_positions_to_session.assert_not_called()
        # No positions, so reconciliation takes the empty branch.
        reconciler_cls.instance.resolve_pending_orders.assert_called_once()
        reconciler_cls.instance.reconcile_startup.assert_not_called()


class TestRecoverExistingSessionRemembersInactiveId:
    """Unit-level: _recover_existing_session captures the inactive session id on
    the clean-restart branch (and only there)."""

    def test_inactive_branch_sets_recovered_id(self):
        db = DatabaseManager("sqlite:///:memory:")
        old_session_id, _ = _seed_inactive_session_with_open_position(db)
        engine = _make_live_engine_with_real_db(db)
        engine._active_symbol = SYMBOL

        recovered = engine._recover_existing_session()

        assert recovered is not None
        # New-session semantics preserved: trading_session_id stays unset...
        assert engine.trading_session_id is None
        # ...but the old session id is remembered for carry-forward.
        assert engine._recovered_inactive_session_id == old_session_id

    def test_active_branch_does_not_set_recovered_id(self):
        db = DatabaseManager("sqlite:///:memory:")
        engine = _make_live_engine_with_real_db(db)
        engine._active_symbol = SYMBOL
        # Force the active (crash-recovery) path.
        engine.db_manager.get_active_session_id = MagicMock(return_value=55)
        engine.db_manager.recover_last_balance = MagicMock(return_value=900.0)

        recovered = engine._recover_existing_session()

        assert recovered == 900.0
        assert engine.trading_session_id == 55
        assert engine._recovered_inactive_session_id is None
