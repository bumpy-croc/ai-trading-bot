"""Live startup sequencer (#486).

Owns the engine bootstrap sequence: session recovery + creation, #668 open-
position carry-forward, #657 self-heal, exchange account sync, runtime-service
startup, and the main trading-loop launch. Extracted verbatim from
``LiveTradingEngine.start`` (the public ``start`` stays on the engine and
delegates here). All engine state lives on the engine and is read/written
through the backref at call time; this class holds no state of its own.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

from src.database.models import EventType, TradeSource
from src.infrastructure.logging.context import set_context, update_context
from src.infrastructure.logging.events import log_engine_event

if TYPE_CHECKING:
    from src.data_providers.data_provider import DataProvider
    from src.database.manager import DatabaseManager
    from src.engines.live.execution.execution_engine import LiveExecutionEngine
    from src.engines.live.execution.position_tracker import LivePositionTracker
    from src.engines.live.logging.event_logger import LiveEventLogger
    from src.engines.live.order_tracker import OrderTracker
    from src.engines.live.reconciliation import BaseAssetLockRegistry, PeriodicReconciler
    from src.position_management.time_exits import TimeExitPolicy

logger = logging.getLogger(__name__)


class LiveStartupEngineState(Protocol):
    """Engine state the startup sequencer reads and mutates at call time."""

    is_running: bool
    enable_live_trading: bool
    resume_from_last_balance: bool
    current_balance: float
    max_position_size: float
    check_interval: int
    timeframe: str | None
    _active_symbol: str | None
    trading_session_id: int | None
    _recovered_inactive_session_id: int | None
    _pending_balance_correction: bool
    _pending_corrected_balance: float | None
    _orphan_sweep_cooldown: dict[str, float]
    _balance_lock: threading.Lock
    main_thread: threading.Thread | None
    strategy: Any
    exchange_interface: Any
    account_synchronizer: Any
    data_provider: DataProvider
    db_manager: DatabaseManager
    event_logger: LiveEventLogger
    live_execution_engine: LiveExecutionEngine
    live_position_tracker: LivePositionTracker
    order_tracker: OrderTracker | None
    time_exit_policy: TimeExitPolicy | None
    _base_asset_locks: BaseAssetLockRegistry
    _periodic_reconciler: PeriodicReconciler | None

    def _recover_existing_session(self) -> float | None: ...

    def _recover_active_positions(self) -> None: ...

    def _reconcile_positions_with_exchange(self) -> None: ...

    def _ensure_positions_registered_with_risk_manager(self) -> None: ...

    def _strategy_name(self) -> str: ...

    def _enter_close_only_mode(self) -> None: ...

    def _start_websocket_streams(self, symbol: str, timeframe: str) -> None: ...

    def _run_trading_loop(
        self, symbol: str, timeframe: str, max_steps: int | None = None
    ) -> None: ...

    def stop(self) -> None: ...

    def _exit_if_loop_crashed(self, exit_on_crash: bool) -> None: ...

    def _record_event(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str = ...,
        component: str | None = ...,
        error_code: str | None = ...,
        exc: BaseException | None = ...,
        alert: bool = ...,
    ) -> None: ...

    def _warn_if_no_alert_channel(self) -> None: ...


class LiveStartupSequencer:
    """Drives the engine's startup sequence against the engine backref (#486)."""

    def __init__(self, engine_state: LiveStartupEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    def run(
        self,
        symbol: str,
        timeframe: str = "1h",
        max_steps: int | None = None,
        exit_on_crash: bool = False,
    ) -> None:
        """Start the live trading engine.

        ``exit_on_crash`` makes an abnormal loop death exit the process non-zero
        so an orchestrator restarts it (#630). It defaults to False so start()
        stays a well-behaved library call for callers that read results after it
        returns (e.g. the migration baseline tool); the production runner opts in.
        """
        state = self._state
        if state.is_running:
            logger.warning("Trading engine is already running")
            return

        self.begin_session_runtime(symbol, timeframe)
        self.bootstrap_trading_session(symbol, timeframe)
        # Now the session exists: record a missing alert channel loudly so the
        # "operator alerts are not deliverable" blind spot lands in system_events
        # on every startup instead of being silent (P0 observability hardening).
        state._warn_if_no_alert_channel()
        self.carry_forward_open_positions()
        self.self_heal_terminal_positions()
        self.synchronize_account_on_start()

        # Set session ID on strategy for logging
        if hasattr(state.strategy, "session_id"):
            state.strategy.session_id = state.trading_session_id

        self.start_runtime_services(symbol, timeframe)
        self.run_main_loop_until_stopped(symbol, timeframe, max_steps)

        # After a clean stop this is a no-op; after an abnormal loop death it
        # exits the process non-zero (when opted in) so the orchestrator
        # restarts it (#630).
        state._exit_if_loop_crashed(exit_on_crash)

    def begin_session_runtime(self, symbol: str, timeframe: str) -> None:
        """Set runtime flags + logging context and emit the startup banner."""
        state = self._state
        state.is_running = True
        state._active_symbol = symbol
        state.timeframe = timeframe  # Store the trading timeframe
        # Set base logging context for this engine run
        set_context(
            component="live_engine",
            strategy=getattr(state.strategy, "__class__", type("_", (), {})).__name__,
            symbol=symbol,
            timeframe=timeframe,
        )
        log_engine_event(
            "engine_start",
            initial_balance=state.current_balance,
            max_position_size=state.max_position_size,
            check_interval=state.check_interval,
            mode="live" if state.enable_live_trading else "paper",
        )
        logger.info("🚀 Starting live trading for %s on %s timeframe", symbol, timeframe)
        logger.info("Initial balance: $%.2f", state.current_balance)
        logger.info("Max position size: %.1f%% of balance", state.max_position_size * 100)
        logger.info("Check interval: %ss", state.check_interval)

        if not state.enable_live_trading:
            logger.warning("⚠️  PAPER TRADING MODE - No real orders will be executed")

    def bootstrap_trading_session(self, symbol: str, timeframe: str) -> None:
        """Recover prior balance/positions and create + wire the trading session."""
        state = self._state
        # Try to recover from existing session first
        if state.resume_from_last_balance:
            recovered_balance = state._recover_existing_session()
            if recovered_balance is not None:
                # _recover_existing_session() already coerced this to a finite float
                # and rejected corrupt (non-finite) state. The float() here is a
                # cheap defensive invariant so current_balance never becomes a
                # Decimal — which would break downstream float arithmetic such as
                # _print_final_stats (CODE.md "Arithmetic & Financial Calculations").
                state.current_balance = float(recovered_balance)
                logger.info(
                    "💾 Recovered balance from previous session: $%.2f",
                    recovered_balance,
                )
                # Also recover active positions
                state._recover_active_positions()
            else:
                logger.info("🆕 No existing session found, starting fresh")

        # Create new trading session in database if none exists
        if state.trading_session_id is None:
            mode = TradeSource.LIVE if state.enable_live_trading else TradeSource.PAPER
            # Prepare time-exit session config for persistence
            tx_cfg = None
            if state.time_exit_policy:
                tx_cfg = {
                    "max_holding_hours": state.time_exit_policy.max_holding_hours,
                    "end_of_day_flat": state.time_exit_policy.end_of_day_flat,
                    "weekend_flat": state.time_exit_policy.weekend_flat,
                    "time_restrictions": {
                        "no_overnight": state.time_exit_policy.time_restrictions.no_overnight,
                        "no_weekend": state.time_exit_policy.time_restrictions.no_weekend,
                        "trading_hours_only": state.time_exit_policy.time_restrictions.trading_hours_only,
                    },
                }

            state.trading_session_id = state.db_manager.create_trading_session(
                strategy_name=state._strategy_name(),
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                initial_balance=state.current_balance,  # Use current balance (might be recovered)
                strategy_config=getattr(state.strategy, "config", {}),
                time_exit_config=tx_cfg,
                market_timezone=(
                    state.time_exit_policy.market_timezone if state.time_exit_policy else None
                ),
            )

            # Update context with session id
            update_context(session_id=state.trading_session_id)

            # Initialize balance tracking
            state.db_manager.update_balance(
                state.current_balance, "session_start", "system", state.trading_session_id
            )

            # Set session ID on strategy for logging
            if hasattr(state.strategy, "session_id"):
                state.strategy.session_id = state.trading_session_id

            # Wire session_id and strategy_name to execution engine for order journaling
            state.live_execution_engine.session_id = state.trading_session_id
            state.live_execution_engine.strategy_name = state._strategy_name()

            # Wire the event logger so snapshot/daily-P&L logging is session
            # scoped; on a clean restart also point day-start recovery at the
            # prior session, where today's earlier snapshots live (#766).
            state.event_logger.set_session_id(state.trading_session_id)
            state.event_logger.set_recovery_session_id(state._recovered_inactive_session_id)

    def carry_forward_open_positions(self) -> None:
        """Carry OPEN positions forward from a recovered inactive session (#668)."""
        state = self._state
        # Carry OPEN positions forward on a clean restart (#668). The inactive
        # session recovered above (balance only) still owns any OPEN position via
        # Position.session_id, so _recover_active_positions() at line ~1281 saw a
        # None session id and loaded nothing — the position would be orphaned.
        # Re-point those positions onto the new session, then reload them into the
        # live tracker. Ordering: reassign → recover-into-tracker (which self-heals
        # first) → the heal + exchange reconciliation below re-verify the position
        # and its server-side stop-loss against the exchange.
        if (
            state._recovered_inactive_session_id is not None
            and state.trading_session_id is not None
        ):
            try:
                moved_ids = state.db_manager.reassign_open_positions_to_session(
                    old_session_id=state._recovered_inactive_session_id,
                    new_session_id=state.trading_session_id,
                    symbol=state._active_symbol,
                    strategy_name=state._strategy_name(),
                )
                if moved_ids:
                    logger.info(
                        "🔁 Carried %d OPEN position(s) forward from inactive session "
                        "#%s into new session #%s; reloading into tracker",
                        len(moved_ids),
                        state._recovered_inactive_session_id,
                        state.trading_session_id,
                    )
                    # Reload now that the rows belong to the new session. Safe and
                    # idempotent if it already ran (empty session ⇒ no-op).
                    state._recover_active_positions()
            except Exception as reassign_err:
                # A failure here must not abort startup, but it is capital-critical
                # (an OPEN position stays orphaned), so log loudly for alerting.
                logger.critical(
                    "Failed to carry OPEN positions forward from inactive session "
                    "#%s into session #%s (positions may be orphaned — MANUAL "
                    "RECONCILIATION REQUIRED): %s",
                    state._recovered_inactive_session_id,
                    state.trading_session_id,
                    reassign_err,
                    exc_info=True,
                )
            finally:
                # Clear so a later start()/stop()/start() re-entry cannot re-trigger a
                # stale-session reassign (#668, P3).
                state._recovered_inactive_session_id = None

    def self_heal_terminal_positions(self) -> None:
        """Close any OPEN position in this session that already has a terminal Trade (#657)."""
        state = self._state
        # Startup self-heal (#657): close any OPEN position in this session that
        # already has a terminal Trade. Deliberately NOT gated behind
        # enable_live_trading/exchange — the whole bug was that closing was
        # paper-blind, so this must run in paper mode too. Pure DB reconciliation
        # (no exchange calls), idempotent, and complements the atomic status flip
        # now performed inside log_trade. Placed before account sync so the books
        # are consistent before any exchange reconciliation reads them.
        if state.trading_session_id is not None:
            try:
                healed = state.db_manager.heal_positions_with_terminal_trades(
                    state.trading_session_id
                )
                if healed:
                    logger.info(
                        "🩹 Startup self-heal closed %d stale-OPEN position(s) with "
                        "terminal trades (session #%s)",
                        healed,
                        state.trading_session_id,
                    )
            except Exception as heal_err:
                logger.warning("Startup position self-heal failed (continuing): %s", heal_err)

    def synchronize_account_on_start(self) -> None:
        """Sync balance/positions with the exchange and persist any balance correction."""
        state = self._state
        # Perform account synchronization if available
        state._pending_balance_correction = False
        state._pending_corrected_balance = None
        if state.account_synchronizer and state.enable_live_trading:
            try:
                logger.info("🔄 Performing initial account synchronization...")
                sync_result = state.account_synchronizer.sync_account_data(
                    force=True, symbol=state._active_symbol
                )
                if sync_result.success:
                    logger.info("✅ Account synchronization completed")
                    # Update session ID for synchronizer
                    if state.trading_session_id:
                        state.account_synchronizer.session_id = state.trading_session_id
                    # Check if balance was corrected
                    balance_sync = sync_result.data.get("balance_sync", {})
                    if balance_sync.get("corrected", False):
                        previous_balance = balance_sync.get("old_balance", state.current_balance)
                        corrected_balance = balance_sync.get("new_balance", state.current_balance)
                        # Atomic balance update with lock to prevent race conditions
                        with state._balance_lock:
                            state.current_balance = corrected_balance
                            state._pending_balance_correction = True
                            state._pending_corrected_balance = corrected_balance
                        logger.info(
                            "💰 Balance corrected from exchange: $%.2f",
                            corrected_balance,
                        )
                        # A silent balance overwrite masked a real capital-erosion
                        # incident before — make every correction auditable.
                        state._record_event(
                            EventType.WARNING,
                            (
                                "Balance overwritten from exchange: "
                                f"{previous_balance} -> {corrected_balance}"
                            ),
                            severity="warning",
                            component="balance",
                            error_code="BALANCE_OVERWRITE",
                        )
                else:
                    logger.warning("⚠️ Account synchronization failed: %s", sync_result.message)

                # Reconcile positions with exchange (detect offline stop-loss triggers)
                state._reconcile_positions_with_exchange()

                # Reconciliation paths (e.g. PositionReconciler._reconcile_filled_entry)
                # may create LivePositions via track_recovered_position without
                # registering them with risk_manager. The DB-recovery path in
                # _recover_active_positions does register; the reconciler path
                # currently does not. Sweep the tracker after reconciliation so
                # every tracked position is known to the risk manager — this
                # restores the parity invariant (also enforced on every
                # backtest entry) that risk_manager has visibility into all
                # active positions for per-symbol caps and correlation gating.
                state._ensure_positions_registered_with_risk_manager()

            except Exception as e:
                logger.error("❌ Account synchronization error: %s", e, exc_info=True)

        # If a balance correction was pending, log it now (outside session creation conditional)
        # Use lock to ensure atomic check and update
        with state._balance_lock:
            if (
                getattr(state, "_pending_balance_correction", False)
                and state.trading_session_id is not None
            ):
                corrected_balance = state._pending_corrected_balance
                state.db_manager.update_balance(
                    corrected_balance, "account_sync", "system", state.trading_session_id
                )
                state._pending_balance_correction = False
                state._pending_corrected_balance = None
                logger.info("💰 Balance corrected in database: $%.2f", corrected_balance)
            elif getattr(state, "_pending_balance_correction", False):
                # Balance correction was pending but no session ID available
                logger.warning(
                    "⚠️ Balance correction pending but no trading session ID available - skipping database update"
                )
                state._pending_balance_correction = False
                state._pending_corrected_balance = None

    def start_runtime_services(self, symbol: str, timeframe: str) -> None:
        """Start the order tracker, periodic reconciler, and WebSocket streams."""
        state = self._state
        # Start order tracker for monitoring order fills (live trading only)
        if state.order_tracker and state.enable_live_trading:
            state.order_tracker.start()
            logger.info("📡 Order tracker started")

        # Start periodic reconciler (live trading only, not paper mode)
        if state.enable_live_trading and state.exchange_interface and state.trading_session_id:
            try:
                from src.engines.live.reconciliation import PeriodicReconciler

                use_margin = getattr(state.exchange_interface, "is_margin_mode", False)
                state._periodic_reconciler = PeriodicReconciler(
                    exchange_interface=state.exchange_interface,
                    position_tracker=state.live_position_tracker,
                    db_manager=state.db_manager,
                    session_id=state.trading_session_id,
                    on_critical=state._enter_close_only_mode,
                    on_event=state._record_event,
                    use_margin=use_margin,
                    symbols=[state._active_symbol] if state._active_symbol else [],
                    sweep_cooldown=state._orphan_sweep_cooldown,
                    lock_registry=state._base_asset_locks,
                    data_provider=state.data_provider,
                )
                state._periodic_reconciler.start()
                logger.info("🔄 Periodic reconciler started")
            except Exception as e:
                logger.warning("Failed to start periodic reconciler: %s", e)
                # A silently-disabled reconciler is exactly the kind of failure
                # that ran invisible for months — surface it in system_events.
                state._record_event(
                    EventType.ERROR,
                    f"Periodic reconciler failed to start: {e}",
                    severity="error",
                    component="reconciler",
                    error_code="RECONCILER_START_FAILED",
                    exc=e,
                )

        # Try to start WebSocket streams for reduced API weight
        state._start_websocket_streams(symbol, timeframe)

    def run_main_loop_until_stopped(
        self, symbol: str, timeframe: str, max_steps: int | None
    ) -> None:
        """Launch the trading-loop thread and block until it stops, then tear down."""
        state = self._state
        # Start main trading loop in separate thread
        state.main_thread = threading.Thread(
            target=state._run_trading_loop, args=(symbol, timeframe, max_steps)
        )
        state.main_thread.daemon = True
        state.main_thread.start()

        try:
            # Keep main thread alive
            while state.is_running and state.main_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            state.stop()
