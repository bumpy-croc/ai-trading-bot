"""Session and position recovery for the live trading engine startup sequence.

Crash/clean-restart balance recovery, persisted-position reload (with
self-healing of stale-OPEN rows), risk-manager re-registration, and the
startup exchange reconciliation (PositionReconciler with legacy SL-based
fallback). Moved verbatim from ``LiveTradingEngine`` (#486) so the engine
orchestrates startup while this module owns the recovery mechanics.

Thread-safety / lock ownership: the recoverer holds no locks and owns no
mutable state of its own. It runs on the startup path before the trading loop
starts; engine attributes it mutates (``trading_session_id``,
``current_balance``, ``_close_only_mode``, ``_recovered_inactive_session_id``)
are read/written through the engine exactly as the original inline code did,
and position mutations go through ``LivePositionTracker``'s internal lock.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from src.database.models import EventType
from src.engines.live.execution.position_tracker import LivePosition as Position
from src.engines.live.execution.position_tracker import LivePositionTracker
from src.engines.live.margin_interest_tracker import MarginInterestTracker
from src.engines.live.order_tracker import OrderTracker
from src.engines.live.trade_close_accounting import (
    _close_entry_fee_usd,
    _close_position_portion,
    _closed_base_quantity,
)
from src.engines.shared.models import BaseTrade as Trade
from src.engines.shared.models import PositionSide
from src.performance.metrics import Side, pnl_percent

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    from src.engines.live.execution.execution_engine import LiveExecutionEngine
    from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
    from src.engines.live.reconciliation import BaseAssetLockRegistry
    from src.performance.tracker import PerformanceTracker
    from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class RecoveryEngineState(Protocol):
    """Live engine state the recoverer reads and mutates at call time.

    Recovery is the most engine-entangled startup subsystem; this protocol
    makes the full coupling surface explicit. Attributes are accessed
    dynamically (not captured at construction) because session id, balance,
    and close-only mode are assigned during ``start()``.
    """

    # Read-only collaborators
    db_manager: DatabaseManager
    live_position_tracker: LivePositionTracker
    live_execution_engine: LiveExecutionEngine
    stop_loss_manager: LiveStopLossManager
    performance_tracker: PerformanceTracker
    risk_manager: RiskManager
    order_tracker: OrderTracker | None
    exchange_interface: Any
    enable_live_trading: bool
    log_trades: bool
    max_position_size: float
    completed_trades: list[Trade]
    _active_symbol: str | None
    _orphan_sweep_cooldown: dict[str, float]
    _base_asset_locks: BaseAssetLockRegistry

    # Mutated during recovery
    trading_session_id: int | None
    current_balance: float
    _close_only_mode: bool
    _recovered_inactive_session_id: int | None

    def _strategy_name(self) -> str: ...

    def _enter_close_only_mode(self) -> None: ...

    def _log_trade(self, trade: Trade) -> None: ...

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


class LiveSessionRecoverer:
    """Startup recovery: session balance, persisted positions, reconciliation."""

    def __init__(self, engine_state: RecoveryEngineState) -> None:
        """Bind to the engine's live state (see protocol for the full surface)."""
        self._state = engine_state

    def recover_existing_session(self) -> float | None:
        """Try to recover balance from an existing session.

        Prefers an active session (crash recovery); otherwise falls back to the
        most recent matching session within 7 days (clean-restart path). When
        there is genuinely no recent session to recover (new symbol/strategy,
        fresh DB, or older than the window), returns None and the engine starts
        fresh. Recovery runs unconditionally in every environment so staging
        mirrors production — there is intentionally no bypass flag. (A
        recovery-bypass env var silently orphaned open positions and reset to a
        phantom balance; removed — see #668.)
        """
        state = self._state
        try:
            # Prefer an active session (crash recovery path).
            session_id = state.db_manager.get_active_session_id()
            source = "active"

            # Fallback: most recent matching session within 7 days (clean-restart path).
            if session_id is None:
                strategy = state._strategy_name()
                session_id = state.db_manager.get_last_session_id(
                    within_hours=168,  # 7 days — covers long-running paper trading sessions
                    strategy_name=strategy,
                    symbol=state._active_symbol,
                )
                source = "recent inactive"

            if session_id is None:
                logger.info("🆕 No recent session found, starting fresh")
                return None

            logger.info("🔍 Found %s session #%s", source, session_id)

            # Clean restart (inactive session): remember it so start() can carry its
            # OPEN positions forward into the new session — INDEPENDENT of whether a
            # positive balance was recovered. A fully-liquidated session (balance 0)
            # can still hold an OPEN position; gating this on balance > 0 would
            # re-orphan it (#668, P2). The active/crash path reuses the session
            # directly below and never needs this.
            if source != "active":
                state._recovered_inactive_session_id = session_id

            recovered_balance = state.db_manager.recover_last_balance(session_id)
            # Sanitize BEFORE the positivity filter below. recover_last_balance()
            # can return a Decimal (Numeric column), and corrupt state can be
            # non-finite — float(Decimal('Infinity'))/NaN -> inf/nan. The `> 0`
            # check would otherwise silently drop -inf/NaN to None (engine starts
            # on the default balance and may trade) or raise on Decimal('NaN').
            # Coerce to a float invariant and fail fast on non-finite: a corrupt
            # persisted balance must halt startup, never feed position sizing
            # (CODE.md "Arithmetic & Financial Calculations").
            if recovered_balance is not None:
                recovered_balance = float(recovered_balance)
                if not math.isfinite(recovered_balance):
                    raise ValueError(
                        f"Recovered balance is not finite ({recovered_balance!r}); "
                        "refusing to start on corrupt persisted state."
                    )
            if recovered_balance and recovered_balance > 0:
                # Crash recovery (active session): reuse the existing session ID so
                # trades stay attributed to the same session row. Clean restarts create
                # a new session below; their OPEN positions are carried forward via
                # _recovered_inactive_session_id (set above).
                if source == "active":
                    state.trading_session_id = session_id
                    # Register the reused session with the DB manager. create_trading_session
                    # sets _current_session_id for NEW sessions, but this active-recovery path
                    # reuses an existing one — without this, every session-scoped write that
                    # falls back to _current_session_id (balance updates, etc.) fails with
                    # "No active trading session" on the first trade after recovery (#41).
                    state.db_manager.set_current_session(session_id)
                    # Wire session context to execution engine so journaling works
                    state.live_execution_engine.session_id = session_id
                    state.live_execution_engine.strategy_name = state._strategy_name()
                logger.info(
                    "💾 Recovered balance $%.2f from %s session #%s",
                    recovered_balance,
                    source,
                    session_id,
                )
                return recovered_balance

            logger.warning("⚠️  Session #%s found but no balance to recover", session_id)
            return None
        except ValueError:
            # Corrupt-balance invariant violation — must not be swallowed by the
            # broad handler below (that would silently fall back to the default
            # balance). Propagate so startup fails fast.
            raise
        except Exception as e:
            logger.error("❌ Error recovering session: %s", e, exc_info=True)
            return None

    def ensure_positions_registered_with_risk_manager(self) -> None:
        """Register every tracked position with the risk manager.

        Idempotent. Re-registering a known position is a no-op for risk
        managers that key on (symbol, side); for the few that count entries,
        registering twice still leaves the position visible — strictly
        better than the recovered-but-invisible state we are guarding
        against.

        Parity rationale:
        - Backtest registers every position at entry
          (src/engines/backtest/execution/entry_handler.py:407-421).
        - Live's DB-recovery path registers
          (src/engines/live/trading_engine.py:_recover_active_positions).
        - Live's reconciler path (PositionReconciler._reconcile_filled_entry)
          can also create positions via track_recovered_position but does
          not register. This sweep closes that gap so per-symbol caps and
          correlation gating see all active positions, matching the
          invariant backtest assumes always holds.
        """
        state = self._state
        if state.risk_manager is None:
            return
        try:
            positions_snapshot = state.live_position_tracker.positions
        except Exception as e:
            logger.warning("Failed to snapshot positions for risk-manager sync: %s", e)
            return

        for position in positions_snapshot.values():
            try:
                # Use current_size (post-partial-exit) — passing the original
                # ``size`` would silently re-inflate risk_manager.daily_risk_used
                # on every re-registration, undoing the prior
                # adjust_position_after_partial_exit. CODE.md "Position Fields"
                # rules: ``current_size`` is the source of truth for capital
                # currently deployed.
                effective_size = (
                    float(position.current_size)
                    if position.current_size is not None
                    else float(position.size)
                )
                if effective_size <= 0:
                    # 100% partial-exit drained the position but the close
                    # ack has not popped it from the tracker yet. Calling
                    # ``update_position(size=0.0)`` would fail the size>0
                    # validator and leave ``daily_risk_used`` inflated at
                    # the original allocation. Drain the slot via
                    # ``close_position`` so the next entry sees the right
                    # remaining budget, then skip re-registration.
                    try:
                        state.risk_manager.close_position(position.symbol)
                    except (KeyError, ValueError, AttributeError) as drain_err:
                        logger.debug(
                            "Risk manager close_position drain skipped for %s: %s",
                            position.symbol,
                            drain_err,
                        )
                    continue
                state.risk_manager.update_position(
                    symbol=position.symbol,
                    # __post_init__ guarantees side is a PositionSide enum.
                    side=cast(PositionSide, position.side).value,
                    size=effective_size,
                    entry_price=position.entry_price,
                )
            except Exception as e:
                logger.warning(
                    "Failed to register recovered position %s with risk manager: %s",
                    position.symbol,
                    e,
                )

    def recover_active_positions(self) -> None:
        """Recover active positions from database"""
        state = self._state
        try:
            if not state.trading_session_id:
                return

            # Self-heal BEFORE reloading (works in paper too, no exchange calls):
            # close any OPEN position in this session that already has a terminal
            # Trade. Such rows are the #657 footgun — historically a closed Trade
            # was logged without flipping positions.status, so the stale-OPEN row
            # gets reloaded here with its old stop_loss and re-closed, producing a
            # phantom duplicate trade. Healing first means get_active_positions
            # below cannot return them. Belt-and-suspenders alongside the atomic
            # status flip now done in log_trade.
            try:
                healed = state.db_manager.heal_positions_with_terminal_trades(
                    state.trading_session_id
                )
                if healed:
                    logger.info(
                        "🩹 Self-healed %d stale-OPEN position(s) with terminal trades "
                        "before recovery (session #%s)",
                        healed,
                        state.trading_session_id,
                    )
            except Exception as heal_err:
                # Never let a heal failure block recovery; the atomic log_trade
                # flip remains the primary defense going forward.
                logger.warning(
                    "Position self-heal failed before recovery (continuing): %s",
                    heal_err,
                )

            # Get active positions from database
            db_positions = state.db_manager.get_active_positions(state.trading_session_id)

            if not db_positions:
                logger.info("📊 No active positions to recover")
                return

            logger.info("🔄 Recovering %s active positions...", len(db_positions))

            for pos_data in db_positions:
                # Convert database position to Position object
                # Handle both uppercase and lowercase side values from database
                side_value = pos_data["side"]
                if isinstance(side_value, str):
                    side_value = side_value.lower()

                stored_entry_balance = pos_data.get("entry_balance")
                try:
                    entry_balance = (
                        float(stored_entry_balance)
                        if stored_entry_balance is not None
                        else float(state.current_balance)
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Recovered position %s has invalid entry balance %s; falling back to current balance",
                        pos_data.get("symbol"),
                        stored_entry_balance,
                    )
                    entry_balance = float(state.current_balance)

                # Resolve the tracker key: prefer entry_order_id (exchange),
                # fall back to database ID string for backward compat
                entry_order_id = pos_data.get("entry_order_id")
                tracker_key = entry_order_id or str(pos_data["id"])

                position = Position(
                    symbol=pos_data["symbol"],
                    side=PositionSide(side_value),
                    size=pos_data["size"],
                    entry_price=pos_data["entry_price"],
                    entry_time=pos_data["entry_time"],
                    entry_balance=entry_balance,
                    stop_loss=pos_data.get("stop_loss"),
                    take_profit=pos_data.get("take_profit"),
                    unrealized_pnl=float(pos_data.get("unrealized_pnl", 0.0) or 0.0),
                    unrealized_pnl_percent=float(
                        pos_data.get("unrealized_pnl_percent", 0.0) or 0.0
                    ),
                    quantity=pos_data.get("quantity"),
                    # Hydrate partial-operation state so a position partially exited
                    # before a restart closes at its REMAINING size (and logs the
                    # remaining base quantity), not the full original. Without this,
                    # current_size/original_size fall back to size and the close
                    # over-reports size, pnl fraction, and trades.quantity.
                    original_size=pos_data.get("original_size"),
                    current_size=pos_data.get("current_size"),
                    partial_exits_taken=pos_data.get("partial_exits_taken", 0) or 0,
                    scale_ins_taken=pos_data.get("scale_ins_taken", 0) or 0,
                    last_partial_exit_price=pos_data.get("last_partial_exit_price"),
                    last_scale_in_price=pos_data.get("last_scale_in_price"),
                    order_id=tracker_key,  # Backward compat: used as _positions dict key
                    tracker_key=tracker_key,
                    exchange_order_id=entry_order_id,
                    client_order_id=pos_data.get("client_order_id"),
                    db_position_id=pos_data["id"],
                    stop_loss_order_id=pos_data.get("stop_loss_order_id"),
                )

                # Validate recovered entry_price before tracking. Positions
                # with invalid entry_price cannot be closed properly and would
                # become orphaned in the tracker.
                if position.entry_price <= 0 or not math.isfinite(position.entry_price):
                    logger.critical(
                        "SKIPPING recovery of position %s (%s): invalid entry_price %.8f. "
                        "MANUAL RECONCILIATION REQUIRED.",
                        position.symbol,
                        position.order_id,
                        position.entry_price,
                    )
                    continue

                if position.order_id:
                    state.live_position_tracker.track_recovered_position(
                        position, db_id=pos_data.get("id")
                    )

                # Register recovered stop-loss order with OrderTracker for monitoring
                if position.stop_loss_order_id and state.order_tracker:
                    state.order_tracker.track_order(position.stop_loss_order_id, position.symbol)
                    logger.info(
                        f"📡 Recovered and tracking stop-loss order {position.stop_loss_order_id} "
                        f"for position {position.symbol}"
                    )

                # Update risk manager tracking for recovered positions
                if state.risk_manager:
                    try:
                        state.risk_manager.update_position(
                            symbol=position.symbol,
                            # __post_init__ guarantees side is a PositionSide enum.
                            side=cast(PositionSide, position.side).value,
                            size=position.size,
                            entry_price=position.entry_price,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to update risk manager for recovered position %s: %s",
                            position.symbol,
                            e,
                        )

                logger.info(
                    "✅ Recovered position: %s %s @ $%.2f",
                    pos_data["symbol"],
                    pos_data["side"],
                    pos_data["entry_price"],
                )

            logger.info("🎯 Successfully recovered %s positions", len(db_positions))

        except Exception as e:
            logger.error("❌ Error recovering positions: %s", e, exc_info=True)

    def reconcile_positions_with_exchange(self) -> None:
        """
        Reconcile local positions with exchange state on startup.

        Delegates to PositionReconciler for comprehensive order-based verification.
        Falls back to legacy SL-based reconciliation if reconciler unavailable.
        """
        state = self._state
        if not state.exchange_interface or not state.enable_live_trading:
            return

        positions_snapshot = state.live_position_tracker.positions

        # Run PositionReconciler regardless of position count — resolve_pending_orders
        # must execute even when no positions were recovered (e.g. entry submitted but
        # never persisted as a position before crash).
        if state.trading_session_id:
            try:
                from src.engines.live.reconciliation import (
                    PositionReconciler,
                    Severity,
                    run_orphaned_borrow_sweep,
                )

                use_margin = getattr(state.exchange_interface, "is_margin_mode", False)
                reconciler = PositionReconciler(
                    exchange_interface=state.exchange_interface,
                    position_tracker=state.live_position_tracker,
                    db_manager=state.db_manager,
                    session_id=state.trading_session_id,
                    max_position_size=state.max_position_size,
                    use_margin=use_margin,
                    fee_rate=state.live_execution_engine.fee_rate,
                )

                if not positions_snapshot:
                    logger.info("📊 No local positions to reconcile — checking pending orders")
                    results = reconciler.resolve_pending_orders()

                    # After pending orders are resolved (so a fill that becomes a
                    # position is adopted first), sweep any orphaned margin borrow.
                    # No-op unless margin + flag enabled; safe when flat.
                    if use_margin and state._active_symbol:
                        run_orphaned_borrow_sweep(
                            exchange=state.exchange_interface,
                            position_tracker=state.live_position_tracker,
                            db_manager=state.db_manager,
                            session_id=state.trading_session_id,
                            use_margin=use_margin,
                            symbols=[state._active_symbol],
                            cooldown_state=state._orphan_sweep_cooldown,
                            lock_registry=state._base_asset_locks,
                        )

                    # Process results even with no positions — a filled entry
                    # order may create a position, and critical issues must
                    # still trigger close-only mode.
                    critical_count = sum(1 for r in results if r.severity == Severity.CRITICAL)
                    if critical_count > 0:
                        logger.critical(
                            "🚨 %d CRITICAL reconciliation issues — entering close-only mode",
                            critical_count,
                        )
                        state._close_only_mode = True

                    for r in results:
                        if r.status == "corrected" and r.severity >= Severity.HIGH:
                            for correction in r.corrections:
                                logger.warning(
                                    "⚠️ Auto-corrected %s #%s: %s",
                                    r.entity_type,
                                    r.entity_id,
                                    correction.reason,
                                )

                    if results:
                        corrections = sum(len(r.corrections) for r in results)
                        logger.info(
                            "✅ Pending order resolution complete: %d results, %d corrections, %d critical",
                            len(results),
                            corrections,
                            critical_count,
                        )
                    return

                logger.info("🔄 Reconciling %s positions with exchange...", len(positions_snapshot))
                results = reconciler.reconcile_startup(positions_snapshot)

                # Check for critical issues
                critical_count = sum(1 for r in results if r.severity == Severity.CRITICAL)
                if critical_count > 0:
                    logger.critical(
                        "🚨 %d CRITICAL reconciliation issues — entering close-only mode",
                        critical_count,
                    )
                    # Route through the guarded helper so the CLOSE_ONLY event is
                    # emitted on this startup-critical path too, not just runtime.
                    state._enter_close_only_mode()

                # Log HIGH severity auto-corrections (cancelled entries, SL fills)
                for r in results:
                    if r.status == "corrected" and r.severity >= Severity.HIGH:
                        for correction in r.corrections:
                            logger.warning(
                                "⚠️ Auto-corrected %s #%s: %s",
                                r.entity_type,
                                r.entity_id,
                                correction.reason,
                            )

                corrections = sum(len(r.corrections) for r in results)
                logger.info(
                    "✅ Reconciliation complete: %d results, %d corrections, %d critical",
                    len(results),
                    corrections,
                    critical_count,
                )
                return
            except Exception as e:
                logger.warning(
                    "PositionReconciler failed, falling back to legacy reconciliation: %s", e
                )
                state._record_event(
                    EventType.ERROR,
                    f"PositionReconciler failed, falling back to legacy reconciliation: {e}",
                    severity="error",
                    component="reconciler",
                    error_code="RECONCILER_FALLBACK",
                    exc=e,
                )

        # Legacy fallback: SL-based reconciliation (requires positions)
        if not positions_snapshot:
            logger.info("📊 No local positions to reconcile")
            return

        try:
            positions_to_close = state.stop_loss_manager.find_offline_filled_stops(
                positions_snapshot
            )

            # Close positions that were stopped out
            for position, exit_price in positions_to_close:
                logger.info(
                    "🔄 Marking position %s as closed (stop-loss triggered offline)",
                    position.symbol,
                )
                # Update balance based on stop-loss exit
                if exit_price:
                    fraction = (
                        position.current_size
                        if position.current_size is not None
                        else position.size
                    )
                    # Guard against division by zero (pnl_percent handles this but we log)
                    if position.entry_price <= 0:
                        logger.error(
                            "Invalid entry_price %s for position %s - skipping reconciliation",
                            position.entry_price,
                            position.symbol,
                        )
                        continue
                    if exit_price <= 0 or not math.isfinite(exit_price):
                        logger.error(
                            f"Invalid exit_price {exit_price} for position "
                            f"{position.symbol} - skipping reconciliation"
                        )
                        continue

                    # Use shared pnl_percent for parity with backtest engine
                    side_enum = Side.LONG if position.side == PositionSide.LONG else Side.SHORT
                    pnl_pct_sized = pnl_percent(
                        position.entry_price, exit_price, side_enum, fraction
                    )

                    # Use entry_balance for PnL calculation to maintain backtest-live parity
                    basis_balance = (
                        float(position.entry_balance)
                        if position.entry_balance is not None and position.entry_balance > 0
                        else state.current_balance
                    )
                    # Calculate exit fee for filled offline stop-loss
                    # Slippage is zero for filled orders - slippage already occurred on exchange
                    # and is reflected in the fill price. Matches execute_filled_exit behavior.
                    exit_position_notional = (
                        basis_balance * fraction * (exit_price / position.entry_price)
                    )
                    exit_fee = state.live_execution_engine.calculate_exit_fee(
                        exit_position_notional
                    )
                    exit_slippage_cost = 0.0  # Slippage already in fill price
                    # Calculate GROSS P&L for Trade.pnl (parity with backtest engine)
                    # and NET P&L for balance updates
                    gross_pnl = pnl_pct_sized * basis_balance
                    realized_pnl = gross_pnl - exit_fee  # Net P&L for balance update

                    # Deduct margin interest for short positions closed offline
                    offline_interest_cost = 0.0
                    if (
                        getattr(state.exchange_interface, "is_margin_mode", False)
                        and position.side == PositionSide.SHORT
                    ):
                        try:
                            from src.engines.live.reconciliation import PositionReconciler

                            tracker = MarginInterestTracker(state.exchange_interface)
                            base_asset = PositionReconciler._extract_base_asset(position.symbol)
                            interest_base = tracker.get_position_interest_cost(
                                base_asset, position.entry_time
                            )
                            offline_interest_cost = interest_base * exit_price
                            if offline_interest_cost > 0:
                                realized_pnl -= offline_interest_cost
                                logger.info(
                                    "Deducted margin interest $%.4f from offline SL PnL for %s",
                                    offline_interest_cost,
                                    position.symbol,
                                )
                        except Exception as e:
                            logger.warning(
                                "Failed to query margin interest for offline SL %s: %s",
                                position.symbol,
                                e,
                            )

                    # Atomic balance update for offline stop-loss reconciliation
                    if state.trading_session_id is not None:
                        try:
                            with state.db_manager.atomic_balance_update(
                                balance_change=realized_pnl,
                                reason=f"offline_stop_loss_{position.symbol}",
                                updated_by="live_engine_reconciliation",
                                correlation_id=position.order_id,
                            ) as balance_result:
                                state.current_balance = balance_result["new_balance"]
                                logger.info(
                                    f"💰 Adjusted balance for offline stop-loss: ${realized_pnl:+,.2f} "
                                    f"(fee: ${exit_fee:.2f}) -> ${state.current_balance:,.2f}"
                                )
                        except Exception as balance_err:
                            logger.error(
                                "Failed to update balance for offline stop-loss %s: %s. Skipping reconciliation.",
                                position.symbol,
                                balance_err,
                            )
                            continue
                    else:
                        # No trading session - update balance directly
                        state.current_balance += realized_pnl
                        logger.info(
                            f"💰 Adjusted balance for offline stop-loss: ${realized_pnl:+,.2f} "
                            f"(fee: ${exit_fee:.2f}) -> ${state.current_balance:,.2f}"
                        )
                    # Store GROSS P&L in Trade.pnl for parity with backtest engine
                    # Fees are tracked separately via performance_tracker.record_trade()
                    trade = Trade(
                        symbol=position.symbol,
                        side=position.side,
                        size=fraction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        entry_time=position.entry_time,
                        exit_time=datetime.now(UTC),
                        pnl=gross_pnl,
                        pnl_percent=pnl_pct_sized,
                        exit_reason="stop_loss_offline",
                    )
                    state.performance_tracker.record_trade(
                        trade=trade,
                        fee=exit_fee + offline_interest_cost,
                        slippage=exit_slippage_cost,
                    )
                    state.completed_trades.append(trade)
                    if state.log_trades:
                        state._log_trade(trade)

                    # Persist trade to DB with margin interest cost
                    if state.trading_session_id is not None:
                        state.db_manager.log_trade(
                            symbol=position.symbol,
                            # __post_init__ guarantees side is a PositionSide enum.
                            side=cast(PositionSide, position.side).value,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            size=fraction,
                            pnl=gross_pnl,
                            strategy_name=state._strategy_name(),
                            exit_reason="stop_loss_offline",
                            entry_time=position.entry_time,
                            exit_time=datetime.now(UTC),
                            session_id=state.trading_session_id,
                            # Round-trip fee in USD: entry_fee (booked at open, or
                            # reconstructed from the fee model for restart-recovered
                            # positions; entry leg scaled to the closed portion) plus this
                            # offline exit_fee. Same units as account_balances, where
                            # realized_pnl above is already net of exit_fee.
                            commission=(
                                _close_entry_fee_usd(position, state.live_execution_engine)
                                * _close_position_portion(position)
                                + exit_fee
                            ),
                            quantity=_closed_base_quantity(position),
                            margin_interest_cost=offline_interest_cost,
                        )

                # Stop tracking the SL order to prevent memory leak
                if position.stop_loss_order_id and state.order_tracker:
                    state.order_tracker.stop_tracking(position.stop_loss_order_id)

                # Remove from local positions
                if position.order_id:
                    state.live_position_tracker.remove_position(position.order_id)
                if state.risk_manager:
                    try:
                        state.risk_manager.close_position(position.symbol)
                    except Exception as e:
                        logger.warning(
                            "Failed to update risk manager for reconciled position %s: %s",
                            position.symbol,
                            e,
                        )

                # Close in database
                db_ids = state.live_position_tracker.position_db_ids
                # Tracked positions always carry a non-None order_id.
                position_db_id = db_ids.get(cast(str, position.order_id))
                if position_db_id:
                    state.db_manager.close_position(position_id=position_db_id)

            if positions_to_close:
                logger.info(
                    f"🔄 Reconciliation complete: {len(positions_to_close)} positions "
                    "closed (stopped out while offline)"
                )
            else:
                logger.info("✅ All positions verified - no offline closures detected")

        except Exception as e:
            logger.error("❌ Error reconciling positions with exchange: %s", e, exc_info=True)
