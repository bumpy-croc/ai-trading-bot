"""Live exit-execution coordinator (#486).

Owns the live engine's exit pipeline: deciding whether to close open positions
(``check_exit_conditions``) and executing the close under the position's
base-asset lock (``execute_exit`` -> ``execute_exit_locked``). This is a
real-money path, so the extraction is behaviour-preserving: the methods are the
verbatim engine methods with ``self.`` rewritten to ``state.`` against an engine
backref.

Locking & ordering (unchanged from the engine):
- ``execute_exit`` serialises the whole close on the symbol's base-asset lock
  (``state._base_asset_locks``, #703). The lock is re-entrant: an entry whose
  stop-loss placement failed routes its emergency close through the engine's
  ``_execute_exit`` wrapper on the same thread and re-acquires the lock without
  deadlock.
- All engine state (balance, trackers, exit handler, session id) lives on the
  engine and is mutated through the ``state`` backref exactly as before; the
  coordinator holds no state of its own beyond that reference.
- ``check_exit_conditions`` invokes the close through ``state._execute_exit``
  (the engine wrapper) rather than ``self.execute_exit`` so engine-level test
  mocks (``engine._execute_exit = Mock()``) still intercept it, preserving the
  existing exit-condition tests. ``execute_exit`` -> ``execute_exit_locked`` is a
  coordinator-internal call (no engine wrapper indirection needed).
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import pandas as pd

from src.config.constants import BORROW_DUST_EPSILON
from src.engines.live.margin_interest_tracker import MarginInterestTracker
from src.engines.live.trade_close_accounting import (
    _close_entry_fee_usd,
    _close_position_portion,
    _closed_base_quantity,
)
from src.engines.shared.models import BaseTrade as Trade
from src.engines.shared.models import PositionSide
from src.engines.shared.validation import is_same_bar_entry
from src.infrastructure.logging.events import log_order_event
from src.performance.metrics import Side, pnl_percent

if TYPE_CHECKING:
    from src.data_providers.data_provider import DataProvider
    from src.database.manager import DatabaseManager
    from src.engines.live.execution.execution_engine import LiveExecutionEngine
    from src.engines.live.execution.exit_handler import LiveExitHandler
    from src.engines.live.execution.position_tracker import LivePosition as Position
    from src.engines.live.execution.position_tracker import LivePositionTracker
    from src.engines.live.reconciliation import BaseAssetLockRegistry
    from src.strategies.components import Strategy as ComponentStrategy
    from src.trading.performance import PerformanceTracker

logger = logging.getLogger(__name__)


class LiveExitEngineState(Protocol):
    """Engine state the exit coordinator reads and mutates at call time.

    All exit-pipeline state stays on the engine; the coordinator reads/writes it
    through this backref. Accessed dynamically because handlers/session are wired
    during construction and ``start()``.
    """

    enable_live_trading: bool
    current_balance: float
    log_trades: bool
    trading_session_id: int | None
    completed_trades: list[Trade]
    _component_strategy: ComponentStrategy | None
    _base_asset_locks: BaseAssetLockRegistry
    live_position_tracker: LivePositionTracker
    live_exit_handler: LiveExitHandler
    live_execution_engine: LiveExecutionEngine
    performance_tracker: PerformanceTracker
    # Genuinely loose: concrete providers expose duck-typed margin/WS extensions
    # beyond the base interface (matches the engine's own ``exchange_interface: Any``).
    exchange_interface: Any
    data_provider: DataProvider
    db_manager: DatabaseManager

    # Engine helpers that stay on the engine; the coordinator calls them via this
    # backref (so subclass/test overrides on the engine still apply).
    def _extract_indicators(self, df: pd.DataFrame, index: int) -> dict: ...

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> dict: ...

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> dict: ...

    def _strategy_name(self) -> str: ...

    def _check_stop_loss_filled(self, position: Position) -> tuple[bool, float | None]: ...

    def _stop_loss_filled_quantity(self, position: Position) -> float | None: ...

    def _cancel_stop_loss_order(self, position: Position) -> bool: ...

    def _reprotect_position(self, position: Position) -> None: ...

    def _log_trade(self, trade: Trade) -> None: ...

    # Mocked in tests via patch.object(engine, ...); routed through state so those
    # mocks still intercept (see module docstring).
    def _execute_exit(
        self,
        position: Position,
        reason: str,
        limit_price: float | None,
        current_price: float,
        candle_high: float | None,
        candle_low: float | None,
        candle: Any,
        skip_live_close: bool = ...,
    ) -> None: ...


class LiveExitCoordinator:
    """Owns the live engine's exit decision + execution pipeline."""

    def __init__(self, engine_state: LiveExitEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    def check_exit_conditions(
        self,
        df: pd.DataFrame,
        current_index: int,
        current_price: float,
        runtime_decision: Any = None,
        candle: Any = None,
        safety_mode: bool = False,
    ) -> None:
        """Check if any positions should be closed."""
        state = self._state
        positions_snapshot = state.live_position_tracker.positions
        if not positions_snapshot:
            return

        # Extract candle high/low for more realistic SL/TP detection (parity with backtest)
        candle_high = None
        candle_low = None
        if df is not None and current_index < len(df):
            row = df.iloc[current_index]
            if "high" in df.columns:
                candle_high = float(row["high"])
            if "low" in df.columns:
                candle_low = float(row["low"])

        # Extract context for logging
        indicators = state._extract_indicators(df, current_index)
        sentiment_data = state._extract_sentiment_data(df, current_index)
        ml_predictions = state._extract_ml_predictions(df, current_index)

        component_strategy = None if safety_mode else state._component_strategy
        decision_for_exit = None if safety_mode else runtime_decision

        # Get current candle timestamp for same-bar exit protection
        candle_time = None
        if df is not None and current_index < len(df):
            candle_time = df.index[current_index]

        for position in positions_snapshot.values():
            # Same-bar exit protection: skip positions entered on the current
            # candle. The candle's high/low may include extremes before the
            # entry fill, making SL/TP evaluation unrealistic. Matches backtest
            # behavior where entered_this_candle prevents same-bar exits.
            if is_same_bar_entry(position.entry_time, candle_time):
                logger.debug(
                    "Skipping exit check for %s: entered on current bar",
                    position.symbol,
                )
                continue

            exit_check = state.live_exit_handler.check_exit_conditions(
                position=position,
                current_price=float(current_price),
                candle_high=candle_high,
                candle_low=candle_low,
                runtime_decision=decision_for_exit,
                component_strategy=component_strategy,
            )

            should_exit = exit_check.should_exit
            exit_reason = exit_check.exit_reason
            limit_price = exit_check.limit_price

            # Log exit decision for each position
            if state.db_manager:
                # Calculate current P&L for context using shared function for consistency
                # Note: Using fraction=1.0 to get raw P&L percentage for logging (unsized)
                if position.entry_price <= 0:
                    logger.error(
                        "Invalid entry_price %s for position %s - skipping P&L calculation for logging",
                        position.entry_price,
                        position.symbol,
                    )
                    current_pnl = 0.0  # Fallback value for logging
                elif float(current_price) <= 0 or not math.isfinite(float(current_price)):
                    logger.error(
                        "Invalid current_price %s for position %s - skipping P&L calculation for logging",
                        current_price,
                        position.symbol,
                    )
                    current_pnl = 0.0  # Fallback value for logging
                else:
                    side_enum = Side.LONG if position.side == PositionSide.LONG else Side.SHORT
                    current_pnl = pnl_percent(
                        position.entry_price, float(current_price), side_enum, 1.0
                    )

                # Prepare logging reasons with TradingDecision data if available
                log_reasons = [
                    exit_reason if should_exit else "holding_position",
                    f"current_pnl_{current_pnl:.4f}",
                    f"position_age_{(datetime.now(UTC).replace(tzinfo=None) - position.entry_time.replace(tzinfo=None)).total_seconds():.0f}s",
                    f"entry_price_{position.entry_price:.2f}",
                ]

                # Add regime context if available from TradingDecision
                if (
                    decision_for_exit
                    and hasattr(decision_for_exit, "regime")
                    and decision_for_exit.regime
                ):
                    regime = decision_for_exit.regime
                    log_reasons.append(
                        f"regime_trend_{regime.trend.value if hasattr(regime.trend, 'value') else regime.trend}"
                    )
                    log_reasons.append(
                        f"regime_volatility_{regime.volatility.value if hasattr(regime.volatility, 'value') else regime.volatility}"
                    )
                    log_reasons.append(f"regime_confidence_{regime.confidence:.2f}")

                # Add risk metrics if available from TradingDecision
                if (
                    decision_for_exit
                    and hasattr(decision_for_exit, "risk_metrics")
                    and decision_for_exit.risk_metrics
                ):
                    for key, value in decision_for_exit.risk_metrics.items():
                        if isinstance(value, int | float):
                            log_reasons.append(f"risk_{key}_{value:.4f}")

                # Extract signal confidence from TradingDecision if available
                confidence_score = indicators.get("prediction_confidence", 0.5)
                if (
                    decision_for_exit
                    and hasattr(decision_for_exit, "signal")
                    and decision_for_exit.signal
                ):
                    confidence_score = decision_for_exit.signal.confidence

                state.db_manager.log_strategy_execution(
                    strategy_name=state._strategy_name(),
                    symbol=position.symbol,
                    signal_type="exit",
                    action_taken="closed_position" if should_exit else "hold_position",
                    price=current_price,
                    timeframe="1m",
                    signal_strength=1.0 if should_exit else 0.0,
                    confidence_score=confidence_score,
                    indicators=indicators,
                    sentiment_data=sentiment_data if sentiment_data else None,
                    ml_predictions=ml_predictions if ml_predictions else None,
                    position_size=position.size,
                    reasons=log_reasons,
                    volume=indicators.get("volume"),
                    volatility=indicators.get("volatility"),
                    session_id=state.trading_session_id,
                )

            if should_exit:
                state._execute_exit(
                    position,
                    exit_reason,
                    limit_price,
                    float(current_price),
                    candle_high,
                    candle_low,
                    candle,
                )

    def execute_exit(
        self,
        position: Position,
        reason: str,
        limit_price: float | None,
        current_price: float,
        candle_high: float | None,
        candle_low: float | None,
        candle: Any,
        skip_live_close: bool = False,
    ) -> None:
        """Serialise the close on the position's base-asset lock, then execute it (#703).

        Re-entrant: an entry that already holds the lock (its SL-failed emergency
        close routes here) re-acquires it on the same thread without deadlock.
        """
        state = self._state
        from src.engines.live.reconciliation import PositionReconciler

        base = PositionReconciler._extract_base_asset(getattr(position, "symbol", "") or "")
        with state._base_asset_locks.lock_for(base):
            self.execute_exit_locked(
                position,
                reason,
                limit_price,
                current_price,
                candle_high,
                candle_low,
                candle,
                skip_live_close=skip_live_close,
            )

    def execute_exit_locked(
        self,
        position: Position,
        reason: str,
        limit_price: float | None,
        current_price: float,
        candle_high: float | None,
        candle_low: float | None,
        candle: Any,
        skip_live_close: bool = False,
    ) -> None:
        """Close a position using shared execution modules."""
        state = self._state
        try:
            # Defensive check: verify position still exists (prevents race with concurrent close)
            if position.order_id and not state.live_position_tracker.has_position(
                position.order_id
            ):
                logger.debug(
                    "Position %s no longer exists (already closed) - skipping exit",
                    position.order_id,
                )
                return

            if position.entry_price <= 0:
                logger.error(
                    "Invalid entry_price %s for position %s - cannot close position safely",
                    position.entry_price,
                    position.symbol,
                )
                return

            # Tracked positions always carry a non-None order_id; __post_init__
            # guarantees side is a PositionSide enum.
            metrics = state.live_position_tracker.mfe_mae_tracker.get_position_metrics(
                cast(str, position.order_id)
            )
            position_side = cast(PositionSide, position.side)

            sl_already_filled = False
            sl_fill_price: float | None = None
            if not skip_live_close:
                sl_already_filled, sl_fill_price = state._check_stop_loss_filled(position)
            else:
                sl_already_filled = True

            base_price = None
            if sl_already_filled and sl_fill_price is not None:
                base_price = float(sl_fill_price)
            elif limit_price is not None:
                base_price = float(limit_price)
            else:
                base_price = float(current_price)

            if sl_already_filled or skip_live_close:
                exit_result = state.live_exit_handler.execute_filled_exit(
                    position=position,
                    exit_reason=reason,
                    filled_price=base_price,
                    current_balance=state.current_balance,
                )
            else:
                # #710: On margin a resting stop-loss order reserves the position's
                # base asset, so a market close submitted while it rests is rejected
                # with -2010 (insufficient balance). We must cancel the stop first to
                # free the balance — but ONLY when it is safe to close the full size.
                #
                # Inventory-awareness: a stop that has ANY fill means held base !=
                # tracked size, so a full-size close would over-sell (long) / over-buy
                # (short). Re-query the stop's filled quantity BEFORE cancelling (no
                # unprotected window — it stays resting) and AGAIN after the confirmed
                # (terminal) cancel; on any fill, or any unconfirmable state, DEFER to
                # the periodic reconciler instead of closing. Only a provably clean
                # (zero-fill) stop is cancelled and the full size closed.
                protective_order_cancelled = False
                if (
                    state.enable_live_trading
                    and state.exchange_interface
                    and position.stop_loss_order_id
                ):
                    pre_filled = state._stop_loss_filled_quantity(position)
                    if pre_filled is None or pre_filled > BORROW_DUST_EPSILON:
                        logger.warning(
                            "Deferring market close of %s: resting stop %s is mid-fill or "
                            "its state is unconfirmable (filled=%s) — reconciler will "
                            "adjust. Not cancelling or closing.",
                            position.symbol,
                            position.stop_loss_order_id,
                            pre_filled,
                        )
                        return
                    if not state._cancel_stop_loss_order(position):
                        logger.warning(
                            "Skipping market close of %s: could not confirm cancel of "
                            "resting stop-loss %s; position remains protected, will retry.",
                            position.symbol,
                            position.stop_loss_order_id,
                        )
                        return
                    protective_order_cancelled = True
                    post_filled = state._stop_loss_filled_quantity(position)
                    if post_filled is None or post_filled > BORROW_DUST_EPSILON:
                        logger.warning(
                            "Deferring market close of %s: stop %s filled during cancel "
                            "(filled=%s) — reconciler will adjust. Not closing.",
                            position.symbol,
                            position.stop_loss_order_id,
                            post_filled,
                        )
                        return

                exit_result = state.live_exit_handler.execute_exit(
                    position=position,
                    exit_reason=reason,
                    current_price=float(current_price),
                    limit_price=limit_price,
                    current_balance=state.current_balance,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    data_provider=state.data_provider,
                )
                if not exit_result.success and protective_order_cancelled:
                    # The close failed after we cancelled a clean (zero-fill) stop, so
                    # the position is momentarily unprotected — re-protect immediately
                    # (verifying it is still held, to avoid orphaning a stop on an
                    # ambiguous / already-executed close). The periodic reconciler is
                    # the ultimate backstop. (#710)
                    state._reprotect_position(position)
            if not exit_result.success:
                logger.error(
                    "Failed to close position %s: %s",
                    position.order_id,
                    exit_result.error,
                )
                return

            realized_pnl = exit_result.realized_pnl - exit_result.exit_fee

            # Deduct margin interest for short positions in margin mode
            interest_cost = 0.0
            if (
                getattr(state.exchange_interface, "is_margin_mode", False)
                and position.side == PositionSide.SHORT
            ):
                try:
                    from src.engines.live.reconciliation import PositionReconciler

                    tracker = MarginInterestTracker(state.exchange_interface)
                    base_asset = PositionReconciler._extract_base_asset(position.symbol)
                    if base_asset == position.symbol:
                        logger.warning(
                            "Could not extract base asset from %s — margin interest may not be queried correctly",
                            position.symbol,
                        )
                    interest_base = tracker.get_position_interest_cost(
                        base_asset, position.entry_time
                    )
                    # Convert from base asset units to USDT using exit price
                    interest_cost = interest_base * float(exit_result.exit_price)
                    if interest_cost > 0:
                        realized_pnl -= interest_cost
                        logger.info(
                            "Deducted margin interest $%.4f (%.8f %s @ %.2f) from PnL for %s",
                            interest_cost,
                            interest_base,
                            base_asset,
                            float(exit_result.exit_price),
                            position.symbol,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to query margin interest for %s — proceeding without deduction: %s",
                        position.symbol,
                        e,
                    )

            # Atomic balance update with full audit trail for realized P&L
            if state.trading_session_id is not None:
                try:
                    with state.db_manager.atomic_balance_update(
                        balance_change=realized_pnl,
                        reason=f"realized_pnl_{position.symbol}_{reason}",
                        updated_by="live_engine",
                        correlation_id=position.order_id,
                    ) as balance_result:
                        state.current_balance = balance_result["new_balance"]
                except Exception as balance_err:
                    logger.error(
                        "Failed to update balance for realized P&L %s: %s. Trade will be logged but balance inconsistent.",
                        position.symbol,
                        balance_err,
                        exc_info=True,
                    )
                    # Continue processing to log the trade even if balance update fails
                    # This allows for manual reconciliation
            else:
                # No trading session - update balance directly (testing/paper trading mode)
                state.current_balance += realized_pnl
                if state.current_balance < 0:
                    logger.critical(
                        "CRITICAL: Balance went negative (%.6f) after realized PnL of "
                        "%.6f for %s (%s). MANUAL RECONCILIATION REQUIRED.",
                        state.current_balance,
                        realized_pnl,
                        position.symbol,
                        reason,
                    )

            exit_price = float(exit_result.exit_price)
            exit_fee = exit_result.exit_fee
            exit_slippage_cost = exit_result.slippage_cost
            pnl_percent = exit_result.realized_pnl_percent

            # Prefer the exact fee booked at open; reconstruct from the fee model for
            # positions recovered after a restart (no entry-fee metadata) so the trade's
            # commission still includes the entry leg. Feeds both performance metrics and
            # the persisted trades.commission below.
            entry_fee = _close_entry_fee_usd(position, state.live_execution_engine)
            entry_slippage_cost = float(position.metadata.get("entry_slippage_cost", 0.0))
            # This close is only the remaining slice of a partially-exited position
            # (1.0 for a full close). The Trade above is portion-level (size =
            # current_size), so the entry leg must be scaled to the same portion or
            # the performance metrics over-attribute the entry fee/slippage by
            # 1/portion — and diverge from the DB ledger, which already scales the
            # entry fee (see commission= below).
            close_portion = _close_position_portion(position)
            scaled_entry_fee = entry_fee * close_portion
            total_fee = scaled_entry_fee + exit_fee
            total_slippage = entry_slippage_cost * close_portion + exit_slippage_cost

            # Store GROSS P&L in Trade.pnl for parity with backtest engine
            # Fees are tracked separately via performance_tracker.record_trade()
            # This matches backtest behavior where Trade.pnl is price movement only
            gross_pnl = exit_result.realized_pnl

            trade = Trade(
                symbol=position.symbol,
                side=position.side,
                size=float(
                    position.current_size if position.current_size is not None else position.size
                ),
                entry_price=position.entry_price,
                exit_price=exit_price,
                entry_time=position.entry_time,
                exit_time=datetime.now(UTC),
                pnl=gross_pnl,
                pnl_percent=pnl_percent,
                exit_reason=reason,
            )

            # Include margin interest in performance tracker fees so
            # reported PnL, win rate, and net metrics account for financing.
            state.performance_tracker.record_trade(
                trade=trade,
                fee=total_fee + interest_cost,
                slippage=total_slippage,
            )

            state.completed_trades.append(trade)
            if state.log_trades:
                state._log_trade(trade)

            if state.trading_session_id is not None:
                # Pass the position's DB row id so log_trade flips the Position
                # to CLOSED in the SAME transaction as the Trade insert. This is
                # the single fix for #657: previously the closed Trade row was
                # written but positions.status stayed OPEN (the dedicated CLOSED
                # setters are all gated on enable_live_trading+exchange, so they
                # were dead in paper mode), leaving positions permanently OPEN
                # and re-closed on restart as phantom duplicate trades.
                #
                # db_position_id is set on the position object at open time
                # (LivePositionTracker.open_position) and on recovery
                # (_recover_active_positions). It may legitimately be None for a
                # position that was never persisted (e.g. db logging failed, or
                # tests without a DB); in that case log_trade falls back to its
                # original behaviour (insert trade only, no status flip).
                close_position_id = getattr(position, "db_position_id", None)
                if close_position_id is None:
                    logger.warning(
                        "Closing %s (%s) without db_position_id — position row "
                        "status will not be flipped to CLOSED; relying on startup "
                        "self-heal to reconcile.",
                        position.symbol,
                        position.order_id,
                    )
                state.db_manager.log_trade(
                    symbol=position.symbol,
                    side=position_side.value,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=float(
                        position.current_size
                        if position.current_size is not None
                        else position.size
                    ),
                    pnl=gross_pnl,
                    strategy_name=state._strategy_name(),
                    exit_reason=reason,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(UTC),
                    session_id=state.trading_session_id,
                    position_id=close_position_id,
                    mfe=(metrics.mfe if metrics else None),
                    mae=(metrics.mae if metrics else None),
                    mfe_price=(metrics.mfe_price if metrics else None),
                    mae_price=(metrics.mae_price if metrics else None),
                    mfe_time=(metrics.mfe_time if metrics else None),
                    mae_time=(metrics.mae_time if metrics else None),
                    # trades.commission is the round-trip fee in USD (entry + exit),
                    # the same values booked to account_balances: entry_fee as the
                    # entry_fee_<symbol> ledger event, exit_fee folded into the
                    # realized_pnl_<symbol> event. Deliberately NOT orders.actual_commission
                    # (raw exchange commission in the received asset — ETH on buys, USDT on
                    # sells — with no commission_asset column, populated asynchronously by
                    # reconciliation, so unit-ambiguous and unreliable at close time).
                    # The entry leg is scaled to the closed portion so a partial final
                    # close's commission matches its portion-level size/quantity/pnl
                    # (ratio is 1.0 for a full close). Reuses the same scaled total fed
                    # to performance_tracker.record_trade above so the metrics and the
                    # DB ledger never diverge.
                    commission=total_fee,
                    quantity=_closed_base_quantity(position),
                    margin_interest_cost=interest_cost,
                )

            # NOTE(#710): the resting stop-loss is now cancelled BEFORE the market
            # close (see the close path above) so it cannot reserve the base asset
            # and trigger -2010. No post-close cancel is needed here; on a successful
            # close the position (and its already-cancelled stop) are removed by the
            # exit handler.
            logger.info(
                "📈 Closed %s position for %s: PnL=$%.2f, Reason=%s, Balance=$%.2f",
                position_side.value,
                position.symbol,
                gross_pnl,
                reason,
                state.current_balance,
            )
            log_order_event(
                "close_position",
                order_id=position.order_id,
                symbol=position.symbol,
                side=position_side.value,
                exit_price=exit_price,
                pnl=gross_pnl,
                pnl_percent=trade.pnl_percent,
                reason=reason,
            )
        except Exception as e:
            logger.error("Failed to close position %s: %s", position.order_id, e, exc_info=True)
