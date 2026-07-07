"""Live entry-execution coordinator (#486).

Owns the live engine's entry pipeline: deciding whether to open a position
(``check_entry_conditions``) and executing it under the symbol's base-asset
lock (``execute_entry`` -> ``execute_entry_locked``). This is a real-money
path, so the extraction is behaviour-preserving: the methods are the verbatim
engine methods with ``self.`` rewritten to ``state.`` against an engine backref.

Locking & ordering (unchanged from the engine):
- ``execute_entry`` serialises the whole entry on the symbol's base-asset lock
  (``state._base_asset_locks``) so the orphaned-borrow sweep can't repay a
  borrow this entry just created (#703). The lock is held across order submit
  -> position tracking -> stop-loss placement, and any emergency-close fallback
  re-acquires it re-entrantly via ``state._execute_exit``.
- All engine state (balance, trackers, risk manager, session id) lives on the
  engine and is mutated through the ``state`` backref exactly as before; the
  coordinator holds no state of its own beyond that reference.
- The two engine methods that callers mock (``_execute_exit`` and
  ``_record_event``) are invoked through ``state`` so engine-level test mocks
  still intercept them.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import pandas as pd

from src.config.constants import DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT
from src.data_providers.exchange_interface import OrderSide, OrderType, SideEffectType
from src.database.models import EventType
from src.engines.live.execution.entry_handler import LiveEntrySignal
from src.engines.live.execution.entry_pause import EntryPauseGate
from src.engines.shared.models import PositionSide
from src.infrastructure.logging.events import log_order_event
from src.strategies.components import Signal, SignalDirection
from src.strategies.components import Strategy as ComponentStrategy
from src.tech.adapters.row_extractors import extract_ml_predictions_from_signal

if TYPE_CHECKING:
    from src.data_providers.data_provider import DataProvider
    from src.database.manager import DatabaseManager
    from src.engines.live.execution.entry_handler import LiveEntryHandler
    from src.engines.live.execution.position_tracker import LivePosition, LivePositionTracker
    from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
    from src.engines.live.order_tracker import OrderTracker
    from src.engines.live.reconciliation import BaseAssetLockRegistry
    from src.risk.risk_manager import RiskManager
    from src.trading.performance import PerformanceTracker

logger = logging.getLogger(__name__)


class LiveEntryEngineState(Protocol):
    """Engine state the entry coordinator reads and mutates at call time.

    All entry pipeline state stays on the engine; the coordinator reads/writes
    it through this backref. Accessed dynamically because handlers/session are
    wired during construction and ``start()``.
    """

    enable_live_trading: bool
    current_balance: float
    max_position_size: float
    timeframe: str | None
    trading_session_id: int | None
    strategy: Any
    risk_manager: RiskManager
    performance_tracker: PerformanceTracker
    live_entry_handler: LiveEntryHandler
    live_position_tracker: LivePositionTracker
    stop_loss_manager: LiveStopLossManager
    order_tracker: OrderTracker | None
    # Genuinely loose: concrete providers expose duck-typed margin/WS extensions
    # beyond the base interface (matches the engine's own ``exchange_interface: Any``).
    exchange_interface: Any
    data_provider: DataProvider
    db_manager: DatabaseManager
    _component_strategy: ComponentStrategy | None
    _close_only_mode: bool
    _base_asset_locks: BaseAssetLockRegistry

    # Engine helpers that stay on the engine; the coordinator calls them via
    # this backref (so subclass/test overrides on the engine still apply).
    def _is_runtime_strategy(self) -> bool: ...

    def _strategy_name(self) -> str: ...

    def _resolve_take_profit_pct(self) -> float: ...

    def _send_alert(self, message: str) -> bool: ...

    def _enter_close_only_mode(self) -> None: ...

    def _extract_indicators(self, df: pd.DataFrame, index: int) -> dict: ...

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> dict: ...

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> dict: ...

    def _build_component_positions(self, current_price: float) -> list: ...

    def _apply_policies_from_decision(self, decision: Any) -> None: ...

    def _apply_dynamic_risk_adjustment(
        self, original_size: float, current_time: datetime
    ) -> float: ...

    def _get_correlation_context(
        self,
        symbol: str,
        df: pd.DataFrame,
        overrides: dict | None,
        *,
        index: int | None = ...,
    ) -> dict | None: ...

    # Mocked in tests via patch.object(engine, ...); routed through state so
    # those mocks still intercept.
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

    def _execute_exit(
        self,
        position: LivePosition,
        reason: str,
        limit_price: float | None,
        current_price: float,
        candle_high: float | None,
        candle_low: float | None,
        candle: Any,
        skip_live_close: bool = ...,
    ) -> None: ...


class LiveEntryCoordinator:
    """Owns the live engine's entry decision + execution pipeline."""

    def __init__(self, engine_state: LiveEntryEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state
        # FEATURE_ENTRY_PAUSE gate for new entries (scale-ins are gated by the
        # exit handler's own instance — see EntryPauseGate).
        self._entry_pause = EntryPauseGate()

    def _entry_paused(self, context: str) -> bool:
        """True when FEATURE_ENTRY_PAUSE suppresses new entries (rate-limit logged)."""
        return self._entry_pause.paused(context)

    def check_entry_conditions(
        self,
        df: pd.DataFrame,
        current_index: int,
        symbol: str,
        current_price: float,
        current_time: datetime,
        runtime_decision=None,
    ):
        """Check if new positions should be opened"""
        state = self._state

        # Close-only mode: skip all entry signals, exits/stops still active
        if state._close_only_mode:
            logger.debug("Close-only mode active — skipping entry check")
            return

        if self._entry_paused(f"entry evaluation for {symbol}"):
            return

        use_runtime = state._is_runtime_strategy()
        entry_signal = False
        position_size = 0.0
        entry_side = PositionSide.LONG
        runtime_strength = 0.0
        runtime_confidence = 0.0
        stop_loss = None
        take_profit = None
        overrides = None

        indicators = state._extract_indicators(df, current_index)
        sentiment_data = state._extract_sentiment_data(df, current_index)
        ml_predictions = state._extract_ml_predictions(df, current_index)
        # Signal whose metadata carries the model outputs (or failure reason)
        # for ml_predictions logging (#914); the component path overwrites it
        # with its locally produced decision below.
        decision_signal = getattr(runtime_decision, "signal", None)

        if use_runtime:
            perf_metrics = state.performance_tracker.get_metrics()
            # Pass symbol/timeframe/df/index so LiveEntryHandler can run
            # correlation control. These are required keyword args of the
            # handler's correlation guard (see LiveEntryHandler.process_runtime_decision
            # at src/engines/live/execution/entry_handler.py:208-222) — without
            # them, correlation_handler.apply_correlation_control is silently
            # skipped, so the live engine over-concentrates in correlated pairs
            # that the backtest engine de-risks.
            entry_signal_result = state.live_entry_handler.process_runtime_decision(
                runtime_decision=runtime_decision,
                balance=state.current_balance,
                current_price=float(current_price),
                current_time=datetime.now(UTC),
                symbol=symbol,
                timeframe=state.timeframe,
                df=df,
                index=current_index,
                peak_balance=perf_metrics.peak_balance or state.current_balance,
                trading_session_id=state.trading_session_id,
            )
            if entry_signal_result.should_enter and entry_signal_result.side is not None:
                entry_signal = True
                entry_side = entry_signal_result.side
                position_size = entry_signal_result.size_fraction
                stop_loss = entry_signal_result.stop_loss
                take_profit = entry_signal_result.take_profit
                runtime_strength = entry_signal_result.signal_strength
                runtime_confidence = entry_signal_result.signal_confidence
        elif isinstance(state.strategy, ComponentStrategy):
            # Component-based strategy: use process_candle() for decision
            # Note: runtime_decision should already be populated if this is a component strategy
            # This branch handles direct ComponentStrategy usage without StrategyRuntime wrapper.
            # Pass the live positions list so strategies that consult
            # current_positions (anti-pyramiding, correlation-aware sizing,
            # etc.) see the same view they would in the StrategyRuntime path
            # and in backtest. Previously this hardcoded ``None`` and silently
            # diverged from backtest.
            try:
                # Fall back to the most recent close (df[-1]) when
                # current_index is past the end — never to 0.0, which would
                # produce a -100% pnl_percent in ComponentPosition and could
                # trigger forced-exit logic in strategies that consult
                # current_positions.
                if len(df) == 0:
                    fallback_price = 0.0
                elif current_index < len(df):
                    fallback_price = float(df["close"].iloc[current_index])
                else:
                    fallback_price = float(df["close"].iloc[-1])
                current_positions = state._build_component_positions(fallback_price)
                decision = state.strategy.process_candle(
                    df,
                    current_index,
                    state.current_balance,
                    current_positions or None,
                )
                decision_signal = decision.signal
                state._apply_policies_from_decision(decision)

                notional_size = float(decision.position_size or 0.0)
                balance = float(state.current_balance or 0.0)
                size_fraction = 0.0 if balance <= 0 else max(0.0, notional_size / balance)
                bounded_fraction = min(size_fraction, state.max_position_size)

                if decision.signal.direction == SignalDirection.BUY and bounded_fraction > 0:
                    entry_signal = True
                    entry_side = PositionSide.LONG
                    position_size = bounded_fraction
                    runtime_strength = decision.signal.strength
                    runtime_confidence = decision.signal.confidence
                elif decision.signal.direction == SignalDirection.SELL and bounded_fraction > 0:
                    entry_signal = True
                    entry_side = PositionSide.SHORT
                    position_size = bounded_fraction
                    runtime_strength = decision.signal.strength
                    runtime_confidence = decision.signal.confidence
            except Exception as e:
                logger.warning("Component strategy decision failed: %s", e)
                entry_signal = False
        else:
            # All strategies should be component-based
            logger.error("Strategy %s is not a component-based strategy", state.strategy.name)
            entry_signal = False

        if entry_signal and not use_runtime:
            # Component strategies supply their own sizing. Retain correlation context computation
            # for downstream consumers that expect it to be populated as part of the entry check.
            state._get_correlation_context(symbol, df, None, index=current_index)

        if position_size > 0 and not use_runtime:
            position_size = state._apply_dynamic_risk_adjustment(position_size, current_time)

        if state.db_manager:
            # Enrich with model outputs from the signal metadata — the
            # dataframe columns the extractor reads are never populated by
            # component strategies, which left ml_predictions null (#914).
            signal_ml = extract_ml_predictions_from_signal(decision_signal)
            if signal_ml:
                ml_predictions = {**ml_predictions, **signal_ml}

            # Prepare logging data - include TradingDecision data if available
            log_reasons = [
                (
                    "runtime_entry"
                    if use_runtime
                    else "entry_conditions_met" if entry_signal else "entry_conditions_not_met"
                ),
                (f"position_size_{position_size:.4f}" if position_size > 0 else "no_position_size"),
                f"max_positions_check_{state.live_position_tracker.position_count}_of_{state.risk_manager.get_max_concurrent_positions() if state.risk_manager else 1}",
                (
                    f"enter_short_{bool(getattr(runtime_decision, 'metadata', {}).get('enter_short'))}"
                    if use_runtime and runtime_decision is not None
                    else "enter_short_n/a"
                ),
            ]

            # Add regime context if available from TradingDecision
            if runtime_decision and hasattr(runtime_decision, "regime") and runtime_decision.regime:
                regime = runtime_decision.regime
                log_reasons.append(
                    f"regime_trend_{regime.trend.value if hasattr(regime.trend, 'value') else regime.trend}"
                )
                log_reasons.append(
                    f"regime_volatility_{regime.volatility.value if hasattr(regime.volatility, 'value') else regime.volatility}"
                )
                log_reasons.append(f"regime_confidence_{regime.confidence:.2f}")

            # Add risk metrics if available from TradingDecision
            if (
                runtime_decision
                and hasattr(runtime_decision, "risk_metrics")
                and runtime_decision.risk_metrics
            ):
                for key, value in runtime_decision.risk_metrics.items():
                    if isinstance(value, int | float):
                        log_reasons.append(f"risk_{key}_{value:.4f}")

            state.db_manager.log_strategy_execution(
                strategy_name=state._strategy_name(),
                symbol=symbol,
                signal_type="entry",
                action_taken=(
                    "opened_long"
                    if entry_signal and position_size > 0 and entry_side == PositionSide.LONG
                    else (
                        "opened_short"
                        if entry_signal and position_size > 0 and entry_side == PositionSide.SHORT
                        else "no_action"
                    )
                ),
                price=current_price,
                timeframe="1m",
                signal_strength=runtime_strength if use_runtime else (1.0 if entry_signal else 0.0),
                confidence_score=(
                    runtime_confidence
                    if use_runtime
                    else indicators.get("prediction_confidence", 0.5)
                ),
                indicators=indicators,
                sentiment_data=sentiment_data if sentiment_data else None,
                ml_predictions=ml_predictions if ml_predictions else None,
                position_size=position_size if position_size > 0 else None,
                reasons=log_reasons,
                volume=indicators.get("volume"),
                volatility=indicators.get("volatility"),
                session_id=state.trading_session_id,
            )

        if not entry_signal or position_size <= 0:
            return

        if use_runtime and state._component_strategy is not None:
            try:
                if stop_loss is None:
                    stop_loss = state._component_strategy.get_stop_loss_price(
                        float(current_price),
                        runtime_decision.signal if runtime_decision else None,
                        runtime_decision.regime if runtime_decision else None,
                    )
            except Exception as e:
                logger.warning(
                    "Stop loss price calculation failed for %s, using default: %s", symbol, e
                )
                if stop_loss is None:
                    stop_loss = float(current_price) * (
                        (1 - DEFAULT_STOP_LOSS_PCT)
                        if entry_side == PositionSide.LONG
                        else (1 + DEFAULT_STOP_LOSS_PCT)
                    )
            if take_profit is None:
                tp_pct = state._resolve_take_profit_pct()
                take_profit = (
                    float(current_price) * (1 + tp_pct)
                    if entry_side == PositionSide.LONG
                    else float(current_price) * (1 - tp_pct)
                )
        elif isinstance(state.strategy, ComponentStrategy):
            # Component-based strategy: use get_stop_loss_price()
            try:
                # Create a signal from the decision
                signal = Signal(
                    direction=(
                        SignalDirection.BUY
                        if entry_side == PositionSide.LONG
                        else SignalDirection.SELL
                    ),
                    strength=runtime_strength,
                    confidence=runtime_confidence,
                    metadata={},
                )
                stop_loss = state.strategy.get_stop_loss_price(
                    float(current_price), signal, None  # regime context
                )
            except Exception as e:
                logger.warning("Component stop loss calculation failed for %s: %s", symbol, e)
                stop_loss = float(current_price) * (
                    (1 - DEFAULT_STOP_LOSS_PCT)
                    if entry_side == PositionSide.LONG
                    else (1 + DEFAULT_STOP_LOSS_PCT)
                )
            tp_pct = state._resolve_take_profit_pct()
            take_profit = (
                float(current_price) * (1 + tp_pct)
                if entry_side == PositionSide.LONG
                else float(current_price) * (1 - tp_pct)
            )
        else:
            try:
                overrides = (
                    state.strategy.get_risk_overrides()
                    if hasattr(state.strategy, "get_risk_overrides")
                    else None
                )
            except Exception as e:
                logger.warning("Failed to get risk overrides for %s: %s", symbol, e)
                overrides = None

            if overrides and ("stop_loss_pct" in overrides or "take_profit_pct" in overrides):
                stop_loss, take_profit = state.risk_manager.compute_sl_tp(
                    df=df,
                    index=current_index,
                    entry_price=current_price,
                    side="long",
                    strategy_overrides=overrides,
                )
                if take_profit is None:
                    take_profit = current_price * (
                        1 + overrides.get("take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)
                    )
            else:
                # All strategies should be component-based
                logger.error(
                    "Strategy %s does not support component-based stop loss calculation",
                    state.strategy.name,
                )
                stop_loss = current_price * (1 - DEFAULT_STOP_LOSS_PCT)  # Default 5% stop for long
                take_profit = current_price * (
                    1 + getattr(state.strategy, "take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)
                )
            entry_side = PositionSide.LONG

        self.execute_entry(
            symbol=symbol,
            side=entry_side,
            size=position_size,
            price=float(current_price),
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_strength=runtime_strength,
            signal_confidence=runtime_confidence,
        )

    def execute_entry(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        price: float,
        stop_loss: float | None,
        take_profit: float | None,
        signal_strength: float,
        signal_confidence: float,
    ) -> None:
        """Serialise the entry on the symbol's base-asset lock, then execute it.

        The lock is held across order submit -> position tracking (and any
        emergency-close fallback, which re-acquires it re-entrantly) so the
        orphaned-borrow sweep can't repay a borrow this entry just created (#703).
        """
        state = self._state
        from src.engines.live.reconciliation import PositionReconciler

        base = PositionReconciler._extract_base_asset(symbol)
        with state._base_asset_locks.lock_for(base):
            self.execute_entry_locked(
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
            )

    def execute_entry_locked(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        price: float,
        stop_loss: float | None,
        take_profit: float | None,
        signal_strength: float,
        signal_confidence: float,
    ) -> None:
        """Execute a new trading position using shared execution modules."""
        state = self._state
        # Defense-in-depth: refuse any entry routed around check_entry_conditions.
        # Close-only mode gates the chokepoint too, so the legacy short path and
        # any direct caller can never add exposure while halted.
        if state._close_only_mode:
            logger.debug("Close-only mode active — refusing entry execution for %s", symbol)
            return
        if self._entry_paused(f"entry execution for {symbol}"):
            return
        try:
            # Prevent duplicate positions on the same symbol (guards against multi-slot
            # risk managers with max_concurrent_positions > 1).
            if state.live_position_tracker.has_position_for_symbol(symbol):
                logger.info(
                    "Position already open for %s — skipping duplicate entry.",
                    symbol,
                )
                return

            # Check max concurrent positions limit (defense-in-depth, also checked in loop)
            max_concurrent = state.risk_manager.get_max_concurrent_positions()
            if state.live_position_tracker.position_count >= max_concurrent:
                logger.warning(
                    "Max concurrent positions limit reached (%d/%d). Rejecting entry for %s.",
                    state.live_position_tracker.position_count,
                    max_concurrent,
                    symbol,
                )
                return

            if size > state.max_position_size:
                logger.warning(
                    "Position size %.2f%% exceeds maximum %.2f%%. Capping at maximum.",
                    size * 100,
                    state.max_position_size * 100,
                )
                size = state.max_position_size

            # Build entry reasons for logging and analysis
            entry_reasons = [
                f"side_{side.value}",
                f"size_{size:.4f}",
                f"strength_{signal_strength:.2f}",
                f"confidence_{signal_confidence:.2f}",
            ]
            if stop_loss is not None:
                entry_reasons.append(f"sl_{stop_loss:.2f}")
            if take_profit is not None:
                entry_reasons.append(f"tp_{take_profit:.2f}")

            entry_signal = LiveEntrySignal(
                should_enter=True,
                side=side,
                size_fraction=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=entry_reasons,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
            )
            result = state.live_entry_handler.execute_entry(
                signal=entry_signal,
                symbol=symbol,
                current_price=price,
                balance=state.current_balance,
            )

            if not result.executed or result.position is None:
                logger.error("Failed to execute entry for %s: %s", symbol, result.error)
                return

            position = result.position
            entry_fee = result.entry_fee
            entry_slippage_cost = result.slippage_cost

            # Atomic balance update with full audit trail when trading session exists
            if state.trading_session_id is not None:
                try:
                    with state.db_manager.atomic_balance_update(
                        balance_change=-entry_fee,
                        reason=f"entry_fee_{symbol}",
                        updated_by="live_engine",
                        correlation_id=position.order_id,
                    ) as balance_result:
                        state.current_balance = balance_result["new_balance"]
                except Exception as balance_err:
                    logger.error(
                        "Failed to update balance for entry fee %s: %s. Aborting entry.",
                        symbol,
                        balance_err,
                        exc_info=True,
                    )
                    # Critical: Entry executed but balance update failed
                    # Attempt emergency close to maintain consistency
                    if state.enable_live_trading and state.exchange_interface:
                        try:
                            close_side = (
                                OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
                            )
                            # Validate entry_price to prevent division by zero
                            if position.entry_price <= 0:
                                logger.error(
                                    "Cannot calculate emergency close quantity - invalid "
                                    "entry_price %s for %s",
                                    position.entry_price,
                                    symbol,
                                    exc_info=True,
                                )
                            else:
                                # Use quantity from position - LiveEntryResult.position.quantity
                                emergency_order = state.exchange_interface.place_order(
                                    symbol=symbol,
                                    side=close_side,
                                    order_type=OrderType.MARKET,
                                    quantity=result.position.quantity,
                                    side_effect_type=SideEffectType.AUTO_REPAY,
                                )
                                if emergency_order is None:
                                    # None is an ambiguous/failed placement, NOT a
                                    # confirmed close: the position may still be open
                                    # and unprotected on the exchange. Escalate to
                                    # close-only instead of logging a false success;
                                    # the reconciler resolves it on restart.
                                    logger.critical(
                                        "CRITICAL: Emergency close for %s UNCONFIRMED "
                                        "(place_order returned None) after balance update "
                                        "failure — position may remain open on the exchange. "
                                        "Entering close-only mode until restart reconciles. "
                                        "MANUAL INTERVENTION REQUIRED.",
                                        symbol,
                                    )
                                    state._enter_close_only_mode()
                                else:
                                    logger.warning(
                                        "Emergency close placed for %s due to balance "
                                        "update failure",
                                        symbol,
                                    )
                        except Exception as close_err:
                            logger.critical(
                                "CRITICAL: Emergency close FAILED after balance update failure for %s. "
                                "MANUAL INTERVENTION REQUIRED. Error: %s",
                                symbol,
                                close_err,
                            )
                    return
            else:
                # No trading session - update balance directly (testing/paper trading mode)
                state.current_balance -= entry_fee
                if state.current_balance < 0:
                    logger.critical(
                        "CRITICAL: Balance went negative (%.6f) after entry fee deduction "
                        "of %.6f for %s. MANUAL RECONCILIATION REQUIRED.",
                        state.current_balance,
                        entry_fee,
                        symbol,
                    )

            position.metadata["entry_fee"] = entry_fee
            position.metadata["entry_slippage_cost"] = entry_slippage_cost

            # CRITICAL: Register position with tracker IMMEDIATELY after execution
            # to minimize race window with OrderTracker callbacks.
            # If this fails after order execution, we have an orphaned position.
            try:
                state.live_position_tracker.open_position(
                    position=position,
                    session_id=state.trading_session_id,
                    strategy_name=state._strategy_name(),
                )
            except Exception as tracker_err:
                # Position executed on exchange but failed to track locally.
                # This is critical - attempt emergency close to avoid orphaned position.
                logger.critical(
                    "CRITICAL: Position tracking failed after order execution for %s. "
                    "Attempting emergency close. Error: %s",
                    symbol,
                    tracker_err,
                )
                emergency_close_confirmed = False
                live_emergency_path = bool(state.enable_live_trading and state.exchange_interface)
                if live_emergency_path:
                    try:
                        close_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY

                        # Use quantity from position - LiveEntryResult.position.quantity
                        # No need to recalculate from entry_price which could introduce errors.
                        # Live executed entries always carry the filled quantity.
                        if cast(float, result.position.quantity) <= 0:
                            logger.critical(
                                "CRITICAL: Cannot place emergency close for %s - "
                                "invalid quantity %.8f. MANUAL INTERVENTION REQUIRED.",
                                symbol,
                                result.position.quantity,
                            )
                        else:
                            emergency_order = state.exchange_interface.place_order(
                                symbol=symbol,
                                side=close_side,
                                order_type=OrderType.MARKET,
                                quantity=result.position.quantity,
                                side_effect_type=SideEffectType.AUTO_REPAY,
                            )
                            if emergency_order is None:
                                # Ambiguous/failed placement (not a confirmed close):
                                # the orphaned position may still be open on the
                                # exchange. Escalate to close-only; the reconciler
                                # resolves the untracked position on restart.
                                logger.critical(
                                    "CRITICAL: Emergency close for orphaned position %s "
                                    "UNCONFIRMED (place_order returned None) — position may "
                                    "remain open on the exchange. Entering close-only mode "
                                    "until restart reconciles. MANUAL INTERVENTION REQUIRED.",
                                    symbol,
                                )
                                state._enter_close_only_mode()
                            else:
                                emergency_close_confirmed = True
                                logger.info(
                                    "Emergency close order placed for orphaned position %s",
                                    symbol,
                                )
                    except Exception as close_err:
                        logger.critical(
                            "CRITICAL: Emergency close FAILED for %s. "
                            "MANUAL INTERVENTION REQUIRED. Error: %s",
                            symbol,
                            close_err,
                        )
                # Refund the entry fee only when no possibly-open live position was left
                # behind: paper (no live path) always refunds; a live path refunds only
                # on a CONFIRMED close. On an unconfirmed/failed live close the position
                # may still be open, so keep the fee charged (the reconciler resolves the
                # real state on restart) rather than optimistically crediting the balance.
                should_refund = (not live_emergency_path) or emergency_close_confirmed
                if should_refund:
                    if state.trading_session_id is not None:
                        try:
                            with state.db_manager.atomic_balance_update(
                                balance_change=entry_fee,
                                reason=f"refund_entry_fee_{symbol}_tracking_failed",
                                updated_by="live_engine",
                                correlation_id=position.order_id,
                            ) as balance_result:
                                state.current_balance = balance_result["new_balance"]
                        except Exception as refund_err:
                            logger.critical(
                                "CRITICAL: Failed to refund entry fee after position tracking "
                                "failure for %s. Balance state inconsistent. Error: %s",
                                symbol,
                                refund_err,
                            )
                    else:
                        # No trading session - update balance directly
                        state.current_balance += entry_fee
                return

            # Update risk manager tracking for new position.
            # If this fails, close the position to maintain state consistency.
            if state.risk_manager:
                try:
                    state.risk_manager.update_position(
                        symbol=symbol,
                        side=side.value,
                        size=size,
                        entry_price=position.entry_price,
                    )
                except (AttributeError, ValueError, KeyError, TypeError) as e:
                    # Risk manager update failed - state is now inconsistent.
                    # Close position to prevent exceeding risk limits.
                    logger.error(
                        "Risk manager update failed for %s position %s. "
                        "Closing position to maintain risk consistency. Error: %s",
                        side.value,
                        symbol,
                        e,
                    )
                    state._execute_exit(
                        position,
                        "Risk manager sync failure",
                        None,
                        price,
                        None,
                        None,
                        None,
                        skip_live_close=False,
                    )
                    return

            logger.info(
                "🚀 Opened %s position: %s @ $%.2f (Size: %.2f%%)",
                side.value,
                symbol,
                position.entry_price,
                size * 100,
            )
            log_order_event(
                "open_position",
                order_id=position.order_id,
                symbol=symbol,
                side=side.value,
                entry_price=position.entry_price,
                size=size,
            )

            # Register with order tracker AFTER position is fully tracked.
            # This ensures callbacks can find the position in the tracker.
            if position.order_id and state.order_tracker:
                try:
                    state.order_tracker.track_order(position.order_id, symbol)
                except Exception as e:
                    # Order tracking failure is non-critical - position exists and is tracked.
                    # Stop-loss monitoring may be affected but position is safe.
                    logger.warning(
                        "Failed to track order %s for %s (position still valid): %s",
                        position.order_id,
                        symbol,
                        e,
                    )

            # Send alert if configured
            state._send_alert(
                f"Position Opened: {symbol} {side.value} @ ${position.entry_price:.2f}"
            )

            # Ambiguous entry: order submission timed out so we don't know if/how
            # much actually filled. Track the phantom position (so the reconciler can
            # resolve it on restart) but do NOT place a stop-loss and immediately
            # enter close-only mode to prevent further exposure.
            if result.ambiguous:
                logger.critical(
                    "Ambiguous order submission for %s (order_id=%s) — "
                    "entering close-only mode until restart reconciles the phantom position. "
                    "No stop-loss placed.",
                    symbol,
                    position.order_id,
                )
                state._enter_close_only_mode()
                return

            # Place server-side stop-loss order for protection with retry logic
            if state.enable_live_trading and stop_loss is not None and state.exchange_interface:
                # Use stored quantity directly to ensure stop-loss covers exact position size
                if position.quantity is not None and position.quantity > 0:
                    quantity = position.quantity
                else:
                    # Fallback for legacy positions without quantity field
                    entry_balance = (
                        float(position.entry_balance)
                        if position.entry_balance is not None and position.entry_balance > 0
                        else float(state.current_balance)
                    )
                    position_value = size * entry_balance
                    quantity = (
                        position_value / float(position.entry_price)
                        if position.entry_price is not None and position.entry_price > 0
                        else 0.0
                    )

                sl_order_id = state.stop_loss_manager.place_protection(
                    position=position,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    stop_price=stop_loss,
                )

                if not sl_order_id:
                    logger.critical(
                        "CRITICAL: Failed to place stop-loss after %s attempts for %s - "
                        "closing position on exchange to prevent unprotected exposure",
                        3,  # placement retry budget lives in LiveStopLossManager
                        symbol,
                    )
                    # Record the structured event and fire the alert in one call
                    # (alert=True dispatches the webhook, so no separate
                    # _send_alert is needed here).
                    state._record_event(
                        EventType.ALERT,
                        f"EMERGENCY: Closing {symbol} position - stop-loss placement failed",
                        severity="critical",
                        component="execution",
                        error_code="EMERGENCY_CLOSE",
                        alert=True,
                    )
                    # Fetch current market price for accurate exit pricing.
                    # The position is still open on the exchange without a stop loss,
                    # so we MUST close it via live order (skip_live_close=False).
                    # Using skip_live_close=True would only remove it from local
                    # tracking while leaving the unprotected position open on the
                    # exchange - a critical fund loss risk.
                    try:
                        emergency_price = state.data_provider.get_current_price(symbol)
                    except Exception as exc:
                        logger.warning(
                            "Failed to fetch current price for emergency close of %s; "
                            "falling back to entry price %s for exit pricing: %s",
                            symbol,
                            price,
                            exc,
                        )
                        emergency_price = price
                    state._execute_exit(
                        position,
                        "Stop-loss placement failed - emergency close",
                        None,
                        emergency_price,
                        None,
                        None,
                        None,
                        skip_live_close=False,
                    )

        except Exception as e:
            logger.error("Failed to open position: %s", e, exc_info=True)
            if state.trading_session_id is not None:
                state.db_manager.log_event(
                    event_type="ERROR",
                    message=f"Failed to open position: {str(e)}",
                    severity="error",
                    component="LiveTradingEngine",
                    details={"stack_trace": str(e)},
                    session_id=state.trading_session_id,
                )
            else:
                logger.warning("⚠️ Cannot log error to database - no trading session ID available")

    def process_legacy_short_entry(
        self,
        df: pd.DataFrame,
        current_index: int,
        symbol: str,
        current_price: float,
        current_time: datetime,
    ) -> None:
        """Evaluate and execute a legacy duck-typed short entry (non-runtime strategies)."""
        state = self._state
        # Close-only mode: skip legacy short evaluation, exits/stops still active
        if state._close_only_mode:
            logger.debug("Close-only mode active — skipping legacy short entry check")
            return
        if self._entry_paused(f"legacy short entry for {symbol}"):
            return
        if (not state._is_runtime_strategy()) and callable(
            getattr(state.strategy, "check_short_entry_conditions", None)
        ):
            # Legacy duck-typed hook; presence is verified by the
            # callable(getattr(...)) guard above.
            short_entry_signal = cast(Any, state.strategy).check_short_entry_conditions(
                df, current_index
            )
            if short_entry_signal:
                try:
                    overrides = (
                        state.strategy.get_risk_overrides()
                        if hasattr(state.strategy, "get_risk_overrides")
                        else None
                    )
                except Exception:
                    logger.warning(
                        "get_risk_overrides() raised for %s; proceeding with overrides=None "
                        "(strategy-configured SL/TP may not apply to this short entry)",
                        getattr(state.strategy, "name", "<unknown>"),
                        exc_info=True,
                    )
                    overrides = None
                indicators = state._extract_indicators(df, current_index)
                # Correlation context for short entries
                short_correlation_ctx = state._get_correlation_context(
                    symbol,
                    df,
                    overrides,
                    index=current_index,
                )
                if overrides and overrides.get("position_sizer"):
                    short_fraction = state.risk_manager.calculate_position_fraction(
                        df=df,
                        index=current_index,
                        balance=state.current_balance,
                        price=current_price,
                        indicators=indicators,
                        strategy_overrides=overrides,
                        correlation_ctx=short_correlation_ctx,
                    )
                    short_fraction = min(short_fraction, state.max_position_size)
                    short_position_size = short_fraction
                else:
                    # All strategies should be component-based
                    logger.error(
                        "Strategy %s does not support component-based position sizing",
                        state.strategy.name,
                    )
                    short_position_size = 0.0

                # Apply dynamic risk adjustments
                short_position_size = state._apply_dynamic_risk_adjustment(
                    short_position_size,
                    current_time,
                )

                # Regime-gated gross exposure cap (#802). This legacy path has no
                # regime context, so the governor uses its most-conservative
                # (unknown-regime) cap by design — tightly bounding legacy shorts.
                # Routed through the same live entry handler as the runtime path
                # so the cap arithmetic is shared (no bypass).
                if short_position_size > 0:
                    short_position_size, short_gate_reason = (
                        state.live_entry_handler.apply_pre_order_gates(
                            short_position_size,
                            regime=None,
                            equity=state.current_balance,
                            now=current_time,
                        )
                    )
                    if short_gate_reason:
                        logger.info(
                            "Legacy short exposure-capped for %s: %s", symbol, short_gate_reason
                        )
                if short_position_size > 0:
                    if overrides and (
                        ("stop_loss_pct" in overrides) or ("take_profit_pct" in overrides)
                    ):
                        short_stop_loss, short_take_profit = state.risk_manager.compute_sl_tp(
                            df=df,
                            index=current_index,
                            entry_price=current_price,
                            side="short",
                            strategy_overrides=overrides,
                        )
                        if short_take_profit is None:
                            short_take_profit = current_price * (
                                1
                                - getattr(
                                    state.strategy,
                                    "take_profit_pct",
                                    DEFAULT_TAKE_PROFIT_PCT,
                                )
                            )
                    else:
                        # All strategies should be component-based
                        logger.error(
                            "Strategy %s does not support component-based stop loss calculation",
                            state.strategy.name,
                        )
                        short_stop_loss = current_price * (
                            1 + DEFAULT_STOP_LOSS_PCT
                        )  # Default 5% stop for short
                        short_take_profit = current_price * (
                            1
                            - getattr(
                                state.strategy,
                                "take_profit_pct",
                                DEFAULT_TAKE_PROFIT_PCT,
                            )
                        )
                    self.execute_entry(
                        symbol=symbol,
                        side=PositionSide.SHORT,
                        size=short_position_size,
                        price=float(current_price),
                        stop_loss=short_stop_loss,
                        take_profit=short_take_profit,
                        signal_strength=0.0,
                        signal_confidence=0.0,
                    )
