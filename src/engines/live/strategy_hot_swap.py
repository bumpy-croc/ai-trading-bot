"""Strategy hot-swap and model-update lifecycle for the live trading engine.

Owns the engine's hot-swap pipeline: the public ``hot_swap_strategy`` /
``update_model`` entry points, the ``StrategyManager`` callbacks
(``handle_strategy_change`` / ``handle_model_update``), the loop-applied
``apply_pending_strategy_update``, and the post-swap refresh of all
strategy-derived engine state (trailing-stop / partial-operations / time-exit
policies, component risk re-binding, correlation-handler strategy reference).
Extracted from ``LiveTradingEngine`` (#486) so the engine orchestrates while
this coordinator owns the swap mechanics.

Thread-safety / lock ownership: the coordinator holds no locks of its own. The
public entry points and the callbacks run on the *caller's* thread and only
queue a pending update (guarded by ``StrategyManager``'s ``update_lock``) plus
synchronous side effects (closing positions); all engine-state mutation happens
later in ``apply_pending_strategy_update`` on the single trading-loop thread, as
the run loop drains the pending update. The refactor relocates these
reads/writes without changing which thread performs them, so the pre-existing
lock-free design is preserved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

from src.config.constants import (
    DEFAULT_END_OF_DAY_FLAT,
    DEFAULT_MARKET_TIMEZONE,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_TIME_RESTRICTIONS,
    DEFAULT_WEEKEND_FLAT,
)
from src.config.feature_flags import is_enabled
from src.engines.shared.partial_operations_manager import PartialOperationsManager
from src.position_management.partial_manager import PartialExitPolicy
from src.position_management.time_exits import TimeExitPolicy, TimeRestrictions
from src.risk.risk_manager import RiskParameters
from src.strategies.components import Strategy as ComponentStrategy

if TYPE_CHECKING:
    from src.data_providers.data_provider import DataProvider
    from src.engines.live.execution.exit_handler import LiveExitHandler
    from src.engines.live.execution.position_tracker import LivePosition, LivePositionTracker
    from src.engines.live.strategy_manager import StrategyManager
    from src.position_management.early_cut import EarlyCutPolicy
    from src.position_management.trailing_stops import TrailingStopPolicy

logger = logging.getLogger(__name__)


class HotSwapEngineState(Protocol):
    """Engine state the coordinator reads and mutates at call time.

    All mutation happens on the trading-loop thread (via
    ``apply_pending_strategy_update``); the public entry points and callbacks
    only read ``strategy_manager`` and fire synchronous side effects.
    """

    strategy_manager: StrategyManager | None
    live_position_tracker: LivePositionTracker
    data_provider: DataProvider
    risk_manager: Any
    strategy: Any
    _component_strategy: ComponentStrategy | None
    enable_partial_operations: bool
    live_exit_handler: LiveExitHandler
    live_entry_handler: Any

    # Mutated during the post-swap refresh.
    trailing_stop_policy: TrailingStopPolicy | None
    _trailing_stop_opt_in: bool
    partial_manager: PartialExitPolicy | None
    _partial_operations_opt_in: bool
    time_exit_policy: TimeExitPolicy | None
    early_cut_policy: EarlyCutPolicy | None
    _runtime_dataset: Any
    _runtime_warmup: int

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

    def _send_alert(self, message: str) -> object: ...

    def _strategy_name(self) -> str: ...

    def _build_trailing_policy(self) -> TrailingStopPolicy | None: ...

    def _configure_strategy(self, strategy: Any) -> None: ...

    def _finalize_runtime(self) -> None: ...


class StrategyHotSwapCoordinator:
    """Drives the strategy hot-swap / model-update lifecycle."""

    def __init__(self, engine_state: HotSwapEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    def handle_strategy_change(self, swap_data: dict[str, Any]) -> None:
        """Handles the strategy-change callback, closing positions if requested."""
        state = self._state
        logger.info("🔄 Strategy change requested: %s", swap_data)

        # If requested to close positions, close them now
        if swap_data.get("close_positions", False):
            logger.info("🚪 Closing all positions before strategy swap")
            for position in list(state.live_position_tracker.positions.values()):
                # Validate price before closing to prevent data corruption
                try:
                    current_price = state.data_provider.get_current_price(position.symbol)
                except Exception as exc:
                    logger.error(
                        "Cannot close position %s during strategy change - price fetch failed: %s. "
                        "Position will remain open.",
                        position.symbol,
                        exc,
                        exc_info=True,
                    )
                    continue
                if current_price is None or current_price <= 0:
                    logger.error(
                        "Cannot close position %s during strategy change - invalid price %s. "
                        "Position will remain open.",
                        position.symbol,
                        current_price,
                    )
                    continue

                state._execute_exit(
                    position,
                    "Strategy change - close requested",
                    None,
                    float(current_price),
                    None,
                    None,
                    None,
                )
        else:
            logger.info("📊 Keeping existing positions during strategy swap")

    def handle_model_update(self, update_data: dict[str, Any]) -> None:
        """Handles the model-update callback (application deferred to the loop)."""
        logger.info("🤖 Model update requested: %s", update_data)
        # Model update logic is handled in strategy_manager.apply_pending_update()

    def hot_swap_strategy(
        self, new_strategy_name: str, close_positions: bool = False, new_config: dict | None = None
    ) -> bool:
        """
        Hot-swap to a new strategy during live trading

        Args:
            new_strategy_name: Name of new strategy
            close_positions: Whether to close existing positions
            new_config: Configuration for new strategy

        Returns:
            True if swap was initiated successfully
        """
        state = self._state

        if not state.strategy_manager:
            logger.error("Strategy manager not initialized - hot swapping disabled")
            return False

        logger.info("🔄 Initiating hot-swap to strategy: %s", new_strategy_name)

        success = state.strategy_manager.hot_swap_strategy(
            new_strategy_name=new_strategy_name,
            new_config=new_config,
            close_existing_positions=close_positions,
        )

        if success:
            logger.info("✅ Hot-swap initiated successfully - will apply on next cycle")
            strategy_name = state._strategy_name()
            state._send_alert(f"Strategy hot-swap initiated: {strategy_name} → {new_strategy_name}")
        else:
            logger.error("❌ Hot-swap initiation failed")

        return success

    def update_model(self, new_model_path: str) -> bool:
        """
        Update ML models during live trading

        Args:
            new_model_path: Path to new model file

        Returns:
            True if update was initiated successfully
        """
        state = self._state

        if not state.strategy_manager:
            logger.error("Strategy manager not initialized - model updates disabled")
            return False

        strategy_name = state._strategy_name().lower()

        logger.info("🤖 Initiating model update for strategy: %s", strategy_name)

        success = state.strategy_manager.update_model(
            strategy_name=strategy_name, new_model_path=new_model_path, validate_model=True
        )

        if success:
            logger.info("✅ Model update initiated successfully - will apply on next cycle")
            state._send_alert(f"Model update initiated for {strategy_name}")
        else:
            logger.error("❌ Model update initiation failed")

        return success

    def apply_pending_strategy_update(self) -> bool:
        """Apply a queued strategy/model update from the StrategyManager.

        Drives the full hot-swap pipeline as observed by the run loop:
        ``strategy_manager.apply_pending_update()`` then engine-side reconfigure
        and refresh of all strategy-dependent state. Returns ``True`` on success.

        Extracted as a method so unit tests can drive the same code path the
        run loop uses.
        """
        state = self._state
        if not state.strategy_manager:
            return False

        success = state.strategy_manager.apply_pending_update()
        if not success:
            logger.error("❌ Failed to apply strategy/model update")
            return False

        state._finalize_runtime()
        # apply_pending_update() success guarantees a current strategy is set.
        updated_strategy = cast(ComponentStrategy, state.strategy_manager.current_strategy)
        state._configure_strategy(updated_strategy)
        state._runtime_dataset = None
        state._runtime_warmup = 0
        # Refresh strategy-dependent state so the new strategy's overrides take
        # effect on the very next decision (matches backtest _switch_strategy).
        self.refresh_strategy_dependencies()
        logger.info("✅ Strategy/model update applied successfully")
        return True

    def refresh_strategy_dependencies(self) -> None:
        """Refresh engine state derived from the active strategy.

        Called after a hot-swap (or model update) to re-bind the new component
        risk adapter to the engine's portfolio risk manager and rebuild engine-
        level policies (trailing stop, partial operations, time exits) from the
        new strategy's risk overrides. Without this, the live engine continues
        to use the previous strategy's risk plumbing until restart, silently
        diverging from a backtest-validated strategy.

        Mirrors the backtest equivalent in ``Backtester._switch_strategy``.
        """
        state = self._state
        component_strategy = getattr(state, "_component_strategy", None)
        component_risk = (
            getattr(component_strategy, "risk_manager", None)
            if component_strategy is not None
            else None
        )

        # 1. Re-bind component risk adapter to the engine's portfolio risk
        #    manager so position-tracking writes hit the canonical instance.
        if component_risk is not None and hasattr(component_risk, "bind_core_manager"):
            try:
                component_risk.bind_core_manager(state.risk_manager)
            except Exception as exc:
                logger.warning(
                    "Hot-swap: failed to re-bind core risk manager to component "
                    "strategy: %s. Component risk limits may not be enforced.",
                    exc,
                    exc_info=True,
                )

        # 2. Push the new strategy's overrides onto the component risk adapter.
        #    The factory typically already does this, but engines have
        #    historically also called it (see __init__ around L250) so the
        #    adapter is the single source of truth post-swap.
        new_overrides: dict[str, Any] = {}
        if component_strategy is not None and hasattr(component_strategy, "get_risk_overrides"):
            try:
                fetched = component_strategy.get_risk_overrides()
            except Exception as exc:
                logger.warning("Hot-swap: get_risk_overrides() failed: %s", exc, exc_info=True)
                fetched = None
            if isinstance(fetched, dict):
                new_overrides = dict(fetched)
        if component_risk is not None and hasattr(component_risk, "set_strategy_overrides"):
            adapter_overrides = getattr(component_risk, "_strategy_overrides", None)
            merged = dict(adapter_overrides) if isinstance(adapter_overrides, dict) else {}
            merged.update(new_overrides)
            try:
                component_risk.set_strategy_overrides(merged)
            except Exception as exc:
                logger.warning(
                    "Hot-swap: failed to propagate strategy overrides to "
                    "component risk manager: %s",
                    exc,
                    exc_info=True,
                )

        # 3. Rebuild engine-level trailing-stop policy from the new strategy /
        #    risk-manager and propagate it into the live exit handler so that
        #    the next trailing-stop tick uses the refreshed configuration.
        try:
            state.trailing_stop_policy = state._build_trailing_policy()
            state._trailing_stop_opt_in = state.trailing_stop_policy is not None
            exit_handler = getattr(state, "live_exit_handler", None)
            if exit_handler is not None:
                exit_handler.trailing_stop_policy = state.trailing_stop_policy
                trailing_manager = getattr(exit_handler, "_trailing_stop_manager", None)
                if trailing_manager is not None:
                    trailing_manager.policy = state.trailing_stop_policy
        except Exception as exc:
            logger.warning(
                "Hot-swap: failed to refresh trailing stop policy: %s",
                exc,
                exc_info=True,
            )

        # 4. Rebuild engine-level partial operations policy from new overrides.
        try:
            self.refresh_partial_manager_after_swap(new_overrides)
        except Exception as exc:
            logger.warning(
                "Hot-swap: failed to refresh partial operations manager: %s",
                exc,
                exc_info=True,
            )

        # 5. Rebuild engine-level time exit policy from new overrides.
        try:
            self.refresh_time_exit_policy_after_swap(new_overrides)
        except Exception as exc:
            logger.warning(
                "Hot-swap: failed to refresh time exit policy: %s",
                exc,
                exc_info=True,
            )

        # 5b. Rebuild engine-level MFE early-cut policy from the new
        #     strategy / risk-manager (shared builder, parity with backtest).
        try:
            from src.engines.shared.risk_configuration import build_early_cut_policy

            state.early_cut_policy = build_early_cut_policy(state.strategy, state.risk_manager)
            exit_handler = getattr(state, "live_exit_handler", None)
            if exit_handler is not None:
                exit_handler.early_cut_policy = state.early_cut_policy
        except Exception as exc:
            logger.warning(
                "Hot-swap: failed to refresh early-cut policy: %s",
                exc,
                exc_info=True,
            )

        # 6. If a correlation handler is wired on the entry handler, refresh
        #    its strategy reference (mirrors backtest engine.py:817).
        entry_handler = getattr(state, "live_entry_handler", None)
        if entry_handler is not None:
            correlation_handler = getattr(entry_handler, "correlation_handler", None)
            if correlation_handler is not None and hasattr(correlation_handler, "set_strategy"):
                try:
                    correlation_handler.set_strategy(state.strategy)
                except Exception as exc:
                    logger.warning(
                        "Hot-swap: correlation_handler.set_strategy failed: %s", exc, exc_info=True
                    )

        # 7. Defensive invariant guard: ConfidenceWeightedSizer enforces
        #    min_confidence_floor <= min_confidence at construction time, but
        #    log a critical signal here if it ever slips through (e.g. via a
        #    mutated sizer instance) so operators can intervene quickly.
        position_sizer = getattr(component_strategy, "position_sizer", None)
        if position_sizer is not None:
            min_conf = getattr(position_sizer, "min_confidence", None)
            min_floor = getattr(position_sizer, "min_confidence_floor", None)
            if min_conf is not None and min_floor is not None and min_floor > min_conf:
                logger.critical(
                    "Hot-swap invariant violation: min_confidence_floor (%s) > "
                    "min_confidence (%s) on new strategy sizer; live engine may "
                    "over-size low-confidence signals until next swap.",
                    min_floor,
                    min_conf,
                )

    def refresh_partial_manager_after_swap(
        self,
        new_overrides: dict[str, Any],
    ) -> None:
        """Rebuild engine-level partial_manager from new strategy overrides.

        Pushes the refreshed policy into the live exit handler's
        :class:`PartialOperationsManager` so that partial-exit / scale-in
        decisions on the next bar use the new configuration.
        """
        state = self._state
        new_policy: PartialExitPolicy | None = None
        # Re-read the live flag here (not the construction-time
        # ``state.settings.partial_operations_allowed``) on purpose:
        # ``live_partial_operations`` is a deliberately runtime-dynamic flag
        # (see config.py / #800) so a hot-swap re-checks the *current* value.
        if not is_enabled("live_partial_operations", False):
            # Same guard as __init__ (#734): partial ops are bookkeeping-only in
            # the live engine, so a hot-swapped strategy's partial_operations
            # overrides must not re-enable them.
            if isinstance(new_overrides, dict) and "partial_operations" in new_overrides:
                logger.warning(
                    "Ignoring partial_operations overrides from hot-swapped "
                    "strategy: partial exits/scale-ins are disabled in the live "
                    "engine (#734)."
                )
            state.partial_manager = None
            state._partial_operations_opt_in = False
            exit_handler = getattr(state, "live_exit_handler", None)
            if exit_handler is not None:
                exit_handler.partial_manager = None
            return
        partial_cfg = (
            new_overrides.get("partial_operations") if isinstance(new_overrides, dict) else None
        )
        if isinstance(partial_cfg, dict):
            new_policy = PartialExitPolicy(
                exit_targets=partial_cfg.get("exit_targets", []),
                exit_sizes=partial_cfg.get("exit_sizes", []),
                scale_in_thresholds=partial_cfg.get("scale_in_thresholds", []),
                scale_in_sizes=partial_cfg.get("scale_in_sizes", []),
                max_scale_ins=partial_cfg.get("max_scale_ins", 0),
            )
        elif state.enable_partial_operations:
            rp = state.risk_manager.params if state.risk_manager else RiskParameters()
            new_policy = PartialExitPolicy(
                exit_targets=rp.partial_exit_targets or [],
                exit_sizes=rp.partial_exit_sizes or [],
                scale_in_thresholds=rp.scale_in_thresholds or [],
                scale_in_sizes=rp.scale_in_sizes or [],
                max_scale_ins=rp.max_scale_ins,
            )

        state.partial_manager = new_policy
        state._partial_operations_opt_in = bool(
            state.enable_partial_operations or state.partial_manager is not None
        )

        exit_handler = getattr(state, "live_exit_handler", None)
        if exit_handler is None:
            return

        ops_manager = getattr(exit_handler, "partial_manager", None)
        if new_policy is None:
            # Disable partial operations on the exit handler.
            exit_handler.partial_manager = None
            return

        if ops_manager is None:
            exit_handler.partial_manager = PartialOperationsManager(policy=new_policy)
        elif hasattr(ops_manager, "set_policy"):
            ops_manager.set_policy(new_policy)
        else:
            ops_manager.policy = new_policy

    def refresh_time_exit_policy_after_swap(
        self,
        new_overrides: dict[str, Any],
    ) -> None:
        """Rebuild engine-level time_exit_policy from new strategy overrides."""
        state = self._state
        time_cfg = new_overrides.get("time_exits") if isinstance(new_overrides, dict) else None
        if not time_cfg and state.risk_manager and getattr(state.risk_manager, "params", None):
            time_cfg = getattr(state.risk_manager.params, "time_exits", None)

        new_policy: TimeExitPolicy | None = None
        if time_cfg:
            tr = time_cfg.get("time_restrictions") or DEFAULT_TIME_RESTRICTIONS
            restrictions = TimeRestrictions(
                no_overnight=bool(tr.get("no_overnight", False)),
                no_weekend=bool(tr.get("no_weekend", False)),
                trading_hours_only=bool(tr.get("trading_hours_only", False)),
            )
            new_policy = TimeExitPolicy(
                max_holding_hours=time_cfg.get("max_holding_hours", DEFAULT_MAX_HOLDING_HOURS),
                end_of_day_flat=time_cfg.get("end_of_day_flat", DEFAULT_END_OF_DAY_FLAT),
                weekend_flat=time_cfg.get("weekend_flat", DEFAULT_WEEKEND_FLAT),
                market_timezone=time_cfg.get("market_timezone", DEFAULT_MARKET_TIMEZONE),
                time_restrictions=restrictions,
            )

        state.time_exit_policy = new_policy
        exit_handler = getattr(state, "live_exit_handler", None)
        if exit_handler is not None:
            exit_handler.time_exit_policy = new_policy
