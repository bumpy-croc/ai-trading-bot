"""Strategy-runtime coordination for the live trading engine.

Owns the engine's strategy normalization and per-candle runtime decision
pipeline: configuring raw ``ComponentStrategy`` vs wrapped ``StrategyRuntime``
inputs, the component risk-context provider (correlation hydration), runtime
dataframe preparation, building the ``RuntimeContext`` from live positions,
processing a runtime decision, and the risk-parameter merge/clone helpers used
at construction. Extracted from ``LiveTradingEngine`` (#486) so the engine
orchestrates while this coordinator owns strategy↔runtime translation.

Thread-safety / lock ownership: the coordinator holds no locks and owns no
mutable state of its own. It reads and writes the engine's strategy-runtime
attributes (``strategy``, ``_component_strategy``, ``_runtime``,
``_runtime_dataset``, ``_runtime_warmup``, and the component risk-context
cache) at call time through a narrow ``Protocol``. All of these are touched
only on the single trading-loop thread (per-candle processing and the
loop-applied hot-swap), exactly as the original inline code did.
``_component_dynamic_risk_config`` is also *written* here on that thread but is
*read* elsewhere for engine-level state reporting.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import pandas as pd

from src.engines.live.execution.position_tracker import LivePosition as Position
from src.engines.shared.models import PositionSide
from src.engines.shared.policy_hydration import apply_policies_to_engine
from src.risk.risk_manager import RiskParameters
from src.strategies.components import Position as ComponentPosition
from src.strategies.components import RuntimeContext, SignalDirection, StrategyRuntime
from src.strategies.components import Strategy as ComponentStrategy

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    from src.engines.live.execution.entry_handler import LiveEntryHandler
    from src.engines.live.execution.position_tracker import LivePositionTracker
    from src.position_management.correlation_engine import CorrelationEngine
    from src.strategies.components.strategy import TradingDecision

logger = logging.getLogger(__name__)


class StrategyRuntimeEngineState(Protocol):
    """Engine state the coordinator reads and mutates at call time.

    Attributes are accessed dynamically (not captured at construction) because
    ``_configure_strategy`` runs early in ``__init__`` — before the entry
    handler exists — and again on every hot-swap, reassigning ``strategy`` /
    ``_component_strategy`` / ``_runtime``.
    """

    strategy: Any
    _component_strategy: ComponentStrategy | None
    _runtime: StrategyRuntime | None
    _runtime_dataset: Any
    _runtime_warmup: int
    _component_risk_context_cache_key: tuple[str, int] | None
    _component_risk_context_cache: dict[str, Any] | None
    _component_dynamic_risk_config: Any
    _active_symbol: str | None
    current_balance: float
    correlation_engine: CorrelationEngine | None
    live_position_tracker: LivePositionTracker
    db_manager: DatabaseManager
    # Created mid-__init__ (after configure_strategy's first call), so it may be
    # ABSENT during early construction; configure_strategy access is hasattr-guarded.
    live_entry_handler: LiveEntryHandler

    def _build_correlation_context(
        self, symbol: str, df: pd.DataFrame, overrides: dict | None
    ) -> dict | None: ...


class StrategyRuntimeCoordinator:
    """Strategy configuration and per-candle runtime decision pipeline."""

    def __init__(self, engine_state: StrategyRuntimeEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    # Risk-parameter helpers ------------------------------------------------------

    def extract_component_risk_parameters(
        self, component_risk_manager: object
    ) -> RiskParameters | None:
        """Clone risk parameters from a component adapter, if available."""

        if component_risk_manager is None:
            return None

        core_manager = getattr(component_risk_manager, "_core_manager", None)
        if core_manager is None:
            return None

        params = getattr(core_manager, "params", None)
        if not isinstance(params, RiskParameters):
            return None

        return self.clone_risk_parameters(params)

    def merge_risk_parameters(
        self,
        engine_params: RiskParameters | None,
        component_params: RiskParameters | None,
    ) -> RiskParameters | None:
        """Merges engine-provided and component-provided risk parameters.

        Component parameters take precedence over engine parameters when both
        are provided. Non-None component values override engine values.

        Args:
            engine_params: Risk parameters from the trading engine
            component_params: Risk parameters from the strategy component

        Returns:
            Merged risk parameters, or None if both inputs are None
        """

        if engine_params is None and component_params is None:
            return None

        if component_params is None:
            return self.clone_risk_parameters(engine_params)

        if engine_params is None:
            return component_params

        component_dict = asdict(component_params)
        engine_dict = asdict(engine_params)
        default_dict = asdict(RiskParameters())

        merged = dict(component_dict)
        for key, value in engine_dict.items():
            default_value = default_dict.get(key)

            # Preserve component overrides when the engine sticks with defaults.
            if value == default_value:
                continue

            merged[key] = value

        return RiskParameters(**merged)

    @staticmethod
    def clone_risk_parameters(params: RiskParameters | None) -> RiskParameters | None:
        """Creates a deep-cloned copy of risk parameters for safe reuse.

        Args:
            params: Risk parameters to clone

        Returns:
            Deep copy of the risk parameters, or None if input is None
        """

        if params is None:
            return None

        return RiskParameters(**asdict(params))

    # Strategy configuration & context provider -----------------------------------

    def configure_strategy(self, strategy: ComponentStrategy | StrategyRuntime) -> None:
        """Normalizes strategy inputs and configures runtime bookkeeping.

        Handles both raw ComponentStrategy instances and wrapped StrategyRuntime
        instances, extracting the underlying strategy and setting up engine state.

        Args:
            strategy: Strategy instance to configure (raw or wrapped)
        """
        state = self._state

        runtime = strategy if isinstance(strategy, StrategyRuntime) else None
        base_strategy = runtime.strategy if runtime is not None else strategy

        previous_component = getattr(state, "_component_strategy", None)

        state.strategy = base_strategy
        state._component_strategy = (
            base_strategy if isinstance(base_strategy, ComponentStrategy) else None
        )

        if (
            previous_component is not None
            and previous_component is not state._component_strategy
            and hasattr(previous_component, "set_additional_risk_context_provider")
        ):
            try:
                previous_component.set_additional_risk_context_provider(None)
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to clear risk context provider on previous strategy: %s", exc)

        if runtime is not None:
            state._runtime = runtime
        elif state._component_strategy is not None:
            state._runtime = StrategyRuntime(state._component_strategy)
        else:
            state._runtime = None

        self.register_component_context_provider()

        if hasattr(state, "live_entry_handler"):
            state.live_entry_handler.set_component_strategy(state._component_strategy)

    def register_component_context_provider(self) -> None:
        """Attaches the engine-provided risk context hook to component strategies.

        Registers a callback function that allows component strategies to request
        additional risk context (correlation data, etc.) during decision-making.
        """
        state = self._state

        strategy = getattr(state, "_component_strategy", None)
        if strategy is None:
            return

        setter = getattr(strategy, "set_additional_risk_context_provider", None)
        if not callable(setter):
            return

        def provider(df: pd.DataFrame, index: int, signal) -> dict[str, Any] | None:
            return self.component_risk_context(df, index, signal)

        try:
            setter(provider)
        except (TypeError, AttributeError) as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to attach risk context provider to component strategy: %s", exc)

    def component_risk_context(self, df: pd.DataFrame, index: int, signal) -> dict[str, Any]:
        """Build supplemental risk context (e.g., correlation data) for components."""
        state = self._state

        strategy = getattr(state, "_component_strategy", None)
        if strategy is None:
            return {}

        if getattr(state, "correlation_engine", None) is None:
            return {}

        symbol = state._active_symbol or getattr(strategy, "trading_pair", None)
        if not symbol:
            state._component_risk_context_cache_key = None
            state._component_risk_context_cache = None
            return {}

        cache_key = (str(symbol), int(index))
        cached_key = getattr(state, "_component_risk_context_cache_key", None)
        if cached_key != cache_key:
            state._component_risk_context_cache_key = None
            state._component_risk_context_cache = None

        overrides = None
        if hasattr(strategy, "get_risk_overrides"):
            try:
                overrides = strategy.get_risk_overrides()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "Failed to fetch component risk overrides for correlation context: %s",
                    exc,
                )

        # Only build correlation context when sizing a potential entry to avoid repeated
        # historical price lookups on every candle. Reuse the cached value if the same bar
        # has already triggered sizing.
        try:
            direction = getattr(signal, "direction", None)
        except Exception:
            direction = None

        if direction == SignalDirection.HOLD:
            return {}

        correlation_ctx = self.get_correlation_context(
            str(symbol),
            df,
            overrides,
            index=index,
        )
        if not correlation_ctx:
            return {}

        return {"correlation_ctx": correlation_ctx}

    def get_correlation_context(
        self,
        symbol: str,
        df: pd.DataFrame,
        overrides: dict | None,
        *,
        index: int | None = None,
    ) -> dict | None:
        """Return cached correlation context for the given bar or build it on demand."""
        state = self._state

        cache_key = (symbol, index) if index is not None else None
        cached_key = getattr(state, "_component_risk_context_cache_key", None)
        cached_ctx = getattr(state, "_component_risk_context_cache", None)
        if cache_key is not None and cache_key == cached_key and cached_ctx is not None:
            return cached_ctx

        context = state._build_correlation_context(symbol, df, overrides)
        if cache_key is not None:
            if context:
                state._component_risk_context_cache_key = cache_key
                state._component_risk_context_cache = context
            else:
                if cache_key == getattr(state, "_component_risk_context_cache_key", None):
                    state._component_risk_context_cache_key = None
                    state._component_risk_context_cache = None
        return context

    def apply_policies_from_decision(self, decision) -> None:
        """Hydrate engine-level policies from component strategy output.

        Uses shared policy hydration logic for consistency with backtest engine.
        """
        state = self._state
        # Use shared policy hydration logic
        apply_policies_to_engine(decision, state, state.db_manager)

        # Cache dynamic risk config for live engine state tracking
        if decision is not None:
            bundle = getattr(decision, "policies", None)
            if bundle:
                try:
                    dynamic_descriptor = getattr(bundle, "dynamic_risk", None)
                    if dynamic_descriptor is not None:
                        state._component_dynamic_risk_config = dynamic_descriptor.to_config()
                except Exception as exc:
                    # Benign: shared hydration above already applied the policy;
                    # only the live-engine state cache misses this descriptor.
                    # Debug level: runs per decision in the trading loop.
                    logger.debug("Dynamic risk config cache update skipped: %s", exc)

    # Runtime integration helpers -------------------------------------------------

    def is_runtime_strategy(self) -> bool:
        return self._state._runtime is not None

    def strategy_name(self) -> str:
        """Returns the configured strategy name for logging and reporting.

        Returns:
            Strategy name, or "UnknownStrategy" if no strategy is configured
        """
        strategy = getattr(self._state, "strategy", None)
        if strategy is None:
            return "UnknownStrategy"
        return getattr(strategy, "name", strategy.__class__.__name__)

    def prepare_strategy_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares dataframe for strategy processing.

        Component-based strategies compute indicators on-demand in process_candle(),
        so the dataframe is returned as-is. Legacy strategies would need upfront
        indicator calculation (not currently used).

        Args:
            df: Raw market data dataframe

        Returns:
            Prepared dataframe ready for strategy processing
        """
        state = self._state
        if not self.is_runtime_strategy():
            # Component-based strategies don't need upfront indicator calculation
            # They compute indicators on-demand in process_candle()
            return df

        # is_runtime_strategy() guard above ensures _runtime is not None.
        dataset = cast(StrategyRuntime, state._runtime).prepare_data(df)
        state._runtime_dataset = dataset
        state._runtime_warmup = max(0, int(dataset.warmup_period or 0))
        return dataset.data

    def build_component_positions(
        self,
        current_price: float,
    ) -> list[ComponentPosition]:
        """Translate live positions into the strategy-side ComponentPosition list.

        Used by both the StrategyRuntime context and the direct
        ComponentStrategy.process_candle call so a strategy that consults
        ``current_positions`` (e.g. for anti-pyramiding or correlation-aware
        sizing) gets the same view in both code paths — and matching the
        backtest path where positions always flow through.
        """
        state = self._state
        positions: list[ComponentPosition] = []
        for position in state.live_position_tracker.positions.values():
            try:
                quantity = self.compute_component_quantity(position)
                component_position = ComponentPosition(
                    symbol=position.symbol,
                    # __post_init__ guarantees side is a PositionSide enum.
                    side=cast(PositionSide, position.side).value,
                    size=quantity,
                    entry_price=float(position.entry_price),
                    current_price=float(current_price),
                    entry_time=position.entry_time,
                )
                positions.append(component_position)
            except Exception as exc:
                logger.debug("Failed to translate live position for runtime: %s", exc)
        return positions

    def build_runtime_context(
        self,
        balance: float,
        current_price: float,
        current_time: datetime,
    ) -> RuntimeContext:
        """Build the StrategyRuntime context with current balance and live positions.

        Delegates the position-translation step to ``build_component_positions``
        so the runtime path and the direct ``ComponentStrategy.process_candle``
        path present an identical position view to the strategy — matching the
        backtest engine's runtime context construction.
        """
        positions = self.build_component_positions(current_price)
        return RuntimeContext(balance=float(balance), current_positions=positions or None)

    def compute_component_quantity(
        self, position: Position, balance_basis: float | None = None
    ) -> float:
        """Translate a position's fractional size into asset quantity for component strategies."""
        entry_price = float(position.entry_price)
        if entry_price <= 0:
            return 0.0

        basis = (
            balance_basis if balance_basis is not None else getattr(position, "entry_balance", None)
        )
        if basis is None or basis <= 0:
            basis = self._state.current_balance

        size_fraction = float(
            position.current_size if position.current_size is not None else position.size
        )
        return (size_fraction * float(basis)) / entry_price

    def runtime_process_decision(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        current_price: float,
        current_time: datetime,
    ) -> TradingDecision | None:
        state = self._state
        if not self.is_runtime_strategy():
            return None
        if state._runtime_dataset is None:
            return None
        if index < state._runtime_warmup:
            return None

        context = self.build_runtime_context(balance, current_price, current_time)
        try:
            # is_runtime_strategy() guard above ensures _runtime is not None.
            decision = cast(StrategyRuntime, state._runtime).process(index, context)
            self.apply_policies_from_decision(decision)
            return decision
        except (ValueError, KeyError, IndexError, AttributeError) as exc:
            logger.warning("Runtime decision failed in live engine at index %s: %s", index, exc)
            return None

    def finalize_runtime(self) -> None:
        state = self._state
        if self.is_runtime_strategy():
            try:
                # is_runtime_strategy() guard ensures _runtime is not None.
                cast(StrategyRuntime, state._runtime).finalize()
            finally:
                state._runtime_dataset = None
                state._runtime_warmup = 0
