"""Adapter utilities exposing engine risk controls to component strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.risk.risk_manager import RiskManager as CoreRiskManager
from src.risk.risk_manager import RiskParameters

from .policies import (
    DynamicRiskDescriptor,
    PartialExitPolicyDescriptor,
    PolicyBundle,
    TrailingStopPolicyDescriptor,
)
from .risk_manager import MarketData, Position, RiskManager

if TYPE_CHECKING:
    from .regime_context import RegimeContext
    from .signal_generator import Signal


@dataclass
class PortfolioStateHooks:
    """Optional callbacks that let the adapter observe portfolio events."""

    on_fill: Callable | None = None
    on_close: Callable | None = None


class CoreRiskAdapter(RiskManager):
    """Wraps :class:`src.risk.risk_manager.RiskManager` for component usage."""

    def __init__(
        self,
        core_manager: CoreRiskManager | None = None,
        *,
        parameters: RiskParameters | None = None,
        dynamic_risk: DynamicRiskDescriptor | None = None,
    ) -> None:
        super().__init__("core_risk_adapter")
        self._core_manager = core_manager or CoreRiskManager(parameters)
        self._strategy_overrides: dict[str, Any] = {}
        self._dynamic_risk_descriptor = dynamic_risk
        self._hooks: PortfolioStateHooks | None = None

    # ------------------------------------------------------------------
    # Binding helpers
    # ------------------------------------------------------------------
    def bind_core_manager(self, manager: CoreRiskManager) -> None:
        """Attach the canonical engine risk manager instance."""

        self._core_manager = manager

    def set_strategy_overrides(self, overrides: dict[str, Any] | None) -> None:
        """Register default strategy overrides used for sizing/policies."""

        self._strategy_overrides = dict(overrides or {})

    def set_portfolio_hooks(self, hooks: PortfolioStateHooks | None) -> None:
        self._hooks = hooks

    # ------------------------------------------------------------------
    # Compatibility properties for legacy integrations
    # ------------------------------------------------------------------
    @property
    def risk_per_trade(self) -> float:
        manager = self._require_core_manager()
        params = getattr(manager, "params", None)
        if params is None:
            return 0.0
        return float(getattr(params, "base_risk_per_trade", 0.0))

    @risk_per_trade.setter
    def risk_per_trade(self, value: float) -> None:
        manager = self._require_core_manager()
        params = getattr(manager, "params", None)
        if params is not None:
            params.base_risk_per_trade = float(value)

    @property
    def stop_loss_pct(self) -> float | None:
        value = self._strategy_overrides.get("stop_loss_pct")
        return float(value) if value is not None else None

    @stop_loss_pct.setter
    def stop_loss_pct(self, value: float | None) -> None:
        if value is None:
            self._strategy_overrides.pop("stop_loss_pct", None)
            return
        self._strategy_overrides["stop_loss_pct"] = float(value)

    @property
    def take_profit_pct(self) -> float | None:
        override = self._strategy_overrides.get("take_profit_pct")
        if override is not None:
            return float(override)
        params = getattr(self._require_core_manager(), "params", None)
        if params is None or params.default_take_profit_pct is None:
            return None
        return float(params.default_take_profit_pct)

    @take_profit_pct.setter
    def take_profit_pct(self, value: float | None) -> None:
        manager = self._require_core_manager()
        params = getattr(manager, "params", None)
        if value is None:
            self._strategy_overrides.pop("take_profit_pct", None)
            if params is not None:
                params.default_take_profit_pct = None
            return
        self._strategy_overrides["take_profit_pct"] = float(value)
        if params is not None:
            params.default_take_profit_pct = float(value)

    # ------------------------------------------------------------------
    # RiskManager interface
    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        manager = self._require_core_manager()
        df = context.get("df")
        index = context.get("index")
        if df is None or index is None:
            return 0.0

        overrides = self._resolve_overrides(context)
        fraction = manager.calculate_position_fraction(
            df=df,
            index=int(index),
            balance=balance,
            price=context.get("price"),
            indicators=context.get("indicators"),
            strategy_overrides=overrides,
            regime=self._normalise_regime(regime),
            correlation_ctx=context.get("correlation_ctx"),
        )
        fraction = max(0.0, min(1.0, float(fraction)))
        return balance * fraction

    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> bool:
        manager = self._require_core_manager()
        params = getattr(manager, "params", RiskParameters())

        price = float(current_data.price)
        entry_price = float(position.entry_price)
        if price <= 0 or entry_price <= 0:
            return False

        if position.side == "long":
            pnl_pct = (price - entry_price) / entry_price
            threshold = -float(params.max_risk_per_trade)
            return pnl_pct <= threshold
        if position.side == "short":
            pnl_pct = (entry_price - price) / entry_price
            threshold = -float(params.max_risk_per_trade)
            return pnl_pct <= threshold
        return False

    def get_stop_loss(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        manager = self._require_core_manager()
        df = context.get("df")
        index = context.get("index")
        overrides = self._resolve_overrides(context)
        if df is None or index is None:
            stop_pct = overrides.get("stop_loss_pct")
            if stop_pct is None:
                return entry_price
            if self._signal_side(signal) == "long":
                return entry_price * (1 - float(stop_pct))
            return entry_price * (1 + float(stop_pct))

        side = self._signal_side(signal)
        stop, _ = manager.compute_sl_tp(
            df=df,
            index=int(index),
            entry_price=entry_price,
            side=side,
            strategy_overrides=overrides,
        )
        return float(stop) if stop is not None else entry_price

    def get_take_profit(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        manager = self._require_core_manager()
        df = context.get("df")
        index = context.get("index")
        overrides = self._resolve_overrides(context)
        if df is None or index is None:
            take_profit_pct = overrides.get("take_profit_pct")
            if take_profit_pct is None:
                params = getattr(manager, "params", None)
                take_profit_pct = params.default_take_profit_pct if params else None
            if take_profit_pct is None:
                return entry_price
            if self._signal_side(signal) == "long":
                return entry_price * (1 + float(take_profit_pct))
            return entry_price * (1 - float(take_profit_pct))

        side = self._signal_side(signal)
        _, take_profit = manager.compute_sl_tp(
            df=df,
            index=int(index),
            entry_price=entry_price,
            side=side,
            strategy_overrides=overrides,
        )
        return float(take_profit) if take_profit is not None else entry_price

    def get_position_policies(
        self,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> PolicyBundle | None:
        partial = self._build_partial_descriptor()
        trailing = self._build_trailing_descriptor()
        dynamic = self._dynamic_risk_descriptor
        bundle = PolicyBundle(partial_exit=partial, trailing_stop=trailing, dynamic_risk=dynamic)
        return None if bundle.is_empty() else bundle

    # ------------------------------------------------------------------
    # Portfolio synchronisation helpers
    # ------------------------------------------------------------------
    def on_fill(self, symbol: str, side: str, fraction: float, entry_price: float) -> None:
        manager = self._require_core_manager()
        manager.update_position(symbol, side, fraction, entry_price)
        if self._hooks and self._hooks.on_fill:
            self._hooks.on_fill(symbol, side, fraction, entry_price)

    def on_close(self, symbol: str) -> None:
        manager = self._require_core_manager()
        manager.close_position(symbol)
        if self._hooks and self._hooks.on_close:
            self._hooks.on_close(symbol)

    def on_partial_exit(self, symbol: str, executed_fraction_of_original: float) -> None:
        manager = self._require_core_manager()
        manager.adjust_position_after_partial_exit(symbol, executed_fraction_of_original)

    def on_scale_in(self, symbol: str, added_fraction_of_original: float) -> None:
        manager = self._require_core_manager()
        manager.adjust_position_after_scale_in(symbol, added_fraction_of_original)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_core_manager(self) -> CoreRiskManager:
        if self._core_manager is None:
            raise RuntimeError("CoreRiskAdapter is not bound to an engine RiskManager")
        return self._core_manager

    def _resolve_overrides(self, context: dict[str, Any]) -> dict[str, Any]:
        overrides = dict(self._strategy_overrides)
        ctx_overrides = context.get("strategy_overrides")
        if isinstance(ctx_overrides, dict):
            overrides.update(ctx_overrides)
        return overrides

    @staticmethod
    def _normalise_regime(regime: RegimeContext | None) -> str:
        if regime is None:
            return "normal"
        trend = getattr(regime, "trend", None)
        if hasattr(trend, "value"):
            return str(trend.value)
        if isinstance(regime, str):
            return regime
        return "normal"

    @staticmethod
    def _signal_side(signal: Signal) -> str:
        direction = getattr(signal.direction, "value", getattr(signal, "direction", "hold"))
        if direction == "buy":
            return "long"
        if direction == "sell":
            return "short"
        return "long"

    def _build_partial_descriptor(self) -> PartialExitPolicyDescriptor | None:
        params = getattr(self._require_core_manager(), "params", None)
        if params is None:
            return None
        if not params.partial_exit_targets or not params.partial_exit_sizes:
            return None
        return PartialExitPolicyDescriptor(
            exit_targets=list(params.partial_exit_targets),
            exit_sizes=list(params.partial_exit_sizes),
            scale_in_thresholds=list(params.scale_in_thresholds or []),
            scale_in_sizes=list(params.scale_in_sizes or []),
            max_scale_ins=int(params.max_scale_ins),
        )

    def _build_trailing_descriptor(self) -> TrailingStopPolicyDescriptor | None:
        params = getattr(self._require_core_manager(), "params", None)
        if params is None:
            return None
        if params.trailing_activation_threshold is None or (
            params.trailing_distance_pct is None and params.trailing_atr_multiplier is None
        ):
            return None
        return TrailingStopPolicyDescriptor(
            activation_threshold=float(params.trailing_activation_threshold),
            trailing_distance_pct=(
                float(params.trailing_distance_pct)
                if params.trailing_distance_pct is not None
                else None
            ),
            atr_multiplier=(
                float(params.trailing_atr_multiplier)
                if params.trailing_atr_multiplier is not None
                else None
            ),
            breakeven_threshold=(
                float(params.breakeven_threshold)
                if params.breakeven_threshold is not None
                else None
            ),
            breakeven_buffer=float(params.breakeven_buffer),
        )


__all__ = ["CoreRiskAdapter", "PortfolioStateHooks"]
