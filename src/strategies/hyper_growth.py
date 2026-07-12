"""
Hyper Growth Strategy - Aggressive Component-Based Implementation

Targets high annual returns by combining three mechanisms from research:
1. ML-driven signal generation using the basic (price-only) model for
   directional alpha
2. High base position sizing (25% of balance per trade)
3. Aggressive risk overrides: tight stops (10%), wide drawdown tolerance,
   partial exits, trailing stops

NOTE: Leverage is DISABLED by default (max_leverage=1.0). Backtesting showed
that enabling leverage (e.g. 3x) hurt overall returns by -32% due to amplified
losses during volatile regime transitions. The LeverageManager infrastructure
and _HYPER_LEVERAGE_MAP remain in place for future experimentation — callers
can pass max_leverage > 1.0 to re-enable it.

Key architectural decision: ML models produce very low raw confidence values
(0.01-0.05 per bar) even when their directional accuracy is meaningful.
Standard risk managers and sizers multiply by confidence, crushing positions
to $10. This strategy uses a FlatRiskManager that returns a fixed fraction
of balance regardless of confidence — the ML direction filter IS the edge,
not the per-bar confidence score.

Model choice: the signal generator uses model_type="basic" (not "sentiment").
MLBasicSignalGenerator installs a 5-column PriceOnlyFeatureExtractor; the
sentiment bundle was trained on 10 features and silently returns 0.0 when
fed price-only inputs, which the generator converts to predicted_return=-1.0
and emits as a constant SELL sentinel. See
.claude/reports/hyper_growth_experiment_sweep_2026-04-17.md.

Reference: docs/research/500_percent_annual_returns.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.strategies.components import (
    EnhancedRegimeDetector,
    FixedFractionSizer,
    MLBasicSignalGenerator,
    MomentumSignalGenerator,
    SignalGenerator,
    Strategy,
)
from src.strategies.components.leverage_manager import LeverageManager
from src.strategies.components.position_sizer import LeveragedPositionSizer
from src.strategies.components.regime_context import TrendLabel, VolLabel
from src.strategies.components.risk_manager import RiskManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.strategies.components.regime_context import RegimeContext
    from src.strategies.components.signal_generator import Signal


class FlatRiskManager(RiskManager):
    """Risk manager that returns a fixed fraction of balance without
    scaling by signal confidence or strength.

    Standard risk managers multiply by confidence (0.01-0.05 for ML signals),
    reducing $1000 risk to $10. This manager trusts the ML direction filter
    and returns the full risk amount, letting the position sizer and leverage
    manager control sizing.
    """

    # Experimentation-runner contract: overrides listed here are honored
    # because this manager reads ``self.<attr>`` directly at trade time.
    # The runner consults this set to decide whether a setattr-based
    # override (as opposed to a ``_strategy_overrides`` dict override) is
    # sufficient for this manager class.
    _direct_runtime_overrides: frozenset[str] = frozenset({"stop_loss_pct", "min_confidence"})

    def __init__(
        self,
        risk_fraction: float = 0.10,
        stop_loss_pct: float = 0.10,
        min_confidence: float = 0.05,
    ):
        """Initialize flat risk manager.

        Args:
            risk_fraction: Fraction of balance to risk per trade (0.10 = 10%).
            stop_loss_pct: Stop loss percentage for exit decisions.
            min_confidence: Minimum signal confidence to allow a trade.
                ML signals below this are filtered out as noise.
        """
        super().__init__("flat_risk_manager")
        if not 0.01 <= risk_fraction <= 0.50:
            raise ValueError(f"risk_fraction must be between 0.01 and 0.50, got {risk_fraction}")
        self.risk_fraction = risk_fraction
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence

    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        """Return balance * risk_fraction without confidence/strength scaling.

        Uses a minimum confidence threshold to filter out low-quality signals.
        ML models produce 0.01-0.05 confidence on most bars but 0.10+ on
        higher-conviction signals that historically have 60%+ accuracy.
        """
        self.validate_inputs(balance)
        if signal.direction.value == "hold":
            return 0.0
        # Filter out very low confidence signals — below this threshold,
        # the ML model is essentially guessing
        if signal.confidence < self.min_confidence:
            return 0.0
        # Flat risk — no further confidence/strength scaling
        return balance * self.risk_fraction

    def should_exit(
        self,
        position: Any,
        current_data: Any,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> bool:
        """Exit when unrealized loss exceeds stop_loss_pct."""
        if not hasattr(position, "unrealized_pnl_pct"):
            return False
        return position.unrealized_pnl_pct <= -self.stop_loss_pct

    def get_stop_loss(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        """Calculate stop loss based on fixed percentage."""
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if signal.direction.value == "buy":
            return entry_price * (1 - self.stop_loss_pct)
        if signal.direction.value == "sell":
            return entry_price * (1 + self.stop_loss_pct)
        return entry_price

    def get_parameters(self) -> dict[str, float]:
        """Return risk manager parameters."""
        return {
            "risk_fraction": self.risk_fraction,
            "stop_loss_pct": self.stop_loss_pct,
        }


# Aggressive leverage map: amplify in bulls, protect in bears.
# Keys are (TrendLabel, VolLabel) tuples matching LeverageManager.get_leverage_multiplier() lookups.
# NOTE: These raw multipliers are clamped by LeverageManager's max_leverage cap.
# With the default max_leverage=1.0 none of these values take effect — they exist
# so that callers who pass max_leverage > 1.0 get regime-aware scaling automatically.
_HYPER_LEVERAGE_MAP: dict[tuple[TrendLabel, VolLabel], float] = {
    (TrendLabel.TREND_UP, VolLabel.LOW): 3.0,  # Full leverage in confirmed bull + low vol
    (TrendLabel.TREND_UP, VolLabel.HIGH): 2.0,  # Moderate leverage in bull + high vol
    (TrendLabel.RANGE, VolLabel.LOW): 1.5,  # Slight leverage in confirmed range + low vol
    (TrendLabel.RANGE, VolLabel.HIGH): 1.0,  # No leverage in range + high vol
    (TrendLabel.TREND_DOWN, VolLabel.LOW): 0.5,  # Minimal exposure in bear + low vol
    (TrendLabel.TREND_DOWN, VolLabel.HIGH): 0.0,  # Cash in confirmed bear + high vol
}


def create_hyper_growth_strategy(
    name: str = "HyperGrowth",
    signal_source: str = "ml",
    # 0.25 keeps realized risk/trade at or below 2%: live confidence scaling
    # lands realized notional at 0.46-0.80 of base_fraction (~11-20% of
    # balance), and the 10% stop bounds the loss at ~1.1-2.0% per trade.
    risk_fraction: float = 0.25,
    base_fraction: float = 0.25,
    min_confidence: float = 0.05,
    max_leverage: float = 1.0,
    leverage_decay_rate: float = 0.20,
    min_regime_bars: int = 3,
    take_profit_pct: float = 0.30,
    stop_loss_pct: float = 0.10,
    symbol: str | None = None,
    enable_trailing_stop: bool = True,
    trailing_activation_threshold: float = 0.03,
    trailing_distance_pct: float = 0.015,
    breakeven_threshold: float | None = 0.05,
    breakeven_buffer: float = 0.008,
    early_cut_mfe_threshold_pct: float | None = None,
    early_cut_evaluation_window_hours: float | None = None,
    model_version: str | None = None,
) -> Strategy:
    """Create hyper-growth strategy targeting high annual returns.

    Combines ML signal generation with aggressive position sizing to
    maximize compounding during favorable conditions while preserving
    capital during drawdowns.

    Leverage is DISABLED by default (max_leverage=1.0) because testing
    showed it reduces returns by -32%. The LeverageManager infrastructure
    is wired up so callers can pass max_leverage > 1.0 to experiment,
    but the default configuration runs without leverage amplification.

    Args:
        name: Strategy name.
        signal_source: "ml" for ML predictions, "momentum" for breakouts.
        risk_fraction: Fraction of balance the risk manager allocates per trade.
        base_fraction: Base position size as fraction of balance.
        min_confidence: Minimum signal confidence to allow a trade.
        max_leverage: Maximum leverage multiplier safety cap (default 1.0 =
            leverage disabled). Set > 1.0 to re-enable regime-based leverage.
        leverage_decay_rate: Exponential decay for smooth transitions.
        min_regime_bars: Bars before conviction scaling begins.
        take_profit_pct: Target profit percentage for scaling out.
        stop_loss_pct: Stop loss percentage (default 0.10 — tighter stops
            outperform 0.20 on BTCUSDT 2024 by ~50 pp of annualized return
            because the trailing stop and partial exits capture upside while
            tight SL caps short-side losses in the 2024 uptrend).
        symbol: Trading symbol threaded to the ML signal generator for model
            registry selection. Runners must pass the symbol they trade;
            None keeps the generator default (BTCUSDT).
        enable_trailing_stop: When False the trailing-stop/breakeven override
            is declared disabled — build_trailing_stop_policy returns None
            instead of falling back to RiskParameters defaults.
        trailing_activation_threshold: Position gain (decimal) at which the
            trailing stop activates.
        trailing_distance_pct: Trailing distance as a fraction of price.
        breakeven_threshold: Position gain (decimal) at which the stop moves
            to breakeven; None disables the breakeven move.
        breakeven_buffer: Buffer above entry (long) / below entry (short)
            for the breakeven stop.
        early_cut_mfe_threshold_pct: MFE-conditioned early-cut: raw price MFE
            (decimal) the position must reach inside the evaluation window,
            else it is flattened. Default None = early cut OFF. Must be set
            together with ``early_cut_evaluation_window_hours``.
        early_cut_evaluation_window_hours: Early-cut evaluation window in
            hours from entry. Must be set together with
            ``early_cut_mfe_threshold_pct``.
        model_version: Pin ML predictions to this exact registry version
            instead of resolving ``latest`` — the backtest harness's
            point-in-time pin (GH #988). Only valid with
            ``signal_source="ml"`` (momentum runs no model).

    Returns:
        Configured Strategy instance.

    Raises:
        ValueError: If exactly one of the two early-cut kwargs is provided,
            or if ``model_version`` is combined with ``signal_source="momentum"``.
    """
    if (early_cut_mfe_threshold_pct is None) != (early_cut_evaluation_window_hours is None):
        raise ValueError(
            "early_cut_mfe_threshold_pct and early_cut_evaluation_window_hours "
            "must be provided together (both set to enable the early cut, "
            "both None to disable it)"
        )
    if signal_source == "momentum" and model_version is not None:
        raise ValueError(
            "model_version cannot be combined with signal_source='momentum': "
            "the momentum generator runs no ML model, so a pinned model "
            "version would silently not be honored."
        )
    # #805: hyper_growth targets high returns via leverage/aggressive sizing and
    # is NOT recommended in a bear/high-vol regime, where exposure is the primary
    # risk. Prefer ml_adaptive with the exposure governor (#802) + vol-targeted
    # sizing (#805) when trading a downtrend.
    logger.warning(
        "hyper_growth is NOT recommended for bear/high-vol regimes: its leveraged, "
        "aggressive sizing amplifies drawdowns. Consider ml_adaptive with the "
        "exposure governor + vol-target sizing instead."
    )
    # Signal generator (declared up-front: branches assign different subtypes)
    signal_generator: SignalGenerator
    if signal_source == "momentum":
        signal_generator = MomentumSignalGenerator(
            name=f"{name}_signals",
            momentum_entry_threshold=0.001,  # 0.1% — very sensitive
            strong_momentum_threshold=0.005,  # 0.5%
        )
    else:
        # ML signals — the basic model predicts direction with a measurable
        # edge (55-57% BUY accuracy at 12-24h horizons) and per-bar confidence
        # of 0.01-0.10. The sentiment model is intentionally NOT used here:
        # MLBasicSignalGenerator installs a 5-column PriceOnlyFeatureExtractor
        # while the sentiment bundle expects 10 features (incl.
        # sentiment_momentum_scaled), so feeding it the price-only tensor
        # silently returns 0.0 on every bar — which the generator converts to
        # predicted_return = -1.0 (a constant SELL sentinel, not a prediction).
        signal_generator = MLBasicSignalGenerator(
            name=f"{name}_signals", model_type="basic", symbol=symbol, model_version=model_version
        )

    risk_manager = FlatRiskManager(
        risk_fraction=risk_fraction,
        stop_loss_pct=stop_loss_pct,
        min_confidence=min_confidence,
    )

    base_sizer = FixedFractionSizer(
        fraction=base_fraction,
        adjust_for_confidence=False,
        adjust_for_strength=False,
    )

    regime_detector = EnhancedRegimeDetector()

    leverage_manager = LeverageManager(
        max_leverage=max_leverage,
        decay_rate=leverage_decay_rate,
        min_regime_bars=min_regime_bars,
        leverage_map=_HYPER_LEVERAGE_MAP,
    )

    position_sizer = LeveragedPositionSizer(
        base_sizer=base_sizer,
        leverage_manager=leverage_manager,
        max_leveraged_fraction=0.50,
    )

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    strategy.leverage_manager = leverage_manager
    strategy.base_position_size = base_fraction
    strategy.take_profit_pct = take_profit_pct
    # Override default 25% cap to match max_leveraged_fraction (50%),
    # otherwise _validate_position_size() clamps leveraged positions to 25%
    strategy._max_position_pct = 0.50

    # Hold positions through signal flips — only exit on SL/TP/trailing stop.
    # ML signals flip direction every bar, but the trend persists for days.
    # Without this, positions are exited after 1 bar with 0.03% P&L.
    strategy._extra_metadata = {"ignore_signal_reversal": True}

    # Trailing/breakeven config: defaults keep the exact pre-#971 dict shape
    # (key-presence matters — build_trailing_stop_policy falls back to
    # RiskParameters for absent keys, so adding keys would change behavior).
    trailing_stop_config: dict[str, Any]
    if enable_trailing_stop:
        trailing_stop_config = {
            "activation_threshold": trailing_activation_threshold,
            "trailing_distance_pct": trailing_distance_pct,
            "breakeven_threshold": breakeven_threshold,
            "breakeven_buffer": breakeven_buffer,
        }
    else:
        trailing_stop_config = {"enabled": False}

    # Engine-level risk overrides
    risk_overrides: dict[str, Any] = {
        "position_sizer": "leveraged_fixed_fraction",
        "base_fraction": base_fraction,
        "min_fraction": 0.01,
        "max_fraction": min(base_fraction * max_leverage, 0.50),
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "leverage": {
            "enabled": True,
            "max_leverage": max_leverage,
            "decay_rate": leverage_decay_rate,
            "min_regime_bars": min_regime_bars,
        },
        "dynamic_risk": {
            "enabled": True,
            # Wider drawdown tolerance for hyper-growth target
            "drawdown_thresholds": [0.15, 0.30, 0.45],
            "risk_reduction_factors": [0.8, 0.5, 0.2],
            "recovery_thresholds": [0.08, 0.15],
        },
        "partial_operations": {
            "exit_targets": [0.08, 0.15, 0.30],
            "exit_sizes": [0.20, 0.30, 0.50],
            "scale_in_thresholds": [0.015, 0.03],
            "scale_in_sizes": [0.40, 0.25],
            "max_scale_ins": 2,
        },
        "trailing_stop": trailing_stop_config,
    }

    # MFE early-cut (default OFF): the key is only present when configured,
    # so existing configs hydrate no policy (build_early_cut_policy).
    if early_cut_mfe_threshold_pct is not None:
        risk_overrides["early_cut"] = {
            "mfe_threshold_pct": early_cut_mfe_threshold_pct,
            "evaluation_window_hours": early_cut_evaluation_window_hours,
        }

    strategy.set_risk_overrides(risk_overrides)

    return strategy
