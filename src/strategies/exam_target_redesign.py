"""Exam-only strategy factory for the TARGET-REDESIGN tournament.

Mirrors ``src/strategies/ml_basic.py``'s ``create_ml_basic_strategy`` wiring
pattern exactly (``CoreRiskAdapter(EngineRiskManager(RiskParameters(...)))``
+ ``ConfidenceWeightedSizer(base_fraction=0.2, min_confidence=0.3)``) --
deliberately NOT HyperGrowth's ``FlatRiskManager`` +
``FixedFractionSizer(adjust_for_confidence=False)`` wiring, which #938
proved cannot express confidence/magnitude differences between candidates
at all (the load-bearing fix this exam harness depends on).

Stop/target geometry defaults to the Board-ratified ``risk-limits.json``
defaults (``DEFAULT_STOP_LOSS_PCT``/``DEFAULT_TAKE_PROFIT_PCT`` = 5%/4%,
``max_holding_hours=336``) -- NOT prod HyperGrowth's 10%/30% (see the
TARGET-REDESIGN tournament preregistration §5/§9 Deviation 2, and entrant
(c)'s triple-barrier labels, which use the same defaults for
self-consistency between what the label encodes and what the exam executes).

Takes a pluggable ``signal_generator`` since one exam harness serves all
four TARGET-REDESIGN entrants: ``ClassificationExamSignalGenerator`` for
(a) meta-labeling / (b) binary direction / (c) triple-barrier, and
``SmoothedReturnExamSignalGenerator`` for (d) smoothed forward return
(both in ``src/strategies/components/exam_signal_generator.py``).
"""

from __future__ import annotations

from src.config.constants import (
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_STRATEGY_BASE_FRACTION,
    DEFAULT_STRATEGY_MIN_CONFIDENCE,
    DEFAULT_TAKE_PROFIT_PCT,
)
from src.risk.risk_manager import RiskManager as EngineRiskManager
from src.risk.risk_manager import RiskParameters
from src.strategies.components import (
    ConfidenceWeightedSizer,
    CoreRiskAdapter,
    EnhancedRegimeDetector,
    SignalGenerator,
    Strategy,
)

DEFAULT_EXAM_STRATEGY_NAME = "TargetRedesignExam"


def create_exam_strategy(
    signal_generator: SignalGenerator,
    name: str = DEFAULT_EXAM_STRATEGY_NAME,
    *,
    base_fraction: float = DEFAULT_STRATEGY_BASE_FRACTION,
    min_confidence: float = DEFAULT_STRATEGY_MIN_CONFIDENCE,
    min_confidence_floor: float = 0.0,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    max_holding_hours: int = DEFAULT_MAX_HOLDING_HOURS,
) -> Strategy:
    """Build the TARGET-REDESIGN tournament's shared exam-only strategy.

    Every entrant's reformed SignalGenerator plugs into this SAME wiring
    (preregistration §5) -- money-metric comparability across entrants
    depends on it (two entrants that agree on direction but differ in
    confidence WILL produce differently-sized trades under
    ConfidenceWeightedSizer, unlike HyperGrowth's flat sizer, per #938).

    Args:
        signal_generator: The entrant's SignalGenerator instance (already
            constructed/configured by the caller).
        name: Strategy name (used in logs/metadata; give each entrant a
            distinct name, e.g. "EntrantB_BinaryDirection").
        base_fraction: ConfidenceWeightedSizer base fraction (default 0.2,
            matching ml_basic/ml_adaptive's DEFAULT_STRATEGY_BASE_FRACTION).
        min_confidence: Minimum signal confidence before any position is
            opened (default 0.3, DEFAULT_STRATEGY_MIN_CONFIDENCE).
        min_confidence_floor: Lower bound on the confidence factor once the
            min_confidence gate has passed (0.0 disables).
        stop_loss_pct: Stop-loss distance (default: ratified risk-limits.json
            default, 5%).
        take_profit_pct: Take-profit distance (default: ratified
            risk-limits.json default, 4%).
        max_holding_hours: Vertical/time exit barrier (default: ratified
            risk-limits.json operational.max_holding_hours, 336).

    Returns:
        Configured Strategy instance.
    """
    risk_parameters = RiskParameters(
        default_take_profit_pct=take_profit_pct,
        time_exits={"max_holding_hours": max_holding_hours},
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_manager = CoreRiskAdapter(core_risk_manager)

    risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": base_fraction,
        "max_fraction": base_fraction,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "time_exits": {"max_holding_hours": max_holding_hours},
    }
    risk_manager.set_strategy_overrides(risk_overrides)

    position_sizer = ConfidenceWeightedSizer(
        base_fraction=base_fraction,
        min_confidence=min_confidence,
        min_confidence_floor=min_confidence_floor,
    )
    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )
    # Also register on the Strategy object itself: build_time_exit_policy
    # (src/engines/shared/risk_configuration.py) checks
    # strategy.get_risk_overrides() BEFORE falling back to the risk
    # manager's own params -- and that fallback only works when the raw
    # PortfolioRiskManager (not the CoreRiskAdapter wrapper) is passed in.
    # Setting it here makes max_holding_hours resolve correctly regardless
    # of which object an engine passes to build_time_exit_policy.
    strategy.set_risk_overrides(risk_overrides)

    return strategy


__all__ = ["DEFAULT_EXAM_STRATEGY_NAME", "create_exam_strategy"]
