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

import os
from typing import Any

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
from src.strategies.components.exam_signal_generator import (
    ClassificationExamSignalGenerator,
    SmoothedReturnExamSignalGenerator,
)

DEFAULT_EXAM_STRATEGY_NAME = "TargetRedesignExam"

# Escape hatch letting the exam harness's fold-runner pin a SPECIFIC
# (non-latest) model version per fold -- PredictionModelRegistry's
# select_bundle()/list_bundles() always resolve "latest", with no override
# (deliberately: select_bundle is also called unconditionally on the LIVE
# trading path in ml_signal_generator.py, so it must never be pinnable via
# ambient env var). This module is exam-only and never imported by the live
# runner, so reading the override here has zero live blast radius. Resolved
# into a full StrategyModel.key string
# (f"{symbol}:{timeframe}:{model_type}:{version}"), which
# PredictionEngine._resolve_bundle finds via PredictionModelRegistry.
# get_bundle_by_key -- the one place a non-latest version is reachable by
# exact key.
_MODEL_VERSION_OVERRIDE_ENV_VAR = "ATB_MODEL_VERSION_OVERRIDE"


def _exam_model_name(symbol: str, timeframe: str, model_type: str) -> str | None:
    """Resolve the model_name to pass an exam SignalGenerator.

    Returns a full bundle key (pinning to a specific version) when
    ``_MODEL_VERSION_OVERRIDE_ENV_VAR`` is set, else None (registry
    default: latest).
    """
    version = os.environ.get(_MODEL_VERSION_OVERRIDE_ENV_VAR)
    if not version:
        return None
    return f"{symbol}:{timeframe}:{model_type}:{version}"


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


def create_exam_binary_direction_strategy(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    model_type: str = "tft",
    sequence_length: int = 120,
    **strategy_kwargs: Any,
) -> Strategy:
    """Entrant (b): binary fixed-horizon direction classification.

    ``model_type`` defaults to "tft" -- the only binary-classification
    architecture (sigmoid/BCE head) in this codebase (see
    task_types.MODEL_TASK_TYPES).
    """
    signal_generator = ClassificationExamSignalGenerator(
        name="exam_binary_direction_signal_generator",
        sequence_length=sequence_length,
        model_name=_exam_model_name(symbol, timeframe, model_type),
    )
    return create_exam_strategy(
        signal_generator, name="EntrantB_BinaryDirection", **strategy_kwargs
    )


def create_exam_triple_barrier_strategy(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    model_type: str = "tft_ternary",
    sequence_length: int = 120,
    **strategy_kwargs: Any,
) -> Strategy:
    """Entrant (c): triple-barrier ternary classification.

    ``model_type`` defaults to "tft_ternary" -- the 3-class softmax sibling
    of "tft" (see models_tft.py::create_tft_ternary_model). Same consuming
    SignalGenerator as entrant (b): both read a classifier bundle's
    probabilities directly via ``result.probabilities``/``result.direction``.
    """
    signal_generator = ClassificationExamSignalGenerator(
        name="exam_triple_barrier_signal_generator",
        sequence_length=sequence_length,
        model_name=_exam_model_name(symbol, timeframe, model_type),
    )
    return create_exam_strategy(signal_generator, name="EntrantC_TripleBarrier", **strategy_kwargs)


def create_exam_smoothed_return_strategy(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    model_type: str = "cnn_lstm",
    sequence_length: int = 120,
    long_entry_threshold: float = 0.0,
    short_entry_threshold: float = 0.0,
    **strategy_kwargs: Any,
) -> Strategy:
    """Entrant (d): smoothed forward return (Board directive).

    ``model_type`` defaults to "cnn_lstm" -- the incumbent's own regression
    architecture, so the (d) vs. incumbent comparison isolates the target
    reformulation (label semantics) rather than also varying architecture
    (preregistration principle: same architecture, different target where
    the target's task type allows it).
    """
    signal_generator = SmoothedReturnExamSignalGenerator(
        name="exam_smoothed_return_signal_generator",
        sequence_length=sequence_length,
        model_name=_exam_model_name(symbol, timeframe, model_type),
        long_entry_threshold=long_entry_threshold,
        short_entry_threshold=short_entry_threshold,
    )
    return create_exam_strategy(signal_generator, name="EntrantD_SmoothedReturn", **strategy_kwargs)


__all__ = [
    "DEFAULT_EXAM_STRATEGY_NAME",
    "create_exam_binary_direction_strategy",
    "create_exam_smoothed_return_strategy",
    "create_exam_strategy",
    "create_exam_triple_barrier_strategy",
]
