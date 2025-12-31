"""
Regime-Aware Strategy Switcher

This module extends the StrategyManager to automatically switch trading strategies
based on detected market regimes for optimal performance.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.config.constants import (
    DEFAULT_REGIME_CONFIG_1D,
    DEFAULT_REGIME_CONFIG_1H,
    DEFAULT_REGIME_CONFIG_4H,
    DEFAULT_REGIME_MAX_DRAWDOWN_SWITCH,
    DEFAULT_REGIME_MIN_CONFIDENCE,
    DEFAULT_REGIME_MIN_DATA_LENGTH,
    DEFAULT_REGIME_MIN_DURATION_BARS,
    DEFAULT_REGIME_MULTIPLIER_BEAR_HIGH_VOL,
    DEFAULT_REGIME_MULTIPLIER_BEAR_LOW_VOL,
    DEFAULT_REGIME_MULTIPLIER_BULL_HIGH_VOL,
    DEFAULT_REGIME_MULTIPLIER_BULL_LOW_VOL,
    DEFAULT_REGIME_MULTIPLIER_RANGE_HIGH_VOL,
    DEFAULT_REGIME_MULTIPLIER_RANGE_LOW_VOL,
    DEFAULT_REGIME_SWITCH_COOLDOWN_MINUTES,
    DEFAULT_REGIME_TIMEFRAME_AGREEMENT,
    DEFAULT_REGIME_TIMEFRAME_WEIGHTS,
    DEFAULT_REGIME_TIMEFRAMES,
    DEFAULT_REGIME_TRANSITION_SIZE_MULTIPLIER,
    DEFAULT_REGIME_UNKNOWN_MULTIPLIER,
)
from src.engines.live.strategy_manager import StrategyManager
from src.regime.detector import RegimeConfig, RegimeDetector, TrendLabel, VolLabel

logger = logging.getLogger(__name__)


@dataclass
class RegimeStrategyMapping:
    """Maps market regimes to optimal strategies"""

    # Bull market strategies
    bull_low_vol: str = "momentum_leverage"  # Use aggressive momentum in calm bull markets
    bull_high_vol: str = "ensemble_weighted"  # Use ensemble in volatile bull markets

    # Bear market strategies
    bear_low_vol: str = "ml_basic"  # Use ML in calm bear markets
    bear_high_vol: str = "ml_adaptive"  # Use adaptive ML in volatile bear markets

    # Range/sideways market strategies
    range_low_vol: str = "ml_basic"  # Use ML in calm range markets
    range_high_vol: str = "ml_basic"  # Use ML with reduced size in volatile ranges

    # Position size adjustments by regime (use centralized constants)
    bull_low_vol_multiplier: float = DEFAULT_REGIME_MULTIPLIER_BULL_LOW_VOL
    bull_high_vol_multiplier: float = DEFAULT_REGIME_MULTIPLIER_BULL_HIGH_VOL
    bear_low_vol_multiplier: float = DEFAULT_REGIME_MULTIPLIER_BEAR_LOW_VOL
    bear_high_vol_multiplier: float = DEFAULT_REGIME_MULTIPLIER_BEAR_HIGH_VOL
    range_low_vol_multiplier: float = DEFAULT_REGIME_MULTIPLIER_RANGE_LOW_VOL
    range_high_vol_multiplier: float = DEFAULT_REGIME_MULTIPLIER_RANGE_HIGH_VOL


@dataclass
class SwitchingConfig:
    """Configuration for regime-based strategy switching"""

    # Switching criteria (use centralized constants)
    min_regime_confidence: float = DEFAULT_REGIME_MIN_CONFIDENCE
    min_regime_duration: int = DEFAULT_REGIME_MIN_DURATION_BARS
    switch_cooldown_minutes: int = DEFAULT_REGIME_SWITCH_COOLDOWN_MINUTES

    # Enhanced regime detection
    enable_multi_timeframe: bool = True  # Use multiple timeframes for confirmation
    timeframes: list = None  # Timeframes to analyze ['1h', '4h', '1d']
    require_timeframe_agreement: float = DEFAULT_REGIME_TIMEFRAME_AGREEMENT

    # Position management during switches
    close_positions_on_switch: bool = False  # Whether to close positions when switching
    reduce_size_during_transition: bool = True  # Reduce position sizes during transitions
    transition_size_multiplier: float = DEFAULT_REGIME_TRANSITION_SIZE_MULTIPLIER

    # Risk management
    max_drawdown_switch_threshold: float = DEFAULT_REGIME_MAX_DRAWDOWN_SWITCH
    emergency_strategy: str = "ml_basic"  # Strategy to use in emergencies


class RegimeStrategySwitcher:
    """
    Intelligent strategy switcher that automatically adapts to market regimes
    """

    def __init__(
        self,
        strategy_manager: StrategyManager,
        regime_config: RegimeConfig | None = None,
        strategy_mapping: RegimeStrategyMapping | None = None,
        switching_config: SwitchingConfig | None = None,
    ):
        self.strategy_manager = strategy_manager
        self.regime_detector = RegimeDetector(regime_config or RegimeConfig())
        self.strategy_mapping = strategy_mapping or RegimeStrategyMapping()
        self.switching_config = switching_config or SwitchingConfig()

        # Initialize timeframes for multi-timeframe analysis
        if self.switching_config.timeframes is None:
            self.switching_config.timeframes = list(DEFAULT_REGIME_TIMEFRAMES)

        # State tracking
        self.current_regime: str | None = None
        self.regime_confidence: float = 0.0
        self.regime_start_time: datetime | None = None
        self.regime_start_candle_index: int | None = None
        self.last_switch_time: datetime | None = None
        self.regime_duration: int = 0

        # Multi-timeframe regime detectors
        self.timeframe_detectors: dict[str, RegimeDetector] = {}
        if self.switching_config.enable_multi_timeframe:
            for tf in self.switching_config.timeframes:
                # Adjust parameters based on timeframe
                config = self._get_timeframe_config(tf)
                self.timeframe_detectors[tf] = RegimeDetector(config)

        # Performance tracking
        self.strategy_performance: dict[str, dict[str, float]] = {}
        self.regime_history: list = []

        # Callbacks
        self.on_regime_change: Callable | None = None
        self.on_strategy_switch: Callable | None = None

        logger.info("RegimeStrategySwitcher initialized")

    def _get_timeframe_config(self, timeframe: str) -> RegimeConfig:
        """Get regime detection config adjusted for timeframe.

        Uses centralized constants for consistent configuration across engines.
        """
        # Map timeframe to centralized config
        config_map = {
            "1h": DEFAULT_REGIME_CONFIG_1H,
            "4h": DEFAULT_REGIME_CONFIG_4H,
            "1d": DEFAULT_REGIME_CONFIG_1D,
        }

        if timeframe in config_map:
            cfg = config_map[timeframe]
            return RegimeConfig(
                slope_window=cfg["slope_window"],
                hysteresis_k=cfg["hysteresis_k"],
                min_dwell=cfg["min_dwell"],
                trend_threshold=cfg["trend_threshold"],
            )

        # Fallback to default config for unknown timeframes
        return RegimeConfig()

    def analyze_market_regime(self, price_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """
        Analyze market regime across multiple timeframes

        Args:
            price_data: Dictionary with timeframe keys and price dataframes

        Returns:
            Dictionary with regime analysis results
        """

        regime_results = {}

        # Analyze each timeframe
        for timeframe in self.switching_config.timeframes:
            if timeframe not in price_data:
                continue

            df = price_data[timeframe]
            if len(df) < DEFAULT_REGIME_MIN_DATA_LENGTH:  # Need sufficient data
                continue

            detector = self.timeframe_detectors.get(timeframe, self.regime_detector)
            df_with_regime = detector.annotate(df)

            # Get current regime for this timeframe
            trend_label, vol_label, confidence = detector.current_labels(df_with_regime)
            regime_label = f"{trend_label}:{vol_label}"

            regime_results[timeframe] = {
                "trend_label": trend_label,
                "vol_label": vol_label,
                "regime_label": regime_label,
                "confidence": confidence,
                "trend_score": df_with_regime["trend_score"].iloc[-1],
                "atr_percentile": df_with_regime["atr_percentile"].iloc[-1],
            }

        # Determine consensus regime
        consensus_regime = self._determine_consensus_regime(regime_results)

        return {
            "timeframe_regimes": regime_results,
            "consensus_regime": consensus_regime,
            "analysis_timestamp": datetime.now(UTC),
        }

    def _determine_consensus_regime(self, regime_results: dict[str, dict]) -> dict[str, Any]:
        """Determine consensus regime across timeframes"""

        if not regime_results:
            return {
                "regime_label": "unknown:unknown",
                "confidence": 0.0,
                "agreement_score": 0.0,
                "participating_timeframes": [],
            }

        # Collect regime votes with weights
        regime_votes = {}
        confidence_sum = 0.0
        weight_sum = 0.0

        # Weight longer timeframes more heavily (use centralized constant)
        timeframe_weights = DEFAULT_REGIME_TIMEFRAME_WEIGHTS

        for tf, result in regime_results.items():
            regime = result["regime_label"]
            confidence = result["confidence"]
            weight = timeframe_weights.get(tf, 1.0)

            if regime not in regime_votes:
                regime_votes[regime] = 0.0

            regime_votes[regime] += weight * confidence
            confidence_sum += confidence * weight
            weight_sum += weight

        # Find winning regime
        if not regime_votes:
            return {
                "regime_label": "unknown:unknown",
                "confidence": 0.0,
                "agreement_score": 0.0,
                "participating_timeframes": [],
            }

        winning_regime = max(regime_votes.keys(), key=lambda k: regime_votes[k])
        winning_score = regime_votes[winning_regime]

        # Calculate agreement score
        total_votes = sum(regime_votes.values())
        agreement_score = winning_score / total_votes if total_votes > 0 else 0.0

        # Calculate average confidence
        avg_confidence = confidence_sum / weight_sum if weight_sum > 0 else 0.0

        return {
            "regime_label": winning_regime,
            "confidence": avg_confidence,
            "agreement_score": agreement_score,
            "participating_timeframes": list(regime_results.keys()),
            "regime_votes": regime_votes,
        }

    def should_switch_strategy(
        self, regime_analysis: dict[str, Any], current_candle_index: int | None = None
    ) -> dict[str, Any]:
        """Determine if strategy should be switched based on regime analysis

        Args:
            regime_analysis: Results from analyze_market_regime
            current_candle_index: Current candle index for accurate duration tracking
        """

        consensus = regime_analysis["consensus_regime"]
        new_regime = consensus["regime_label"]
        confidence = consensus["confidence"]
        agreement = consensus["agreement_score"]

        # Get optimal strategy for this regime
        optimal_strategy = self._get_optimal_strategy(new_regime)
        current_strategy_name = (
            self.strategy_manager.current_strategy.name
            if self.strategy_manager.current_strategy
            else None
        )

        # Decision criteria
        decision = {
            "should_switch": False,
            "reason": "",
            "new_regime": new_regime,
            "optimal_strategy": optimal_strategy,
            "current_strategy": current_strategy_name,
            "confidence": confidence,
            "agreement": agreement,
        }

        # Check if we should switch
        if confidence < self.switching_config.min_regime_confidence:
            decision["reason"] = (
                f"Low confidence: {confidence:.3f} < {self.switching_config.min_regime_confidence}"
            )
            return decision

        if agreement < self.switching_config.require_timeframe_agreement:
            decision["reason"] = (
                f"Low timeframe agreement: {agreement:.3f} < {self.switching_config.require_timeframe_agreement}"
            )
            return decision

        # Check cooldown
        if self.last_switch_time:
            time_since_switch = datetime.now(UTC) - self.last_switch_time
            cooldown = timedelta(minutes=self.switching_config.switch_cooldown_minutes)
            if time_since_switch < cooldown:
                decision["reason"] = f"Switch cooldown: {time_since_switch} < {cooldown}"
                return decision

        # Check if regime has been stable long enough
        if self.current_regime == new_regime:
            # Calculate duration based on actual candle count if index is provided
            if current_candle_index is not None and self.regime_start_candle_index is not None:
                self.regime_duration = current_candle_index - self.regime_start_candle_index + 1
            else:
                # Fallback to old method if candle index not available
                self.regime_duration += 1
        else:
            self.regime_duration = 1
            self.current_regime = new_regime
            self.regime_start_time = datetime.now(UTC)
            if current_candle_index is not None:
                self.regime_start_candle_index = current_candle_index

        if self.regime_duration < self.switching_config.min_regime_duration:
            decision["reason"] = (
                f"Regime not stable: {self.regime_duration} < {self.switching_config.min_regime_duration}"
            )
            return decision

        # Check if optimal strategy is different from current
        if optimal_strategy == current_strategy_name:
            decision["reason"] = f"Already using optimal strategy: {optimal_strategy}"
            return decision

        # All checks passed - recommend switch
        decision["should_switch"] = True
        decision["reason"] = "Regime stable, high confidence, different optimal strategy"

        return decision

    def _get_optimal_strategy(self, regime_label: str) -> str:
        """Get optimal strategy for a given regime"""

        if ":" not in regime_label:
            return self.switching_config.emergency_strategy

        trend_label, vol_label = regime_label.split(":")

        # Map regime to strategy
        if trend_label == TrendLabel.TREND_UP.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bull_low_vol
            else:
                return self.strategy_mapping.bull_high_vol
        elif trend_label == TrendLabel.TREND_DOWN.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bear_low_vol
            else:
                return self.strategy_mapping.bear_high_vol
        else:  # Range market
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.range_low_vol
            else:
                return self.strategy_mapping.range_high_vol

    def get_position_size_multiplier(self, regime_label: str) -> float:
        """Get position size multiplier for regime"""

        if ":" not in regime_label:
            return DEFAULT_REGIME_UNKNOWN_MULTIPLIER

        trend_label, vol_label = regime_label.split(":")

        # Map regime to position size multiplier
        if trend_label == TrendLabel.TREND_UP.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bull_low_vol_multiplier
            else:
                return self.strategy_mapping.bull_high_vol_multiplier
        elif trend_label == TrendLabel.TREND_DOWN.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bear_low_vol_multiplier
            else:
                return self.strategy_mapping.bear_high_vol_multiplier
        else:  # Range market
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.range_low_vol_multiplier
            else:
                return self.strategy_mapping.range_high_vol_multiplier

    def execute_strategy_switch(self, switch_decision: dict[str, Any]) -> bool:
        """Execute the strategy switch"""

        if not switch_decision["should_switch"]:
            return False

        try:
            new_strategy = switch_decision["optimal_strategy"]

            # For now, just pass the name parameter to avoid constructor issues
            # The regime-specific configuration can be added later when strategies support it
            regime_config = {"name": f"{new_strategy}_RegimeAdaptive"}

            # Execute hot swap
            success = self.strategy_manager.hot_swap_strategy(
                new_strategy_name=new_strategy,
                new_config=regime_config,
                close_existing_positions=self.switching_config.close_positions_on_switch,
            )

            if success:
                self.last_switch_time = datetime.now(UTC)

                # Record the switch
                switch_record = {
                    "timestamp": self.last_switch_time,
                    "from_strategy": switch_decision["current_strategy"],
                    "to_strategy": new_strategy,
                    "regime": switch_decision["new_regime"],
                    "confidence": switch_decision["confidence"],
                    "reason": switch_decision["reason"],
                }
                self.regime_history.append(switch_record)

                # Notify callbacks
                if self.on_strategy_switch:
                    self.on_strategy_switch(switch_record)

                logger.info(
                    f"✅ Strategy switched: {switch_decision['current_strategy']} → {new_strategy} "
                    f"(regime: {switch_decision['new_regime']}, confidence: {switch_decision['confidence']:.3f})"
                )

                return True
            else:
                logger.error(f"❌ Strategy switch failed: {switch_decision}")
                return False

        except Exception as e:
            logger.error(f"❌ Strategy switch execution failed: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get current status of regime-based switching"""

        return {
            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "regime_duration": self.regime_duration,
            "current_strategy": (
                self.strategy_manager.current_strategy.name
                if self.strategy_manager.current_strategy
                else None
            ),
            "last_switch_time": (
                self.last_switch_time.isoformat() if self.last_switch_time else None
            ),
            "total_switches": len(self.regime_history),
            "switching_enabled": True,
            "multi_timeframe_enabled": self.switching_config.enable_multi_timeframe,
            "timeframes": self.switching_config.timeframes,
        }

    def get_switching_history(self) -> list:
        """Get history of strategy switches"""
        return [
            {**record, "timestamp": record["timestamp"].isoformat()}
            for record in self.regime_history
        ]
