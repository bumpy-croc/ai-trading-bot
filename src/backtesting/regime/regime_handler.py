"""RegimeHandler manages regime-based strategy switching.

Coordinates market regime analysis and strategy switching during backtests.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.live.regime_strategy_switcher import RegimeStrategySwitcher
    from src.live.strategy_manager import StrategyManager
    from src.strategies.components import Strategy as ComponentStrategy

logger = logging.getLogger(__name__)

# Buffer for regime lookback calculation
REGIME_LOOKBACK_BUFFER = 5

# Frequency of regime analysis (every N candles)
REGIME_CHECK_FREQUENCY = 50

# Minimum candles before first regime check
REGIME_WARMUP_CANDLES = 60


def compute_regime_lookback(regime_switcher: Any) -> int:
    """Determine how many candles are needed for regime analysis.

    Args:
        regime_switcher: The regime switcher instance.

    Returns:
        Number of lookback candles required.
    """
    if not regime_switcher:
        return 0

    configs: list[Any] = []

    detector = getattr(regime_switcher, "regime_detector", None)
    if detector is not None:
        cfg = getattr(detector, "config", None)
        if cfg is not None:
            configs.append(cfg)

    timeframe_detectors = getattr(regime_switcher, "timeframe_detectors", {}) or {}
    for detector in timeframe_detectors.values():
        cfg = getattr(detector, "config", None)
        if cfg is not None:
            configs.append(cfg)

    if not configs:
        return 0

    slope_window = max((getattr(cfg, "slope_window", 0) or 0) for cfg in configs)
    atr_lookback = max((getattr(cfg, "atr_percentile_lookback", 0) or 0) for cfg in configs)

    base_lookback = max(slope_window, atr_lookback)
    return int(base_lookback + REGIME_LOOKBACK_BUFFER)


class RegimeHandler:
    """Handles regime analysis and strategy switching during backtests.

    This class encapsulates regime-aware strategy switching logic including:
    - Periodic regime analysis
    - Strategy selection based on regime
    - Strategy switching execution
    - Regime history tracking
    """

    def __init__(
        self,
        regime_switcher: RegimeStrategySwitcher,
        strategy_manager: StrategyManager,
        initial_strategy_name: str,
        check_frequency: int = REGIME_CHECK_FREQUENCY,
        warmup_candles: int = REGIME_WARMUP_CANDLES,
    ) -> None:
        """Initialize regime handler.

        Args:
            regime_switcher: Switcher for regime-based strategy changes.
            strategy_manager: Manager for loading strategies.
            initial_strategy_name: Name of the initial strategy.
            check_frequency: How often to check regime (in candles).
            warmup_candles: Minimum candles before first check.
        """
        self.regime_switcher = regime_switcher
        self.strategy_manager = strategy_manager
        self.initial_strategy_name = initial_strategy_name
        self.check_frequency = check_frequency
        self.warmup_candles = warmup_candles

        self.regime_history: list[dict] = []
        self.strategy_switches: list[dict] = []
        self._current_strategy_name = initial_strategy_name

    @property
    def current_strategy_name(self) -> str:
        """Get the current strategy name."""
        return self._current_strategy_name

    def should_analyze_regime(self, candle_index: int) -> bool:
        """Check if regime analysis should run at this candle.

        Args:
            candle_index: Current candle index.

        Returns:
            True if regime analysis should be performed.
        """
        if candle_index < self.warmup_candles:
            return False
        return candle_index % self.check_frequency == 0

    def analyze_and_switch_if_needed(
        self,
        df: pd.DataFrame,
        candle_index: int,
        current_time: datetime,
        timeframe: str,
        balance: float,
        current_strategy: ComponentStrategy,
    ) -> tuple[ComponentStrategy | None, bool, dict | None]:
        """Analyze market regime and switch strategy if needed.

        Args:
            df: DataFrame with market data.
            candle_index: Current candle index.
            current_time: Current timestamp.
            timeframe: Candle timeframe.
            balance: Current account balance.
            current_strategy: Currently active strategy.

        Returns:
            Tuple of (new_strategy_or_None, switched, switch_info).
        """
        try:
            # Prepare data slice for regime analysis
            regime_lookback = compute_regime_lookback(self.regime_switcher)
            start_idx = max(0, (candle_index + 1) - max(regime_lookback, 0))
            analysis_df = df.iloc[start_idx : candle_index + 1]

            # Build price data dictionary
            price_data = self._build_price_data(analysis_df, timeframe)

            # Analyze current market regime
            regime_analysis = self.regime_switcher.analyze_market_regime(price_data)

            # Record regime for history
            self._record_regime(candle_index, current_time, regime_analysis)

            # Check if strategy should be switched
            switch_decision = self.regime_switcher.should_switch_strategy(
                regime_analysis, current_candle_index=candle_index
            )

            if not switch_decision["should_switch"]:
                return None, False, None

            # Attempt strategy switch
            new_strategy_name = switch_decision["optimal_strategy"]
            old_strategy_name = current_strategy.name

            # Record switch info
            switch_info = {
                "timestamp": current_time,
                "candle_index": candle_index,
                "old_strategy": old_strategy_name,
                "new_strategy": new_strategy_name,
                "regime": switch_decision["new_regime"],
                "confidence": switch_decision["confidence"],
                "reason": switch_decision["reason"],
                "balance_at_switch": balance,
            }
            self.strategy_switches.append(switch_info)

            # Load new strategy
            new_strategy = self._load_strategy(new_strategy_name)
            if new_strategy is None:
                logger.warning("Failed to load strategy %s", new_strategy_name)
                return None, False, switch_info

            logger.info(
                "Strategy switch at %s (candle %d): %s -> %s (regime: %s)",
                current_time,
                candle_index,
                old_strategy_name,
                new_strategy_name,
                switch_decision["new_regime"],
            )

            # Update regime switcher state
            self._execute_switch(switch_decision, new_strategy)
            self._current_strategy_name = new_strategy_name

            return new_strategy, True, switch_info

        except Exception as e:
            logger.debug("Regime analysis error at candle %d: %s", candle_index, e)
            return None, False, None

    def _build_price_data(
        self,
        analysis_df: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, pd.DataFrame]:
        """Build price data dictionary for regime analysis.

        Args:
            analysis_df: DataFrame slice for analysis.
            timeframe: Candle timeframe.

        Returns:
            Dictionary mapping timeframes to DataFrames.
        """
        price_data: dict[str, pd.DataFrame] = {}

        switching_config = getattr(self.regime_switcher, "switching_config", None)
        if switching_config is not None:
            timeframes = getattr(switching_config, "timeframes", None) or []
            multi_tf = getattr(switching_config, "enable_multi_timeframe", False)
            if multi_tf and timeframes:
                for tf in timeframes:
                    price_data[tf] = analysis_df

        if not price_data:
            price_data[timeframe] = analysis_df

        return price_data

    def _record_regime(
        self,
        candle_index: int,
        current_time: datetime,
        regime_analysis: dict,
    ) -> None:
        """Record regime analysis result to history.

        Args:
            candle_index: Current candle index.
            current_time: Current timestamp.
            regime_analysis: Result from regime analysis.
        """
        self.regime_history.append({
            "timestamp": current_time,
            "candle_index": candle_index,
            "regime": regime_analysis["consensus_regime"]["regime_label"],
            "confidence": regime_analysis["consensus_regime"]["confidence"],
            "agreement": regime_analysis["consensus_regime"]["agreement_score"],
        })

    def _load_strategy(self, strategy_name: str) -> ComponentStrategy | None:
        """Load strategy by name.

        Args:
            strategy_name: Name of strategy to load.

        Returns:
            Strategy instance, or None on failure.
        """
        strategy_factories = {
            "ml_basic": ("src.strategies.ml_basic", "create_ml_basic_strategy"),
            "ml_adaptive": ("src.strategies.ml_adaptive", "create_ml_adaptive_strategy"),
            "ml_sentiment": ("src.strategies.ml_sentiment", "create_ml_sentiment_strategy"),
            "ensemble_weighted": (
                "src.strategies.ensemble_weighted",
                "create_ensemble_weighted_strategy",
            ),
            "momentum_leverage": (
                "src.strategies.momentum_leverage",
                "create_momentum_leverage_strategy",
            ),
        }

        if strategy_name not in strategy_factories:
            logger.warning("Unknown strategy for switching: %s", strategy_name)
            return None

        try:
            module_path, factory_name = strategy_factories[strategy_name]
            module = __import__(module_path, fromlist=[factory_name])
            factory_function = getattr(module, factory_name)
            return factory_function()
        except Exception as e:
            logger.error("Failed to load strategy %s: %s", strategy_name, e)
            return None

    def _execute_switch(
        self,
        switch_decision: dict,
        new_strategy: ComponentStrategy,
    ) -> None:
        """Execute the strategy switch in the regime switcher.

        Args:
            switch_decision: Switch decision from regime analysis.
            new_strategy: New strategy instance.
        """
        try:
            self.regime_switcher.execute_strategy_switch(switch_decision)
        except Exception as e:
            # Fallback to manual state update
            logger.debug("Using fallback state update: %s", e)
            self.regime_switcher.last_switch_time = datetime.now()

            # Update strategy manager's current strategy if available
            if hasattr(self.regime_switcher, "strategy_manager"):
                if self.regime_switcher.strategy_manager:
                    self.regime_switcher.strategy_manager.current_strategy = new_strategy

    def get_results(self) -> dict:
        """Get regime switching results for backtest output.

        Returns:
            Dictionary with regime switching summary.
        """
        return {
            "regime_switching_enabled": True,
            "strategy_switches": self.strategy_switches,
            "regime_history": self.regime_history,
            "final_strategy": self._current_strategy_name,
            "initial_strategy": self.initial_strategy_name,
            "total_strategy_switches": len(self.strategy_switches),
        }

    @staticmethod
    def get_disabled_results() -> dict:
        """Get results when regime switching is disabled.

        Returns:
            Dictionary with empty regime switching data.
        """
        return {
            "regime_switching_enabled": False,
            "strategy_switches": [],
            "regime_history": [],
            "total_strategy_switches": 0,
        }
