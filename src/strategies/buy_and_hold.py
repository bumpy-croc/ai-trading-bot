"""
Buy-and-Hold Strategy - Benchmark Implementation

This strategy implements a pure buy-and-hold approach:
- Enters a long position on the first available candle
- Holds the position indefinitely (never exits)
- No stop loss, no take profit
- Maximum position size (all-in)
- Serves as the benchmark that active strategies must beat

This strategy is used to establish performance baselines for comparison.
It represents the simplest possible strategy: buying and holding without
any active trading decisions.

Performance Characteristics:
- Returns = Price appreciation from start to end
- Maximum drawdown = Worst peak-to-trough decline in asset price
- Sharpe ratio = Risk-adjusted returns of the underlying asset
- No trading costs beyond initial entry
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.risk.risk_manager import RiskManager as EngineRiskManager
from src.risk.risk_manager import RiskParameters
from src.strategies.components import (
    CoreRiskAdapter,
    EnhancedRegimeDetector,
    FixedFractionSizer,
    RegimeContext,
    Signal,
    SignalDirection,
    SignalGenerator,
    Strategy,
)


class BuyAndHoldSignalGenerator(SignalGenerator):
    """
    Signal generator for buy-and-hold strategy

    Generates BUY signal on first candle, then HOLD forever.
    This creates a position that is never exited, simulating
    perfect buy-and-hold behavior.
    """

    def __init__(self):
        super().__init__("buy_and_hold_signal_generator")
        self._first_signal_generated = False

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None
    ) -> Signal:
        """
        Generate buy-and-hold signal

        Args:
            df: DataFrame containing OHLCV data
            index: Current index position
            regime: Optional regime context (unused)

        Returns:
            BUY signal on first call, HOLD on all subsequent calls
        """
        self.validate_inputs(df, index)

        # Generate BUY signal on first candle
        if not self._first_signal_generated:
            self._first_signal_generated = True
            return Signal(
                direction=SignalDirection.BUY,
                strength=1.0,  # Maximum strength
                confidence=1.0,  # Maximum confidence
                metadata={
                    "generator": self.name,
                    "index": index,
                    "timestamp": df.index[index] if hasattr(df.index, "__getitem__") else None,
                    "entry_price": df["close"].iloc[index],
                    "strategy": "buy_and_hold",
                },
            )

        # HOLD forever after initial buy
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=1.0,  # High confidence in holding
            metadata={
                "generator": self.name,
                "index": index,
                "timestamp": df.index[index] if hasattr(df.index, "__getitem__") else None,
                "strategy": "buy_and_hold",
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Always return maximum confidence"""
        self.validate_inputs(df, index)
        return 1.0

    def reset(self):
        """Reset the signal generator state"""
        self._first_signal_generated = False


def create_buy_and_hold_strategy(
    name: str = "BuyAndHold",
    position_fraction: float = 1.0,  # 100% (TRUE buy-and-hold - fully invested)
) -> Strategy:
    """
    Create a buy-and-hold strategy for benchmarking

    This strategy serves as the baseline that active trading strategies
    must beat. It represents the simplest possible approach: buy on day 1
    and hold forever, with no active management.

    Args:
        name: Strategy name for identification
        position_fraction: Fraction of capital to invest (default: 1.0 = 100%)
            IMPORTANT: Should always be 1.0 for true buy-and-hold benchmark.
            Any value < 1.0 creates a hybrid cash/investment portfolio that
            artificially reduces returns and makes the baseline easier to beat.

    Returns:
        Configured Strategy instance that buys and holds

    Example:
        >>> strategy = create_buy_and_hold_strategy()
        >>> backtester = Backtester(strategy=strategy, ...)
        >>> results = backtester.run(symbol="BTCUSDT", timeframe="1d", days=1825)
        >>> print(f"5-year buy-and-hold return: {results['total_return']:.2f}%")
    """
    # Create minimal risk manager (no stop loss, no take profit)
    risk_parameters = RiskParameters(
        base_risk_per_trade=position_fraction,  # Use position fraction as base risk
        max_risk_per_trade=position_fraction,  # Same as base risk
        default_take_profit_pct=None,  # No take profit
        max_position_size=position_fraction,  # All-in or custom fraction
    )
    core_risk_manager = EngineRiskManager(risk_parameters)

    # Override with buy-and-hold specific settings
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": position_fraction,  # All-in
        "max_fraction": position_fraction,
        "stop_loss_pct": None,  # No stop loss
        "take_profit_pct": None,  # No take profit
    }

    # Create components
    signal_generator = BuyAndHoldSignalGenerator()
    risk_manager = CoreRiskAdapter(core_risk_manager)
    risk_manager.set_strategy_overrides(risk_overrides)
    position_sizer = FixedFractionSizer(fraction=position_fraction)
    regime_detector = EnhancedRegimeDetector()

    # Compose strategy
    return Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )
