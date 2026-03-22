"""
Chaos Test Strategy

High-frequency alternating strategy for reconciliation system validation.
Trades approximately every 3 candles on 1-minute timeframes by combining
RSI signals with forced direction alternation, generating enough trade
volume to exercise all reconciliation code paths.

NOT for production use -- paper trading validation only.
"""

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.strategies.components.position_sizer import FixedFractionSizer
from src.strategies.components.regime_context import EnhancedRegimeDetector
from src.strategies.components.signal_generator import (
    Signal,
    SignalDirection,
    SignalGenerator,
)
from src.strategies.hyper_growth import FlatRiskManager

if TYPE_CHECKING:
    from src.strategies.components.regime_context import RegimeContext
    from src.strategies.components.strategy import Strategy

logger = logging.getLogger(__name__)


class ChaosSignalGenerator(SignalGenerator):
    """Generates frequent alternating BUY/SELL signals for chaos testing.

    Uses RSI as the base indicator with forced alternation after a configurable
    number of candles. Guarantees constant trade flow regardless of market
    conditions by flipping direction when max_hold_candles is reached.

    Args:
        rsi_buy_threshold: RSI level below which a BUY signal is generated.
        rsi_sell_threshold: RSI level above which a SELL signal is generated.
        max_hold_candles: Force direction flip after this many candles in a position.
        rsi_period: Lookback period for RSI calculation.
    """

    def __init__(
        self,
        rsi_buy_threshold: float = 35.0,
        rsi_sell_threshold: float = 65.0,
        max_hold_candles: int = 3,
        rsi_period: int = 14,
    ):
        super().__init__("chaos_signal_generator")

        if rsi_buy_threshold >= rsi_sell_threshold:
            raise ValueError(
                f"rsi_buy_threshold ({rsi_buy_threshold}) must be less than "
                f"rsi_sell_threshold ({rsi_sell_threshold})"
            )
        if max_hold_candles < 1:
            raise ValueError(f"max_hold_candles must be >= 1, got {max_hold_candles}")
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")

        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.max_hold_candles = max_hold_candles
        self.rsi_period = rsi_period
        self._warmup_period = rsi_period + 1

        # Internal state for forced alternation
        self._candles_in_position = 0
        self._last_direction: SignalDirection = SignalDirection.HOLD

    @property
    def warmup_period(self) -> int:
        """Minimum history required before generating signals."""
        return self._warmup_period

    def generate_signal(
        self,
        df: pd.DataFrame,
        index: int,
        regime: "RegimeContext | None" = None,
    ) -> Signal:
        """Generate a trading signal based on RSI with forced alternation.

        During warmup (index < warmup_period), returns HOLD. After warmup,
        checks RSI thresholds first, then forces alternation if the position
        has been held for max_hold_candles.
        """
        self.validate_inputs(df, index)

        # Warmup: need enough bars for RSI calculation
        if index < self._warmup_period:
            return self._hold_signal(index, reason="warmup")

        rsi_value = self._calculate_rsi(df, index)
        if rsi_value is None:
            return self._hold_signal(index, reason="rsi_unavailable")

        # Forced alternation: flip direction after max_hold_candles
        if (
            self._last_direction != SignalDirection.HOLD
            and self._candles_in_position >= self.max_hold_candles
        ):
            forced_direction = (
                SignalDirection.SELL
                if self._last_direction == SignalDirection.BUY
                else SignalDirection.BUY
            )
            self._last_direction = forced_direction
            self._candles_in_position = 1
            return Signal(
                direction=forced_direction,
                strength=0.8,
                confidence=0.9,
                metadata={
                    "generator": self.name,
                    "index": index,
                    "rsi": rsi_value,
                    "trigger": "forced_alternation",
                    "candles_held": self.max_hold_candles,
                },
            )

        # RSI-based signals
        direction = SignalDirection.HOLD
        trigger = "none"

        if rsi_value < self.rsi_buy_threshold:
            direction = SignalDirection.BUY
            trigger = "rsi_oversold"
        elif rsi_value > self.rsi_sell_threshold:
            direction = SignalDirection.SELL
            trigger = "rsi_overbought"

        if direction != SignalDirection.HOLD:
            self._last_direction = direction
            self._candles_in_position = 1
            return Signal(
                direction=direction,
                strength=0.8,
                confidence=0.9,
                metadata={
                    "generator": self.name,
                    "index": index,
                    "rsi": rsi_value,
                    "trigger": trigger,
                },
            )

        # No signal -- increment hold counter
        if self._last_direction != SignalDirection.HOLD:
            self._candles_in_position += 1

        return self._hold_signal(index, reason="no_trigger", rsi=rsi_value)

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Return fixed high confidence for chaos testing."""
        self.validate_inputs(df, index)
        return 0.9

    def _calculate_rsi(self, df: pd.DataFrame, index: int) -> float | None:
        """Calculate RSI at the given index using the close price series."""
        if "rsi" in df.columns:
            val = df["rsi"].iloc[index]
            if pd.notna(val):
                return float(val)

        # Manual RSI calculation as fallback
        if index < self.rsi_period:
            return None

        close = df["close"].iloc[max(0, index - self.rsi_period) : index + 1]
        delta = close.diff()
        gain = delta.clip(lower=0).mean()
        loss = (-delta.clip(upper=0)).mean()

        if loss == 0:
            return 100.0
        rs = gain / loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _hold_signal(
        self,
        index: int,
        reason: str,
        rsi: float | None = None,
    ) -> Signal:
        """Create a HOLD signal with metadata."""
        metadata: dict[str, Any] = {
            "generator": self.name,
            "index": index,
            "trigger": reason,
        }
        if rsi is not None:
            metadata["rsi"] = rsi
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=0.0,
            metadata=metadata,
        )

    def get_parameters(self) -> dict[str, Any]:
        """Return generator configuration for logging."""
        params = super().get_parameters()
        params.update(
            {
                "rsi_buy_threshold": self.rsi_buy_threshold,
                "rsi_sell_threshold": self.rsi_sell_threshold,
                "max_hold_candles": self.max_hold_candles,
                "rsi_period": self.rsi_period,
                "candles_in_position": self._candles_in_position,
                "last_direction": self._last_direction.value,
            }
        )
        return params


def create_chaos_test_strategy(
    name: str = "ChaosTest",
    risk_fraction: float = 0.02,
    base_fraction: float = 0.02,
    stop_loss_pct: float = 0.01,
    take_profit_pct: float = 0.02,
    max_hold_candles: int = 3,
) -> "Strategy":
    """Create a chaos test strategy for reconciliation validation.

    Composes a high-frequency alternating signal generator with tight risk
    parameters to maximise trade volume while keeping individual trade sizes
    small (~$20 per trade on a $1000 balance).

    Args:
        name: Strategy name for logging and DB references.
        risk_fraction: Fraction of balance risked per trade.
        base_fraction: Base position size as fraction of balance.
        stop_loss_pct: Stop-loss distance as decimal (0.01 = 1%).
        take_profit_pct: Take-profit distance as decimal (0.02 = 2%).
        max_hold_candles: Force direction flip after this many candles.

    Returns:
        Fully composed Strategy ready for the live trading engine.
    """
    from src.strategies.components.strategy import Strategy

    signal_generator = ChaosSignalGenerator(
        rsi_buy_threshold=35.0,
        rsi_sell_threshold=65.0,
        max_hold_candles=max_hold_candles,
    )

    risk_manager = FlatRiskManager(
        risk_fraction=risk_fraction,
        stop_loss_pct=stop_loss_pct,
        min_confidence=0.05,
    )

    position_sizer = FixedFractionSizer(
        fraction=base_fraction,
        adjust_for_confidence=False,
        adjust_for_strength=False,
    )

    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    strategy._extra_metadata = {"ignore_signal_reversal": True}

    strategy.set_risk_overrides(
        {
            "position_sizer": "fixed_fraction",
            "base_fraction": base_fraction,
            "min_fraction": 0.01,
            "max_fraction": 0.05,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "partial_operations": {
                "exit_targets": [0.005, 0.01],
                "exit_sizes": [0.3, 0.3],
                "scale_in_thresholds": [-0.003],
                "scale_in_sizes": [0.5],
                "max_scale_ins": 1,
            },
            "trailing_stop": {
                "activation_threshold": 0.005,
                "trailing_distance_pct": 0.003,
            },
        }
    )

    return strategy
