"""
Exit policies for live trading.

Implements a simple, robust model-outage exit policy that applies only when
ML predictions are unavailable. It does NOT open new positions; it only
manages exits for existing ones using deterministic rules.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from src.config.constants import (
    MODEL_OUTAGE_EXIT_ENABLED,
    MODEL_OUTAGE_MAX_HOLD_SECONDS,
    MODEL_OUTAGE_BREAKEVEN_TRIGGER_PCT,
    MODEL_OUTAGE_TRAIL_ATR_MULTIPLIER,
    MODEL_OUTAGE_TRAIL_MIN_PCT,
    MODEL_OUTAGE_VOLATILITY_GUARD_MULTIPLIER,
)


@dataclass
class PositionSnapshot:
    symbol: str
    side: str  # 'long' | 'short'
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]


class ModelOutageExitPolicy:
    """Applies deterministic exits when the model is unavailable."""

    def __init__(self, enabled: bool = MODEL_OUTAGE_EXIT_ENABLED):
        self.enabled = enabled

    def should_exit(
        self,
        df: pd.DataFrame,
        index: int,
        position: PositionSnapshot,
    ) -> bool:
        """Return True if the policy indicates we should exit now."""
        if not self.enabled or index <= 0 or index >= len(df):
            return False

        current_price = float(df['close'].iloc[index])
        entry_price = float(position.entry_price)
        hold_seconds = (datetime.now() - position.entry_time).total_seconds()

        # 1) Time-based exit
        if hold_seconds > MODEL_OUTAGE_MAX_HOLD_SECONDS:
            return True

        # 2) Volatility shock guard using ATR and rolling median
        atr = float(df['atr'].iloc[index]) if 'atr' in df.columns and pd.notna(df['atr'].iloc[index]) else None
        if atr is not None:
            # Use a 20-bar rolling median as baseline if available
            if 'atr' in df.columns and index >= 20:
                recent_atr = df['atr'].iloc[max(0, index-20):index].dropna()
                if len(recent_atr) >= 5:
                    median_atr = float(recent_atr.median())
                    if median_atr > 0 and atr > MODEL_OUTAGE_VOLATILITY_GUARD_MULTIPLIER * median_atr:
                        return True

        # 3) Breakeven move and trailing are handled by stop management; do not force exit here.
        #    Exits occur via existing stop/take-profit or time/volatility guards above.

        # No forced exit
        return False

    def adjust_protective_stops(
        self,
        df: pd.DataFrame,
        index: int,
        position: PositionSnapshot,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Suggest new (stop_loss, take_profit) based on breakeven and trailing rules.
        Returns a tuple of possibly-updated (stop_loss, take_profit).
        """
        if not self.enabled or index <= 0 or index >= len(df):
            return position.stop_loss, position.take_profit

        current_price = float(df['close'].iloc[index])
        entry_price = float(position.entry_price)

        new_stop = position.stop_loss
        take_profit = position.take_profit

        # Breakeven move when unrealized return exceeds trigger
        if position.side == 'long':
            unrealized = (current_price - entry_price) / entry_price
            if unrealized >= MODEL_OUTAGE_BREAKEVEN_TRIGGER_PCT:
                new_stop = max(new_stop or 0.0, entry_price)
        else:
            unrealized = (entry_price - current_price) / entry_price
            if unrealized >= MODEL_OUTAGE_BREAKEVEN_TRIGGER_PCT:
                new_stop = min(new_stop or 1e18, entry_price)

        # Trailing stop using ATR or fixed percent
        atr = float(df['atr'].iloc[index]) if 'atr' in df.columns and pd.notna(df['atr'].iloc[index]) else None
        trail_pct = (atr / current_price) * MODEL_OUTAGE_TRAIL_ATR_MULTIPLIER if atr and current_price > 0 else MODEL_OUTAGE_TRAIL_MIN_PCT

        if position.side == 'long':
            candidate = current_price * (1 - trail_pct)
            if new_stop is None:
                new_stop = candidate
            else:
                new_stop = max(new_stop, candidate)
        else:
            candidate = current_price * (1 + trail_pct)
            if new_stop is None:
                new_stop = candidate
            else:
                new_stop = min(new_stop, candidate)

        return new_stop, take_profit


