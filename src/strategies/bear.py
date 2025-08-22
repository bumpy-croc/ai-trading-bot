"""
Bear market strategy.

This strategy is designed for bear market conditions.
"""

import numpy as np
import pandas as pd

from src.indicators.technical import (
    calculate_atr,
    calculate_macd,
    calculate_rsi,
)
from src.strategies.base import BaseStrategy


class BearStrategy(BaseStrategy):
    # Configuration constants (percentages expressed as decimals)
    BASE_POSITION_SIZE = 0.12  # Base position size (12% of balance)
    MIN_POSITION_SIZE_RATIO = 0.04  # 4% minimum
    MAX_POSITION_SIZE_RATIO = 0.2  # 20% maximum

    def __init__(
        self,
        name: str = "BearStrategy",
        short_ma_period: int = 50,
        long_ma_period: int = 200,
        stop_loss_pct: float = 0.02,  # 2% default SL (engine may also use ATR-based level)
        take_profit_pct: float = 0.04,  # 4% default TP
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return df

        # Moving Averages
        df["short_ma"] = df["close"].rolling(window=self.short_ma_period, min_periods=1).mean()
        df["long_ma"] = df["close"].rolling(window=self.long_ma_period, min_periods=1).mean()

        # RSI and MACD
        df["rsi"] = calculate_rsi(df, period=14)
        df = calculate_macd(df, fast_period=12, slow_period=26, signal_period=9)

        # ATR for volatility-aware stops
        df = calculate_atr(df, period=self.atr_period)

        # Confidence metric for short bias (0..1)
        # - Stronger when short_ma << long_ma, MACD negative, RSI below 50
        ma_gap = (df["long_ma"] - df["short_ma"]).clip(lower=0)
        ma_gap_ratio = (ma_gap / df["long_ma"].replace(0, np.nan)).fillna(0).clip(0, 1)
        rsi_component = ((50.0 - df["rsi"]) / 50.0).clip(lower=0, upper=1).fillna(0)
        macd_component = (-df["macd"]).clip(lower=0)
        macd_norm = macd_component / (macd_component.rolling(50).max().replace(0, np.nan))
        macd_norm = macd_norm.fillna(0).clip(0, 1)
        bear_confidence = (0.5 * ma_gap_ratio + 0.3 * rsi_component + 0.2 * macd_norm).clip(0, 1)
        df["prediction_confidence"] = bear_confidence

        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # This strategy does not open long positions
        return False

    def check_short_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False

        row = df.iloc[index]
        # Require trend confirmation and momentum alignment
        trend_down = bool(row.get("short_ma", np.nan) < row.get("long_ma", np.nan))
        price_below_long = bool(row.get("close", np.nan) < row.get("long_ma", np.nan))
        macd_negative = bool(row.get("macd", np.nan) < 0)
        rsi_bearish = bool(row.get("rsi", 50) < 50)

        # Minimum confidence threshold for entry
        confidence = float(row.get("prediction_confidence", 0.0) or 0.0)
        meets_confidence = confidence >= 0.25

        return (
            trend_down and price_below_long and macd_negative and rsi_bearish and meets_confidence
        )

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False

        row = df.iloc[index]
        current_price = float(row["close"])

        # Generic profit/loss thresholds (complementary to engine-level SL/TP)
        move_from_entry = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0
        hit_take_profit = move_from_entry >= self.take_profit_pct
        hit_stop_loss = move_from_entry <= -self.stop_loss_pct

        # Reversal conditions: RSI recovery and MACD improvement
        rsi_reversal = float(row.get("rsi", 50)) > 55
        macd_reversal = float(row.get("macd_hist", 0.0)) > 0
        ma_reversal = float(row.get("short_ma", np.nan)) > float(row.get("long_ma", np.nan))

        return hit_take_profit or hit_stop_loss or rsi_reversal or macd_reversal or ma_reversal

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if index >= len(df) or balance <= 0:
            return 0.0
        confidence = (
            float(df["prediction_confidence"].iloc[index])
            if "prediction_confidence" in df.columns
            else 0.0
        )
        dynamic_size = self.BASE_POSITION_SIZE * max(0.0, min(1.0, confidence))
        sized_ratio = max(
            self.MIN_POSITION_SIZE_RATIO, min(self.MAX_POSITION_SIZE_RATIO, dynamic_size)
        )
        # Follow existing convention: return ratio scaled by balance (engine caps to max ratio)
        return sized_ratio * balance

    def calculate_stop_loss(self, df, index, price, side: str = "long") -> float:
        # Handle both enum and string inputs
        side_str = side.value if hasattr(side, "value") else str(side)
        price = float(price)

        # ATR-based stop for volatility awareness; fallback to fixed percentage
        atr_value = None
        try:
            atr_value = float(df["atr"].iloc[index]) if "atr" in df.columns else None
        except Exception:
            atr_value = None

        if side_str == "short":
            if atr_value and atr_value > 0:
                return price + self.atr_multiplier * atr_value
            return price * (1 + self.stop_loss_pct)
        else:
            # Not used by this strategy, but provided for completeness
            if atr_value and atr_value > 0:
                return price - self.atr_multiplier * atr_value
            return price * (1 - self.stop_loss_pct)

    def get_parameters(self) -> dict:
        return {
            "name": self.name,
            "short_ma_period": self.short_ma_period,
            "long_ma_period": self.long_ma_period,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
        }
