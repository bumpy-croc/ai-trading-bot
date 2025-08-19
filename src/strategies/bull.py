import numpy as np
import pandas as pd
from indicators.technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
)

from strategies.base import BaseStrategy


class Bull(BaseStrategy):
    """
    Bull Market Strategy

    Optimized for strong uptrends with trend confirmation and momentum filters.

    - Trend confirmation using MA(50) > MA(200) and price above MA(20)/MA(50)
    - Momentum confirmation using MACD histogram and RSI (avoid overbought extremes)
    - Volatility-aware sizing using ATR percentage
    - Simple exit rules via stop-loss/take-profit plus momentum/trend weakening
    """

    # Core risk parameters
    BASE_POSITION_SIZE = 0.15  # 15% base size in bull regimes
    MIN_POSITION_SIZE_RATIO = 0.05
    MAX_POSITION_SIZE_RATIO = 0.25

    def __init__(self, name: str = "Bull"):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"

        # Stop loss / take profit tuned for bull trends
        self.stop_loss_pct = 0.02  # 2% default stop
        self.take_profit_pct = 0.06  # 6% take profit target

        # Sizing and filters
        self.max_atr_pct_for_full_size = 0.04  # If ATR% <= 4%, allow larger size
        self.min_atr_pct_for_entry = 0.005  # If ATR% < 0.5%, avoid chop (need minimal movement)

        # Momentum thresholds
        self.rsi_overbought = 75
        self.min_macd_hist = 0.0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Technicals
        df = calculate_moving_averages(df, periods=[20, 50, 200])
        df = calculate_macd(df)
        df["rsi"] = calculate_rsi(df, period=14)
        df = calculate_atr(df, period=14)
        df = calculate_bollinger_bands(df, period=20, std_dev=2.0)

        # Derived metrics
        df["atr_pct"] = df["atr"] / df["close"]
        df["trend_strength_50_200"] = (df["ma_50"] - df["ma_200"]) / df["ma_200"]
        df["price_above_ma20"] = (df["close"] > df["ma_20"]).astype(int)
        df["price_above_ma50"] = (df["close"] > df["ma_50"]).astype(int)

        # Basic regime classification for internal use
        df["bull_regime"] = (
            (df["ma_50"] > df["ma_200"])
            & (df["close"] > df["ma_50"])
            & (df["ma_20"] > df["ma_50"])  # strong alignment
        ).astype(int)

        # Minimal NaN handling: leave core OHLCV untouched; consumers will dropna on essentials
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False

        close = df["close"].iloc[index]
        rsi = df["rsi"].iloc[index]
        macd_hist = df["macd_hist"].iloc[index]
        atr_pct = df["atr_pct"].iloc[index]

        # Trend alignment
        ma50 = df["ma_50"].iloc[index]
        ma200 = df["ma_200"].iloc[index]
        ma20 = df["ma_20"].iloc[index]

        # Filters
        in_bull = ma50 > ma200 and close > ma50 and ma20 > ma50
        momentum_positive = macd_hist > self.min_macd_hist
        not_overbought = rsi < self.rsi_overbought
        volatility_ok = atr_pct >= self.min_atr_pct_for_entry

        entry_signal = in_bull and momentum_positive and not_overbought and volatility_ok

        # Log decision
        self.log_execution(
            signal_type="entry",
            action_taken="entry_signal" if entry_signal else "no_action",
            price=close,
            signal_strength=(
                float(df["trend_strength_50_200"].iloc[index])
                if "trend_strength_50_200" in df.columns
                else None
            ),
            indicators={
                "ma20": float(ma20) if not np.isnan(ma20) else None,
                "ma50": float(ma50) if not np.isnan(ma50) else None,
                "ma200": float(ma200) if not np.isnan(ma200) else None,
                "rsi": float(rsi) if not np.isnan(rsi) else None,
                "macd_hist": float(macd_hist) if not np.isnan(macd_hist) else None,
                "atr_pct": float(atr_pct) if not np.isnan(atr_pct) else None,
            },
            reasons=[
                "bull_regime" if in_bull else "not_bull_regime",
                "momentum_positive" if momentum_positive else "momentum_weak",
                "not_overbought" if not_overbought else "overbought",
                "volatility_ok" if volatility_ok else "volatility_too_low",
            ],
        )

        return entry_signal

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False

        current_price = df["close"].iloc[index]
        returns = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

        # Stop loss or take profit
        hit_stop = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct

        # Momentum/trend weakening
        macd_hist = df["macd_hist"].iloc[index]
        ma20 = df["ma_20"].iloc[index]
        price_below_ma20 = current_price < ma20 if not np.isnan(ma20) else False
        momentum_turned = macd_hist < 0

        exit_signal = hit_stop or hit_take_profit or (price_below_ma20 and momentum_turned)

        self.log_execution(
            signal_type="exit",
            action_taken="exit_signal" if exit_signal else "hold",
            price=current_price,
            reasons=[
                "hit_stop" if hit_stop else "",
                "hit_take_profit" if hit_take_profit else "",
                (
                    "price_below_ma20_and_momentum_down"
                    if (price_below_ma20 and momentum_turned)
                    else ""
                ),
            ],
        )

        return exit_signal

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if index >= len(df) or balance <= 0:
            return 0.0

        # Trend factor based on MA slope alignment and distance
        trend = (
            float(df["trend_strength_50_200"].iloc[index])
            if "trend_strength_50_200" in df.columns
            else 0.0
        )
        trend_factor = max(0.5, min(1.5, 1.0 + 2.0 * max(0.0, trend)))  # amplify with strong trend

        # Volatility adjustment: scale down if ATR% is high, scale up if moderate
        atr_pct = df["atr_pct"].iloc[index] if "atr_pct" in df.columns else np.nan
        if not np.isnan(atr_pct):
            if atr_pct <= self.max_atr_pct_for_full_size:
                vol_factor = 1.0
            else:
                # Linearly reduce beyond threshold; cap at 50%
                excess = atr_pct - self.max_atr_pct_for_full_size
                vol_factor = max(0.5, 1.0 - 5.0 * excess)
        else:
            vol_factor = 1.0

        base = self.BASE_POSITION_SIZE * trend_factor * vol_factor
        position_ratio = max(self.MIN_POSITION_SIZE_RATIO, min(self.MAX_POSITION_SIZE_RATIO, base))

        # Follow existing convention: return ratio scaled by balance
        return position_ratio * balance

    def calculate_stop_loss(
        self, df: pd.DataFrame, index: int, price: float, side: str = "long"
    ) -> float:
        side_str = side.value if hasattr(side, "value") else str(side)

        # ATR-based stop for longs: max(fixed_pct, k * ATR)
        atr = df["atr"].iloc[index] if "atr" in df.columns else np.nan
        if not np.isnan(atr) and price > 0:
            atr_stop_pct = min(0.04, max(self.stop_loss_pct, float(atr / price) * 1.5))
        else:
            atr_stop_pct = self.stop_loss_pct

        if side_str == "long":
            return price * (1 - atr_stop_pct)
        else:
            return price * (1 + atr_stop_pct)

    def get_parameters(self) -> dict:
        return {
            "name": self.name,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "base_position_size": self.BASE_POSITION_SIZE,
            "min_position_size_ratio": self.MIN_POSITION_SIZE_RATIO,
            "max_position_size_ratio": self.MAX_POSITION_SIZE_RATIO,
            "rsi_overbought": self.rsi_overbought,
            "min_macd_hist": self.min_macd_hist,
            "max_atr_pct_for_full_size": self.max_atr_pct_for_full_size,
            "min_atr_pct_for_entry": self.min_atr_pct_for_entry,
        }
