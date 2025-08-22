import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.config.constants import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_CORRELATION_UPDATE_FREQUENCY_HOURS,
    DEFAULT_CORRELATION_WINDOW_DAYS,
    DEFAULT_MAX_CORRELATED_EXPOSURE,
)
from src.indicators.technical import calculate_atr


@dataclass
class RiskParameters:
    """Risk management parameters"""

    base_risk_per_trade: float = 0.02  # 2% risk per trade
    max_risk_per_trade: float = 0.03  # 3% maximum risk per trade
    max_position_size: float = 0.25  # 25% maximum position size (fraction of balance)
    max_daily_risk: float = 0.06  # 6% maximum daily risk (fraction of balance)
    max_correlated_risk: float = 0.10  # 10% maximum risk for correlated positions
    max_drawdown: float = 0.20  # 20% maximum drawdown (fraction)
    position_size_atr_multiplier: float = 1.0
    default_take_profit_pct: Optional[float] = None  # if None, engine/strategy may supply
    atr_period: int = 14
    # Time exit config (optional; strategies may override)
    time_exits: Optional[dict] = None
    # Correlation control configuration (used by correlation engine/integration)
    correlation_window_days: int = DEFAULT_CORRELATION_WINDOW_DAYS
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    max_correlated_exposure: float = DEFAULT_MAX_CORRELATED_EXPOSURE
    correlation_update_frequency_hours: int = DEFAULT_CORRELATION_UPDATE_FREQUENCY_HOURS

    def __post_init__(self):
        """Validate risk parameters after initialization"""
        if self.base_risk_per_trade <= 0:
            raise ValueError("base_risk_per_trade must be positive")
        if self.max_risk_per_trade <= 0:
            raise ValueError("max_risk_per_trade must be positive")
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")
        if self.max_daily_risk <= 0:
            raise ValueError("max_daily_risk must be positive")
        if self.max_correlated_risk <= 0:
            raise ValueError("max_correlated_risk must be positive")
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if self.position_size_atr_multiplier <= 0:
            raise ValueError("position_size_atr_multiplier must be positive")

        # Logical consistency checks
        if self.base_risk_per_trade > self.max_risk_per_trade:
            raise ValueError("base_risk_per_trade cannot be greater than max_risk_per_trade")
        if self.correlation_threshold < -1 or self.correlation_threshold > 1:
            raise ValueError("correlation_threshold must be between -1 and 1")
        if self.max_correlated_exposure <= 0 or self.max_correlated_exposure > 1:
            raise ValueError("max_correlated_exposure must be in (0,1]")


class RiskManager:
    """Handles position sizing and risk management"""

    def __init__(
        self, parameters: Optional[RiskParameters] = None, max_concurrent_positions: int = 3
    ):
        self.params = parameters or RiskParameters()
        self.daily_risk_used = 0.0
        self.positions: dict[str, dict] = {}
        self.max_concurrent_positions = max_concurrent_positions

    def reset_daily_risk(self):
        """Reset daily risk counter"""
        self.daily_risk_used = 0.0

    def _ensure_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        if "atr" not in df.columns:
            df = calculate_atr(df, period=self.params.atr_period)
        return df

    # LEGACY/LOW-LEVEL API (quantity based)
    def calculate_position_size(
        self, price: float, atr: float, balance: float, regime: str = "normal"
    ) -> float:
        """
        Calculate position size in units (quantity) based on ATR and risk.
        Kept for backward compatibility and direct usage in tests.
        """
        # Validate inputs
        if price <= 0 or balance <= 0:
            return 0.0

        # Handle zero or very small ATR
        if atr <= 0 or atr < price * 0.001:  # Less than 0.1% of price
            atr = price * 0.01  # Use 1% of price as minimum ATR

        # Adjust risk based on market regime
        base_risk = self.params.base_risk_per_trade
        if regime == "trending":
            risk = base_risk * 1.5
        elif regime == "volatile":
            risk = base_risk * 0.6
        else:
            risk = base_risk

        # Ensure we don't exceed maximum risk limits
        risk = min(risk, self.params.max_risk_per_trade)
        remaining_daily_risk = self.params.max_daily_risk - self.daily_risk_used
        risk = min(risk, remaining_daily_risk)

        # If no remaining daily risk, return 0
        if risk <= 0:
            return 0.0

        # Calculate position size based on ATR
        risk_amount = balance * risk
        atr_stop = atr * self.params.position_size_atr_multiplier
        position_size = risk_amount / atr_stop

        # Ensure position size doesn't exceed maximum (convert to units)
        max_position_value = balance * self.params.max_position_size
        position_size = min(position_size, max_position_value / price)

        # Ensure position size is positive
        return max(0.0, position_size)

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str = "long") -> float:
        """Calculate adaptive stop loss level (ATR-based)"""
        atr_multiple = self.params.position_size_atr_multiplier
        stop_distance = atr * atr_multiple

        if side == "long":
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance

    # NEW HIGHER-LEVEL API (fraction-based + SL/TP policies)
    def calculate_position_fraction(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        price: Optional[float] = None,
        indicators: Optional[dict[str, Any]] = None,
        strategy_overrides: Optional[dict[str, Any]] = None,
        regime: str = "normal",
        correlation_ctx: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Return the fraction of balance to allocate (0..1), enforcing risk limits.
        Uses policy selected by strategy_overrides['position_sizer'].
        Supported sizers: 'fixed_fraction', 'confidence_weighted', 'atr_risk'.

        Examples
        --------
        atr_risk sizer (ATR-based sizing converted to fraction):
            overrides = {
                'position_sizer': 'atr_risk',
                'base_fraction': 0.02,  # ignored for atr_risk
            }
            fraction = risk_manager.calculate_position_fraction(
                df=df, index=i, balance=10_000, price=df['close'].iloc[i],
                strategy_overrides=overrides
            )
            # -> returns a fraction such as 0.03 meaning 3% of balance
        """
        if balance <= 0 or index < 0 or index >= len(df):
            return 0.0
        strategy_overrides = strategy_overrides or {}
        indicators = indicators or {}
        sizer = strategy_overrides.get("position_sizer", "fixed_fraction")
        min_fraction = float(strategy_overrides.get("min_fraction", 0.0))
        max_fraction = float(strategy_overrides.get("max_fraction", self.params.max_position_size))
        max_fraction = min(max_fraction, self.params.max_position_size)

        # Respect remaining daily risk
        remaining_daily_risk = max(0.0, self.params.max_daily_risk - self.daily_risk_used)
        max_fraction = min(max_fraction, remaining_daily_risk)

        # Default fraction baseline
        base_fraction = float(
            strategy_overrides.get("base_fraction", self.params.base_risk_per_trade)
        )
        base_fraction = max(0.0, min(base_fraction, self.params.max_position_size))

        fraction = 0.0
        if sizer == "fixed_fraction":
            fraction = base_fraction
        elif sizer == "confidence_weighted":
            # Look up model confidence from indicators or df
            conf_key = strategy_overrides.get("confidence_key", "prediction_confidence")
            confidence = None
            if conf_key in indicators:
                confidence = indicators.get(conf_key)
            elif conf_key in df.columns:
                confidence = df[conf_key].iloc[index]
            try:
                confidence = (
                    float(confidence) if confidence is not None and not pd.isna(confidence) else 0.0
                )
            except Exception:
                confidence = 0.0
            fraction = base_fraction * max(0.0, min(1.0, confidence))
        elif sizer == "atr_risk":
            # Convert legacy quantity sizing to fraction-of-balance
            df = self._ensure_atr(df)
            atr = float(df["atr"].iloc[index]) if not pd.isna(df["atr"].iloc[index]) else 0.0
            px = float(price if price is not None else df["close"].iloc[index])
            qty = self.calculate_position_size(price=px, atr=atr, balance=balance, regime=regime)
            position_value = qty * px
            fraction = position_value / balance if balance > 0 else 0.0
        else:
            # Unknown sizer -> safest fallback: 0
            fraction = 0.0

        # Clamp
        fraction = max(min_fraction, min(max_fraction, fraction))
        
        # Optional correlation-based size reduction
        try:
            if correlation_ctx and fraction > 0:
                engine = correlation_ctx.get("engine")
                candidate_symbol = correlation_ctx.get("candidate_symbol")
                corr_matrix = correlation_ctx.get("corr_matrix")
                if engine is not None and candidate_symbol:
                    positions = self.positions
                    factor = float(
                        engine.compute_size_reduction_factor(
                            positions=positions,
                            corr_matrix=corr_matrix,
                            candidate_symbol=str(candidate_symbol),
                            candidate_fraction=float(fraction),
                        )
                    )
                    if factor < 1.0:
                        fraction = max(0.0, fraction * factor)
        except Exception:
            # Fail-safe: never raise from correlation logic
            logging.exception("Exception in correlation-based size reduction logic")
        return max(0.0, min(self.params.max_position_size, fraction))

    def compute_sl_tp(
        self,
        df: pd.DataFrame,
        index: int,
        entry_price: float,
        side: str = "long",
        strategy_overrides: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute stop-loss and take-profit prices.
        Priority:
          1) Explicit overrides: stop_loss_pct / take_profit_pct
          2) ATR-based SL via parameters; TP via params.default_take_profit_pct if set
          3) Fallback: (None, None) and engine/strategy may decide
        """
        strategy_overrides = strategy_overrides or {}
        stop_loss_pct = strategy_overrides.get("stop_loss_pct")
        take_profit_pct = strategy_overrides.get(
            "take_profit_pct", self.params.default_take_profit_pct
        )

        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if stop_loss_pct is not None:
            if side == "long":
                sl_price = entry_price * (1 - float(stop_loss_pct))
            else:
                sl_price = entry_price * (1 + float(stop_loss_pct))
        else:
            # ATR-based if available
            df = self._ensure_atr(df)
            atr_value = (
                float(df["atr"].iloc[index])
                if "atr" in df.columns and not pd.isna(df["atr"].iloc[index])
                else 0.0
            )
            if atr_value > 0:
                sl_price = self.calculate_stop_loss(
                    entry_price=entry_price, atr=atr_value, side=side
                )

        if take_profit_pct is not None:
            if side == "long":
                tp_price = entry_price * (1 + float(take_profit_pct))
            else:
                tp_price = entry_price * (1 - float(take_profit_pct))
        else:
            tp_price = None

        return sl_price, tp_price

    def check_drawdown(self, current_balance: float, peak_balance: float) -> bool:
        """Check if maximum drawdown has been exceeded"""
        if peak_balance == 0:
            return False

        drawdown = (peak_balance - current_balance) / peak_balance
        return drawdown > self.params.max_drawdown

    def update_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Update position tracking.

        Parameters
        ----------
        symbol : str
            Symbol identifier (e.g., 'BTCUSDT').
        side : str
            'long' or 'short'.
        size : float
            Fraction of account balance allocated to this position (0..1).
            This function treats `size` as a fraction for daily risk accounting.
        entry_price : float
            Entry price of the position.

        Notes
        -----
        - Daily risk accounting increments `daily_risk_used` by `size` so that
          the sum of concurrently open position fractions respects
          `params.max_daily_risk` across multiple positions.
        """
        self.positions[symbol] = {"side": side, "size": size, "entry_price": entry_price}

        # Update daily risk used (approximate: count fraction of balance put at risk)
        self.daily_risk_used += size  # size here is treated as fraction of balance allocated

    def close_position(self, symbol: str):
        """Close position tracking"""
        if symbol in self.positions:
            del self.positions[symbol]

    def get_total_exposure(self) -> float:
        """Calculate total position exposure (sum of fractions)"""
        return float(sum(pos["size"] for pos in self.positions.values()))

    def get_position_correlation_risk(self, symbols: list, corr_matrix: pd.DataFrame | None = None, threshold: Optional[float] = None) -> float:
        """Calculate correlated exposure across provided symbols.

        If a correlation matrix is provided, group symbols whose pairwise correlation
        exceeds the threshold (defaults to params.correlation_threshold) and return
        the maximum group exposure among the groups that intersect the input symbols.
        Fallback: sum exposures of provided symbols.
        """
        if not symbols:
            return 0.0
        try:
            sym_set = set(map(str, symbols))
            # Fallback: sum exposures for given symbols
            if corr_matrix is None or corr_matrix.empty:
                exposure = sum(
                    float(pos.get("size", 0.0))
                    for s, pos in self.positions.items()
                    if s in sym_set
                )
                return round(float(exposure), 8)

            thr = float(self.params.correlation_threshold if threshold is None else threshold)
            cols = [c for c in corr_matrix.columns if c in sym_set]
            if len(cols) < 1:
                return 0.0
            parent = {s: s for s in cols}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: str, b: str) -> None:
                ra = find(a)
                rb = find(b)
                if ra != rb:
                    parent[rb] = ra

            # Build unions among provided symbols
            for i, a in enumerate(cols):
                for j in range(i + 1, len(cols)):
                    b = cols[j]
                    val = corr_matrix.at[a, b] if (a in corr_matrix.index and b in corr_matrix.columns) else None
                    if pd.notna(val) and float(val) >= thr:
                        union(a, b)

            groups: dict[str, list[str]] = {}
            for s in cols:
                root = find(s)
                groups.setdefault(root, []).append(s)

            # Compute exposures per group; return the maximum group exposure
            max_exposure = 0.0
            for g in groups.values():
                if not g:
                    continue
                total = 0.0
                for s in g:
                    total += float(self.positions.get(s, {}).get("size", 0.0))
                max_exposure = max(max_exposure, total)
            # If no groups formed (all singletons), fall back to sum of the specified symbols
            if max_exposure == 0.0 and len(groups) == len(cols):
                max_exposure = sum(float(self.positions.get(s, {}).get("size", 0.0)) for s in cols)
            return round(max_exposure, 8)
        except Exception:
            # Fail-safe
            exposure = sum(
                float(pos.get("size", 0.0)) for s, pos in self.positions.items() if s in symbols
            )
            return round(float(exposure), 8)

    def get_max_concurrent_positions(self) -> int:
        """Return the maximum number of concurrent positions allowed."""
        return self.max_concurrent_positions
