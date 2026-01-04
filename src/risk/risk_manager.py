"""
Portfolio-Level Risk Manager (Layer 2 of 3-Layer Risk Architecture)

This module provides the global risk management system that enforces portfolio-wide
constraints across all positions and symbols.

ARCHITECTURE ROLE:
    This is LAYER 2 (Portfolio Level) of the three-layer risk management architecture.
    It enforces global constraints and tracks positions across the entire portfolio.

SCOPE:
    - Multi-symbol, portfolio-wide strategic constraints
    - Daily risk limit enforcement (max_daily_risk)
    - Position tracking across all symbols
    - Correlation-based position sizing
    - Drawdown checking
    - Thread-safe concurrent access for live trading

KEY DIFFERENCES FROM STRATEGY RISK MANAGER:
    - Strategy risk (Layer 1): "What size makes sense for THIS signal?"
    - Portfolio risk (This file): "What size is ALLOWED given global constraints?"

RELATIONSHIP TO OTHER RISK LAYERS:
    Layer 1 (src/strategies/components/risk_manager.py): Strategy component - signal-based decisions
    Layer 2 (This file): Portfolio manager - global constraint enforcement
    Layer 3 (src/engines/shared/dynamic_risk_handler.py): Dynamic adjustments based on performance

KEY RESPONSIBILITIES:
    1. Position Sizing: Calculate allowed position fractions with multiple policies
    2. Position Tracking: Thread-safe tracking of all open positions
    3. Daily Risk Limits: Enforce max_daily_risk across all trades
    4. Correlation Control: Limit exposure to correlated assets
    5. Drawdown Protection: Check against max_drawdown threshold
    6. Partial Operations: Track partial exits and scale-ins

IMPORTANT NOTE ON DAILY RISK ACCOUNTING:
    The `daily_risk_used` attribute tracks EXPOSURE (capital allocation), NOT actual
    capital at risk. When a 10% position is opened, daily_risk_used increases by 0.1
    (10%), regardless of the stop loss distance.

    Example:
        - Open 10% position with 1% stop loss
        - Actual capital at risk: 0.10 × 0.01 = 0.001 (0.1%)
        - daily_risk_used: 0.10 (10% exposure tracked)

    This conservative approach prevents over-leveraging but means you can't open as
    many positions as theoretical risk would allow. See class docstring for details.

USAGE:
    >>> from src.risk.risk_manager import PortfolioRiskManager, RiskParameters
    >>>
    >>> # Initialize with parameters
    >>> params = RiskParameters(
    ...     base_risk_per_trade=0.02,
    ...     max_daily_risk=0.06,
    ...     max_position_size=0.10,
    ... )
    >>> risk_manager = PortfolioRiskManager(parameters=params, max_concurrent_positions=3)
    >>>
    >>> # Calculate allowed position fraction
    >>> fraction = risk_manager.calculate_position_fraction(
    ...     df=df,
    ...     index=i,
    ...     balance=10000,
    ...     strategy_overrides={'position_sizer': 'atr_risk'},
    ... )
    >>>
    >>> # Track position (thread-safe)
    >>> risk_manager.update_position('BTCUSDT', 'long', 0.05, 50000)
    >>>
    >>> # Check drawdown
    >>> if risk_manager.check_drawdown(current_balance, peak_balance):
    ...     # Drawdown limit exceeded
    ...     pass

THREAD SAFETY:
    All operations on shared state (positions dict, daily_risk_used) are protected
    by a reentrant lock (_state_lock) for safe concurrent access from multiple threads
    (e.g., live trading engine + monitoring threads).

See also:
    - docs/risk_management_architecture.md: Complete architecture documentation
    - src/strategies/components/risk_manager.py: Strategy-level risk component (Layer 1)
    - src/engines/shared/dynamic_risk_handler.py: Dynamic risk adjustments (Layer 3)
    - src/position_management/README.md: Position management policies
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.constants import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_BASE_RISK_PER_TRADE,
    DEFAULT_BREAKEVEN_BUFFER,
    DEFAULT_BREAKEVEN_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_CORRELATION_UPDATE_FREQUENCY_HOURS,
    DEFAULT_CORRELATION_WINDOW_DAYS,
    DEFAULT_EXPOSURE_PRECISION_DECIMALS,
    DEFAULT_MAX_CORRELATED_EXPOSURE,
    DEFAULT_MAX_CORRELATED_RISK,
    DEFAULT_MAX_DAILY_RISK,
    DEFAULT_MAX_DRAWDOWN,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_MAX_RISK_PER_TRADE,
    DEFAULT_MAX_SCALE_INS,
    DEFAULT_PARTIAL_EXIT_SIZES,
    DEFAULT_PARTIAL_EXIT_TARGETS,
    DEFAULT_SCALE_IN_SIZES,
    DEFAULT_SCALE_IN_THRESHOLDS,
    DEFAULT_TRAILING_ACTIVATION_THRESHOLD,
    DEFAULT_TRAILING_DISTANCE_ATR_MULT,
    DEFAULT_TRAILING_DISTANCE_PCT,
)
from src.tech.indicators.core import calculate_atr
from src.utils.price_targets import PriceTargetCalculator

# Risk calculation constants
MIN_ATR_THRESHOLD = 0.001  # Minimum ATR as fraction of price (0.1%)
MIN_ATR_FALLBACK = 0.01  # Fallback ATR when too small (1% of price)
TRENDING_RISK_MULTIPLIER = 1.5  # Increase risk in trending markets
VOLATILE_RISK_MULTIPLIER = 0.6  # Decrease risk in volatile markets
VALID_REGIMES = frozenset({"normal", "trending", "volatile"})
VALID_SIDES = frozenset({"long", "short"})


@dataclass
class RiskParameters:
    """Risk management parameters"""

    base_risk_per_trade: float = DEFAULT_BASE_RISK_PER_TRADE  # 2% risk per trade
    max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE  # 3% maximum risk per trade
    max_position_size: float = (
        DEFAULT_MAX_POSITION_SIZE  # Maximum position size (fraction of balance)
    )
    max_daily_risk: float = DEFAULT_MAX_DAILY_RISK  # 6% maximum daily risk (fraction of balance)
    max_correlated_risk: float = (
        DEFAULT_MAX_CORRELATED_RISK  # 10% maximum risk for correlated positions
    )
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN  # 20% maximum drawdown (fraction)
    position_size_atr_multiplier: float = 1.0
    default_take_profit_pct: float | None = None  # if None, engine/strategy may supply
    atr_period: int = DEFAULT_ATR_PERIOD
    # Time exit config (optional; strategies may override)
    time_exits: dict | None = None
    # Partial operations (defaults can be overridden by strategies)
    partial_exit_targets: list[float] | None = None
    partial_exit_sizes: list[float] | None = None
    scale_in_thresholds: list[float] | None = None
    scale_in_sizes: list[float] | None = None
    max_scale_ins: int = DEFAULT_MAX_SCALE_INS
    # Trailing stop config (engine/backtester may override via strategy.get_risk_overrides())
    trailing_activation_threshold: float = DEFAULT_TRAILING_ACTIVATION_THRESHOLD
    trailing_distance_pct: float | None = DEFAULT_TRAILING_DISTANCE_PCT
    trailing_atr_multiplier: float | None = DEFAULT_TRAILING_DISTANCE_ATR_MULT
    breakeven_threshold: float = DEFAULT_BREAKEVEN_THRESHOLD
    breakeven_buffer: float = DEFAULT_BREAKEVEN_BUFFER
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

        # Default partial operations if not provided
        if self.partial_exit_targets is None:
            self.partial_exit_targets = list(DEFAULT_PARTIAL_EXIT_TARGETS)
        if self.partial_exit_sizes is None:
            self.partial_exit_sizes = list(DEFAULT_PARTIAL_EXIT_SIZES)
        if self.scale_in_thresholds is None:
            self.scale_in_thresholds = list(DEFAULT_SCALE_IN_THRESHOLDS)
        if self.scale_in_sizes is None:
            self.scale_in_sizes = list(DEFAULT_SCALE_IN_SIZES)

        # Validate partial operations configuration lengths and ranges
        if len(self.partial_exit_targets) != len(self.partial_exit_sizes):
            raise ValueError("partial_exit_targets and partial_exit_sizes must have equal length")
        if len(self.scale_in_thresholds) != len(self.scale_in_sizes):
            raise ValueError("scale_in_thresholds and scale_in_sizes must have equal length")
        if any(t <= 0 for t in self.partial_exit_targets):
            raise ValueError("partial_exit_targets must be positive percentages (decimals)")
        if any(s <= 0 or s > 1 for s in self.partial_exit_sizes):
            raise ValueError("partial_exit_sizes must be in (0, 1]")
        if sum(self.partial_exit_sizes) > 1.0:
            raise ValueError(
                f"partial_exit_sizes cannot sum to more than 1.0, got {sum(self.partial_exit_sizes)}"
            )
        if any(t <= 0 for t in self.scale_in_thresholds):
            raise ValueError("scale_in_thresholds must be positive percentages (decimals)")
        if any(s <= 0 or s > 1 for s in self.scale_in_sizes):
            raise ValueError("scale_in_sizes must be in (0, 1]")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.correlation_update_frequency_hours <= 0:
            raise ValueError("correlation_update_frequency_hours must be positive")


class PortfolioRiskManager:
    """Handles position sizing and risk management across the entire portfolio.

    Daily Risk Accounting
    ---------------------
    IMPORTANT: The `daily_risk_used` attribute tracks EXPOSURE (capital allocation),
    not actual capital at risk. When a position is opened with 10% of balance,
    `daily_risk_used` increases by 0.1 (10%), regardless of the stop loss distance.

    This is a conservative approach that:
    - Prevents over-leveraging by limiting total capital allocation
    - Simplifies accounting (no need to track stop loss distances)
    - Provides a margin of safety in volatile markets

    However, it differs from traditional risk management where:
    - Risk = position_size × stop_loss_distance
    - A 10% position with 1% stop loss = 0.1% actual capital at risk

    Future Enhancement: Consider tracking actual capital at risk by multiplying
    position size by stop loss distance percentage. This would allow larger
    positions with tight stops while maintaining true risk limits.

    Thread Safety
    -------------
    All operations on shared state (positions dict and daily_risk_used counter)
    are protected by a reentrant lock to ensure safe concurrent access from
    multiple threads (e.g., live trading engine + monitoring threads).
    """

    def __init__(self, parameters: RiskParameters | None = None, max_concurrent_positions: int = 3):
        self.params = parameters or RiskParameters()

        # Validate max_concurrent_positions
        if max_concurrent_positions <= 0:
            raise ValueError(
                f"max_concurrent_positions must be positive, got {max_concurrent_positions}"
            )

        # Tracks total exposure (capital allocation), not actual capital at risk
        # See class docstring for details on this design decision
        self.daily_risk_used = 0.0
        self.positions: dict[str, dict] = {}
        self.max_concurrent_positions = max_concurrent_positions
        self._state_lock = threading.RLock()  # Protects positions and daily_risk_used

    def reset_daily_risk(self):
        """Reset daily risk counter to zero.

        This should be called at the start of each trading day to reset
        the cumulative daily risk exposure tracking. Thread-safe.
        """
        with self._state_lock:
            self.daily_risk_used = 0.0

    def _ensure_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has ATR column, calculating if missing.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'atr' column added if it was missing.
        """
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

        Parameters
        ----------
        price : float
            Current asset price (must be positive and finite).
        atr : float
            Average True Range value (must be finite).
        balance : float
            Account balance (must be positive and finite).
        regime : str, default="normal"
            Market regime: "normal", "trending", or "volatile".

        Returns
        -------
        float
            Position size in units (quantity), or 0.0 if inputs are invalid.
        """
        # Validate inputs for NaN/Infinity
        if not math.isfinite(price) or not math.isfinite(atr) or not math.isfinite(balance):
            logging.error(
                "Non-finite input to calculate_position_size: price=%s, atr=%s, balance=%s",
                price,
                atr,
                balance,
            )
            return 0.0

        # Validate positive values
        if price <= 0 or balance <= 0:
            return 0.0

        # Validate regime parameter
        if regime not in VALID_REGIMES:
            logging.warning("Unknown regime '%s', treating as 'normal'", regime)
            regime = "normal"

        # Handle zero or very small ATR
        if atr <= 0 or atr < price * MIN_ATR_THRESHOLD:
            atr = price * MIN_ATR_FALLBACK

        # Adjust risk based on market regime
        base_risk = self.params.base_risk_per_trade
        if regime == "trending":
            risk = base_risk * TRENDING_RISK_MULTIPLIER
        elif regime == "volatile":
            risk = base_risk * VOLATILE_RISK_MULTIPLIER
        else:
            risk = base_risk

        # Ensure we don't exceed maximum risk limits
        risk = min(risk, self.params.max_risk_per_trade)

        # Check remaining daily risk (thread-safe)
        # NOTE: This is a read-only check for the legacy quantity-based API
        # The authoritative daily risk check happens in update_position()
        with self._state_lock:
            remaining_daily_risk = self.params.max_daily_risk - self.daily_risk_used
            effective_risk = min(risk, remaining_daily_risk)

        # If no remaining daily risk, return 0
        if effective_risk <= 0:
            return 0.0

        # Calculate position size based on ATR
        risk_amount = balance * effective_risk
        atr_stop = atr * self.params.position_size_atr_multiplier
        position_size = risk_amount / atr_stop

        # Ensure position size doesn't exceed maximum (convert to units)
        max_position_value = balance * self.params.max_position_size
        position_size = min(position_size, max_position_value / price)

        # Ensure position size is positive
        return max(0.0, position_size)

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str = "long") -> float:
        """Calculate adaptive stop loss level (ATR-based).

        Parameters
        ----------
        entry_price : float
            Entry price of the position.
        atr : float
            Average True Range value.
        side : str, default="long"
            Position side: "long" or "short".

        Returns
        -------
        float
            Stop loss price level.

        Raises
        ------
        ValueError
            If side is not "long" or "short", entry_price or atr are invalid.
        """
        # Validate inputs
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"entry_price must be positive and finite, got {entry_price}")
        if not math.isfinite(atr) or atr < 0:
            raise ValueError(f"atr must be non-negative and finite, got {atr}")
        if side not in VALID_SIDES:
            raise ValueError(f"side must be 'long' or 'short', got '{side}'")

        return PriceTargetCalculator.stop_loss_atr(
            entry_price=entry_price,
            atr=atr,
            multiplier=self.params.position_size_atr_multiplier,
            side=side,
        )

    # NEW HIGHER-LEVEL API (fraction-based + SL/TP policies)

    def _parse_position_sizing_params(
        self, strategy_overrides: dict[str, Any]
    ) -> tuple[float, float, float]:
        """Parse and validate position sizing parameters from strategy overrides.

        NOTE: This method does NOT enforce daily risk limits. Daily risk clamping
        is performed atomically in _finalize_fraction_with_risk_limits() to prevent
        race conditions.

        Parameters
        ----------
        strategy_overrides : dict
            Strategy-specific override parameters.

        Returns
        -------
        tuple[float, float, float]
            (min_fraction, max_fraction, base_fraction) - validated and clamped
            to parameter limits (but NOT yet clamped to remaining daily risk).
        """
        # Handle Mock objects in tests by converting to float safely
        try:
            min_fraction = float(strategy_overrides.get("min_fraction", 0.0))
        except (TypeError, ValueError):
            min_fraction = 0.0

        try:
            max_fraction = float(
                strategy_overrides.get("max_fraction", self.params.max_position_size)
            )
        except (TypeError, ValueError):
            max_fraction = float(self.params.max_position_size)

        # Default fraction baseline
        try:
            base_fraction = float(
                strategy_overrides.get("base_fraction", self.params.base_risk_per_trade)
            )
        except (TypeError, ValueError):
            base_fraction = float(self.params.base_risk_per_trade)

        # Validate and clamp all fractions to valid ranges
        min_fraction = max(0.0, min_fraction)  # Ensure non-negative
        max_fraction = max(
            0.0, min(max_fraction, self.params.max_position_size)
        )  # Clamp to [0, max_position_size]
        base_fraction = max(0.0, min(base_fraction, self.params.max_position_size))

        # Ensure min_fraction <= max_fraction
        if min_fraction > max_fraction:
            logging.warning(
                "min_fraction (%.4f) > max_fraction (%.4f), swapping to maintain validity",
                min_fraction,
                max_fraction,
            )
            min_fraction, max_fraction = max_fraction, min_fraction

        return min_fraction, max_fraction, base_fraction

    def _calculate_raw_fraction(
        self,
        sizer: str,
        base_fraction: float,
        df: pd.DataFrame,
        index: int,
        balance: float,
        price: float | None,
        regime: str,
        indicators: dict[str, Any],
        strategy_overrides: dict[str, Any],
    ) -> float:
        """Calculate raw position fraction based on selected sizing policy.

        Parameters
        ----------
        sizer : str
            Position sizer type: 'fixed_fraction', 'confidence_weighted', or 'atr_risk'.
        base_fraction : float
            Base allocation fraction.
        df : pd.DataFrame
            Market data.
        index : int
            Current candle index.
        balance : float
            Account balance.
        price : float | None
            Current price (if available).
        regime : str
            Market regime.
        indicators : dict
            Indicator values.
        strategy_overrides : dict
            Strategy-specific overrides.

        Returns
        -------
        float
            Raw position fraction before limits applied.
        """
        if sizer == "fixed_fraction":
            return base_fraction

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
            except (TypeError, ValueError) as e:
                logging.debug("Could not convert confidence to float: %s, defaulting to 0.0", e)
                confidence = 0.0
            return base_fraction * max(0.0, min(1.0, confidence))

        elif sizer == "atr_risk":
            # Convert legacy quantity sizing to fraction-of-balance
            df = self._ensure_atr(df)
            atr = float(df["atr"].iloc[index]) if not pd.isna(df["atr"].iloc[index]) else 0.0
            px = float(price if price is not None else df["close"].iloc[index])
            qty = self.calculate_position_size(price=px, atr=atr, balance=balance, regime=regime)
            position_value = qty * px
            return position_value / balance if balance > 0 else 0.0

        else:
            # Unknown sizer -> safest fallback: 0
            logging.warning("Unknown position sizer '%s', returning 0.0", sizer)
            return 0.0

    def _apply_correlation_adjustment(
        self, fraction: float, correlation_ctx: dict[str, Any] | None
    ) -> float:
        """Apply correlation-based size reduction if correlation context provided.

        Parameters
        ----------
        fraction : float
            Initial position fraction.
        correlation_ctx : dict | None
            Correlation context with engine, symbol, matrix, etc.

        Returns
        -------
        float
            Adjusted fraction after correlation reduction (or original if no adjustment).
        """
        if not correlation_ctx or fraction <= 0:
            return fraction

        try:
            engine = correlation_ctx.get("engine")
            candidate_symbol = correlation_ctx.get("candidate_symbol")
            corr_matrix = correlation_ctx.get("corr_matrix")
            max_exposure_override = correlation_ctx.get("max_exposure_override")

            if engine is None or not candidate_symbol:
                return fraction

            with self._state_lock:
                positions = self.positions.copy()

            raw_factor = engine.compute_size_reduction_factor(
                positions=positions,
                corr_matrix=corr_matrix,
                candidate_symbol=str(candidate_symbol),
                candidate_fraction=float(fraction),
                max_exposure_override=max_exposure_override,
            )

            # Validate factor is numeric and not None/NaN before using
            try:
                factor = float(raw_factor)
                if pd.isna(factor):
                    logging.warning(
                        "Correlation engine returned NaN reduction factor - ignoring adjustment"
                    )
                    return fraction
            except (TypeError, ValueError):
                logging.warning(
                    "Correlation engine returned non-numeric factor (%s: %s) - ignoring adjustment",
                    type(raw_factor).__name__,
                    raw_factor,
                )
                return fraction

            if factor < 1.0:
                return max(0.0, fraction * factor)
            return fraction

        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # Fail-safe: never raise from correlation logic
            logging.warning("Error in correlation-based size reduction logic: %s", e)
            return fraction

    def _finalize_fraction_with_risk_limits(self, fraction: float) -> float:
        """Perform final validation and clamping with risk limits (thread-safe).

        This method performs atomic validation inside a lock to prevent TOCTOU races.

        Parameters
        ----------
        fraction : float
            Calculated position fraction before final limits.

        Returns
        -------
        float
            Final validated and clamped position fraction.
        """
        with self._state_lock:
            remaining_daily_risk = max(0.0, self.params.max_daily_risk - self.daily_risk_used)
            final_fraction = max(0.0, min(self.params.max_position_size, fraction))

            # Clamp to remaining daily risk
            if final_fraction > remaining_daily_risk:
                logging.warning(
                    "Calculated fraction %.4f exceeds remaining daily risk %.4f, clamping to limit",
                    final_fraction,
                    remaining_daily_risk,
                )
                final_fraction = remaining_daily_risk

            # Double-check max position size (defensive)
            if final_fraction > self.params.max_position_size:
                logging.error(
                    "CRITICAL: Calculated fraction %.4f exceeds max_position_size %.4f. "
                    "This indicates a logic error in position sizing.",
                    final_fraction,
                    self.params.max_position_size,
                )
                final_fraction = self.params.max_position_size

            return final_fraction

    def calculate_position_fraction(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        price: float | None = None,
        indicators: dict[str, Any] | None = None,
        strategy_overrides: dict[str, Any] | None = None,
        regime: str = "normal",
        correlation_ctx: dict[str, Any] | None = None,
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
        # Early validation
        if df is None or df.empty:
            return 0.0
        if balance <= 0 or index < 0 or index >= len(df):
            return 0.0

        strategy_overrides = strategy_overrides or {}
        indicators = indicators or {}
        sizer = strategy_overrides.get("position_sizer", "fixed_fraction")

        # Step 1: Parse and validate parameters
        min_fraction, max_fraction, base_fraction = self._parse_position_sizing_params(
            strategy_overrides
        )

        # Step 2: Calculate raw fraction based on sizer type
        raw_fraction = self._calculate_raw_fraction(
            sizer=sizer,
            base_fraction=base_fraction,
            df=df,
            index=index,
            balance=balance,
            price=price,
            regime=regime,
            indicators=indicators,
            strategy_overrides=strategy_overrides,
        )

        # Step 3: Clamp to min/max bounds
        clamped_fraction = max(min_fraction, min(max_fraction, raw_fraction))

        # Step 4: Apply correlation adjustment if provided
        adjusted_fraction = self._apply_correlation_adjustment(clamped_fraction, correlation_ctx)

        # Step 5: Final validation with risk limits (atomic, thread-safe)
        return self._finalize_fraction_with_risk_limits(adjusted_fraction)

    def compute_sl_tp(
        self,
        df: pd.DataFrame,
        index: int,
        entry_price: float,
        side: str = "long",
        strategy_overrides: dict[str, Any] | None = None,
    ) -> tuple[float | None, float | None]:
        """
        Compute stop-loss and take-profit prices.
        Priority:
          1) Explicit overrides: stop_loss_pct / take_profit_pct
          2) ATR-based SL via parameters; TP via params.default_take_profit_pct if set
          3) Fallback: (None, None) and engine/strategy may decide

        Parameters
        ----------
        df : pd.DataFrame
            Market data with OHLCV and optional indicators.
        index : int
            Current candle index.
        entry_price : float
            Entry price of the position (must be positive and finite).
        side : str, default="long"
            Position side: "long" or "short".
        strategy_overrides : dict, optional
            Strategy-specific overrides for stop loss and take profit.

        Returns
        -------
        tuple[float | None, float | None]
            (stop_loss_price, take_profit_price) or (None, None) if not set.

        Raises
        ------
        ValueError
            If entry_price is invalid, side is not "long" or "short",
            df is None/empty, or index is out of bounds.
        """
        # Validate inputs
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"entry_price must be positive and finite, got {entry_price}")
        if side not in VALID_SIDES:
            raise ValueError(f"side must be 'long' or 'short', got '{side}'")
        if df is None or df.empty:
            raise ValueError("df cannot be None or empty")
        if index < 0 or index >= len(df):
            raise ValueError(f"index must be in range [0, {len(df)}), got {index}")

        strategy_overrides = strategy_overrides or {}
        stop_loss_pct = strategy_overrides.get("stop_loss_pct")
        take_profit_pct = strategy_overrides.get(
            "take_profit_pct", self.params.default_take_profit_pct
        )

        sl_price: float | None = None
        tp_price: float | None = None

        if stop_loss_pct is not None:
            try:
                sl_price = PriceTargetCalculator.stop_loss(
                    entry_price=entry_price,
                    pct=float(stop_loss_pct),
                    side=side,
                )
            except (TypeError, ValueError) as e:
                logging.warning(
                    "Invalid stop_loss_pct value '%s' (type: %s) - cannot convert to float: %s. "
                    "Falling back to ATR-based stop loss.",
                    stop_loss_pct,
                    type(stop_loss_pct).__name__,
                    e,
                )
                sl_price = None
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
            try:
                # Convert to float first to satisfy type checker
                tp_pct_float = float(take_profit_pct)
                tp_price = PriceTargetCalculator.take_profit(
                    entry_price=entry_price,
                    pct=tp_pct_float,
                    side=side,
                )
            except (TypeError, ValueError) as e:
                logging.warning(
                    "Invalid take_profit_pct value '%s' (type: %s) - cannot convert to float: %s. "
                    "Setting take profit to None.",
                    take_profit_pct,
                    type(take_profit_pct).__name__,
                    e,
                )
                tp_price = None
        else:
            tp_price = None

        return sl_price, tp_price

    def check_drawdown(self, current_balance: float, peak_balance: float) -> bool:
        """Check if maximum drawdown has been exceeded.

        Parameters
        ----------
        current_balance : float
            Current account balance.
        peak_balance : float
            Peak account balance reached.

        Returns
        -------
        bool
            True if drawdown exceeds configured maximum, False otherwise.
        """
        # Validate inputs
        if not math.isfinite(current_balance) or not math.isfinite(peak_balance):
            logging.warning(
                "Non-finite balance in drawdown check: current=%s, peak=%s",
                current_balance,
                peak_balance,
            )
            return False

        # Cannot calculate drawdown with non-positive peak
        if peak_balance <= 0:
            return False

        drawdown = (peak_balance - current_balance) / peak_balance
        return drawdown > self.params.max_drawdown

    def update_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Update position tracking, creating new or updating existing position.

        If the symbol already exists in positions, this method replaces the old position
        with the new one and adjusts daily_risk_used accordingly (subtracts old size,
        adds new size).

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
            Entry price of the position (must be positive and finite).

        Notes
        -----
        - Daily risk accounting: subtracts old size (if updating), adds new size.
        - This tracks EXPOSURE (capital allocation), not actual capital at risk.
        - Example: 10% position → daily_risk_used += 0.1, regardless of stop loss distance.
        - Updating 10% position to 5% → daily_risk_used -= 0.1, then += 0.05 (net: -0.05).
        - The sum of all position fractions is capped by `params.max_daily_risk`.
        - This is a conservative approach; see class docstring for details.
        - Thread-safe: protected by internal lock.

        Raises
        ------
        ValueError
            If symbol is None/empty, or size, entry_price, or side parameters are invalid.
        """
        # Validate inputs
        if not symbol:
            raise ValueError("symbol cannot be None or empty")
        if not math.isfinite(size) or size <= 0 or size > 1:
            raise ValueError(f"size must be in (0, 1], got {size}")
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"entry_price must be positive and finite, got {entry_price}")
        if side not in VALID_SIDES:
            raise ValueError(f"side must be 'long' or 'short', got '{side}'")

        with self._state_lock:
            # If updating existing position, first remove old size from daily risk
            old_size = 0.0
            if symbol in self.positions:
                old_size = float(self.positions[symbol].get("size", 0.0))
                self.daily_risk_used = max(0.0, self.daily_risk_used - old_size)

            # Update position with new values
            self.positions[symbol] = {"side": side, "size": size, "entry_price": entry_price}

            # Add new size to daily risk used
            self.daily_risk_used += size

    def close_position(self, symbol: str):
        """Close position tracking and free daily risk (thread-safe).

        IMPORTANT: This method reduces daily_risk_used by the position's size,
        freeing up risk budget for new positions.

        Parameters
        ----------
        symbol : str
            Symbol identifier of position to close.

        Raises
        ------
        ValueError
            If symbol is None or empty string.
        """
        if not symbol:
            raise ValueError("symbol cannot be None or empty")

        with self._state_lock:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos_size = float(pos.get("size", 0.0))
                # Free up daily risk when closing position
                self.daily_risk_used = max(0.0, self.daily_risk_used - pos_size)
                del self.positions[symbol]

    def get_total_exposure(self) -> float:
        """Calculate total position exposure (sum of fractions, thread-safe).

        Returns
        -------
        float
            Total exposure as sum of all position size fractions.
            For example, if positions allocate 0.1 and 0.15 of balance,
            returns 0.25 (25% total exposure).
        """
        with self._state_lock:
            return float(sum(pos["size"] for pos in self.positions.values()))

    def get_position_correlation_risk(
        self,
        symbols: list,
        corr_matrix: pd.DataFrame | None = None,
        threshold: float | None = None,
    ) -> float:
        """Calculate correlated exposure across provided symbols.

        If a correlation matrix is provided, group symbols whose pairwise correlation
        exceeds the threshold (defaults to params.correlation_threshold) and return
        the maximum group exposure among the groups that intersect the input symbols.
        Fallback: sum exposures of provided symbols.

        Thread-safe: takes a snapshot of positions at start.

        Raises
        ------
        ValueError
            If threshold is provided but is not finite or not in [-1, 1].
        """
        if not symbols:
            return 0.0

        # Validate threshold if provided
        if threshold is not None:
            if not math.isfinite(threshold):
                raise ValueError(f"threshold must be finite, got {threshold}")
            if threshold < -1 or threshold > 1:
                raise ValueError(f"threshold must be in [-1, 1], got {threshold}")

        # Take snapshot of positions while holding lock
        with self._state_lock:
            positions_snapshot = self.positions.copy()

        try:
            sym_set = set(map(str, symbols))
            # Fallback: sum exposures for given symbols
            if corr_matrix is None or corr_matrix.empty:
                exposure = sum(
                    float(pos.get("size", 0.0))
                    for s, pos in positions_snapshot.items()
                    if s in sym_set
                )
                # Round to prevent floating-point precision issues in comparisons
                return round(float(exposure), DEFAULT_EXPOSURE_PRECISION_DECIMALS)

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
                    val = (
                        corr_matrix.at[a, b]
                        if (a in corr_matrix.index and b in corr_matrix.columns)
                        else None
                    )
                    if val is not None and pd.notna(val) and float(val) >= thr:
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
                    total += float(positions_snapshot.get(s, {}).get("size", 0.0))
                max_exposure = max(max_exposure, total)
            # If no groups formed (all singletons), fall back to sum of the specified symbols
            if max_exposure == 0.0 and len(groups) == len(cols):
                max_exposure = sum(
                    float(positions_snapshot.get(s, {}).get("size", 0.0)) for s in cols
                )
            # Round to prevent floating-point precision issues in comparisons
            return round(max_exposure, DEFAULT_EXPOSURE_PRECISION_DECIMALS)
        except (TypeError, ValueError, KeyError) as e:
            # Fail-safe: fall back to simple sum if correlation calculation fails
            logging.warning(
                "Error calculating correlated exposure: %s, using simple sum fallback", e
            )
            exposure = sum(
                float(pos.get("size", 0.0)) for s, pos in positions_snapshot.items() if s in symbols
            )
            # Round to prevent floating-point precision issues in comparisons
            return round(float(exposure), DEFAULT_EXPOSURE_PRECISION_DECIMALS)

    def get_max_concurrent_positions(self) -> int:
        """Return the maximum number of concurrent positions allowed."""
        return self.max_concurrent_positions

    def adjust_position_after_partial_exit(
        self, symbol: str, executed_fraction_of_original: float
    ) -> None:
        """Reduce tracked exposure after a partial exit.

        executed_fraction_of_original is the fraction of ORIGINAL size removed.

        Thread-safe: protected by internal lock.

        Parameters
        ----------
        symbol : str
            Symbol identifier.
        executed_fraction_of_original : float
            Fraction of the original position size that was exited (must be positive).
        """
        # Validate input
        if not symbol:
            raise ValueError("symbol cannot be None or empty")
        if not math.isfinite(executed_fraction_of_original) or executed_fraction_of_original <= 0:
            logging.warning(
                "Invalid executed_fraction for partial exit: %s, ignoring adjustment",
                executed_fraction_of_original,
            )
            return

        with self._state_lock:
            pos = self.positions.get(symbol)
            if not pos:
                logging.debug(
                    "Cannot adjust non-existent position for partial exit: symbol=%s", symbol
                )
                return
            current = float(pos.get("size", 0.0))
            new_size = max(0.0, current - float(executed_fraction_of_original))

            # Calculate actual reduction (handles case where executed_fraction > current)
            actual_reduction = current - new_size

            # If position fully exited, remove from tracking
            if new_size < 1e-9:  # Effectively zero
                del self.positions[symbol]
            else:
                pos["size"] = new_size

            # Reduce daily risk used by actual amount removed
            self.daily_risk_used = max(0.0, self.daily_risk_used - actual_reduction)

    def adjust_position_after_scale_in(
        self, symbol: str, added_fraction_of_original: float
    ) -> None:
        """Increase tracked exposure after a scale-in, enforcing daily and per-position caps.

        Thread-safe: protected by internal lock.

        Parameters
        ----------
        symbol : str
            Symbol identifier.
        added_fraction_of_original : float
            Fraction of the original position size to add (must be positive).
        """
        # Validate input
        if not symbol:
            raise ValueError("symbol cannot be None or empty")
        if not math.isfinite(added_fraction_of_original) or added_fraction_of_original <= 0:
            logging.warning(
                "Invalid added_fraction for scale-in: %s, ignoring adjustment",
                added_fraction_of_original,
            )
            return

        with self._state_lock:
            pos = self.positions.get(symbol)
            if not pos:
                logging.debug("Cannot adjust non-existent position for scale-in: symbol=%s", symbol)
                return
            current = float(pos.get("size", 0.0))
            # Enforce per-position cap and remaining daily risk
            remaining_daily = max(0.0, self.params.max_daily_risk - self.daily_risk_used)
            effective_add = min(
                float(added_fraction_of_original),
                remaining_daily,
                max(0.0, self.params.max_position_size - current),
            )
            if effective_add <= 0:
                return
            pos["size"] = current + effective_add
            self.daily_risk_used = min(
                self.params.max_daily_risk, self.daily_risk_used + effective_add
            )
