"""
Risk Manager Components

This module defines the abstract RiskManager interface and related data models
for managing position sizing, stop losses, and risk controls in the component-based
strategy architecture.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from src.config.constants import DEFAULT_BASE_RISK_PER_TRADE, DEFAULT_STOP_LOSS_PCT

if TYPE_CHECKING:
    from .regime_context import RegimeContext
    from .runtime import FeatureGeneratorSpec
    from .signal_generator import Signal


@dataclass
class Position:
    """
    Data class representing a trading position

    Attributes:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        side: Position side ('long' or 'short')
        size: Position size in base currency
        entry_price: Entry price for the position
        current_price: Current market price
        entry_time: Timestamp when position was opened
        unrealized_pnl: Current unrealized profit/loss
        realized_pnl: Realized profit/loss from partial exits
    """

    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        """Validate position parameters after initialization"""
        self._validate_position()

    def _validate_position(self):
        """Validate position parameters are within acceptable bounds"""
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")

        if self.side not in ["long", "short"]:
            raise ValueError(f"side must be 'long' or 'short', got {self.side}")

        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")

        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {self.entry_price}")

        if self.current_price <= 0:
            raise ValueError(f"current_price must be positive, got {self.current_price}")

    def update_current_price(self, price: float) -> None:
        """Update current price and recalculate unrealized PnL"""
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        self.current_price = price

        # Calculate unrealized PnL
        if self.side == "long":
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.size

    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl

    def get_pnl_percentage(self) -> float:
        """Get PnL as percentage of entry value"""
        entry_value = self.entry_price * self.size
        return (self.get_total_pnl() / entry_value) * 100 if entry_value > 0 else 0.0


@dataclass
class MarketData:
    """
    Data class representing current market data

    Attributes:
        symbol: Trading symbol
        price: Current price
        volume: Current volume
        bid: Current bid price
        ask: Current ask price
        timestamp: Data timestamp
        volatility: Current volatility measure (e.g., ATR)
    """

    symbol: str
    price: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    timestamp: datetime | None = None
    volatility: float | None = None

    def __post_init__(self):
        """Validate market data parameters after initialization"""
        self._validate_market_data()

    def _validate_market_data(self):
        """Validate market data parameters are within acceptable bounds"""
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")

        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")

        if self.volume < 0:
            raise ValueError(f"volume must be non-negative, got {self.volume}")

        if self.bid is not None and self.bid <= 0:
            raise ValueError(f"bid must be positive when provided, got {self.bid}")

        if self.ask is not None and self.ask <= 0:
            raise ValueError(f"ask must be positive when provided, got {self.ask}")

        if self.volatility is not None and self.volatility < 0:
            raise ValueError(
                f"volatility must be non-negative when provided, got {self.volatility}"
            )

    def get_spread(self) -> float | None:
        """Get bid-ask spread if both bid and ask are available"""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    def get_spread_percentage(self) -> float | None:
        """Get bid-ask spread as percentage of mid price"""
        spread = self.get_spread()
        if spread is not None and self.bid is not None and self.ask is not None:
            mid_price = (self.bid + self.ask) / 2
            return (spread / mid_price) * 100 if mid_price > 0 else None
        return None


class RiskManager(ABC):
    """
    Abstract base class for risk managers

    Risk managers are responsible for calculating position sizes, determining
    exit conditions, and managing stop losses based on risk parameters and
    market conditions.
    """

    def __init__(self, name: str):
        """
        Initialize the risk manager

        Args:
            name: Unique name for this risk manager
        """
        self.name = name

    @abstractmethod
    def calculate_position_size(
        self,
        signal: "Signal",
        balance: float,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """
        Calculate position size based on signal strength and risk parameters

        Args:
            signal: Trading signal with strength and confidence
            balance: Available account balance
            regime: Optional regime context for regime-aware sizing

        Returns:
            Position size in base currency

        Raises:
            ValueError: If input parameters are invalid
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> bool:
        """
        Determine if a position should be exited based on risk criteria

        Args:
            position: Current position information
            current_data: Current market data
            regime: Optional regime context for regime-aware exit decisions

        Returns:
            True if position should be exited, False otherwise
        """
        pass

    @abstractmethod
    def get_stop_loss(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """
        Calculate stop loss level for a new position

        Args:
            entry_price: Entry price for the position
            signal: Trading signal that triggered the position
            regime: Optional regime context for regime-aware stop loss

        Returns:
            Stop loss price level

        Raises:
            ValueError: If input parameters are invalid
        """
        pass

    def validate_inputs(self, balance: float) -> None:
        """
        Validate common input parameters

        Args:
            balance: Account balance to validate

        Raises:
            ValueError: If balance is invalid
        """
        if balance <= 0:
            raise ValueError(f"balance must be positive, got {balance}")

    def get_parameters(self) -> dict[str, Any]:
        """
        Get risk manager parameters for logging and serialization

        Returns:
            Dictionary of parameter names and values
        """
        return {"name": self.name, "type": self.__class__.__name__}

    @property
    def warmup_period(self) -> int:
        """Return the minimum history required for the risk manager."""

        return 0

    def get_feature_generators(self) -> Sequence["FeatureGeneratorSpec"]:
        """Return feature generators used by the risk manager."""

        return []

    def get_take_profit(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Return a take-profit level for the given signal.

        Implementations should return the entry price when no explicit take profit
        is available so that downstream consumers can treat the value as optional.
        """

        return entry_price

    def get_position_policies(
        self,
        signal: "Signal",
        balance: float,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> Any:
        """Return optional position-management policy descriptors.

        Concrete implementations may override this hook to surface structured
        policy information (e.g., partial exits, trailing stops). The default
        implementation returns ``None`` to indicate that no additional policies
        are supplied by the risk manager.
        """

        return None


class FixedRiskManager(RiskManager):
    """
    Simple fixed-risk manager for testing

    Uses fixed percentage risk per trade and simple stop loss rules
    """

    def __init__(
        self,
        risk_per_trade: float = DEFAULT_BASE_RISK_PER_TRADE,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    ):
        """
        Initialize fixed risk manager

        Args:
            risk_per_trade: Risk percentage per trade (0.02 = 2%)
            stop_loss_pct: Stop loss percentage (0.05 = 5%)
        """
        super().__init__("fixed_risk_manager")

        if not 0.001 <= risk_per_trade <= 0.1:  # 0.1% to 10%
            raise ValueError(f"risk_per_trade must be between 0.001 and 0.1, got {risk_per_trade}")

        if not 0.01 <= stop_loss_pct <= 0.5:  # 1% to 50%
            raise ValueError(f"stop_loss_pct must be between 0.01 and 0.5, got {stop_loss_pct}")

        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct

    def calculate_position_size(
        self,
        signal: "Signal",
        balance: float,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate position size based on fixed risk percentage"""
        self.validate_inputs(balance)

        if signal.direction.value == "hold":
            return 0.0

        # Base risk amount
        risk_amount = balance * self.risk_per_trade

        # Adjust for signal confidence (lower confidence = smaller position)
        confidence_multiplier = max(0.1, signal.confidence)  # Minimum 10% of base size

        # Adjust for signal strength
        strength_multiplier = max(0.1, signal.strength)  # Minimum 10% of base size

        # Calculate final position size
        position_size = risk_amount * confidence_multiplier * strength_multiplier

        # Apply regime-based adjustments if available
        if regime is not None:
            regime_multiplier = self._get_regime_multiplier(regime)
            position_size *= regime_multiplier

        # Ensure minimum position size
        min_position = balance * 0.001  # 0.1% minimum
        max_position = balance * 0.1  # 10% maximum

        return max(min_position, min(max_position, position_size))

    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> bool:
        """Determine exit based on stop loss percentage"""
        # Calculate current loss percentage
        loss_pct = abs(position.get_pnl_percentage()) / 100

        # Exit if loss exceeds stop loss threshold
        if position.get_pnl_percentage() < 0 and loss_pct >= self.stop_loss_pct:
            return True

        return False

    def get_stop_loss(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate stop loss based on fixed percentage"""
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        if signal.direction.value == "buy":
            # For long positions, stop loss is below entry price
            return entry_price * (1 - self.stop_loss_pct)
        elif signal.direction.value == "sell":
            # For short positions, stop loss is above entry price
            return entry_price * (1 + self.stop_loss_pct)
        else:
            # No stop loss for hold signals
            return entry_price

    def _get_regime_multiplier(self, regime: "RegimeContext") -> float:
        """Get position size multiplier based on regime"""
        multiplier = 1.0

        # Reduce size in high volatility
        if hasattr(regime, "volatility") and regime.volatility.value == "high_vol":
            multiplier *= 0.7

        # Reduce size in bear markets
        if hasattr(regime, "trend") and regime.trend.value == "trend_down":
            multiplier *= 0.8

        # Reduce size when regime confidence is low
        if hasattr(regime, "confidence") and regime.confidence < 0.5:
            multiplier *= 0.9

        return max(0.2, multiplier)  # Minimum 20% of base size

    def get_parameters(self) -> dict[str, Any]:
        """Get fixed risk manager parameters"""
        params = super().get_parameters()
        params.update({"risk_per_trade": self.risk_per_trade, "stop_loss_pct": self.stop_loss_pct})
        return params

    def get_take_profit(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Return a symmetric take-profit target unless signal indicates hold."""

        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        # Default to a 2:1 reward-to-risk ratio when explicit overrides are absent.
        take_profit_pct = self.stop_loss_pct * 2
        if signal.direction.value == "buy":
            return entry_price * (1 + take_profit_pct)
        if signal.direction.value == "sell":
            return entry_price * (1 - take_profit_pct)
        return entry_price


class VolatilityRiskManager(RiskManager):
    """
    ATR-based volatility risk manager

    Adjusts position sizing and stop losses based on market volatility
    using Average True Range (ATR) calculations.
    """

    def __init__(
        self,
        base_risk: float = DEFAULT_BASE_RISK_PER_TRADE,
        atr_multiplier: float = 2.0,
        min_risk: float = 0.005,
        max_risk: float = DEFAULT_STOP_LOSS_PCT,
    ):
        """
        Initialize volatility risk manager

        Args:
            base_risk: Base risk percentage per trade (0.02 = 2%)
            atr_multiplier: ATR multiplier for stop loss calculation
            min_risk: Minimum risk percentage (0.005 = 0.5%)
            max_risk: Maximum risk percentage (0.05 = 5%)
        """
        super().__init__("volatility_risk_manager")

        if not 0.001 <= base_risk <= 0.1:
            raise ValueError(f"base_risk must be between 0.001 and 0.1, got {base_risk}")

        if not 0.5 <= atr_multiplier <= 5.0:
            raise ValueError(f"atr_multiplier must be between 0.5 and 5.0, got {atr_multiplier}")

        if not 0.001 <= min_risk <= max_risk:
            raise ValueError(f"min_risk must be between 0.001 and max_risk, got {min_risk}")

        if not min_risk <= max_risk <= 0.2:
            raise ValueError(f"max_risk must be between min_risk and 0.2, got {max_risk}")

        self.base_risk = base_risk
        self.atr_multiplier = atr_multiplier
        self.min_risk = min_risk
        self.max_risk = max_risk

    def calculate_position_size(
        self,
        signal: "Signal",
        balance: float,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate position size based on volatility-adjusted risk"""
        self.validate_inputs(balance)

        if signal.direction.value == "hold":
            return 0.0

        # Get volatility from signal metadata or use default
        volatility = signal.metadata.get("atr", 0.02)  # Default 2% ATR

        # Adjust risk based on volatility (higher volatility = lower risk)
        volatility_multiplier = min(2.0, max(0.5, 0.02 / max(volatility, 0.005)))
        adjusted_risk = self.base_risk * volatility_multiplier

        # Apply bounds
        adjusted_risk = max(self.min_risk, min(self.max_risk, adjusted_risk))

        # Base risk amount
        risk_amount = balance * adjusted_risk

        # Adjust for signal confidence and strength
        confidence_multiplier = max(0.1, signal.confidence)
        strength_multiplier = max(0.1, signal.strength)

        position_size = risk_amount * confidence_multiplier * strength_multiplier

        # Apply regime-based adjustments if available
        if regime is not None:
            regime_multiplier = self._get_regime_multiplier(regime)
            position_size *= regime_multiplier

        # Ensure reasonable bounds
        min_position = balance * 0.001  # 0.1% minimum
        max_position = balance * 0.15  # 15% maximum

        return max(min_position, min(max_position, position_size))

    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> bool:
        """Determine exit based on volatility-adjusted stop loss"""
        # Use volatility from market data if available
        volatility = current_data.volatility or 0.02

        # Calculate dynamic stop loss percentage based on volatility
        stop_loss_pct = volatility * self.atr_multiplier
        stop_loss_pct = max(0.01, min(0.15, stop_loss_pct))  # 1% to 15% bounds

        # Calculate current loss percentage
        loss_pct = abs(position.get_pnl_percentage()) / 100

        # Exit if loss exceeds dynamic stop loss threshold
        if position.get_pnl_percentage() < 0 and loss_pct >= stop_loss_pct:
            return True

        return False

    def get_stop_loss(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate stop loss based on ATR"""
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        # Get ATR from signal metadata or use default
        atr = signal.metadata.get("atr", entry_price * 0.02)  # Default 2% of price

        # Calculate stop loss distance
        stop_distance = atr * self.atr_multiplier

        if signal.direction.value == "buy":
            # For long positions, stop loss is below entry price
            return entry_price - stop_distance
        elif signal.direction.value == "sell":
            # For short positions, stop loss is above entry price
            return entry_price + stop_distance
        else:
            # No stop loss for hold signals
            return entry_price

    def _get_regime_multiplier(self, regime: "RegimeContext") -> float:
        """Get position size multiplier based on regime"""
        multiplier = 1.0

        # Reduce size in high volatility regimes
        if hasattr(regime, "volatility") and regime.volatility.value == "high_vol":
            multiplier *= 0.6
        elif hasattr(regime, "volatility") and regime.volatility.value == "low_vol":
            multiplier *= 1.2

        # Adjust for trend
        if hasattr(regime, "trend") and regime.trend.value == "trend_down":
            multiplier *= 0.7

        # Adjust for regime confidence
        if hasattr(regime, "confidence") and regime.confidence < 0.5:
            multiplier *= 0.8

        return max(0.2, min(2.0, multiplier))  # 20% to 200% of base size

    def get_parameters(self) -> dict[str, Any]:
        """Get volatility risk manager parameters"""
        params = super().get_parameters()
        params.update(
            {
                "base_risk": self.base_risk,
                "atr_multiplier": self.atr_multiplier,
                "min_risk": self.min_risk,
                "max_risk": self.max_risk,
            }
        )
        return params

    def get_take_profit(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Mirror stop distance for take-profit targeting."""

        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        atr_multiplier = getattr(self, "atr_multiplier", self.atr_multiplier)
        atr = context.get("atr") if context else None
        distance = None
        if atr is not None and atr > 0:
            distance = atr * atr_multiplier
        elif context and "volatility" in context:
            try:
                distance = float(context["volatility"]) * entry_price
            except (ValueError, TypeError):
                distance = None

        if distance is None:
            distance = entry_price * self.base_risk

        if signal.direction.value == "buy":
            return entry_price + distance
        if signal.direction.value == "sell":
            return entry_price - distance
        return entry_price


class RegimeAdaptiveRiskManager(RiskManager):
    """
    Regime-adaptive risk manager

    Adjusts risk parameters based on detected market regimes,
    with different risk profiles for different market conditions.
    """

    def __init__(
        self,
        base_risk: float = DEFAULT_BASE_RISK_PER_TRADE,
        regime_multipliers: dict[str, float] | None = None,
    ):
        """
        Initialize regime-adaptive risk manager

        Args:
            base_risk: Base risk percentage per trade (0.02 = 2%)
            regime_multipliers: Custom multipliers for different regimes
        """
        super().__init__("regime_adaptive_risk_manager")

        if not 0.001 <= base_risk <= 0.1:
            raise ValueError(f"base_risk must be between 0.001 and 0.1, got {base_risk}")

        self.base_risk = base_risk

        # Default regime multipliers
        default_multipliers = {
            "bull_low_vol": 1.5,  # Aggressive in favorable conditions
            "bull_high_vol": 1.0,  # Normal in volatile bull market
            "bear_low_vol": 0.5,  # Conservative in bear market
            "bear_high_vol": 0.3,  # Very conservative in volatile bear
            "sideways_low_vol": 0.8,  # Reduced in sideways markets
            "sideways_high_vol": 0.4,  # Very reduced in volatile sideways
            "unknown": 0.6,  # Conservative when regime unclear
        }

        # Merge custom multipliers with defaults
        if regime_multipliers:
            self.regime_multipliers = {**default_multipliers, **regime_multipliers}
        else:
            self.regime_multipliers = default_multipliers
        # Validate multipliers
        for regime, multiplier in self.regime_multipliers.items():
            if not 0.1 <= multiplier <= 3.0:
                raise ValueError(
                    f"regime multiplier for {regime} must be between 0.1 and 3.0, got {multiplier}"
                )

    def calculate_position_size(
        self,
        signal: "Signal",
        balance: float,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate position size based on regime-specific risk parameters"""
        self.validate_inputs(balance)

        if signal.direction.value == "hold":
            return 0.0

        # Get regime-specific risk multiplier
        regime_multiplier = self._get_regime_risk_multiplier(regime)
        adjusted_risk = self.base_risk * regime_multiplier

        # Base risk amount
        risk_amount = balance * adjusted_risk

        # Adjust for signal confidence and strength
        confidence_multiplier = max(0.1, signal.confidence)
        strength_multiplier = max(0.1, signal.strength)

        position_size = risk_amount * confidence_multiplier * strength_multiplier

        # Apply regime confidence scaling
        if regime is not None and hasattr(regime, "confidence"):
            confidence_scaling = max(0.5, regime.confidence)  # Min 50% scaling
            position_size *= confidence_scaling

        # Ensure reasonable bounds
        min_position = balance * 0.001  # 0.1% minimum
        max_position = balance * 0.2  # 20% maximum

        return max(min_position, min(max_position, position_size))

    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> bool:
        """Determine exit based on regime-specific criteria"""
        # Get regime-specific stop loss percentage
        stop_loss_pct = self._get_regime_stop_loss(regime)

        # Calculate current loss percentage
        loss_pct = abs(position.get_pnl_percentage()) / 100

        # Exit if loss exceeds regime-specific stop loss threshold
        if position.get_pnl_percentage() < 0 and loss_pct >= stop_loss_pct:
            return True

        # Check for regime transition exit conditions
        if regime is not None and self._should_exit_on_regime_change(regime):
            return True

        return False

    def get_stop_loss(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate stop loss based on regime-specific parameters"""
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        # Get regime-specific stop loss percentage
        stop_loss_pct = self._get_regime_stop_loss(regime)

        if signal.direction.value == "buy":
            # For long positions, stop loss is below entry price
            return entry_price * (1 - stop_loss_pct)
        elif signal.direction.value == "sell":
            # For short positions, stop loss is above entry price
            return entry_price * (1 + stop_loss_pct)
        else:
            # No stop loss for hold signals
            return entry_price

    def get_take_profit(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate a regime-adjusted take-profit level."""

        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        stop_loss_pct = self._get_regime_stop_loss(regime)
        reward_to_risk = context.get("reward_to_risk", 2.0)
        try:
            reward_to_risk = max(1.0, float(reward_to_risk))
        except (TypeError, ValueError):
            reward_to_risk = 2.0
        take_profit_pct = stop_loss_pct * reward_to_risk

        if signal.direction.value == "buy":
            return entry_price * (1 + take_profit_pct)
        if signal.direction.value == "sell":
            return entry_price * (1 - take_profit_pct)
        return entry_price

    def _get_regime_risk_multiplier(self, regime: Optional["RegimeContext"]) -> float:
        """Get risk multiplier based on current regime"""
        if regime is None:
            return self.regime_multipliers["unknown"]

        # Determine regime key based on trend and volatility
        trend_key = "unknown"
        if hasattr(regime, "trend"):
            if regime.trend.value == "trend_up":
                trend_key = "bull"
            elif regime.trend.value == "trend_down":
                trend_key = "bear"
            else:
                trend_key = "sideways"

        vol_key = "high_vol"
        if hasattr(regime, "volatility"):
            vol_key = "low_vol" if regime.volatility.value == "low_vol" else "high_vol"

        regime_key = f"{trend_key}_{vol_key}"

        return self.regime_multipliers.get(regime_key, self.regime_multipliers["unknown"])

    def _get_regime_stop_loss(self, regime: Optional["RegimeContext"]) -> float:
        """Get stop loss percentage based on regime"""
        if regime is None:
            return 0.05  # 5% default

        # More aggressive stops in favorable regimes, wider in unfavorable
        if hasattr(regime, "trend") and hasattr(regime, "volatility"):
            if regime.trend.value == "trend_up" and regime.volatility.value == "low_vol":
                return 0.03  # 3% tight stop in bull low vol
            elif regime.trend.value == "trend_down":
                return 0.08  # 8% wider stop in bear market
            elif regime.volatility.value == "high_vol":
                return 0.07  # 7% wider stop in high volatility

        return 0.05  # 5% default

    def _should_exit_on_regime_change(self, regime: "RegimeContext") -> bool:
        """Check if position should be exited due to regime change"""
        # Exit if regime confidence is very low (regime transition)
        if hasattr(regime, "confidence") and regime.confidence < 0.3:
            return True

        # Exit if regime duration is very short (unstable regime)
        if hasattr(regime, "duration") and regime.duration < 3:
            return True

        return False

    def get_parameters(self) -> dict[str, Any]:
        """Get regime-adaptive risk manager parameters"""
        params = super().get_parameters()
        params.update({"base_risk": self.base_risk, "regime_multipliers": self.regime_multipliers})
        return params
