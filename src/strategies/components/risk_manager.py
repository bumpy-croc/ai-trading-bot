"""
Risk Manager Components

This module defines the abstract RiskManager interface and related data models
for managing position sizing, stop losses, and risk controls in the component-based
strategy architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


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
        
        if self.side not in ['long', 'short']:
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
        if self.side == 'long':
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
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: Optional[datetime] = None
    volatility: Optional[float] = None
    
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
            raise ValueError(f"volatility must be non-negative when provided, got {self.volatility}")
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread if both bid and ask are available"""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    def get_spread_percentage(self) -> Optional[float]:
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
    def calculate_position_size(self, signal: 'Signal', balance: float, 
                              regime: Optional['RegimeContext'] = None) -> float:
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
    def should_exit(self, position: Position, current_data: MarketData, 
                   regime: Optional['RegimeContext'] = None) -> bool:
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
    def get_stop_loss(self, entry_price: float, signal: 'Signal', 
                     regime: Optional['RegimeContext'] = None) -> float:
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
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get risk manager parameters for logging and serialization
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__
        }


class FixedRiskManager(RiskManager):
    """
    Simple fixed-risk manager for testing
    
    Uses fixed percentage risk per trade and simple stop loss rules
    """
    
    def __init__(self, risk_per_trade: float = 0.02, stop_loss_pct: float = 0.05):
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
    
    def calculate_position_size(self, signal: 'Signal', balance: float, 
                              regime: Optional['RegimeContext'] = None) -> float:
        """Calculate position size based on fixed risk percentage"""
        self.validate_inputs(balance)
        
        if signal.direction.value == 'hold':
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
        max_position = balance * 0.1    # 10% maximum
        
        return max(min_position, min(max_position, position_size))
    
    def should_exit(self, position: Position, current_data: MarketData, 
                   regime: Optional['RegimeContext'] = None) -> bool:
        """Determine exit based on stop loss percentage"""
        # Calculate current loss percentage
        loss_pct = abs(position.get_pnl_percentage()) / 100
        
        # Exit if loss exceeds stop loss threshold
        if position.get_pnl_percentage() < 0 and loss_pct >= self.stop_loss_pct:
            return True
        
        return False
    
    def get_stop_loss(self, entry_price: float, signal: 'Signal', 
                     regime: Optional['RegimeContext'] = None) -> float:
        """Calculate stop loss based on fixed percentage"""
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        
        if signal.direction.value == 'buy':
            # For long positions, stop loss is below entry price
            return entry_price * (1 - self.stop_loss_pct)
        elif signal.direction.value == 'sell':
            # For short positions, stop loss is above entry price
            return entry_price * (1 + self.stop_loss_pct)
        else:
            # No stop loss for hold signals
            return entry_price
    
    def _get_regime_multiplier(self, regime: 'RegimeContext') -> float:
        """Get position size multiplier based on regime"""
        multiplier = 1.0
        
        # Reduce size in high volatility
        if hasattr(regime, 'volatility') and regime.volatility.value == 'high_vol':
            multiplier *= 0.7
        
        # Reduce size in bear markets
        if hasattr(regime, 'trend') and regime.trend.value == 'trend_down':
            multiplier *= 0.8
        
        # Reduce size when regime confidence is low
        if hasattr(regime, 'confidence') and regime.confidence < 0.5:
            multiplier *= 0.9
        
        return max(0.2, multiplier)  # Minimum 20% of base size
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get fixed risk manager parameters"""
        params = super().get_parameters()
        params.update({
            'risk_per_trade': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct
        })
        return params