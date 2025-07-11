from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np

@dataclass
class RiskParameters:
    """Risk management parameters"""
    base_risk_per_trade: float = 0.02  # 2% risk per trade
    max_risk_per_trade: float = 0.03   # 3% maximum risk per trade
    max_position_size: float = 0.25    # 25% maximum position size
    max_daily_risk: float = 0.06       # 6% maximum daily risk
    max_correlated_risk: float = 0.10  # 10% maximum risk for correlated positions
    max_drawdown: float = 0.20         # 20% maximum drawdown
    position_size_atr_multiplier: float = 1.0
    
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

class RiskManager:
    """Handles position sizing and risk management"""
    
    def __init__(self, parameters: Optional[RiskParameters] = None, max_concurrent_positions: int = 3):
        self.params = parameters or RiskParameters()
        self.daily_risk_used = 0.0
        self.positions: Dict[str, dict] = {}
        self.max_concurrent_positions = max_concurrent_positions
        
    def reset_daily_risk(self):
        """Reset daily risk counter"""
        self.daily_risk_used = 0.0
        
    def calculate_position_size(
        self,
        price: float,
        atr: float,
        balance: float,
        regime: str = 'normal'
    ) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            price: Current asset price
            atr: Average True Range
            balance: Account balance
            regime: Market regime ('normal', 'trending', 'volatile')
            
        Returns:
            Position size in base currency
        """
        # Validate inputs
        if price <= 0 or balance <= 0:
            return 0.0
            
        # Handle zero or very small ATR
        if atr <= 0 or atr < price * 0.001:  # Less than 0.1% of price
            atr = price * 0.01  # Use 1% of price as minimum ATR
            
        # Adjust risk based on market regime
        base_risk = self.params.base_risk_per_trade
        if regime == 'trending':
            risk = base_risk * 1.5  # Increase risk in trending markets (more aggressive)
        elif regime == 'volatile':
            risk = base_risk * 0.6  # Reduce risk in volatile markets (more conservative)
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
        
        # Ensure position size doesn't exceed maximum
        max_position_value = balance * self.params.max_position_size
        position_size = min(position_size, max_position_value / price)
        
        # Ensure position size is positive
        return max(0.0, position_size)
        
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        side: str = 'long'
    ) -> float:
        """Calculate adaptive stop loss level"""
        atr_multiple = self.params.position_size_atr_multiplier
        stop_distance = atr * atr_multiple
        
        if side == 'long':
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
            
    def check_drawdown(self, current_balance: float, peak_balance: float) -> bool:
        """Check if maximum drawdown has been exceeded"""
        if peak_balance == 0:
            return False
            
        drawdown = (peak_balance - current_balance) / peak_balance
        return drawdown > self.params.max_drawdown
        
    def update_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float
    ):
        """Update position tracking"""
        self.positions[symbol] = {
            'side': side,
            'size': size,
            'entry_price': entry_price
        }
        
        # Update daily risk used
        position_value = size * entry_price
        self.daily_risk_used += position_value * self.params.base_risk_per_trade
        
    def close_position(self, symbol: str):
        """Close position tracking"""
        if symbol in self.positions:
            del self.positions[symbol]
            
    def get_total_exposure(self) -> float:
        """Calculate total position exposure"""
        return sum(pos['size'] * pos['entry_price'] for pos in self.positions.values())
        
    def get_position_correlation_risk(self, symbols: list) -> float:
        """Calculate risk from correlated positions"""
        # This is a simplified correlation check
        # In practice, you would want to use actual price correlation calculations
        exposure = sum(
            pos['size'] * pos['entry_price']
            for symbol, pos in self.positions.items()
            if symbol in symbols
        )
        return round(exposure, 2)  # Round to avoid floating point precision issues 

    def get_max_concurrent_positions(self) -> int:
        """Return the maximum number of concurrent positions allowed."""
        return self.max_concurrent_positions 