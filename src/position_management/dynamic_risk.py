"""
Dynamic Risk Management System

This module implements dynamic risk throttling that automatically adjusts position sizing,
stop-loss levels, and risk limits based on recent performance, drawdown levels, and market conditions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import logging

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    
from src.risk.risk_manager import RiskParameters

logger = logging.getLogger(__name__)


@dataclass
class DynamicRiskConfig:
    """Configuration for dynamic risk management"""
    
    # Core settings
    enabled: bool = True
    performance_window_days: int = 30
    
    # Drawdown thresholds and adjustments
    drawdown_thresholds: List[float] = None  # [0.05, 0.10, 0.15] = [5%, 10%, 15%]
    risk_reduction_factors: List[float] = None  # [0.8, 0.6, 0.4] = reduction at each threshold
    
    # Recovery thresholds
    recovery_thresholds: List[float] = None  # [0.02, 0.05] = [2%, 5%] positive returns
    
    # Volatility adjustments
    volatility_adjustment_enabled: bool = True
    volatility_window_days: int = 30
    high_volatility_threshold: float = 0.03  # 3% daily volatility threshold
    low_volatility_threshold: float = 0.01   # 1% daily volatility threshold
    volatility_risk_multipliers: Tuple[float, float] = (0.7, 1.3)  # (high_vol, low_vol)
    
    def __post_init__(self):
        """Set defaults and validate configuration"""
        if self.drawdown_thresholds is None:
            self.drawdown_thresholds = [0.05, 0.10, 0.15]
        if self.risk_reduction_factors is None:
            self.risk_reduction_factors = [0.8, 0.6, 0.4]
        if self.recovery_thresholds is None:
            self.recovery_thresholds = [0.02, 0.05]
            
        # Validation
        if len(self.drawdown_thresholds) != len(self.risk_reduction_factors):
            raise ValueError("drawdown_thresholds and risk_reduction_factors must have same length")
        
        if not all(0 < factor <= 1 for factor in self.risk_reduction_factors):
            raise ValueError("risk_reduction_factors must be between 0 and 1")
            
        if not all(threshold > 0 for threshold in self.drawdown_thresholds):
            raise ValueError("drawdown_thresholds must be positive")


@dataclass
class RiskAdjustments:
    """Container for calculated risk adjustments"""
    
    position_size_factor: float = 1.0  # Multiplier for position sizing
    stop_loss_tightening: float = 1.0  # Multiplier for stop-loss distances
    daily_risk_factor: float = 1.0     # Multiplier for daily risk limits
    
    # Metadata
    primary_reason: str = "normal"
    adjustment_details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.adjustment_details is None:
            self.adjustment_details = {}


class DynamicRiskManager:
    """
    Manages dynamic risk adjustments based on performance and market conditions.
    
    This class works alongside the existing RiskManager to provide adaptive
    risk management that responds to changing performance and market conditions.
    """
    
    def __init__(
        self,
        config: Optional[DynamicRiskConfig] = None,
        db_manager: Optional["DatabaseManager"] = None
    ):
        self.config = config or DynamicRiskConfig()
        self.db_manager = db_manager
        
        # Cache for performance calculations
        self._performance_cache: Dict[str, Any] = {}
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minutes
        
    def calculate_dynamic_risk_adjustments(
        self,
        current_balance: float,
        peak_balance: float,
        session_id: Optional[int] = None
    ) -> RiskAdjustments:
        """
        Calculate dynamic risk adjustments based on current performance metrics.
        
        Args:
            current_balance: Current account balance
            peak_balance: Peak account balance (for drawdown calculation)
            session_id: Trading session ID for database queries
            
        Returns:
            RiskAdjustments object with calculated adjustment factors
        """
        if not self.config.enabled:
            return RiskAdjustments(primary_reason="disabled")
            
        # Calculate current drawdown
        current_drawdown = self._calculate_current_drawdown(current_balance, peak_balance)
        
        # Get performance metrics
        performance_metrics = self._get_performance_metrics(session_id)
        
        # Calculate adjustments based on different factors
        drawdown_adjustment = self._calculate_drawdown_adjustment(current_drawdown)
        performance_adjustment = self._calculate_performance_adjustment(performance_metrics)
        volatility_adjustment = self._calculate_volatility_adjustment(performance_metrics)
        
        # Combine adjustments (take the most conservative)
        final_position_factor = min(
            drawdown_adjustment.position_size_factor,
            performance_adjustment.position_size_factor,
            volatility_adjustment.position_size_factor
        )
        
        final_stop_loss_factor = max(
            drawdown_adjustment.stop_loss_tightening,
            performance_adjustment.stop_loss_tightening,
            volatility_adjustment.stop_loss_tightening
        )
        
        final_daily_risk_factor = min(
            drawdown_adjustment.daily_risk_factor,
            performance_adjustment.daily_risk_factor,
            volatility_adjustment.daily_risk_factor
        )
        
        # Determine primary reason
        primary_reason = self._determine_primary_reason(
            drawdown_adjustment, performance_adjustment, volatility_adjustment
        )
        
        return RiskAdjustments(
            position_size_factor=final_position_factor,
            stop_loss_tightening=final_stop_loss_factor,
            daily_risk_factor=final_daily_risk_factor,
            primary_reason=primary_reason,
            adjustment_details={
                "current_drawdown": current_drawdown,
                "drawdown_adjustment": drawdown_adjustment,
                "performance_adjustment": performance_adjustment,
                "volatility_adjustment": volatility_adjustment,
                "performance_metrics": performance_metrics
            }
        )
    
    def apply_risk_adjustments(
        self,
        risk_parameters: RiskParameters,
        adjustments: RiskAdjustments
    ) -> RiskParameters:
        """
        Apply dynamic adjustments to risk parameters.
        
        Args:
            risk_parameters: Original risk parameters
            adjustments: Calculated adjustments to apply
            
        Returns:
            New RiskParameters with adjustments applied
        """
        # Create a copy to avoid modifying the original
        adjusted_params = RiskParameters(
            base_risk_per_trade=risk_parameters.base_risk_per_trade * adjustments.position_size_factor,
            max_risk_per_trade=risk_parameters.max_risk_per_trade * adjustments.position_size_factor,
            max_position_size=risk_parameters.max_position_size * adjustments.position_size_factor,
            max_daily_risk=risk_parameters.max_daily_risk * adjustments.daily_risk_factor,
            max_correlated_risk=risk_parameters.max_correlated_risk * adjustments.position_size_factor,
            max_drawdown=risk_parameters.max_drawdown,  # Don't adjust max drawdown threshold
            position_size_atr_multiplier=risk_parameters.position_size_atr_multiplier * adjustments.stop_loss_tightening,
            default_take_profit_pct=risk_parameters.default_take_profit_pct,
            atr_period=risk_parameters.atr_period
        )
        
        return adjusted_params
    
    def _calculate_current_drawdown(self, current_balance: float, peak_balance: float) -> float:
        """Calculate current drawdown percentage"""
        if peak_balance <= 0:
            return 0.0
        return max(0.0, (peak_balance - current_balance) / peak_balance)
    
    def _get_performance_metrics(self, session_id: Optional[int]) -> Dict[str, Any]:
        """Get cached performance metrics or calculate new ones"""
        now = datetime.utcnow()
        
        # Check cache validity
        if (self._cache_timestamp and 
            (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds and
            self._performance_cache):
            return self._performance_cache
        
        # Calculate new metrics
        metrics = {}
        
        if self.db_manager and session_id:
            try:
                # Get recent performance data
                start_date = now - timedelta(days=self.config.performance_window_days)
                db_metrics = self.db_manager.get_performance_metrics(
                    start_date=start_date,
                    session_id=session_id
                )
                
                metrics.update(db_metrics)
                
                # Calculate additional metrics if we have trade data
                # This would typically include volatility, Sharpe ratio, etc.
                # For now, we'll use the basic metrics from the database
                
            except Exception as e:
                logger.warning(f"Failed to get performance metrics from database: {e}")
        
        # Cache the results
        self._performance_cache = metrics
        self._cache_timestamp = now
        
        return metrics
    
    def _calculate_drawdown_adjustment(self, current_drawdown: float) -> RiskAdjustments:
        """Calculate adjustments based on current drawdown level"""
        if current_drawdown <= 0:
            return RiskAdjustments(primary_reason="normal")
        
        # Find the appropriate threshold
        position_factor = 1.0
        stop_loss_factor = 1.0
        daily_risk_factor = 1.0
        
        for i, threshold in enumerate(self.config.drawdown_thresholds):
            if current_drawdown >= threshold:
                position_factor = self.config.risk_reduction_factors[i]
                daily_risk_factor = self.config.risk_reduction_factors[i]
                stop_loss_factor = 1.0 + (0.2 * i)  # Tighten stops progressively
        
        reason = f"drawdown_{current_drawdown:.1%}"
        
        return RiskAdjustments(
            position_size_factor=position_factor,
            stop_loss_tightening=stop_loss_factor,
            daily_risk_factor=daily_risk_factor,
            primary_reason=reason
        )
    
    def _calculate_performance_adjustment(self, performance_metrics: Dict[str, Any]) -> RiskAdjustments:
        """Calculate adjustments based on recent performance"""
        win_rate = performance_metrics.get("win_rate", 0.5)
        profit_factor = performance_metrics.get("profit_factor", 1.0)
        total_trades = performance_metrics.get("total_trades", 0)
        
        # Need minimum trades for reliable adjustment
        if total_trades < 10:
            return RiskAdjustments(primary_reason="insufficient_data")
        
        # Calculate performance score
        performance_score = (win_rate * 0.6) + (min(profit_factor / 2.0, 1.0) * 0.4)
        
        if performance_score < 0.3:  # Poor performance
            return RiskAdjustments(
                position_size_factor=0.6,
                stop_loss_tightening=1.2,
                daily_risk_factor=0.7,
                primary_reason="poor_performance"
            )
        elif performance_score > 0.7:  # Good performance
            return RiskAdjustments(
                position_size_factor=1.2,
                stop_loss_tightening=0.9,
                daily_risk_factor=1.1,
                primary_reason="good_performance"
            )
        else:
            return RiskAdjustments(primary_reason="normal_performance")
    
    def _calculate_volatility_adjustment(self, performance_metrics: Dict[str, Any]) -> RiskAdjustments:
        """Calculate adjustments based on market volatility"""
        if not self.config.volatility_adjustment_enabled:
            return RiskAdjustments(primary_reason="volatility_disabled")
        
        # This is a simplified volatility calculation
        # In a real implementation, you'd calculate this from price data
        # For now, we'll use a placeholder
        estimated_volatility = 0.02  # 2% default
        
        if estimated_volatility > self.config.high_volatility_threshold:
            return RiskAdjustments(
                position_size_factor=self.config.volatility_risk_multipliers[0],
                stop_loss_tightening=1.1,
                daily_risk_factor=self.config.volatility_risk_multipliers[0],
                primary_reason="high_volatility"
            )
        elif estimated_volatility < self.config.low_volatility_threshold:
            return RiskAdjustments(
                position_size_factor=self.config.volatility_risk_multipliers[1],
                stop_loss_tightening=0.9,
                daily_risk_factor=self.config.volatility_risk_multipliers[1],
                primary_reason="low_volatility"
            )
        else:
            return RiskAdjustments(primary_reason="normal_volatility")
    
    def _determine_primary_reason(
        self,
        drawdown_adj: RiskAdjustments,
        performance_adj: RiskAdjustments,
        volatility_adj: RiskAdjustments
    ) -> str:
        """Determine the primary reason for risk adjustment"""
        # Priority: drawdown > performance > volatility
        if drawdown_adj.position_size_factor < 1.0:
            return drawdown_adj.primary_reason
        elif performance_adj.position_size_factor != 1.0:
            return performance_adj.primary_reason
        elif volatility_adj.position_size_factor != 1.0:
            return volatility_adj.primary_reason
        else:
            return "normal"