"""
Strategy Signal Generator

Standardizes signal generation across all strategies and execution environments.
Provides consistent interface for entry/exit signals and confidence scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from performance.metrics import Side
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Standardized trading signal"""
    timestamp: datetime
    symbol: str
    action: str  # 'enter', 'exit', 'hold'
    side: Optional[Side] = None  # Required for 'enter'
    confidence: float = 0.5  # 0.0 to 1.0
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Optional[Dict] = None  # Strategy-specific data
    reasons: Optional[List[str]] = None  # Human-readable reasons


@dataclass
class MarketContext:
    """Market context for signal generation"""
    symbol: str
    current_price: float
    timestamp: datetime
    timeframe: str
    data: pd.DataFrame  # Historical data
    index: int  # Current position in data


class SignalGenerator:
    """
    Signal generation system.
    
    Standardizes the interface between strategies and execution engines,
    ensuring consistent signal format and metadata across all environments.
    """
    
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.last_signal: Optional[Signal] = None
        self.signal_history: List[Signal] = []
        
    def generate_signal(
        self, 
        context: MarketContext, 
        current_balance: float,
        active_positions: Optional[Dict] = None
    ) -> Signal:
        """
        Generate trading signal based on market context.
        
        Args:
            context: Current market context
            current_balance: Available trading balance
            active_positions: Currently active positions (for exit signals)
            
        Returns:
            Signal with action, confidence, and metadata
        """
        try:
            # Prepare data with indicators
            df = self.strategy.prepare_data(context.data)
            
            # Check for exit signals first (if we have positions)
            if active_positions:
                exit_signal = self._check_exit_signals(df, context, active_positions)
                if exit_signal:
                    self._log_signal(exit_signal)
                    return exit_signal
            
            # Check for entry signals
            entry_signal = self._check_entry_signals(df, context, current_balance)
            self._log_signal(entry_signal)
            return entry_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._create_hold_signal(context, error=str(e))
    
    def _check_entry_signals(
        self, 
        df: pd.DataFrame, 
        context: MarketContext, 
        current_balance: float
    ) -> Signal:
        """Check for entry signal conditions"""
        
        # Check if strategy conditions are met
        should_enter = self.strategy.check_entry_conditions(df, context.index)
        
        if not should_enter:
            return self._create_hold_signal(context, reason="Entry conditions not met")
        
        # Calculate position size
        position_size = self.strategy.calculate_position_size(
            df, context.index, current_balance
        )
        
        if position_size <= 0:
            return self._create_hold_signal(context, reason="Invalid position size")
        
        # Calculate stop loss
        stop_loss = self.strategy.calculate_stop_loss(
            df, context.index, context.current_price, side='long'
        )
        
        # Calculate confidence (if strategy supports it)
        confidence = self._calculate_confidence(df, context)
        
        # Get strategy-specific metadata
        metadata = self._extract_metadata(df, context)
        
        # Generate reasons
        reasons = self._generate_entry_reasons(df, context)
        
        return Signal(
            timestamp=context.timestamp,
            symbol=context.symbol,
            action='enter',
            side=Side.LONG,  # Currently only support long positions
            confidence=confidence,
            price=context.current_price,
            stop_loss=stop_loss,
            position_size=min(position_size / current_balance, 1.0) if current_balance > 0 else 0.0,  # Convert to fraction
            metadata=metadata,
            reasons=reasons
        )
    
    def _check_exit_signals(
        self, 
        df: pd.DataFrame, 
        context: MarketContext, 
        active_positions: Dict
    ) -> Optional[Signal]:
        """Check for exit signal conditions"""
        
        for position_id, position in active_positions.items():
            if position.symbol != context.symbol:
                continue
                
            # Check strategy exit conditions
            should_exit = self.strategy.check_exit_conditions(
                df, context.index, position.entry_price
            )
            
            if should_exit:
                # Calculate confidence for exit
                confidence = self._calculate_exit_confidence(df, context, position)
                
                # Get exit metadata
                metadata = self._extract_exit_metadata(df, context, position)
                
                # Generate exit reasons
                reasons = self._generate_exit_reasons(df, context, position)
                
                return Signal(
                    timestamp=context.timestamp,
                    symbol=context.symbol,
                    action='exit',
                    confidence=confidence,
                    price=context.current_price,
                    metadata={
                        **metadata,
                        'position_id': position_id,
                        'entry_price': position.entry_price,
                        'entry_time': position.entry_time
                    },
                    reasons=reasons
                )
        
        return None
    
    def _create_hold_signal(
        self, 
        context: MarketContext, 
        reason: str = "No action", 
        error: Optional[str] = None
    ) -> Signal:
        """Create a hold signal"""
        metadata = {'reason': reason}
        if error:
            metadata['error'] = error
            
        return Signal(
            timestamp=context.timestamp,
            symbol=context.symbol,
            action='hold',
            confidence=0.0,
            price=context.current_price,
            metadata=metadata,
            reasons=[reason]
        )
    
    def _calculate_confidence(self, df: pd.DataFrame, context: MarketContext) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence_factors = []
        
        # Volume confirmation
        if 'volume' in df.columns and context.index > 0:
            current_volume = df['volume'].iloc[context.index]
            avg_volume = df['volume'].rolling(20).mean().iloc[context.index]
            if pd.notna(avg_volume) and avg_volume > 0:
                volume_factor = min(current_volume / avg_volume, 2.0) / 2.0
                confidence_factors.append(volume_factor)
        
        # Trend strength
        if 'trend_strength' in df.columns:
            trend_strength = df['trend_strength'].iloc[context.index]
            if pd.notna(trend_strength):
                trend_factor = min(abs(trend_strength), 0.1) / 0.1
                confidence_factors.append(trend_factor)
        
        # RSI confirmation
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[context.index]
            if pd.notna(rsi):
                # Higher confidence when RSI is in optimal range
                if 30 <= rsi <= 70:
                    rsi_factor = 1.0 - abs(rsi - 50) / 20  # Peak at RSI 50
                else:
                    rsi_factor = 0.3  # Lower confidence at extremes
                confidence_factors.append(rsi_factor)
        
        # ML model confidence (if available)
        if 'prediction_confidence' in df.columns:
            ml_confidence = df['prediction_confidence'].iloc[context.index]
            if pd.notna(ml_confidence):
                confidence_factors.append(ml_confidence)
        
        # Return average confidence, default to 0.5 if no factors
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_exit_confidence(self, df: pd.DataFrame, context: MarketContext, position) -> float:
        """Calculate exit signal confidence"""
        # Simple exit confidence based on profit/loss
        current_pnl = (context.current_price - position.entry_price) / position.entry_price
        
        if current_pnl > 0.02:  # Profitable position
            return 0.8
        elif current_pnl < -0.01:  # Losing position
            return 0.9  # High confidence to cut losses
        else:
            return 0.6  # Neutral
    
    def _extract_metadata(self, df: pd.DataFrame, context: MarketContext) -> Dict:
        """Extract strategy-specific metadata"""
        metadata = {
            'strategy': self.strategy.name,
            'timeframe': context.timeframe,
            'data_points': len(df)
        }
        
        # Add indicator values
        indicator_columns = ['rsi', 'atr', 'trend_strength', 'volume_trend']
        for col in indicator_columns:
            if col in df.columns:
                value = df[col].iloc[context.index]
                if pd.notna(value):
                    metadata[col] = float(value)
        
        # Add strategy parameters
        try:
            strategy_params = self.strategy.get_parameters()
            metadata['strategy_config'] = strategy_params
        except Exception:
            pass
        
        return metadata
    
    def _extract_exit_metadata(self, df: pd.DataFrame, context: MarketContext, position) -> Dict:
        """Extract exit-specific metadata"""
        metadata = self._extract_metadata(df, context)
        
        # Add position-specific data
        current_pnl = (context.current_price - position.entry_price) / position.entry_price
        metadata.update({
            'position_duration': (context.timestamp - position.entry_time).total_seconds() / 3600,  # hours
            'unrealized_pnl_pct': current_pnl * 100,
            'hit_stop_loss': context.current_price <= (position.stop_loss or 0),
            'hit_take_profit': context.current_price >= (position.take_profit or float('inf'))
        })
        
        return metadata
    
    def _generate_entry_reasons(self, df: pd.DataFrame, context: MarketContext) -> List[str]:
        """Generate human-readable entry reasons"""
        reasons = []
        
        # Check various conditions and add explanations
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[context.index]
            if pd.notna(rsi):
                if rsi < 30:
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    reasons.append(f"RSI overbought ({rsi:.1f})")
        
        if 'trend_strength' in df.columns:
            trend = df['trend_strength'].iloc[context.index]
            if pd.notna(trend) and trend > 0.01:
                reasons.append(f"Strong uptrend ({trend:.3f})")
        
        if 'volume_trend' in df.columns:
            vol_trend = df['volume_trend'].iloc[context.index]
            if pd.notna(vol_trend) and vol_trend > 0.1:
                reasons.append("High volume confirmation")
        
        return reasons if reasons else ["Strategy conditions met"]
    
    def _generate_exit_reasons(self, df: pd.DataFrame, context: MarketContext, position) -> List[str]:
        """Generate human-readable exit reasons"""
        reasons = []
        
        # Check stop loss / take profit
        if position.stop_loss and context.current_price <= position.stop_loss:
            reasons.append("Stop loss triggered")
        
        if position.take_profit and context.current_price >= position.take_profit:
            reasons.append("Take profit triggered")
        
        # Check technical reasons
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[context.index]
            if pd.notna(rsi) and rsi > 80:
                reasons.append(f"RSI extremely overbought ({rsi:.1f})")
        
        # Check trend reversal
        if 'trend_strength' in df.columns:
            trend = df['trend_strength'].iloc[context.index]
            if pd.notna(trend) and trend < -0.01:
                reasons.append("Trend reversal detected")
        
        return reasons if reasons else ["Strategy exit conditions met"]
    
    def _log_signal(self, signal: Signal) -> None:
        """Log signal for tracking and debugging"""
        self.last_signal = signal
        self.signal_history.append(signal)
        
        # Keep only last 100 signals
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
        
        # Log significant signals
        if signal.action != 'hold':
            logger.info(
                f"🎯 Signal: {signal.action.upper()} {signal.symbol} "
                f"(confidence: {signal.confidence:.2f}) - {', '.join(signal.reasons or [])}"
            )
    
    def get_signal_history(self, limit: int = 10) -> List[Signal]:
        """Get recent signal history"""
        return self.signal_history[-limit:] 