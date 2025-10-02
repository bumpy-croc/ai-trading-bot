"""
Composable Strategy Class

This module defines the Strategy class that composes SignalGenerator, RiskManager,
and PositionSizer components to create a unified trading strategy with comprehensive
logging and decision tracking.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .signal_generator import Signal, SignalGenerator, SignalDirection
from .risk_manager import RiskManager, Position, MarketData
from .position_sizer import PositionSizer
from .regime_context import RegimeContext, EnhancedRegimeDetector


@dataclass
class TradingDecision:
    """
    Complete trading decision with all component outputs
    
    Attributes:
        timestamp: When the decision was made
        signal: Generated trading signal
        position_size: Calculated position size
        regime: Market regime context
        risk_metrics: Risk-related metrics
        execution_time_ms: Time taken for decision
        metadata: Additional decision metadata
    """
    timestamp: datetime
    signal: Signal
    position_size: float
    regime: Optional[RegimeContext]
    risk_metrics: Dict[str, float]
    execution_time_ms: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'signal': {
                'direction': self.signal.direction.value,
                'strength': self.signal.strength,
                'confidence': self.signal.confidence,
                'metadata': self.signal.metadata
            },
            'position_size': self.position_size,
            'regime': {
                'trend': self.regime.trend.value if self.regime else None,
                'volatility': self.regime.volatility.value if self.regime else None,
                'confidence': self.regime.confidence if self.regime else None,
                'duration': self.regime.duration if self.regime else None,
                'strength': self.regime.strength if self.regime else None
            } if self.regime else None,
            'risk_metrics': self.risk_metrics,
            'execution_time_ms': self.execution_time_ms,
            'metadata': self.metadata
        }


class Strategy:
    """
    Composable strategy class that orchestrates components
    
    This class composes SignalGenerator, RiskManager, and PositionSizer components
    to create a unified trading strategy with comprehensive logging and decision tracking.
    """
    
    def __init__(self, name: str, signal_generator: SignalGenerator,
                 risk_manager: RiskManager, position_sizer: PositionSizer,
                 regime_detector: Optional[EnhancedRegimeDetector] = None,
                 enable_logging: bool = True, max_history: int = 1000):
        """
        Initialize composable strategy
        
        Args:
            name: Strategy name for identification
            signal_generator: Component for generating trading signals
            risk_manager: Component for risk management and position sizing
            position_sizer: Component for final position size calculation
            regime_detector: Optional regime detection component
            enable_logging: Whether to enable detailed logging
            max_history: Maximum number of decisions to keep in history
        """
        self.name = name
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.regime_detector = regime_detector or EnhancedRegimeDetector()
        
        # Logging setup
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(f"Strategy.{name}")
        if enable_logging:
            self.logger.setLevel(logging.INFO)
        
        # Decision history
        self.decision_history: List[TradingDecision] = []
        self.max_history = max_history
        
        # Performance metrics
        self.metrics = {
            'total_decisions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_execution_time_ms': 0.0,
            'avg_signal_confidence': 0.0,
            'avg_position_size': 0.0,
            'last_updated': datetime.now()
        }
        
        self.logger.info(f"Strategy '{name}' initialized with components: "
                        f"SignalGen={signal_generator.name}, "
                        f"RiskMgr={risk_manager.name}, "
                        f"PosSizer={position_sizer.name}")
    
    def process_candle(self, df: pd.DataFrame, index: int, balance: float,
                      current_positions: Optional[List[Position]] = None) -> TradingDecision:
        """
        Process a single candle and make trading decision
        
        This is the main method that coordinates all components to make a trading decision.
        
        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame
            balance: Available account balance
            current_positions: List of current positions (optional)
            
        Returns:
            TradingDecision containing all decision information
            
        Raises:
            ValueError: If input parameters are invalid
            IndexError: If index is out of bounds
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Validate inputs
            self._validate_inputs(df, index, balance)
            
            # Step 1: Detect market regime
            regime = self._detect_regime(df, index)
            
            # Step 2: Generate trading signal
            signal = self._generate_signal(df, index, regime)
            
            # Step 3: Calculate risk-based position size
            risk_position_size = self._calculate_risk_position_size(signal, balance, regime)
            
            # Step 4: Apply position sizer adjustments
            final_position_size = self._calculate_final_position_size(
                signal, balance, risk_position_size, regime
            )
            
            # Step 5: Validate and bound final position size
            validated_position_size = self._validate_position_size(
                final_position_size, signal, balance, regime
            )
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Create risk metrics
            risk_metrics = self._calculate_risk_metrics(
                signal, balance, risk_position_size, validated_position_size, regime
            )
            
            # Create decision metadata
            metadata = self._create_decision_metadata(
                df, index, balance, current_positions, regime, signal,
                risk_position_size, validated_position_size
            )
            
            # Create trading decision
            decision = TradingDecision(
                timestamp=timestamp,
                signal=signal,
                position_size=validated_position_size,
                regime=regime,
                risk_metrics=risk_metrics,
                execution_time_ms=execution_time_ms,
                metadata=metadata
            )
            
            # Record decision
            self._record_decision(decision)
            
            # Log decision
            if self.enable_logging:
                self._log_decision(decision)
            
            return decision
            
        except Exception as e:
            # Handle errors gracefully
            execution_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"Error processing candle at index {index}: {e}")
            
            # Return safe decision
            safe_signal = Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={'error': str(e), 'component': 'strategy'}
            )
            
            decision = TradingDecision(
                timestamp=timestamp,
                signal=safe_signal,
                position_size=0.0,
                regime=None,
                risk_metrics={'error': True},
                execution_time_ms=execution_time_ms,
                metadata={'error': str(e), 'safe_mode': True}
            )
            
            self._record_decision(decision)
            return decision
    
    def should_exit_position(self, position: Position, current_data: MarketData,
                           regime: Optional[RegimeContext] = None) -> bool:
        """
        Determine if a position should be exited
        
        Args:
            position: Current position to evaluate
            current_data: Current market data
            regime: Optional regime context
            
        Returns:
            True if position should be exited, False otherwise
        """
        try:
            return self.risk_manager.should_exit(position, current_data, regime)
        except Exception as e:
            self.logger.error(f"Error in exit decision: {e}")
            return False  # Conservative default
    
    def get_stop_loss_price(self, entry_price: float, signal: Signal,
                          regime: Optional[RegimeContext] = None) -> float:
        """
        Get stop loss price for a position
        
        Args:
            entry_price: Entry price for the position
            signal: Trading signal that triggered the position
            regime: Optional regime context
            
        Returns:
            Stop loss price level
        """
        try:
            return self.risk_manager.get_stop_loss(entry_price, signal, regime)
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            # Return conservative stop loss
            if signal.direction == SignalDirection.BUY:
                return entry_price * 0.95  # 5% stop loss for long
            elif signal.direction == SignalDirection.SELL:
                return entry_price * 1.05  # 5% stop loss for short
            else:
                return entry_price
    
    def get_performance_metrics(self, lookback_decisions: int = 100) -> Dict[str, Any]:
        """
        Get strategy performance metrics
        
        Args:
            lookback_decisions: Number of recent decisions to analyze
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.decision_history:
            return self.metrics.copy()
        
        # Get recent decisions
        recent_decisions = self.decision_history[-lookback_decisions:]
        
        # Calculate metrics
        total_decisions = len(recent_decisions)
        buy_signals = sum(1 for d in recent_decisions if d.signal.direction == SignalDirection.BUY)
        sell_signals = sum(1 for d in recent_decisions if d.signal.direction == SignalDirection.SELL)
        hold_signals = sum(1 for d in recent_decisions if d.signal.direction == SignalDirection.HOLD)
        
        avg_execution_time = sum(d.execution_time_ms for d in recent_decisions) / total_decisions
        avg_confidence = sum(d.signal.confidence for d in recent_decisions) / total_decisions
        avg_position_size = sum(d.position_size for d in recent_decisions) / total_decisions
        
        # Regime analysis
        regime_distribution = {}
        for decision in recent_decisions:
            if decision.regime:
                regime_key = f"{decision.regime.trend.value}_{decision.regime.volatility.value}"
                regime_distribution[regime_key] = regime_distribution.get(regime_key, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'buy_signal_pct': (buy_signals / total_decisions) * 100 if total_decisions > 0 else 0,
            'sell_signal_pct': (sell_signals / total_decisions) * 100 if total_decisions > 0 else 0,
            'hold_signal_pct': (hold_signals / total_decisions) * 100 if total_decisions > 0 else 0,
            'avg_execution_time_ms': avg_execution_time,
            'avg_signal_confidence': avg_confidence,
            'avg_position_size': avg_position_size,
            'regime_distribution': regime_distribution,
            'component_info': {
                'signal_generator': self.signal_generator.get_parameters(),
                'risk_manager': self.risk_manager.get_parameters(),
                'position_sizer': self.position_sizer.get_parameters()
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent trading decisions
        
        Args:
            count: Number of recent decisions to return
            
        Returns:
            List of decision dictionaries
        """
        recent = self.decision_history[-count:] if self.decision_history else []
        return [decision.to_dict() for decision in recent]
    
    def clear_history(self) -> None:
        """Clear decision history and reset metrics"""
        self.decision_history.clear()
        self.metrics = {
            'total_decisions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_execution_time_ms': 0.0,
            'avg_signal_confidence': 0.0,
            'avg_position_size': 0.0,
            'last_updated': datetime.now()
        }
        self.logger.info("Strategy history and metrics cleared")
    
    def get_component_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all components"""
        return {
            'signal_generator': self.signal_generator.get_parameters(),
            'risk_manager': self.risk_manager.get_parameters(),
            'position_sizer': self.position_sizer.get_parameters(),
            'regime_detector': {'type': 'EnhancedRegimeDetector'}
        }
    
    def _validate_inputs(self, df: pd.DataFrame, index: int, balance: float) -> None:
        """Validate input parameters"""
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} is out of bounds for DataFrame of length {len(df)}")
        
        if balance <= 0:
            raise ValueError(f"Balance must be positive, got {balance}")
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    def _detect_regime(self, df: pd.DataFrame, index: int) -> Optional[RegimeContext]:
        """Detect market regime"""
        try:
            return self.regime_detector.detect_regime(df, index)
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return None
    
    def _generate_signal(self, df: pd.DataFrame, index: int, 
                        regime: Optional[RegimeContext]) -> Signal:
        """Generate trading signal"""
        try:
            return self.signal_generator.generate_signal(df, index, regime)
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={'error': str(e), 'component': 'signal_generator'}
            )
    
    def _calculate_risk_position_size(self, signal: Signal, balance: float,
                                    regime: Optional[RegimeContext]) -> float:
        """Calculate risk-based position size"""
        try:
            return self.risk_manager.calculate_position_size(signal, balance, regime)
        except Exception as e:
            self.logger.error(f"Risk position size calculation failed: {e}")
            return 0.0
    
    def _calculate_final_position_size(self, signal: Signal, balance: float,
                                     risk_amount: float, regime: Optional[RegimeContext]) -> float:
        """Calculate final position size using position sizer"""
        try:
            return self.position_sizer.calculate_size(signal, balance, risk_amount, regime)
        except Exception as e:
            self.logger.error(f"Final position size calculation failed: {e}")
            return risk_amount  # Fallback to risk manager's calculation
    
    def _validate_position_size(self, position_size: float, signal: Signal,
                              balance: float, regime: Optional[RegimeContext]) -> float:
        """Validate and bound position size"""
        if signal.direction == SignalDirection.HOLD:
            return 0.0
        
        # Apply reasonable bounds
        max_position = balance * 0.25  # Maximum 25% of balance
        min_position = balance * 0.001  # Minimum 0.1% of balance
        
        # Respect zero position (no trade decision)
        if position_size == 0.0:
            return 0.0
        
        # Apply bounds only for positive positions
        return max(min_position, min(max_position, position_size))
    
    def _calculate_risk_metrics(self, signal: Signal, balance: float,
                              risk_position_size: float, final_position_size: float,
                              regime: Optional[RegimeContext]) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        return {
            'risk_position_size': risk_position_size,
            'final_position_size': final_position_size,
            'position_size_ratio': final_position_size / risk_position_size if risk_position_size > 0 else 0,
            'balance_risk_pct': (final_position_size / balance) * 100 if balance > 0 else 0,
            'signal_confidence': signal.confidence,
            'signal_strength': signal.strength,
            'regime_confidence': regime.confidence if regime else 0.0
        }
    
    def _create_decision_metadata(self, df: pd.DataFrame, index: int, balance: float,
                                current_positions: Optional[List[Position]],
                                regime: Optional[RegimeContext], signal: Signal,
                                risk_position_size: float, final_position_size: float) -> Dict[str, Any]:
        """Create comprehensive decision metadata"""
        metadata = {
            'strategy_name': self.name,
            'index': index,
            'timestamp_data': df.index[index] if hasattr(df.index, '__getitem__') else None,
            'balance': balance,
            'current_positions_count': len(current_positions) if current_positions else 0,
            'components': {
                'signal_generator': self.signal_generator.name,
                'risk_manager': self.risk_manager.name,
                'position_sizer': self.position_sizer.name
            },
            'market_data': {
                'open': float(df.iloc[index]['open']),
                'high': float(df.iloc[index]['high']),
                'low': float(df.iloc[index]['low']),
                'close': float(df.iloc[index]['close']),
                'volume': float(df.iloc[index]['volume'])
            },
            'decision_flow': {
                'risk_position_size': risk_position_size,
                'final_position_size': final_position_size,
                'size_adjustment_ratio': final_position_size / risk_position_size if risk_position_size > 0 else 0
            }
        }
        
        # Add regime information if available
        if regime:
            metadata['regime'] = {
                'trend': regime.trend.value,
                'volatility': regime.volatility.value,
                'confidence': regime.confidence,
                'duration': regime.duration,
                'strength': regime.strength
            }
        
        return metadata
    
    def _record_decision(self, decision: TradingDecision) -> None:
        """Record decision in history and update metrics"""
        self.decision_history.append(decision)
        
        # Limit history size
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-int(self.max_history * 0.8):]
        
        # Update metrics
        self.metrics['total_decisions'] += 1
        
        if decision.signal.direction == SignalDirection.BUY:
            self.metrics['buy_signals'] += 1
        elif decision.signal.direction == SignalDirection.SELL:
            self.metrics['sell_signals'] += 1
        else:
            self.metrics['hold_signals'] += 1
        
        # Update running averages
        total = self.metrics['total_decisions']
        self.metrics['avg_execution_time_ms'] = (
            (self.metrics['avg_execution_time_ms'] * (total - 1) + decision.execution_time_ms) / total
        )
        self.metrics['avg_signal_confidence'] = (
            (self.metrics['avg_signal_confidence'] * (total - 1) + decision.signal.confidence) / total
        )
        self.metrics['avg_position_size'] = (
            (self.metrics['avg_position_size'] * (total - 1) + decision.position_size) / total
        )
        
        self.metrics['last_updated'] = datetime.now()
    
    def _log_decision(self, decision: TradingDecision) -> None:
        """Log trading decision"""
        regime_str = ""
        if decision.regime:
            regime_str = f" | Regime: {decision.regime.trend.value}/{decision.regime.volatility.value} (conf: {decision.regime.confidence:.2f})"
        
        self.logger.info(
            f"Decision: {decision.signal.direction.value.upper()} "
            f"| Size: {decision.position_size:.2f} "
            f"| Confidence: {decision.signal.confidence:.2f} "
            f"| Strength: {decision.signal.strength:.2f} "
            f"| Time: {decision.execution_time_ms:.1f}ms"
            f"{regime_str}"
        )
    
    def __str__(self) -> str:
        """String representation of strategy"""
        return (f"Strategy(name='{self.name}', "
                f"signal_gen='{self.signal_generator.name}', "
                f"risk_mgr='{self.risk_manager.name}', "
                f"pos_sizer='{self.position_sizer.name}')")
    
    def __repr__(self) -> str:
        """Detailed representation of strategy"""
        return self.__str__()