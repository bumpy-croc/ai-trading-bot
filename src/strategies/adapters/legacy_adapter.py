"""
Legacy Strategy Adapter

This module provides the LegacyStrategyAdapter class that wraps new component-based
strategies to maintain compatibility with the existing BaseStrategy interface.
"""

import logging
import time
from typing import Any, Dict, Optional

import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.components.signal_generator import SignalGenerator, SignalDirection
from src.strategies.components.risk_manager import RiskManager, Position, MarketData
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import EnhancedRegimeDetector, RegimeContext


class LegacyStrategyAdapter(BaseStrategy):
    """
    Adapter class that wraps component-based strategies to maintain BaseStrategy compatibility
    
    This adapter allows new component-based strategies to work with existing backtesting
    and live trading infrastructure that expects the BaseStrategy interface.
    """
    
    def __init__(self, 
                 signal_generator: SignalGenerator,
                 risk_manager: RiskManager,
                 position_sizer: PositionSizer,
                 regime_detector: Optional[EnhancedRegimeDetector] = None,
                 name: Optional[str] = None):
        """
        Initialize the legacy strategy adapter
        
        Args:
            signal_generator: Component for generating trading signals
            risk_manager: Component for managing risk and exits
            position_sizer: Component for calculating position sizes
            regime_detector: Optional regime detector for regime-aware behavior
            name: Strategy name (auto-generated if not provided)
        """
        # Generate name if not provided
        if name is None:
            name = f"adapter_{signal_generator.name}_{risk_manager.name}_{position_sizer.name}"
        
        super().__init__(name)
        
        # Store components
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.regime_detector = regime_detector or EnhancedRegimeDetector()
        
        # Performance tracking
        self.performance_metrics = {
            'signals_generated': 0,
            'entry_conditions_checked': 0,
            'exit_conditions_checked': 0,
            'position_sizes_calculated': 0,
            'regime_detections': 0,
            'component_errors': 0,
            'execution_times': {
                'signal_generation': [],
                'risk_management': [],
                'position_sizing': [],
                'regime_detection': []
            }
        }
        
        # Current regime context cache
        self._current_regime: Optional[RegimeContext] = None
        self._regime_cache_index: int = -1
        
        # Component state tracking
        self._last_signal = None
        self._last_regime_context = None
        
        # Logging setup
        self.adapter_logger = logging.getLogger(f"{self.name}.adapter")
        self.adapter_logger.info(f"Initialized LegacyStrategyAdapter with components: "
                               f"signal={signal_generator.name}, "
                               f"risk={risk_manager.name}, "
                               f"sizer={position_sizer.name}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific indicators on the data
        
        This method ensures regime annotations are available and adds any
        component-specific indicators needed.
        """
        try:
            # Start with a copy of the input data
            result_df = df.copy()
            
            # Ensure regime annotations exist
            if 'regime_label' not in result_df.columns:
                self.adapter_logger.debug("Adding regime annotations to DataFrame")
                result_df = self.regime_detector.base_detector.annotate(result_df)
            
            # Log successful indicator calculation
            self.adapter_logger.debug(f"Calculated indicators for {len(result_df)} rows")
            
            return result_df
            
        except Exception as e:
            self.adapter_logger.error(f"Error calculating indicators: {e}")
            self.performance_metrics['component_errors'] += 1
            # Return original DataFrame if calculation fails
            return df.copy()
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """
        Check if entry conditions are met at the given index
        
        Uses the signal generator component to determine entry conditions.
        """
        start_time = time.time()
        
        try:
            self.performance_metrics['entry_conditions_checked'] += 1
            
            # Get current regime context
            regime_context = self._get_regime_context(df, index)
            
            # Generate signal
            signal = self.signal_generator.generate_signal(df, index, regime_context)
            self._last_signal = signal
            
            # Log signal generation
            self.log_execution(
                signal_type="entry_check",
                action_taken=f"signal_{signal.direction.value}",
                price=float(df.iloc[index]['close']),
                signal_strength=signal.strength,
                confidence_score=signal.confidence,
                additional_context={
                    'regime_trend': regime_context.trend.value if regime_context else 'unknown',
                    'regime_volatility': regime_context.volatility.value if regime_context else 'unknown',
                    'regime_confidence': regime_context.confidence if regime_context else 0.0,
                    'signal_metadata': str(signal.metadata)
                }
            )
            
            self.performance_metrics['signals_generated'] += 1
            
            # Entry condition is met if signal is BUY
            entry_condition = signal.direction == SignalDirection.BUY
            
            # Record execution time
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['signal_generation'].append(execution_time)
            
            self.adapter_logger.debug(f"Entry check at index {index}: {entry_condition} "
                                    f"(signal: {signal.direction.value}, "
                                    f"strength: {signal.strength:.3f}, "
                                    f"confidence: {signal.confidence:.3f})")
            
            return entry_condition
            
        except Exception as e:
            self.adapter_logger.error(f"Error checking entry conditions at index {index}: {e}")
            self.performance_metrics['component_errors'] += 1
            
            # Record execution time even for errors
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['signal_generation'].append(execution_time)
            
            return False
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """
        Check if exit conditions are met at the given index
        
        Uses the risk manager component to determine exit conditions.
        """
        start_time = time.time()
        
        try:
            self.performance_metrics['exit_conditions_checked'] += 1
            
            # Get current regime context
            regime_context = self._get_regime_context(df, index)
            
            # Create position object for risk manager
            current_price = float(df.iloc[index]['close'])
            position = Position(
                symbol=self.trading_pair,
                side='long',  # Assuming long positions for legacy compatibility
                size=1.0,     # Normalized size for exit decision
                entry_price=entry_price,
                current_price=current_price,
                entry_time=pd.Timestamp.now(),
                unrealized_pnl=(current_price - entry_price)
            )
            
            # Create market data object
            market_data = MarketData(
                symbol=self.trading_pair,
                price=current_price,
                volume=float(df.iloc[index]['volume']),
                timestamp=pd.Timestamp.now()
            )
            
            # Check exit conditions using risk manager
            should_exit = self.risk_manager.should_exit(position, market_data, regime_context)
            
            # Log exit decision
            self.log_execution(
                signal_type="exit_check",
                action_taken=f"exit_{should_exit}",
                price=current_price,
                additional_context={
                    'entry_price': entry_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'pnl_percentage': position.get_pnl_percentage(),
                    'regime_trend': regime_context.trend.value if regime_context else 'unknown',
                    'regime_volatility': regime_context.volatility.value if regime_context else 'unknown'
                }
            )
            
            # Record execution time
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['risk_management'].append(execution_time)
            
            self.adapter_logger.debug(f"Exit check at index {index}: {should_exit} "
                                    f"(entry: {entry_price:.2f}, current: {current_price:.2f}, "
                                    f"pnl: {position.get_pnl_percentage():.2f}%)")
            
            return should_exit
            
        except Exception as e:
            self.adapter_logger.error(f"Error checking exit conditions at index {index}: {e}")
            self.performance_metrics['component_errors'] += 1
            
            # Record execution time even for errors
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['risk_management'].append(execution_time)
            
            return False
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """
        Calculate the position size for a new trade
        
        Uses both the risk manager and position sizer components to determine optimal size.
        """
        start_time = time.time()
        
        try:
            self.performance_metrics['position_sizes_calculated'] += 1
            
            # Get current regime context
            regime_context = self._get_regime_context(df, index)
            
            # Use the last generated signal or generate a new one
            if self._last_signal is None:
                signal = self.signal_generator.generate_signal(df, index, regime_context)
            else:
                signal = self._last_signal
            
            # Calculate risk amount using risk manager
            risk_amount = self.risk_manager.calculate_position_size(signal, balance, regime_context)
            
            # Calculate final position size using position sizer
            position_size = self.position_sizer.calculate_size(signal, balance, risk_amount, regime_context)
            
            # Log position sizing decision
            self.log_execution(
                signal_type="position_sizing",
                action_taken="size_calculated",
                price=float(df.iloc[index]['close']),
                position_size=position_size,
                signal_strength=signal.strength,
                confidence_score=signal.confidence,
                additional_context={
                    'balance': balance,
                    'risk_amount': risk_amount,
                    'position_fraction': position_size / balance if balance > 0 else 0,
                    'regime_trend': regime_context.trend.value if regime_context else 'unknown',
                    'regime_volatility': regime_context.volatility.value if regime_context else 'unknown'
                }
            )
            
            # Record execution time
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['position_sizing'].append(execution_time)
            
            self.adapter_logger.debug(f"Position size calculated at index {index}: {position_size:.4f} "
                                    f"(balance: {balance:.2f}, risk: {risk_amount:.4f}, "
                                    f"fraction: {position_size/balance*100:.2f}%)")
            
            return position_size
            
        except Exception as e:
            self.adapter_logger.error(f"Error calculating position size at index {index}: {e}")
            self.performance_metrics['component_errors'] += 1
            
            # Record execution time even for errors
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['position_sizing'].append(execution_time)
            
            # Return conservative fallback size
            return balance * 0.01  # 1% of balance as fallback
    
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
        """
        Calculate stop loss level for a position
        
        Uses the risk manager component to determine stop loss level.
        """
        try:
            # Get current regime context
            regime_context = self._get_regime_context(df, index)
            
            # Use the last generated signal or generate a new one
            if self._last_signal is None:
                signal = self.signal_generator.generate_signal(df, index, regime_context)
            else:
                signal = self._last_signal
            
            # Calculate stop loss using risk manager
            stop_loss = self.risk_manager.get_stop_loss(price, signal, regime_context)
            
            self.adapter_logger.debug(f"Stop loss calculated at index {index}: {stop_loss:.4f} "
                                    f"(entry: {price:.4f}, side: {side})")
            
            return stop_loss
            
        except Exception as e:
            self.adapter_logger.error(f"Error calculating stop loss at index {index}: {e}")
            self.performance_metrics['component_errors'] += 1
            
            # Return conservative fallback stop loss
            if side == "long":
                return price * 0.95  # 5% stop loss for long
            else:
                return price * 1.05  # 5% stop loss for short
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return strategy parameters for logging
        
        Combines parameters from all components.
        """
        parameters = {
            'adapter_name': self.name,
            'trading_pair': self.trading_pair,
            'signal_generator': self.signal_generator.get_parameters(),
            'risk_manager': self.risk_manager.get_parameters(),
            'position_sizer': self.position_sizer.get_parameters(),
            'performance_metrics': self.get_performance_metrics()
        }
        
        return parameters
    
    def _get_regime_context(self, df: pd.DataFrame, index: int) -> Optional[RegimeContext]:
        """
        Get regime context for the given index with caching
        
        Args:
            df: DataFrame with market data
            index: Current index position
            
        Returns:
            RegimeContext or None if detection fails
        """
        start_time = time.time()
        
        try:
            # Use cached regime if available for same index
            if self._current_regime is not None and self._regime_cache_index == index:
                return self._current_regime
            
            # Detect regime
            regime_context = self.regime_detector.detect_regime(df, index)
            
            # Cache the result
            self._current_regime = regime_context
            self._regime_cache_index = index
            self._last_regime_context = regime_context
            
            self.performance_metrics['regime_detections'] += 1
            
            # Record execution time
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['regime_detection'].append(execution_time)
            
            return regime_context
            
        except Exception as e:
            self.adapter_logger.warning(f"Error detecting regime at index {index}: {e}")
            self.performance_metrics['component_errors'] += 1
            
            # Record execution time even for errors
            execution_time = time.time() - start_time
            self.performance_metrics['execution_times']['regime_detection'].append(execution_time)
            
            # Return last known regime or None
            return self._last_regime_context
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get adapter performance metrics
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = self.performance_metrics.copy()
        
        # Calculate average execution times
        for component, times in metrics['execution_times'].items():
            if times:
                metrics[f'avg_{component}_time'] = sum(times) / len(times)
                metrics[f'max_{component}_time'] = max(times)
                metrics[f'min_{component}_time'] = min(times)
            else:
                metrics[f'avg_{component}_time'] = 0.0
                metrics[f'max_{component}_time'] = 0.0
                metrics[f'min_{component}_time'] = 0.0
        
        # Remove raw timing data to keep metrics clean
        del metrics['execution_times']
        
        # Add component information
        metrics['component_info'] = {
            'signal_generator_type': self.signal_generator.__class__.__name__,
            'risk_manager_type': self.risk_manager.__class__.__name__,
            'position_sizer_type': self.position_sizer.__class__.__name__,
            'regime_detector_type': self.regime_detector.__class__.__name__
        }
        
        return metrics
    
    def reset_performance_metrics(self) -> None:
        """Reset performance tracking metrics"""
        self.performance_metrics = {
            'signals_generated': 0,
            'entry_conditions_checked': 0,
            'exit_conditions_checked': 0,
            'position_sizes_calculated': 0,
            'regime_detections': 0,
            'component_errors': 0,
            'execution_times': {
                'signal_generation': [],
                'risk_management': [],
                'position_sizing': [],
                'regime_detection': []
            }
        }
        
        self.adapter_logger.info("Performance metrics reset")
    
    def get_component_status(self) -> Dict[str, str]:
        """
        Get status information for all components
        
        Returns:
            Dictionary with component status information
        """
        return {
            'signal_generator': f"{self.signal_generator.__class__.__name__} ({self.signal_generator.name})",
            'risk_manager': f"{self.risk_manager.__class__.__name__} ({self.risk_manager.name})",
            'position_sizer': f"{self.position_sizer.__class__.__name__} ({self.position_sizer.name})",
            'regime_detector': f"{self.regime_detector.__class__.__name__}",
            'current_regime': self._current_regime.get_regime_label() if self._current_regime else 'unknown',
            'last_signal': self._last_signal.direction.value if self._last_signal else 'none'
        }
    
    def __str__(self) -> str:
        """String representation of the adapter"""
        return (f"LegacyStrategyAdapter(name={self.name}, "
                f"signal={self.signal_generator.name}, "
                f"risk={self.risk_manager.name}, "
                f"sizer={self.position_sizer.name})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the adapter"""
        return (f"LegacyStrategyAdapter("
                f"name='{self.name}', "
                f"signal_generator={self.signal_generator.__class__.__name__}, "
                f"risk_manager={self.risk_manager.__class__.__name__}, "
                f"position_sizer={self.position_sizer.__class__.__name__}, "
                f"regime_detector={self.regime_detector.__class__.__name__})")