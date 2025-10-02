"""
Regime-Specific Testing Framework

This module provides comprehensive testing capabilities for strategies and components
in specific market regimes, with regime filtering and regime-specific performance metrics.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.regime.detector import TrendLabel, VolLabel

from ..position_sizer import PositionSizer
from ..regime_context import RegimeContext
from ..risk_manager import RiskManager
from ..signal_generator import SignalGenerator
from ..strategy import Strategy

logger = logging.getLogger(__name__)


@dataclass
class RegimeTestResults:
    """Results from regime-specific testing"""
    regime_type: str
    regime_description: str
    test_duration: float

    # Data coverage
    total_periods: int
    regime_periods: int
    regime_coverage: float  # Percentage of data in this regime

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Trade statistics
    total_trades: int
    avg_trade_duration: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float

    # Regime-specific metrics
    regime_entry_accuracy: float  # How well strategy performs when entering regime
    regime_exit_timing: float     # How well strategy handles regime transitions
    regime_stability_score: float # Performance consistency within regime

    # Risk metrics
    volatility: float
    downside_deviation: float
    calmar_ratio: float  # Return / Max Drawdown

    # Component performance (if available)
    signal_accuracy_in_regime: Optional[float] = None
    risk_control_effectiveness: Optional[float] = None
    position_sizing_optimality: Optional[float] = None

    # Regime transition analysis
    transition_performance: Dict[str, float] = None  # Performance during regime changes

    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0


@dataclass
class RegimeComparisonResults:
    """Results from comparing performance across multiple regimes"""
    regime_results: Dict[str, RegimeTestResults]

    # Cross-regime analysis
    best_regime: str
    worst_regime: str
    regime_consistency: float  # How consistent performance is across regimes

    # Adaptation analysis
    regime_adaptation_score: float  # How well strategy adapts to different regimes
    transition_handling_score: float  # How well strategy handles regime changes

    # Overall metrics
    overall_performance: float
    regime_diversification_benefit: float  # Benefit from regime-aware approach


class RegimeTester:
    """
    Comprehensive regime-specific testing framework
    
    Provides capabilities to test strategies and components in specific market regimes,
    with regime filtering, transition analysis, and regime-specific performance metrics.
    """

    def __init__(self, test_data: pd.DataFrame, regime_detection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize regime tester
        
        Args:
            test_data: Historical market data for testing (OHLCV format)
            regime_detection_params: Parameters for regime detection algorithm
        """
        self.test_data = test_data.copy()
        self.regime_detection_params = regime_detection_params or {}

        # Validate test data
        self._validate_test_data()

        # Detect regimes in the data
        self.regime_data = self._detect_regimes()

        # Create regime-filtered datasets
        self.regime_datasets = self._create_regime_datasets()

    def _parse_trend_label(self, trend_str: str) -> TrendLabel:
        """Parse trend string to TrendLabel enum"""
        trend_map = {
            'trend_up': TrendLabel.TREND_UP,
            'trend_down': TrendLabel.TREND_DOWN,
            'range': TrendLabel.RANGE
        }
        return trend_map.get(trend_str, TrendLabel.RANGE)

    def _parse_vol_label(self, vol_str: str) -> VolLabel:
        """Parse volatility string to VolLabel enum"""
        vol_map = {
            'low_vol': VolLabel.LOW,
            'high_vol': VolLabel.HIGH,
            'medium_vol': VolLabel.LOW  # Default to LOW for medium
        }
        return vol_map.get(vol_str, VolLabel.LOW)

    def _validate_test_data(self) -> None:
        """Validate that test data has required columns and format"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.test_data.columns]

        if missing_columns:
            raise ValueError(f"Test data missing required columns: {missing_columns}")

        if len(self.test_data) < 200:
            raise ValueError("Test data must have at least 200 rows for meaningful regime analysis")

        # Check for NaN values
        if self.test_data[required_columns].isnull().any().any():
            raise ValueError("Test data contains NaN values")

    def _detect_regimes(self) -> pd.DataFrame:
        """
        Detect market regimes in the test data
        
        Returns:
            DataFrame with regime labels for each period
        """
        regime_data = pd.DataFrame(index=self.test_data.index)

        # Calculate technical indicators for regime detection
        self.test_data['sma_20'] = self.test_data['close'].rolling(20).mean()
        self.test_data['sma_50'] = self.test_data['close'].rolling(50).mean()
        self.test_data['returns'] = self.test_data['close'].pct_change()
        self.test_data['volatility'] = self.test_data['returns'].rolling(20).std()

        # Trend detection using moving averages
        trend_condition = self.test_data['sma_20'] > self.test_data['sma_50']
        price_trend = self.test_data['close'] > self.test_data['close'].shift(20)

        # Combine conditions for trend classification
        regime_data['trend'] = 'range'
        regime_data.loc[trend_condition & price_trend, 'trend'] = 'trend_up'
        regime_data.loc[~trend_condition & ~price_trend, 'trend'] = 'trend_down'

        # Volatility classification
        vol_median = self.test_data['volatility'].median()
        vol_threshold_high = vol_median * 1.5
        vol_threshold_low = vol_median * 0.7

        regime_data['volatility'] = 'medium_vol'
        regime_data.loc[self.test_data['volatility'] > vol_threshold_high, 'volatility'] = 'high_vol'
        regime_data.loc[self.test_data['volatility'] < vol_threshold_low, 'volatility'] = 'low_vol'

        # Calculate regime confidence based on signal strength
        regime_data['confidence'] = self._calculate_regime_confidence()

        # Calculate regime duration (how long current regime has persisted)
        regime_data['duration'] = self._calculate_regime_duration(regime_data)

        # Calculate regime strength (how strong the regime characteristics are)
        regime_data['strength'] = self._calculate_regime_strength()

        # Create combined regime type
        regime_data['regime_type'] = regime_data['trend'] + '_' + regime_data['volatility']

        return regime_data.dropna()

    def _calculate_regime_confidence(self) -> pd.Series:
        """Calculate confidence in regime detection"""
        # Base confidence on multiple factors

        # Trend strength (distance between moving averages)
        ma_distance = abs(self.test_data['sma_20'] - self.test_data['sma_50']) / self.test_data['close']
        trend_confidence = np.clip(ma_distance * 10, 0, 1)  # Scale to 0-1

        # Volatility consistency (how stable volatility is)
        vol_consistency = 1 - (self.test_data['volatility'].rolling(10).std() / self.test_data['volatility'])
        vol_consistency = np.clip(vol_consistency, 0, 1)

        # Combine confidences
        overall_confidence = (trend_confidence + vol_consistency) / 2
        return overall_confidence.fillna(0.5)  # Default to medium confidence

    def _calculate_regime_duration(self, regime_data: pd.DataFrame) -> pd.Series:
        """Calculate how long each regime has persisted"""
        duration = pd.Series(index=regime_data.index, dtype=int)

        current_regime = None
        current_duration = 0

        for i, (idx, row) in enumerate(regime_data.iterrows()):
            regime_key = f"{row['trend']}_{row['volatility']}"

            if regime_key == current_regime:
                current_duration += 1
            else:
                current_regime = regime_key
                current_duration = 1

            duration.iloc[i] = current_duration

        return duration

    def _calculate_regime_strength(self) -> pd.Series:
        """Calculate strength of regime characteristics"""
        # Trend strength
        price_momentum = self.test_data['close'] / self.test_data['close'].shift(20) - 1
        trend_strength = np.clip(abs(price_momentum) * 2, 0, 1)

        # Volatility strength (how extreme the volatility is)
        vol_percentile = self.test_data['volatility'].rolling(100).rank(pct=True)
        vol_strength = np.maximum(vol_percentile, 1 - vol_percentile)  # High for extreme values

        # Combine strengths
        overall_strength = (trend_strength + vol_strength) / 2
        return overall_strength.fillna(0.5)

    def _create_regime_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create filtered datasets for each regime type"""
        regime_datasets = {}

        # Get unique regime types
        regime_types = self.regime_data['regime_type'].unique()

        for regime_type in regime_types:
            # Filter data for this regime
            regime_mask = self.regime_data['regime_type'] == regime_type
            regime_indices = self.regime_data[regime_mask].index

            # Create filtered dataset
            filtered_data = self.test_data.loc[regime_indices].copy()

            # Add regime context
            filtered_data['regime_confidence'] = self.regime_data.loc[regime_indices, 'confidence']
            filtered_data['regime_duration'] = self.regime_data.loc[regime_indices, 'duration']
            filtered_data['regime_strength'] = self.regime_data.loc[regime_indices, 'strength']

            regime_datasets[regime_type] = filtered_data

        return regime_datasets

    def test_strategy_in_regime(self, strategy: Strategy, regime_type: str,
                              initial_balance: float = 10000.0) -> RegimeTestResults:
        """
        Test strategy performance in a specific market regime
        
        Args:
            strategy: Strategy to test
            regime_type: Regime type to test in (e.g., 'trend_up_low_vol')
            initial_balance: Starting balance for testing
            
        Returns:
            RegimeTestResults with regime-specific performance metrics
        """
        if regime_type not in self.regime_datasets:
            raise ValueError(f"Regime type '{regime_type}' not found in data")

        start_time = time.time()
        regime_data = self.regime_datasets[regime_type]

        if len(regime_data) < 50:
            raise ValueError(f"Insufficient data for regime '{regime_type}': {len(regime_data)} periods")

        # Initialize tracking variables
        balance = initial_balance
        positions = []
        trades = []
        portfolio_values = [balance]
        error_count = 0

        # Simulate trading in this regime
        for i in range(len(regime_data) - 1):
            try:
                current_data = regime_data.iloc[:i+1]

                # Create regime context with safe parsing and enum conversion
                regime_parts = regime_type.split('_')
                if len(regime_parts) >= 3:
                    trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                    volatility_str = regime_parts[2]
                else:
                    # Log warning for unexpected format
                    logger.warning(f"Unexpected regime_type format: '{regime_type}'. Using fallback values.")
                    trend_str = 'range'
                    volatility_str = 'low_vol'

                # Convert to enums
                trend = self._parse_trend_label(trend_str)
                volatility = self._parse_vol_label(volatility_str)
                
                # Validate and bound duration value
                raw_duration = regime_data.iloc[i]['regime_duration']
                duration = int(max(1, min(raw_duration, 1_000_000))) if not np.isnan(raw_duration) else 1
                
                regime_context = RegimeContext(
                    trend=trend,
                    volatility=volatility,
                    confidence=regime_data.iloc[i]['regime_confidence'],
                    duration=duration,
                    strength=regime_data.iloc[i]['regime_strength'],
                    metadata={'regime_type': regime_type}
                )
                
                # Process candle with strategy (pass balance, strategy detects regime internally)
                decision = strategy.process_candle(current_data, i, balance)

                # Execute trades based on decision (TradingDecision is a dataclass)
                if decision and hasattr(decision, 'signal'):
                    signal_direction = decision.signal.direction
                    direction_value = signal_direction.value if hasattr(signal_direction, 'value') else signal_direction

                    if direction_value in ['buy', 'sell']:
                        trade_result = self._execute_trade(
                            decision, regime_data.iloc[i], regime_data.iloc[i+1], balance
                        )

                        if trade_result:
                            trades.append(trade_result)
                            balance = trade_result['new_balance']

                portfolio_values.append(balance)

            except Exception as e:
                error_count += 1
                logger.error(f"Error testing strategy in regime at index {i}: {e}", exc_info=True)
                continue

        # Calculate performance metrics
        total_return = (balance - initial_balance) / initial_balance
        returns = pd.Series(portfolio_values).pct_change().dropna()

        # Performance metrics
        annualized_return = (1 + total_return) ** (252 / len(regime_data)) - 1
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(pd.Series(portfolio_values))

        # Trade statistics
        if trades:
            trade_returns = [t['return'] for t in trades]
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_trade_return = np.mean(trade_returns)
            best_trade = max(trade_returns)
            worst_trade = min(trade_returns)
            avg_trade_duration = np.mean([t['duration'] for t in trades])
        else:
            win_rate = 0.0
            avg_trade_return = 0.0
            best_trade = 0.0
            worst_trade = 0.0
            avg_trade_duration = 0.0

        # Regime-specific metrics
        regime_entry_accuracy = self._calculate_regime_entry_accuracy(trades, regime_data)
        regime_exit_timing = self._calculate_regime_exit_timing(trades, regime_data)
        regime_stability_score = self._calculate_regime_stability_score(returns)

        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0.0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        test_duration = time.time() - start_time

        return RegimeTestResults(
            regime_type=regime_type,
            regime_description=self._get_regime_description(regime_type),
            test_duration=test_duration,
            total_periods=len(self.test_data),
            regime_periods=len(regime_data),
            regime_coverage=len(regime_data) / len(self.test_data),
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            regime_entry_accuracy=regime_entry_accuracy,
            regime_exit_timing=regime_exit_timing,
            regime_stability_score=regime_stability_score,
            volatility=volatility,
            downside_deviation=downside_deviation,
            calmar_ratio=calmar_ratio,
            error_count=error_count,
            error_rate=error_count / len(regime_data) if len(regime_data) > 0 else 0.0
        )

    def _execute_trade(self, decision, entry_data: pd.Series,
                      exit_data: pd.Series, balance: float) -> Optional[Dict[str, Any]]:
        """Execute a trade based on strategy decision"""
        try:
            entry_price = entry_data['close']
            exit_price = exit_data['close']

            # Handle TradingDecision dataclass
            if hasattr(decision, 'position_size'):
                # TradingDecision dataclass
                position_size = decision.position_size
                signal_direction = decision.signal.direction
                action = signal_direction.value if hasattr(signal_direction, 'value') else signal_direction
            else:
                # Legacy dict format
                position_size = decision.get('size', balance * 0.02)  # Default 2% position
                action = decision['action']

            if action == 'buy':
                trade_return = (exit_price - entry_price) / entry_price
            else:  # sell
                trade_return = (entry_price - exit_price) / entry_price

            pnl = position_size * trade_return
            new_balance = balance + pnl

            return {
                'action': action,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'return': trade_return,
                'pnl': pnl,
                'new_balance': new_balance,
                'duration': 1,  # Single period trade for simplicity
                'entry_time': entry_data.name,
                'exit_time': exit_data.name
            }

        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return None

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown from portfolio values"""
        if len(portfolio_values) == 0:
            return 0.0

        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_regime_entry_accuracy(self, trades: List[Dict[str, Any]],
                                       regime_data: pd.DataFrame) -> float:
        """Calculate how well strategy performs when entering regime"""
        if not trades:
            return 0.0

        # Look at trades that occurred early in regime periods (high duration values indicate stable regime)
        early_regime_trades = [t for t in trades if self._is_early_regime_trade(t, regime_data)]

        if not early_regime_trades:
            return 0.0

        successful_entries = sum(1 for t in early_regime_trades if t['return'] > 0)
        return successful_entries / len(early_regime_trades)

    def _is_early_regime_trade(self, trade: Dict[str, Any], regime_data: pd.DataFrame) -> bool:
        """Check if trade occurred early in a regime period"""
        try:
            trade_time = trade['entry_time']
            if trade_time in regime_data.index:
                duration = regime_data.loc[trade_time, 'regime_duration']
                return duration <= 5  # Consider first 5 periods as "early"
        except:
            pass
        return False

    def _calculate_regime_exit_timing(self, trades: List[Dict[str, Any]],
                                    regime_data: pd.DataFrame) -> float:
        """Calculate how well strategy handles regime transitions"""
        if not trades:
            return 0.0

        # Look at trades that occurred near regime transitions (low confidence periods)
        transition_trades = [t for t in trades if self._is_transition_trade(t, regime_data)]

        if not transition_trades:
            return 0.5  # Neutral score if no transition trades

        successful_transitions = sum(1 for t in transition_trades if t['return'] > -0.02)  # Small loss acceptable
        return successful_transitions / len(transition_trades)

    def _is_transition_trade(self, trade: Dict[str, Any], regime_data: pd.DataFrame) -> bool:
        """Check if trade occurred during regime transition"""
        try:
            trade_time = trade['entry_time']
            if trade_time in regime_data.index:
                confidence = regime_data.loc[trade_time, 'regime_confidence']
                return confidence < 0.5  # Low confidence indicates transition
        except:
            pass
        return False

    def _calculate_regime_stability_score(self, returns: pd.Series) -> float:
        """Calculate performance consistency within regime"""
        if len(returns) < 10:
            return 0.0

        # Measure consistency using rolling Sharpe ratio stability
        rolling_sharpe = returns.rolling(10).apply(
            lambda x: x.mean() / x.std() if x.std() > 0 else 0
        )

        sharpe_stability = 1 - (rolling_sharpe.std() / (abs(rolling_sharpe.mean()) + 0.1))
        return max(0.0, min(1.0, sharpe_stability))

    def _get_regime_description(self, regime_type: str) -> str:
        """Get human-readable description of regime type"""
        parts = regime_type.split('_')

        trend_desc = {
            'trend_up': 'Bull Market',
            'trend_down': 'Bear Market',
            'range': 'Sideways Market'
        }.get('_'.join(parts[:2]), 'Unknown Trend')

        vol_desc = {
            'low_vol': 'Low Volatility',
            'high_vol': 'High Volatility',
            'medium_vol': 'Medium Volatility'
        }.get(parts[-1], 'Unknown Volatility')

        return f"{trend_desc} with {vol_desc}"

    def compare_regime_performance(self, strategy: Strategy,
                                 regime_types: Optional[List[str]] = None,
                                 initial_balance: float = 10000.0) -> RegimeComparisonResults:
        """
        Compare strategy performance across multiple regimes
        
        Args:
            strategy: Strategy to test
            regime_types: List of regime types to compare (None = all available)
            initial_balance: Starting balance for testing
            
        Returns:
            RegimeComparisonResults with cross-regime analysis
        """
        if regime_types is None:
            regime_types = list(self.regime_datasets.keys())

        # Test strategy in each regime
        regime_results = {}
        for regime_type in regime_types:
            try:
                result = self.test_strategy_in_regime(strategy, regime_type, initial_balance)
                regime_results[regime_type] = result
            except Exception as e:
                logger.error(f"Error testing regime {regime_type}: {e}", exc_info=True)
                continue

        if not regime_results:
            raise ValueError("No valid regime test results obtained")

        # Find best and worst regimes
        regime_scores = {k: v.sharpe_ratio for k, v in regime_results.items()}
        best_regime = max(regime_scores, key=regime_scores.get)
        worst_regime = min(regime_scores, key=regime_scores.get)

        # Calculate regime consistency
        returns = [r.total_return for r in regime_results.values()]
        regime_consistency = 1 - (np.std(returns) / (abs(np.mean(returns)) + 0.1))
        regime_consistency = max(0.0, min(1.0, regime_consistency))

        # Calculate adaptation score
        sharpe_ratios = [r.sharpe_ratio for r in regime_results.values()]
        adaptation_score = np.mean([max(0, s) for s in sharpe_ratios])  # Average positive Sharpe ratios

        # Calculate transition handling score
        transition_scores = [r.regime_exit_timing for r in regime_results.values()]
        transition_handling_score = np.mean(transition_scores)

        # Overall performance (weighted by regime coverage)
        total_coverage = sum(r.regime_coverage for r in regime_results.values())
        if total_coverage > 0:
            overall_performance = sum(
                r.total_return * r.regime_coverage for r in regime_results.values()
            ) / total_coverage
        else:
            overall_performance = np.mean([r.total_return for r in regime_results.values()])

        # Regime diversification benefit (compare to single-regime performance)
        single_regime_performance = max(r.total_return for r in regime_results.values())
        regime_diversification_benefit = overall_performance / single_regime_performance if single_regime_performance > 0 else 0.0

        return RegimeComparisonResults(
            regime_results=regime_results,
            best_regime=best_regime,
            worst_regime=worst_regime,
            regime_consistency=regime_consistency,
            regime_adaptation_score=adaptation_score,
            transition_handling_score=transition_handling_score,
            overall_performance=overall_performance,
            regime_diversification_benefit=regime_diversification_benefit
        )

    def test_component_in_regime(self, component: Union[SignalGenerator, RiskManager, PositionSizer],
                               regime_type: str) -> Dict[str, Any]:
        """
        Test individual component performance in specific regime
        
        Args:
            component: Component to test
            regime_type: Regime type to test in
            
        Returns:
            Dictionary with component-specific regime performance metrics
        """
        if regime_type not in self.regime_datasets:
            raise ValueError(f"Regime type '{regime_type}' not found in data")

        regime_data = self.regime_datasets[regime_type]

        if isinstance(component, SignalGenerator):
            return self._test_signal_generator_in_regime(component, regime_data, regime_type)
        elif isinstance(component, RiskManager):
            return self._test_risk_manager_in_regime(component, regime_data, regime_type)
        elif isinstance(component, PositionSizer):
            return self._test_position_sizer_in_regime(component, regime_data, regime_type)
        else:
            raise ValueError(f"Unsupported component type: {type(component)}")

    def _test_signal_generator_in_regime(self, generator: SignalGenerator,
                                       regime_data: pd.DataFrame, regime_type: str) -> Dict[str, Any]:
        """Test signal generator in specific regime"""
        signals = []
        accuracies = []

        for i in range(len(regime_data) - 1):
            try:
                # Create regime context with safe parsing and enum conversion
                regime_parts = regime_type.split('_')
                if len(regime_parts) >= 3:
                    trend_str = f"{regime_parts[0]}_{regime_parts[1]}"
                    volatility_str = regime_parts[2]
                else:
                    # Log warning for unexpected format
                    logger.warning(f"Unexpected regime_type format: '{regime_type}'. Using fallback values.")
                    trend_str = 'range'
                    volatility_str = 'low_vol'

                # Convert to enums
                trend = self._parse_trend_label(trend_str)
                volatility = self._parse_vol_label(volatility_str)
                
                # Validate and bound duration value
                raw_duration = regime_data.iloc[i]['regime_duration']
                duration = int(max(1, min(raw_duration, 1_000_000))) if not np.isnan(raw_duration) else 1
                
                regime_context = RegimeContext(
                    trend=trend,
                    volatility=volatility,
                    confidence=regime_data.iloc[i]['regime_confidence'],
                    duration=duration,
                    strength=regime_data.iloc[i]['regime_strength'],
                    metadata={'regime_type': regime_type}
                )
                
                signal = generator.generate_signal(regime_data, i, regime_context)
                future_return = regime_data.iloc[i + 1]['close'] / regime_data.iloc[i]['close'] - 1

                # Calculate accuracy (handle both enum and string signal directions)
                direction_value = signal.direction.value if hasattr(signal.direction, 'value') else signal.direction

                if direction_value == 'buy' and future_return > 0:
                    accurate = True
                elif direction_value == 'sell' and future_return < 0:
                    accurate = True
                elif direction_value == 'hold' and abs(future_return) < 0.01:
                    accurate = True
                else:
                    accurate = False

                signals.append(signal)
                accuracies.append(accurate)

            except Exception as e:
                logger.error(f"Error testing signal generator in regime: {e}", exc_info=True)
                continue

        if not signals:
            return {'error': 'No valid signals generated'}

        accuracy = sum(accuracies) / len(accuracies)
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_strength = np.mean([s.strength for s in signals])

        return {
            'component_type': 'SignalGenerator',
            'regime_type': regime_type,
            'total_signals': len(signals),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_strength': avg_strength,
            'regime_adaptation': avg_confidence  # Use confidence as proxy for adaptation
        }

    def _test_risk_manager_in_regime(self, risk_manager: RiskManager,
                                   regime_data: pd.DataFrame, regime_type: str) -> Dict[str, Any]:
        """Test risk manager in specific regime"""
        # Placeholder implementation - would need more sophisticated testing
        return {
            'component_type': 'RiskManager',
            'regime_type': regime_type,
            'risk_control_score': 0.8,  # Placeholder
            'regime_adaptation': 0.75   # Placeholder
        }

    def _test_position_sizer_in_regime(self, position_sizer: PositionSizer,
                                     regime_data: pd.DataFrame, regime_type: str) -> Dict[str, Any]:
        """Test position sizer in specific regime"""
        # Placeholder implementation - would need more sophisticated testing
        return {
            'component_type': 'PositionSizer',
            'regime_type': regime_type,
            'sizing_optimality': 0.7,   # Placeholder
            'regime_adaptation': 0.8    # Placeholder
        }

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regimes in the test data"""
        regime_stats = {}

        for regime_type, data in self.regime_datasets.items():
            regime_stats[regime_type] = {
                'periods': len(data),
                'coverage': len(data) / len(self.test_data),
                'avg_confidence': data['regime_confidence'].mean(),
                'avg_duration': data['regime_duration'].mean(),
                'avg_strength': data['regime_strength'].mean(),
                'return_in_regime': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) if len(data) > 1 else 0.0,
                'volatility_in_regime': data['returns'].std() if 'returns' in data.columns else 0.0
            }

        return regime_stats

    def create_regime_transition_analysis(self) -> Dict[str, Any]:
        """Analyze regime transitions in the data"""
        transitions = []

        prev_regime = None
        for i, regime in enumerate(self.regime_data['regime_type']):
            if prev_regime is not None and regime != prev_regime:
                transitions.append({
                    'from_regime': prev_regime,
                    'to_regime': regime,
                    'transition_index': i,
                    'transition_date': self.regime_data.index[i]
                })
            prev_regime = regime

        # Analyze transition patterns
        transition_matrix = {}
        for transition in transitions:
            from_regime = transition['from_regime']
            to_regime = transition['to_regime']

            if from_regime not in transition_matrix:
                transition_matrix[from_regime] = {}

            if to_regime not in transition_matrix[from_regime]:
                transition_matrix[from_regime][to_regime] = 0

            transition_matrix[from_regime][to_regime] += 1

        return {
            'total_transitions': len(transitions),
            'transition_frequency': len(transitions) / len(self.regime_data),
            'transition_matrix': transition_matrix,
            'transitions': transitions
        }
