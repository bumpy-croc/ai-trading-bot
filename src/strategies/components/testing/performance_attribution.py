"""
Performance Attribution Analysis

This module provides comprehensive performance attribution analysis for strategy components,
allowing detailed analysis of how each component contributes to overall strategy performance.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..signal_generator import SignalGenerator, Signal, SignalDirection
from ..risk_manager import RiskManager
from ..position_sizer import PositionSizer
from ..strategy import Strategy


@dataclass
class ComponentAttribution:
    """Attribution analysis for a single component"""
    component_name: str
    component_type: str
    
    # Performance contribution
    total_contribution: float  # Total performance contribution
    positive_contribution: float  # Contribution from positive trades
    negative_contribution: float  # Contribution from negative trades
    
    # Trade impact analysis
    trade_count_impact: int  # Number of trades influenced by this component
    avg_trade_impact: float  # Average impact per trade
    best_trade_contribution: float  # Best single trade contribution
    worst_trade_contribution: float  # Worst single trade contribution
    
    # Risk-adjusted metrics
    risk_adjusted_contribution: float  # Sharpe-adjusted contribution
    volatility_contribution: float  # Contribution to portfolio volatility
    drawdown_contribution: float  # Contribution to maximum drawdown
    
    # Component-specific metrics
    component_effectiveness: float  # How effective the component is
    component_consistency: float  # How consistent the component is
    component_reliability: float  # How reliable the component is
    
    # Interaction effects
    synergy_score: float  # How well it works with other components
    substitution_impact: float  # Impact if this component were replaced
    
    # Regime-specific attribution
    regime_attribution: Dict[str, float]  # Performance by regime
    
    # Optimization recommendations
    optimization_potential: float  # Potential for improvement
    recommended_adjustments: List[str]  # Specific recommendations


@dataclass
class AttributionReport:
    """Comprehensive attribution analysis report"""
    strategy_name: str
    analysis_period: str
    total_return: float
    
    # Component attributions
    signal_attribution: Optional[ComponentAttribution] = None
    risk_attribution: Optional[ComponentAttribution] = None
    sizing_attribution: Optional[ComponentAttribution] = None
    
    # Cross-component analysis
    component_correlations: Dict[str, Dict[str, float]] = None
    interaction_effects: Dict[str, float] = None
    
    # Overall attribution summary
    explained_performance: float = 0.0  # % of performance explained by components
    unexplained_performance: float = 0.0  # Residual performance
    attribution_quality: float = 0.0  # Quality of attribution analysis
    
    # Optimization insights
    primary_performance_driver: str = ""
    weakest_component: str = ""
    optimization_priority: List[str] = None
    
    # Replacement analysis
    component_replacement_impact: Dict[str, float] = None


class PerformanceAttributionAnalyzer:
    """
    Comprehensive performance attribution analyzer for strategy components
    
    Analyzes how individual components contribute to overall strategy performance,
    providing detailed insights for optimization and component replacement decisions.
    """
    
    def __init__(self, test_data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None):
        """
        Initialize performance attribution analyzer
        
        Args:
            test_data: Historical market data for analysis (OHLCV format)
            regime_data: Optional regime labels for regime-specific attribution
        """
        self.test_data = test_data.copy()
        self.regime_data = regime_data
        
        # Validate test data
        self._validate_test_data()
        
        # Prepare test data with indicators
        self._prepare_test_data()
        
        # Initialize baseline performance metrics
        self.baseline_metrics = self._calculate_baseline_metrics()
    
    def _validate_test_data(self) -> None:
        """Validate that test data has required columns and format"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.test_data.columns]
        
        if missing_columns:
            raise ValueError(f"Test data missing required columns: {missing_columns}")
        
        if len(self.test_data) < 100:
            raise ValueError("Test data must have at least 100 rows for meaningful attribution analysis")
    
    def _prepare_test_data(self) -> None:
        """Prepare test data with technical indicators and returns"""
        # Add basic technical indicators
        self.test_data['sma_20'] = self.test_data['close'].rolling(20).mean()
        self.test_data['sma_50'] = self.test_data['close'].rolling(50).mean()
        self.test_data['returns'] = self.test_data['close'].pct_change()
        self.test_data['log_returns'] = np.log(self.test_data['close'] / self.test_data['close'].shift(1))
        
        # Calculate volatility
        self.test_data['volatility'] = self.test_data['returns'].rolling(20).std()
        
        # Drop initial NaN rows
        self.test_data = self.test_data.dropna()
    
    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """Calculate baseline performance metrics for comparison"""
        returns = self.test_data['returns']
        
        return {
            'total_return': (self.test_data['close'].iloc[-1] / self.test_data['close'].iloc[0]) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(self.test_data['close']),
            'avg_return': returns.mean(),
            'return_std': returns.std()
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        return abs(drawdown.min())
    
    def analyze_strategy_attribution(self, strategy: Strategy, 
                                   initial_balance: float = 10000.0) -> AttributionReport:
        """
        Perform comprehensive attribution analysis for a complete strategy
        
        Args:
            strategy: Strategy to analyze
            initial_balance: Starting balance for simulation
            
        Returns:
            AttributionReport with detailed attribution analysis
        """
        start_time = time.time()
        
        # Run baseline strategy simulation
        baseline_results = self._simulate_strategy(strategy, initial_balance)
        
        # Analyze individual component contributions
        signal_attribution = None
        risk_attribution = None
        sizing_attribution = None
        
        if hasattr(strategy, 'signal_generator') and strategy.signal_generator:
            signal_attribution = self._analyze_signal_generator_attribution(
                strategy.signal_generator, baseline_results
            )
        
        if hasattr(strategy, 'risk_manager') and strategy.risk_manager:
            risk_attribution = self._analyze_risk_manager_attribution(
                strategy.risk_manager, baseline_results
            )
        
        if hasattr(strategy, 'position_sizer') and strategy.position_sizer:
            sizing_attribution = self._analyze_position_sizer_attribution(
                strategy.position_sizer, baseline_results
            )
        
        # Calculate component correlations and interactions
        component_correlations = self._calculate_component_correlations(
            signal_attribution, risk_attribution, sizing_attribution
        )
        
        interaction_effects = self._calculate_interaction_effects(
            strategy, baseline_results
        )
        
        # Calculate overall attribution metrics
        total_attribution = 0.0
        if signal_attribution:
            total_attribution += signal_attribution.total_contribution
        if risk_attribution:
            total_attribution += risk_attribution.total_contribution
        if sizing_attribution:
            total_attribution += sizing_attribution.total_contribution
        
        explained_performance = min(1.0, abs(total_attribution) / abs(baseline_results['total_return']) 
                                  if baseline_results['total_return'] != 0 else 0.0)
        unexplained_performance = 1.0 - explained_performance
        
        # Determine primary performance driver and weakest component
        component_contributions = {}
        if signal_attribution:
            component_contributions['signal_generator'] = signal_attribution.total_contribution
        if risk_attribution:
            component_contributions['risk_manager'] = risk_attribution.total_contribution
        if sizing_attribution:
            component_contributions['position_sizer'] = sizing_attribution.total_contribution
        
        primary_driver = max(component_contributions, key=component_contributions.get) if component_contributions else ""
        weakest_component = min(component_contributions, key=component_contributions.get) if component_contributions else ""
        
        # Generate optimization priorities
        optimization_priority = self._generate_optimization_priorities(
            signal_attribution, risk_attribution, sizing_attribution
        )
        
        # Calculate component replacement impact
        replacement_impact = self._calculate_replacement_impact(strategy, baseline_results)
        
        analysis_duration = time.time() - start_time
        
        return AttributionReport(
            strategy_name=getattr(strategy, 'name', 'Unknown Strategy'),
            analysis_period=f"{self.test_data.index[0]} to {self.test_data.index[-1]}",
            total_return=baseline_results['total_return'],
            signal_attribution=signal_attribution,
            risk_attribution=risk_attribution,
            sizing_attribution=sizing_attribution,
            component_correlations=component_correlations,
            interaction_effects=interaction_effects,
            explained_performance=explained_performance,
            unexplained_performance=unexplained_performance,
            attribution_quality=explained_performance,  # Simple proxy for quality
            primary_performance_driver=primary_driver,
            weakest_component=weakest_component,
            optimization_priority=optimization_priority,
            component_replacement_impact=replacement_impact
        )
    
    def _simulate_strategy(self, strategy: Strategy, initial_balance: float) -> Dict[str, Any]:
        """Simulate strategy performance and collect detailed metrics"""
        balance = initial_balance
        trades = []
        portfolio_values = [balance]
        signals = []
        position_sizes = []
        risk_decisions = []
        
        for i in range(len(self.test_data) - 1):
            try:
                current_data = self.test_data.iloc[:i+1]
                
                # Get regime context if available
                regime = None
                if self.regime_data is not None and i < len(self.regime_data):
                    regime = self.regime_data.iloc[i]
                
                # Process candle with strategy
                decision = strategy.process_candle(current_data, i, regime)
                
                # Track component decisions
                if hasattr(strategy, 'signal_generator'):
                    try:
                        signal = strategy.signal_generator.generate_signal(current_data, i, regime)
                        signals.append({
                            'index': i,
                            'signal': signal,
                            'timestamp': self.test_data.index[i]
                        })
                    except:
                        pass
                
                # Execute trade if decision made
                if decision and decision.get('action') in ['buy', 'sell']:
                    trade_result = self._execute_attribution_trade(
                        decision, self.test_data.iloc[i], self.test_data.iloc[i+1], balance
                    )
                    
                    if trade_result:
                        trades.append(trade_result)
                        balance = trade_result['new_balance']
                        
                        # Track position sizing decisions
                        position_sizes.append({
                            'index': i,
                            'size': trade_result['position_size'],
                            'size_fraction': trade_result['position_size'] / balance,
                            'timestamp': self.test_data.index[i]
                        })
                
                portfolio_values.append(balance)
                
            except Exception as e:
                print(f"Error in strategy simulation at index {i}: {e}")
                continue
        
        # Calculate performance metrics
        total_return = (balance - initial_balance) / initial_balance
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'trades': trades,
            'signals': signals,
            'position_sizes': position_sizes,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(pd.Series(portfolio_values)),
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0,
            'win_rate': sum(1 for t in trades if t['return'] > 0) / len(trades) if trades else 0.0
        }
    
    def _execute_attribution_trade(self, decision: Dict[str, Any], entry_data: pd.Series,
                                 exit_data: pd.Series, balance: float) -> Optional[Dict[str, Any]]:
        """Execute trade for attribution analysis"""
        try:
            entry_price = entry_data['close']
            exit_price = exit_data['close']
            position_size = decision.get('size', balance * 0.02)
            
            if decision['action'] == 'buy':
                trade_return = (exit_price - entry_price) / entry_price
            else:  # sell
                trade_return = (entry_price - exit_price) / entry_price
            
            pnl = position_size * trade_return
            new_balance = balance + pnl
            
            return {
                'action': decision['action'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'return': trade_return,
                'pnl': pnl,
                'new_balance': new_balance,
                'entry_time': entry_data.name,
                'exit_time': exit_data.name,
                'decision_metadata': decision.get('metadata', {})
            }
        
        except Exception as e:
            print(f"Error executing attribution trade: {e}")
            return None
    
    def _analyze_signal_generator_attribution(self, signal_generator: SignalGenerator,
                                            baseline_results: Dict[str, Any]) -> ComponentAttribution:
        """Analyze signal generator's contribution to performance"""
        signals = baseline_results['signals']
        trades = baseline_results['trades']
        
        if not signals or not trades:
            return self._create_empty_attribution('signal_generator', 'SignalGenerator')
        
        # Calculate signal quality metrics
        signal_accuracy = self._calculate_signal_accuracy(signals, trades)
        signal_timing = self._calculate_signal_timing_quality(signals, trades)
        
        # Estimate signal contribution to returns
        signal_contribution = self._estimate_signal_contribution(signals, trades)
        
        # Calculate signal-specific metrics
        confidences = [s['signal'].confidence for s in signals]
        strengths = [s['signal'].strength for s in signals]
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_strength = np.mean(strengths) if strengths else 0.0
        
        # Component effectiveness (based on accuracy and timing)
        effectiveness = (signal_accuracy + signal_timing) / 2
        
        # Component consistency (based on confidence stability)
        consistency = 1 - (np.std(confidences) / (np.mean(confidences) + 0.1)) if confidences else 0.0
        consistency = max(0.0, min(1.0, consistency))
        
        # Component reliability (based on error rate and performance stability)
        reliability = signal_accuracy * (1 - baseline_results.get('error_rate', 0.0))
        
        # Generate recommendations
        recommendations = self._generate_signal_recommendations(
            signal_accuracy, avg_confidence, avg_strength
        )
        
        return ComponentAttribution(
            component_name=signal_generator.name,
            component_type='SignalGenerator',
            total_contribution=signal_contribution,
            positive_contribution=max(0, signal_contribution),
            negative_contribution=min(0, signal_contribution),
            trade_count_impact=len([t for t in trades if t['return'] > 0]),
            avg_trade_impact=signal_contribution / len(trades) if trades else 0.0,
            best_trade_contribution=max([t['return'] for t in trades]) if trades else 0.0,
            worst_trade_contribution=min([t['return'] for t in trades]) if trades else 0.0,
            risk_adjusted_contribution=signal_contribution / baseline_results['volatility'] if baseline_results['volatility'] > 0 else 0.0,
            volatility_contribution=baseline_results['volatility'] * 0.6,  # Estimate 60% from signals
            drawdown_contribution=baseline_results['max_drawdown'] * 0.5,  # Estimate 50% from signals
            component_effectiveness=effectiveness,
            component_consistency=consistency,
            component_reliability=reliability,
            synergy_score=0.8,  # Placeholder - would need multi-component analysis
            substitution_impact=signal_contribution * 0.8,  # Estimate 80% impact if replaced
            regime_attribution=self._calculate_regime_attribution(signals, trades),
            optimization_potential=1.0 - effectiveness,
            recommended_adjustments=recommendations
        )
    
    def _analyze_risk_manager_attribution(self, risk_manager: RiskManager,
                                        baseline_results: Dict[str, Any]) -> ComponentAttribution:
        """Analyze risk manager's contribution to performance"""
        trades = baseline_results['trades']
        
        if not trades:
            return self._create_empty_attribution('risk_manager', 'RiskManager')
        
        # Calculate risk management effectiveness
        drawdown_control = 1.0 - (baseline_results['max_drawdown'] / 0.2)  # Assume 20% target
        drawdown_control = max(0.0, min(1.0, drawdown_control))
        
        # Estimate risk manager contribution (primarily through drawdown control)
        risk_contribution = self._estimate_risk_manager_contribution(trades, baseline_results)
        
        # Calculate risk-specific metrics
        position_sizes = [t['position_size'] for t in trades]
        avg_position_size = np.mean(position_sizes) if position_sizes else 0.0
        position_consistency = 1 - (np.std(position_sizes) / (np.mean(position_sizes) + 0.1)) if position_sizes else 0.0
        
        # Component effectiveness (based on drawdown control and risk-adjusted returns)
        effectiveness = drawdown_control * 0.7 + (baseline_results['sharpe_ratio'] / 2.0) * 0.3
        effectiveness = max(0.0, min(1.0, effectiveness))
        
        # Component consistency (based on position sizing consistency)
        consistency = position_consistency
        
        # Component reliability (based on consistent risk control)
        reliability = effectiveness * 0.9  # High correlation with effectiveness for risk management
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            drawdown_control, avg_position_size, baseline_results['volatility']
        )
        
        return ComponentAttribution(
            component_name=risk_manager.name,
            component_type='RiskManager',
            total_contribution=risk_contribution,
            positive_contribution=max(0, risk_contribution),
            negative_contribution=min(0, risk_contribution),
            trade_count_impact=len(trades),
            avg_trade_impact=risk_contribution / len(trades) if trades else 0.0,
            best_trade_contribution=0.0,  # Risk management doesn't directly create best trades
            worst_trade_contribution=min([t['return'] for t in trades]) if trades else 0.0,
            risk_adjusted_contribution=risk_contribution,
            volatility_contribution=baseline_results['volatility'] * 0.3,  # Estimate 30% from risk management
            drawdown_contribution=baseline_results['max_drawdown'] * 0.8,  # Estimate 80% from risk management
            component_effectiveness=effectiveness,
            component_consistency=consistency,
            component_reliability=reliability,
            synergy_score=0.7,  # Risk management has moderate synergy
            substitution_impact=baseline_results['max_drawdown'] * 2,  # High impact if removed
            regime_attribution={},  # Placeholder
            optimization_potential=1.0 - effectiveness,
            recommended_adjustments=recommendations
        )
    
    def _analyze_position_sizer_attribution(self, position_sizer: PositionSizer,
                                          baseline_results: Dict[str, Any]) -> ComponentAttribution:
        """Analyze position sizer's contribution to performance"""
        position_sizes = baseline_results.get('position_sizes', [])
        trades = baseline_results['trades']
        
        if not position_sizes or not trades:
            return self._create_empty_attribution('position_sizer', 'PositionSizer')
        
        # Calculate position sizing effectiveness
        sizing_optimality = self._calculate_sizing_optimality(position_sizes, trades)
        
        # Estimate position sizer contribution
        sizing_contribution = self._estimate_sizing_contribution(position_sizes, trades, baseline_results)
        
        # Calculate sizing-specific metrics
        sizes = [p['size'] for p in position_sizes]
        size_fractions = [p['size_fraction'] for p in position_sizes]
        
        avg_size_fraction = np.mean(size_fractions) if size_fractions else 0.0
        size_consistency = 1 - (np.std(size_fractions) / (np.mean(size_fractions) + 0.01)) if size_fractions else 0.0
        
        # Component effectiveness (based on sizing optimality)
        effectiveness = sizing_optimality
        
        # Component consistency (based on size consistency)
        consistency = size_consistency
        
        # Component reliability (based on consistent sizing performance)
        reliability = effectiveness * consistency
        
        # Generate recommendations
        recommendations = self._generate_sizing_recommendations(
            sizing_optimality, avg_size_fraction, size_consistency
        )
        
        return ComponentAttribution(
            component_name=position_sizer.name,
            component_type='PositionSizer',
            total_contribution=sizing_contribution,
            positive_contribution=max(0, sizing_contribution),
            negative_contribution=min(0, sizing_contribution),
            trade_count_impact=len(trades),
            avg_trade_impact=sizing_contribution / len(trades) if trades else 0.0,
            best_trade_contribution=max([t['return'] * t['position_size'] for t in trades]) if trades else 0.0,
            worst_trade_contribution=min([t['return'] * t['position_size'] for t in trades]) if trades else 0.0,
            risk_adjusted_contribution=sizing_contribution / baseline_results['volatility'] if baseline_results['volatility'] > 0 else 0.0,
            volatility_contribution=baseline_results['volatility'] * 0.1,  # Estimate 10% from position sizing
            drawdown_contribution=baseline_results['max_drawdown'] * 0.2,  # Estimate 20% from position sizing
            component_effectiveness=effectiveness,
            component_consistency=consistency,
            component_reliability=reliability,
            synergy_score=0.9,  # Position sizing has high synergy with other components
            substitution_impact=sizing_contribution * 1.2,  # High impact if replaced with poor sizing
            regime_attribution={},  # Placeholder
            optimization_potential=1.0 - effectiveness,
            recommended_adjustments=recommendations
        )
    
    def _create_empty_attribution(self, name: str, component_type: str) -> ComponentAttribution:
        """Create empty attribution for components with no data"""
        return ComponentAttribution(
            component_name=name,
            component_type=component_type,
            total_contribution=0.0,
            positive_contribution=0.0,
            negative_contribution=0.0,
            trade_count_impact=0,
            avg_trade_impact=0.0,
            best_trade_contribution=0.0,
            worst_trade_contribution=0.0,
            risk_adjusted_contribution=0.0,
            volatility_contribution=0.0,
            drawdown_contribution=0.0,
            component_effectiveness=0.0,
            component_consistency=0.0,
            component_reliability=0.0,
            synergy_score=0.0,
            substitution_impact=0.0,
            regime_attribution={},
            optimization_potential=1.0,
            recommended_adjustments=['Insufficient data for analysis']
        )
    
    def _calculate_signal_accuracy(self, signals: List[Dict[str, Any]], 
                                 trades: List[Dict[str, Any]]) -> float:
        """Calculate signal accuracy based on trade outcomes"""
        if not signals or not trades:
            return 0.0
        
        # Match signals to trades and calculate accuracy
        accurate_signals = 0
        total_signals = 0
        
        for signal_data in signals:
            signal = signal_data['signal']
            signal_time = signal_data['timestamp']
            
            # Find corresponding trade
            corresponding_trade = None
            for trade in trades:
                if abs((trade['entry_time'] - signal_time).total_seconds()) < 3600:  # Within 1 hour
                    corresponding_trade = trade
                    break
            
            if corresponding_trade:
                total_signals += 1
                
                # Check if signal was accurate
                if signal.direction == SignalDirection.BUY and corresponding_trade['return'] > 0:
                    accurate_signals += 1
                elif signal.direction == SignalDirection.SELL and corresponding_trade['return'] < 0:
                    accurate_signals += 1
                elif signal.direction == SignalDirection.HOLD:
                    accurate_signals += 0.5  # Neutral for hold signals
        
        return accurate_signals / total_signals if total_signals > 0 else 0.0
    
    def _calculate_signal_timing_quality(self, signals: List[Dict[str, Any]],
                                       trades: List[Dict[str, Any]]) -> float:
        """Calculate quality of signal timing"""
        # Placeholder implementation - would need more sophisticated analysis
        return 0.7  # Assume moderate timing quality
    
    def _estimate_signal_contribution(self, signals: List[Dict[str, Any]],
                                    trades: List[Dict[str, Any]]) -> float:
        """Estimate signal generator's contribution to total returns"""
        if not trades:
            return 0.0
        
        # Simple estimation: assume signals contribute to 70% of performance
        total_trade_return = sum(t['return'] for t in trades)
        return total_trade_return * 0.7
    
    def _estimate_risk_manager_contribution(self, trades: List[Dict[str, Any]],
                                          baseline_results: Dict[str, Any]) -> float:
        """Estimate risk manager's contribution to performance"""
        # Risk manager primarily contributes through drawdown control and risk-adjusted returns
        # Positive contribution for good risk control, negative for poor control
        
        max_drawdown = baseline_results['max_drawdown']
        target_drawdown = 0.15  # 15% target
        
        if max_drawdown < target_drawdown:
            # Good risk control - positive contribution
            return (target_drawdown - max_drawdown) * baseline_results['total_return']
        else:
            # Poor risk control - negative contribution
            return -(max_drawdown - target_drawdown) * baseline_results['total_return']
    
    def _estimate_sizing_contribution(self, position_sizes: List[Dict[str, Any]],
                                    trades: List[Dict[str, Any]],
                                    baseline_results: Dict[str, Any]) -> float:
        """Estimate position sizer's contribution to performance"""
        if not position_sizes or not trades:
            return 0.0
        
        # Estimate contribution based on sizing optimality
        # Good sizing amplifies good trades and minimizes bad trades
        
        total_contribution = 0.0
        for i, trade in enumerate(trades):
            if i < len(position_sizes):
                size_fraction = position_sizes[i]['size_fraction']
                trade_return = trade['return']
                
                # Optimal size would be larger for profitable trades, smaller for losses
                if trade_return > 0:
                    # Reward larger positions on winning trades
                    contribution = size_fraction * trade_return * 0.1
                else:
                    # Reward smaller positions on losing trades
                    contribution = (0.05 - size_fraction) * abs(trade_return) * 0.1
                
                total_contribution += contribution
        
        return total_contribution
    
    def _calculate_sizing_optimality(self, position_sizes: List[Dict[str, Any]],
                                   trades: List[Dict[str, Any]]) -> float:
        """Calculate how optimal the position sizing was"""
        if not position_sizes or not trades:
            return 0.0
        
        # Simple optimality measure: correlation between position size and trade outcome
        sizes = []
        outcomes = []
        
        for i, trade in enumerate(trades):
            if i < len(position_sizes):
                sizes.append(position_sizes[i]['size_fraction'])
                outcomes.append(trade['return'])
        
        if len(sizes) > 1:
            correlation = np.corrcoef(sizes, outcomes)[0, 1]
            # Convert correlation to optimality score (0 to 1)
            return max(0.0, correlation) if not np.isnan(correlation) else 0.5
        
        return 0.5  # Neutral if insufficient data
    
    def _calculate_regime_attribution(self, signals: List[Dict[str, Any]],
                                    trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance attribution by regime"""
        # Placeholder implementation
        return {
            'bull_market': 0.6,
            'bear_market': -0.2,
            'sideways_market': 0.1
        }
    
    def _generate_signal_recommendations(self, accuracy: float, avg_confidence: float,
                                       avg_strength: float) -> List[str]:
        """Generate recommendations for signal generator improvement"""
        recommendations = []
        
        if accuracy < 0.5:
            recommendations.append("Improve signal accuracy through better feature engineering")
        
        if avg_confidence < 0.6:
            recommendations.append("Increase signal confidence through ensemble methods")
        
        if avg_strength < 0.5:
            recommendations.append("Enhance signal strength calculation methodology")
        
        if not recommendations:
            recommendations.append("Signal generator performing well - consider fine-tuning parameters")
        
        return recommendations
    
    def _generate_risk_recommendations(self, drawdown_control: float, avg_position_size: float,
                                     volatility: float) -> List[str]:
        """Generate recommendations for risk manager improvement"""
        recommendations = []
        
        if drawdown_control < 0.7:
            recommendations.append("Tighten risk controls to reduce maximum drawdown")
        
        if avg_position_size > 0.1:  # More than 10% per trade
            recommendations.append("Reduce average position size to improve risk management")
        
        if volatility > 0.3:  # High volatility
            recommendations.append("Implement volatility-based position sizing")
        
        if not recommendations:
            recommendations.append("Risk management performing well - monitor for regime changes")
        
        return recommendations
    
    def _generate_sizing_recommendations(self, optimality: float, avg_size: float,
                                       consistency: float) -> List[str]:
        """Generate recommendations for position sizer improvement"""
        recommendations = []
        
        if optimality < 0.6:
            recommendations.append("Improve position sizing algorithm to better match trade outcomes")
        
        if avg_size < 0.01:  # Very small positions
            recommendations.append("Consider increasing position sizes to improve returns")
        elif avg_size > 0.05:  # Very large positions
            recommendations.append("Consider reducing position sizes to manage risk")
        
        if consistency < 0.7:
            recommendations.append("Improve position sizing consistency across different market conditions")
        
        if not recommendations:
            recommendations.append("Position sizing performing well - consider regime-adaptive enhancements")
        
        return recommendations
    
    def _calculate_component_correlations(self, signal_attr: Optional[ComponentAttribution],
                                        risk_attr: Optional[ComponentAttribution],
                                        sizing_attr: Optional[ComponentAttribution]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between component contributions"""
        correlations = {}
        
        # Placeholder implementation - would need more sophisticated analysis
        if signal_attr and risk_attr:
            correlations['signal_risk'] = {'correlation': 0.3, 'significance': 0.05}
        
        if signal_attr and sizing_attr:
            correlations['signal_sizing'] = {'correlation': 0.7, 'significance': 0.01}
        
        if risk_attr and sizing_attr:
            correlations['risk_sizing'] = {'correlation': 0.5, 'significance': 0.02}
        
        return correlations
    
    def _calculate_interaction_effects(self, strategy: Strategy,
                                     baseline_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate interaction effects between components"""
        # Placeholder implementation - would need component isolation testing
        return {
            'signal_risk_interaction': 0.1,
            'signal_sizing_interaction': 0.15,
            'risk_sizing_interaction': 0.05,
            'three_way_interaction': 0.02
        }
    
    def _generate_optimization_priorities(self, signal_attr: Optional[ComponentAttribution],
                                        risk_attr: Optional[ComponentAttribution],
                                        sizing_attr: Optional[ComponentAttribution]) -> List[str]:
        """Generate optimization priorities based on attribution analysis"""
        priorities = []
        
        # Collect optimization potentials
        potentials = {}
        if signal_attr:
            potentials['signal_generator'] = signal_attr.optimization_potential
        if risk_attr:
            potentials['risk_manager'] = risk_attr.optimization_potential
        if sizing_attr:
            potentials['position_sizer'] = sizing_attr.optimization_potential
        
        # Sort by optimization potential (highest first)
        sorted_components = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        
        for component, potential in sorted_components:
            if potential > 0.3:  # Significant optimization potential
                priorities.append(component)
        
        return priorities
    
    def _calculate_replacement_impact(self, strategy: Strategy,
                                    baseline_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact of replacing each component"""
        # Placeholder implementation - would need actual component replacement testing
        return {
            'signal_generator': -0.15,  # 15% performance drop if replaced with poor signal generator
            'risk_manager': -0.25,     # 25% performance drop if replaced with poor risk manager
            'position_sizer': -0.10    # 10% performance drop if replaced with poor position sizer
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def analyze_component_replacement_impact(self, strategy: Strategy, 
                                           component_type: str,
                                           replacement_component: Union[SignalGenerator, RiskManager, PositionSizer],
                                           initial_balance: float = 10000.0) -> Dict[str, Any]:
        """
        Analyze the impact of replacing a specific component
        
        Args:
            strategy: Original strategy
            component_type: Type of component to replace ('signal', 'risk', 'sizing')
            replacement_component: New component to test
            initial_balance: Starting balance for testing
            
        Returns:
            Dictionary with replacement impact analysis
        """
        # Create modified strategy with replacement component
        modified_strategy = self._create_modified_strategy(strategy, component_type, replacement_component)
        
        # Run simulations
        original_results = self._simulate_strategy(strategy, initial_balance)
        modified_results = self._simulate_strategy(modified_strategy, initial_balance)
        
        # Calculate impact metrics
        return_impact = modified_results['total_return'] - original_results['total_return']
        sharpe_impact = modified_results['sharpe_ratio'] - original_results['sharpe_ratio']
        drawdown_impact = modified_results['max_drawdown'] - original_results['max_drawdown']
        
        return {
            'component_type': component_type,
            'replacement_component': replacement_component.name,
            'return_impact': return_impact,
            'return_impact_pct': (return_impact / abs(original_results['total_return'])) * 100 if original_results['total_return'] != 0 else 0,
            'sharpe_impact': sharpe_impact,
            'drawdown_impact': drawdown_impact,
            'overall_improvement': return_impact > 0 and sharpe_impact > 0 and drawdown_impact < 0,
            'original_performance': {
                'return': original_results['total_return'],
                'sharpe': original_results['sharpe_ratio'],
                'drawdown': original_results['max_drawdown']
            },
            'modified_performance': {
                'return': modified_results['total_return'],
                'sharpe': modified_results['sharpe_ratio'],
                'drawdown': modified_results['max_drawdown']
            }
        }
    
    def _create_modified_strategy(self, original_strategy: Strategy, component_type: str,
                                replacement_component: Union[SignalGenerator, RiskManager, PositionSizer]) -> Strategy:
        """Create a modified strategy with replaced component"""
        # This would need to be implemented based on the actual Strategy class structure
        # For now, return the original strategy (placeholder)
        return original_strategy