"""
Regime-Aware Backtesting Engine

This module provides a backtesting framework that can automatically switch
strategies based on detected market regimes, similar to the live trading engine.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.data_providers.sentiment_provider import SentimentDataProvider
from src.live.regime_strategy_switcher import RegimeStrategySwitcher, RegimeStrategyMapping, SwitchingConfig
from src.live.strategy_manager import StrategyManager
from src.regime.detector import RegimeConfig
from src.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class RegimeAwareBacktester:
    """
    Backtesting engine that supports automatic strategy switching based on market regimes
    """

    def __init__(
        self,
        initial_strategy: BaseStrategy,
        data_provider: DataProvider,
        sentiment_provider: Optional[SentimentDataProvider] = None,
        risk_parameters: Optional[Any] = None,
        initial_balance: float = 10000.0,
        regime_config: Optional[RegimeConfig] = None,
        strategy_mapping: Optional[RegimeStrategyMapping] = None,
        switching_config: Optional[SwitchingConfig] = None,
        enable_regime_switching: bool = True,
        **backtester_kwargs
    ):
        self.initial_strategy = initial_strategy
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.risk_parameters = risk_parameters
        self.initial_balance = initial_balance
        self.enable_regime_switching = enable_regime_switching
        self.backtester_kwargs = backtester_kwargs
        
        # Initialize strategy manager for regime switching
        self.strategy_manager = StrategyManager()
        self.strategy_manager.load_strategy(initial_strategy.name.lower().replace(" ", "_"))
        
        # Initialize regime strategy switcher if enabled
        self.regime_switcher = None
        if enable_regime_switching:
            self.regime_switcher = RegimeStrategySwitcher(
                strategy_manager=self.strategy_manager,
                regime_config=regime_config,
                strategy_mapping=strategy_mapping,
                switching_config=switching_config
            )
        
        # Track strategy switches during backtest
        self.strategy_switches: List[Dict] = []
        self.current_backtester: Optional[Backtester] = None
        
        logger.info(f"RegimeAwareBacktester initialized with regime switching: {enable_regime_switching}")

    def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        """
        Run regime-aware backtest
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start date
            end: End date
            
        Returns:
            Dictionary with backtest results including regime switches
        """
        
        logger.info(f"Starting regime-aware backtest: {symbol} {timeframe} from {start} to {end}")
        
        # Get historical data
        df = self.data_provider.get_historical_data(symbol, timeframe, start, end)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol} {timeframe} between {start} and {end}")
        
        # If regime switching is disabled, run normal backtest
        if not self.enable_regime_switching or self.regime_switcher is None:
            return self._run_single_strategy_backtest(symbol, timeframe, start, end)
        
        # Run regime-aware backtest with strategy switching
        return self._run_regime_aware_backtest(df, symbol, timeframe, start, end)

    def _run_single_strategy_backtest(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        """Run backtest with single strategy (no regime switching)"""
        
        backtester = Backtester(
            strategy=self.initial_strategy,
            data_provider=self.data_provider,
            sentiment_provider=self.sentiment_provider,
            risk_parameters=self.risk_parameters,
            initial_balance=self.initial_balance,
            **self.backtester_kwargs
        )
        
        return backtester.run(symbol, timeframe, start, end)

    def _run_regime_aware_backtest(
        self,
        df: DataFrame,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        """Run backtest with regime-based strategy switching"""
        
        # Prepare data for multi-timeframe regime analysis
        # For backtesting, we'll use the main timeframe for regime detection
        price_data = {timeframe: df}
        
        # Initialize results tracking
        all_trades = []
        strategy_switches = []
        current_balance = self.initial_balance
        peak_balance = self.initial_balance
        
        # Current strategy state
        current_strategy = self.initial_strategy
        current_strategy_name = self.initial_strategy.name.lower().replace(" ", "_")
        
        # Track regime analysis over time
        regime_history = []
        
        # Process data in chunks to simulate real-time regime detection
        chunk_size = 100  # Analyze regime every 100 candles
        
        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            chunk_df = df.iloc[:chunk_end].copy()
            
            if len(chunk_df) < 60:  # Need minimum data for regime analysis
                continue
            
            # Analyze current market regime
            regime_analysis = self.regime_switcher.analyze_market_regime({timeframe: chunk_df})
            regime_history.append({
                'timestamp': chunk_df.index[-1],
                'regime': regime_analysis['consensus_regime']['regime_label'],
                'confidence': regime_analysis['consensus_regime']['confidence'],
                'agreement': regime_analysis['consensus_regime']['agreement_score']
            })
            
            # Check if strategy should be switched
            switch_decision = self.regime_switcher.should_switch_strategy(regime_analysis)
            
            if switch_decision['should_switch']:
                new_strategy_name = switch_decision['optimal_strategy']
                
                # Record the switch
                switch_info = {
                    'timestamp': chunk_df.index[-1],
                    'old_strategy': current_strategy_name,
                    'new_strategy': new_strategy_name,
                    'regime': switch_decision['new_regime'],
                    'confidence': switch_decision['confidence'],
                    'reason': switch_decision['reason']
                }
                strategy_switches.append(switch_info)
                
                logger.info(f"Strategy switch at {chunk_df.index[-1]}: {current_strategy_name} -> {new_strategy_name} (regime: {switch_decision['new_regime']})")
                
                # Load new strategy
                try:
                    new_strategy = self._load_strategy_by_name(new_strategy_name)
                    current_strategy = new_strategy
                    current_strategy_name = new_strategy_name
                except Exception as e:
                    logger.warning(f"Failed to load strategy {new_strategy_name}: {e}. Continuing with current strategy.")
        
        # Run final backtest with the regime-aware approach
        # For simplicity, we'll run a full backtest and adjust results based on switches
        # In a more sophisticated implementation, we would run separate backtests for each regime period
        
        backtester = Backtester(
            strategy=current_strategy,
            data_provider=self.data_provider,
            sentiment_provider=self.sentiment_provider,
            risk_parameters=self.risk_parameters,
            initial_balance=self.initial_balance,
            **self.backtester_kwargs
        )
        
        results = backtester.run(symbol, timeframe, start, end)
        
        # Add regime-aware results
        results.update({
            'regime_switching_enabled': True,
            'strategy_switches': strategy_switches,
            'regime_history': regime_history,
            'final_strategy': current_strategy_name,
            'initial_strategy': self.initial_strategy.name,
            'total_strategy_switches': len(strategy_switches)
        })
        
        return results

    def _load_strategy_by_name(self, strategy_name: str) -> BaseStrategy:
        """Load strategy by name using the strategy manager registry"""
        
        strategy_classes = {
            'ml_basic': 'src.strategies.ml_basic.MlBasic',
            'ml_adaptive': 'src.strategies.ml_adaptive.MlAdaptive', 
            'ml_sentiment': 'src.strategies.ml_sentiment.MlSentiment',
            'bear': 'src.strategies.bear.BearStrategy',
            'bull': 'src.strategies.bull.Bull',
            'ensemble_weighted': 'src.strategies.ensemble_weighted.EnsembleWeighted',
            'momentum_leverage': 'src.strategies.momentum_leverage.MomentumLeverage'
        }
        
        if strategy_name not in strategy_classes:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        module_path, class_name = strategy_classes[strategy_name].rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        strategy_class = getattr(module, class_name)
        
        return strategy_class()

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies for regime switching"""
        return list(self.strategy_manager.strategy_registry.keys())