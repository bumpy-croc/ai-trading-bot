"""
Unit tests for ensemble strategies
"""


import numpy as np
import pandas as pd
import pytest

from src.strategies.ensemble_weighted import EnsembleWeighted


class TestEnsembleWeighted:
    """Test the weighted ensemble strategy"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data
        base_price = 30000
        returns = np.random.normal(0, 0.01, len(dates))  # 1% hourly volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure high >= close >= low and high >= open >= low
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def test_ensemble_weighted_initialization(self):
        """Test that EnsembleWeighted initializes correctly"""
        strategy = EnsembleWeighted()
        
        assert strategy.name == "EnsembleWeighted"
        assert strategy.trading_pair == "BTCUSDT"
        assert len(strategy.strategies) >= 2  # Should have at least ML Basic and Adaptive
        assert "ml_basic" in strategy.strategies
        assert "ml_adaptive" in strategy.strategies
        
        # Check weights are normalized
        total_weight = sum(strategy.strategy_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_ensemble_weighted_indicators(self, sample_data):
        """Test indicator calculation for weighted ensemble"""
        strategy = EnsembleWeighted()
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_data)
        
        # Check that ensemble columns were added
        assert "ensemble_entry_score" in df_with_indicators.columns
        assert "ensemble_confidence" in df_with_indicators.columns
        assert "strategy_agreement" in df_with_indicators.columns
        assert "active_strategies" in df_with_indicators.columns
        
        # Check that strategy-specific columns were added
        assert any(col.startswith("ml_basic_") for col in df_with_indicators.columns)
        assert any(col.startswith("ml_adaptive_") for col in df_with_indicators.columns)
    
    def test_ensemble_weighted_entry_conditions(self, sample_data):
        """Test entry condition checking"""
        strategy = EnsembleWeighted()
        
        df_with_indicators = strategy.calculate_indicators(sample_data)
        
        # Test entry conditions at various points
        for i in [130, 140, 150]:  # Test after sufficient history
            result = strategy.check_entry_conditions(df_with_indicators, i)
            assert isinstance(result, (bool, np.bool_))
    
    def test_ensemble_weighted_position_sizing(self, sample_data):
        """Test position sizing calculation"""
        strategy = EnsembleWeighted()
        
        df_with_indicators = strategy.calculate_indicators(sample_data)
        balance = 10000.0
        
        # Test position sizing
        position_size = strategy.calculate_position_size(df_with_indicators, 130, balance)
        
        assert position_size >= 0
        assert position_size <= balance * strategy.MAX_POSITION_SIZE_RATIO
        assert position_size >= balance * strategy.MIN_POSITION_SIZE_RATIO or position_size == 0
    
    def test_ensemble_weighted_parameters(self):
        """Test parameter retrieval"""
        strategy = EnsembleWeighted()
        params = strategy.get_parameters()
        
        assert isinstance(params, dict)
        assert "name" in params
        assert "strategies" in params
        assert "current_weights" in params
        assert params["name"] == "EnsembleWeighted"


class TestEnsembleOptimized:
    """Test the optimized ensemble strategy features"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing optimized features"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
        np.random.seed(42)
        
        base_price = 30000
        returns = np.random.normal(0, 0.01, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def test_optimized_position_sizing(self, sample_data):
        """Test that optimized position sizing uses higher allocations"""
        strategy = EnsembleWeighted()
        
        # Check increased position size limits
        assert strategy.BASE_POSITION_SIZE == 0.50  # 50%
        assert strategy.MAX_POSITION_SIZE_RATIO == 0.80  # 80%
        assert strategy.MIN_POSITION_SIZE_RATIO == 0.20  # 20%
        
        df_with_indicators = strategy.calculate_indicators(sample_data)
        balance = 10000.0
        
        position_size = strategy.calculate_position_size(df_with_indicators, 130, balance)
        
        # Should allow larger positions
        assert position_size >= balance * 0.20  # At least 20%
        assert position_size <= balance * 0.80  # At most 80%
    
    def test_optimized_risk_parameters(self):
        """Test that risk parameters are optimized for higher returns"""
        strategy = EnsembleWeighted()
        
        # Check wider stops and targets
        assert strategy.STOP_LOSS_PCT == 0.06  # 6%
        assert strategy.TAKE_PROFIT_PCT == 0.20  # 20%
        
        risk_overrides = strategy.get_risk_overrides()
        
        # Check trailing stops are included
        assert "trailing_stop" in risk_overrides
        assert risk_overrides["trailing_stop"]["activation_threshold"] == 0.04
    
    def test_momentum_indicators(self, sample_data):
        """Test that momentum indicators are calculated"""
        strategy = EnsembleWeighted()
        
        df_with_indicators = strategy.calculate_indicators(sample_data)
        
        # Check momentum indicators exist
        momentum_cols = [
            "momentum_fast", "momentum_medium", "momentum_slow", "momentum_score",
            "volatility_fast", "volatility_slow", "volatility_ratio",
            "trend_strength_fast", "trend_strength_slow", "trend_alignment",
            "strong_breakout_up", "strong_breakout_down", "strong_bull", "strong_bear"
        ]
        
        for col in momentum_cols:
            assert col in df_with_indicators.columns
    
    def test_enhanced_strategy_components(self):
        """Test that ML strategies are included"""
        strategy = EnsembleWeighted()
        
        # Should include ML strategies (bull and bear were removed)
        assert "ml_basic" in strategy.strategies
        assert "ml_adaptive" in strategy.strategies
        assert len(strategy.strategies) >= 2  # ML Basic, ML Adaptive