"""
Unit tests for ensemble strategies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.ensemble_weighted import EnsembleWeighted
from src.strategies.ensemble_adaptive import EnsembleAdaptive


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


class TestEnsembleAdaptive:
    """Test the adaptive ensemble strategy"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
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
    
    def test_ensemble_adaptive_initialization(self):
        """Test that EnsembleAdaptive initializes correctly"""
        strategy = EnsembleAdaptive()
        
        assert strategy.name == "EnsembleAdaptive"
        assert strategy.trading_pair == "BTCUSDT"
        assert len(strategy.strategies) >= 2
        
        # Check that regime detector is initialized
        assert strategy.regime_detector is not None
    
    def test_ensemble_adaptive_indicators(self, sample_data):
        """Test indicator calculation for adaptive ensemble"""
        strategy = EnsembleAdaptive()
        
        df_with_indicators = strategy.calculate_indicators(sample_data)
        
        # Check ensemble-specific columns
        expected_columns = [
            "ensemble_signal_strength",
            "ensemble_confidence", 
            "strategy_consensus",
            "regime_adjusted_weights"
        ]
        
        for col in expected_columns:
            assert col in df_with_indicators.columns
        
        # Check regime columns
        regime_columns = ["trend_label", "vol_label", "regime_confidence"]
        for col in regime_columns:
            assert col in df_with_indicators.columns
    
    def test_ensemble_adaptive_regime_weights(self, sample_data):
        """Test regime-adjusted weight calculation"""
        strategy = EnsembleAdaptive()
        
        df_with_indicators = strategy.calculate_indicators(sample_data)
        
        # Test regime weight adjustment
        regime_weights = strategy._get_regime_adjusted_weights(df_with_indicators, 130)
        
        assert isinstance(regime_weights, dict)
        assert len(regime_weights) > 0
        
        # Weights should be normalized
        total_weight = sum(regime_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_ensemble_adaptive_risk_overrides(self):
        """Test risk management overrides"""
        strategy = EnsembleAdaptive()
        
        risk_overrides = strategy.get_risk_overrides()
        
        assert isinstance(risk_overrides, dict)
        assert "dynamic_risk" in risk_overrides
        assert "partial_operations" in risk_overrides
        assert "trailing_stop" in risk_overrides
        
        # Check specific risk parameters
        assert "drawdown_thresholds" in risk_overrides["dynamic_risk"]
        assert "exit_targets" in risk_overrides["partial_operations"]
        assert "activation_threshold" in risk_overrides["trailing_stop"]


class TestEnsembleComparison:
    """Compare ensemble strategies"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for comparison"""
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
    
    def test_both_ensembles_work(self, sample_data):
        """Test that both ensemble strategies can process the same data"""
        weighted = EnsembleWeighted()
        adaptive = EnsembleAdaptive()
        
        # Both should be able to calculate indicators
        df_weighted = weighted.calculate_indicators(sample_data)
        df_adaptive = adaptive.calculate_indicators(sample_data)
        
        # Both should have basic OHLCV columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in df_weighted.columns
            assert col in df_adaptive.columns
        
        # Both should be able to check entry conditions
        weighted_entry = weighted.check_entry_conditions(df_weighted, 130)
        adaptive_entry = adaptive.check_entry_conditions(df_adaptive, 130)
        
        assert isinstance(weighted_entry, (bool, np.bool_))
        assert isinstance(adaptive_entry, (bool, np.bool_))
    
    def test_parameter_compatibility(self):
        """Test that both strategies return compatible parameter structures"""
        weighted = EnsembleWeighted()
        adaptive = EnsembleAdaptive()
        
        weighted_params = weighted.get_parameters()
        adaptive_params = adaptive.get_parameters()
        
        # Both should have basic parameter structure
        for params in [weighted_params, adaptive_params]:
            assert isinstance(params, dict)
            assert "name" in params
            assert "stop_loss_pct" in params or "STOP_LOSS_PCT" in str(params)
            assert "take_profit_pct" in params or "TAKE_PROFIT_PCT" in str(params)