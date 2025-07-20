"""
Test ML Adaptive Strategy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.ml_adaptive import MlAdaptive

class TestMlAdaptive:
    """Test ML Adaptive strategy functionality"""
    
    @pytest.fixture
    def strategy(self):
        """Create ML Adaptive strategy instance"""
        return MlAdaptive()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2020-01-01', end='2020-06-01', freq='H')
        np.random.seed(42)
        
        # Create realistic price data with trend and volatility
        base_price = 10000
        trend = np.linspace(0, 1000, len(dates))
        noise = np.random.normal(0, 100, len(dates))
        prices = base_price + trend + noise
        
        # Add a crash period (similar to COVID crash)
        crash_start = len(dates) // 3
        crash_end = crash_start + 96  # 4 days
        crash_magnitude = 0.35  # 35% drop
        prices[crash_start:crash_end] *= (1 - crash_magnitude)
        
        # Recovery period
        recovery_end = crash_end + 240  # 10 days
        recovery_prices = np.linspace(
            prices[crash_end-1], 
            prices[crash_start-1] * 0.9,  # Recover to 90% of pre-crash
            recovery_end - crash_end
        )
        prices[crash_end:recovery_end] = recovery_prices
        
        df = pd.DataFrame({
            'open': prices * np.random.uniform(0.995, 1.005, len(dates)),
            'high': prices * np.random.uniform(1.001, 1.02, len(dates)),
            'low': prices * np.random.uniform(0.98, 0.999, len(dates)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, len(dates))
        }, index=dates)
        
        return df
    
    def test_initialization(self, strategy):
        """Test strategy initialization"""
        assert strategy.name == "MlAdaptive"
        assert strategy.trading_pair == 'BTCUSDT'
        assert strategy.base_stop_loss_pct == 0.02
        assert strategy.base_take_profit_pct == 0.04
        assert strategy.base_position_size == 0.10
        assert strategy.max_daily_loss_pct == 0.05
        assert strategy.consecutive_losses == 0
        assert not strategy.in_crisis_mode
    
    def test_market_regime_detection(self, strategy, sample_data):
        """Test market regime detection"""
        # Calculate indicators
        df = strategy.calculate_indicators(sample_data)
        
        # Check that regime detection works
        assert 'market_regime' in df.columns
        regimes = df['market_regime'].unique()
        
        # Should detect multiple regimes in our test data
        assert 'normal' in regimes
        assert len(regimes) > 1  # Should detect at least normal and volatile/crisis
        
        # Check crisis detection during crash period
        crash_period = df.iloc[len(df)//3:len(df)//3 + 96]
        crisis_regimes = crash_period['market_regime'].value_counts()
        
        # Should detect crisis or volatile during crash
        assert ('crisis' in crisis_regimes or 'volatile' in crisis_regimes)
    
    def test_volatility_calculations(self, strategy, sample_data):
        """Test volatility metric calculations"""
        df = strategy.calculate_indicators(sample_data)
        
        # Check volatility columns exist
        assert 'volatility_20' in df.columns
        assert 'volatility_50' in df.columns
        assert 'atr_pct' in df.columns
        
        # Volatility should be higher during crash period
        normal_period = df.iloc[:len(df)//3]
        crash_period = df.iloc[len(df)//3:len(df)//3 + 96]
        
        normal_vol = normal_period['volatility_20'].mean()
        crash_vol = crash_period['volatility_20'].mean()
        
        assert crash_vol > normal_vol * 2  # Volatility should spike during crash
    
    def test_dynamic_stop_loss(self, strategy, sample_data):
        """Test dynamic stop loss calculation"""
        df = strategy.calculate_indicators(sample_data)
        
        # Test stop loss in different regimes
        normal_idx = df[df['market_regime'] == 'normal'].index[0]
        normal_idx_pos = df.index.get_loc(normal_idx)
        normal_sl = strategy._get_dynamic_stop_loss(df, normal_idx_pos, 'normal')
        
        # Find crisis period if exists
        crisis_indices = df[df['market_regime'] == 'crisis'].index
        if len(crisis_indices) > 0:
            crisis_idx = crisis_indices[0]
            crisis_idx_pos = df.index.get_loc(crisis_idx)
            crisis_sl = strategy._get_dynamic_stop_loss(df, crisis_idx_pos, 'crisis')
            
            # Crisis stop loss should be larger
            assert crisis_sl > normal_sl
            assert crisis_sl <= strategy.max_stop_loss_pct
    
    def test_position_sizing(self, strategy, sample_data):
        """Test adaptive position sizing"""
        df = strategy.calculate_indicators(sample_data)
        balance = 10000
        
        # Test position sizing in different market conditions
        positions = []
        regimes = []
        
        for i in range(len(df)):
            if i >= 120:  # After warm-up period
                pos_size = strategy.calculate_position_size(df, i, balance)
                positions.append(pos_size / balance)  # As percentage
                regimes.append(df['market_regime'].iloc[i])
        
        positions = pd.Series(positions, index=regimes)
        
        # Average position size should be lower in crisis/volatile markets
        if 'crisis' in positions.index:
            crisis_pos = positions[positions.index == 'crisis'].mean()
            normal_pos = positions[positions.index == 'normal'].mean()
            assert crisis_pos < normal_pos
    
    def test_entry_conditions_crisis(self, strategy, sample_data):
        """Test entry conditions during crisis"""
        df = strategy.calculate_indicators(sample_data)
        
        # Count entries in different regimes
        entries_by_regime = {'normal': 0, 'volatile': 0, 'crisis': 0}
        
        for i in range(150, len(df)):
            if strategy.check_entry_conditions(df, i):
                regime = df['market_regime'].iloc[i]
                if regime in entries_by_regime:
                    entries_by_regime[regime] += 1
        
        # Should have fewer entries in crisis mode
        if entries_by_regime['crisis'] > 0 and entries_by_regime['normal'] > 0:
            assert entries_by_regime['crisis'] < entries_by_regime['normal']
    
    def test_daily_loss_limit(self, strategy):
        """Test daily loss limit enforcement"""
        # Set up a daily loss
        test_date = datetime(2020, 3, 15).date()
        strategy.daily_losses[test_date] = -0.06  # 6% loss (exceeds limit)
        
        # Create minimal test data
        dates = pd.date_range(start='2020-03-15', periods=200, freq='H')
        df = pd.DataFrame({
            'close': 10000,
            'open': 10000,
            'high': 10100,
            'low': 9900,
            'volume': 1000
        }, index=dates)
        
        df = strategy.calculate_indicators(df)
        
        # Should not allow entry when daily loss limit exceeded
        assert not strategy.check_entry_conditions(df, 150)
    
    def test_consecutive_loss_tracking(self, strategy):
        """Test consecutive loss tracking"""
        # Set consecutive losses to limit
        strategy.consecutive_losses = 3
        
        # Create minimal test data
        dates = pd.date_range(start='2020-01-01', periods=200, freq='H')
        df = pd.DataFrame({
            'close': 10000,
            'open': 10000,
            'high': 10100,
            'low': 9900,
            'volume': 1000
        }, index=dates)
        
        df = strategy.calculate_indicators(df)
        
        # Should not allow entry when consecutive loss limit reached
        assert not strategy.check_entry_conditions(df, 150)
    
    def test_ml_predictions(self, strategy, sample_data):
        """Test ML prediction generation"""
        df = strategy.calculate_indicators(sample_data)
        
        # Check prediction columns exist
        assert 'onnx_pred' in df.columns
        assert 'prediction_confidence' in df.columns
        
        # Predictions should start after sequence length
        first_pred_idx = strategy.sequence_length
        assert pd.isna(df['onnx_pred'].iloc[:first_pred_idx]).all()
        assert not pd.isna(df['onnx_pred'].iloc[first_pred_idx:]).all()
        
        # Confidence should be calculated
        valid_confidence = df['prediction_confidence'].dropna()
        assert len(valid_confidence) > 0
        assert (valid_confidence >= 0).all()
    
    def test_parameter_export(self, strategy):
        """Test parameter export"""
        params = strategy.get_parameters()
        
        assert params['name'] == 'MlAdaptive'
        assert 'volatility_thresholds' in params
        assert params['base_position_size'] == 0.10
        assert params['max_daily_loss_pct'] == 0.05