"""
Tests for Feature Extractors

This module contains unit tests for all feature extractors in the prediction engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.prediction.features.sentiment import SentimentFeatureExtractor
from src.prediction.features.market import MarketFeatureExtractor
from src.prediction.features.schemas import TECHNICAL_FEATURES_SCHEMA


class TestTechnicalFeatureExtractor:
    """Test cases for TechnicalFeatureExtractor."""
    
    # Test configuration constants
    VALIDATION_THRESHOLD = 0.8  # At least 80% should be valid
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='1h')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 30000
        price_changes = np.random.normal(0, 0.02, len(dates))
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # Ensure positive prices
        
        # Create OHLCV data
        high_offset = np.random.uniform(0.001, 0.02, len(dates))
        low_offset = np.random.uniform(-0.02, -0.001, len(dates))
        volume = np.random.uniform(100, 1000, len(dates))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + h) for p, h in zip(prices, high_offset)],
            'low': [p * (1 + l) for p, l in zip(prices, low_offset)],
            'close': prices,
            'volume': volume
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def extractor(self):
        """Create a TechnicalFeatureExtractor instance."""
        return TechnicalFeatureExtractor()
    
    def test_initialization(self, extractor):
        """Test proper initialization of the extractor."""
        assert extractor.name == "technical"
        assert extractor.sequence_length == 120
        assert extractor.normalization_window == 120
        assert extractor.rsi_period == 14
        assert extractor.atr_period == 14
        assert len(extractor.get_feature_names()) > 0
    
    def test_extract_basic_functionality(self, extractor, sample_ohlcv_data):
        """Test basic feature extraction functionality."""
        result = extractor.extract(sample_ohlcv_data)
        
        # Check that original data is preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], sample_ohlcv_data[col])
        
        # Check that technical indicators are added
        expected_features = extractor.get_feature_names()
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_normalized_price_features(self, extractor, sample_ohlcv_data):
        """Test that normalized price features are correctly calculated."""
        result = extractor.extract(sample_ohlcv_data)
        
        normalized_features = extractor.get_normalized_features()
        for feature in normalized_features:
            assert feature in result.columns
            
            # Check that normalized values are between 0 and 1 (allowing for some NaN)
            feature_data = result[feature].dropna()
            if len(feature_data) > 0:
                assert feature_data.min() >= 0, f"{feature} has values < 0"
                assert feature_data.max() <= 1, f"{feature} has values > 1"
    
    def test_technical_indicators(self, extractor, sample_ohlcv_data):
        """Test that technical indicators are calculated correctly."""
        result = extractor.extract(sample_ohlcv_data)
        
        # Test RSI
        assert 'rsi' in result.columns
        rsi_data = result['rsi'].dropna()
        if len(rsi_data) > 0:
            assert rsi_data.min() >= 0
            assert rsi_data.max() <= 100
        
        # Test ATR
        assert 'atr' in result.columns
        assert 'atr_pct' in result.columns
        atr_data = result['atr'].dropna()
        if len(atr_data) > 0:
            assert (atr_data >= 0).all()
        
        # Test Moving Averages
        for period in extractor.ma_periods:
            ma_col = f'ma_{period}'
            assert ma_col in result.columns
        
        # Test Bollinger Bands
        for band in ['bb_upper', 'bb_middle', 'bb_lower']:
            assert band in result.columns
        
        # Test MACD
        for macd_col in ['macd', 'macd_signal', 'macd_hist']:
            assert macd_col in result.columns
    
    def test_derived_features(self, extractor, sample_ohlcv_data):
        """Test that derived features are calculated correctly."""
        result = extractor.extract(sample_ohlcv_data)
        
        # Test returns
        assert 'returns' in result.columns
        
        # Test volatility
        assert 'volatility_20' in result.columns
        assert 'volatility_50' in result.columns
        volatility_data = result['volatility_20'].dropna()
        if len(volatility_data) > 0:
            assert (volatility_data >= 0).all()
        
        # Test trend features
        assert 'trend_strength' in result.columns
        assert 'trend_direction' in result.columns
        trend_direction_data = result['trend_direction'].dropna()
        if len(trend_direction_data) > 0:
            assert set(trend_direction_data.unique()).issubset({-1, 1})
    
    def test_invalid_input_handling(self, extractor):
        """Test handling of invalid input data."""
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Invalid input data"):
            extractor.extract(empty_df)
        
        # Test missing required columns
        invalid_df = pd.DataFrame({'price': [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid input data"):
            extractor.extract(invalid_df)
    
    def test_feature_validation(self, extractor, sample_ohlcv_data):
        """Test feature validation functionality."""
        result = extractor.extract(sample_ohlcv_data)
        validation_results = extractor.validate_features(result)
        
        # Most features should be valid (allowing for some NaN due to window requirements)
        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        assert valid_count / total_count > self.VALIDATION_THRESHOLD  # At least 80% should be valid
    
    def test_configuration(self, extractor):
        """Test configuration retrieval."""
        config = extractor.get_config()
        
        assert config['name'] == 'technical'
        assert config['type'] == 'TechnicalFeatureExtractor'
        assert 'sequence_length' in config
        assert 'rsi_period' in config
        assert 'ma_periods' in config
    
    def test_feature_importance_weights(self, extractor):
        """Test feature importance weights."""
        weights = extractor.get_feature_importance_weights()
        
        # All weights should be between 0 and 1
        for feature, weight in weights.items():
            assert 0 <= weight <= 1, f"Invalid weight for {feature}: {weight}"
        
        # Normalized features should have highest importance
        normalized_features = extractor.get_normalized_features()
        for feature in normalized_features:
            if feature in weights:
                assert weights[feature] == 1.0


class TestSentimentFeatureExtractor:
    """Test cases for SentimentFeatureExtractor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        return pd.DataFrame({
            'open': np.random.uniform(29000, 31000, 100),
            'high': np.random.uniform(30000, 32000, 100),
            'low': np.random.uniform(28000, 30000, 100),
            'close': np.random.uniform(29000, 31000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
    
    def test_initialization_disabled(self):
        """Test initialization with sentiment disabled (MVP mode)."""
        extractor = SentimentFeatureExtractor(enabled=False)
        assert extractor.name == "sentiment"
        assert not extractor.enabled
        assert len(extractor.get_feature_names()) > 0
    
    def test_extract_neutral_values(self, sample_data):
        """Test extraction returns neutral values when disabled."""
        extractor = SentimentFeatureExtractor(enabled=False)
        result = extractor.extract(sample_data)
        
        # Check that sentiment features are added with neutral values
        feature_names = extractor.get_feature_names()
        for feature in feature_names:
            assert feature in result.columns
            
            # Check for reasonable neutral values
            feature_data = result[feature].dropna()
            if len(feature_data) > 0:
                if 'primary' in feature:
                    assert (feature_data == 0.5).all()
                elif 'momentum' in feature:
                    assert (feature_data == 0.0).all()
                elif 'confidence' in feature:
                    assert (feature_data == 0.7).all()
    
    def test_configuration(self):
        """Test configuration retrieval."""
        extractor = SentimentFeatureExtractor(enabled=False)
        config = extractor.get_config()
        
        assert config['name'] == 'sentiment'
        assert config['enabled'] is False
        assert config['mvp_mode'] is True


class TestMarketFeatureExtractor:
    """Test cases for MarketFeatureExtractor."""
    
    def test_initialization_disabled(self):
        """Test initialization with market features disabled (MVP mode)."""
        extractor = MarketFeatureExtractor(enabled=False)
        assert extractor.name == "market_microstructure"
        assert not extractor.enabled
        assert len(extractor.get_feature_names()) == 0  # No features when disabled
    
    def test_extract_raises_error_when_disabled(self):
        """Test that extraction raises error when disabled."""
        extractor = MarketFeatureExtractor(enabled=False)
        sample_data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1.1, 2.1, 3.1],
            'low': [0.9, 1.9, 2.9],
            'close': [1, 2, 3],
            'volume': [100, 200, 300]
        })
        
        with pytest.raises(RuntimeError, match="Market microstructure extraction is disabled"):
            extractor.extract(sample_data)
    
    def test_configuration(self):
        """Test configuration retrieval."""
        extractor = MarketFeatureExtractor(enabled=False)
        config = extractor.get_config()
        
        assert config['name'] == 'market_microstructure'
        assert config['enabled'] is False
        assert config['mvp_mode'] is True


class TestFeatureSchemas:
    """Test cases for feature schemas."""
    
    def test_technical_features_schema(self):
        """Test technical features schema structure."""
        schema = TECHNICAL_FEATURES_SCHEMA
        
        assert schema.name == "technical_features_v1"
        assert schema.version == "1.0.0"
        assert len(schema.features) > 0
        assert schema.sequence_length == 120
        
        # Check that required features exist
        feature_names = schema.get_feature_names()
        required_features = schema.get_required_features()
        
        # All required features should be in the feature list
        for required in required_features:
            assert required in feature_names
        
        # Check specific expected features
        expected_features = [
            'close_normalized', 'rsi', 'atr', 'ma_20', 'ma_50', 'ma_200',
            'bb_upper', 'bb_lower', 'macd', 'returns', 'volatility_20'
        ]
        for feature in expected_features:
            assert feature in feature_names
    
    def test_feature_definition_validation(self):
        """Test feature definition validation."""
        schema = TECHNICAL_FEATURES_SCHEMA
        
        for feature_def in schema.features:
            # Check that all features have required attributes
            assert hasattr(feature_def, 'name')
            assert hasattr(feature_def, 'feature_type')
            assert hasattr(feature_def, 'description')
            assert feature_def.name is not None
            assert len(feature_def.description) > 0
            
            # Check value constraints
            if feature_def.min_value is not None:
                assert isinstance(feature_def.min_value, (int, float))
            if feature_def.max_value is not None:
                assert isinstance(feature_def.max_value, (int, float))
            if feature_def.default_value is not None:
                assert isinstance(feature_def.default_value, (int, float))


@pytest.fixture
def sample_market_data():
    """Create sample market data for integration tests."""
    dates = pd.date_range('2023-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # Generate realistic crypto price data
    base_price = 30000
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 100))
    
    # Add realistic spread and volume
    spread_pct = np.random.uniform(0.001, 0.005, len(dates))
    volume = np.random.uniform(50, 500, len(dates))
    
    return pd.DataFrame({
        'open': prices,
        'high': [p * (1 + s + np.random.uniform(0, 0.01)) for p, s in zip(prices, spread_pct)],
        'low': [p * (1 - s - np.random.uniform(0, 0.01)) for p, s in zip(prices, spread_pct)],
        'close': prices,
        'volume': volume
    }, index=dates)