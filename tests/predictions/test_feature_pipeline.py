"""
Tests for Feature Pipeline

This module contains integration tests for the FeaturePipeline class
and its interaction with feature extractors.
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import patch, MagicMock
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.prediction.features.sentiment import SentimentFeatureExtractor
from src.prediction.features.base import FeatureExtractor
from src.prediction.utils.caching import FeatureCache


class MockCustomExtractor(FeatureExtractor):
    """Mock custom extractor for testing."""
    
    def __init__(self):
        super().__init__("mock_custom")
        self._feature_names = ['custom_feature_1', 'custom_feature_2']
    
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract mock features."""
        df = data.copy()
        df['custom_feature_1'] = 1.0
        df['custom_feature_2'] = 2.0
        return df
    
    def get_feature_names(self):
        return self._feature_names.copy()


class TestFeaturePipeline:
    """Test cases for FeaturePipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=150, freq='1h')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 30000
        returns = np.random.normal(0, 0.015, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 100))
        
        spread_pct = np.random.uniform(0.001, 0.003, len(dates))
        volume = np.random.uniform(100, 500, len(dates))
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * (1 + s + np.random.uniform(0, 0.005)) for p, s in zip(prices, spread_pct)],
            'low': [p * (1 - s - np.random.uniform(0, 0.005)) for p, s in zip(prices, spread_pct)],
            'close': prices,
            'volume': volume
        }, index=dates)
    
    @pytest.fixture
    def pipeline_mvp(self):
        """Create MVP pipeline with only technical features."""
        return FeaturePipeline(
            enable_technical=True,
            enable_sentiment=False,
            enable_market_microstructure=False,
            use_cache=True
        )
    
    @pytest.fixture
    def pipeline_no_cache(self):
        """Create pipeline without caching."""
        return FeaturePipeline(
            enable_technical=True,
            enable_sentiment=False,
            enable_market_microstructure=False,
            use_cache=False
        )
    
    def test_initialization_mvp(self, pipeline_mvp):
        """Test MVP pipeline initialization."""
        assert pipeline_mvp.enable_technical is True
        assert pipeline_mvp.enable_sentiment is False
        assert pipeline_mvp.enable_market_microstructure is False
        assert pipeline_mvp.use_cache is True
        
        # Should have technical extractor only
        extractors = pipeline_mvp.get_extractor_names()
        assert 'technical' in extractors
        assert 'sentiment' not in extractors
        assert 'market' not in extractors
        assert len(extractors) == 1
    
    def test_initialization_with_sentiment(self):
        """Test pipeline initialization with sentiment enabled."""
        pipeline = FeaturePipeline(
            enable_technical=True,
            enable_sentiment=True,
            enable_market_microstructure=False
        )
        
        extractors = pipeline.get_extractor_names()
        assert 'technical' in extractors
        assert 'sentiment' in extractors
        assert 'market' not in extractors
        assert len(extractors) == 2
    
    def test_initialization_with_custom_extractors(self):
        """Test pipeline initialization with custom extractors."""
        custom_extractor = MockCustomExtractor()
        pipeline = FeaturePipeline(
            enable_technical=True,
            enable_sentiment=False,
            enable_market_microstructure=False,
            custom_extractors=[custom_extractor]
        )
        
        extractors = pipeline.get_extractor_names()
        assert 'technical' in extractors
        assert 'mock_custom' in extractors
        assert len(extractors) == 2
    
    def test_transform_basic(self, pipeline_mvp, sample_data):
        """Test basic feature transformation."""
        result = pipeline_mvp.transform(sample_data)
        
        # Check that original data is preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], sample_data[col])
        
        # Check that features are added
        feature_names = pipeline_mvp.get_feature_names()
        for feature in feature_names:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check data quality
        assert not result.empty
        assert len(result) == len(sample_data)
    
    def test_transform_with_caching(self, pipeline_mvp, sample_data):
        """Test transformation with caching enabled."""
        # Clear cache first
        pipeline_mvp.clear_cache()
        
        # First transform - should miss cache
        result1 = pipeline_mvp.transform(sample_data)
        stats1 = pipeline_mvp.get_performance_stats()
        
        # Second transform with same data - should hit cache
        result2 = pipeline_mvp.transform(sample_data)
        stats2 = pipeline_mvp.get_performance_stats()
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Cache hit rate should improve
        assert stats2['cache_hits'] > stats1['cache_hits']
    
    def test_transform_without_caching(self, pipeline_no_cache, sample_data):
        """Test transformation without caching."""
        result = pipeline_no_cache.transform(sample_data)
        
        # Should work without cache
        assert not result.empty
        assert len(result) == len(sample_data)
        
        # Cache stats should be None
        cache_stats = pipeline_no_cache.get_cache_stats()
        assert cache_stats is None
    
    def test_transform_with_custom_extractor(self, sample_data):
        """Test transformation with custom extractor."""
        custom_extractor = MockCustomExtractor()
        pipeline = FeaturePipeline(
            enable_technical=True,
            enable_sentiment=False,
            enable_market_microstructure=False,
            custom_extractors=[custom_extractor]
        )
        
        result = pipeline.transform(sample_data)
        
        # Check that custom features are added
        assert 'custom_feature_1' in result.columns
        assert 'custom_feature_2' in result.columns
        assert (result['custom_feature_1'] == 1.0).all()
        assert (result['custom_feature_2'] == 2.0).all()
    
    def test_invalid_input_handling(self, pipeline_mvp):
        """Test handling of invalid input data."""
        # Test None input
        with pytest.raises(ValueError, match="Input data is empty or None"):
            pipeline_mvp.transform(None)
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Input data is empty or None"):
            pipeline_mvp.transform(empty_df)
    
    def test_missing_value_handling(self, pipeline_mvp, sample_data):
        """Test handling of missing values in output."""
        # Add some NaN values to input
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.iloc[50:60, 1] = np.nan  # Add NaN to 'high' column
        
        result = pipeline_mvp.transform(sample_data_with_nan)
        
        # Pipeline should handle missing values
        assert not result.empty
        # Some NaN might remain due to indicator calculations, but shouldn't be excessive
        nan_ratio = result.isna().sum().sum() / result.size
        assert nan_ratio < 0.3  # Less than 30% NaN
    
    def test_performance_tracking(self, pipeline_mvp, sample_data):
        """Test performance statistics tracking."""
        # Clear cache and stats
        pipeline_mvp.clear_cache()
        pipeline_mvp.reset_performance_stats()
        
        # Perform transformation
        pipeline_mvp.transform(sample_data)
        
        stats = pipeline_mvp.get_performance_stats()
        
        # Check that stats are tracked
        assert stats['total_transforms'] == 1
        assert stats['total_time'] > 0
        assert 'extractor_times' in stats
        assert 'extractor_avg_times' in stats
        
        # Technical extractor should have timing data
        assert 'technical' in stats['extractor_avg_times']
        assert stats['extractor_avg_times']['technical'] > 0
    
    def test_get_feature_names(self, pipeline_mvp):
        """Test getting feature names from pipeline."""
        feature_names = pipeline_mvp.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Should include technical features
        technical_features = ['rsi', 'atr', 'ma_20', 'close_normalized']
        for feature in technical_features:
            assert feature in feature_names
    
    def test_get_config(self, pipeline_mvp):
        """Test pipeline configuration retrieval."""
        config = pipeline_mvp.get_config()
        
        assert config['enable_technical'] is True
        assert config['enable_sentiment'] is False
        assert config['enable_market_microstructure'] is False
        assert config['use_cache'] is True
        assert 'extractors' in config
        assert 'total_features' in config
        assert config['total_features'] > 0
    
    def test_cache_management(self, pipeline_mvp, sample_data):
        """Test cache management functionality."""
        # Clear cache
        pipeline_mvp.clear_cache()
        
        # Transform and check cache stats
        pipeline_mvp.transform(sample_data)
        cache_stats = pipeline_mvp.get_cache_stats()
        
        assert cache_stats is not None
        assert 'total_entries' in cache_stats
        assert 'hit_rate' in cache_stats
        
        # Clear cache again
        pipeline_mvp.clear_cache()
        cache_stats_after_clear = pipeline_mvp.get_cache_stats()
        assert cache_stats_after_clear['total_entries'] == 0
    
    def test_feature_validation(self, pipeline_mvp, sample_data):
        """Test feature validation functionality."""
        result = pipeline_mvp.transform(sample_data)
        validation_results = pipeline_mvp.validate_features(result)
        
        assert isinstance(validation_results, dict)
        assert 'technical' in validation_results
        
        # Most features should be valid
        technical_validation = validation_results['technical']
        valid_features = sum(technical_validation.values())
        total_features = len(technical_validation)
        assert valid_features / total_features > 0.8
    
    def test_add_remove_extractors(self, pipeline_mvp):
        """Test adding and removing extractors dynamically."""
        initial_extractors = pipeline_mvp.get_extractor_names()
        
        # Add custom extractor
        custom_extractor = MockCustomExtractor()
        pipeline_mvp.add_extractor(custom_extractor)
        
        after_add = pipeline_mvp.get_extractor_names()
        assert len(after_add) == len(initial_extractors) + 1
        assert 'mock_custom' in after_add
        
        # Remove extractor
        pipeline_mvp.remove_extractor('mock_custom')
        
        after_remove = pipeline_mvp.get_extractor_names()
        assert len(after_remove) == len(initial_extractors)
        assert 'mock_custom' not in after_remove
    
    def test_cache_override(self, pipeline_mvp, sample_data):
        """Test cache override functionality."""
        # Transform with cache enabled (default)
        result1 = pipeline_mvp.transform(sample_data, use_cache=True)
        
        # Transform with cache disabled (override)
        result2 = pipeline_mvp.transform(sample_data, use_cache=False)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_error_handling_in_extraction(self, sample_data):
        """Test error handling when feature extraction fails."""
        # Create a pipeline with a mock extractor that raises an error
        class FailingExtractor(FeatureExtractor):
            def __init__(self):
                super().__init__("failing")
                self._feature_names = ['failing_feature']
            
            def extract(self, data):
                raise RuntimeError("Extraction failed")
            
            def get_feature_names(self):
                return self._feature_names
        
        failing_extractor = FailingExtractor()
        pipeline = FeaturePipeline(
            enable_technical=False,
            enable_sentiment=False,
            enable_market_microstructure=False,
            custom_extractors=[failing_extractor]
        )
        
        with pytest.raises(RuntimeError, match="Feature pipeline transformation failed"):
            pipeline.transform(sample_data)
    
    def test_output_validation(self, sample_data):
        """Test output validation catches invalid results."""
        # Create a pipeline with an extractor that produces invalid output
        class InvalidOutputExtractor(FeatureExtractor):
            def __init__(self):
                super().__init__("invalid")
                self._feature_names = ['inf_feature']
            
            def extract(self, data):
                df = data.copy()
                df['inf_feature'] = np.inf  # Add infinite values
                return df
            
            def get_feature_names(self):
                return self._feature_names
        
        invalid_extractor = InvalidOutputExtractor()
        pipeline = FeaturePipeline(
            enable_technical=False,
            enable_sentiment=False,
            enable_market_microstructure=False,
            custom_extractors=[invalid_extractor]
        )
        
        with pytest.raises(RuntimeError, match="infinite values"):
            pipeline.transform(sample_data)


class TestFeaturePipelineIntegration:
    """Integration tests for feature pipeline with real data."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic crypto market data."""
        dates = pd.date_range('2023-01-01', periods=500, freq='1h')
        np.random.seed(123)
        
        # Generate realistic BTC-like price movements
        base_price = 25000
        volatility = 0.02
        trend = 0.0001
        
        prices = [base_price]
        for i in range(1, len(dates)):
            # Add trend and random walk
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000))  # Floor at $1000
        
        # Generate realistic OHLC from close prices
        data = []
        for i, price in enumerate(prices):
            if i == 0:
                open_price = price
            else:
                open_price = prices[i-1]
            
            # High/low based on volatility
            daily_range = price * np.random.uniform(0.005, 0.03)
            high = price + np.random.uniform(0, daily_range)
            low = price - np.random.uniform(0, daily_range)
            
            # Volume correlated with price movement
            price_change = abs(price - open_price) / open_price if open_price > 0 else 0
            base_volume = 100
            volume = base_volume + (price_change * 1000) + np.random.uniform(0, 50)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    def test_end_to_end_mvp_pipeline(self, realistic_data):
        """Test complete end-to-end MVP pipeline functionality."""
        pipeline = FeaturePipeline(
            enable_technical=True,
            enable_sentiment=False,
            enable_market_microstructure=False,
            use_cache=True
        )
        
        # Transform the data
        result = pipeline.transform(realistic_data)
        
        # Comprehensive validation
        assert len(result) == len(realistic_data)
        assert not result.empty
        
        # Check feature completeness
        feature_names = pipeline.get_feature_names()
        for feature in feature_names:
            assert feature in result.columns
        
        # Check data quality
        total_values = result.size
        nan_values = result.isna().sum().sum()
        nan_ratio = nan_values / total_values
        assert nan_ratio < 0.2  # Less than 20% NaN values
        
        # Check specific feature ranges
        if 'rsi' in result.columns:
            rsi_data = result['rsi'].dropna()
            if len(rsi_data) > 0:
                assert rsi_data.min() >= 0
                assert rsi_data.max() <= 100
        
        # Check normalized features
        normalized_features = ['close_normalized', 'volume_normalized']
        for feature in normalized_features:
            if feature in result.columns:
                feature_data = result[feature].dropna()
                if len(feature_data) > 0:
                    assert feature_data.min() >= 0
                    assert feature_data.max() <= 1
        
        # Performance validation
        stats = pipeline.get_performance_stats()
        assert stats['total_transforms'] == 1
        assert stats['total_time'] > 0
        assert stats['avg_time_per_transform'] > 0
    
    def test_pipeline_performance_benchmark(self, realistic_data):
        """Test pipeline performance meets requirements."""
        pipeline = FeaturePipeline(
            enable_technical=True,
            enable_sentiment=False,
            enable_market_microstructure=False,
            use_cache=True
        )
        
        # First run (cache miss)
        start_time = time.time()
        result1 = pipeline.transform(realistic_data)
        first_run_time = time.time() - start_time
        
        # Second run (cache hit)
        start_time = time.time()
        result2 = pipeline.transform(realistic_data)
        second_run_time = time.time() - start_time
        
        # Validate results are identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Performance requirements (adjust based on actual requirements)
        assert first_run_time < 10.0  # Should complete within 10 seconds
        assert second_run_time < first_run_time  # Cache should improve performance
        
        # Cache effectiveness
        stats = pipeline.get_performance_stats()
        assert stats['cache_hit_rate'] > 0  # Should have some cache hits
    
    def test_pipeline_with_real_strategy_data(self, realistic_data):
        """Test pipeline produces features compatible with existing strategies."""
        pipeline = FeaturePipeline(
            enable_technical=True,
            enable_sentiment=False,
            enable_market_microstructure=False,
            use_cache=True
        )
        
        result = pipeline.transform(realistic_data)
        
        # Features required by MlBasic strategy
        ml_basic_features = ['close_normalized', 'volume_normalized', 'high_normalized', 
                           'low_normalized', 'open_normalized']
        for feature in ml_basic_features:
            assert feature in result.columns
        
        # Features required by MlAdaptive strategy
        ml_adaptive_features = ['rsi', 'atr', 'atr_pct', 'ma_20', 'ma_50', 'ma_200',
                              'bb_upper', 'bb_lower', 'bb_middle', 'macd', 'macd_signal',
                              'macd_hist', 'returns', 'volatility_20', 'volatility_50',
                              'trend_strength', 'trend_direction']
        for feature in ml_adaptive_features:
            assert feature in result.columns
        
        # Verify feature data quality for strategy compatibility
        # RSI should be reasonable
        rsi_data = result['rsi'].dropna()
        if len(rsi_data) > 10:
            assert 0 <= rsi_data.median() <= 100
        
        # Trend direction should only be -1 or 1
        trend_data = result['trend_direction'].dropna()
        if len(trend_data) > 10:
            unique_values = set(trend_data.unique())
            assert unique_values.issubset({-1, 1})
        
        # Volatility should be non-negative
        vol_data = result['volatility_20'].dropna()
        if len(vol_data) > 10:
            assert (vol_data >= 0).all()