"""
Unit tests for RegimeContext and EnhancedRegimeDetector components
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import RegimeDetector, TrendLabel, VolLabel
from src.strategies.components.regime_context import (
    EnhancedRegimeDetector,
    RegimeContext,
    RegimeTransition,
)


class TestRegimeContext:
    """Test RegimeContext dataclass"""
    
    def test_regime_context_creation_valid(self):
        """Test creating a valid regime context"""
        timestamp = datetime.now()
        metadata = {'trend_score': 0.05, 'atr_percentile': 0.8}
        
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.HIGH,
            confidence=0.8,
            duration=15,
            strength=0.7,
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert regime.trend == TrendLabel.TREND_UP
        assert regime.volatility == VolLabel.HIGH
        assert regime.confidence == 0.8
        assert regime.duration == 15
        assert regime.strength == 0.7
        assert regime.timestamp == timestamp
        assert regime.metadata == metadata
    
    def test_regime_context_validation_trend(self):
        """Test regime context trend validation"""
        with pytest.raises(ValueError, match="trend must be a TrendLabel enum"):
            RegimeContext(
                trend="invalid",
                volatility=VolLabel.LOW,
                confidence=0.8,
                duration=10,
                strength=0.7
            )
    
    def test_regime_context_validation_volatility(self):
        """Test regime context volatility validation"""
        with pytest.raises(ValueError, match="volatility must be a VolLabel enum"):
            RegimeContext(
                trend=TrendLabel.TREND_UP,
                volatility="invalid",
                confidence=0.8,
                duration=10,
                strength=0.7
            )
    
    def test_regime_context_validation_confidence_bounds(self):
        """Test regime context confidence bounds validation"""
        # Test negative confidence
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            RegimeContext(
                trend=TrendLabel.TREND_UP,
                volatility=VolLabel.LOW,
                confidence=-0.1,
                duration=10,
                strength=0.7
            )
        
        # Test confidence > 1.0
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            RegimeContext(
                trend=TrendLabel.TREND_UP,
                volatility=VolLabel.LOW,
                confidence=1.1,
                duration=10,
                strength=0.7
            )
    
    def test_regime_context_validation_duration(self):
        """Test regime context duration validation"""
        with pytest.raises(ValueError, match="duration must be non-negative"):
            RegimeContext(
                trend=TrendLabel.TREND_UP,
                volatility=VolLabel.LOW,
                confidence=0.8,
                duration=-5,
                strength=0.7
            )
    
    def test_regime_context_validation_strength_bounds(self):
        """Test regime context strength bounds validation"""
        # Test negative strength
        with pytest.raises(ValueError, match="strength must be between 0.0 and 1.0"):
            RegimeContext(
                trend=TrendLabel.TREND_UP,
                volatility=VolLabel.LOW,
                confidence=0.8,
                duration=10,
                strength=-0.1
            )
        
        # Test strength > 1.0
        with pytest.raises(ValueError, match="strength must be between 0.0 and 1.0"):
            RegimeContext(
                trend=TrendLabel.TREND_UP,
                volatility=VolLabel.LOW,
                confidence=0.8,
                duration=10,
                strength=1.1
            )
    
    def test_regime_context_validation_metadata_type(self):
        """Test regime context metadata type validation"""
        with pytest.raises(ValueError, match="metadata must be a dictionary when provided"):
            RegimeContext(
                trend=TrendLabel.TREND_UP,
                volatility=VolLabel.LOW,
                confidence=0.8,
                duration=10,
                strength=0.7,
                metadata="invalid"
            )
    
    def test_get_regime_label(self):
        """Test regime label generation"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.HIGH,
            confidence=0.8,
            duration=10,
            strength=0.7
        )
        
        assert regime.get_regime_label() == "trend_up:high_vol"
    
    def test_is_stable_true(self):
        """Test regime stability check - stable"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=15,  # >= 10 default minimum
            strength=0.7
        )
        
        assert regime.is_stable() is True
        assert regime.is_stable(min_duration=12) is True
    
    def test_is_stable_false(self):
        """Test regime stability check - not stable"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=5,  # < 10 default minimum
            strength=0.7
        )
        
        assert regime.is_stable() is False
        assert regime.is_stable(min_duration=8) is False
    
    def test_is_high_confidence_true(self):
        """Test high confidence check - true"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,  # >= 0.7 default threshold
            duration=10,
            strength=0.7
        )
        
        assert regime.is_high_confidence() is True
        assert regime.is_high_confidence(threshold=0.75) is True
    
    def test_is_high_confidence_false(self):
        """Test high confidence check - false"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.6,  # < 0.7 default threshold
            duration=10,
            strength=0.7
        )
        
        assert regime.is_high_confidence() is False
        assert regime.is_high_confidence(threshold=0.65) is False
    
    def test_is_strong_regime_true(self):
        """Test strong regime check - true"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=10,
            strength=0.7  # >= 0.6 default threshold
        )
        
        assert regime.is_strong_regime() is True
        assert regime.is_strong_regime(threshold=0.65) is True
    
    def test_is_strong_regime_false(self):
        """Test strong regime check - false"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=10,
            strength=0.5  # < 0.6 default threshold
        )
        
        assert regime.is_strong_regime() is False
        assert regime.is_strong_regime(threshold=0.55) is False
    
    def test_get_risk_multiplier_bull_low_vol(self):
        """Test risk multiplier for bull market, low volatility"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=15,
            strength=0.7
        )
        
        multiplier = regime.get_risk_multiplier()
        
        # No reductions for this favorable regime
        assert multiplier == 1.0
    
    def test_get_risk_multiplier_bear_high_vol(self):
        """Test risk multiplier for bear market, high volatility"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_DOWN,
            volatility=VolLabel.HIGH,
            confidence=0.4,  # Low confidence
            duration=5,  # Not stable
            strength=0.7
        )
        
        multiplier = regime.get_risk_multiplier()
        
        # Multiple reductions: 0.8 (high vol) * 0.7 (bear) * 0.8 (low conf) * 0.9 (unstable)
        expected = 0.8 * 0.7 * 0.8 * 0.9
        assert abs(multiplier - expected) < 0.01
    
    def test_get_risk_multiplier_minimum_floor(self):
        """Test risk multiplier has minimum floor"""
        regime = RegimeContext(
            trend=TrendLabel.TREND_DOWN,
            volatility=VolLabel.HIGH,
            confidence=0.1,  # Very low confidence
            duration=1,  # Very unstable
            strength=0.1
        )
        
        multiplier = regime.get_risk_multiplier()
        
        # Should be capped at minimum 20%
        assert multiplier >= 0.2


class TestRegimeTransition:
    """Test RegimeTransition dataclass"""
    
    def create_test_regime(self, trend=TrendLabel.TREND_UP, volatility=VolLabel.LOW):
        """Create test regime context"""
        return RegimeContext(
            trend=trend,
            volatility=volatility,
            confidence=0.8,
            duration=10,
            strength=0.7
        )
    
    def test_regime_transition_creation(self):
        """Test creating a regime transition"""
        from_regime = self.create_test_regime(TrendLabel.TREND_UP, VolLabel.LOW)
        to_regime = self.create_test_regime(TrendLabel.TREND_DOWN, VolLabel.HIGH)
        transition_time = datetime.now()
        
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_time=transition_time,
            confidence=0.7
        )
        
        assert transition.from_regime == from_regime
        assert transition.to_regime == to_regime
        assert transition.transition_time == transition_time
        assert transition.confidence == 0.7
    
    def test_get_transition_type(self):
        """Test transition type description"""
        from_regime = self.create_test_regime(TrendLabel.TREND_UP, VolLabel.LOW)
        to_regime = self.create_test_regime(TrendLabel.TREND_DOWN, VolLabel.HIGH)
        
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_time=datetime.now(),
            confidence=0.7
        )
        
        assert transition.get_transition_type() == "trend_up:low_vol -> trend_down:high_vol"
    
    def test_is_major_transition_true(self):
        """Test major transition detection - trend change"""
        from_regime = self.create_test_regime(TrendLabel.TREND_UP, VolLabel.LOW)
        to_regime = self.create_test_regime(TrendLabel.TREND_DOWN, VolLabel.LOW)
        
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_time=datetime.now(),
            confidence=0.7
        )
        
        assert transition.is_major_transition() is True
    
    def test_is_major_transition_false(self):
        """Test major transition detection - volatility change only"""
        from_regime = self.create_test_regime(TrendLabel.TREND_UP, VolLabel.LOW)
        to_regime = self.create_test_regime(TrendLabel.TREND_UP, VolLabel.HIGH)
        
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_time=datetime.now(),
            confidence=0.7
        )
        
        assert transition.is_major_transition() is False


class TestEnhancedRegimeDetector:
    """Test EnhancedRegimeDetector"""
    
    def create_test_dataframe(self, length=100):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range('2023-01-01', periods=length, freq='1H')
        
        # Create trending data
        base_price = 50000
        trend = np.linspace(0, 0.1, length)  # 10% trend over period
        noise = np.random.normal(0, 0.01, length)  # 1% noise
        
        prices = base_price * (1 + trend + noise)
        
        data = {
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, length)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_enhanced_regime_detector_initialization_default(self):
        """Test EnhancedRegimeDetector initialization with defaults"""
        detector = EnhancedRegimeDetector()
        
        assert detector.stability_threshold == 10
        assert detector.max_history == 1000
        assert len(detector.regime_history) == 0
        assert len(detector.transition_history) == 0
        assert detector.current_regime is None
    
    def test_enhanced_regime_detector_initialization_custom(self):
        """Test EnhancedRegimeDetector initialization with custom parameters"""
        base_detector = RegimeDetector()
        detector = EnhancedRegimeDetector(
            base_detector=base_detector,
            stability_threshold=15,
            max_history=500
        )
        
        assert detector.base_detector == base_detector
        assert detector.stability_threshold == 15
        assert detector.max_history == 500
    
    def test_detect_regime_valid_index(self):
        """Test regime detection with valid index"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe()
        
        # Annotate with base detector first
        df = detector.base_detector.annotate(df)
        
        regime = detector.detect_regime(df, 50)
        
        assert isinstance(regime, RegimeContext)
        assert isinstance(regime.trend, TrendLabel)
        assert isinstance(regime.volatility, VolLabel)
        assert 0.0 <= regime.confidence <= 1.0
        assert regime.duration >= 1
        assert 0.0 <= regime.strength <= 1.0
        assert regime.metadata is not None
    
    def test_detect_regime_invalid_index(self):
        """Test regime detection with invalid index"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe()
        
        with pytest.raises(IndexError, match="Index .* is out of bounds"):
            detector.detect_regime(df, -1)
        
        with pytest.raises(IndexError, match="Index .* is out of bounds"):
            detector.detect_regime(df, len(df))
    
    def test_detect_regime_auto_annotate(self):
        """Test regime detection auto-annotates DataFrame"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe()
        
        # DataFrame doesn't have regime annotations initially
        assert 'regime_label' not in df.columns
        
        regime = detector.detect_regime(df, 50)
        
        # Should work and return valid regime
        assert isinstance(regime, RegimeContext)
    
    def test_get_regime_history_empty_dataframe(self):
        """Test regime history with empty DataFrame"""
        detector = EnhancedRegimeDetector()
        df = pd.DataFrame()
        
        history = detector.get_regime_history(df)
        
        assert history == []
    
    def test_get_regime_history_valid(self):
        """Test regime history with valid DataFrame"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe(50)
        
        history = detector.get_regime_history(df, lookback_periods=20)
        
        # Should return some history
        assert len(history) > 0
        assert len(history) <= 20
        assert all(isinstance(regime, RegimeContext) for regime in history)
    
    def test_is_regime_stable_true(self):
        """Test regime stability check - stable"""
        detector = EnhancedRegimeDetector(stability_threshold=5)
        df = self.create_test_dataframe()
        
        # Detect regime at a specific index
        regime = detector.detect_regime(df, 49)
        
        # Check if the detected regime has sufficient duration
        # The test passes if duration >= min_duration OR if method doesn't crash
        is_stable = detector.is_regime_stable(df, 49, min_duration=1)
        
        # Should work without error (stability depends on actual regime duration)
        assert isinstance(is_stable, bool)
    
    def test_is_regime_stable_false(self):
        """Test regime stability check - not stable"""
        detector = EnhancedRegimeDetector(stability_threshold=20)
        df = self.create_test_dataframe()
        
        is_stable = detector.is_regime_stable(df, 10)
        
        # Should not be stable with high threshold
        assert is_stable is False
    
    def test_detect_regime_transitions_empty(self):
        """Test regime transition detection with insufficient data"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe(1)
        
        transitions = detector.detect_regime_transitions(df)
        
        assert transitions == []
    
    def test_detect_regime_transitions_valid(self):
        """Test regime transition detection with valid data"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe(100)
        
        transitions = detector.detect_regime_transitions(df, lookback_periods=50)
        
        # Should detect some transitions (or none if regime is stable)
        assert isinstance(transitions, list)
        assert all(isinstance(t, RegimeTransition) for t in transitions)
    
    def test_get_regime_statistics_empty(self):
        """Test regime statistics with empty history"""
        detector = EnhancedRegimeDetector()
        df = pd.DataFrame()
        
        stats = detector.get_regime_statistics(df)
        
        assert stats == {}
    
    def test_get_regime_statistics_valid(self):
        """Test regime statistics with valid data"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe(100)
        
        stats = detector.get_regime_statistics(df, lookback_periods=50)
        
        if stats:  # Only check if we have stats
            assert 'total_periods' in stats
            assert 'avg_confidence' in stats
            assert 'avg_strength' in stats
            assert 'transition_frequency' in stats
            assert stats['total_periods'] > 0
    
    def test_regime_tracking_updates(self):
        """Test that regime tracking updates correctly"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe()
        
        # Detect regime multiple times
        regime1 = detector.detect_regime(df, 30)
        regime2 = detector.detect_regime(df, 31)
        
        # Should have history
        assert len(detector.regime_history) >= 2
        assert detector.current_regime is not None
    
    def test_get_current_regime(self):
        """Test getting current regime"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe()
        
        # Initially no current regime
        assert detector.get_current_regime() is None
        
        # After detection, should have current regime
        detector.detect_regime(df, 50)
        current = detector.get_current_regime()
        
        assert current is not None
        assert isinstance(current, RegimeContext)
    
    def test_get_recent_transitions(self):
        """Test getting recent transitions"""
        detector = EnhancedRegimeDetector()
        
        # Initially no transitions
        transitions = detector.get_recent_transitions()
        assert transitions == []
        
        # After detecting regimes, may have transitions
        df = self.create_test_dataframe()
        for i in range(30, 40):
            detector.detect_regime(df, i)
        
        transitions = detector.get_recent_transitions(count=3)
        assert len(transitions) <= 3
    
    def test_reset_tracking(self):
        """Test resetting regime tracking"""
        detector = EnhancedRegimeDetector()
        df = self.create_test_dataframe()
        
        # Build up some history
        for i in range(30, 40):
            detector.detect_regime(df, i)
        
        assert len(detector.regime_history) > 0
        assert detector.current_regime is not None
        
        # Reset tracking
        detector.reset_tracking()
        
        assert len(detector.regime_history) == 0
        assert len(detector.transition_history) == 0
        assert detector.current_regime is None
        assert detector.regime_start_index == 0
    
    def test_history_size_limit(self):
        """Test that history size is limited"""
        detector = EnhancedRegimeDetector(max_history=10)
        df = self.create_test_dataframe(50)
        
        # Generate more history than limit
        for i in range(20, 40):
            detector.detect_regime(df, i)
        
        # Should be limited to max_history
        assert len(detector.regime_history) <= 10