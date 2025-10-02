"""
Regime Context and Detection Utilities

This module defines the RegimeContext data model and enhanced RegimeDetector
for providing market regime information to strategy components.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# Import existing regime detection components
from src.regime.detector import RegimeDetector as BaseRegimeDetector, TrendLabel, VolLabel


@dataclass
class RegimeContext:
    """
    Enhanced regime context for strategy components
    
    Attributes:
        trend: Market trend direction (TREND_UP, TREND_DOWN, RANGE)
        volatility: Market volatility level (HIGH, LOW)
        confidence: Confidence in regime classification (0.0 to 1.0)
        duration: How long current regime has been active (in periods)
        strength: Strength of the regime signal (0.0 to 1.0)
        timestamp: When this regime context was created
        metadata: Additional regime information
    """
    trend: TrendLabel
    volatility: VolLabel
    confidence: float
    duration: int
    strength: float
    timestamp: Optional[datetime] = None
    metadata: Optional[dict[str, float]] = None
    
    def __post_init__(self):
        """Validate regime context parameters after initialization"""
        self._validate_regime_context()
    
    def _validate_regime_context(self):
        """Validate regime context parameters are within acceptable bounds"""
        if not isinstance(self.trend, TrendLabel):
            raise ValueError(f"trend must be a TrendLabel enum, got {type(self.trend)}")
        
        if not isinstance(self.volatility, VolLabel):
            raise ValueError(f"volatility must be a VolLabel enum, got {type(self.volatility)}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.duration < 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")
        
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be between 0.0 and 1.0, got {self.strength}")
        
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary when provided, got {type(self.metadata)}")
    
    def get_regime_label(self) -> str:
        """Get combined regime label string"""
        return f"{self.trend.value}:{self.volatility.value}"
    
    def is_stable(self, min_duration: int = 10) -> bool:
        """Check if regime has been stable for minimum duration"""
        return self.duration >= min_duration
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if regime confidence is above threshold"""
        return self.confidence >= threshold
    
    def is_strong_regime(self, threshold: float = 0.6) -> bool:
        """Check if regime strength is above threshold"""
        return self.strength >= threshold
    
    def get_risk_multiplier(self) -> float:
        """Get risk adjustment multiplier based on regime characteristics"""
        multiplier = 1.0
        
        # Reduce risk in high volatility
        if self.volatility == VolLabel.HIGH:
            multiplier *= 0.8
        
        # Reduce risk in bear markets
        if self.trend == TrendLabel.TREND_DOWN:
            multiplier *= 0.7
        
        # Reduce risk in range markets
        if self.trend == TrendLabel.RANGE:
            multiplier *= 0.9
        
        # Reduce risk when confidence is low
        if self.confidence < 0.5:
            multiplier *= 0.8
        
        # Reduce risk when regime is not stable
        if not self.is_stable():
            multiplier *= 0.9
        
        return max(0.2, multiplier)  # Minimum 20% of base risk


@dataclass
class RegimeTransition:
    """
    Information about a regime transition
    
    Attributes:
        from_regime: Previous regime context
        to_regime: New regime context
        transition_time: When the transition occurred
        confidence: Confidence in the transition
    """
    from_regime: RegimeContext
    to_regime: RegimeContext
    transition_time: datetime
    confidence: float
    
    def get_transition_type(self) -> str:
        """Get description of transition type"""
        from_label = self.from_regime.get_regime_label()
        to_label = self.to_regime.get_regime_label()
        return f"{from_label} -> {to_label}"
    
    def is_major_transition(self) -> bool:
        """Check if this is a major regime transition (trend change)"""
        return self.from_regime.trend != self.to_regime.trend


class EnhancedRegimeDetector:
    """
    Enhanced regime detector with component-specific adaptations
    
    Extends the base regime detector with additional functionality for
    strategy components including regime stability detection, history tracking,
    and transition analysis.
    """
    
    def __init__(self, base_detector: Optional[BaseRegimeDetector] = None,
                 stability_threshold: int = 10, max_history: int = 1000):
        """
        Initialize enhanced regime detector
        
        Args:
            base_detector: Base regime detector instance
            stability_threshold: Minimum periods for regime stability
            max_history: Maximum regime history to maintain
        """
        self.base_detector = base_detector or BaseRegimeDetector()
        self.stability_threshold = stability_threshold
        self.max_history = max_history
        
        # Regime history tracking
        self.regime_history: List[RegimeContext] = []
        self.transition_history: List[RegimeTransition] = []
        
        # Current regime state
        self.current_regime: Optional[RegimeContext] = None
        self.regime_start_index: int = 0
    
    def detect_regime(self, df: pd.DataFrame, index: int) -> RegimeContext:
        """
        Detect current market regime with enhanced context
        
        Args:
            df: DataFrame with OHLCV data and regime annotations
            index: Current index position
            
        Returns:
            RegimeContext with comprehensive regime information
        """
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} is out of bounds for DataFrame of length {len(df)}")
        
        # Ensure regime annotations exist
        if 'regime_label' not in df.columns:
            df = self.base_detector.annotate(df)
        
        # Get current regime labels
        current_row = df.iloc[index]
        trend_label = TrendLabel(current_row['trend_label'])
        vol_label = VolLabel(current_row['vol_label'])
        base_confidence = float(current_row.get('regime_confidence', 0.5))
        
        # Calculate regime duration
        duration = self._calculate_regime_duration(df, index, trend_label)
        
        # Calculate regime strength
        strength = self._calculate_regime_strength(df, index)
        
        # Enhance confidence with stability and strength
        enhanced_confidence = self._enhance_confidence(base_confidence, duration, strength)
        
        # Create regime context
        regime_context = RegimeContext(
            trend=trend_label,
            volatility=vol_label,
            confidence=enhanced_confidence,
            duration=duration,
            strength=strength,
            timestamp=datetime.now(),
            metadata={
                'trend_score': float(current_row.get('trend_score', 0.0)),
                'atr_percentile': float(current_row.get('atr_percentile', 0.5)),
                'base_confidence': base_confidence,
                'index': index
            }
        )
        
        # Update regime tracking
        self._update_regime_tracking(regime_context, index)
        
        return regime_context
    
    def get_regime_history(self, df: pd.DataFrame, lookback_periods: int = 100) -> List[RegimeContext]:
        """
        Get historical regime contexts
        
        Args:
            df: DataFrame with OHLCV data
            lookback_periods: Number of periods to look back
            
        Returns:
            List of historical RegimeContext objects
        """
        if df.empty:
            return []
        
        # Ensure we have regime annotations
        if 'regime_label' not in df.columns:
            df = self.base_detector.annotate(df)
        
        history = []
        start_index = max(0, len(df) - lookback_periods)
        
        for i in range(start_index, len(df)):
            try:
                regime_context = self.detect_regime(df, i)
                history.append(regime_context)
            except (IndexError, ValueError):
                continue
        
        return history
    
    def is_regime_stable(self, df: pd.DataFrame, index: int, min_duration: Optional[int] = None) -> bool:
        """
        Check if current regime is stable
        
        Args:
            df: DataFrame with OHLCV data
            index: Current index position
            min_duration: Minimum duration for stability (uses default if None)
            
        Returns:
            True if regime is stable, False otherwise
        """
        min_dur = min_duration or self.stability_threshold
        
        try:
            regime_context = self.detect_regime(df, index)
            return regime_context.is_stable(min_dur)
        except (IndexError, ValueError):
            return False
    
    def detect_regime_transitions(self, df: pd.DataFrame, 
                                lookback_periods: int = 50) -> List[RegimeTransition]:
        """
        Detect recent regime transitions
        
        Args:
            df: DataFrame with OHLCV data
            lookback_periods: Number of periods to analyze
            
        Returns:
            List of detected regime transitions
        """
        if df.empty or len(df) < 2:
            return []
        
        transitions = []
        start_index = max(1, len(df) - lookback_periods)
        
        prev_regime = None
        for i in range(start_index, len(df)):
            try:
                current_regime = self.detect_regime(df, i)
                
                if prev_regime is not None:
                    # Check for regime change
                    if (prev_regime.trend != current_regime.trend or 
                        prev_regime.volatility != current_regime.volatility):
                        
                        transition = RegimeTransition(
                            from_regime=prev_regime,
                            to_regime=current_regime,
                            transition_time=current_regime.timestamp or datetime.now(),
                            confidence=min(prev_regime.confidence, current_regime.confidence)
                        )
                        transitions.append(transition)
                
                prev_regime = current_regime
                
            except (IndexError, ValueError):
                continue
        
        return transitions
    
    def get_regime_statistics(self, df: pd.DataFrame, 
                            lookback_periods: int = 252) -> dict[str, float]:
        """
        Get statistical information about regime behavior
        
        Args:
            df: DataFrame with OHLCV data
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with regime statistics
        """
        history = self.get_regime_history(df, lookback_periods)
        
        if not history:
            return {}
        
        # Count regime occurrences
        regime_counts = {}
        confidence_sum = 0.0
        strength_sum = 0.0
        duration_sum = 0
        
        for regime in history:
            label = regime.get_regime_label()
            regime_counts[label] = regime_counts.get(label, 0) + 1
            confidence_sum += regime.confidence
            strength_sum += regime.strength
            duration_sum += regime.duration
        
        total_periods = len(history)
        
        # Calculate statistics
        stats = {
            'total_periods': total_periods,
            'avg_confidence': confidence_sum / total_periods,
            'avg_strength': strength_sum / total_periods,
            'avg_duration': duration_sum / total_periods,
        }
        
        # Add regime distribution
        for label, count in regime_counts.items():
            stats[f'{label}_pct'] = (count / total_periods) * 100
        
        # Calculate transition frequency
        transitions = self.detect_regime_transitions(df, lookback_periods)
        stats['transition_frequency'] = len(transitions) / total_periods if total_periods > 0 else 0
        
        return stats
    
    def _calculate_regime_duration(self, df: pd.DataFrame, index: int, 
                                 current_trend: TrendLabel) -> int:
        """Calculate how long current regime has been active"""
        duration = 1
        
        # Look backwards to find regime start
        for i in range(index - 1, -1, -1):
            if i < 0 or 'trend_label' not in df.columns:
                break
            
            try:
                prev_trend = TrendLabel(df.iloc[i]['trend_label'])
                if prev_trend == current_trend:
                    duration += 1
                else:
                    break
            except (KeyError, ValueError):
                break
        
        return duration
    
    def _calculate_regime_strength(self, df: pd.DataFrame, index: int, 
                                 window: int = 20) -> float:
        """Calculate regime strength based on trend consistency"""
        if index < window:
            return 0.5  # Default strength for insufficient data
        
        try:
            # Get trend scores for recent period
            start_idx = max(0, index - window + 1)
            trend_scores = df.iloc[start_idx:index + 1]['trend_score'].values
            
            # Calculate strength as consistency of trend direction
            if len(trend_scores) == 0:
                return 0.5
            
            # Remove NaN values
            valid_scores = trend_scores[~np.isnan(trend_scores)]
            if len(valid_scores) == 0:
                return 0.5
            
            # Calculate strength as normalized absolute mean
            mean_score = np.mean(np.abs(valid_scores))
            
            # Normalize to 0-1 range (assuming trend scores typically range -0.1 to 0.1)
            strength = min(1.0, mean_score / 0.05)
            
            return max(0.0, strength)
            
        except (KeyError, IndexError):
            return 0.5
    
    def _enhance_confidence(self, base_confidence: float, duration: int, 
                          strength: float) -> float:
        """Enhance confidence based on regime stability and strength"""
        enhanced = base_confidence
        
        # Boost confidence for stable regimes
        if duration >= self.stability_threshold:
            stability_boost = min(0.2, duration / (self.stability_threshold * 5))
            enhanced += stability_boost
        
        # Boost confidence for strong regimes
        if strength > 0.7:
            strength_boost = (strength - 0.7) * 0.3
            enhanced += strength_boost
        
        # Penalize confidence for very short regimes
        if duration < 3:
            enhanced *= 0.8
        
        return max(0.0, min(1.0, enhanced))
    
    def _update_regime_tracking(self, regime_context: RegimeContext, index: int) -> None:
        """Update internal regime tracking state"""
        # Check for regime change
        if (self.current_regime is None or 
            self.current_regime.trend != regime_context.trend or
            self.current_regime.volatility != regime_context.volatility):
            
            # Record transition if we had a previous regime
            if self.current_regime is not None:
                transition = RegimeTransition(
                    from_regime=self.current_regime,
                    to_regime=regime_context,
                    transition_time=regime_context.timestamp or datetime.now(),
                    confidence=min(self.current_regime.confidence, regime_context.confidence)
                )
                self.transition_history.append(transition)
                
                # Limit transition history
                if len(self.transition_history) > self.max_history // 10:
                    self.transition_history = self.transition_history[-self.max_history // 10:]
            
            self.regime_start_index = index
        
        # Update current regime
        self.current_regime = regime_context
        
        # Add to history
        self.regime_history.append(regime_context)
        
        # Limit history size
        if len(self.regime_history) > self.max_history:
            self.regime_history = self.regime_history[-self.max_history:]
    
    def get_current_regime(self) -> Optional[RegimeContext]:
        """Get the current regime context"""
        return self.current_regime
    
    def get_recent_transitions(self, count: int = 5) -> List[RegimeTransition]:
        """Get most recent regime transitions"""
        return self.transition_history[-count:] if self.transition_history else []
    
    def reset_tracking(self) -> None:
        """Reset regime tracking state"""
        self.regime_history.clear()
        self.transition_history.clear()
        self.current_regime = None
        self.regime_start_index = 0