"""
Enhanced Regime Detector

An improved regime detection system that combines multiple indicators
and timeframes for better accuracy and faster response times.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.regime.detector import RegimeConfig, RegimeDetector

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRegimeConfig:
    """Enhanced configuration for regime detection"""
    
    # Base regime detection
    slope_window: int = 40
    band_window: int = 20
    atr_window: int = 14
    atr_percentile_lookback: int = 50  # Reduced for performance
    trend_threshold: float = 0.001
    r2_min: float = 0.3
    atr_high_percentile: float = 0.7
    hysteresis_k: int = 3
    min_dwell: int = 12
    
    # Enhanced momentum indicators
    rsi_window: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    momentum_windows: list[int] = None  # [5, 10, 20]
    
    # Volume analysis
    volume_sma_window: int = 20
    volume_spike_threshold: float = 2.0
    
    # Volatility regime detection
    garch_window: int = 50
    volatility_regime_threshold: float = 1.5
    
    # Multi-indicator weights
    trend_weight: float = 0.4
    momentum_weight: float = 0.3
    volume_weight: float = 0.2
    volatility_weight: float = 0.1
    
    # Confidence calculation
    min_confidence_threshold: float = 0.3
    confidence_smoothing: int = 10


class MarketRegime(str, Enum):
    """Enhanced market regime labels"""
    STRONG_BULL = "strong_bull"
    MILD_BULL = "mild_bull"
    STRONG_BEAR = "strong_bear"
    MILD_BEAR = "mild_bear"
    CHOPPY_RANGE = "choppy_range"
    STABLE_RANGE = "stable_range"
    HIGH_VOLATILITY = "high_volatility"
    TRANSITION = "transition"


class EnhancedRegimeDetector:
    """
    Enhanced regime detector using multiple indicators and ensemble methods
    """
    
    def __init__(self, config: Optional[EnhancedRegimeConfig] = None):
        self.config = config or EnhancedRegimeConfig()
        
        # Initialize default momentum windows
        if self.config.momentum_windows is None:
            self.config.momentum_windows = [5, 10, 20]
        
        # Base regime detector for comparison
        base_config = RegimeConfig(
            slope_window=self.config.slope_window,
            hysteresis_k=self.config.hysteresis_k,
            min_dwell=self.config.min_dwell,
            trend_threshold=self.config.trend_threshold
        )
        self.base_detector = RegimeDetector(base_config)
        
        # State tracking
        self._last_regime: Optional[str] = None
        self._regime_strength: float = 0.0
        self._consecutive: int = 0
        self._dwell: int = 0
        
        logger.info("Enhanced regime detector initialized")
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical indicators for regime detection"""
        
        result_df = df.copy()
        
        # Basic price indicators
        result_df = self._add_trend_indicators(result_df)
        result_df = self._add_momentum_indicators(result_df)
        result_df = self._add_volume_indicators(result_df)
        result_df = self._add_volatility_indicators(result_df)
        
        return result_df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators"""
        
        # Multiple EMAs for trend analysis
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Trend strength indicators
        df['trend_short'] = (df['ema_12'] - df['ema_26']) / df['ema_26']
        df['trend_medium'] = (df['ema_26'] - df['ema_50']) / df['ema_50']
        df['trend_long'] = (df['ema_50'] - df['ema_100']) / df['ema_100']
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_window = self.config.band_window
        df['bb_middle'] = df['close'].rolling(bb_window).mean()
        bb_std = df['close'].rolling(bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Trend alignment score
        df['trend_alignment'] = (
            np.sign(df['trend_short']) * 0.5 +
            np.sign(df['trend_medium']) * 0.3 +
            np.sign(df['trend_long']) * 0.2
        )
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.rsi_window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        stoch_window = 14
        df['stoch_k'] = (
            (df['close'] - df['low'].rolling(stoch_window).min()) /
            (df['high'].rolling(stoch_window).max() - df['low'].rolling(stoch_window).min())
        ) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Multi-timeframe momentum
        for window in self.config.momentum_windows:
            df[f'momentum_{window}'] = df['close'].pct_change(window)
        
        # Momentum score (composite)
        momentum_cols = [f'momentum_{w}' for w in self.config.momentum_windows]
        weights = np.array([1.0 / w for w in self.config.momentum_windows])
        weights = weights / weights.sum()
        
        momentum_values = df[momentum_cols].fillna(0).values
        df['momentum_score'] = np.dot(momentum_values, weights)
        
        # Rate of change
        df['roc'] = df['close'].pct_change(self.config.slope_window)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # Volume moving average
        df['volume_sma'] = df['volume'].rolling(self.config.volume_sma_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = np.sign(df['obv'] - df['obv_sma'])
        
        # Volume Price Trend (VPT)
        df['vpt'] = (df['close'].pct_change() * df['volume']).fillna(0).cumsum()
        df['vpt_sma'] = df['vpt'].rolling(20).mean()
        
        # Volume spike detection
        df['volume_spike'] = df['volume_ratio'] > self.config.volume_spike_threshold
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        
        # Returns for volatility calculation
        df['returns'] = df['close'].pct_change()
        
        # Simple volatility measures
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        
        # GARCH-like volatility estimate
        df['vol_garch'] = self._calculate_garch_volatility(df['returns'])
        
        # Volatility regime
        df['vol_regime'] = df['volatility_20'] / df['volatility_50']
        
        # True Range and ATR
        df['true_range'] = np.maximum(
            np.maximum(
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1))
            ),
            abs(df['low'] - df['close'].shift(1))
        )
        df['atr'] = df['true_range'].rolling(self.config.atr_window).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate GARCH-like volatility estimate using vectorized operations"""
        
        # Simple GARCH(1,1) approximation
        alpha = 0.1
        beta = 0.85
        
        returns_array = returns.to_numpy()
        n = len(returns_array)
        volatility = np.full(n, np.nan, dtype=np.float64)
        
        # Use the std of returns (ignoring NaNs) for initial volatility
        initial_vol = np.nanstd(returns_array)
        volatility[0] = initial_vol
        
        for i in range(1, n):
            if not np.isnan(returns_array[i]) and not np.isnan(volatility[i-1]):
                volatility[i] = np.sqrt(
                    alpha * returns_array[i]**2 +
                    beta * volatility[i-1]**2
                )
            else:
                volatility[i] = volatility[i-1] if i > 0 else initial_vol
        
        return pd.Series(volatility, index=returns.index)
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main regime detection using enhanced indicators"""
        
        # Calculate enhanced indicators
        df_enhanced = self.calculate_enhanced_indicators(df)
        
        # Get base regime detection
        df_base = self.base_detector.annotate(df.copy())
        
        # Combine with enhanced analysis
        df_result = self._combine_regime_signals(df_enhanced, df_base)
        
        return df_result
    
    def _combine_regime_signals(self, df_enhanced: pd.DataFrame, df_base: pd.DataFrame) -> pd.DataFrame:
        """Combine multiple regime signals into final regime classification"""
        
        result_df = df_enhanced.copy()
        
        # Add base regime signals
        result_df['base_trend_label'] = df_base['trend_label']
        result_df['base_regime_confidence'] = df_base['regime_confidence']
        
        # Calculate enhanced regime scores
        regime_scores = []
        enhanced_regimes = []
        confidence_scores = []
        
        for i in range(len(result_df)):
            if i < max(self.config.momentum_windows + [self.config.slope_window]):
                regime_scores.append(0.0)
                enhanced_regimes.append(MarketRegime.TRANSITION.value)
                confidence_scores.append(0.0)
                continue
            
            # Get current values
            row = result_df.iloc[i]
            
            # Calculate regime scores
            trend_score = self._calculate_trend_score(row)
            momentum_score = self._calculate_momentum_score(row)
            volume_score = self._calculate_volume_score(row)
            volatility_score = self._calculate_volatility_score(row)
            
            # Weighted ensemble score
            total_score = (
                trend_score * self.config.trend_weight +
                momentum_score * self.config.momentum_weight +
                volume_score * self.config.volume_weight +
                volatility_score * self.config.volatility_weight
            )
            
            # Determine enhanced regime
            enhanced_regime = self._score_to_regime(
                total_score, trend_score, momentum_score, volatility_score
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                trend_score, momentum_score, volume_score, volatility_score, row
            )
            
            regime_scores.append(total_score)
            enhanced_regimes.append(enhanced_regime)
            confidence_scores.append(confidence)
        
        result_df['enhanced_regime_score'] = regime_scores
        result_df['enhanced_regime'] = enhanced_regimes
        result_df['enhanced_confidence'] = confidence_scores
        
        # Apply hysteresis to enhanced regime
        result_df['final_regime'] = self._apply_hysteresis(enhanced_regimes)
        
        # Smooth confidence scores
        result_df['final_confidence'] = pd.Series(confidence_scores).rolling(
            self.config.confidence_smoothing, min_periods=1
        ).mean()
        
        return result_df
    
    def _calculate_trend_score(self, row: pd.Series) -> float:
        """Calculate trend component score"""
        
        trend_alignment = row.get('trend_alignment', 0)
        macd_signal = 1 if row.get('macd_histogram', 0) > 0 else -1
        bb_position = row.get('bb_position', 0.5)
        
        # Combine trend signals
        trend_score = (
            trend_alignment * 0.6 +
            macd_signal * 0.3 +
            (bb_position - 0.5) * 0.1
        )
        
        return np.clip(trend_score, -1, 1)
    
    def _calculate_momentum_score(self, row: pd.Series) -> float:
        """Calculate momentum component score"""
        
        rsi = row.get('rsi', 50)
        momentum_score = row.get('momentum_score', 0)
        roc = row.get('roc', 0)
        
        # RSI contribution
        rsi_normalized = (rsi - 50) / 50  # Convert to -1 to 1 range
        
        # Combine momentum signals
        momentum_score_combined = (
            momentum_score * 0.5 +
            rsi_normalized * 0.3 +
            np.sign(roc) * min(abs(roc) * 10, 1) * 0.2
        )
        
        return np.clip(momentum_score_combined, -1, 1)
    
    def _calculate_volume_score(self, row: pd.Series) -> float:
        """Calculate volume component score"""
        
        volume_ratio = row.get('volume_ratio', 1)
        obv_trend = row.get('obv_trend', 0)
        volume_spike = row.get('volume_spike', False)
        
        # Volume confirmation score
        volume_score = (
            np.sign(volume_ratio - 1) * min(abs(volume_ratio - 1), 1) * 0.5 +
            obv_trend * 0.3 +
            (0.2 if volume_spike else 0) * 0.2
        )
        
        return np.clip(volume_score, -1, 1)
    
    def _calculate_volatility_score(self, row: pd.Series) -> float:
        """Calculate volatility component score"""
        
        vol_regime = row.get('vol_regime', 1)
        atr_pct = row.get('atr_pct', 0.02)
        
        # High volatility reduces confidence in trend signals
        # Combine vol_regime and atr_pct for more comprehensive volatility assessment
        if vol_regime > self.config.volatility_regime_threshold or atr_pct > 0.04:
            volatility_penalty = -0.5
        elif vol_regime < 0.5 and atr_pct < 0.02:
            volatility_penalty = 0.3  # Low volatility is good for trends
        else:
            volatility_penalty = 0.0
        
        return np.clip(volatility_penalty, -1, 1)
    
    def _score_to_regime(self, total_score: float, trend_score: float, 
                        momentum_score: float, volatility_score: float) -> str:
        """Convert scores to enhanced regime classification"""
        
        # Strong signals
        if total_score > 0.6 and trend_score > 0.5:
            return MarketRegime.STRONG_BULL.value
        elif total_score < -0.6 and trend_score < -0.5:
            return MarketRegime.STRONG_BEAR.value
        
        # Moderate signals
        elif total_score > 0.3:
            return MarketRegime.MILD_BULL.value
        elif total_score < -0.3:
            return MarketRegime.MILD_BEAR.value
        
        # High volatility override
        elif volatility_score < -0.3:
            return MarketRegime.HIGH_VOLATILITY.value
        
        # Range markets
        elif abs(total_score) < 0.15:
            if abs(momentum_score) < 0.1:
                return MarketRegime.STABLE_RANGE.value
            else:
                return MarketRegime.CHOPPY_RANGE.value
        
        # Default to transition
        else:
            return MarketRegime.TRANSITION.value
    
    def _calculate_confidence(self, trend_score: float, momentum_score: float,
                            volume_score: float, volatility_score: float,
                            row: pd.Series) -> float:
        """Calculate confidence in regime classification"""
        
        # Base confidence from signal strength
        signal_strength = abs(trend_score) * 0.4 + abs(momentum_score) * 0.3 + abs(volume_score) * 0.3
        
        # Reduce confidence for high volatility
        vol_penalty = max(0, volatility_score * -0.5)
        
        # Increase confidence for signal alignment
        signal_alignment = 1 - abs(trend_score - momentum_score) / 2
        
        # Base regime confidence
        base_confidence = row.get('base_regime_confidence', 0.5)
        
        # Combined confidence
        confidence = (
            signal_strength * 0.5 +
            signal_alignment * 0.3 +
            base_confidence * 0.2 -
            vol_penalty
        )
        
        return np.clip(confidence, 0, 1)
    
    def _apply_hysteresis(self, regimes: list[str]) -> list[str]:
        """Apply hysteresis to prevent regime flip-flopping"""
        
        result = []
        last_regime = None
        consecutive = 0
        dwell = 0
        
        for regime in regimes:
            if last_regime is None:
                last_regime = regime
                consecutive = 1
                dwell = 1
                result.append(regime)
                continue
            
            if regime == last_regime:
                consecutive += 1
                dwell += 1
                result.append(last_regime)
                continue
            
            # Different regime proposed
            consecutive += 1
            
            # Check if we should switch
            if dwell >= self.config.min_dwell and consecutive >= self.config.hysteresis_k:
                last_regime = regime
                dwell = 1
                consecutive = 1
            else:
                # Don't switch yet
                pass
            
            result.append(last_regime)
        
        return result
    
    def get_current_regime(self, df: pd.DataFrame) -> tuple[str, float]:
        """Get current market regime and confidence"""
        
        if df.empty or 'final_regime' not in df.columns:
            return MarketRegime.TRANSITION.value, 0.0
        
        current_regime = df['final_regime'].iloc[-1]
        current_confidence = df['final_confidence'].iloc[-1]
        
        return current_regime, current_confidence
    
    def get_regime_summary(self, df: pd.DataFrame, lookback: int = 100) -> dict[str, any]:
        """Get summary of recent regime behavior"""
        
        if df.empty or len(df) < lookback:
            return {}
        
        recent_df = df.tail(lookback)
        
        # Regime distribution
        regime_counts = recent_df['final_regime'].value_counts()
        regime_distribution = (regime_counts / len(recent_df)).to_dict()
        
        # Regime stability (fewer unique regimes = more stable)
        unique_regimes = recent_df['final_regime'].nunique()
        stability_score = 1.0 / unique_regimes if unique_regimes > 0 else 0.0
        
        # Average confidence
        avg_confidence = recent_df['final_confidence'].mean()
        
        # Regime transitions
        transitions = []
        prev_regime = None
        for i, regime in enumerate(recent_df['final_regime']):
            if prev_regime and regime != prev_regime:
                transitions.append((i, prev_regime, regime))
            prev_regime = regime
        
        return {
            'regime_distribution': regime_distribution,
            'stability_score': stability_score,
            'average_confidence': avg_confidence,
            'num_transitions': len(transitions),
            'recent_transitions': transitions[-5:],  # Last 5 transitions
            'current_regime': recent_df['final_regime'].iloc[-1],
            'current_confidence': recent_df['final_confidence'].iloc[-1]
        }