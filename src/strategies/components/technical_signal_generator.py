"""
Technical Signal Generator Components

This module contains technical indicator-based signal generators that use
traditional technical analysis methods to generate trading signals.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.indicators.technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
)
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator


class TechnicalSignalGenerator(SignalGenerator):
    """
    Technical Signal Generator using traditional technical indicators
    
    This signal generator combines multiple technical indicators to generate
    trading signals including RSI, MACD, moving averages, and Bollinger Bands.
    
    Features:
    - RSI overbought/oversold signals
    - MACD crossover signals
    - Moving average trend signals
    - Bollinger Bands mean reversion signals
    - Configurable parameters for all indicators
    """
    
    def __init__(
        self,
        name: str = "technical_signal_generator",
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ma_short: int = 20,
        ma_long: int = 50,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        atr_period: int = 14,
    ):
        """
        Initialize Technical Signal Generator
        
        Args:
            name: Name for this signal generator
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            ma_short: Short moving average period
            ma_long: Long moving average period
            bb_period: Bollinger Bands period
            bb_std_dev: Bollinger Bands standard deviation multiplier
            atr_period: ATR period for volatility measurement
        """
        super().__init__(name)
        
        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        # MACD parameters
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # Moving average parameters
        self.ma_short = ma_short
        self.ma_long = ma_long
        
        # Bollinger Bands parameters
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        
        # ATR parameters
        self.atr_period = atr_period
        
        # Minimum periods required for signal generation
        self.min_periods = max(
            self.rsi_period,
            self.macd_slow + self.macd_signal,
            self.ma_long,
            self.bb_period,
            self.atr_period
        )
    
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None) -> Signal:
        """
        Generate trading signal based on technical indicators
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            index: Current index position in the DataFrame
            regime: Optional regime context for regime-aware signal generation
            
        Returns:
            Signal object with direction, strength, confidence, and metadata
        """
        self.validate_inputs(df, index)
        
        # Ensure we have enough history for indicators
        if index < self.min_periods:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'insufficient_history',
                    'index': index,
                    'required_periods': self.min_periods
                }
            )
        
        # Calculate technical indicators if not already present
        df_with_indicators = self._ensure_indicators(df)
        
        # Get individual signal components
        rsi_signal = self._get_rsi_signal(df_with_indicators, index)
        macd_signal = self._get_macd_signal(df_with_indicators, index)
        ma_signal = self._get_ma_signal(df_with_indicators, index)
        bb_signal = self._get_bb_signal(df_with_indicators, index)
        
        # Combine signals
        combined_signal = self._combine_signals(
            rsi_signal, macd_signal, ma_signal, bb_signal, regime
        )
        
        # Calculate overall strength and confidence
        strength = self._calculate_signal_strength(
            rsi_signal, macd_signal, ma_signal, bb_signal
        )
        confidence = self._calculate_signal_confidence(
            rsi_signal, macd_signal, ma_signal, bb_signal, df_with_indicators, index
        )
        
        # Create metadata
        metadata = {
            'generator': self.name,
            'index': index,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'ma_signal': ma_signal,
            'bb_signal': bb_signal,
            'rsi_value': float(df_with_indicators['rsi'].iloc[index]),
            'macd_value': float(df_with_indicators['macd'].iloc[index]),
            'macd_signal_value': float(df_with_indicators['macd_signal'].iloc[index]),
            'ma_short_value': float(df_with_indicators[f'ma_{self.ma_short}'].iloc[index]),
            'ma_long_value': float(df_with_indicators[f'ma_{self.ma_long}'].iloc[index]),
        }
        
        # Add regime information if available
        if regime:
            metadata.update({
                'regime_trend': regime.trend.value,
                'regime_volatility': regime.volatility.value,
                'regime_confidence': regime.confidence
            })
        
        return Signal(
            direction=combined_signal,
            strength=strength,
            confidence=confidence,
            metadata=metadata
        )
    
    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """
        Get confidence score for signal generation at the given index
        
        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        self.validate_inputs(df, index)
        
        if index < self.min_periods:
            return 0.0
        
        # Calculate technical indicators if not already present
        df_with_indicators = self._ensure_indicators(df)
        
        # Get individual signal components
        rsi_signal = self._get_rsi_signal(df_with_indicators, index)
        macd_signal = self._get_macd_signal(df_with_indicators, index)
        ma_signal = self._get_ma_signal(df_with_indicators, index)
        bb_signal = self._get_bb_signal(df_with_indicators, index)
        
        return self._calculate_signal_confidence(
            rsi_signal, macd_signal, ma_signal, bb_signal, df_with_indicators, index
        )
    
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required technical indicators are calculated"""
        df_copy = df.copy()
        
        # Calculate RSI if not present
        if 'rsi' not in df_copy.columns:
            df_copy['rsi'] = calculate_rsi(df_copy, self.rsi_period)
        
        # Calculate MACD if not present
        macd_columns = ['macd', 'macd_signal', 'macd_hist']
        if not all(col in df_copy.columns for col in macd_columns):
            df_copy = calculate_macd(
                df_copy, self.macd_fast, self.macd_slow, self.macd_signal
            )
        
        # Calculate moving averages if not present
        ma_columns = [f'ma_{self.ma_short}', f'ma_{self.ma_long}']
        if not all(col in df_copy.columns for col in ma_columns):
            df_copy = calculate_moving_averages(df_copy, [self.ma_short, self.ma_long])
        
        # Calculate Bollinger Bands if not present
        bb_columns = ['bb_upper', 'bb_middle', 'bb_lower']
        if not all(col in df_copy.columns for col in bb_columns):
            df_copy = calculate_bollinger_bands(df_copy, self.bb_period, self.bb_std_dev)
        
        # Calculate ATR if not present
        if 'atr' not in df_copy.columns:
            df_copy = calculate_atr(df_copy, self.atr_period)
        
        return df_copy
    
    def _get_rsi_signal(self, df: pd.DataFrame, index: int) -> int:
        """
        Get RSI signal (-1: sell, 0: hold, 1: buy)
        
        Args:
            df: DataFrame with RSI calculated
            index: Current index
            
        Returns:
            RSI signal value
        """
        rsi = df['rsi'].iloc[index]
        
        if pd.isna(rsi):
            return 0
        
        if rsi >= self.rsi_overbought:
            return -1  # Sell signal (overbought)
        elif rsi <= self.rsi_oversold:
            return 1   # Buy signal (oversold)
        else:
            return 0   # Hold
    
    def _get_macd_signal(self, df: pd.DataFrame, index: int) -> int:
        """
        Get MACD signal (-1: sell, 0: hold, 1: buy)
        
        Args:
            df: DataFrame with MACD calculated
            index: Current index
            
        Returns:
            MACD signal value
        """
        if index < 1:
            return 0
        
        macd_current = df['macd'].iloc[index]
        macd_signal_current = df['macd_signal'].iloc[index]
        macd_prev = df['macd'].iloc[index - 1]
        macd_signal_prev = df['macd_signal'].iloc[index - 1]
        
        if pd.isna(macd_current) or pd.isna(macd_signal_current):
            return 0
        
        # MACD crossover signals
        if macd_prev <= macd_signal_prev and macd_current > macd_signal_current:
            return 1   # Bullish crossover
        elif macd_prev >= macd_signal_prev and macd_current < macd_signal_current:
            return -1  # Bearish crossover
        else:
            return 0   # No crossover
    
    def _get_ma_signal(self, df: pd.DataFrame, index: int) -> int:
        """
        Get Moving Average signal (-1: sell, 0: hold, 1: buy)
        
        Args:
            df: DataFrame with moving averages calculated
            index: Current index
            
        Returns:
            Moving average signal value
        """
        ma_short = df[f'ma_{self.ma_short}'].iloc[index]
        ma_long = df[f'ma_{self.ma_long}'].iloc[index]
        current_price = df['close'].iloc[index]
        
        if pd.isna(ma_short) or pd.isna(ma_long):
            return 0
        
        # Price above both MAs and short MA above long MA = bullish
        if current_price > ma_short > ma_long:
            return 1
        # Price below both MAs and short MA below long MA = bearish
        elif current_price < ma_short < ma_long:
            return -1
        else:
            return 0
    
    def _get_bb_signal(self, df: pd.DataFrame, index: int) -> int:
        """
        Get Bollinger Bands signal (-1: sell, 0: hold, 1: buy)
        
        Args:
            df: DataFrame with Bollinger Bands calculated
            index: Current index
            
        Returns:
            Bollinger Bands signal value
        """
        current_price = df['close'].iloc[index]
        bb_upper = df['bb_upper'].iloc[index]
        bb_lower = df['bb_lower'].iloc[index]
        bb_middle = df['bb_middle'].iloc[index]
        
        if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(bb_middle):
            return 0
        
        # Mean reversion signals
        if current_price <= bb_lower:
            return 1   # Buy at lower band (oversold)
        elif current_price >= bb_upper:
            return -1  # Sell at upper band (overbought)
        else:
            return 0   # Hold in middle range
    
    def _combine_signals(
        self,
        rsi_signal: int,
        macd_signal: int,
        ma_signal: int,
        bb_signal: int,
        regime: Optional[RegimeContext] = None
    ) -> SignalDirection:
        """
        Combine individual signals into final signal direction
        
        Args:
            rsi_signal: RSI signal value
            macd_signal: MACD signal value
            ma_signal: Moving average signal value
            bb_signal: Bollinger Bands signal value
            regime: Optional regime context
            
        Returns:
            Combined signal direction
        """
        # Weight the signals (can be adjusted based on regime)
        weights = {
            'rsi': 0.2,
            'macd': 0.3,
            'ma': 0.3,
            'bb': 0.2
        }
        
        # Adjust weights based on regime if available
        if regime:
            # In trending markets, give more weight to trend-following indicators
            if regime.trend.value in ['trend_up', 'trend_down']:
                weights['ma'] = 0.4
                weights['macd'] = 0.3
                weights['rsi'] = 0.15
                weights['bb'] = 0.15
            # In ranging markets, give more weight to mean reversion indicators
            elif regime.trend.value == 'range':
                weights['rsi'] = 0.3
                weights['bb'] = 0.3
                weights['ma'] = 0.2
                weights['macd'] = 0.2
        
        # Calculate weighted signal
        weighted_signal = (
            rsi_signal * weights['rsi'] +
            macd_signal * weights['macd'] +
            ma_signal * weights['ma'] +
            bb_signal * weights['bb']
        )
        
        # Convert to signal direction with threshold
        threshold = 0.1  # Require some conviction for signal
        
        if weighted_signal > threshold:
            return SignalDirection.BUY
        elif weighted_signal < -threshold:
            return SignalDirection.SELL
        else:
            return SignalDirection.HOLD
    
    def _calculate_signal_strength(
        self,
        rsi_signal: int,
        macd_signal: int,
        ma_signal: int,
        bb_signal: int
    ) -> float:
        """
        Calculate signal strength based on agreement between indicators
        
        Args:
            rsi_signal: RSI signal value
            macd_signal: MACD signal value
            ma_signal: Moving average signal value
            bb_signal: Bollinger Bands signal value
            
        Returns:
            Signal strength between 0.0 and 1.0
        """
        signals = [rsi_signal, macd_signal, ma_signal, bb_signal]
        
        # Count agreements
        buy_signals = sum(1 for s in signals if s > 0)
        sell_signals = sum(1 for s in signals if s < 0)
        total_signals = len(signals)
        
        # Calculate strength as proportion of agreeing signals
        if buy_signals > sell_signals:
            strength = buy_signals / total_signals
        elif sell_signals > buy_signals:
            strength = sell_signals / total_signals
        else:
            strength = 0.0  # No clear direction
        
        return min(1.0, max(0.0, strength))
    
    def _calculate_signal_confidence(
        self,
        rsi_signal: int,
        macd_signal: int,
        ma_signal: int,
        bb_signal: int,
        df: pd.DataFrame,
        index: int
    ) -> float:
        """
        Calculate signal confidence based on indicator strength and market conditions
        
        Args:
            rsi_signal: RSI signal value
            macd_signal: MACD signal value
            ma_signal: Moving average signal value
            bb_signal: Bollinger Bands signal value
            df: DataFrame with indicators
            index: Current index
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence_factors = []
        
        # RSI confidence (stronger at extremes)
        rsi = df['rsi'].iloc[index]
        if not pd.isna(rsi):
            if rsi <= 20 or rsi >= 80:
                confidence_factors.append(0.9)  # Very extreme
            elif rsi <= 30 or rsi >= 70:
                confidence_factors.append(0.7)  # Extreme
            else:
                confidence_factors.append(0.3)  # Neutral
        
        # MACD confidence (stronger with larger histogram)
        macd_hist = df['macd_hist'].iloc[index]
        if not pd.isna(macd_hist):
            hist_strength = min(1.0, abs(macd_hist) * 100)  # Scale histogram
            confidence_factors.append(hist_strength)
        
        # MA confidence (stronger with larger separation)
        ma_short = df[f'ma_{self.ma_short}'].iloc[index]
        ma_long = df[f'ma_{self.ma_long}'].iloc[index]
        if not pd.isna(ma_short) and not pd.isna(ma_long):
            ma_separation = abs(ma_short - ma_long) / ma_long if ma_long > 0 else 0
            ma_confidence = min(1.0, ma_separation * 20)  # Scale separation
            confidence_factors.append(ma_confidence)
        
        # BB confidence (stronger near bands)
        current_price = df['close'].iloc[index]
        bb_upper = df['bb_upper'].iloc[index]
        bb_lower = df['bb_lower'].iloc[index]
        bb_middle = df['bb_middle'].iloc[index]
        
        if not pd.isna(bb_upper) and not pd.isna(bb_lower) and not pd.isna(bb_middle):
            bb_width = bb_upper - bb_lower
            if bb_width > 0:
                # Distance from middle as proportion of band width
                distance_from_middle = abs(current_price - bb_middle)
                bb_position = distance_from_middle / (bb_width / 2)
                bb_confidence = min(1.0, bb_position)  # Higher confidence near bands
                confidence_factors.append(bb_confidence)
        
        # Overall confidence as average of factors
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default confidence
    
    def get_parameters(self) -> dict[str, Any]:
        """Get signal generator parameters for logging and serialization"""
        params = super().get_parameters()
        params.update({
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'ma_short': self.ma_short,
            'ma_long': self.ma_long,
            'bb_period': self.bb_period,
            'bb_std_dev': self.bb_std_dev,
            'atr_period': self.atr_period,
            'min_periods': self.min_periods
        })
        return params


class RSISignalGenerator(SignalGenerator):
    """
    Simple RSI-based signal generator
    
    Generates signals based solely on RSI overbought/oversold conditions.
    """
    
    def __init__(
        self,
        name: str = "rsi_signal_generator",
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0
    ):
        """
        Initialize RSI Signal Generator
        
        Args:
            name: Name for this signal generator
            period: RSI calculation period
            overbought: RSI overbought threshold
            oversold: RSI oversold threshold
        """
        super().__init__(name)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None) -> Signal:
        """Generate signal based on RSI levels"""
        self.validate_inputs(df, index)

        # Calculate RSI if not present
        if 'rsi' not in df.columns:
            # Check if we have enough data for RSI calculation
            if len(df) < self.period:
                return Signal(
                    direction=SignalDirection.HOLD,
                    strength=0.0,
                    confidence=0.0,
                    metadata={
                        'generator': self.name,
                        'reason': 'insufficient_history',
                        'index': index
                    }
                )
            df = df.copy()
            df['rsi'] = calculate_rsi(df, self.period)
        
        rsi = df['rsi'].iloc[index]
        
        if pd.isna(rsi):
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'rsi_calculation_failed',
                    'index': index
                }
            )
        
        # Determine signal
        if rsi >= self.overbought:
            direction = SignalDirection.SELL
            strength = min(1.0, (rsi - self.overbought) / (100 - self.overbought))
        elif rsi <= self.oversold:
            direction = SignalDirection.BUY
            strength = min(1.0, (self.oversold - rsi) / self.oversold)
        else:
            direction = SignalDirection.HOLD
            strength = 0.0
        
        # Calculate confidence (higher at extremes)
        if rsi <= 20 or rsi >= 80:
            confidence = 0.9
        elif rsi <= 30 or rsi >= 70:
            confidence = 0.7
        else:
            confidence = 0.3
        
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'generator': self.name,
                'rsi_value': float(rsi),
                'overbought_threshold': self.overbought,
                'oversold_threshold': self.oversold,
                'index': index
            }
        )
    
    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence based on RSI extremity"""
        self.validate_inputs(df, index)

        if 'rsi' not in df.columns:
            if len(df) < self.period:
                return 0.0
            df = df.copy()
            df['rsi'] = calculate_rsi(df, self.period)
        
        rsi = df['rsi'].iloc[index]
        
        if pd.isna(rsi):
            return 0.0
        
        # Higher confidence at extremes
        if rsi <= 20 or rsi >= 80:
            return 0.9
        elif rsi <= 30 or rsi >= 70:
            return 0.7
        else:
            return 0.3
    
    def get_parameters(self) -> dict[str, Any]:
        """Get RSI signal generator parameters"""
        params = super().get_parameters()
        params.update({
            'period': self.period,
            'overbought': self.overbought,
            'oversold': self.oversold
        })
        return params


class MACDSignalGenerator(SignalGenerator):
    """
    Simple MACD-based signal generator
    
    Generates signals based on MACD line crossovers with signal line.
    """
    
    def __init__(
        self,
        name: str = "macd_signal_generator",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        """
        Initialize MACD Signal Generator
        
        Args:
            name: Name for this signal generator
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_periods = slow_period + signal_period
    
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None) -> Signal:
        """Generate signal based on MACD crossovers"""
        self.validate_inputs(df, index)
        
        if index < max(1, self.min_periods):
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'insufficient_history',
                    'index': index
                }
            )
        
        # Calculate MACD if not present
        macd_columns = ['macd', 'macd_signal', 'macd_hist']
        if not all(col in df.columns for col in macd_columns):
            df = calculate_macd(df.copy(), self.fast_period, self.slow_period, self.signal_period)
        
        macd_current = df['macd'].iloc[index]
        macd_signal_current = df['macd_signal'].iloc[index]
        macd_prev = df['macd'].iloc[index - 1]
        macd_signal_prev = df['macd_signal'].iloc[index - 1]
        macd_hist = df['macd_hist'].iloc[index]
        
        if pd.isna(macd_current) or pd.isna(macd_signal_current):
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'macd_calculation_failed',
                    'index': index
                }
            )
        
        # Determine signal based on crossover
        if macd_prev <= macd_signal_prev and macd_current > macd_signal_current:
            direction = SignalDirection.BUY
            strength = min(1.0, abs(macd_hist) * 100)
        elif macd_prev >= macd_signal_prev and macd_current < macd_signal_current:
            direction = SignalDirection.SELL
            strength = min(1.0, abs(macd_hist) * 100)
        else:
            direction = SignalDirection.HOLD
            strength = 0.0
        
        # Calculate confidence based on histogram strength
        if not pd.isna(macd_hist):
            # Scale histogram to confidence - use a more gradual scaling
            hist_abs = abs(macd_hist)
            if hist_abs >= 0.05:
                confidence = 0.9
            elif hist_abs >= 0.02:
                confidence = 0.7
            elif hist_abs >= 0.01:
                confidence = 0.5
            else:
                confidence = 0.3
        else:
            confidence = 0.5
        
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'generator': self.name,
                'macd_value': float(macd_current),
                'macd_signal_value': float(macd_signal_current),
                'macd_histogram': float(macd_hist) if not pd.isna(macd_hist) else None,
                'crossover_detected': direction != SignalDirection.HOLD,
                'index': index
            }
        )
    
    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence based on MACD histogram strength"""
        self.validate_inputs(df, index)
        
        if index < self.min_periods:
            return 0.0
        
        macd_columns = ['macd', 'macd_signal', 'macd_hist']
        if not all(col in df.columns for col in macd_columns):
            df = calculate_macd(df.copy(), self.fast_period, self.slow_period, self.signal_period)
        
        macd_hist = df['macd_hist'].iloc[index]
        
        if pd.isna(macd_hist):
            return 0.0
        # Use same scaling as in generate_signal
        hist_abs = abs(macd_hist)
        if hist_abs >= 0.05:
            return 0.9
        elif hist_abs >= 0.02:
            return 0.7
        elif hist_abs >= 0.01:
            return 0.5
        else:
            return 0.3

    def get_parameters(self) -> dict[str, Any]:
        """Get MACD signal generator parameters"""
        params = super().get_parameters()
        params.update({
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'min_periods': self.min_periods
        })
        return params