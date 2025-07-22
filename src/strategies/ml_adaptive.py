"""
ML Adaptive Strategy

This strategy improves upon ML Basic to handle extreme market volatility and bear markets.
It incorporates lessons learned from the 2020 COVID crash analysis.

Key Improvements:
- Volatility-based position sizing using ATR
- Market regime detection (trending, ranging, volatile, crisis)
- Dynamic stop loss based on market conditions
- Crisis mode protocols for extreme volatility
- Maximum daily loss limits
- Enhanced risk management
- Multiple timeframe analysis

Ideal for:
- Volatile market conditions
- Bear market protection
- Crisis situations
- Risk-conscious trading
"""

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from strategies.base import BaseStrategy
from indicators.technical import (
    calculate_atr, calculate_rsi, calculate_bollinger_bands,
    calculate_moving_averages, calculate_macd
)
from datetime import datetime, timedelta

class MlAdaptive(BaseStrategy):
    # * Constants for magic numbers
    MIN_PREDICTION_CONFIDENCE = 0.007  # 0.7% minimum predicted move
    SECONDS_PER_DAY = 86400  # * Number of seconds in a day
    LOSS_REDUCTION_FACTOR = 0.2  # * 20% reduction per consecutive loss

    def __init__(self, name="MlAdaptive", model_path="ml/btcusdt_price.onnx", sequence_length=120):
        super().__init__(name)
        
        # Set strategy-specific trading pair - ML model trained on BTC
        self.trading_pair = 'BTCUSDT'
        
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        
        # Adaptive risk management parameters
        self.base_stop_loss_pct = 0.02  # 2% base stop loss
        self.base_take_profit_pct = 0.04  # 4% base take profit
        self.max_stop_loss_pct = 0.08  # 8% maximum stop loss in crisis
        self.min_stop_loss_pct = 0.015  # 1.5% minimum stop loss
        
        # Position sizing parameters
        self.base_position_size = 0.10  # 10% base position
        self.min_position_size = 0.02  # 2% minimum position
        self.max_position_size = 0.15  # 15% maximum position
        
        # Market regime thresholds
        self.volatility_low_threshold = 0.02  # 2% daily volatility
        self.volatility_high_threshold = 0.05  # 5% daily volatility
        self.volatility_crisis_threshold = 0.20  # 20% daily volatility (crisis mode)
        
        # Risk limits
        self.max_daily_loss_pct = 0.05  # 5% maximum daily loss
        self.max_consecutive_losses = 3  # Maximum consecutive losing trades
        self.crisis_mode_cooldown_hours = 24  # Hours to wait after crisis
        
        # Tracking variables
        self.daily_losses = {}
        self.consecutive_losses = 0
        self.last_crisis_time = None
        self.in_crisis_mode = False
        
        # ML confidence thresholds
        # * Lower minimum confidence slightly to allow more frequent trades
        self.min_prediction_confidence = self.MIN_PREDICTION_CONFIDENCE  # 0.7% minimum predicted move
        self.crisis_confidence_multiplier = 2.0  # Double confidence requirement in crisis

        # * Expose a generic take_profit_pct attribute expected by the trading engine
        self.take_profit_pct = self.base_take_profit_pct  # For short positions handled by engine

        # * Rebound detection parameters
        self.rebound_confidence_multiplier = 0.5  # Reduce confidence threshold by 50% on rebounds

        # * Bear market detection parameters
        self.bear_trend_threshold = -0.05  # 5% negative trend strength considered bear

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate technical indicators
        df = calculate_atr(df, period=14)
        df = calculate_moving_averages(df, periods=[20, 50, 200])
        df = calculate_bollinger_bands(df, period=20, std_dev=2.0)
        df = calculate_macd(df)
        df['rsi'] = calculate_rsi(df, period=14)
        
        # Calculate volatility metrics
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()
        df['atr_pct'] = df['atr'] / df['close']  # ATR as percentage of price
        
        # We need trend and bear/rebound information before detecting regime
        # Calculate trend strength
        df['trend_strength'] = (df['close'] - df['ma_50']) / df['ma_50']
        df['trend_direction'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)

        # * Bear market flag – MA50 below MA200 and significant negative trend strength
        df['bear_market'] = np.where(
            (df['ma_50'] < df['ma_200']) & (df['trend_strength'] < self.bear_trend_threshold),
            1,
            0
        )

        # * Rebound signal – previous candle in bear market but current trend turns positive (ma_20 > ma_50)
        df['rebound_signal'] = (
            (df['bear_market'].shift(1) == 1) &  # Previously in bear
            (df['trend_direction'] > 0) &        # Trend flips positive
            (df['close'] > df['ma_50'])           # Price reclaims MA50
        ).astype(int)

        # * Now that we have bear/rebound flags, detect market regime
        df['market_regime'] = self._detect_market_regime(df)
        
        # Normalize price features for ML model
        price_features = ['close', 'volume', 'high', 'low', 'open']
        for feature in price_features:
            if feature in df.columns:
                df[f'{feature}_normalized'] = df[feature].rolling(
                    window=self.sequence_length, min_periods=1
                ).apply(
                    lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0.5,
                    raw=True
                )
        
        # Generate ML predictions
        df['onnx_pred'] = np.nan
        df['prediction_confidence'] = np.nan
        
        for i in range(self.sequence_length, len(df)):
            # Prepare input features
            feature_columns = [f'{feature}_normalized' for feature in price_features]
            input_data = df[feature_columns].iloc[i-self.sequence_length:i].values
            
            # Reshape for ONNX model
            input_data = input_data.astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run prediction
            try:
                output = self.ort_session.run(None, {self.input_name: input_data})
                pred = output[0][0][0]
                
                # Denormalize prediction
                recent_close = df['close'].iloc[i-self.sequence_length:i].values
                min_close = np.min(recent_close)
                max_close = np.max(recent_close)
                
                if max_close != min_close:
                    pred_denormalized = pred * (max_close - min_close) + min_close
                else:
                    pred_denormalized = df['close'].iloc[i-1]
                
                df.at[df.index[i], 'onnx_pred'] = pred_denormalized
                
                # Calculate prediction confidence based on expected return
                current_close = df['close'].iloc[i]
                expected_return = abs((pred_denormalized - current_close) / current_close)
                df.at[df.index[i], 'prediction_confidence'] = expected_return
                
            except Exception as e:
                print(f"Prediction error at index {i}: {e}")
                df.at[df.index[i], 'onnx_pred'] = df['close'].iloc[i-1]
                df.at[df.index[i], 'prediction_confidence'] = 0.0
        
        return df

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime including bear and bull phases"""
        regimes = []
        
        for i in range(len(df)):
            if i < 20:  # Not enough data
                regimes.append('normal')
                continue
            
            volatility = df['volatility_20'].iloc[i] if not pd.isna(df['volatility_20'].iloc[i]) else 0
            atr_pct = df['atr_pct'].iloc[i] if not pd.isna(df['atr_pct'].iloc[i]) else 0
            
            # * Crisis overrides all other regimes
            if volatility > self.volatility_crisis_threshold or atr_pct > 0.15:
                regimes.append('crisis')
                self.in_crisis_mode = True
                self.last_crisis_time = df.index[i]
            # * High volatility
            elif volatility > self.volatility_high_threshold or atr_pct > 0.08:
                regimes.append('volatile')
            # * Recovery phase directly after crisis (within 24h)
            elif self.last_crisis_time and (df.index[i] - self.last_crisis_time).total_seconds() < self.SECONDS_PER_DAY:
                regimes.append('recovery')
            else:
                # * Trend-based classification
                is_bear = (df['bear_market'].iloc[i] == 1)
                if is_bear:
                    regimes.append('bear')
                else:
                    regimes.append('bull')
                self.in_crisis_mode = False
        
        return pd.Series(regimes, index=df.index)

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False
        
        # Check daily loss limit
        current_date = df.index[index].date()
        if current_date in self.daily_losses and self.daily_losses[current_date] <= -self.max_daily_loss_pct:
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=df['close'].iloc[index],
                reasons=['daily_loss_limit_reached']
            )
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=df['close'].iloc[index],
                reasons=['consecutive_loss_limit_reached']
            )
            return False
        
        # Get current conditions
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        regime = df['market_regime'].iloc[index]
        volatility = df['volatility_20'].iloc[index]
        rsi = df['rsi'].iloc[index]
        trend_direction = df['trend_direction'].iloc[index]
        confidence = df['prediction_confidence'].iloc[index]
        
        # Check if we have valid prediction
        if pd.isna(pred) or pd.isna(confidence):
            return False
        
        # Calculate predicted return
        predicted_return = (pred - close) / close if close > 0 else 0
        
        # * Base confidence threshold adjustments
        min_confidence = self.min_prediction_confidence
        if regime == 'crisis':
            min_confidence *= self.crisis_confidence_multiplier
        elif regime == 'volatile':
            min_confidence *= 1.5
        elif regime == 'bull':
            # * Be more aggressive in confirmed bull runs
            min_confidence *= 0.75  # Reduce threshold by 25 %

        # * Rebound – be more aggressive (lower confidence threshold)
        rebound = df['rebound_signal'].iloc[index] == 1
        if rebound:
            min_confidence *= self.rebound_confidence_multiplier
        
        # Entry conditions
        entry_conditions = [
            pred > close,  # Positive prediction (expecting price rise)
            confidence >= min_confidence,  # Sufficient confidence
            regime not in ['crisis', 'bear'] or confidence >= min_confidence * 2,  # Extra caution in crisis/bear
            rsi < 70,  # Not overbought
            (trend_direction > 0) or rebound or regime in ['bull', 'recovery'],  # Positive trend or rebound/bull
        ]
        
        # Additional filters for volatile markets
        if regime in ['volatile', 'crisis']:
            entry_conditions.extend([
                close > df['bb_lower'].iloc[index],  # Above lower Bollinger Band
                df['macd_hist'].iloc[index] > 0,  # MACD histogram positive
            ])
        
        entry_signal = all(entry_conditions)
        
        # Log decision
        ml_predictions = {
            'raw_prediction': pred,
            'current_price': close,
            'predicted_return': predicted_return,
            'confidence': confidence
        }
        
        market_conditions = {
            'regime': regime,
            'volatility': volatility,
            'rsi': rsi,
            'trend_direction': trend_direction
        }
        
        reasons = [
            f'regime_{regime}',
            f'predicted_return_{predicted_return:.4f}',
            f'confidence_{confidence:.4f}',
            f'volatility_{volatility:.4f}',
            'entry_signal_met' if entry_signal else 'entry_conditions_not_met'
        ]
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if entry_signal else 'no_action',
            price=close,
            signal_strength=confidence if entry_signal else 0.0,
            confidence_score=min(1.0, confidence * 10),
            ml_predictions=ml_predictions,
            volatility=volatility,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_adaptive',
                'min_confidence_threshold': min_confidence,
                'crisis_mode': self.in_crisis_mode,
                'market_conditions': market_conditions
            }
        )
        
        return entry_signal

    # * ------------------------------------------------------------------
    # * Short entry conditions for bear markets
    # * ------------------------------------------------------------------
    def check_short_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False

        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        regime = df['market_regime'].iloc[index]
        rsi = df['rsi'].iloc[index]
        trend_direction = df['trend_direction'].iloc[index]

        # * Validate prediction availability
        if pd.isna(pred):
            return False

        # * Expected downward move for short
        predicted_return = (close - pred) / close if close > 0 else 0
        confidence = predicted_return  # Use magnitude as confidence proxy

        # * Minimum confidence requirement (reuse long threshold)
        min_confidence = self.min_prediction_confidence
        if regime in ['crisis', 'volatile']:
            min_confidence *= 1.5

        # * Entry rules
        entry_conditions = [
            pred < close,               # Model predicts lower price
            confidence >= min_confidence,  # Enough expected drop
            regime in ['bear', 'crisis', 'volatile'],  # Only short in non-bull phases
            rsi > 30,                   # Not oversold
            trend_direction < 0,        # Downward trend
        ]

        short_entry = all(entry_conditions)

        # * Simple logging (re-use BaseStrategy logger)
        self.log_execution(
            signal_type='entry_short',
            action_taken='entry_signal' if short_entry else 'no_action',
            price=close,
            signal_strength=confidence if short_entry else 0.0,
            confidence_score=min(1.0, confidence * 10),
            reasons=[
                f'regime_{regime}',
                f'predicted_return_{-predicted_return:.4f}',  # Negative means drop
                'short_entry_signal_met' if short_entry else 'short_entry_conditions_not_met'
            ],
            additional_context={'model_type': 'ml_adaptive'}
        )

        return short_entry

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False
        
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        regime = df['market_regime'].iloc[index]
        
        # Get dynamic stop loss and take profit
        stop_loss = self._get_dynamic_stop_loss(df, index, regime)
        take_profit = self._get_dynamic_take_profit(df, index, regime)
        
        # Exit conditions
        hit_stop_loss = returns <= -stop_loss
        hit_take_profit = returns >= take_profit
        
        # Additional exit conditions for risk management
        emergency_exit = False
        if regime == 'crisis' and returns < -stop_loss * 0.5:
            emergency_exit = True  # Exit early in crisis if losing
        
        # Update tracking
        if hit_stop_loss or emergency_exit:
            self.consecutive_losses += 1
            current_date = df.index[index].date()
            if current_date not in self.daily_losses:
                self.daily_losses[current_date] = 0
            self.daily_losses[current_date] += returns
        elif hit_take_profit:
            self.consecutive_losses = 0  # Reset on winning trade
        
        return hit_stop_loss or hit_take_profit or emergency_exit

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get current market conditions
        regime = df['market_regime'].iloc[index]
        volatility = df['volatility_20'].iloc[index] if not pd.isna(df['volatility_20'].iloc[index]) else 0.02
        atr_pct = df['atr_pct'].iloc[index] if not pd.isna(df['atr_pct'].iloc[index]) else 0.02
        confidence = df['prediction_confidence'].iloc[index] if not pd.isna(df['prediction_confidence'].iloc[index]) else 0.0
        
        # Base position size
        position_size = self.base_position_size
        
        # Adjust for volatility (inverse relationship)
        volatility_factor = self.volatility_low_threshold / max(volatility, 0.01)
        volatility_factor = np.clip(volatility_factor, 0.5, 1.5)
        position_size *= volatility_factor
        
        # Adjust for market regime
        regime_multipliers = {
            'normal': 1.0,
            'bull': 1.2,  # Increase size in bull markets
            'volatile': 0.5,
            'crisis': 0.25,
            'recovery': 0.75,
            'bear': 0.75  # Slightly reduce size in bear markets
        }
        position_size *= regime_multipliers.get(regime, 1.0)
        
        # Adjust for prediction confidence
        confidence_factor = min(1.0, confidence / self.min_prediction_confidence)
        position_size *= confidence_factor
        
        # Apply limits
        position_size = np.clip(position_size, self.min_position_size, self.max_position_size)
        
        # Further reduce if consecutive losses
        if self.consecutive_losses > 0:
            position_size *= (1 - self.LOSS_REDUCTION_FACTOR * self.consecutive_losses)  # Apply reduction factor
        
        return balance * position_size

    def _get_dynamic_stop_loss(self, df: pd.DataFrame, index: int, regime: str) -> float:
        """Calculate dynamic stop loss based on market conditions"""
        atr_pct = df['atr_pct'].iloc[index] if not pd.isna(df['atr_pct'].iloc[index]) else 0.02
        
        # Base stop loss on ATR
        stop_loss = max(self.base_stop_loss_pct, atr_pct * 1.5)
        
        # Adjust for market regime
        if regime == 'crisis':
            stop_loss = min(self.max_stop_loss_pct, stop_loss * 2.0)
        elif regime == 'volatile':
            stop_loss = min(self.max_stop_loss_pct * 0.75, stop_loss * 1.5)
        
        # Apply limits
        stop_loss = np.clip(stop_loss, self.min_stop_loss_pct, self.max_stop_loss_pct)
        
        return stop_loss

    def _get_dynamic_take_profit(self, df: pd.DataFrame, index: int, regime: str) -> float:
        """Calculate dynamic take profit based on market conditions"""
        atr_pct = df['atr_pct'].iloc[index] if not pd.isna(df['atr_pct'].iloc[index]) else 0.02
        
        # Base take profit on ATR
        take_profit = max(self.base_take_profit_pct, atr_pct * 3.0)
        
        # Adjust for market regime
        if regime == 'crisis':
            take_profit *= 0.75  # Take profits earlier in crisis
        elif regime == 'volatile':
            take_profit *= 1.25  # Allow more room in volatile markets
        
        return take_profit

    def calculate_stop_loss(self, df, index, price, side: str = 'long') -> float:
        """Calculate stop loss price (handle enum or string)"""
        # * Normalize side to string value ('long' / 'short') to handle PositionSide enum
        side_str = side.value if hasattr(side, 'value') else str(side)

        regime = df['market_regime'].iloc[index] if 'market_regime' in df.columns else 'normal'
        stop_loss_pct = self._get_dynamic_stop_loss(df, index, regime)
        
        if side_str == 'long':
            return price * (1 - stop_loss_pct)
        else:  # short
            return price * (1 + stop_loss_pct)

    def get_parameters(self) -> dict:
        return {
            'name': self.name,
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'base_stop_loss_pct': self.base_stop_loss_pct,
            'base_take_profit_pct': self.base_take_profit_pct,
            'take_profit_pct': self.take_profit_pct,
            'base_position_size': self.base_position_size,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'volatility_thresholds': {
                'low': self.volatility_low_threshold,
                'high': self.volatility_high_threshold,
                'crisis': self.volatility_crisis_threshold
            }
        }
    
    def _load_model(self):
        """Load or reload the ONNX model"""
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Failed to load model {self.model_path}: {e}")
            raise