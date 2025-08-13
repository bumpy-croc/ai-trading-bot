"""
ML Basic Strategy

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Normalized price inputs for better model performance
- 2% stop loss, 4% take profit risk management
- No external API dependencies

Ideal for:
- Consistent, reliable trading signals
- Backtesting historical periods
- Environments where sentiment data is unavailable
- Simple deployment scenarios
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from src.strategies.base import BaseStrategy
from src.config.feature_flags import is_enabled
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.technical import TechnicalFeatureExtractor
from typing import Optional
from src.prediction import PredictionEngine, PredictionConfig
from src.config import get_config
from src.config.constants import (
    DEFAULT_USE_PREDICTION_ENGINE
)
from pathlib import Path
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
import os


class MlBasic(BaseStrategy):
    # * Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 10  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.1  # Base position size (10% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.2  # Maximum position size (20% of balance)
    # Regime and volatility controls
    LONG_TREND_FILTER_PERIOD = 200  # Only take longs when above MA200
    VOL_FULL_SIZE_ATR_PCT = 0.02    # <=2% ATR pct allows full base sizing
    VOL_SIZE_DECAY = 4.0            # Decay factor for size as volatility rises
    BEAR_SIZE_MULTIPLIER = 0.5      # Halve sizing in bear regime
    MIN_CONFIDENCE_TO_TRADE = 0.05  # Skip very low-confidence signals
    # Exit tuning
    UNFAVORABLE_PRED_EXIT_BULL = -0.02   # -2% predicted return threshold in bull
    UNFAVORABLE_PRED_EXIT_BEAR = -0.005  # -0.5% threshold in bear
    BEAR_DEFENSIVE_EXIT_DD = -0.01       # Exit if open loss worse than -1% in bear
    # Adaptive stops
    STOP_LOSS_PCT_BEAR = 0.015  # 1.5% stop in bear regime
    # Volatility cutoff
    EXTREME_ATR_THRESHOLD = 0.08  # Avoid entries when ATR% exceeds this
    
    def __init__(self, name="MlBasic", model_path="src/ml/btcusdt_price.onnx", sequence_length=120, use_prediction_engine: Optional[bool] = None, model_name: Optional[str] = None):
        super().__init__(name)
        
        # Set strategy-specific trading pair - ML model trained on BTC
        self.trading_pair = 'BTCUSDT'
        
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.stop_loss_pct = 0.02  # 2% stop loss (bull/default)
        self.take_profit_pct = 0.04  # 4% take profit

        # Optional prediction engine integration (disabled by default to preserve behavior)
        cfg = get_config()
        self.use_prediction_engine = (
            use_prediction_engine
            if use_prediction_engine is not None
            else cfg.get_bool('USE_PREDICTION_ENGINE', default=DEFAULT_USE_PREDICTION_ENGINE)
        )
        # Prefer explicit, then config, then fallback to stem of ONNX path to match prior behavior
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get('PREDICTION_ENGINE_MODEL_NAME', default=None)
        if self.model_name is None:
            try:
                self.model_name = Path(self.model_path).stem
            except Exception:
                self.model_name = None
        self.prediction_engine = None
        self._engine_warning_emitted = False
        # Optional batch inference flag (default off to preserve exact behavior)
        self.use_engine_batch = get_config().get_bool('ENGINE_BATCH_INFERENCE', default=False)

        # Allow configuration overrides for key tunables
        try:
            self.LONG_TREND_FILTER_PERIOD = int(cfg.get('MLBASIC_LONG_TREND_FILTER_PERIOD', default=self.LONG_TREND_FILTER_PERIOD))
        except Exception:
            pass
        try:
            self.MIN_CONFIDENCE_TO_TRADE = float(cfg.get('MLBASIC_MIN_CONFIDENCE_TO_TRADE', default=self.MIN_CONFIDENCE_TO_TRADE))
        except Exception:
            pass
        try:
            self.BEAR_SIZE_MULTIPLIER = float(cfg.get('MLBASIC_BEAR_SIZE_MULTIPLIER', default=self.BEAR_SIZE_MULTIPLIER))
        except Exception:
            pass
        try:
            self.UNFAVORABLE_PRED_EXIT_BULL = float(cfg.get('MLBASIC_UNFAV_PRED_EXIT_BULL', default=self.UNFAVORABLE_PRED_EXIT_BULL))
            self.UNFAVORABLE_PRED_EXIT_BEAR = float(cfg.get('MLBASIC_UNFAV_PRED_EXIT_BEAR', default=self.UNFAVORABLE_PRED_EXIT_BEAR))
        except Exception:
            pass
        try:
            self.BEAR_DEFENSIVE_EXIT_DD = float(cfg.get('MLBASIC_BEAR_DEFENSIVE_EXIT_DD', default=self.BEAR_DEFENSIVE_EXIT_DD))
        except Exception:
            pass
        try:
            self.STOP_LOSS_PCT_BEAR = float(cfg.get('MLBASIC_STOP_LOSS_PCT_BEAR', default=self.STOP_LOSS_PCT_BEAR))
        except Exception:
            pass
        try:
            self.VOL_FULL_SIZE_ATR_PCT = float(cfg.get('MLBASIC_VOL_FULL_SIZE_ATR_PCT', default=self.VOL_FULL_SIZE_ATR_PCT))
            self.VOL_SIZE_DECAY = float(cfg.get('MLBASIC_VOL_SIZE_DECAY', default=self.VOL_SIZE_DECAY))
        except Exception:
            pass
        try:
            self.EXTREME_ATR_THRESHOLD = float(cfg.get('MLBASIC_EXTREME_ATR_THRESHOLD', default=self.EXTREME_ATR_THRESHOLD))
        except Exception:
            pass

        # Initialize feature pipeline with a technical extractor matching our normalization window
        technical_extractor = TechnicalFeatureExtractor(
            sequence_length=self.sequence_length,
            normalization_window=self.sequence_length
        )
        # Disable default technical extractor to avoid duplicate; use our custom ones
        if self.use_prediction_engine:
            # When engine is enabled, include price-only extractor for model input and technical extractor for regime/risk
            price_only = PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
            self.feature_pipeline = FeaturePipeline(
                enable_technical=False,
                enable_sentiment=False,
                enable_market_microstructure=False,
                custom_extractors=[price_only, technical_extractor]
            )
        else:
            self.feature_pipeline = FeaturePipeline(
                enable_technical=False,
                enable_sentiment=False,
                enable_market_microstructure=False,
                custom_extractors=[technical_extractor]
            )

        # Early health check for engine (non-fatal)
        if self.use_prediction_engine:
            try:
                config = PredictionConfig.from_config_manager()
                config.enable_sentiment = False
                config.enable_market_microstructure = False
                engine = PredictionEngine(config)
                # Ensure price-only extractor
                engine.feature_pipeline = FeaturePipeline(
                    enable_technical=False,
                    enable_sentiment=False,
                    enable_market_microstructure=False,
                    custom_extractors=[PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)]
                )
                health = engine.health_check()
                if health.get('status') != 'healthy' and not self._engine_warning_emitted:
                    print(f"[MlBasic] Prediction engine health degraded: {health}")
                    self._engine_warning_emitted = True
                self.prediction_engine = engine
            except Exception as _e:
                if not self._engine_warning_emitted:
                    print("[MlBasic] Prediction engine initialization failed; falling back to local ONNX session.")
                    self._engine_warning_emitted = True
                self.prediction_engine = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Use the prediction feature pipeline to generate normalized price features identically
        df = self.feature_pipeline.transform(df)
        
        # Prepare predictions columns
        df['onnx_pred'] = np.nan
        df['ml_prediction'] = np.nan
        df['prediction_confidence'] = np.nan
        df['engine_direction'] = np.nan
        df['engine_confidence'] = np.nan
        # Ensure regime helpers exist when technicals are available
        if f'ma_{self.LONG_TREND_FILTER_PERIOD}' in df.columns:
            df['is_bull_regime'] = (df['close'] >= df[f'ma_{self.LONG_TREND_FILTER_PERIOD}']).astype(int)
        else:
            df['is_bull_regime'] = 1  # default to permissive if not available
        
        price_features = ['close', 'volume', 'high', 'low', 'open']
        
        # Decide whether to use engine or ONNX locally
        use_engine_flag = is_enabled("use_prediction_engine", default=False) or bool(self.use_prediction_engine)
        if use_engine_flag and self.prediction_engine is None:
            # Lazy engine init
            try:
                config = PredictionConfig.from_config_manager()
                config.enable_sentiment = False
                config.enable_market_microstructure = False
                engine = PredictionEngine(config)
                engine.feature_pipeline = FeaturePipeline(
                    enable_technical=False,
                    enable_sentiment=False,
                    enable_market_microstructure=False,
                    custom_extractors=[PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)]
                )
                self.prediction_engine = engine
            except Exception:
                self.prediction_engine = None

        # Prefer batched, denormalized predictions when using the engine to ensure price-scale outputs
        used_engine_series = False
        if use_engine_flag and self.prediction_engine is not None and self.model_name:
            try:
                series = self.prediction_engine.predict_series(
                    df[['open', 'high', 'low', 'close', 'volume']],
                    model_name=self.model_name,
                    batch_size=1024,
                    return_denormalized=True,
                    sequence_length_override=self.sequence_length
                )
                indices = series.get('indices')
                preds = series.get('preds')
                if indices is not None and preds is not None and len(indices) == len(preds):
                    for idx, pred_denormalized in zip(indices, preds):
                        df.at[df.index[int(idx)], 'onnx_pred'] = float(pred_denormalized)
                        df.at[df.index[int(idx)], 'ml_prediction'] = float(pred_denormalized)
                        close_i = float(df['close'].iloc[int(idx)])
                        if close_i > 0:
                            predicted_return = abs(float(pred_denormalized) - close_i) / close_i
                            confidence = min(1.0, predicted_return * self.CONFIDENCE_MULTIPLIER)
                            df.at[df.index[int(idx)], 'prediction_confidence'] = confidence
                    used_engine_series = True
            except Exception as _e:
                used_engine_series = False

        # Fallback: per-row prediction (engine single-step or local ONNX)
        if not used_engine_series:
            for i in range(self.sequence_length, len(df)):
                # Prepare input features
                feature_columns = [f'{feature}_normalized' for feature in price_features]
                input_data = df[feature_columns].iloc[i-self.sequence_length:i].values

                # Reshape for ONNX model: (batch_size, sequence_length, features)
                input_data = input_data.astype(np.float32)
                input_data = np.expand_dims(input_data, axis=0)

                try:
                    if use_engine_flag and self.prediction_engine is not None and self.model_name:
                        # Engine single-step returns normalized scalar; denormalize using prior window close range
                        result = self.prediction_engine.predict(
                            df[['open', 'high', 'low', 'close', 'volume']].iloc[i-self.sequence_length:i],
                            model_name=self.model_name
                        )
                        # Denormalize using previous window on close
                        recent_close = df['close'].iloc[i-self.sequence_length:i].values
                        min_close = np.min(recent_close)
                        max_close = np.max(recent_close)
                        raw_pred = float(result.price)
                        if max_close != min_close:
                            pred_denormalized = raw_pred * (max_close - min_close) + min_close
                        else:
                            pred_denormalized = df['close'].iloc[i-1]
                    else:
                        output = self.ort_session.run(None, {self.input_name: input_data})
                        pred = float(output[0][0][0])
                        recent_close = df['close'].iloc[i-self.sequence_length:i].values
                        min_close = np.min(recent_close)
                        max_close = np.max(recent_close)
                        if max_close != min_close:
                            pred_denormalized = pred * (max_close - min_close) + min_close
                        else:
                            pred_denormalized = df['close'].iloc[i-1]

                    df.at[df.index[i], 'onnx_pred'] = pred_denormalized
                    df.at[df.index[i], 'ml_prediction'] = pred_denormalized

                    close_i = df['close'].iloc[i]
                    if close_i > 0:
                        predicted_return = abs(pred_denormalized - close_i) / close_i
                        confidence = min(1.0, predicted_return * self.CONFIDENCE_MULTIPLIER)
                        df.at[df.index[i], 'prediction_confidence'] = confidence

                except Exception as e:
                    print(f"Prediction error at index {i}: {e}")
                    fallback_price = df['close'].iloc[i-1]
                    df.at[df.index[i], 'onnx_pred'] = fallback_price
                    df.at[df.index[i], 'ml_prediction'] = fallback_price
                    df.at[df.index[i], 'prediction_confidence'] = np.nan
 
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Go long if the predicted price for the next bar is higher than the current close
        if index < 1 or index >= len(df):
            return False
        
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        # Regime and volatility context
        ma_col = f'ma_{self.LONG_TREND_FILTER_PERIOD}'
        in_bull_regime = True
        if ma_col in df.columns and not pd.isna(df[ma_col].iloc[index]):
            in_bull_regime = close >= float(df[ma_col].iloc[index])
        atr_pct = float(df['atr_pct'].iloc[index]) if 'atr_pct' in df.columns and not pd.isna(df['atr_pct'].iloc[index]) else None
        
        # Check if we have a valid prediction
        if pd.isna(pred):
            # Log the missing prediction
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=close,
                reasons=['missing_ml_prediction'],
                additional_context={'prediction_available': False}
            )
            return False
        
        # Calculate predicted return
        predicted_return = (pred - close) / close if close > 0 else 0
        
        # Determine entry signal
        entry_signal = (pred > close)
        # Apply trend filter: avoid longs in bear regime
        if not in_bull_regime:
            entry_signal = False
        # Skip extremely low-confidence signals
        conf_val = df['prediction_confidence'].iloc[index] if 'prediction_confidence' in df.columns else np.nan
        if not pd.isna(conf_val) and conf_val < self.MIN_CONFIDENCE_TO_TRADE:
            entry_signal = False
        # Avoid entries during extreme volatility
        if atr_pct is not None and atr_pct > self.EXTREME_ATR_THRESHOLD:
            entry_signal = False
        
        # Log the decision process
        ml_predictions = {
            'raw_prediction': pred,
            'current_price': close,
            'predicted_return': predicted_return
        }
        # Engine metadata for auditability
        if self.use_prediction_engine and self.prediction_engine is not None:
            ml_predictions.update({
                'engine_enabled': True,
                'engine_model_name': self.model_name,
                'engine_batch': self.use_engine_batch,
            })
        
        reasons = [
            f'predicted_return_{predicted_return:.4f}',
            f'prediction_{pred:.2f}_vs_current_{close:.2f}',
            'entry_signal_met' if entry_signal else 'entry_signal_not_met'
        ]
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if entry_signal else 'no_action',
            price=close,
            signal_strength=abs(predicted_return) if entry_signal else 0.0,
            confidence_score=float(df['prediction_confidence'].iloc[index]) if 'prediction_confidence' in df.columns and not pd.isna(df['prediction_confidence'].iloc[index]) else min(1.0, abs(predicted_return) * 10),
            ml_predictions=ml_predictions,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_basic',
                'sequence_length': self.sequence_length,
                'prediction_available': True
            }
        )
        
        return entry_signal

    def check_short_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        if pd.isna(pred):
            return False
        predicted_return = (pred - close) / close if close > 0 else 0
        return predicted_return < self.SHORT_ENTRY_THRESHOLD

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        
        # * Basic exit conditions (stop loss and take profit)
        basic_exit = hit_stop_loss or hit_take_profit
        
        # * ML-based exit signal for unfavorable predictions
        pred = df['onnx_pred'].iloc[index]
        # Determine regime
        ma_col = f'ma_{self.LONG_TREND_FILTER_PERIOD}'
        in_bull_regime = True
        if ma_col in df.columns and not pd.isna(df[ma_col].iloc[index]):
            in_bull_regime = current_price >= float(df[ma_col].iloc[index])

        if not pd.isna(pred):
            # For long positions: exit earlier in bear regime on unfavorable prediction
            predicted_return_next = (pred - current_price) / current_price if current_price > 0 else 0
            unfavorable_threshold = self.UNFAVORABLE_PRED_EXIT_BULL if in_bull_regime else self.UNFAVORABLE_PRED_EXIT_BEAR
            significant_unfavorable_prediction = predicted_return_next < unfavorable_threshold

            # Defensive technical exit in bear: MA20 below MA50 and price below MA20
            bear_defensive = False
            if not in_bull_regime and 'ma_20' in df.columns and 'ma_50' in df.columns:
                ma20 = df['ma_20'].iloc[index]
                ma50 = df['ma_50'].iloc[index]
                if not pd.isna(ma20) and not pd.isna(ma50):
                    bear_defensive = (ma20 < ma50) and (current_price < ma20)

            # Also cut early if open loss exceeds small threshold in bear regime
            bear_small_dd_exit = (not in_bull_regime) and (returns <= self.BEAR_DEFENSIVE_EXIT_DD)

            return basic_exit or significant_unfavorable_prediction or bear_defensive or bear_small_dd_exit
        
        return basic_exit

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if index >= len(df) or balance <= 0:
            return 0.0
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        if pd.isna(pred):
            return 0.0
        predicted_return = abs(pred - close) / close if close > 0 else 0
        confidence = min(1.0, predicted_return * self.CONFIDENCE_MULTIPLIER)
        # Volatility-aware penalty: reduce size as ATR% rises above comfort level
        atr_pct = float(df['atr_pct'].iloc[index]) if 'atr_pct' in df.columns and not pd.isna(df['atr_pct'].iloc[index]) else self.VOL_FULL_SIZE_ATR_PCT
        excess_vol = max(0.0, atr_pct - self.VOL_FULL_SIZE_ATR_PCT)
        vol_penalty = 1.0 / (1.0 + self.VOL_SIZE_DECAY * excess_vol)
        # Regime penalty for bear
        ma_col = f'ma_{self.LONG_TREND_FILTER_PERIOD}'
        in_bull_regime = True
        if ma_col in df.columns and not pd.isna(df[ma_col].iloc[index]):
            in_bull_regime = close >= float(df[ma_col].iloc[index])
        regime_multiplier = 1.0 if in_bull_regime else self.BEAR_SIZE_MULTIPLIER
        dynamic_size_ratio = self.BASE_POSITION_SIZE * confidence * vol_penalty * regime_multiplier
        capped_ratio = max(self.MIN_POSITION_SIZE_RATIO, min(self.MAX_POSITION_SIZE_RATIO, dynamic_size_ratio))
        return capped_ratio * balance

    def get_parameters(self) -> dict:
        return {
            'name': self.name,
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'use_prediction_engine': self.use_prediction_engine,
            'engine_model_name': self.model_name,
            'bear_stop_loss_pct': self.STOP_LOSS_PCT_BEAR,
            'vol_full_size_atr_pct': self.VOL_FULL_SIZE_ATR_PCT,
            'vol_size_decay': self.VOL_SIZE_DECAY,
            'bear_size_multiplier': self.BEAR_SIZE_MULTIPLIER,
        }

    def calculate_stop_loss(self, df, index, price, side) -> float:
        """Calculate stop loss price"""
        # * Handle both string and enum inputs for backward compatibility
        side_str = side.value if hasattr(side, 'value') else str(side)
        # Determine regime at the given index for adaptive stops
        in_bull_regime = True
        try:
            ma_col = f'ma_{self.LONG_TREND_FILTER_PERIOD}'
            if ma_col in df.columns and not pd.isna(df[ma_col].iloc[index]):
                in_bull_regime = df['close'].iloc[index] >= float(df[ma_col].iloc[index])
        except Exception:
            in_bull_regime = True

        sl_pct = self.stop_loss_pct if in_bull_regime else self.STOP_LOSS_PCT_BEAR
        if side_str == 'long':
            return price * (1 - sl_pct)
        else:  # short
            return price * (1 + sl_pct)
    
    def _load_model(self):
        """Load or reload the ONNX model"""
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Failed to load model {self.model_path}: {e}")
            raise 