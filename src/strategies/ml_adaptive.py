"""
ML Adaptive Strategy

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

from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_USE_PREDICTION_ENGINE
from src.prediction import PredictionConfig, PredictionEngine
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.regime.detector import RegimeDetector, TrendLabel, VolLabel
from src.strategies.base import BaseStrategy


class MlAdaptive(BaseStrategy):
    # * Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries (base threshold)
    CONFIDENCE_MULTIPLIER = 12  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.2  # Base position size (20% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.25  # Maximum position size (25% of balance)
    
    # * Dynamic short entry threshold configuration
    # Base thresholds for different market regimes (more aggressive to match original performance)
    SHORT_THRESHOLD_TREND_UP = -0.0003  # Less conservative in uptrend (-0.03%)
    SHORT_THRESHOLD_TREND_DOWN = -0.0007  # More conservative in downtrend (-0.07%)
    SHORT_THRESHOLD_RANGE = -0.0005  # Standard threshold in range-bound market (-0.05%)
    SHORT_THRESHOLD_HIGH_VOL = -0.0004  # Less conservative in high volatility (-0.04%)
    SHORT_THRESHOLD_LOW_VOL = -0.0006  # More conservative in low volatility (-0.06%)
    # Confidence-based adjustment (reduced impact)
    SHORT_THRESHOLD_CONFIDENCE_MULTIPLIER = 0.2  # Adjust threshold based on regime confidence

    def __init__(
        self,
        name="MlAdaptive",
        model_path="src/ml/btcusdt_price.onnx",
        sequence_length=120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
        enable_short_selling: bool = False,
    ):
        super().__init__(name, enable_short_selling)

        # Set strategy-specific trading pair - ML model trained on BTC
        self.trading_pair = "BTCUSDT"

        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit

        # Optional prediction engine integration (disabled by default to preserve behavior)
        cfg = get_config()
        self.use_prediction_engine = (
            use_prediction_engine
            if use_prediction_engine is not None
            else cfg.get_bool("USE_PREDICTION_ENGINE", default=DEFAULT_USE_PREDICTION_ENGINE)
        )
        # Prefer explicit, then config, then fallback to stem of ONNX path to match prior behavior
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)
        if self.model_name is None:
            try:
                self.model_name = Path(self.model_path).stem
            except Exception:
                self.model_name = None
        self.prediction_engine = None
        self._engine_warning_emitted = False
        # Optional batch inference flag (default off to preserve exact behavior)
        self.use_engine_batch = get_config().get_bool("ENGINE_BATCH_INFERENCE", default=False)
        
        # Initialize regime detector for dynamic threshold adjustment
        self.regime_detector = RegimeDetector()

        # Initialize feature pipeline with a technical extractor matching our normalization window
        technical_extractor = TechnicalFeatureExtractor(
            sequence_length=self.sequence_length, normalization_window=self.sequence_length
        )
        # Disable default technical extractor to avoid duplicate; use our custom one
        if self.use_prediction_engine:
            # When engine is enabled, use a price-only extractor to guarantee 5 features in expected order
            price_only = PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
            config = {
                "technical_features": {"enabled": False},
                "sentiment_features": {"enabled": False},
                "market_features": {"enabled": False},
                "price_only_features": {"enabled": False},
            }
            self.feature_pipeline = FeaturePipeline(
                config=config,
                custom_extractors=[price_only],
            )
        else:
            config = {
                "technical_features": {"enabled": False},
                "sentiment_features": {"enabled": False},
                "market_features": {"enabled": False},
                "price_only_features": {"enabled": False},
            }
            self.feature_pipeline = FeaturePipeline(
                config=config,
                custom_extractors=[technical_extractor],
            )

        # Early health check for engine (non-fatal)
        if self.use_prediction_engine:
            try:
                config = PredictionConfig.from_config_manager()
                config.enable_sentiment = False
                config.enable_market_microstructure = False
                engine = PredictionEngine(config)
                # Ensure price-only extractor
                config = {
                    "technical_features": {"enabled": False},
                    "sentiment_features": {"enabled": False},
                    "market_features": {"enabled": False},
                    "price_only_features": {"enabled": False},
                }
                engine.feature_pipeline = FeaturePipeline(
                    config=config,
                    custom_extractors=[
                        PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
                    ],
                )
                health = engine.health_check()
                if health.get("status") != "healthy" and not self._engine_warning_emitted:
                    print(f"[MlAdaptive] Prediction engine health degraded: {health}")
                    self._engine_warning_emitted = True
                self.prediction_engine = engine
            except Exception as _e:
                if not self._engine_warning_emitted:
                    print(
                        "[MlAdaptive] Prediction engine initialization failed; falling back to local ONNX session."
                    )
                    self._engine_warning_emitted = True
                self.prediction_engine = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # * Gate ML predictions using the instance configuration (stable across tests)
        use_prediction_engine = bool(self.use_prediction_engine)

        # Use the prediction feature pipeline to generate normalized price features identically
        df = self.feature_pipeline.transform(df)
        
        # * Add regime detection for dynamic threshold adjustment
        try:
            df = self.regime_detector.annotate(df)
        except Exception as e:
            # If regime detection fails, continue without it
            print(f"[MlAdaptive] Regime detection failed: {e}")
            # Add default regime columns to prevent errors
            df["trend_label"] = "range"
            df["vol_label"] = "low_vol"
            df["regime_confidence"] = 0.5

        # Prepare predictions columns
        df["onnx_pred"] = np.nan
        df["ml_prediction"] = np.nan
        df["prediction_confidence"] = np.nan
        df["engine_direction"] = np.nan
        df["engine_confidence"] = np.nan

        price_features = ["close", "volume", "high", "low", "open"]

        # Generate predictions for each row that has enough history.
        # Always compute predictions; choose engine or local ONNX per configuration.
        # Lazy engine init if needed
        if use_prediction_engine and self.prediction_engine is None:
            try:
                config = PredictionConfig.from_config_manager()
                config.enable_sentiment = False
                config.enable_market_microstructure = False
                engine = PredictionEngine(config)
                engine.feature_pipeline = FeaturePipeline(
                    enable_technical=False,
                    enable_sentiment=False,
                    enable_market_microstructure=False,
                    custom_extractors=[
                        PriceOnlyFeatureExtractor(normalization_window=self.sequence_length)
                    ],
                )
                self.prediction_engine = engine
            except Exception:
                self.prediction_engine = None

        for i in range(self.sequence_length, len(df)):
            # Prepare input features
            feature_columns = [f"{feature}_normalized" for feature in price_features]
            input_data = df[feature_columns].iloc[i - self.sequence_length : i].values

            # Reshape for ONNX model: (batch_size, sequence_length, features)
            input_data = input_data.astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)

            try:
                if use_prediction_engine and self.prediction_engine is not None and self.model_name:
                    window_df = df[["open", "high", "low", "close", "volume"]].iloc[
                        i - self.sequence_length : i
                    ]
                    result = self.prediction_engine.predict(window_df, model_name=self.model_name)
                    pred = float(result.price)
                else:
                    output = self.ort_session.run(None, {self.input_name: input_data})
                    pred = output[0][0][0]

                recent_close = df["close"].iloc[i - self.sequence_length : i].values
                min_close = np.min(recent_close)
                max_close = np.max(recent_close)

                if max_close != min_close:
                    pred_denormalized = pred * (max_close - min_close) + min_close
                else:
                    pred_denormalized = df["close"].iloc[i - 1]

                df.at[df.index[i], "onnx_pred"] = pred_denormalized
                df.at[df.index[i], "ml_prediction"] = pred_denormalized

                close_i = df["close"].iloc[i]
                if close_i > 0:
                    predicted_return = abs(pred_denormalized - close_i) / close_i
                    confidence = min(1.0, predicted_return * self.CONFIDENCE_MULTIPLIER)
                    df.at[df.index[i], "prediction_confidence"] = confidence

            except Exception as e:
                print(f"Prediction error at index {i}: {e}")
                fallback_price = df["close"].iloc[i - 1]
                df.at[df.index[i], "onnx_pred"] = fallback_price
                df.at[df.index[i], "ml_prediction"] = fallback_price
                df.at[df.index[i], "prediction_confidence"] = np.nan

        return df

    def _calculate_dynamic_short_threshold(self, df: pd.DataFrame, index: int) -> float:
        """
        Calculate dynamic short entry threshold based on current market regime.
        
        Args:
            df: DataFrame with regime annotations
            index: Current index in the DataFrame
            
        Returns:
            Dynamic threshold for short entries
        """
        if index >= len(df) or index < 0:
            return self.SHORT_ENTRY_THRESHOLD
            
        # Get current regime information
        trend_label = df["trend_label"].iloc[index] if "trend_label" in df.columns else "range"
        vol_label = df["vol_label"].iloc[index] if "vol_label" in df.columns else "low_vol"
        confidence = df["regime_confidence"].iloc[index] if "regime_confidence" in df.columns else 0.5
        
        # Start with base threshold based on trend
        if trend_label == TrendLabel.TREND_UP.value:
            base_threshold = self.SHORT_THRESHOLD_TREND_UP
        elif trend_label == TrendLabel.TREND_DOWN.value:
            base_threshold = self.SHORT_THRESHOLD_TREND_DOWN
        else:  # range
            base_threshold = self.SHORT_THRESHOLD_RANGE
            
        # Adjust for volatility
        if vol_label == VolLabel.HIGH.value:
            vol_adjustment = self.SHORT_THRESHOLD_HIGH_VOL
        else:  # low_vol
            vol_adjustment = self.SHORT_THRESHOLD_LOW_VOL
            
        # Combine trend and volatility adjustments (weighted average)
        threshold = (base_threshold + vol_adjustment) / 2
        
        # Adjust based on regime confidence
        # Higher confidence = more aggressive threshold (closer to 0)
        # Lower confidence = more conservative threshold (further from 0)
        confidence_adjustment = (1 - confidence) * self.SHORT_THRESHOLD_CONFIDENCE_MULTIPLIER
        threshold = threshold * (1 - confidence_adjustment)
        
        # Ensure threshold is within reasonable bounds
        threshold = max(-0.01, min(-0.0001, threshold))  # Between -1% and -0.01%
        
        return threshold

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Go long if the predicted price for the next bar is higher than the current close
        if index < 1 or index >= len(df):
            return False

        pred = df["onnx_pred"].iloc[index]
        close = df["close"].iloc[index]

        # Check if we have a valid prediction
        if pd.isna(pred):
            # Log the missing prediction
            self.log_execution(
                signal_type="entry",
                action_taken="no_action",
                price=close,
                reasons=["missing_ml_prediction"],
                additional_context={"prediction_available": False},
            )
            return False

        # Calculate predicted return
        predicted_return = (pred - close) / close if close > 0 else 0

        # Determine entry signal
        entry_signal = pred > close

        # Log the decision process
        ml_predictions = {
            "raw_prediction": pred,
            "current_price": close,
            "predicted_return": predicted_return,
        }
        # Engine metadata for auditability
        if self.use_prediction_engine and self.prediction_engine is not None:
            ml_predictions.update(
                {
                    "engine_enabled": True,
                    "engine_model_name": self.model_name,
                    "engine_batch": self.use_engine_batch,
                }
            )

        reasons = [
            f"predicted_return_{predicted_return:.4f}",
            f"prediction_{pred:.2f}_vs_current_{close:.2f}",
            "entry_signal_met" if entry_signal else "entry_signal_not_met",
        ]

        self.log_execution(
            signal_type="entry",
            action_taken="entry_signal" if entry_signal else "no_action",
            price=close,
            signal_strength=abs(predicted_return) if entry_signal else 0.0,
            confidence_score=(
                float(df["prediction_confidence"].iloc[index])
                if "prediction_confidence" in df.columns
                and not pd.isna(df["prediction_confidence"].iloc[index])
                else min(1.0, abs(predicted_return) * 10)
            ),
            ml_predictions=ml_predictions,
            reasons=reasons,
            additional_context={
                "model_type": "ml_adaptive",
                "sequence_length": self.sequence_length,
                "prediction_available": True,
            },
        )

        return entry_signal

    def check_short_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False
        pred = df["onnx_pred"].iloc[index]
        close = df["close"].iloc[index]
        if pd.isna(pred):
            return False
        predicted_return = (pred - close) / close if close > 0 else 0
        
        # * Use dynamic threshold based on market regime
        dynamic_threshold = self._calculate_dynamic_short_threshold(df, index)
        
        # Log the dynamic threshold decision for debugging
        trend_label = df["trend_label"].iloc[index] if "trend_label" in df.columns else "unknown"
        vol_label = df["vol_label"].iloc[index] if "vol_label" in df.columns else "unknown"
        confidence = df["regime_confidence"].iloc[index] if "regime_confidence" in df.columns else 0.0
        
        self.log_execution(
            signal_type="short_entry",
            action_taken="short_signal" if predicted_return < dynamic_threshold else "no_action",
            price=close,
            signal_strength=abs(predicted_return) if predicted_return < dynamic_threshold else 0.0,
            confidence_score=confidence,
            reasons=[
                f"predicted_return_{predicted_return:.4f}",
                f"dynamic_threshold_{dynamic_threshold:.4f}",
                f"regime_{trend_label}_{vol_label}",
                f"confidence_{confidence:.3f}",
                "short_signal_met" if predicted_return < dynamic_threshold else "short_signal_not_met",
            ],
            additional_context={
                "regime_trend": trend_label,
                "regime_volatility": vol_label,
                "regime_confidence": confidence,
                "dynamic_threshold": dynamic_threshold,
                "base_threshold": self.SHORT_ENTRY_THRESHOLD,
            },
        )
        
        return predicted_return < dynamic_threshold

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False
        current_price = df["close"].iloc[index]
        returns = (current_price - entry_price) / entry_price
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct

        # * Basic exit conditions (stop loss and take profit)
        basic_exit = hit_stop_loss or hit_take_profit

        # * ML-based exit signal for unfavorable predictions
        pred = df["onnx_pred"].iloc[index]
        if not pd.isna(pred):
            # * For long positions: exit if prediction suggests significant price drop
            # * Only exit if prediction is significantly unfavorable (>2% drop predicted)
            predicted_return = (pred - current_price) / current_price if current_price > 0 else 0
            significant_unfavorable_prediction = predicted_return < -0.02  # 2% threshold

            return basic_exit or significant_unfavorable_prediction

        return basic_exit

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if index >= len(df) or balance <= 0:
            return 0.0
        pred = df["onnx_pred"].iloc[index]
        close = df["close"].iloc[index]
        if pd.isna(pred):
            return 0.0
        predicted_return = abs(pred - close) / close if close > 0 else 0
        confidence = min(1.0, predicted_return * self.CONFIDENCE_MULTIPLIER)
        dynamic_size = self.BASE_POSITION_SIZE * confidence
        return (
            max(self.MIN_POSITION_SIZE_RATIO, min(self.MAX_POSITION_SIZE_RATIO, dynamic_size))
            * balance
        )

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        base_params = self.get_base_parameters()
        strategy_params = {
            "model_path": self.model_path,
            "sequence_length": self.sequence_length,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "use_prediction_engine": self.use_prediction_engine,
            "model_name": self.model_name,
        }
        return {**base_params, **strategy_params}

    def get_risk_overrides(self) -> Optional[dict[str, Any]]:
        """
        Provide risk management overrides including dynamic risk, partial operations, trailing stops, and time-based exits.
        
        This strategy uses:
        
        Dynamic Risk Management:
        - Adaptive position sizing based on performance and market conditions
        - Drawdown-based risk reduction: 5%, 10%, 15% thresholds
        - Recovery-based risk restoration: 2%, 5% positive return thresholds
        - Volatility-based adjustments for high/low volatility periods
        
        Conservative Partial Operations:
        - Take 25% profit at 3% gain, 25% at 6% gain, 50% at 10% gain
        - Scale in 25% at 2% gain, 25% at 5% gain (max 2 scale-ins)
        
        Conservative Trailing Stops:
        - Activate at 1.5% profit
        - 0.5% trailing distance
        - Breakeven at 2% profit
        
        Time-Based Exits:
        - Maximum 24-hour holding period for crypto positions
        - No weekend restrictions (crypto trades 24/7)
        - No overnight restrictions (crypto trades 24/7)
        """
        return {
            "dynamic_risk": {
                "enabled": True,
                "performance_window_days": 30,
                "drawdown_thresholds": [0.05, 0.10, 0.15],  # 5%, 10%, 15%
                "risk_reduction_factors": [0.8, 0.6, 0.4],   # 80%, 60%, 40% of normal size
                "recovery_thresholds": [0.02, 0.05],         # 2%, 5% positive returns
                "volatility_adjustment_enabled": True,
                "volatility_window_days": 30,
                "high_volatility_threshold": 0.03,           # 3% daily volatility
                "low_volatility_threshold": 0.01,            # 1% daily volatility
                "volatility_risk_multipliers": (0.7, 1.3),   # (high_vol, low_vol) multipliers
            },
            "partial_operations": {
                "exit_targets": [0.03, 0.06, 0.10],  # 3%, 6%, 10%
                "exit_sizes": [0.25, 0.25, 0.50],     # 25%, 25%, 50%
                "scale_in_thresholds": [0.02, 0.05],  # 2%, 5%
                "scale_in_sizes": [0.25, 0.25],       # 25%, 25%
                "max_scale_ins": 2,
            },
            "trailing_stop": {
                "activation_threshold": 0.015,  # 1.5%
                "trailing_distance_pct": 0.005,  # 0.5%
                "breakeven_threshold": 0.02,  # 2.0%
                "breakeven_buffer": 0.001,  # 0.1%
            },
            "time_exits": {
                "max_holding_hours": 24,  # Maximum 24-hour holding period
                "end_of_day_flat": False,  # No end-of-day restrictions for crypto
                "weekend_flat": False,     # No weekend restrictions for crypto
                "market_timezone": "UTC",  # Use UTC for crypto markets
                "time_restrictions": {
                    "no_overnight": False,     # No overnight restrictions for crypto
                    "no_weekend": False,       # No weekend restrictions for crypto
                    "trading_hours_only": False,  # No trading hours restrictions for crypto
                }
            }
        }

    def calculate_stop_loss(self, df, index, price, side) -> float:
        """Calculate stop loss price"""
        # * Handle both string and enum inputs for backward compatibility
        side_str = side.value if hasattr(side, "value") else str(side)

        if side_str == "long":
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct)

    def _load_model(self):
        """Load or reload the ONNX model"""
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Failed to load model {self.model_path}: {e}")
            raise
