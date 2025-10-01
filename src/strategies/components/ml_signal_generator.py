"""
ML Signal Generator Components

This module contains ML-based signal generators extracted from existing strategies.
These components generate trading signals using machine learning models with
regime-aware threshold adjustments and confidence calculations.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_USE_PREDICTION_ENGINE
from src.prediction import PredictionConfig, PredictionEngine
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.price_only import PriceOnlyFeatureExtractor
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.regime.detector import TrendLabel, VolLabel
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator


class MLSignalGenerator(SignalGenerator):
    """
    ML Signal Generator extracted from MlAdaptive strategy
    
    This signal generator uses machine learning models to predict price movements
    and generates trading signals with regime-aware threshold adjustments.
    
    Features:
    - ONNX model inference for price predictions
    - Regime-aware dynamic threshold adjustment
    - Confidence calculation based on prediction quality
    - Optional prediction engine integration
    """
    
    # Configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # Base threshold for short entries (-0.05%)
    CONFIDENCE_MULTIPLIER = 12  # Multiplier for confidence calculation
    
    # Dynamic threshold configuration for different market regimes
    SHORT_THRESHOLD_TREND_UP = -0.0003  # Less conservative in uptrend (-0.03%)
    SHORT_THRESHOLD_TREND_DOWN = -0.0007  # More conservative in downtrend (-0.07%)
    SHORT_THRESHOLD_RANGE = -0.0005  # Standard threshold in range-bound market (-0.05%)
    SHORT_THRESHOLD_HIGH_VOL = -0.0004  # Less conservative in high volatility (-0.04%)
    SHORT_THRESHOLD_LOW_VOL = -0.0006  # More conservative in low volatility (-0.06%)
    SHORT_THRESHOLD_CONFIDENCE_MULTIPLIER = 0.2  # Adjust threshold based on regime confidence
    
    def __init__(
        self,
        name: str = "ml_signal_generator",
        model_path: str = "src/ml/btcusdt_price.onnx",
        sequence_length: int = 120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize ML Signal Generator
        
        Args:
            name: Name for this signal generator
            model_path: Path to ONNX model file
            sequence_length: Sequence length for model input
            use_prediction_engine: Whether to use prediction engine (optional)
            model_name: Model name for prediction engine (optional)
        """
        super().__init__(name)
        
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        
        # Prediction engine configuration
        cfg = get_config()
        self.use_prediction_engine = (
            use_prediction_engine
            if use_prediction_engine is not None
            else cfg.get_bool("USE_PREDICTION_ENGINE", default=DEFAULT_USE_PREDICTION_ENGINE)
        )
        
        # Model name configuration
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
        self.use_engine_batch = get_config().get_bool("ENGINE_BATCH_INFERENCE", default=False)
        
        # Initialize feature pipeline
        self._setup_feature_pipeline()
        
        # Initialize prediction engine if enabled
        if self.use_prediction_engine:
            self._initialize_prediction_engine()
    
    def _setup_feature_pipeline(self):
        """Setup feature pipeline for data preprocessing"""
        if self.use_prediction_engine:
            # Use price-only extractor for prediction engine
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
            # Use technical extractor for local ONNX
            technical_extractor = TechnicalFeatureExtractor(
                sequence_length=self.sequence_length, 
                normalization_window=self.sequence_length
            )
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
    
    def _initialize_prediction_engine(self):
        """Initialize prediction engine with health check"""
        try:
            config = PredictionConfig.from_config_manager()
            config.enable_sentiment = False
            config.enable_market_microstructure = False
            engine = PredictionEngine(config)
            
            # Setup feature pipeline for engine
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
            
            # Health check
            health = engine.health_check()
            if health.get("status") != "healthy" and not self._engine_warning_emitted:
                print(f"[MLSignalGenerator] Prediction engine health degraded: {health}")
                self._engine_warning_emitted = True
            
            self.prediction_engine = engine
            
        except Exception as e:
            if not self._engine_warning_emitted:
                print(f"[MLSignalGenerator] Prediction engine initialization failed: {e}")
                self._engine_warning_emitted = True
            self.prediction_engine = None
    
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None) -> Signal:
        """
        Generate trading signal based on ML prediction
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            index: Current index position in the DataFrame
            regime: Optional regime context for regime-aware signal generation
            
        Returns:
            Signal object with direction, strength, confidence, and metadata
        """
        self.validate_inputs(df, index)
        
        # Ensure we have enough history for prediction
        if index < self.sequence_length:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'insufficient_history',
                    'index': index,
                    'required_length': self.sequence_length
                }
            )
        
        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'prediction_failed',
                    'index': index
                }
            )
        
        current_price = df['close'].iloc[index]
        predicted_return = (prediction - current_price) / current_price if current_price > 0 else 0
        
        # Determine signal direction
        if predicted_return > 0:
            direction = SignalDirection.BUY
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        elif self._should_generate_short_signal(predicted_return, regime):
            direction = SignalDirection.SELL
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        else:
            direction = SignalDirection.HOLD
            strength = 0.0
        
        # Calculate confidence
        confidence = self._calculate_confidence(predicted_return)
        
        # Create metadata
        metadata = {
            'generator': self.name,
            'prediction': prediction,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'index': index,
            'model_path': self.model_path,
            'sequence_length': self.sequence_length
        }
        
        # Add regime information if available
        if regime:
            metadata.update({
                'regime_trend': regime.trend.value,
                'regime_volatility': regime.volatility.value,
                'regime_confidence': regime.confidence,
                'dynamic_threshold': self._calculate_dynamic_short_threshold(regime)
            })
        
        # Add prediction engine metadata
        if self.use_prediction_engine and self.prediction_engine is not None:
            metadata.update({
                'engine_enabled': True,
                'engine_model_name': self.model_name,
                'engine_batch': self.use_engine_batch
            })
        
        return Signal(
            direction=direction,
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
        
        if index < self.sequence_length:
            return 0.0
        
        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return 0.0
        
        current_price = df['close'].iloc[index]
        predicted_return = (prediction - current_price) / current_price if current_price > 0 else 0
        
        return self._calculate_confidence(predicted_return)
    
    def _get_ml_prediction(self, df: pd.DataFrame, index: int) -> Optional[float]:
        """
        Get ML prediction for the given index
        
        Args:
            df: DataFrame with processed features
            index: Current index position
            
        Returns:
            Predicted price or None if prediction fails
        """
        try:
            # Prepare input features
            price_features = ["close", "volume", "high", "low", "open"]
            feature_columns = [f"{feature}_normalized" for feature in price_features]
            
            # Check if features exist
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                # Apply feature pipeline if features are missing
                df_processed = self.feature_pipeline.transform(df.copy())
                input_data = df_processed[feature_columns].iloc[index - self.sequence_length : index].values
            else:
                input_data = df[feature_columns].iloc[index - self.sequence_length : index].values
            
            # Reshape for ONNX model: (batch_size, sequence_length, features)
            input_data = input_data.astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Get prediction
            if self.use_prediction_engine and self.prediction_engine is not None and self.model_name:
                # Use prediction engine
                window_df = df[["open", "high", "low", "close", "volume"]].iloc[
                    index - self.sequence_length : index
                ]
                result = self.prediction_engine.predict(window_df, model_name=self.model_name)
                pred = float(result.price)
            else:
                # Use local ONNX session
                output = self.ort_session.run(None, {self.input_name: input_data})
                pred = output[0][0][0]
            
            # Ensure pred is a scalar value
            if isinstance(pred, (list, tuple, np.ndarray)):
                pred = float(np.array(pred).flatten()[0])
            else:
                pred = float(pred)
            
            # Denormalize prediction
            recent_close = df["close"].iloc[index - self.sequence_length : index].values
            min_close = np.min(recent_close)
            max_close = np.max(recent_close)
            
            if max_close != min_close:
                pred_denormalized = pred * (max_close - min_close) + min_close
            else:
                pred_denormalized = df["close"].iloc[index - 1]
            
            return pred_denormalized
            
        except Exception as e:
            print(f"[MLSignalGenerator] Prediction error at index {index}: {e}")
            return None
    
    def _should_generate_short_signal(self, predicted_return: float, regime: Optional[RegimeContext]) -> bool:
        """
        Determine if a short signal should be generated based on predicted return and regime
        
        Args:
            predicted_return: Predicted return value
            regime: Current regime context
            
        Returns:
            True if short signal should be generated
        """
        if regime is not None:
            dynamic_threshold = self._calculate_dynamic_short_threshold(regime)
        else:
            dynamic_threshold = self.SHORT_ENTRY_THRESHOLD
        
        return predicted_return < dynamic_threshold
    
    def _calculate_dynamic_short_threshold(self, regime: RegimeContext) -> float:
        """
        Calculate dynamic short entry threshold based on current market regime
        
        Args:
            regime: Current regime context
            
        Returns:
            Dynamic threshold for short entries
        """
        # Start with base threshold based on trend
        if regime.trend == TrendLabel.TREND_UP:
            base_threshold = self.SHORT_THRESHOLD_TREND_UP
        elif regime.trend == TrendLabel.TREND_DOWN:
            base_threshold = self.SHORT_THRESHOLD_TREND_DOWN
        else:  # range
            base_threshold = self.SHORT_THRESHOLD_RANGE
        
        # Adjust for volatility
        if regime.volatility == VolLabel.HIGH:
            vol_adjustment = self.SHORT_THRESHOLD_HIGH_VOL
        else:  # low_vol
            vol_adjustment = self.SHORT_THRESHOLD_LOW_VOL
        
        # Combine trend and volatility adjustments (weighted average)
        threshold = (base_threshold + vol_adjustment) / 2
        
        # Adjust based on regime confidence
        # Higher confidence = more aggressive threshold (closer to 0)
        # Lower confidence = more conservative threshold (further from 0)
        confidence_adjustment = (1 - regime.confidence) * self.SHORT_THRESHOLD_CONFIDENCE_MULTIPLIER
        threshold = threshold * (1 - confidence_adjustment)
        
        # Ensure threshold is within reasonable bounds
        threshold = max(-0.01, min(-0.0001, threshold))  # Between -1% and -0.01%
        
        return threshold
    
    def _calculate_confidence(self, predicted_return: float) -> float:
        """
        Calculate confidence score based on predicted return magnitude
        
        Args:
            predicted_return: Predicted return value
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = min(1.0, abs(predicted_return) * self.CONFIDENCE_MULTIPLIER)
        return max(0.0, confidence)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get signal generator parameters for logging and serialization"""
        params = super().get_parameters()
        params.update({
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'use_prediction_engine': self.use_prediction_engine,
            'model_name': self.model_name,
            'short_entry_threshold': self.SHORT_ENTRY_THRESHOLD,
            'confidence_multiplier': self.CONFIDENCE_MULTIPLIER
        })
        return params


class MLBasicSignalGenerator(SignalGenerator):
    """
    ML Basic Signal Generator extracted from MlBasic strategy
    
    This signal generator uses machine learning models to predict price movements
    and generates basic trading signals without regime awareness.
    
    Features:
    - ONNX model inference for price predictions
    - Basic signal generation without regime-specific adjustments
    - Confidence calculation based on prediction quality
    - Optional prediction engine integration with model registry support
    """
    
    # Configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 12  # Multiplier for confidence calculation
    
    def __init__(
        self,
        name: str = "ml_basic_signal_generator",
        model_path: str = "src/ml/btcusdt_price.onnx",
        sequence_length: int = 120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
    ):
        """
        Initialize ML Basic Signal Generator
        
        Args:
            name: Name for this signal generator
            model_path: Path to ONNX model file
            sequence_length: Sequence length for model input
            use_prediction_engine: Whether to use prediction engine (optional)
            model_name: Model name for prediction engine (optional)
            model_type: Model type for registry selection (optional)
            timeframe: Timeframe for registry selection (optional)
        """
        super().__init__(name)
        
        self.model_path = model_path
        self.sequence_length = sequence_length
        
        # Registry model selection preferences
        self.model_type = model_type or "basic"
        self.model_timeframe = timeframe or "1h"
        
        # Defer ONNX session init if prediction engine/registry is used
        self.ort_session = None
        self.input_name = None
        
        # Prediction engine configuration
        cfg = get_config()
        self.use_prediction_engine = (
            use_prediction_engine
            if use_prediction_engine is not None
            else cfg.get_bool("USE_PREDICTION_ENGINE", default=DEFAULT_USE_PREDICTION_ENGINE)
        )
        
        # Model name configuration
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)
        if self.model_name is None:
            try:
                self.model_name = Path(self.model_path).stem
            except Exception:
                self.model_name = None
        
        self.prediction_engine = None
        self._registry = None
        self._engine_warning_emitted = False
        self.use_engine_batch = get_config().get_bool("ENGINE_BATCH_INFERENCE", default=False)
        
        # Initialize feature pipeline
        self._setup_feature_pipeline()
        
        # Initialize prediction engine if enabled
        if self.use_prediction_engine:
            self._initialize_prediction_engine()
    
    def _setup_feature_pipeline(self):
        """Setup feature pipeline for data preprocessing"""
        if self.use_prediction_engine:
            # Use price-only extractor for prediction engine
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
            # Use technical extractor for local ONNX
            technical_extractor = TechnicalFeatureExtractor(
                sequence_length=self.sequence_length, 
                normalization_window=self.sequence_length
            )
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
    
    def _initialize_prediction_engine(self):
        """Initialize prediction engine with health check and registry support"""
        try:
            config = PredictionConfig.from_config_manager()
            config.enable_sentiment = False
            config.enable_market_microstructure = False
            engine = PredictionEngine(config)
            
            # Setup feature pipeline for engine
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
            
            # Health check
            health = engine.health_check()
            if health.get("status") != "healthy" and not self._engine_warning_emitted:
                print(f"[MLBasicSignalGenerator] Prediction engine health degraded: {health}")
                self._engine_warning_emitted = True
            
            self.prediction_engine = engine
            
            # Initialize registry for structured selection
            try:
                self._registry = engine.model_registry
            except Exception:
                self._registry = None
                
        except Exception as e:
            if not self._engine_warning_emitted:
                print(f"[MLBasicSignalGenerator] Prediction engine initialization failed: {e}")
                self._engine_warning_emitted = True
            self.prediction_engine = None
    
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None) -> Signal:
        """
        Generate trading signal based on ML prediction (basic, no regime awareness)
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            index: Current index position in the DataFrame
            regime: Optional regime context (ignored in basic implementation)
            
        Returns:
            Signal object with direction, strength, confidence, and metadata
        """
        self.validate_inputs(df, index)
        
        # Ensure we have enough history for prediction
        if index < self.sequence_length:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'insufficient_history',
                    'index': index,
                    'required_length': self.sequence_length
                }
            )
        
        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    'generator': self.name,
                    'reason': 'prediction_failed',
                    'index': index
                }
            )
        
        current_price = df['close'].iloc[index]
        predicted_return = (prediction - current_price) / current_price if current_price > 0 else 0
        
        # Determine signal direction (basic logic without regime awareness)
        if predicted_return > 0:
            direction = SignalDirection.BUY
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        elif predicted_return < self.SHORT_ENTRY_THRESHOLD:
            direction = SignalDirection.SELL
            strength = min(1.0, abs(predicted_return) * 10)  # Scale to 0-1
        else:
            direction = SignalDirection.HOLD
            strength = 0.0
        
        # Calculate confidence
        confidence = self._calculate_confidence(predicted_return)
        
        # Create metadata
        metadata = {
            'generator': self.name,
            'prediction': prediction,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'index': index,
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'short_threshold': self.SHORT_ENTRY_THRESHOLD
        }
        
        # Add prediction engine metadata
        if self.use_prediction_engine and self.prediction_engine is not None:
            metadata.update({
                'engine_enabled': True,
                'engine_model_name': self.model_name,
                'engine_batch': self.use_engine_batch,
                'model_type': self.model_type,
                'model_timeframe': self.model_timeframe
            })
        
        return Signal(
            direction=direction,
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
        
        if index < self.sequence_length:
            return 0.0
        
        # Get ML prediction
        prediction = self._get_ml_prediction(df, index)
        if prediction is None:
            return 0.0
        
        current_price = df['close'].iloc[index]
        predicted_return = (prediction - current_price) / current_price if current_price > 0 else 0
        
        return self._calculate_confidence(predicted_return)
    
    def _get_ml_prediction(self, df: pd.DataFrame, index: int) -> Optional[float]:
        """
        Get ML prediction for the given index with registry support
        
        Args:
            df: DataFrame with processed features
            index: Current index position
            
        Returns:
            Predicted price or None if prediction fails
        """
        try:
            # Prepare input features
            price_features = ["close", "volume", "high", "low", "open"]
            feature_columns = [f"{feature}_normalized" for feature in price_features]
            
            # Check if features exist
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                # Apply feature pipeline if features are missing
                df_processed = self.feature_pipeline.transform(df.copy())
                input_data = df_processed[feature_columns].iloc[index - self.sequence_length : index].values
            else:
                input_data = df[feature_columns].iloc[index - self.sequence_length : index].values
            
            # Reshape for ONNX model: (batch_size, sequence_length, features)
            input_data = input_data.astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Try to bind bundle session if using registry
            selected_bundle_key = None
            if self.use_prediction_engine and self._registry is not None:
                try:
                    bundle = self._registry.select_bundle(
                        symbol="BTCUSDT",  # Default trading pair
                        model_type=self.model_type,
                        timeframe=self.model_timeframe,
                    )
                    selected_bundle_key = bundle.key
                    if getattr(bundle.runner, "session", None) is not None:
                        self.ort_session = bundle.runner.session
                        self.input_name = self.ort_session.get_inputs()[0].name
                except Exception:
                    selected_bundle_key = None
            
            # Get prediction
            if self.use_prediction_engine and self.prediction_engine is not None:
                # Use prediction engine
                window_df = df[["open", "high", "low", "close", "volume"]].iloc[
                    index - self.sequence_length : index
                ]
                
                # Prefer registry selection by symbol/type/timeframe when available
                engine_model_name = selected_bundle_key or self.model_name
                try:
                    if engine_model_name:
                        result = self.prediction_engine.predict(
                            window_df, model_name=engine_model_name
                        )
                    else:
                        result = self.prediction_engine.predict(window_df)
                except Exception:
                    # Fall back to default registry resolution if explicit lookup fails
                    result = self.prediction_engine.predict(window_df)
                
                pred = float(result.price)
            else:
                # Use local ONNX session; initialize lazily
                if self.ort_session is None:
                    self.ort_session = ort.InferenceSession(self.model_path)
                    self.input_name = self.ort_session.get_inputs()[0].name
                
                output = self.ort_session.run(None, {self.input_name: input_data})
                pred = output[0][0][0]
            
            # Ensure pred is a scalar value
            if isinstance(pred, (list, tuple, np.ndarray)):
                pred = float(np.array(pred).flatten()[0])
            else:
                pred = float(pred)
            
            # Denormalize prediction
            recent_close = df["close"].iloc[index - self.sequence_length : index].values
            min_close = np.min(recent_close)
            max_close = np.max(recent_close)
            
            if max_close != min_close:
                pred_denormalized = pred * (max_close - min_close) + min_close
            else:
                pred_denormalized = df["close"].iloc[index - 1]
            
            return pred_denormalized
            
        except Exception as e:
            print(f"[MLBasicSignalGenerator] Prediction error at index {index}: {e}")
            return None
    
    def _calculate_confidence(self, predicted_return: float) -> float:
        """
        Calculate confidence score based on predicted return magnitude
        
        Args:
            predicted_return: Predicted return value
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = min(1.0, abs(predicted_return) * self.CONFIDENCE_MULTIPLIER)
        return max(0.0, confidence)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get signal generator parameters for logging and serialization"""
        params = super().get_parameters()
        params.update({
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'use_prediction_engine': self.use_prediction_engine,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_timeframe': self.model_timeframe,
            'short_entry_threshold': self.SHORT_ENTRY_THRESHOLD,
            'confidence_multiplier': self.CONFIDENCE_MULTIPLIER
        })
        return params