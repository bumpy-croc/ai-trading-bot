"""
ONNX model runner for inference.

This module provides classes for running ONNX models and handling
prediction results in a standardized way.
"""
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ModelPrediction:
    """Result of a model prediction."""
    price: float
    confidence: float
    direction: int  # 1 for up, 0 for neutral, -1 for down
    model_name: str
    inference_time: float
    model_metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate prediction values after initialization."""
        if not isinstance(self.direction, int) or self.direction not in [-1, 0, 1]:
            self.direction = 0  # Default to neutral if invalid
        
        if self.confidence < 0:
            self.confidence = 0.0
        elif self.confidence > 1:
            self.confidence = 1.0


class OnnxRunner:
    """
    ONNX model runner for ML inference.
    
    This class handles loading and running ONNX models for prediction,
    with support for different normalization methods and metadata tracking.
    """
    
    def __init__(self, 
                 model_path: str, 
                 model_name: str,
                 normalization_fn: Optional[callable] = None,
                 expected_features: Optional[int] = None,
                 sequence_length: int = 120,
                 model_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize ONNX runner.
        
        Args:
            model_path: Path to the ONNX model file
            model_name: Name identifier for the model
            normalization_fn: Function to normalize input features
            expected_features: Expected number of input features
            sequence_length: Expected sequence length for LSTM models
            model_metadata: Additional metadata about the model
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.normalization_fn = normalization_fn
        self.expected_features = expected_features
        self.sequence_length = sequence_length
        self.model_metadata = model_metadata or {}
        
        # Validate model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Lazy session initialization
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        
        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._last_prediction_time = 0.0
    
    @property
    def session(self) -> ort.InferenceSession:
        """Get or create the ONNX runtime session."""
        if self._session is None:
            try:
                self._session = ort.InferenceSession(
                    str(self.model_path), 
                    providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
            except Exception as e:
                raise RuntimeError(f"Failed to load ONNX model {self.model_path}: {e}")
        
        return self._session
    
    @property
    def input_name(self) -> str:
        """Get the input tensor name."""
        if self._input_name is None:
            # Accessing session property will initialize it
            _ = self.session
        return self._input_name
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """
        Make a prediction using the ONNX model.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            ModelPrediction with prediction results
            
        Raises:
            ValueError: If input features are invalid
            RuntimeError: If model inference fails
        """
        start_time = time.time()
        
        try:
            # Validate and prepare input data
            input_array = self._prepare_input(features)
            
            # Run model inference
            inference_start = time.time()
            output = self.session.run(None, {self.input_name: input_array})
            inference_time = time.time() - inference_start
            
            # Process output and create prediction
            prediction = self._process_output(output, inference_time)
            
            # Update performance statistics
            self._prediction_count += 1
            self._total_inference_time += inference_time
            self._last_prediction_time = inference_time
            
            return prediction
            
        except Exception as e:
            # Return error prediction
            inference_time = time.time() - start_time
            return ModelPrediction(
                price=0.0,
                confidence=0.0,
                direction=0,
                model_name=self.model_name,
                inference_time=inference_time,
                model_metadata={**self.model_metadata, 'error': str(e)}
            )
    
    def _prepare_input(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare input data for the model."""
        # Apply normalization if provided
        if self.normalization_fn:
            features = self.normalization_fn(features, self.sequence_length)
            if isinstance(features, tuple):
                features, _ = features  # Handle functions that return (df, feature_names)
        
        # For LSTM models, we need the last sequence_length rows
        if len(features) < self.sequence_length:
            raise ValueError(f"Insufficient data: {len(features)} rows, need {self.sequence_length}")
        
        # Get the most recent sequence
        recent_data = features.tail(self.sequence_length)
        
        # Select appropriate features for model input
        feature_columns = self._select_feature_columns(recent_data)
        input_data = recent_data[feature_columns].values
        
        # Validate feature count
        if self.expected_features and input_data.shape[1] != self.expected_features:
            if input_data.shape[1] < self.expected_features:
                # Pad with zeros
                padding = np.zeros((input_data.shape[0], self.expected_features - input_data.shape[1]))
                input_data = np.concatenate([input_data, padding], axis=1)
            else:
                # Truncate to expected features
                input_data = input_data[:, :self.expected_features]
        
        # Reshape for ONNX model: (batch_size, sequence_length, features)
        input_data = input_data.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        
        # Check for NaN values
        if np.isnan(input_data).any():
            raise ValueError("Input data contains NaN values")
        
        return input_data
    
    def _select_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Select appropriate feature columns for model input."""
        # Priority order for feature selection
        priority_features = [
            'close_normalized', 'volume_normalized', 'high_normalized', 
            'low_normalized', 'open_normalized'
        ]
        
        # Start with basic normalized features
        selected_features = []
        for feature in priority_features:
            if feature in data.columns:
                selected_features.append(feature)
        
        # Add technical indicators if available
        technical_features = [
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 
            'macd', 'rsi', 'bb_upper', 'bb_lower', 'atr'
        ]
        for feature in technical_features:
            if feature in data.columns and feature not in selected_features:
                selected_features.append(feature)
        
        # If we still don't have enough features, add any numerical columns
        if len(selected_features) < 3:  # Minimum features needed
            for col in data.columns:
                if (col not in selected_features and 
                    data[col].dtype in ['float64', 'float32', 'int64', 'int32']):
                    selected_features.append(col)
        
        # Ensure we have at least basic features
        if not selected_features:
            raise ValueError("No suitable features found in input data")
        
        return selected_features
    
    def _process_output(self, output: List[np.ndarray], inference_time: float) -> ModelPrediction:
        """Process model output into a standardized prediction."""
        # Extract prediction value (assuming single output)
        pred_value = float(output[0][0][0]) if len(output[0].shape) == 3 else float(output[0][0])
        
        # Determine price prediction (model-specific logic)
        predicted_price = self._interpret_prediction_value(pred_value)
        
        # Calculate confidence (simple heuristic based on prediction magnitude)
        confidence = min(abs(pred_value), 1.0) if abs(pred_value) <= 1.0 else 1.0
        
        # Determine direction
        direction = 1 if pred_value > 0.001 else (-1 if pred_value < -0.001 else 0)
        
        return ModelPrediction(
            price=predicted_price,
            confidence=confidence,
            direction=direction,
            model_name=self.model_name,
            inference_time=inference_time,
            model_metadata=self.model_metadata
        )
    
    def _interpret_prediction_value(self, pred_value: float) -> float:
        """Interpret raw prediction value as a price prediction."""
        # This is model-specific logic that may need to be customized
        # For now, assume the prediction is a normalized price change
        
        # If the model outputs absolute prices (like price models)
        if abs(pred_value) > 10:  # Likely an absolute price
            return abs(pred_value)
        
        # If the model outputs normalized values or changes
        # We'll return a placeholder price (would need actual current price to denormalize)
        return abs(pred_value) * 50000  # Rough scaling for BTC prices
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'expected_features': self.expected_features,
            'sequence_length': self.sequence_length,
            'metadata': self.model_metadata,
            'prediction_count': self._prediction_count,
            'avg_inference_time': (
                self._total_inference_time / self._prediction_count 
                if self._prediction_count > 0 else 0.0
            ),
            'last_inference_time': self._last_prediction_time
        }
        
        # Add session info if loaded
        if self._session is not None:
            try:
                inputs = self._session.get_inputs()
                outputs = self._session.get_outputs()
                info.update({
                    'input_shape': inputs[0].shape if inputs else None,
                    'output_shape': outputs[0].shape if outputs else None,
                    'input_type': str(inputs[0].type) if inputs else None,
                    'output_type': str(outputs[0].type) if outputs else None
                })
            except Exception:
                # If we can't get session info, skip it
                pass
        
        return info
    
    def warmup(self, sample_features: pd.DataFrame) -> None:
        """Warm up the model with a sample prediction."""
        try:
            self.predict(sample_features)
        except Exception:
            # Warmup failures are not critical
            pass