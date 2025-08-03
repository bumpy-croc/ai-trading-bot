"""
ONNX Model Runner for inference operations.
"""

import os
import json
import time
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..config import PredictionConfig


@dataclass
class ModelPrediction:
    """Result of a model prediction"""
    price: float
    confidence: float
    direction: int  # 1, 0, -1
    model_name: str
    inference_time: float


class OnnxRunner:
    """Handles ONNX model loading and inference"""
    
    def __init__(self, model_path: str, config: PredictionConfig):
        """
        Initialize ONNX runner with model path and configuration.
        
        Args:
            model_path: Path to the ONNX model file
            config: Prediction engine configuration
        """
        self.model_path = model_path
        self.config = config
        self.session = None
        self.model_metadata = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model and metadata"""
        try:
            # Load ONNX session
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Load model metadata
            self.model_metadata = self._load_metadata()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from JSON file"""
        metadata_path = self.model_path.replace('.onnx', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default metadata if file doesn't exist
            return {
                'sequence_length': 120,
                'feature_count': 5,
                'normalization_params': {}
            }
    
    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Run prediction on features"""
        start_time = time.time()
        
        try:
            # Prepare input
            input_data = self._prepare_input(features)
            
            # Get input name dynamically from ONNX session
            input_name = self.session.get_inputs()[0].name
            
            # Run inference
            output = self.session.run(None, {input_name: input_data})
            
            # Process output
            prediction = self._process_output(output[0])
            
            inference_time = time.time() - start_time
            
            return ModelPrediction(
                price=prediction['price'],
                confidence=prediction['confidence'],
                direction=prediction['direction'],
                model_name=os.path.basename(self.model_path),
                inference_time=inference_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _prepare_input(self, features: np.ndarray) -> np.ndarray:
        """Prepare features for model input"""
        # Ensure correct shape (batch_size, sequence_length, features)
        if len(features.shape) == 2:
            features = features.reshape(1, -1, features.shape[1])
        
        # Normalize features if metadata contains normalization params
        if self.model_metadata.get('normalization_params'):
            features = self._normalize_features(features)
        
        return features.astype(np.float32)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using model metadata"""
        norm_params = self.model_metadata['normalization_params']
        
        for i, feature_name in enumerate(norm_params.keys()):
            if i < features.shape[2]:  # Check feature index bounds
                mean = norm_params[feature_name].get('mean', 0.0)
                std = norm_params[feature_name].get('std', 1.0)
                features[:, :, i] = (features[:, :, i] - mean) / std
        
        return features
    
    def _process_output(self, output: np.ndarray) -> Dict[str, Any]:
        """Process model output into prediction"""
        # Extract scalar prediction
        if output.shape == (1, 1, 1):
            pred = output[0][0][0]
        elif output.shape == (1, 1):
            pred = output[0][0]
        else:
            pred = output.flatten()[0]
        
        # Denormalize prediction if needed
        if self.model_metadata.get('price_normalization'):
            pred = self._denormalize_price(pred)
        
        # Calculate confidence and direction
        confidence = self._calculate_confidence(pred)
        direction = self._calculate_direction(pred)
        
        return {
            'price': float(pred),
            'confidence': confidence,
            'direction': direction
        }
    
    def _denormalize_price(self, pred: float) -> float:
        """Denormalize price prediction"""
        price_params = self.model_metadata['price_normalization']
        return pred * price_params['std'] + price_params['mean']
    
    def _calculate_confidence(self, pred: float) -> float:
        """Calculate prediction confidence"""
        # Simple confidence based on prediction magnitude
        # Can be enhanced with model uncertainty estimation
        return min(1.0, abs(pred) * self.config.confidence_scale_factor)
    
    def _calculate_direction(self, pred: float) -> int:
        """Calculate prediction direction"""
        if pred > self.config.direction_threshold:
            return 1
        elif pred < -self.config.direction_threshold:
            return -1
        else:
            return 0