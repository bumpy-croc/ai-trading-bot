"""
ONNX model runner for inference.

This module provides functionality to run ONNX models for prediction.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

from ..config import PredictionConfig
from ..utils.caching import PredictionCacheManager

# Constants for numerical stability
EPSILON = 1e-8  # Small value to prevent division by zero


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

    def __init__(self, model_path: str, config: PredictionConfig, cache_manager: Optional[PredictionCacheManager] = None):
        """
        Initialize ONNX runner with model path and configuration.

        Args:
            model_path: Path to the ONNX model file
            config: Prediction engine configuration
            cache_manager: Optional prediction cache manager
        """
        self.model_path = model_path
        self.config = config
        self.cache_manager = cache_manager
        self.session = None
        self.model_metadata = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model and metadata"""
        try:
            # Load ONNX session
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

            # Load model metadata
            self.model_metadata = self._load_metadata()

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}") from e

    def _load_metadata(self) -> dict[str, Any]:
        """Load model metadata from JSON file"""
        metadata_path = self.model_path.replace(".onnx", "_metadata.json")
        try:
            with open(metadata_path) as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default metadata if file doesn't exist
            return {"sequence_length": 120, "feature_count": 5, "normalization_params": {}}

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Run prediction on features with optional caching"""
        start_time = time.time()

        try:
            # Check cache first if enabled
            if self.cache_manager and self.config.prediction_cache_enabled:
                cache_result = self._check_cache(features)
                if cache_result is not None:
                    # Return cached result
                    inference_time = time.time() - start_time
                    return ModelPrediction(
                        price=cache_result["price"],
                        confidence=cache_result["confidence"],
                        direction=cache_result["direction"],
                        model_name=os.path.basename(self.model_path),
                        inference_time=inference_time,
                    )

            # Prepare input
            input_data = self._prepare_input(features)

            # Get input name dynamically from ONNX session
            input_name = self.session.get_inputs()[0].name

            # Run inference
            output = self.session.run(None, {input_name: input_data})

            # Process output
            prediction = self._process_output(output[0])

            inference_time = time.time() - start_time

            # Cache the result if enabled
            if self.cache_manager and self.config.prediction_cache_enabled:
                self._cache_result(features, prediction)

            return ModelPrediction(
                price=prediction["price"],
                confidence=prediction["confidence"],
                direction=prediction["direction"],
                model_name=os.path.basename(self.model_path),
                inference_time=inference_time,
            )

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

    def _check_cache(self, features: np.ndarray) -> Optional[dict]:
        """Check cache for existing prediction result"""
        if not self.cache_manager:
            return None

        model_name = os.path.basename(self.model_path)
        try:
            config = {
                "confidence_scale_factor": self.config.confidence_scale_factor,
                "direction_threshold": self.config.direction_threshold,
                "normalization_params": self.model_metadata.get("normalization_params", {}) if self.model_metadata else {},
            }

            return self.cache_manager.get(features, model_name, config)
        except Exception as e:
            # Log cache errors at debug level to aid in troubleshooting while maintaining fallback behavior
            logging.debug(
                "Cache lookup failed for model %s: %s: %s. Falling back to model inference.",
                model_name, type(e).__name__, str(e)
            )
            return None

    def _cache_result(self, features: np.ndarray, prediction: dict[str, Any]) -> None:
        """Cache prediction result"""
        if not self.cache_manager:
            return

        model_name = os.path.basename(self.model_path)
        try:
            config = {
                "confidence_scale_factor": self.config.confidence_scale_factor,
                "direction_threshold": self.config.direction_threshold,
                "normalization_params": self.model_metadata.get("normalization_params", {}) if self.model_metadata else {},
            }

            self.cache_manager.set(
                features, model_name, config,
                prediction["price"], prediction["confidence"], prediction["direction"]
            )
        except Exception as e:
            # Log cache errors at debug level to aid in troubleshooting while not affecting prediction
            logging.debug(
                "Failed to cache prediction result for model %s: %s: %s. Continuing with prediction.",
                model_name, type(e).__name__, str(e)
            )

    def _prepare_input(self, features: np.ndarray) -> np.ndarray:
        """Prepare features for model input"""
        # Handle different input shapes
        if len(features.shape) == 1:
            # 1D input: (features,) -> (1, 1, features)
            features = features.reshape(1, 1, -1)
        elif len(features.shape) == 2:
            # 2D input: (sequence_length, features) -> (1, sequence_length, features)
            features = features.reshape(1, -1, features.shape[1])
        elif len(features.shape) == 3:
            # 3D input: (batch_size, sequence_length, features) - already correct
            pass
        else:
            raise ValueError(f"Features must be 1D, 2D, or 3D array, got shape {features.shape}")

        # Normalize features if metadata contains normalization params
        if self.model_metadata and self.model_metadata.get("normalization_params"):
            features = self._normalize_features(features)

        return features.astype(np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using model metadata"""
        norm_params = self.model_metadata["normalization_params"]

        # Ensure features is 3D
        if len(features.shape) != 3:
            raise ValueError(f"Features must be 3D for normalization, got shape {features.shape}")

        for i, feature_name in enumerate(norm_params.keys()):
            if i < features.shape[2]:  # Check feature index bounds
                mean = norm_params[feature_name].get("mean", 0.0)
                std = norm_params[feature_name].get("std", 1.0)

                # Prevent ZeroDivisionError by using a minimum std value
                if std == 0.0:
                    std = EPSILON  # Small epsilon to prevent division by zero

                features[:, :, i] = (features[:, :, i] - mean) / std

        return features

    def _process_output(self, output: np.ndarray) -> dict[str, Any]:
        """Process model output into prediction"""
        # Extract scalar prediction
        if output.shape == (1, 1, 1):
            pred = output[0][0][0]
        elif output.shape == (1, 1):
            pred = output[0][0]
        else:
            pred = output.flatten()[0]

        # Denormalize prediction if needed
        if self.model_metadata and self.model_metadata.get("price_normalization"):
            pred = self._denormalize_price(pred)

        # Calculate confidence and direction
        confidence = self._calculate_confidence(pred)
        direction = self._calculate_direction(pred)

        return {"price": float(pred), "confidence": confidence, "direction": direction}

    def _denormalize_price(self, pred: float) -> float:
        """Denormalize price prediction"""
        price_params = self.model_metadata["price_normalization"]
        return pred * price_params["std"] + price_params["mean"]

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
