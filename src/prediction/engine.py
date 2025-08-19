"""
Core Prediction Engine

This module provides the main PredictionEngine class that serves as the unified
facade for all prediction operations, orchestrating configuration, feature
engineering, and model inference.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from regime.detector import RegimeConfig, RegimeDetector

from .config import PredictionConfig
from .ensemble import SimpleEnsembleAggregator
from .exceptions import (
    FeatureExtractionError,
    InvalidInputError,
    ModelNotFoundError,
)
from .features.pipeline import FeaturePipeline
from .models.registry import PredictionModelRegistry


@dataclass
class PredictionResult:
    """Result of a prediction engine operation"""

    price: float
    confidence: float
    direction: int  # 1, 0, -1
    model_name: str
    timestamp: datetime
    inference_time: float
    features_used: int
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictionEngine:
    """Main prediction engine facade that orchestrates all components"""

    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize prediction engine with configuration

        Args:
            config: Optional prediction configuration. If None, loads from ConfigManager
        """
        self.config = config or PredictionConfig.from_config_manager()
        self.config.validate()

        # Initialize components
        self.feature_pipeline = FeaturePipeline(
            enable_sentiment=self.config.enable_sentiment,
            enable_market_microstructure=self.config.enable_market_microstructure,
            cache_ttl=self.config.feature_cache_ttl,
        )

        self.model_registry = PredictionModelRegistry(self.config)

        # Optional helpers
        self._ensemble_aggregator = None
        if getattr(self.config, "enable_ensemble", False):
            self._ensemble_aggregator = SimpleEnsembleAggregator(
                getattr(self.config, "ensemble_method", "mean")
            )
        self._regime_detector = None
        if getattr(self.config, "enable_regime_aware_confidence", False):
            self._regime_detector = RegimeDetector(RegimeConfig())

        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._feature_extraction_time = 0.0
        # Track per-model inference times
        self._model_inference_times: Dict[str, List[float]] = {}
        # Track feature extraction times for averaging
        self._total_feature_extraction_time = 0.0
        self._feature_extraction_count = 0

    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> PredictionResult:
        """
        Main prediction method - unified interface for all predictions

        Args:
            data: Input market data (OHLCV format)
            model_name: Optional specific model to use. If None, uses default model

        Returns:
            PredictionResult: Unified prediction result with metadata
        """
        start_time = time.time()

        try:
            # Validate input data
            self._validate_input_data(data)

            # Extract features
            feature_start_time = time.time()
            features = self._extract_features(data)
            feature_time = time.time() - feature_start_time
            self._feature_extraction_time = feature_time
            # Track for averaging
            self._total_feature_extraction_time += feature_time
            self._feature_extraction_count += 1

            # Get model for prediction
            model = self._get_model(model_name)

            # Make prediction (with optional ensemble)
            if self._ensemble_aggregator is None:
                prediction = model.predict(features)
                final_price = prediction.price
                final_conf = prediction.confidence
                final_dir = prediction.direction
                final_model_name = prediction.model_name
                member_preds = None
            else:
                # Run all available models for ensemble
                preds = []
                for mname in self.model_registry.list_models():
                    m = self.model_registry.get_model(mname)
                    if m is None:
                        continue
                    preds.append(m.predict(features))
                ens = self._ensemble_aggregator.aggregate(preds)
                final_price = ens.price
                final_conf = ens.confidence
                final_dir = ens.direction
                final_model_name = f"ensemble:{self.config.ensemble_method}"
                member_preds = ens.member_predictions

            # Calculate total inference time
            inference_time = time.time() - start_time

            # Predictions that exceed max_prediction_latency should return an error result,
            # even if the prediction completed successfully.
            if (
                hasattr(self.config, "max_prediction_latency")
                and isinstance(self.config.max_prediction_latency, (int, float))
                and inference_time > self.config.max_prediction_latency
            ):
                return PredictionResult(
                    price=0.0,
                    confidence=0.0,
                    direction=0,
                    model_name=final_model_name,
                    timestamp=datetime.now(timezone.utc),
                    inference_time=inference_time,
                    features_used=features.shape[1] if hasattr(features, "shape") else 0,
                    error=f"Prediction timeout after {inference_time:.3f}s (max: {self.config.max_prediction_latency}s)",
                    metadata={
                        "error_type": "PredictionTimeoutError",
                        "data_length": len(data),
                        "feature_extraction_time": feature_time,
                        "model_inference_time": None,
                        "config_version": self._get_config_version(),
                    },
                )

            # Check if cache was hit (from feature pipeline)
            cache_hit = self._was_cache_hit()

            # Optional regime-aware confidence adjustment
            adjusted_conf = final_conf
            if self._regime_detector is not None:
                annotated = self._regime_detector.annotate(data)
                _, vol_label, regime_conf = self._regime_detector.current_labels(annotated)
                if vol_label == "high_vol":
                    adjusted_conf *= 0.85
                if regime_conf < 0.5:
                    adjusted_conf *= 0.9
                adjusted_conf = float(max(0.0, min(1.0, adjusted_conf)))

            # Create unified result
            result = PredictionResult(
                price=final_price,
                confidence=adjusted_conf,
                direction=final_dir,
                model_name=final_model_name,
                timestamp=datetime.now(timezone.utc),
                inference_time=inference_time,
                features_used=features.shape[1] if hasattr(features, "shape") else 0,
                cache_hit=cache_hit,
                metadata={
                    "data_length": len(data),
                    "feature_extraction_time": feature_time,
                    "model_inference_time": None,
                    "member_models": [p.model_name for p in (member_preds or [])],
                    "config_version": self._get_config_version(),
                },
            )

            # Update performance statistics
            self._update_performance_stats(result)

            return result

        except Exception as e:
            # Calculate total time for both timeout check and error result
            total_time = time.time() - start_time

            # Check for timeout but preserve original error
            error_message = str(e)
            error_type = type(e).__name__

            if (
                hasattr(self.config, "max_prediction_latency")
                and isinstance(self.config.max_prediction_latency, (int, float))
                and total_time > self.config.max_prediction_latency
            ):
                # Add timeout information to the original error message.
                error_message = f"Prediction timeout after {total_time:.3f}s (max: {self.config.max_prediction_latency}s). Original error: {error_message}"
                error_type = f"PredictionTimeoutError+{error_type}"

            # Return error result
            return PredictionResult(
                price=0.0,
                confidence=0.0,
                direction=0,
                model_name=model_name or "unknown",
                timestamp=datetime.now(timezone.utc),
                inference_time=total_time,
                features_used=0,
                error=error_message,
                metadata={
                    "error_type": error_type,
                    "data_length": len(data) if isinstance(data, pd.DataFrame) else 0,
                },
            )

    def predict_series(
        self,
        data: pd.DataFrame,
        model_name: Optional[str] = None,
        batch_size: int = 1024,
        return_denormalized: bool = False,
        sequence_length_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict over a long OHLCV series efficiently.
        Returns a dict with keys: 'indices' (np.ndarray), 'preds' (np.ndarray), 'normalized' (bool)
        If return_denormalized is True, applies rolling-window denormalization of close based on previous window.
        """
        self._validate_input_data(data)
        # Extract features once
        features_df = self.feature_pipeline.transform(data, use_cache=True)
        original_columns = ["open", "high", "low", "close", "volume"]
        feature_columns = [c for c in features_df.columns if c not in original_columns]
        if not feature_columns:
            raise FeatureExtractionError("No feature columns found for series prediction")
        feat = features_df[feature_columns].values.astype(np.float32)
        total = len(features_df)
        seq = (
            sequence_length_override or self.config.prediction_horizons[0]
            if hasattr(self.config, "prediction_horizons")
            else 120
        )
        # Fallback to 120 if missing
        if not isinstance(seq, int) or seq <= 0:
            seq = 120
        if total <= seq:
            return {
                "indices": np.array([], dtype=int),
                "preds": np.array([], dtype=np.float32),
                "normalized": not return_denormalized,
            }

        model = self._get_model(model_name)
        session = model.session
        input_name = session.get_inputs()[0].name

        # Build windows (N, seq, features)
        try:
            from numpy.lib.stride_tricks import sliding_window_view

            windows = sliding_window_view(feat, (seq, feat.shape[1]))
            windows = windows[:, 0, :, :]
        except Exception:
            num_windows = total - seq
            windows = np.empty((num_windows, seq, feat.shape[1]), dtype=np.float32)
            for idx in range(num_windows):
                windows[idx] = feat[idx : idx + seq]

        num_windows = windows.shape[0]
        preds_norm = np.empty((num_windows,), dtype=np.float32)
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            batch = windows[start:end]
            output = session.run(None, {input_name: batch})
            out = output[0]
            preds_norm[start:end] = out.reshape(out.shape[0], -1)[:, 0].astype(np.float32)

        indices = np.arange(seq, total, dtype=int)
        if not return_denormalized:
            return {"indices": indices, "preds": preds_norm, "normalized": True}

        # Denormalize using rolling window on close based on previous seq bars
        close = features_df["close"]
        min_prev = close.shift(1).rolling(window=seq).min().values
        max_prev = close.shift(1).rolling(window=seq).max().values
        min_vals = min_prev[indices]
        max_vals = max_prev[indices]
        same_range = (max_vals == min_vals) | np.isnan(max_vals) | np.isnan(min_vals)
        preds_denorm = preds_norm * (max_vals - min_vals) + min_vals
        prev_close = features_df["close"].shift(1).values[indices]
        preds_denorm[same_range] = prev_close[same_range]
        return {"indices": indices, "preds": preds_denorm, "normalized": False}

    def predict_batch(
        self, data_batches: List[pd.DataFrame], model_name: Optional[str] = None
    ) -> List[PredictionResult]:
        """
        Batch prediction for multiple data sets

        Args:
            data_batches: List of market data DataFrames
            model_name: Optional specific model to use

        Returns:
            List[PredictionResult]: List of prediction results
        """
        results = []

        # Get model once for efficiency
        try:
            model = self._get_model(model_name)
        except Exception as e:
            # If model loading fails, return error results for all batches
            # Create individual PredictionResult objects to avoid shared mutable state
            error_results = []
            for i, _ in enumerate(data_batches):
                error_result = PredictionResult(
                    price=0.0,
                    confidence=0.0,
                    direction=0,
                    model_name=model_name or "unknown",
                    timestamp=datetime.now(timezone.utc),
                    inference_time=0.0,
                    features_used=0,
                    error=str(e),
                    metadata={
                        "error_type": type(e).__name__,
                        "batch_index": i,
                        "batch_size": len(data_batches),
                    },
                )
                error_results.append(error_result)
            return error_results

        for i, data in enumerate(data_batches):
            start_time = time.time()

            try:
                # Validate input data
                self._validate_input_data(data)

                # Extract features
                feature_start_time = time.time()
                features = self._extract_features(data)
                feature_time = time.time() - feature_start_time
                # Track for averaging
                self._total_feature_extraction_time += feature_time
                self._feature_extraction_count += 1

                # Make prediction with pre-loaded model
                prediction = model.predict(features)

                # Calculate total inference time
                inference_time = time.time() - start_time

                # Create result
                result = PredictionResult(
                    price=prediction.price,
                    confidence=prediction.confidence,
                    direction=prediction.direction,
                    model_name=prediction.model_name,
                    timestamp=datetime.now(timezone.utc),
                    inference_time=inference_time,
                    features_used=features.shape[1] if hasattr(features, "shape") else 0,
                    cache_hit=self._was_cache_hit(),
                    metadata={
                        "data_length": len(data),
                        "feature_extraction_time": feature_time,
                        "model_inference_time": prediction.inference_time,
                        "batch_index": i,
                        "batch_size": len(data_batches),
                    },
                )

                # Update performance statistics
                self._update_performance_stats(result)

                results.append(result)

            except Exception as e:
                # Create error result for this batch
                error_result = PredictionResult(
                    price=0.0,
                    confidence=0.0,
                    direction=0,
                    model_name=model_name or "unknown",
                    timestamp=datetime.now(timezone.utc),
                    inference_time=time.time() - start_time,
                    features_used=0,
                    error=str(e),
                    metadata={
                        "error_type": type(e).__name__,
                        "batch_index": i,
                        "batch_size": len(data_batches),
                        "data_length": len(data) if isinstance(data, pd.DataFrame) else 0,
                    },
                )
                results.append(error_result)

        return results

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.model_registry.list_models()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model

        Args:
            model_name: Name of the model

        Returns:
            Dict containing model information
        """
        model = self.model_registry.get_model(model_name)
        if not model:
            return {}

        return {
            "name": model_name,
            "path": model.model_path,
            "metadata": model.model_metadata,
            "loaded": True,
            "inference_time_avg": self._get_model_avg_inference_time(model_name),
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return {
            "total_predictions": self._prediction_count,
            "avg_inference_time": (
                self._total_inference_time / self._prediction_count
                if self._prediction_count > 0
                else 0.0
            ),
            "cache_hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "available_models": len(self.get_available_models()),
            "avg_feature_extraction_time": (
                self._total_feature_extraction_time / self._feature_extraction_count
                if self._feature_extraction_count > 0
                else 0.0
            ),
        }

    def clear_caches(self) -> None:
        """Clear all caches"""
        self.feature_pipeline.clear_cache()
        # Reset cache hit status after clearing
        if hasattr(self.feature_pipeline, "_last_cache_hit"):
            self.feature_pipeline._last_cache_hit = False
        # Reset performance stats related to caching
        self._cache_hits = 0
        self._cache_misses = 0
        # Reset feature extraction tracking
        self._total_feature_extraction_time = 0.0
        self._feature_extraction_count = 0

    def reload_models(self) -> None:
        """Reload all models"""
        self.model_registry.reload_models()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of all components

        Returns:
            Dict containing health status of all components
        """
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Check feature pipeline
        try:
            # Test with minimal valid data
            test_data = pd.DataFrame(
                {
                    "open": [100.0, 101.0, 102.0] * 40,  # 120 rows minimum
                    "high": [102.0, 103.0, 104.0] * 40,
                    "low": [99.0, 100.0, 101.0] * 40,
                    "close": [101.0, 102.0, 103.0] * 40,
                    "volume": [1000, 1100, 1200] * 40,
                }
            )
            features = self.feature_pipeline.transform(test_data, use_cache=False)
            health["components"]["feature_pipeline"] = {
                "status": "healthy",
                "test_features_count": features.shape[1] if hasattr(features, "shape") else 0,
            }
        except Exception as e:
            health["components"]["feature_pipeline"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"

        # Check model registry
        try:
            models = self.model_registry.list_models()
            default_model = self.model_registry.get_default_model()
            health["components"]["model_registry"] = {
                "status": "healthy",
                "available_models": len(models),
                "model_names": models,
                "default_model": default_model.model_path if default_model else None,
            }
        except Exception as e:
            health["components"]["model_registry"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"

        # Check configuration
        try:
            self.config.validate()
            health["components"]["configuration"] = {
                "status": "healthy",
                "config_version": self._get_config_version(),
            }
        except Exception as e:
            health["components"]["configuration"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"

        return health

    # Private methods
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns and sufficient length"""
        if not isinstance(data, pd.DataFrame):
            raise InvalidInputError("Input data must be a pandas DataFrame")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise InvalidInputError(f"Missing required columns: {missing_columns}")

        if len(data) < 120:  # Minimum for LSTM sequence
            raise InvalidInputError(f"Insufficient data: {len(data)} rows, minimum 120 required")

        # Check for invalid values
        if data[required_columns].isnull().any().any():
            raise InvalidInputError("Input data contains null values")

        if (data[required_columns] <= 0).any().any():
            raise InvalidInputError("Input data contains non-positive values")

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features using feature pipeline"""
        try:
            # Get features from pipeline (returns DataFrame with original data + features)
            features_result = self.feature_pipeline.transform(data, use_cache=True)

            # Handle different return types from feature pipeline
            if isinstance(features_result, np.ndarray):
                # Feature pipeline returned numpy array directly
                return features_result
            elif isinstance(features_result, pd.DataFrame):
                # Feature pipeline returned DataFrame - extract feature columns
                original_columns = ["open", "high", "low", "close", "volume"]
                feature_columns = [
                    col for col in features_result.columns if col not in original_columns
                ]

                if not feature_columns:
                    raise FeatureExtractionError("No feature columns found in pipeline output")

                # Convert feature columns to numpy array
                features_array = features_result[feature_columns].values
                return features_array
            else:
                raise FeatureExtractionError(
                    f"Unexpected feature pipeline output type: {type(features_result)}"
                )
        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}") from e

    def _get_model(self, model_name: Optional[str]):
        """Get model for prediction"""
        if model_name:
            model = self.model_registry.get_model(model_name)
            if not model:
                raise ModelNotFoundError(f"Model '{model_name}' not found")
            return model

        # Use default model
        default_model = self.model_registry.get_default_model()
        if not default_model:
            raise ModelNotFoundError("No models available for prediction")

        return default_model

    def _was_cache_hit(self) -> bool:
        """Check if last operation was a cache hit"""
        # Get cache hit status directly from feature pipeline for current operation
        return self.feature_pipeline.get_last_cache_hit_status()

    def _get_config_version(self) -> str:
        """Get configuration version/hash for tracking"""
        config_str = str(self.config).encode("utf-8")
        config_hash = hashlib.sha256(config_str).hexdigest()[:12]
        return f"v1.0-{config_hash}"

    def _update_performance_stats(self, result: PredictionResult) -> None:
        """Update internal performance statistics"""
        self._prediction_count += 1
        self._total_inference_time += result.inference_time

        # Track per-model inference times
        model_name = result.model_name
        if model_name not in self._model_inference_times:
            self._model_inference_times[model_name] = []
        self._model_inference_times[model_name].append(result.inference_time)

        if result.cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def _get_model_avg_inference_time(self, model_name: str) -> float:
        """Get average inference time for specific model"""
        # Return per-model timing if available
        if model_name in self._model_inference_times and self._model_inference_times[model_name]:
            return np.mean(self._model_inference_times[model_name])

        # Fallback to global average when no model-specific data is available
        return (
            self._total_inference_time / self._prediction_count
            if self._prediction_count > 0
            else 0.0
        )
