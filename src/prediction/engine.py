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
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.regime.detector import RegimeConfig, RegimeDetector

from .config import PredictionConfig
from .ensemble import SimpleEnsembleAggregator
from .exceptions import (
    FeatureExtractionError,
    InvalidInputError,
    ModelNotFoundError,
)
from .features.pipeline import FeaturePipeline
from .features.selector import FeatureSelector
from .models.registry import PredictionModelRegistry, StrategyModel
from .utils.caching import PredictionCacheManager


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
    metadata: dict[str, Any] = field(default_factory=dict)


class PredictionEngine:
    """Main prediction engine facade that orchestrates all components"""

    def __init__(self, config: Optional[PredictionConfig] = None, database_manager=None):
        """
        Initialize prediction engine with configuration

        Args:
            config: Optional prediction configuration. If None, loads from ConfigManager
            database_manager: Optional database manager for prediction caching
        """
        self.config = config or PredictionConfig.from_config_manager()
        self.config.validate()

        # Initialize prediction cache manager if enabled
        self.cache_manager = None
        if self.config.prediction_cache_enabled and database_manager:
            self.cache_manager = PredictionCacheManager(
                database_manager,
                ttl=self.config.prediction_cache_ttl,
                max_size=self.config.prediction_cache_max_size
            )

        # Initialize components
        self.feature_pipeline = FeaturePipeline(
            config={
                "technical_features": {"enabled": True},
                "sentiment_features": {"enabled": self.config.enable_sentiment},
                "market_features": {"enabled": self.config.enable_market_microstructure},
            },
            use_cache=True,
            cache_ttl=self.config.feature_cache_ttl,
        )

        self.model_registry = PredictionModelRegistry(self.config, self.cache_manager)

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
        self._model_inference_times: dict[str, list[float]] = {}
        # Track feature extraction times for averaging
        self._total_feature_extraction_time = 0.0
        self._feature_extraction_count = 0

        # No cached structured selection helpers (avoid hidden state)

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

            # Extract features once; if FeatureSelector is used below we will reuse the
            # DataFrame from the pipeline to avoid duplicate work.
            feature_start_time = time.time()
            try:
                features_df_or_arr = self.feature_pipeline.transform(data, use_cache=True)
            except Exception as exc:
                raise FeatureExtractionError(f"Feature extraction failed: {exc}") from exc
            if isinstance(features_df_or_arr, np.ndarray):
                features = features_df_or_arr
                features_df = None
            else:
                features_df = features_df_or_arr
                original_columns = ["open", "high", "low", "close", "volume"]
                feature_columns = [
                    col for col in features_df.columns if col not in original_columns
                ]
                if not feature_columns:
                    raise FeatureExtractionError("No feature columns found in pipeline output")
                features = features_df[feature_columns].values
            feature_time = time.time() - feature_start_time
            self._feature_extraction_time = feature_time
            # Track for averaging
            self._total_feature_extraction_time += feature_time
            self._feature_extraction_count += 1

            # Resolve bundle for prediction and align features to its schema
            bundle = self._resolve_bundle(model_name)
            model = bundle.runner
            prepared_features = self._prepare_features_for_bundle(
                bundle, features, features_df
            )

            # Make prediction (with optional ensemble)
            if self._ensemble_aggregator is None:
                prediction = model.predict(prepared_features)
                final_price = prediction.price
                final_conf = prediction.confidence
                final_dir = prediction.direction
                final_model_name = prediction.model_name
                member_preds = None
                features_used = self._count_features_used(prepared_features)
            else:
                # Run all available structured runners for ensemble
                preds = []
                features_used = 0
                for ensemble_bundle in self.model_registry.list_bundles():
                    ensemble_features = self._prepare_features_for_bundle(
                        ensemble_bundle, features, features_df
                    )
                    if not features_used:
                        features_used = self._count_features_used(ensemble_features)
                    preds.append(ensemble_bundle.runner.predict(ensemble_features))
                ens = self._ensemble_aggregator.aggregate(preds)
                final_price = ens.price
                final_conf = ens.confidence
                final_dir = ens.direction
                final_model_name = f"ensemble:{self.config.ensemble_method}"
                member_preds = ens.member_predictions
                if not features_used:
                    features_used = self._count_features_used(features)

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
                    features_used=features_used,
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
                features_used=features_used,
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
    ) -> dict[str, Any]:
        """
        Predict over a long OHLCV series efficiently.
        Returns a dict with keys: 'indices' (np.ndarray), 'preds' (np.ndarray), 'normalized' (bool)
        If return_denormalized is True, applies rolling-window denormalization of close based on previous window.
        """
        self._validate_input_data(data)
        # Extract features once and align to the target bundle's schema when available
        try:
            features_df_or_arr = self.feature_pipeline.transform(data, use_cache=True)
        except Exception as exc:
            raise FeatureExtractionError(f"Feature extraction failed: {exc}") from exc

        bundle = self._resolve_bundle(model_name)
        model = bundle.runner

        features_df: Optional[pd.DataFrame]
        if isinstance(features_df_or_arr, pd.DataFrame):
            features_df = features_df_or_arr
        else:
            features_df = None

        base_features: np.ndarray
        if features_df is not None:
            original_columns = ["open", "high", "low", "close", "volume"]
            feature_columns = [
                c for c in features_df.columns if c not in original_columns
            ]
            if not feature_columns and not bundle.feature_schema:
                raise FeatureExtractionError(
                    "No feature columns found for series prediction"
                )
            base_features = (
                features_df[feature_columns].to_numpy(dtype=np.float32)
                if feature_columns
                else features_df.to_numpy(dtype=np.float32)
            )
        else:
            base_features = np.asarray(features_df_or_arr, dtype=np.float32)

        feat = base_features
        schema_sequence_length: Optional[int] = None
        if features_df is not None and bundle.feature_schema:
            try:
                aligned_matrix, schema_sequence_length = self._select_schema_features(
                    bundle, features_df
                )
                feat = aligned_matrix.astype(np.float32, copy=False)
            except Exception:
                feat = base_features.astype(np.float32, copy=False)
        else:
            feat = base_features.astype(np.float32, copy=False)

        total = int(feat.shape[0])
        seq = (
            sequence_length_override
            or schema_sequence_length
            or (
                self.config.prediction_horizons[0]
                if hasattr(self.config, "prediction_horizons")
                else 120
            )
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
        if features_df is None:
            raise FeatureExtractionError(
                "Denormalization requires feature pipeline to return a DataFrame"
            )
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
        self, data_batches: list[pd.DataFrame], model_name: Optional[str] = None
    ) -> list[PredictionResult]:
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
            bundle = self._resolve_bundle(model_name)
            model = bundle.runner
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
                try:
                    features_df_or_arr = self.feature_pipeline.transform(
                        data, use_cache=True
                    )
                except Exception as exc:
                    raise FeatureExtractionError(
                        f"Feature extraction failed: {exc}"
                    ) from exc

                if isinstance(features_df_or_arr, pd.DataFrame):
                    features_df = features_df_or_arr
                    original_columns = ["open", "high", "low", "close", "volume"]
                    feature_columns = [
                        col for col in features_df.columns if col not in original_columns
                    ]
                    if not feature_columns and not bundle.feature_schema:
                        raise FeatureExtractionError(
                            "No feature columns found in pipeline output"
                        )
                    base_features = (
                        features_df[feature_columns].to_numpy(dtype=np.float32)
                        if feature_columns
                        else features_df.to_numpy(dtype=np.float32)
                    )
                else:
                    features_df = None
                    base_features = np.asarray(features_df_or_arr, dtype=np.float32)

                prepared_features = self._prepare_features_for_bundle(
                    bundle, base_features, features_df
                )
                feature_time = time.time() - feature_start_time
                # Track for averaging
                self._total_feature_extraction_time += feature_time
                self._feature_extraction_count += 1

                # Make prediction with pre-loaded model
                prediction = model.predict(prepared_features)

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
                    features_used=self._count_features_used(prepared_features),
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

    def get_available_models(self) -> list[str]:
        """Get list of available model bundle keys."""

        return [bundle.key for bundle in self.model_registry.list_bundles()]

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Get information about a specific model

        Args:
            model_name: Name of the model

        Returns:
            Dict containing model information
        """
        for bundle in self.model_registry.list_bundles():
            if bundle.key == model_name:
                return {
                    "name": bundle.key,
                    "path": getattr(bundle.runner, "model_path", None),
                    "metadata": bundle.metadata,
                    "loaded": True,
                    "inference_time_avg": self._get_model_avg_inference_time(bundle.key),
                }
        return {}

    def get_performance_stats(self) -> dict[str, Any]:
        """Get engine performance statistics"""
        avg_inference_time = (
            self._total_inference_time / self._prediction_count
            if self._prediction_count > 0
            else 0.0
        )
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0.0
        )
        model_times = {
            k: np.mean(v) for k, v in self._model_inference_times.items()
        }
        avg_feature_time = (
            self._total_feature_extraction_time / self._feature_extraction_count
            if self._feature_extraction_count > 0
            else 0.0
        )

        return {
            "prediction_count": self._prediction_count,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_inference_time,
            "cache_hit_rate": cache_hit_rate,
            "feature_extraction_time": self._feature_extraction_time,
            "average_feature_extraction_time": avg_feature_time,
            "model_inference_times": model_times,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    def clear_caches(self) -> None:
        """Clear all caches (feature and prediction)"""
        # Clear feature cache
        self.feature_pipeline.cache.clear()

        # Clear prediction cache
        if self.cache_manager:
            self.cache_manager.clear()

        # Reset internal cache-related performance statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._feature_extraction_time = 0.0
        self._total_feature_extraction_time = 0.0
        self._feature_extraction_count = 0

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {}
        
        # Feature cache stats
        if self.feature_pipeline.cache:
            stats["feature_cache"] = self.feature_pipeline.cache.get_stats()
        
        # Prediction cache stats
        if self.cache_manager:
            stats["prediction_cache"] = self.cache_manager.get_stats()
        
        return stats

    def invalidate_model_cache(self, model_name: Optional[str] = None) -> int:
        """
        Invalidate prediction cache for specific model or all models.
        
        Args:
            model_name: Specific model name to invalidate, or None for all models
            
        Returns:
            Number of cache entries invalidated
        """
        if not self.cache_manager:
            return 0
            
        return self.model_registry.invalidate_cache(model_name)

    def reload_models_and_clear_cache(self) -> None:
        """Reload all models and clear prediction cache"""
        # Clear prediction cache first
        if self.cache_manager:
            self.cache_manager.clear()
        
        # Reload models
        self.model_registry.reload_models()

    def health_check(self) -> dict[str, Any]:
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
            bundles = list(self.model_registry.list_bundles())
            default_bundle = self.model_registry.get_default_bundle() if bundles else None
            health["components"]["model_registry"] = {
                "status": "healthy",
                "available_models": len(bundles),
                "default_model": getattr(default_bundle.runner, "model_path", None)
                if default_bundle is not None
                else None,
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

    def _select_schema_features(
        self, bundle: StrategyModel, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, int]:
        """Return all rows aligned to a bundle's schema with normalization applied."""

        selector = FeatureSelector(
            bundle.feature_schema,
            sequence_length=bundle.feature_schema.get("sequence_length"),
        )
        columns: list[str] = []
        normalizers: list[dict[str, float] | None] = []
        for feature_cfg in selector.ordered_features:
            name = feature_cfg.get("name")
            if not name:
                raise ValueError("Feature schema entry missing 'name'")
            required = bool(feature_cfg.get("required", True))
            if required and name not in features_df.columns:
                raise ValueError(f"Required feature '{name}' missing in pipeline output")
            if name in features_df.columns:
                columns.append(name)
                normalizers.append(feature_cfg.get("normalization"))

        if not columns:
            raise ValueError("No matching features found for selection")

        matrix = features_df[columns].to_numpy(dtype=np.float32, copy=True)
        for idx, norm in enumerate(normalizers):
            if not norm:
                continue
            mean = float(norm.get("mean", 0.0))
            std = float(norm.get("std", 1.0))
            if std == 0.0:
                std = 1e-8
            matrix[:, idx] = (matrix[:, idx] - mean) / std

        return matrix, selector.sequence_length

    def _prepare_features_for_bundle(
        self,
        bundle: StrategyModel,
        raw_features: np.ndarray,
        features_df: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """Align feature matrix to a bundle's schema when available."""

        if bundle.feature_schema and features_df is not None:
            try:
                selected, sequence_length = self._select_schema_features(
                    bundle, features_df
                )
                window = selected[-sequence_length:, :]
                if window.ndim == 2:
                    return window[None, :, :]
                return window
            except Exception:
                # Fall back to raw features if selection fails
                pass

        if isinstance(raw_features, np.ndarray):
            return raw_features.astype(np.float32, copy=False)

        return np.asarray(raw_features, dtype=np.float32)

    def _count_features_used(self, features: Any) -> int:
        """Return the feature dimension from an input array if possible."""

        if isinstance(features, np.ndarray) and features.ndim >= 1:
            return int(features.shape[-1])
        return 0

    def _resolve_bundle(self, model_name: Optional[str]) -> StrategyModel:
        """Resolve the model bundle to use for an inference request."""

        bundles = list(self.model_registry.list_bundles())

        if model_name:
            for bundle in bundles:
                if bundle.key == model_name:
                    return bundle
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        if bundles:
            try:
                return self.model_registry.get_default_bundle()
            except Exception:
                pass
            return bundles[0]

        raise ModelNotFoundError("No prediction models available")

    def _get_model(self, model_name: Optional[str]):
        """Get model runner for prediction (structured-only)."""
        bundle = self._resolve_bundle(model_name)
        return bundle.runner

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
