"""
Model registry for prediction models - replacement for existing ModelRegistry.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import PredictionConfig
from .onnx_runner import ModelPrediction, OnnxRunner

# Set up logger
logger = logging.getLogger(__name__)


class PredictionModelRegistry:
    """Registry for prediction models - replacement for existing ModelRegistry"""

    def __init__(self, config: PredictionConfig):
        """
        Initialize the prediction model registry.

        Args:
            config: Prediction engine configuration
        """
        self.config = config
        self.models: Dict[str, OnnxRunner] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load all available models from the model registry path"""
        model_path = Path(self.config.model_registry_path)

        # Find all ONNX files
        onnx_files = list(model_path.glob("*.onnx"))

        for onnx_file in onnx_files:
            model_name = onnx_file.stem
            try:
                self.models[model_name] = OnnxRunner(str(onnx_file), self.config)
                logger.info(f"✓ Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"⚠ Warning: Failed to load model {model_name}: {e}")

    def get_model(self, model_name: str) -> Optional[OnnxRunner]:
        """Get model by name"""
        return self.models.get(model_name)

    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())

    def predict(self, model_name: str, features: np.ndarray) -> ModelPrediction:
        """Run prediction with specified model"""
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        return model.predict(features)

    def get_default_model(self) -> Optional[OnnxRunner]:
        """Get the default model"""
        available_models = self.list_models()

        # Priority order: price model first, then sentiment model
        for model_name in available_models:
            if "price" in model_name.lower():
                return self.get_model(model_name)

        # Fallback to first available model
        if available_models:
            return self.get_model(available_models[0])

        return None

    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a specific model"""
        model = self.get_model(model_name)
        if model:
            return model.model_metadata
        return None

    def reload_models(self) -> None:
        """Reload all models from disk"""
        self.models.clear()
        self._load_models()

    def get_model_count(self) -> int:
        """Get the number of loaded models"""
        return len(self.models)
