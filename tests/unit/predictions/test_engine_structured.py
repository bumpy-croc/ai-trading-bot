from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.prediction.engine import PredictionEngine, PredictionResult
from src.prediction.models.onnx_runner import ModelPrediction


@patch("src.prediction.engine.PredictionModelRegistry")
@patch("src.prediction.engine.FeaturePipeline")
def test_predict_uses_feature_selector_with_schema(mock_pipeline, mock_registry):
    engine = PredictionEngine()

    # Mock pipeline returns a DataFrame with required columns
    df = pd.DataFrame(
        {
            "open": np.random.rand(200),
            "high": np.random.rand(200),
            "low": np.random.rand(200),
            "close": np.random.rand(200),
            "volume": np.random.rand(200),
            "close_normalized": np.random.rand(200),
            "rsi": np.random.rand(200),
        }
    )
    mock_pipeline.return_value.transform.return_value = df

    # Mock structured bundle with schema
    bundle = Mock()
    bundle.feature_schema = {
        "sequence_length": 120,
        "features": [
            {"name": "close_normalized", "required": True, "normalization": {"mean": 0, "std": 1}},
            {"name": "rsi", "required": True},
        ],
    }
    engine.model_registry.get_default_bundle.return_value = bundle

    # Runner predict gets called with selected array
    mock_runner = Mock()
    mock_pred = ModelPrediction(price=101.0, confidence=0.7, direction=1, model_name="runner", inference_time=0.01)
    mock_runner.predict.return_value = mock_pred
    engine.model_registry.get_default_runner.return_value = mock_runner

    data = df
    res = engine.predict(data)
    assert isinstance(res, PredictionResult)
    assert res.price == 101.0
    # Ensure predict was called with shaped array (1, 120, 2)
    args, _ = mock_runner.predict.call_args
    assert isinstance(args[0], np.ndarray)
    assert args[0].ndim == 3 and args[0].shape[1] == 120 and args[0].shape[2] == 2


