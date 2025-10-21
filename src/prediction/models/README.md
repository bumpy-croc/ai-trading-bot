# Prediction Models

Core prediction model components for ONNX runtime and model registry.

## Modules

- `onnx_runner.py`: ONNX model loader and inference engine
- `registry.py`: Model discovery and prediction registry for all strategies

## Usage

```python
from src.prediction.models.registry import PredictionModelRegistry
from src.prediction.config import PredictionConfig

config = PredictionConfig()
registry = PredictionModelRegistry(config)

# Make prediction with registered model
features = np.random.rand(120, 5).astype(np.float32)
result = registry.predict('btcusdt_price', features)
print(f"Price: {result.price}, Confidence: {result.confidence}")
```

## Model Discovery

Models are automatically discovered from the `src/ml` directory:
- `*.onnx` files contain the trained models
- `*_metadata.json` files contain model configuration and metadata