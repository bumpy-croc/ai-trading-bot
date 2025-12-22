# GPU Configuration for Mac Training

This repository supports GPU acceleration for training ML models on Apple Silicon Macs using Metal Performance Shaders (MPS).

## Requirements

1. **Apple Silicon Mac** (M1, M2, M3, or later)
2. **macOS 12.3+** (for MPS support)
3. **TensorFlow 2.19.0** (already in requirements.txt)
4. **tensorflow-metal plugin** (required for GPU acceleration)

### Installation

Install the tensorflow-metal plugin to enable GPU acceleration:

```bash
pip install tensorflow-metal
```

This plugin enables TensorFlow to use Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon Macs.

## How It Works

The training pipeline automatically detects and configures GPU devices when available:

- **Apple Silicon Macs**: Uses Metal Performance Shaders (MPS) backend
- **Other platforms**: Detects NVIDIA/AMD GPUs if available
- **Fallback**: Uses CPU if no GPU is detected

## Usage

GPU detection happens automatically when you run training commands. No additional configuration is needed:

```bash
# Train a model - GPU will be automatically detected and used
atb train model BTCUSDT --epochs 100

# Train price model - GPU will be automatically detected and used
atb train price BTCUSDT --epochs 100
```

## Verification

When training starts, you'll see device information in the logs:

```
✅ Apple Silicon GPU detected (Metal Performance Shaders)
   Using device: /physical_device:GPU:0
🚀 Training will use device: /physical_device:GPU:0
```

Or if no GPU is available:

```
ℹ️  No GPU detected, using CPU
🚀 Training will use CPU
```

## Mixed Precision Training

Mixed precision training (FP16) is automatically enabled when a GPU is detected to improve training speed. You can disable it with:

```bash
atb train model BTCUSDT --disable-mixed-precision
```

## Troubleshooting

### GPU Not Detected

1. **Check your Mac**: Ensure you have an Apple Silicon Mac (not Intel)
2. **Check macOS version**: Requires macOS 12.3 or later
3. **Check TensorFlow version**: Ensure TensorFlow 2.19.0 is installed
   ```bash
   pip show tensorflow
   ```
4. **Install tensorflow-metal**: The plugin is required for GPU support
   ```bash
   pip install tensorflow-metal
   ```
   
   After installation, restart your Python environment and try training again. You should see:
   ```
   ✅ Apple Silicon GPU detected (Metal Performance Shaders)
   ```
   
   If you see a warning about tensorflow-metal not being installed, GPU training will fall back to CPU.

### Verification Script

You can verify GPU detection works by running:

```python
from src.ml.training_pipeline.gpu_config import configure_gpu, get_compute_device

device = configure_gpu()
print(f"Using device: {device}")
print(f"Compute device: {get_compute_device()}")
```

## Performance Notes

- **MPS Performance**: Apple Silicon GPUs provide significant speedup for training, especially for larger models
- **Memory**: MPS uses unified memory, so training can use more memory than CPU-only training
- **Compatibility**: Most TensorFlow operations work with MPS, but some operations may fall back to CPU automatically

## Related Files

- `src/ml/training_pipeline/gpu_config.py` - GPU detection and configuration
- `src/ml/training_pipeline/pipeline.py` - Training pipeline with GPU integration

