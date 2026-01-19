# Phase 1: Training & Benchmarking Results Guide

**Purpose:** Guide for running Phase 1 tests and interpreting results
**Last Updated:** November 2025

---

## Prerequisites

### Environment Setup

**1. Install Dependencies:**
```bash
# If not already done
pip install -r requirements.txt

# This includes:
# - tensorflow==2.19.0 (~500MB, takes 5-10 minutes)
# - numpy, pandas, scikit-learn
# - All model dependencies
```

**2. Install CLI:**
```bash
# From project root
pip install -e .

# Verify
atb --version
```

**3. Set up Database (Optional for training):**
```bash
# Only needed for logging
docker compose up -d postgres
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
```

**4. Verify GPU (Optional but Recommended):**
```bash
atb dev gpu-check

# Expected output:
# ✓ GPU available: [GPU name]
# or
# ⚠ No GPU detected - training will use CPU
```

---

## Testing Workflow

### Phase 1A: Quick Validation (10-15 minutes)

**Purpose:** Verify all models can train without errors

```bash
# Automated quick validation
python scripts/phase1_benchmark.py --quick

# Or manual tests
atb train model BTCUSDT --model-type cnn_lstm --days 7 --epochs 5
atb train model BTCUSDT --model-type attention_lstm --days 7 --epochs 5
atb train model BTCUSDT --model-type tcn --days 7 --epochs 5
```

**Expected Time:** ~2-3 minutes per model (CPU), ~1 minute (GPU)

**Success Criteria:**
- ✅ All models complete without errors
- ✅ Models saved to `src/ml/models/BTCUSDT/price/`
- ✅ ONNX exports created
- ✅ Metadata files present

**If Failures Occur:**
- Check TensorFlow installation
- Verify data can be downloaded
- Check disk space (models ~50MB each)
- Review error logs

---

### Phase 1B: Full Training (1-2 hours per model)

**Purpose:** Real training for meaningful comparison

```bash
# Automated full benchmark
python scripts/phase1_benchmark.py --full

# Or manual training
atb train model BTCUSDT --model-type cnn_lstm --days 30 --epochs 20
atb train model BTCUSDT --model-type attention_lstm --days 30 --epochs 20
atb train model BTCUSDT --model-type tcn --days 30 --epochs 20
```

**Recommended Settings:**
- **Dataset**: 30 days (minimum for meaningful results)
- **Epochs**: 20 (balance speed vs convergence)
- **Timeframe**: 1h (good balance)

**Expected Training Times (30 days, 20 epochs):**
- CNN-LSTM: 60-90 minutes (CPU), 15-20 minutes (GPU)
- Attention-LSTM: 70-100 minutes (CPU), 18-25 minutes (GPU)
- TCN: 20-30 minutes (CPU), 5-8 minutes (GPU) ← **Fastest**

---

### Phase 1C: Benchmarking Suite (30 minutes)

**Purpose:** Systematic comparison with synthetic data

```bash
# Run all benchmarks
pytest tests/benchmark/test_model_architectures.py -v

# Specific tests
pytest tests/benchmark/test_model_architectures.py::test_comprehensive_model_comparison -v
pytest tests/benchmark/test_model_architectures.py::test_attention_lstm_vs_baseline -v
pytest tests/benchmark/test_model_architectures.py::test_inference_speed_benchmark -v
```

**What This Tests:**
- Model creation and compilation
- Training on synthetic data
- Inference speed
- Memory usage
- Parameter counts

---

## Results Interpretation

### Key Metrics Explained

**1. RMSE (Root Mean Squared Error)**
- Lower is better
- Units: Same as price (dollars/cents)
- Sensitive to large errors
- **Target**: 10-15% improvement vs baseline

**2. MAE (Mean Absolute Error)**
- Lower is better
- Average absolute prediction error
- Less sensitive to outliers than RMSE
- **Target**: 12-15% improvement vs baseline

**3. MAPE (Mean Absolute Percentage Error)**
- Lower is better
- Normalized, comparable across assets
- **< 5%**: Excellent
- **5-10%**: Good
- **> 10%**: Poor

**4. Directional Accuracy**
- % of correct up/down predictions
- **> 55%**: Can be profitable
- **> 60%**: Good
- **> 65%**: Excellent

**5. Training Time**
- Total time to train model
- Compare relative speed

**6. Inference Time**
- Time per prediction
- **Target**: < 100ms for production
- **Ideal**: < 50ms

---

### Expected Results

Based on research literature, here's what to expect:

#### Baseline (CNN-LSTM)

```
Symbol: BTCUSDT (30 days, 20 epochs)
Training Time: 60-90 min (CPU)
Test RMSE: 0.0250 (example)
Test MAE: 0.0190
MAPE: 3.5%
Directional Accuracy: 60%
Inference: 50ms
```

#### Attention-LSTM (Expected Improvement)

```
Training Time: 70-100 min (CPU) [~10% slower]
Test RMSE: 0.0215 (-14%) ✓ IMPROVEMENT
Test MAE: 0.0162 (-15%) ✓ IMPROVEMENT
MAPE: 3.0% (-14%)
Directional Accuracy: 65% (+5pp) ✓
Inference: 70ms
```

**Key Indicators:**
- ✅ **12-15% better MAE/RMSE** (matches research)
- ✅ **Higher directional accuracy**
- ⚠️ Slightly slower training (~10%)
- ⚠️ Slightly slower inference (~40%)

#### TCN (Expected Speed)

```
Training Time: 20-30 min (CPU) [3-4x FASTER] ✓
Test RMSE: 0.0245 (-2%)
Test MAE: 0.0185 (-3%)
MAPE: 3.4%
Directional Accuracy: 62% (+2pp)
Inference: 30ms ✓ VERY FAST
```

**Key Indicators:**
- ✅ **3-5x faster training** (matches research)
- ✅ **Very fast inference (<50ms)**
- ✓ Competitive accuracy
- ✓ Good for real-time applications

---

### Comparison Analysis

**View Training Results:**
```bash
# Check latest model
cat src/ml/models/BTCUSDT/price/latest/metadata.json | jq '.evaluation_results'

# Compare architectures
cat src/ml/models/BTCUSDT/price/latest/metadata.json | jq '.training_params.architecture'

# View all versions
ls -la src/ml/models/BTCUSDT/price/
```

**Create Comparison Table:**
```bash
# Extract key metrics from each model
for model in src/ml/models/BTCUSDT/price/*/metadata.json; do
    echo "=== $(dirname $model) ==="
    jq '{architecture: .training_params.architecture, rmse: .evaluation_results.test_rmse, mape: .evaluation_results.mape}' $model
done
```

---

### Decision Matrix

After reviewing results, use this matrix to decide next steps:

#### Scenario 1: Attention-LSTM shows 10%+ improvement ✅

**Action:** **PROCEED TO PHASE 2**
- Implement ensemble stacking
- Deploy best model to production
- Set up A/B testing

**Why:** Clear evidence of improvement, worth investing in ensemble

#### Scenario 2: Attention-LSTM shows 5-10% improvement ⚠️

**Action:** **CONDITIONAL**
- If training time acceptable → Proceed to Phase 2
- If need more evidence → Train on longer data (90-365 days)
- Consider production deployment

**Why:** Good improvement, but verify on longer timeframes

#### Scenario 3: Marginal improvement (<5%) ⚠️

**Action:** **INVESTIGATE**
- Try different hyperparameters (deep variant)
- Train on longer data (90-365 days)
- Check data quality
- Review feature engineering
- Consider TCN for speed benefits

**Why:** Results don't match research expectations, investigate causes

#### Scenario 4: No improvement or degradation ❌

**Action:** **DEBUG**
- Check data quality and preprocessing
- Verify model implementations
- Review hyperparameters
- Ensure proper train/val/test split
- Check for bugs in feature engineering

**Why:** Results contradict research, likely implementation issue

#### Scenario 5: TCN is fastest but accuracy similar ✓

**Action:** **USE FOR SPEED-CRITICAL APPLICATIONS**
- Deploy TCN for real-time inference
- Use Attention-LSTM for batch predictions
- Consider TCN+Attention hybrid

**Why:** TCN provides speed benefits even with similar accuracy

---

## Detailed Analysis Steps

### Step 1: Validate Training Completed

```bash
# Check all models exist
ls -la src/ml/models/BTCUSDT/price/

# Expected:
# - Multiple timestamped directories
# - 'latest' symlink pointing to newest
# - Each directory has: model.keras, model.onnx, metadata.json
```

### Step 2: Extract Metrics

```bash
# Create comparison script
cat > compare_models.sh << 'EOF'
#!/bin/bash
echo "Model Architecture Comparison"
echo "=============================="
for dir in src/ml/models/BTCUSDT/price/2025-*; do
    if [ -f "$dir/metadata.json" ]; then
        arch=$(jq -r '.training_params.architecture' "$dir/metadata.json")
        variant=$(jq -r '.training_params.architecture_variant' "$dir/metadata.json")
        rmse=$(jq -r '.evaluation_results.test_rmse' "$dir/metadata.json")
        mae=$(jq -r '.evaluation_results.test_mae' "$dir/metadata.json")
        mape=$(jq -r '.evaluation_results.mape' "$dir/metadata.json")

        echo ""
        echo "Model: $arch ($variant)"
        echo "  RMSE: $rmse"
        echo "  MAE: $mae"
        echo "  MAPE: $mape%"
    fi
done
EOF

chmod +x compare_models.sh
./compare_models.sh
```

### Step 3: Calculate Improvements

```python
# Python script to calculate improvements
import json
from pathlib import Path

def compare_models():
    models_dir = Path("src/ml/models/BTCUSDT/price")

    results = {}
    for metadata_path in models_dir.glob("*/metadata.json"):
        with open(metadata_path) as f:
            metadata = json.load(f)

        arch = metadata['training_params']['architecture']
        eval_results = metadata['evaluation_results']

        results[arch] = {
            'rmse': eval_results.get('test_rmse', 0),
            'mae': eval_results.get('test_mae', 0),
            'mape': eval_results.get('mape', 0)
        }

    # Find baseline
    baseline = results.get('cnn_lstm', {})
    baseline_rmse = baseline.get('rmse', 1)

    print("\nComparison vs CNN-LSTM Baseline:")
    print("-" * 50)

    for arch, metrics in results.items():
        if arch == 'cnn_lstm':
            continue

        rmse_improvement = ((baseline_rmse - metrics['rmse']) / baseline_rmse) * 100

        print(f"\n{arch}:")
        print(f"  RMSE Improvement: {rmse_improvement:+.1f}%")

        if rmse_improvement >= 10:
            print("  ✓ EXCELLENT - Proceed to Phase 2")
        elif rmse_improvement >= 5:
            print("  ⚠ GOOD - Consider Phase 2")
        else:
            print("  ⚠ MARGINAL - Investigate further")

if __name__ == "__main__":
    compare_models()
```

### Step 4: Visualize Results

```python
# Create comparison plots
import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with actual results)
models = ['CNN-LSTM\n(Baseline)', 'Attention-LSTM', 'TCN']
rmse = [0.0250, 0.0215, 0.0245]  # Replace with actual
mae = [0.0190, 0.0162, 0.0185]   # Replace with actual
train_time = [75, 85, 25]         # Minutes

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RMSE comparison
axes[0].bar(models, rmse, color=['gray', 'green', 'blue'])
axes[0].set_ylabel('RMSE')
axes[0].set_title('Test RMSE (Lower is Better)')
axes[0].axhline(y=rmse[0], color='r', linestyle='--', label='Baseline')
axes[0].legend()

# MAE comparison
axes[1].bar(models, mae, color=['gray', 'green', 'blue'])
axes[1].set_ylabel('MAE')
axes[1].set_title('Mean Absolute Error (Lower is Better)')
axes[1].axhline(y=mae[0], color='r', linestyle='--', label='Baseline')
axes[1].legend()

# Training time
axes[2].bar(models, train_time, color=['gray', 'green', 'blue'])
axes[2].set_ylabel('Minutes')
axes[2].set_title('Training Time (30 days, 20 epochs)')

plt.tight_layout()
plt.savefig('artifacts/phase1_comparison.png', dpi=300)
print("✓ Comparison plot saved to artifacts/phase1_comparison.png")
```

---

## Troubleshooting

### Common Issues

**1. TensorFlow Import Errors**
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution:**
```bash
pip install tensorflow==2.19.0 --timeout 1000
# Large package, may take 5-10 minutes
```

**2. GPU Not Detected**
```
No GPU detected - training will use CPU
```
**Solution:**
- Install CUDA (if NVIDIA GPU available)
- Training will be slower but still work
- Consider cloud GPU (Google Colab, AWS)

**3. Out of Memory (OOM)**
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solutions:**
```bash
# Use lightweight variant
atb train model BTCUSDT --model-type attention_lstm --model-variant lightweight

# Reduce batch size
atb train model BTCUSDT --batch-size 16

# Reduce sequence length
atb train model BTCUSDT --sequence-length 60
```

**4. Training Very Slow**
**Solutions:**
- Enable GPU if available
- Use TCN instead (3-5x faster)
- Reduce data size for testing (--days 7)
- Check CPU usage (other processes)

**5. High Error Metrics**
```
MAPE > 10%, RMSE very high
```
**Possible Causes:**
- Insufficient training data
- Too few epochs
- Bad data quality
- Feature engineering issues

**Solutions:**
- Increase training data (--days 90)
- Increase epochs (--epochs 50)
- Check data quality
- Review feature engineering

---

## Next Steps Based on Results

### If All Models Validated Successfully

1. ✅ **Phase 1 Complete**
2. **Analyze which model performs best**
3. **Make Phase 2 decision** based on improvement %

### If Attention-LSTM Shows 10%+ Improvement

1. ✅ **Success!** Research predictions confirmed
2. **Proceed to Phase 2:** Ensemble implementation
3. **Document results** for future reference
4. **Prepare for production deployment**

### If Results Are Marginal

1. **Try longer training:** 90-365 days
2. **Experiment with hyperparameters:** deep variant, more epochs
3. **Check feature quality:** Are technical indicators helping?
4. **Consider TCN for speed:** Even if accuracy similar

### If Results Are Poor

1. **Debug implementation:** Verify model code
2. **Check data quality:** Is data clean and complete?
3. **Review preprocessing:** Feature normalization correct?
4. **Consult research:** Re-read implementation details

---

## Summary Checklist

- [ ] Environment set up (TensorFlow installed)
- [ ] CLI installed and working (`atb --version`)
- [ ] Quick validation passed (all models train)
- [ ] Full training completed (3 models, 30 days, 20 epochs)
- [ ] Metrics extracted and compared
- [ ] Comparison plots created
- [ ] Decision made on Phase 2
- [ ] Results documented

---

**Document Version:** 1.0
**Last Updated:** November 2025
**Status:** Complete - Ready for Testing
