# Phase 1: Complete Testing & Benchmarking Guide

**Quick Reference:** How to run Phase 1 tests and what to expect
**Status:** Ready to Execute
**Last Updated:** November 2025

---

## TL;DR - Run These Commands

### Install Dependencies (One-time, 5-10 minutes)
```bash
pip install -r requirements.txt
pip install -e .
```

### Quick Test (15 minutes)
```bash
python scripts/phase1_benchmark.py --quick
```

### Full Benchmark (3-4 hours total)
```bash
python scripts/phase1_benchmark.py --full
```

### Analyze Results
```bash
# View latest model metrics
cat src/ml/models/BTCUSDT/price/latest/metadata.json | jq '.evaluation_results'

# Compare all models
python -c "
import json
from pathlib import Path
for p in Path('src/ml/models/BTCUSDT/price').glob('*/metadata.json'):
    m = json.load(open(p))
    print(f\"{m['training_params']['architecture']}: RMSE={m['evaluation_results'].get('test_rmse', 'N/A')}\")
"
```

---

## Environment Requirements

### ✅ What You Need

1. **Python 3.11+** ← Have it
2. **TensorFlow 2.19** ← Need to install (`pip install tensorflow`)
3. **~2GB free disk space** ← For models and data
4. **8GB+ RAM** ← For training
5. **GPU (optional)** ← 3-5x faster training

### ⚙️ Installation Steps

```bash
# Step 1: Install all dependencies (~5-10 min)
pip install -r requirements.txt

# Step 2: Install CLI
pip install -e .

# Step 3: Verify
atb --version
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__}')"
```

**Note:** TensorFlow is ~500MB and may take 5-10 minutes to install.

---

## Three Testing Phases

### Phase 1A: Quick Validation ⚡ (15 minutes)

**What:** Verify all models can train without errors
**How:** 7 days of data, 5 epochs

```bash
# Automated
python scripts/phase1_benchmark.py --quick

# OR Manual
atb train model BTCUSDT --model-type cnn_lstm --days 7 --epochs 5
atb train model BTCUSDT --model-type attention_lstm --days 7 --epochs 5
atb train model BTCUSDT --model-type tcn --days 7 --epochs 5
```

**Success Criteria:**
- ✅ All 3 models complete without errors
- ✅ Models saved to registry
- ✅ Inference time < 100ms

**Expected Time:** 15-20 minutes total

---

### Phase 1B: Full Training 🏋️ (3-4 hours)

**What:** Real training for meaningful comparison
**How:** 30 days of data, 20 epochs

```bash
# Automated
python scripts/phase1_benchmark.py --full

# OR Manual
atb train model BTCUSDT --model-type cnn_lstm --days 30 --epochs 20        # ~75 min
atb train model BTCUSDT --model-type attention_lstm --days 30 --epochs 20  # ~85 min
atb train model BTCUSDT --model-type tcn --days 30 --epochs 20             # ~25 min
```

**Success Criteria:**
- ✅ Attention-LSTM shows 10%+ improvement in MAE/RMSE
- ✅ TCN trains 3x+ faster
- ✅ All models have inference < 100ms

**Expected Time:**
- **CPU:** 3-4 hours total
- **GPU:** 45-60 minutes total

---

### Phase 1C: Benchmark Suite 📊 (30 minutes)

**What:** Systematic comparison with synthetic data
**How:** Automated pytest suite

```bash
# Run all benchmarks
pytest tests/benchmark/test_model_architectures.py -v

# Specific comparisons
pytest tests/benchmark/test_model_architectures.py::test_attention_lstm_vs_baseline -v
pytest tests/benchmark/test_model_architectures.py::test_inference_speed_benchmark -v
```

**What This Tests:**
- Model creation and compilation
- Training on synthetic data
- Parameter counts and model sizes
- Inference speed profiling

**Expected Time:** 20-30 minutes

---

## Expected Results

### 📊 Performance Targets

Based on research (50+ papers), here's what to expect:

| Metric | CNN-LSTM (Baseline) | Attention-LSTM (Target) | TCN (Target) |
|--------|-------------------|------------------------|-------------|
| **RMSE** | 1.0x | **0.85-0.88x** ✓ | 0.95-1.05x |
| **MAE** | 1.0x | **0.85-0.88x** ✓ | 0.95-1.05x |
| **Directional Acc** | ~60% | **~65%** ✓ | ~62% |
| **Training Time** | 100% | ~110% | **~30%** ✓ |
| **Inference** | 50ms | 70ms | **30ms** ✓ |

✓ = Expected significant improvement

### 🎯 Success Criteria

**Attention-LSTM:**
- ✅ 12-15% better MAE/RMSE
- ✅ +5pp directional accuracy
- ⚠️ ~10% slower training (acceptable)

**TCN:**
- ✅ 3-5x faster training
- ✅ <50ms inference
- ✓ Competitive accuracy

**If Results Match Research:**
→ **Proceed to Phase 2** (Ensemble implementation)

---

## Quick Decision Guide

### ✅ If Attention-LSTM improves by 10%+
**Action:** PROCEED TO PHASE 2
- Implement ensemble stacking
- Expected additional 6-18% improvement
- Deploy to production

### ⚠️ If improvement is 5-10%
**Action:** CONDITIONAL
- Good but verify on longer data (90 days)
- Consider deployment
- May still proceed to Phase 2

### ⚠️ If improvement < 5%
**Action:** INVESTIGATE
- Try deep variant
- Train on longer data (365 days)
- Check hyperparameters
- Review data quality

### ❌ If no improvement or worse
**Action:** DEBUG
- Check implementation
- Verify data preprocessing
- Review feature engineering
- Consult research papers

---

## Files & Scripts

### Training Scripts

**Automated Benchmarking:**
```bash
scripts/phase1_benchmark.py
  --quick          # 15 min validation
  --full           # 3-4 hour full benchmark
  --analyze-only   # Just analyze existing results
```

**Manual Training:**
```bash
# Via CLI
atb train model SYMBOL --model-type TYPE --days N --epochs M

# Options:
  --model-type: cnn_lstm | attention_lstm | tcn | tcn_attention
  --model-variant: default | lightweight | deep
  --days: 7 (quick) | 30 (standard) | 365 (full)
  --epochs: 5 (quick) | 20 (standard) | 100 (full)
```

### Analysis Scripts

**Compare Models:**
```bash
# Simple comparison
for model in src/ml/models/BTCUSDT/price/*/metadata.json; do
    jq '{arch: .training_params.architecture, rmse: .evaluation_results.test_rmse}' $model
done

# Detailed analysis (see PHASE1_RESULTS_GUIDE.md)
```

**Visualization:**
```python
# See PHASE1_RESULTS_GUIDE.md for plotting code
```

---

## Documentation

### Complete Guides

1. **`PHASE1_INTEGRATION.md`** - Integration and usage
2. **`PHASE1_RESULTS_GUIDE.md`** - Detailed results interpretation ← **Read this**
3. **`ml_model_research_report.md`** - Research findings and expectations
4. **`ml_model_implementation_guide.md`** - How to use and extend models

### Key Sections

**From PHASE1_RESULTS_GUIDE.md:**
- Prerequisites and setup
- Testing workflow (quick/full/benchmark)
- Results interpretation (what metrics mean)
- Expected results (based on research)
- Decision matrix (what to do with results)
- Troubleshooting common issues

**From ml_model_research_report.md:**
- Research backing for each model
- Expected performance ranges
- Architecture comparisons
- Production readiness assessment

---

## Example Workflow

### Day 1: Setup & Quick Validation (30 min)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Quick validation
python scripts/phase1_benchmark.py --quick

# Expected output:
# ✓ CNN-LSTM validation passed
# ✓ Attention-LSTM validation passed
# ✓ TCN validation passed
# All models validated successfully!
```

### Day 2: Full Training (3-4 hours, can run overnight)

```bash
# Start full benchmark (runs in background)
nohup python scripts/phase1_benchmark.py --full > benchmark.log 2>&1 &

# Check progress
tail -f benchmark.log

# Or run manually with more control
atb train model BTCUSDT --model-type cnn_lstm --days 30 --epochs 20
atb train model BTCUSDT --model-type attention_lstm --days 30 --epochs 20
atb train model BTCUSDT --model-type tcn --days 30 --epochs 20
```

### Day 3: Analysis & Decision (30 min)

```bash
# View results
cat src/ml/models/BTCUSDT/price/latest/metadata.json | jq

# Compare models
python compare_models.py  # From PHASE1_RESULTS_GUIDE.md

# Make decision
# - 10%+ improvement? → Phase 2
# - 5-10%? → Conditional
# - < 5%? → Investigate
```

---

## Troubleshooting

### Problem: TensorFlow not installed
```bash
pip install tensorflow==2.19.0 --timeout 1000
```

### Problem: Out of memory
```bash
# Use lightweight variant
--model-variant lightweight

# Reduce batch size
--batch-size 16

# Reduce sequence length
--sequence-length 60
```

### Problem: Training too slow
```bash
# Check for GPU
atb dev gpu-check

# Use TCN (much faster)
--model-type tcn

# Reduce data for testing
--days 7 --epochs 5
```

### Problem: Models not improving
- Check data quality
- Try longer training (--days 90)
- Try more epochs (--epochs 50)
- Try deep variant (--model-variant deep)

---

## Next Steps After Phase 1

### If Successful (10%+ improvement):

1. ✅ **Phase 2:** Implement ensemble stacking
   - Combine Attention-LSTM + TCN + LightGBM
   - Expected: Additional 6-18% improvement
   - Duration: 1-2 days implementation

2. ✅ **Phase 3:** Production deployment
   - Deploy best model to Railway
   - A/B test vs current model
   - Monitor performance

3. ✅ **Phase 4:** Advanced features
   - Wavelet transforms (+15% RMSE improvement)
   - Enhanced sentiment (FinBERT)
   - On-chain metrics

### If Marginal (5-10% improvement):

1. ⚠️ **Extended training:** 90-365 days
2. ⚠️ **Hyperparameter tuning:** Deep variants, more epochs
3. ⚠️ **Consider TCN:** For speed benefits alone

### If Unsuccessful (<5% improvement):

1. ❌ **Debug:** Implementation, data, features
2. ❌ **Research:** Re-review papers
3. ❌ **Consult:** Domain experts

---

## Support & Resources

### Documentation
- `docs/PHASE1_RESULTS_GUIDE.md` - Detailed interpretation guide
- `docs/ml_model_research_report.md` - Research findings
- `docs/ml_model_implementation_guide.md` - Usage guide

### Scripts
- `scripts/phase1_benchmark.py` - Automated benchmarking
- `scripts/validate_models.py` - Quick validation

### Tests
- `tests/benchmark/test_model_architectures.py` - Benchmark suite

---

## Summary Checklist

**Before Starting:**
- [ ] Python 3.11+ installed
- [ ] ~2GB free disk space
- [ ] Time allocated (15 min quick OR 3-4 hours full)

**Phase 1A - Quick Validation:**
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] CLI installed (`pip install -e .`)
- [ ] Quick validation passed (`python scripts/phase1_benchmark.py --quick`)

**Phase 1B - Full Training:**
- [ ] CNN-LSTM trained (baseline)
- [ ] Attention-LSTM trained
- [ ] TCN trained
- [ ] All models in registry

**Phase 1C - Analysis:**
- [ ] Metrics extracted
- [ ] Comparison completed
- [ ] Improvement calculated
- [ ] Decision made on Phase 2

**Documentation:**
- [ ] Results documented
- [ ] Comparison plots created
- [ ] Lessons learned noted

---

**Version:** 1.0
**Status:** Ready to Execute
**Next:** Run `python scripts/phase1_benchmark.py --quick`
