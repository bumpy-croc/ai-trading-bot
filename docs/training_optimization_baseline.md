# ML Training Pipeline Optimization - Baseline & Results

## Overview
This document tracks the comprehensive optimization of the ML training pipeline for speed improvements while maintaining model quality.

**Goal**: Achieve 2-3x speedup in total training time without degrading model quality metrics.

## Environment
- **Hardware**: CPU only (no GPU available)
- **TensorFlow Version**: 2.20.0
- **Python Version**: 3.11.14
- **CPU Features**: AVX2, AVX512F, AVX512_VNNI, FMA, oneDNN enabled

## Baseline Analysis (Code Review)

### Current Pipeline Architecture
The training pipeline (src/ml/training_pipeline/pipeline.py) follows this flow:

```
1. Data Ingestion (ingestion.py)
   - Download price data via Binance API
   - Load sentiment data from Fear & Greed API

2. Feature Engineering (features.py)
   - Scale price features (OHLCV) using MinMaxScaler
   - Calculate technical indicators (SMA 7/14/30, RSI)
   - Scale indicators
   - Merge and scale sentiment features (optional)

3. Dataset Creation (datasets.py)
   - Create sequences using sliding windows
   - Split into train/validation (80/20)
   - Build TensorFlow datasets with caching/prefetching

4. Model Training (models.py)
   - CNN+LSTM architecture (Conv1D -> MaxPool -> Conv1D -> MaxPool -> LSTM -> LSTM -> Dense)
   - Adam optimizer with MSE loss
   - Early stopping and learning rate reduction callbacks

5. Artifact Saving (artifacts.py)
   - Save Keras model
   - Export to ONNX
   - Generate plots (optional)
   - Validate robustness (optional)
```

### Identified Bottlenecks (Code Analysis)

#### 1. Data Loading Bottlenecks
**Location**: `src/ml/training_pipeline/ingestion.py`

Issues:
- Synchronous API calls (no parallelization)
- Downloads happen sequentially
- No connection pooling or retry optimization
- File I/O could be optimized (read_csv vs read_feather)

**Estimated Impact**: Medium (10-20% of total time)

#### 2. Feature Engineering Bottlenecks
**Location**: `src/ml/training_pipeline/features.py`

Issues:
- Multiple MinMaxScaler instances created and fitted separately
- Rolling operations (SMA, RSI) are computed sequentially
- DataFrame operations use default dtypes (float64) instead of float32
- No parallelization of independent feature calculations
- Scaling happens after rolling operations (optimal), but could pre-allocate arrays

**Code Example (Current)**:
```python
for feature in price_features:
    scaler = MinMaxScaler()  # New scaler each time
    data[f"{feature}_scaled"] = scaler.fit_transform(data[[feature]])
```

**Estimated Impact**: High (20-30% of total time)

#### 3. Dataset Creation Bottlenecks
**Location**: `src/ml/training_pipeline/datasets.py`

Issues:
- Uses sliding_window_view (good!) but no result caching
- Creates new tf.data.Dataset each time (no persistence)
- Shuffle buffer size is fixed (2048) - could be optimized per dataset size
- No prefetch tuning (uses AUTOTUNE which is good)

**Estimated Impact**: Low-Medium (5-10% of total time)

#### 4. Model Architecture Bottlenecks
**Location**: `src/ml/training_pipeline/models.py`

Issues:
- Model architecture is reasonable but not optimized for speed
- Uses standard Conv1D (could use depthwise separable convolutions)
- LSTM layers are heavy (could test GRU or smaller hidden dims)
- No batch normalization (could help convergence)
- Adam optimizer uses default learning rate (could optimize schedule)

**Estimated Impact**: High (40-50% of total time - training dominates)

#### 5. Training Loop Bottlenecks
**Location**: `src/ml/training_pipeline/pipeline.py`

Issues:
- Mixed precision is enabled but requires GPU (disabled on CPU)
- XLA JIT compilation enabled but may not work on all CPUs
- No gradient accumulation for larger effective batch sizes
- Early stopping patience is fixed (15 epochs)
- Learning rate schedule is reactive (ReduceLROnPlateau) not proactive

**Estimated Impact**: Very High (50%+ of total time)

#### 6. Artifact Saving Bottlenecks
**Location**: `src/ml/training_pipeline/artifacts.py`

(Need to read this file to analyze)

**Estimated Impact**: Low (5% of total time)

### Performance Metrics (Expected Baseline)

Based on code analysis and typical training times:

| Stage | Estimated Time (30 days, 50 epochs) | Percentage |
|-------|-------------------------------------|-----------|
| Data Download | 5-10s | 5% |
| Feature Engineering | 10-15s | 10% |
| Dataset Creation | 5-10s | 5% |
| Model Training | 120-180s | 75% |
| Artifact Saving | 5-10s | 5% |
| **Total** | **145-225s** | **100%** |

**Key Metrics to Track**:
- Total training time
- Time per epoch
- Data loading time
- Feature engineering time
- Dataset creation time
- Memory usage
- Model quality (validation loss, test RMSE, MAPE)

## Optimization Strategy

### Phase 1: Low-Hanging Fruit (Quick Wins)
1. ✅ Use float32 instead of float64 in feature engineering
2. ✅ Optimize DataFrame dtypes
3. ✅ Batch MinMaxScaler operations
4. ✅ Pre-allocate arrays where possible
5. ✅ Vectorize all feature calculations

**Target**: 15-20% speedup

### Phase 2: Data Pipeline Optimization
1. ✅ Implement parallel data downloads
2. ✅ Optimize file I/O (prefer feather over CSV)
3. ✅ Cache preprocessed features
4. ✅ Parallelize independent feature calculations

**Target**: 20-30% speedup (cumulative)

### Phase 3: Model & Training Optimization
1. ✅ Test smaller/faster model architectures
2. ✅ Implement better learning rate schedules (OneCycleLR)
3. ✅ Optimize batch size
4. ✅ Add batch normalization for faster convergence
5. ✅ Test GRU vs LSTM (GRU is typically faster)

**Target**: 40-50% speedup (cumulative)

### Phase 4: Advanced Optimizations
1. ✅ Implement smart caching for features
2. ✅ Create training presets (fast/balanced/quality)
3. ✅ Optimize ONNX export
4. ✅ CPU threading optimization

**Target**: 2-3x speedup (cumulative)

## Next Steps
1. Implement optimizations systematically
2. Create synthetic benchmark data for testing
3. Measure improvements at each stage
4. Validate model quality doesn't degrade
5. Document all changes and results
