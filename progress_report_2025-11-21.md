# Strategy Optimization Progress Report
**Date**: 2025-11-21
**Session**: Initial Setup and Baseline Measurement

## Summary

Initiated systematic strategy optimization effort for AI Trading Bot. Successfully overcame environment challenges (offline mode, missing dependencies) and established infrastructure for backtesting optimization experiments.

## Accomplishments

### 1. Created Comprehensive ExecPlan ✅
- **Location**: `docs/execplans/maximize_risk_adjusted_returns.md`
- **Content**: Detailed plan for maximizing Sharpe ratio, reducing drawdowns, and improving win rates
- **Approach**: Systematic experimentation in batches (risk management, ML features, architecture, regime detection, signal thresholds)
- **Target Metrics**: Sharpe >1.5, Max DD <20%, Win Rate >55%

### 2. Environment Setup ✅
- Installed `atb` CLI and all dependencies (tensorflow, matplotlib, seaborn, tf2onnx, pyarrow)
- Resolved package conflicts (blinker, matplotlib)
- Configured Python 3.11 environment

### 3. Test Data Infrastructure ✅
- **Challenge**: Offline environment with no Binance API access
- **Solution**: Located test data file (`tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather`)
- **Data Volume**: 18,000 hourly candles (2 years: 2023-2024)
- **Cache Setup**: Converted feather → parquet, populated cache directory with hashed keys
- **Cache Location**: `/home/user/ai-trading-bot/cache/market_data/`
- **Files**:
  - `9274aafe30b18069...55d.parquet` (BTCUSDT 1h 2023: 8,759 candles)
  - `f51c07ae82d0c760...da64.parquet` (BTCUSDT 1h 2024: 8,784 candles)

### 4. Baseline Backtest Launched 🔄
- **Command**: `atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --start 2023-01-01 --end 2024-12-31 --initial-balance 10000`
- **Status**: Running (started 23:18 UTC)
- **Progress**: Processing 17,520 candles
- **Log File**: `baseline_btc_basic_final.log`

## Key Observations

### ML Model Performance (Preliminary)
From initial backtest logs (first ~150 candles):
- **Model Confidence**: Very low (mostly <0.1, below 0.3 threshold)
- **Signals**: Predominantly HOLD with occasional weak BUY/SELL signals
- **Regime Detection**: Working (detecting range/low_vol, trend_up/low_vol, trend_down/low_vol)
- **Inference Time**: ~37-60ms per candle (acceptable for 1h timeframe)

### Root Cause Analysis
The low confidence signals suggest:
1. **Model Quality**: Current ONNX model (`2025-10-30_12h_v1`) may be undertrained or overfitted
2. **Feature Mismatch**: Model trained on price-only features (normalized OHLCV) lacks technical indicators
3. **Threshold Calibration**: Min confidence 0.3 may be too conservative OR model truly has low predictive power
4. **Data Distribution**: Model trained on different time period than 2023-2024 backtest window

## Next Steps (Prioritized)

### Immediate (Session Continuation)
1. ✅ **Wait for baseline backtest completion** (~10-15 min remaining)
2. **Extract baseline metrics**: Sharpe, Sortino, Max DD, Win Rate, Total Return
3. **Document results** in ExecPlan baseline table

### High Priority Experiments
Based on low model confidence finding:

#### Experiment 1: Model Retraining with More Data
```bash
atb train model BTCUSDT --timeframe 1h \
  --start-date 2020-01-01 --end-date 2024-12-31 \
  --epochs 100 --batch-size 64 --sequence-length 120 \
  --auto-deploy
```
**Hypothesis**: More training data (5 years vs current) will improve generalization

#### Experiment 2: Add Technical Indicator Features
- Modify `src/ml/training_pipeline/features.py`
- Add RSI(14), MACD(12,26,9), Bollinger Bands
- Retrain model with augmented features
**Hypothesis**: Technical indicators provide signal quality boost

#### Experiment 3: Lower Confidence Threshold Test
- Create variant with `min_confidence=0.1` instead of 0.3
- Backtest on same period
**Hypothesis**: Model predictions may be directionally correct even at low confidence

#### Experiment 4: Ensemble Strategy
- Combine ml_basic + technical indicators (RSI oversold/overbought)
- Only trade when both agree
**Hypothesis**: Hybrid approach reduces false signals

### Medium Priority
- Train ETHUSDT basic model (currently only sentiment models exist)
- Run ml_adaptive baseline
- Test different position sizing approaches (fixed fraction vs confidence-weighted)

### Documentation
- Update ExecPlan with baseline results
- Create decision log entries for each experiment
- Track Surprises & Discoveries section

## Blockers & Constraints

### Resolved ✅
- ✅ Offline environment → Used cached test data
- ✅ Missing dependencies → Installed via pip
- ✅ Cache directory mismatch → Copied to correct location

### Current
- ⏳ Backtest still running (expected ~5-10 min more)
- ⚠️ ETHUSDT lacks basic model (only has sentiment models)
- ⚠️ No live API access for real-time validation

### Known Limitations
- **Data Window**: Limited to 2023-2024 (2 years)
- **Offline Mode**: Cannot download fresh data or test on recent market conditions
- **No Database**: Backtests not persisted to PostgreSQL (running in-memory only)
- **Single Symbol**: ETHUSDT experiments blocked until model trained

## Files Created/Modified

### Created
- `docs/execplans/maximize_risk_adjusted_returns.md` - Master optimization plan
- `cache/market_data/*.parquet` - Cached BTCUSDT hourly data
- `baseline_btc_basic_final.log` - Backtest output log
- `progress_report_2025-11-21.md` - This file

### Modified
- None (all changes in new files or cache)

## Resource Usage
- **Token Budget**: ~88k / 200k (44% used)
- **Time**: ~30 minutes for setup + baseline launch
- **Disk**: ~1.6MB cache data

## Recommendations for Next Session

1. **Complete baseline measurement**: Finish BTCUSDT ml_basic, run ml_adaptive
2. **Quick wins first**: Test confidence threshold adjustment (fastest experiment)
3. **Model retraining**: Allocate sufficient time (model training takes 30-60 min)
4. **Systematic logging**: Use ExecPlan Decision Log for all experiment rationale
5. **Commit often**: After each successful experiment, commit with clear messages

## Code Quality Notes
- All setup work used existing infrastructure (no code changes needed)
- Followed CLAUDE.md guidelines: used `atb` CLI, no temporary scripts
- Test data already existed in repository
- No secrets or credentials required (offline mode)

---

**Status**: Infrastructure ready, baseline measurement in progress
**Next Action**: Wait for backtest completion, extract metrics, start experiments
**Estimated Time to First Experiment**: ~15-20 minutes (after baseline completes)
