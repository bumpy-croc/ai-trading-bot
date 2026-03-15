# Hyper-Growth Strategy Implementation Plan

## Problem Statement
All existing strategies are too conservative — 0-1 trades/year with near-zero returns. Target: 500% annual returns.

## Key Findings from Report
1. **Leveraged Regime Switching** (Priority 1) — structural amplification converts small alpha into 100%+ gains
2. **Kelly Criterion** (Priority 2) — optimal compounding, Half-Kelly → Full Kelly based on confidence
3. **Deep Learning TFT** (Priority 3) — predictive alpha from multi-dimensional features
4. **Walk-Forward Validation** — robustness checking
5. **Enhanced Features** — onchain, macro, sentiment data fusion

## Baseline Analysis (1yr BTCUSDT 1h)
- ml_basic: 21 trades, 66.7% win rate, -0.03% return (tiny positions)
- All other strategies: 0-1 trades (thresholds too conservative)
- BTC hold return: -15.98%

## Root Causes of Underperformance
1. **Signal thresholds too high** — momentum needs >1-2.5% move to trigger (too rare on 1h)
2. **Position sizes too small** — ml_basic wins 66% but risks almost nothing
3. **No leverage** — 1x only, can't amplify positive edge
4. **Static risk** — same parameters regardless of market regime

## Strategy Design: "HyperGrowth"

### Architecture
```
MLBasicSignalGenerator (lowered thresholds)
    ↓
VolatilityRiskManager (wider risk bands)
    ↓
KellyCriterionSizer (Half-Kelly base, scale to Full Kelly on high confidence)
    × LeverageManager (regime-based 1x-3x)
    ↓
LeveragedPositionSizer (final position with leverage cap)
```

### Key Parameters
- **Signal Generation**: Use ML predictions with lowered confidence threshold (0.3 → 0.5 vs current 0.6+)
- **Risk Manager**: base_risk=0.12, atr_multiplier=2.5, min_risk=0.02, max_risk=0.20
- **Kelly Sizing**: kelly_fraction=0.5 (Half-Kelly), scale to 0.75 on high-confidence signals
  - min_trades=5 (faster cold start)
  - fallback_fraction=0.08 (aggressive cold start)
  - max_fraction=0.35 (allow larger positions)
- **Leverage**: Bull+LowVol=3.0x, Bull+HighVol=2.0x, Range=1.5x, Bear+LowVol=0.5x, Bear+HighVol=0.0x
- **Risk Overrides**:
  - Drawdown thresholds: [0.15, 0.30, 0.45] (wider tolerance)
  - Trailing stop: activate at 3%, trail at 1.5%
  - Partial exits: [0.08, 0.15, 0.30] at [0.20, 0.30, 0.50]

### Implementation Steps
- [x] Step 1: Register new strategies in backtest CLI
- [ ] Step 2: Create `src/strategies/hyper_growth.py` strategy factory
- [ ] Step 3: Run 1-year backtest to validate trade frequency
- [ ] Step 4: Tune parameters based on results
- [ ] Step 5: Run 5-year backtest for final validation
- [ ] Step 6: Create PR, review, merge

## Status
- Step 1: DONE (backtest CLI updated)
- Step 2: IN PROGRESS
