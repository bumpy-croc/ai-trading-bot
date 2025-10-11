# Phase 2 Benchmark Artifacts

**Date**: 2025-10-11  
**Status**: 🚨 CRITICAL REGRESSIONS IDENTIFIED

## Quick Navigation

### 📊 For Quick Assessment
- **[REGRESSION_SUMMARY.md](REGRESSION_SUMMARY.md)** - Executive summary (5 min read)
- **[METRICS_COMPARISON.md](METRICS_COMPARISON.md)** - Side-by-side metrics table

### 🔍 For Detailed Analysis
- **[PHASE2_ANALYSIS.md](PHASE2_ANALYSIS.md)** - Full behavioral analysis
- **[REGRESSION_ANALYSIS.md](REGRESSION_ANALYSIS.md)** - Technical investigation

### 📈 Raw Data
- `baseline_summary.json` - Aggregated results
- `baseline_summary.md` - Human-readable summary
- `baseline_backtest_ml_basic.json` - Backtest results
- `baseline_live_ml_basic.json` - Live trading results
- `*.csv` - Trade logs
- `*.log` - Execution logs

## Key Findings

### 🚨 Critical Issues (Block Phase 2)

1. **Trade Count Explosion**: 9 trades → 300 trades (+3,233%)
2. **Live Trading Activation**: 0 trades → 50 trades (100% activation rate)
3. **CPU Performance Regression**: Live engine +584% CPU time

### ⚠️ Warning Issues

4. **Suspicious Metric Improvements**: Better Sharpe, lower drawdown despite more trades
5. **RegimeDetector Errors**: Hundreds of missing method errors

## Root Cause

Between baseline (Sep 10) and Phase 2 (Oct 11), the codebase underwent **major refactoring** from legacy strategy architecture to component-based strategies. This introduced behavioral changes that weren't caught because:

1. Baseline captured **before** refactoring
2. No parity validation during refactoring  
3. No automated regression tests
4. Migration proposal process not followed

## Recommendations

### Immediate (This Week)
1. Capture true baseline on pre-refactor code
2. Debug trade generation logic
3. Fix RegimeDetector errors
4. Profile live engine performance

### Short-term (Next 2 Weeks)
1. Implement automated parity tests
2. Fix critical behavioral regressions
3. Validate across multiple seeds
4. Document all behavioral changes

### Long-term
1. Add CI regression checks
2. Version strategies explicitly
3. Maintain golden datasets
4. Follow incremental migration process

## Phase 2 Status

**Can we proceed to Phase 3?** ❌ **NO**

**Blockers**:
- Behavioral parity not established
- Critical regressions unresolved
- No baseline for comparison
- Live engine unsuitable for production

**Timeline**: +2-4 weeks to resolve

## File Index

### Analysis Documents
```
REGRESSION_SUMMARY.md       - Quick executive summary
REGRESSION_ANALYSIS.md      - Detailed technical analysis
PHASE2_ANALYSIS.md          - Behavioral drift investigation
METRICS_COMPARISON.md       - Side-by-side metrics
README.md                   - This file
```

### Benchmark Results (ml_basic)
```
baseline_backtest_ml_basic.json        - Backtest metrics
baseline_backtest_ml_basic_trades.csv  - Trade log
baseline_backtest_ml_basic.log         - Execution log
baseline_live_ml_basic.json            - Live trading metrics
baseline_live_ml_basic_trades.csv      - Live trade log
baseline_live_ml_basic.log             - Live execution log
```

### Benchmark Results (ml_adaptive)
```
baseline_backtest_ml_adaptive.json        - Backtest metrics
baseline_backtest_ml_adaptive_trades.csv  - Trade log
baseline_backtest_ml_adaptive.log         - Execution log
baseline_live_ml_adaptive.json            - Live trading metrics
baseline_live_ml_adaptive_trades.csv      - Live trade log
baseline_live_ml_adaptive.log             - Live execution log
```

### Summary Files
```
baseline_summary.json  - JSON aggregate of all results
baseline_summary.md    - Markdown summary table
```

## Related Documentation

- Migration Proposal: `docs/strategy_migration_proposal.md`
- Baseline Artifacts: `artifacts/strategy-migration/baseline/`
- Benchmark Script: `scripts/benchmark_legacy_baseline.py`
- Strategy Migration Guide: `src/strategies/MIGRATION.md`

## Usage

### Re-run Benchmarks
```bash
python scripts/benchmark_legacy_baseline.py \
  --strategies ml_basic ml_adaptive \
  --output-dir artifacts/strategy-migration/phase2 \
  --timeframe 1h \
  --backtest-days 30 \
  --live-steps 50
```

### Compare with Baseline
```bash
# View summary
cat artifacts/strategy-migration/baseline/baseline_summary.md
cat artifacts/strategy-migration/phase2/baseline_summary.md

# Compare trade counts
jq '.[] | select(.mode=="backtest") | {strategy, trades: .results.total_trades}' \
  artifacts/strategy-migration/baseline/baseline_summary.json

jq '.[] | select(.mode=="backtest") | {strategy, trades: .results.total_trades}' \
  artifacts/strategy-migration/phase2/baseline_summary.json
```

### View Metrics Comparison
```bash
cat artifacts/strategy-migration/phase2/METRICS_COMPARISON.md
```

## Contacts

For questions about:
- **Migration strategy**: See `docs/strategy_migration_proposal.md`
- **Benchmark results**: Review analysis documents in this directory
- **Technical details**: See `REGRESSION_ANALYSIS.md`

---

**Last Updated**: 2025-10-11  
**Phase**: 2 (Engine Integration) - BLOCKED  
**Next Review**: After critical issues resolved

