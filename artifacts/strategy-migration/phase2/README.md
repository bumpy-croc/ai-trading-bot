# Phase 2 Benchmark Results

**Date**: 2025-10-11  
**Branch**: Current Phase 2 implementation  
**Status**: ❌ **BEHAVIORAL PARITY NOT ACHIEVED**

## Quick Start

**Looking for the quick answer?** → Read `SUMMARY.md` (2 min)

**Want detailed analysis?** → Read `ANALYSIS.md` (10 min)

**Need metrics comparison?** → Read `COMPARISON.md` (5 min)

## What Happened?

We ran benchmarks on the current Phase 2 implementation and compared with the baseline (legacy code). The results show **identical behavioral issues** as previous Phase 2 analysis:

**Key Finding**: Component-based strategies generate 300 trades instead of 9 (baseline), indicating the adapter layer has bugs.

## Documents in This Directory

### Analysis Documents
- **README.md** - This file (navigation guide)
- **SUMMARY.md** - Quick reference (TL;DR)
- **ANALYSIS.md** - Full behavioral analysis
- **COMPARISON.md** - Side-by-side metrics tables

### Benchmark Data
- `baseline_summary.md` - Results table (human-readable)
- `baseline_summary.json` - Aggregate JSON results
- `baseline_backtest_ml_basic.json` - ml_basic backtest details
- `baseline_backtest_ml_adaptive.json` - ml_adaptive backtest details
- `baseline_live_ml_basic.json` - ml_basic live trading details
- `baseline_live_ml_adaptive.json` - ml_adaptive live trading details
- `*.csv` - Trade logs
- `*.log` - Execution logs

## The Problem

| What We Expected | What We Got | Status |
|-----------------|-------------|---------|
| 9 trades (backtest) | 300 trades | ❌ 3,233% increase |
| 0-5 trades (live) | 50 trades | ❌ 100% activation |
| Same behavior | Different behavior | ❌ Parity failed |

## Root Cause

The `LegacyStrategyAdapter` (in `src/strategies/adapters/legacy_adapter.py`) is not correctly translating component-based strategy signals into legacy-compatible behavior.

**Likely bugs**:
1. Not tracking position state (enters even when already in position)
2. Missing trade cooldown (immediate re-entry after exit)
3. Entry signals too frequent (thresholds misaligned)

## What Needs to Be Fixed?

1. **Debug adapter** - Create diagnostic script to compare signals
2. **Fix position tracking** - Don't enter if already have position
3. **Add cooldown** - Enforce minimum bars between trades
4. **Fix RegimeDetector** - Implement missing methods
5. **Optimize live engine** - Reduce 573% CPU overhead

**For detailed action plan**: Check git history for previous comprehensive action plan document.

## How to Re-run Benchmarks

```bash
# Run benchmarks
python scripts/benchmark_legacy_baseline.py \
  --output-dir artifacts/strategy-migration/phase2 \
  --strategies ml_basic ml_adaptive \
  --timeframe 1h \
  --backtest-days 30 \
  --live-steps 50

# View summary
cat artifacts/strategy-migration/phase2/baseline_summary.md

# Compare with baseline
diff artifacts/strategy-migration/baseline/baseline_summary.md \
     artifacts/strategy-migration/phase2/baseline_summary.md
```

## Success Criteria

Phase 2 is complete when:

- ✅ Trade count matches baseline (~9 trades, not 300)
- ✅ Live mode doesn't have 100% activation rate
- ✅ CPU time acceptable for production (< 5s for 50 steps)
- ✅ No RegimeDetector errors in logs
- ✅ Automated parity tests pass
- ✅ Behavioral parity validated across multiple seeds

## Current Status: ❌ BLOCKED

**Blocking issues**:
- Behavioral parity not achieved
- Adapter translation bugs
- Live engine performance regression
- No automated parity tests

**Timeline**: ~2-4 weeks to resolve (see action plan in previous analysis)

## Migration Proposal Compliance

Per `docs/strategy_migration_proposal.md`, Phase 2 requires:

> "Results produced by the engines must remain equivalent to the current legacy path"

**Status**: ❌ **NOT COMPLIANT**

**Violations**:
- Trade sequences not equivalent
- Risk metrics different
- Live behavior completely changed

## Related Documentation

- **Migration Proposal**: `docs/strategy_migration_proposal.md`
- **Baseline Results**: `artifacts/strategy-migration/baseline/`
- **Strategy Migration Guide**: `src/strategies/MIGRATION.md`
- **Benchmark Script**: `scripts/benchmark_legacy_baseline.py`

## Previous Analysis

Check git history for previous comprehensive documents including:
- Detailed regression analysis
- Step-by-step action plan
- AI agent brief
- Root cause investigation

## Contact

For questions about:
- **Migration strategy**: See `docs/strategy_migration_proposal.md`
- **Benchmark methodology**: See `scripts/benchmark_legacy_baseline.py`
- **Technical details**: See `ANALYSIS.md` in this directory

---

**Last Updated**: 2025-10-11  
**Phase**: 2 (Engine Integration) - BLOCKED  
**Next Review**: After adapter fixes implemented

