# Adaptive Trend Strategy - Optimization Journal

## Objective
Create a profitable adaptive trading strategy that beats BTC buy-and-hold by at least 10% margin over 5+ years (2020-2025).

**Baseline**: BTC buy-and-hold 2020-2025 = +1199.51% ($7,200 initial -> ~$93K)
**Target**: >1,310% return (buy-and-hold + 10% margin)

---

## Session 1 Findings (Previous Context)

### Architecture Discoveries
- **Single-position backtest engine**: Only allows ONE position at a time (if/elif in main loop)
- **Cascading position size caps**: Strategy (25%) -> PositionSizer (20%) -> RiskManager (15%) -> EntryHandler (10%)
- **Stop loss clamped to max 20%** by `clamp_stop_loss_pct` in `src/utils/bounds.py`
- **Exit priority**: SL > TP > TimeLimit > StrategySignal > Hold

### Iteration 1: EMA Crossover (fast=21, slow=55)
- **Result**: -7.82%, 60 trades
- **Problem**: Tiny position sizes (~3.8%), too many whipsaw trades
- **Fix needed**: Custom RiskManager/PositionSizer to bypass conservative caps

### Iteration 2: Custom RM/PS + EMA Crossover
- Created `TrendFollowingRiskManager` (90% target allocation)
- Created `TrendFollowingPositionSizer` (passes through risk_amount)
- **Result**: -9.50%, 40 trades
- **Problem**: EMA crossover still too frequent

### Iteration 3: Slower EMA (fast=50, slow=100)
- **Result**: +25.91%, 31 trades, 74% win rate
- **Problem**: Avoided 2022 crash (good!) but missed 2023-2024 bull runs

### Iteration 4: Single-EMA with Consecutive-Day Confirmation
- Rewrote signal generator to use price vs single EMA (period=100)
- Asymmetric confirmation: 5 days entry, 10 days exit
- 2% buffer zones
- **Result**: +25.60%, 39 trades, 71.8% win rate
- **Problem**: Still too many trades (target: 3-8), losses >20% despite SL

---

## Session 2 Findings

### Exit Handler Deep Dive

Investigated `/src/engines/backtest/execution/exit_handler.py` thoroughly.

**Priority ordering** (lines 477-491):
1. Stop Loss (HIGHEST) - exit price = candle_low (worst case for longs)
2. Take Profit - exit price = actual TP value
3. Time Limit - exit price = close
4. Strategy Signal - exit price = close (LOWEST)

**Critical insight**: SL exit uses `candle_low`, not the SL price itself. On volatile days, candle_low can be far below the SL level, causing realized losses >20% even with a 20% SL.

**`enable_engine_risk_exits=False`**: Completely skips SL/TP checks. Only strategy signals, time limits, trailing stops, and partial operations remain active.

### Why 20% SL Kills Trend Following on BTC

BTC regularly has 20-30% corrections DURING bull markets:
- May 2021: -53% (then recovered to ATH)
- Sep 2021: -20% (then recovered to ATH)
- Jan 2024: -20% (then recovered)

A 20% SL triggers during these corrections, forcing exit. Re-entry at higher prices loses compounding.

**Strategy change**: Disable engine SL/TP entirely. Use EMA trend signal as sole exit mechanism. The EMA naturally holds through corrections (price stays above long-period EMA during minor dips).

### Current Strategy Parameters (Pre-Fix)
```
trend_ema_period=100
entry_confirmation_days=5
exit_confirmation_days=10
entry_buffer_pct=0.02
exit_buffer_pct=0.02
stop_loss_pct=0.20
take_profit_pct=5.0
target_allocation=0.90
max_position_pct=0.95
```

---

## Iteration 5: [IN PROGRESS]

### Planned Changes
1. Disable engine SL/TP (`enable_engine_risk_exits=False`)
2. Increase EMA period to 150-200 (slower, fewer signals)
3. Increase exit confirmation to 15-20 days (hold through corrections)
4. Increase buffer to 3-5% (reduce noise)
5. Position size: 95% consistently
6. Target: 2-4 round-trip trades over 5 years

### Rationale
To beat buy-and-hold, the strategy must:
- Be IN during major bull runs (2020-2021, 2023-2024)
- Be OUT during the 2022 bear market (BTC -77%)
- Use nearly 100% of capital
- Make very few trades (minimize fee drag)

Theoretical max with perfect timing:
- 2020 ($7.2K) -> 2021 peak ($69K) = ~858% -> capital ~$95.8K
- Exit before crash, re-enter ~$20K
- 2023 ($20K) -> 2025 ($93K) = ~365% -> capital ~$445K
- Total: ~4,450% return (vs 1,199% buy-and-hold)

Even with imperfect timing, avoiding the 2022 crash should put us well above buy-and-hold.
