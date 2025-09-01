# Regime Detection MVP and Roadmap

## Summary
Introduce a lightweight, conservative regime detector to classify market conditions (trend up/down/range with high/low vol overlay) and use it to modulate strategy behavior with hysteresis, minimizing churn.

## Goals (MVP)
- Implement `RegimeDetector` with:
  - Trend via rolling OLS slope on log-price weighted by R²
  - Vol via ATR percentile (rolling lookback)
  - Hysteresis: K confirmations + minimum dwell time
- Integrate into live engine behind feature flags
  - Annotate dataframe with regime columns
  - Log current regime alongside strategy decisions
  - Optionally adjust position size by regime (off by default)
- Keep default behavior unchanged (flags off)

## Acceptance Criteria
- New module `src/regime/detector.py` providing `RegimeDetector` and `RegimeConfig`
- Live engine adds columns: `trend_score`, `trend_label`, `vol_label`, `regime_label`, `regime_confidence` when enabled
- Database logs for entries include regime context (trend/vol/conf)
- Feature flags (all default false unless noted):
  - `enable_regime_detection` (default: false)
  - `regime_adjust_position_size` (default: false)
  - `regime_hysteresis_k` (default: 3)
  - `regime_min_dwell` (default: 12)
  - `regime_min_confidence` (default: 0.5)
- Test suite runs green

## Implementation Notes (MVP)
- Trend score = slope(log(price), W) × R² with R² floor; labels by sign with small threshold
- Volatility via ATR(W=14) and percentile over 252 bars; high if ≥ 70th percentile
- Confidence from z-score of trend score over 252 bars
- Hysteresis:
  - Require K consecutive labels before switching and dwell ≥ M bars
  - Only applied to trend label; vol is overlay
- Sizing adjustments (optional):
  - Long multiplier defaults: high vol ×0.8, range ×0.9, trend down ×0.7, low confidence ×0.8 (clipped to [0.2, 1.0])

## Rollout Plan
1. Ship with detection disabled by default
2. Enable in paper trading and monitor logs for false flips
3. Tune K/M thresholds and multipliers per symbol/timeframe
4. Enable position-size modulation once stable

## Follow-ups (Incremental Improvements)
- Add change-point detectors (CUSUM/Page–Hinkley) and ensemble voting
- Multi-horizon features (1h/4h/1d) with agreement gating
- Hidden Markov / HSMM for explicit duration modeling
- Regime-aware strategy mapping and hot-swapping via `StrategyManager`
  - Map regime → strategy or parameter set; add cooldown after switches
- Uncertainty handling
  - If `max_prob < p_min`: reduce exposure or switch to safest strategy
  - Drift monitors on features (KS/MMD) to alert on distribution shifts
- Backtesting support
  - Per-regime performance breakdown in reports
  - Walk-forward tuning of thresholds and dwell
- Observability
  - Dashboard for current regime and flip history; alerts on rapid flip-flops
- Data/markets
  - Include liquidity metrics (spread/depth) when available
  - Event-driven flags (macro releases), funding/oi overlays for perps

## Risks / Mitigations
- Overfitting thresholds → prefer broad, conservative ranges; test across years
- Flip-flopping in noisy ranges → increase K/M and require multi-horizon agreement
- Latency/CPU → keep OLS windows small and incremental; compute once per loop

## How to Use (for now)
- Toggle detection: set `enable_regime_detection=true` in `feature_flags.json`
- To size by regime: set `regime_adjust_position_size=true`
- Inspect logs for fields: `regime_trend`, `regime_vol`, `regime_conf`

## Future Issue(s) to File
- "Regime-aware hot-swapping": define mapping, dwell/cooldown policies, and tests
- "Ensemble detectors": integrate CUSUM/HMM and calibrate probabilities
