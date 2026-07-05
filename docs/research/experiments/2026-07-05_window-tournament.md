# Training-Window Tournament — ETHUSDT Basic Model

Date: 2026-07-05
Author: quant-researcher
Status: IN PROGRESS (framing + protocol; results appended as each variant completes)

## Hypothesis

Given real, measured feature/price-relationship drift in this market (~2.5-3.5%/month
decay per the 2026-07-05 00:15 log synthesis) AND the finding that naive
recency-chasing (daily retrain on trailing window) was the **worst** performer in its
only head-to-head so far, we hypothesize:

> **H1**: A model trained on a *long* history with a hard cutoff (no exponential
> recency weighting, no continuous retrain) generalizes better to an unseen 2026 bear
> market than a model trained on a short recent window, because crypto regime
> variety (multiple bull/bear/chop cycles) in the training set matters more than
> recency once naive recency-chasing has already been ruled out.

This directly mirrors the research synthesis referenced in the task: Kaggle-winning
approaches for financial time series favor hard cutoffs / expanding windows over
decay-weighted recency; our own soft-regime architecture doesn't need short windows
to adapt (regime detection lives elsewhere in the stack, not in training-window
choice).

Competing hypothesis (**H0**): shorter windows win because market structure genuinely
shifted (e.g., post-2024-halving liquidity/microstructure) and older regimes (2017-19
ICO/exchange-era ETH, 2020-21 DeFi summer, 2022 Terra/FTX contagion) are actively
harmful noise for a 2026 bear-market target.

## Metric

Primary: out-of-sample (2026-01-01 → 2026-07-04, ~185 days, unseen bear market)
backtest return using `hyper_growth` strategy wired to each candidate model via the
`latest` symlink, at prod-matched risk params (`--risk-per-trade 0.02
--max-risk-per-trade 0.03 --max-position-size 0.20`, `--initial-balance 85`).

Secondary: directional accuracy + test RMSE on the *training-time* temporal holdout
(chronological split within the ≤2025-12-31 training data — this is a model-quality
sanity check, not the decision metric), confidence-score distribution of decisions
during the OOS backtest, Sharpe/Sortino-adjacent stats (PF, MaxDD, win rate, trade
count).

## Success threshold

A variant is "ready to consider for staging" only if **all** of:
1. OOS return strictly beats both the current `2026-07-04_22h_v1` (full-history, but
   trained with an in-sample-contaminated eval per the flaw this protocol fixes) and
   a hold/no-trade baseline over the same window.
2. Trade count ≥ 15 over the 185-day OOS window (per CODE.md guidance: a high
   Sharpe/return with a tiny trade count is not a result).
3. No parameter collapse under the sensitivity checks already on file for
   `hyper_growth` risk params (not re-litigated here — this experiment isolates the
   training-window variable only).

If no variant clears (1), the honest conclusion is "drift dominates and no fixed
training window solves it" — which is itself a valid, useful result and should be
reported as such, not papered over.

## Risks of false positive

- **185 days is one bear-market draw, not a distribution.** A win here is one sample
  path, not a proof the window choice is causally better across regimes. Flag this
  explicitly in the verdict.
- **Trade count risk.** ETHUSDT hourly bear-market conditions may produce few
  qualifying entries under `hyper_growth`'s risk filters — a variant could "win" on
  a handful of trades. Report trade count prominently; do not let a high headline
  return from <15 trades pass as a real signal.
- **Confidence distribution is not causally tied to the window choice** — differences
  here are informative but secondary; do not over-read them as the mechanism.
- **Cache/provider drift**: the worktree has no local cache; if Binance API responses
  differ subtly from what produced the `2026-07-04_22h_v1` numbers (e.g., minor
  historical revisions), absolute comparisons to that run carry some noise. Noted,
  not expected to be material given both are full-history closes.

## Protocol (fixes in-sample flaw in the earlier same-day validation)

- Training data for every variant: **strictly ≤ 2025-12-31** (`--end-date 2025-12-31`
  in `atb train price`). No variant's training window may include any timestamp in
  the 2026-01-01→2026-07-04 evaluation window.
- Evaluation: identical OOS window for every variant, `hyper_growth` strategy,
  identical risk params, identical `--initial-balance`.
- Strictly sequential: one training or backtest process at a time, in an isolated
  worktree (`.claude/worktrees/window-tournament`, detached at `develop`, includes
  #867 symbol-wiring + #886 native ETHUSDT model). Nothing here touches the main
  checkout, staging, or prod.
- Fees/slippage: default `CostCalculator` settings (no debug fee-free runs).

## Variants

| Variant | `--start-date` | Approx wall-clock | Training regimes covered |
|---|---|---|---|
| `W_full` | 2017-08-17 | ~60 min | Full ETH history: ICO era, 2017-18 crash, 2020-21 DeFi bull, 2021-22 top, 2022 Terra/FTX bear, 2023-24 recovery, 2024 halving bull |
| `W_3y` | 2023-01-01 | ~25 min | Post-FTX-crash recovery through 2024 halving bull and 2025 |
| `W_18m` | 2024-07-01 | ~15 min | Pre-halving-rally through 2025 only — most "recent regime," least regime variety |

All variants: `atb train price ETHUSDT --timeframe 1h --epochs 50 --batch-size 256
--sequence-length 120 --end-date 2025-12-31 --start-date <variant>`.

---

## Results

_(populated per-variant below as each completes)_
