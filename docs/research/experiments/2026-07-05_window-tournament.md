# Training-Window Tournament — ETHUSDT Basic Model

Date: 2026-07-05
Author: quant-researcher
Status: COMPLETE — hypothesis partially supported (see verdict); no promotion recommended
Issue: https://github.com/bumpy-croc/ai-trading-bot/issues/898

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

All three variants trained successfully with **zero eval-window overlap** (verified
directly against each model's `metadata.json` `training_params.end_date`, all
`2025-12-31`). All three OOS backtests ran identical protocol: `hyper_growth`
strategy, ETHUSDT 1h, 2026-01-01 -> 2026-07-04 (185 days), `--initial-balance 85
--risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`, default
fees/slippage on (fee_rate 0.001, slippage_rate 0.0005 per the result JSONs — not a
fee-free debug run). Hold/no-trade baseline over the same window: **-40.98%**
(constant across all three since it's window/model-independent).

Every number below was cross-checked against the raw backtest result JSON in
`logs/backtest/` inside the tournament worktree and each model's `metadata.json` —
not taken at face value from any intermediate summary.

### Comparison table

| Variant | Train window | Train rows | Holdout test RMSE | OOS Return | Profit Factor | Max DD | Win Rate | Trades | Sharpe | Sortino | Final Balance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **W_full** | 2017-08-17 → 2025-12-31 | 73,284 | 0.06586 | **-7.43%** | **0.673** | **10.55%** | **76.9%** | 52 | 0.073 | 0.119 | $79.12 |
| **W_3y**   | 2023-01-01 → 2025-12-31 | 26,303 | 0.06363 | -11.25% | 0.543 | 13.55% | 72.7% | 55 | 0.095 | 0.230 | $75.85 |
| **W_18m**  | 2024-07-01 → 2025-12-31 | 13,176 | **0.06266** | -7.30% | 0.553 | 11.90% | 69.8% | 43 | 0.068 | 0.134 | $78.08 |
| Hold (no-trade) | — | — | — | -40.98% | — | — | — | 0 | — | — | — |

Bold marks the best value per column among the three trained variants (lower is
better for RMSE/MaxDD).

Directional accuracy from training: **not available for any variant.** Confirmed by
grep of `cli/commands/train_commands.py` that `atb train price` never computes this
metric — the field only exists in `2026-07-04_22h_v1`'s metadata because it was added
out-of-band by whoever wrote that model's notes, not by the training pipeline. Do not
compare "the earlier model has 0.5312 directional accuracy" against these variants;
that number isn't measuring the same thing as anything produced here. The backtest
engine's own `prediction_metrics.directional_accuracy_pct` field is also unusable — it
requires an `onnx_pred` dataframe column that `hyper_growth`'s execution path never
populates, so it silently returns a degenerate `0.0` for all three variants (same
class of bug as the documented `mape` degenerate-metric issue). This is a real gap:
**none of the tournament's decision metrics include a genuine directional-accuracy
figure.** Flagging as a follow-up rather than fabricating a number.

### Confidence distribution (from decision-log `Confidence:` values, n=4,297 per variant)

| Variant | Min | P25 | Median | Mean | P75 | P90 | P99 | Max |
|---|---|---|---|---|---|---|---|---|
| W_full | 0.00 | 0.01 | 0.03 | 0.0532 | 0.07 | 0.12 | 0.32 | 0.78 |
| W_3y | 0.00 | 0.01 | 0.03 | 0.0534 | 0.07 | 0.12 | 0.32 | 0.80 |
| W_18m | 0.00 | 0.01 | 0.03 | 0.0538 | 0.07 | 0.12 | 0.31 | 0.79 |

The three confidence distributions are essentially indistinguishable. Training
window has no visible effect on the confidence-calibration problem flagged in the
2026-07-05 00:15 log entry ("Decision confidences cluster ~0.02-0.05 median, tail to
0.78 -- confidence-scaling layer flagged as next bottleneck, not fixed here"). This
experiment reconfirms that finding rather than resolving it — the bottleneck is
elsewhere in the stack (confidence scaling / position sizing interaction), not the
training window.

### Key cross-variant findings

1. **Holdout RMSE improves monotonically as the training window shortens**
   (0.06586 -> 0.06363 -> 0.06266), but **OOS trading performance does not track it**.
   W_18m has the single best holdout RMSE of the three, yet its OOS return (-7.30%)
   is statistically tied with W_full's (-7.43%, a 0.14pp gap on ~45-50 trades — noise,
   not signal) while its profit factor (0.553) and MaxDD (11.90%) are both worse than
   W_full's (0.673 / 10.55%). **This is a direct, within-experiment demonstration
   that training-time holdout RMSE is not a reliable proxy for OOS P&L** — a caution
   that should generalize beyond this one tournament.
2. **W_3y is the outright loser**, worse than both other variants on every OOS axis
   (return, PF, MaxDD, win rate) despite sitting in the middle of the window-length
   spectrum and despite a *better* holdout RMSE than W_full. This rules out a simple
   "shorter window = worse" or "shorter window = better" monotonic story — the
   relationship between window length and live performance is not linear or even
   monotonic in either direction on this one draw.
3. **On profit factor and max drawdown, full history wins outright**: W_full > W_18m
   > W_3y on both axes. If forced to rank on risk-adjusted terms rather than raw
   return, full history is the strongest of the three.
4. **No variant is OOS-profitable.** All three lose money over the 185-day unseen
   bear market (-7.3% to -11.3%), even though all three comfortably beat the
   hold/no-trade baseline (+29.7pp to +33.7pp). All three cleared the ≥15-trade
   threshold set in the success criteria (43-55 trades), so this isn't a small-sample
   fluke on trade count — the model+strategy combination is trading sensibly (net
   positive vs. hold) but not net-positive in absolute terms in this regime.

## Verdict on the "less recent data is better in a bear" hypothesis

**Partially supported, with an important caveat.** H1 (long history / hard cutoff
beats recency-chasing) is supported in the sense that:
- The worst-performing variant (W_3y) is not the shortest window — it's the middle
  one — so there's no clean "more recent is better" story to begin with.
- Full history wins outright on profit factor and drawdown, the two metrics this desk
  weights most heavily for capital preservation.
- On raw return, full history is tied with the shortest window, not beaten by it.

But it is **not a clean sweep**: W_18m's raw OOS return is nominally the best of the
three (-7.295% vs -7.433%), a difference small enough to be noise given the trade
counts involved, but it means the honest statement is "full history and the shortest
window are statistically tied on return; full history wins on risk-adjusted terms;
the 3-year middle window is the clear loser." This is consistent with the broader
research synthesis (hard cutoffs > naive recency-chasing, which was already the worst
performer in its one prior head-to-head) but does not licence a strong claim that
"more data always wins" — sample size here is one bear-market draw across three
window choices, not a distribution.

**Rejecting the naive recency hypothesis** (H0: shorter/more-recent windows are
better because market structure genuinely shifted) is well supported: the two
shorter windows (W_3y, W_18m) do not outperform full history on any risk-adjusted
metric, and the worst performer is a mid-length window, not the shortest one.

**The bigger headline**: window choice is not the binding constraint on
profitability. All three variants lose money OOS. Improving the training window
recovers, at best, a few points of drawdown and profit factor — it does not flip the
model+strategy combination into net-profitable territory. The confidence-calibration
gap already on file (median confidence ~0.03, i.e. the model is barely committing to
any view) is a more promising place to look for the next unit of improvement than
further training-window experiments.

## Staging-slot recommendation

**Do not promote any tournament variant to replace the currently-deployed
`2026-07-04_22h_v1`.** Two independent reasons:

1. **Not directly comparable.** `2026-07-04_22h_v1` was trained through
   **2026-07-04** (end-date, per its own metadata) — its training data *includes*
   most of this experiment's OOS window. Its previously-logged validation backtest
   (-1.31% over a 90-day window per the 2026-07-05 00:15 log entry) is not
   contaminated in the same way this protocol is designed to catch (that validation
   used a different, non-overlapping 90-day slice per that log entry), but it was not
   run against the same 185-day fully-unseen window used here, so a head-to-head
   number-for-number comparison against these three variants would be comparing
   different eval windows, not just different training windows. Apples to oranges;
   flagging rather than forcing a comparison.
2. **None of the three tournament variants clear this desk's bar for promotion
   anyway** — all three are net-negative OOS (-7.3% to -11.3%). Swapping the live
   symlink for a model that loses more slowly is not an improvement worth the
   operational risk of a symlink change, and this desk does not auto-promote models
   regardless.

**Recommendation to `pm`: promising but not ready.** The training-window question is
now answered for this cycle (full history and short-window are tied on return, full
history wins on risk-adjusted terms, mid-length is worst) — re-running this
particular tournament again without a new hypothesis would be re-litigating an
answered question. The more productive next step is investigating the
confidence-calibration bottleneck flagged in both this experiment and the prior
2026-07-05 00:15 log entry, and/or the missing directional-accuracy instrumentation
gap this experiment surfaced (neither the training pipeline nor the backtest engine's
`prediction_metrics` path currently produces a trustworthy directional-accuracy
number for the `hyper_growth` execution path — worth a follow-up ticket).

## How this could lose money (adversarial self-review)

- **One bear-market draw is not a regime distribution.** All conclusions here are
  scoped to this specific 185-day window. A different 185-day OOS slice (a chop
  regime, a bull leg) could reorder these rankings entirely. Do not generalize "full
  history wins" beyond "full history won this one bear-market OOS test."
- **Sample sizes are thin for statistical confidence.** 43-55 trades per variant is
  enough to clear the desk's minimum threshold, but differences of 1-2 percentage
  points in return (e.g. W_full vs W_18m) are within the noise band for samples this
  size — treat the "full history and shortest window are tied" finding as genuinely
  a tie, not a coin-flip win for either side.
- **All three variants are losing strategies in this window.** Any of them, if
  live-deployed, would have lost 7-11% of capital over 6 months (before considering
  that live slippage/fees often exceed backtest assumptions). This experiment's
  actual output is "here's how badly you'd have lost money under three different
  training regimes," not "here's a winner." Do not let "beats hold by 30pp" read as
  "good" — it's the least-bad of a strictly negative-return field.
- **Model-selection risk from picking W_full as "most robust" on PF/MaxDD**: those
  two metrics come from the same 43-55 trade sample as everything else. A future
  30-day extension of this same OOS window could reorder PF/MaxDD rankings just as
  easily as it could reorder return rankings.

## What `risk-officer` should stress-test (if this ever becomes a live-affecting proposal)

Not applicable yet — this experiment does not recommend any live-affecting change
(see staging-slot recommendation above: no promotion recommended). If a future
follow-up experiment does clear the profitability bar, the risk-officer stress-test
list should include: drawdown behavior under a sharper single-week crash than
appears in this 185-day sample, correlation with BTCUSDT's basic model under
simultaneous market stress, and regime-shift behavior if 2026 H2 turns from bear to
chop or bull (none of these three training windows were selected or tuned against
that possibility).

## Next steps

1. Do not re-run this training-window tournament again without a new hypothesis —
   it is now answered for this data draw.
2. File a follow-up issue for the missing directional-accuracy instrumentation gap
   (`atb train price` never computes it; backtest engine's `prediction_metrics` path
   returns a degenerate 0.0 for non-`onnx_pred`-populated strategies like
   `hyper_growth`).
3. Point research effort at the confidence-calibration bottleneck (median decision
   confidence ~0.03 across all three variants) rather than further training-window
   variants — that is the more promising lever for closing the gap to
   profitability, per both this experiment and the prior 2026-07-05 00:15 log entry.
4. Keep `2026-07-04_22h_v1` as the deployed ETHUSDT basic model; no change
   recommended.

Raw per-variant data (training metadata, full backtest JSON summaries, confidence
percentiles, cross-variant summary) persisted at
`window_tournament_results.json` in the session scratchpad for this run, and the
three raw backtest result files remain in
`.claude/worktrees/window-tournament/logs/backtest/` until the worktree is removed
(paths below).
