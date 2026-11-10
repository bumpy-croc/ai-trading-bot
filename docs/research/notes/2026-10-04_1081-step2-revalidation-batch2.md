# #1081 Step 2 — re-validation batch 2 (items 2, 3, 7)

**Date**: 2026-10-04
**Researcher**: quant-researcher
**Status**: PARTIAL. Item 7 reproduced exactly; item 3 conclusion held but point numbers moved; item 2 not re-run (cost).
Items 4, 5, 6, 8 still pending.
**Worktree**: `.claude/worktrees/agent-ada011782934a2317`, branch `research/1081-revalidation`, off
`origin/develop @ 812b5701`.
**Parent**: GH #1081. Item 1 (short suppression) was closed out in
`2026-08-24_short-suppression-counterfactual-rerun.md`.

## Provenance and method

- Every run printed `src.__file__` and `git rev-parse HEAD` first. All runs here:
  `src` under `.claude/worktrees/agent-ada011782934a2317/src`, HEAD `812b5701`, `PYTHONPATH` unset,
  interpreter `/Users/alex/Sites/ai-trading-bot/.venv/bin/python`. `python tools/install_worktree_shim.py --check`
  passed. #1080's `verify_source_root()` is live on this commit, so a wrong-checkout run would have raised.
- `atb backtest` through the CLI, one run at a time, each under `nice -n 15`. Fees and slippage on
  (default `CostCalculator`). No sentiment. `--drawdown-cap-mode enforce` (CLI default).
- Model: `ETHUSDT/basic/2026-07-04_22h_v1` (the only version in the registry).
- Machine load average was 35-48 for the whole session. A 365d/1h backtest ran at about 2.6 bars/s.

### Controls for engine changes since the originals

Each original predates the following changes, so a changed number is not automatically a
#1070 effect. All four apply to every run below:

| Change | Effect on a re-run |
|---|---|
| #1020 (merged 2026-07-14) HyperGrowth/ETHUSDT long-only | Originals dated 07-12 ran with shorts enabled. A current run is long-only. |
| #1073/#986 + #1089 early-stop cap from the risk manager, 20% | Truncates any run whose drawdown passes 20%. Not reached by items 3/7. |
| #1246 partial-exit trades record whole-position P&L | Per-trade P&L in the printed stats is not comparable to originals. |
| #1255/#1257 run timeframe reaches the ML preset | Not material for 1h. |

The original docs for items 3 and 7 are observational comparisons, not preregistered studies. They
carry no thresholds, so there is no pass/fail bar to apply. The verdict is "same or changed" against the
original reported figures.

## Item 3 — live-trade-review + parity-gap-investigation (shared comparator)

Original claims (`2026-07-12_live-trade-review.md`, `2026-07-12_parity-gap-investigation.md`): the matched-config
backtest does not reproduce live; backtest is quiet and barely actionable, trade count and return are
nowhere near live. This was used as evidence that live-vs-backtest divergence is far outside the charter's 15% band.

| Run | Window | Original | Re-run (develop @ 812b5701, long-only) | Verdict |
|---|---|---|---|---|
| live-trade-review | 2026-06-02 → 2026-07-12, $84 | 6 trades, -0.78%, win rate 50% | 5 trades, -1.26%, win rate 80%, MaxDD 3.85%, Sharpe 0.02, final $82.87 | SAME conclusion, numbers moved |
| parity-gap | 2026-06-01 → 2026-07-01, $84 | 3 trades, +0.91%, win rate 66.7% | 4 trades, -2.27%, win rate 75%, MaxDD 3.84%, Sharpe 0.03, final $82.65 | Sign flipped, same conclusion |

Reading:

- The conclusion both docs draw is that the backtest is a near-flat, low-count series that is nowhere near live's
  ~+9.0% over the same window. That still holds: 4-5 trades and a return within about 2pp of zero.
  The divergence from live is, if anything, larger now.
- The point numbers did not reproduce. The cause is not isolated. The like-for-like confound is #1020: the
  originals ran with shorts enabled, the re-run is long-only (the current prod config). A shorts-enabled run
  would separate #1020 from #1070-style staleness; I did not have the time. So I cannot say whether
  the original numbers were stale-checkout artifacts. I can say the original numbers are not the numbers
  to quote for the current prod config.
- Internal inconsistency worth a follow-up: for the 06-01 → 07-01 run, the CLI prints Total Return -2.27% and
  the Yearly Returns row prints -1.60% for the same run. This is the post-#1246 whole-position P&L
  path vs. the balance path. I did not chase it.
- Shipped decisions affected: none. Neither doc backs a config change. Both only feed the parity
  hypothesis (forming-bar gating), which is untouched.

## Item 7 — 2026-08-09 ETHUSDT retrain, incumbent OOS leg (#1048)

Original claim (PR #1048): incumbent `2026-07-04_22h_v1`, hyper_growth ETHUSDT 1h, 2026-06-10 → 2026-08-09,
`--initial-balance 85 --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`:
PF 0.4814, -1.33%, MaxDD 3.26%, 6 trades, win rate 83.33%, Sharpe 0.03, final $83.86.

Re-run (develop @ 812b5701, same flags, `--model-version 2026-07-04_22h_v1`, long-only as in prod):

| Metric | Original | Re-run | Match |
|---|---:|---:|---|
| Profit factor | 0.4814 | 0.4814 | exact |
| Total return | -1.33% | -1.34% | rounding ($83.86 / $85) |
| Max drawdown | 3.26% | 3.26% | exact |
| Trades / win rate | 6 / 83.33% | 6 / 83.33% | exact |
| Final balance | $83.86 | $83.86 | exact |
| Sharpe | 0.03 | 0.02 | rounding-level |
| Hold return | +18.34% | +16.55% | cache end date differs, not a strategy number |

Verdict: SAME. The incumbent leg reproduces to the cent. This also clears the #1048 retain-incumbent table of the
#1070 concern, and it shows the early-stop cap (#1089), whole-position P&L (#1246) and timeframe threading
(#1255) did not move a long-only 60-day run with drawdown 3.26%. Shipped decisions affected: none. The decision
rested on test RMSE (cloud-isolated, never at risk); the OOS table was informational and challenger columns
were "not measurable" anyway.

## Item 2 — hypergrowth-365d-drawdown-stress-review (NOT re-run)

Original claims: 365d baseline -20.15% / 21.84% MaxDD / 104 trades; CF-A (ratified tiers) 17.01% MaxDD;
CF-B (`--max-drawdown 0.20`) 20.50% MaxDD / 81 trades.

What I did: launched the exact original command (`--start 2025-07-04 --end 2026-07-04`, same flags) in a
detached worktree at the original commit `e1d24239`, so the old code and the corrected provenance are both
known. It was aborted after about 100 decisions: at the measured 2.6 bars/s (load average 35-48) the 8,760-bar window
costs about 55 minutes per run, and the item needs three. This is a cost call, not a result. The 365d window is
what the original pre-committed; I did not shorten it.

What the aborted run did establish (read from its log, not from a result):

- At `e1d24239` the registry has no `ETHUSDT/basic` model. The run scored ETHUSDT with
  `BTCUSDT/basic/2025-10-30_12h_v1` (`latest`). The native ETHUSDT model was added on 2026-07-05 (#886, `6a8e8ff0`),
  the day after the original review.
- Every clean-provenance run since (the 08-13 tier-restore doc, and #1101's 12-month arm) used the native model
  `2026-07-04_22h_v1`.

So the failed reproduction of CF-A (22.23% vs 17.01%) and of the baseline has a non-#1070 candidate cause that the
audit did not list: the model. This is a candidate, not a finding. It was not isolated. The 08-13 shorts-enabled
re-run gave 116 trades / -17.75% / 18.99% against the original 104 / -20.15% / 21.84%, and that run had correct
provenance, so the residual gap is real whatever #1070 did.

Still needed to close item 2: run the baseline at `e1d24239` to completion (about 55 minutes at current load, much less
on an idle machine). If it reproduces -20.15% / 21.84% / 104, the original was clean and the cause of the gap is
model plus engine drift. If not, #1070 is confirmed for this doc.

Shipped decisions affected: the tier-restore and cap decisions already rest on the 08-13 clean-provenance numbers
(22.23% still breaching the cap), not on CF-A. So nothing shipped depends on the unverified 17.01% any more.

## Still pending

- Item 2: baseline/CF-A/CF-B at `e1d24239` (above).
- Item 4 `kelly-active-evaluation`, item 5 `tournament-v2-corrected`, item 6 `confidence-calibration`,
  item 8 #1078 incumbent legs (window 2026-06-24 → 2026-08-23, flags in `model-promotions.md`).
- A shorts-enabled in-process run of the item-3 windows, to separate #1020 from staleness.

## Recommendation

Do not run the remaining 365d items on a loaded machine. Item 2's cost estimate above is the planning number.
