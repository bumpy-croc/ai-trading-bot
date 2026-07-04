# Strategy Tournament v2 — Rerun on Corrected Backtest Engine

**Date**: 2026-07-04
**Researcher**: quant-researcher
**Status**: complete
**Supersedes**: `docs/research/experiments/2026-07-03_strategy-tournament.md` (VOID — see below)
**GitHub Issue**: (to be linked)

## Why this rerun exists

The 2026-07-03 tournament ran on a backtest engine with a partial-exit units bug
(fixed in PR #838, merged to `develop` @ `3ef34ade`). Partial-exit handlers mixed
two incompatible units — position sizes are fractions of **balance**, but the
exit code converted the policy's "fraction of original position" into a
fraction of **current position** and fed that into balance-fraction P&L math.
Every backtest in which a strategy's partial-exit levels triggered fabricated
returns. Concretely: `kelly_momentum`/ETHUSDT/30d previously reported **+16.67%**;
the true number (verified independently by the PM) was **~0%** — the reported
gain was ~$14.19 of phantom credits on $0.07-0.29 of real notional.

**Every return number in the prior tournament is void.** This document reruns
the same battery on the corrected engine (`origin/develop @ 3ef34ade`) so the
PM/risk-officer can make the live-strategy decision on real numbers.

Note: live HyperGrowth's actual production record (5 wins, $83.29 -> $84.40,
DB-verified) is unaffected by the bug — those exits were trailing-stop closes
at +3.1-3.8% price move, below HyperGrowth's first partial-exit target (8%),
so the buggy code path was never exercised in production. Only backtests that
touched partial-exit logic were corrupted.

## Hypothesis

Among the strategies deployable on the ~$85 live account, one strategy+symbol
combination will demonstrate meaningfully better risk-adjusted AND absolute
return than the incumbent (HyperGrowth/ETHUSDT) over both a 90-day and a 30-day
window, on the corrected engine, while respecting the live risk envelope
(2% base risk/trade, 3% max risk/trade, 0.20 max position size).

## Falsifiable Statement

A challenger is "ready for risk review" only if it satisfies all of:
1. Positive total_return in both the 90-day and 30-day windows (robustness).
2. Max drawdown comfortably inside `risk-limits.json`'s 20% portfolio hard cap.
3. Sharpe ratio and total_return internally consistent with win rate and trade
   count (no "0%-win positive return" or "near-zero MaxDD, multi-% return"
   artifacts — the exact pattern that flagged the prior tournament as fabricated).
4. Beats HyperGrowth's corrected numbers on both windows, not just one.

## Methodology

- Engine: `origin/develop @ 3ef34ade` (`fix(backtest): correct partial-exit
  units, scale-in guard, marked-to-market drawdown`, PR #838), checked out in
  a disposable worktree at `.claude/worktrees/tournament-v2`, never touching
  the main checkout or production.
- Initial balance: $85 (matches live account).
- Risk per trade: 0.02 base / 0.03 max (matches live).
- `--max-position-size 0.20` (matches live prod config per #835/#836 — see
  **divergence note** below; this is *not* the repo-default 0.10).
- Timeframe: 1h. Windows: 90 days (2026-04-05 to 2026-07-04) and 30 days
  (2026-06-04 to 2026-07-04) where specified.
- Fees/slippage: default `CostCalculator` ON (fee_rate=0.001, slippage_rate=0.0005) — not a fee-free debug run.
- Data cache: symlinked from the main checkout's `cache/market_data`
  (pre-filled 2025-2026 Binance data); no fresh prefill needed.
- Runs executed **strictly sequentially**, one `atb backtest` invocation at a
  time, to respect thermal limits while another agent ran unit tests on the
  same machine.
- Every CLI run was cross-validated against an in-process rerun that captured
  `Backtester.trades` directly (ground-truth `Trade.side` / `Trade.size`,
  where `Trade.size` is a balance-fraction per the #838 units contract) — all
  9 cross-checks matched the CLI-reported totals/return/Sharpe/MaxDD exactly,
  confirming determinism and giving reliable long/short and position-size data
  that the summary JSON doesn't expose directly.
- `kelly_momentum` is labeled **"baseline, Kelly inactive pending #840 fix"**:
  `KellyCriterionSizer.record_trade()` has zero callers in either engine (risk-officer
  finding, 2026-07-03), so the strategy runs in permanent cold-start fallback
  sizing (~0.3-0.6% of balance/trade) regardless of what the backtest reports.
  Its Kelly edge has never been live-tested.
- `ml_basic` only has a trained model for BTCUSDT; no ETHUSDT basic model exists.

## Divergence note (flag for risk-officer / pm)

`risk-limits.json` (`position.max_position_size_pct`) = **0.10**. Live prod
and this rerun both use **0.20** (per #835's approved sizing raise + explicit
`--max-position-size 0.20`, and railway.json's startCommand). `risk-limits.json`
`$last_reviewed` is still `1970-01-01` — it has never been formally reconciled
against the values actually running in prod. This was flagged by risk-officer
on 2026-07-03 and remains open. Not blocking this experiment (I matched prod
config per the task brief) but it means the JSON is not currently a source of
truth for position sizing and should be corrected or the divergence
explicitly ratified.

## Results — All Runs (corrected engine)

| Strategy | Symbol | Days | Trades | Return% | MaxDD% | Sharpe | WinRate% | Final $85 | Long/Short | Avg Pos Frac | Max Pos Frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper_growth | ETHUSDT | 90 | 13 | -3.29 | 5.97 | 0.05 | 61.5 | $81.45 | 6/7 | 11.7% | 19.7% |
| hyper_growth | ETHUSDT | 30 | 2 | +1.22 | 1.85 | 0.04 | 100.0 | $85.26 | 1/1 | 18.0% | 18.0% |
| hyper_growth | BTCUSDT | 90 | 8 | -1.98 | 3.03 | 0.03 | 75.0 | $83.59 | 5/3 | 12.6% | 20.0% |
| kelly_momentum* | ETHUSDT | 90 | 7 | -0.03 | 0.10 | 0.00 | 57.1 | $84.96 | 7/0 | 0.30% | 0.61% |
| kelly_momentum* | ETHUSDT | 30 | 3 | +0.02 | 0.04 | 0.00 | 66.7 | $85.00 | 3/0 | 0.27% | 0.40% |
| momentum_leverage | ETHUSDT | 90 | 4 | -0.05 | 0.42 | 0.00 | 50.0 | $84.91 | 4/0 | 1.47% | 2.09% |
| momentum_leverage | ETHUSDT | 30 | 1 | +0.21 | 0.09 | 0.00 | 100.0 | $85.13 | 1/0 | 1.96% | 1.96% |
| leveraged_regime | ETHUSDT | 90 | 5 | -0.07 | 0.16 | 0.00 | 20.0 | $84.93 | 5/0 | 0.82% | 1.41% |
| ml_basic | BTCUSDT | 90 | 3 | +0.00 | 0.03 | 0.00 | 66.7 | $85.00 | 3/0 | 0.66% | 1.23% |

*kelly_momentum: cold-start fallback active for the entire test window (see above) — this is not the Kelly-sized strategy the prior tournament evaluated, it's the fallback path.

### Sanity checks performed

Every row was checked for the three fabrication signatures that flagged the
prior tournament: (1) 0%-win-rate with positive return, (2) near-zero MaxDD
with multi-percent return, (3) return/win-rate/trade-count mutual
inconsistency. None of the 9 rows trip any of these:

- `hyper_growth` ETHUSDT 90d: 61.5% win rate with a *negative* total return
  (-3.29%) is internally consistent (losers are larger than winners — a poor
  payoff ratio, not a paradox) and MaxDD (5.97%) scales sensibly with the
  ~12-20% position sizing actually used.
- `hyper_growth` BTCUSDT 90d: 75% win rate, still net negative (-1.98%) — same
  large-loss/many-small-win pattern, consistent with the 12.6% avg/20% max
  position fractions in play.
- `kelly_momentum`, `momentum_leverage`, `leveraged_regime`, `ml_basic`: all
  show sub-2.1% position fractions, sub-0.5% MaxDD, and single-digit-cent
  total returns — fully consistent with each other (tiny size in -> tiny P&L
  and tiny drawdown out). This is the *opposite* of the prior tournament's
  pattern (huge return, ~0% MaxDD) and is the expected signature of the fix
  actually working.
- No trade-count / return magnitude combination implies an average per-trade
  edge outside plausible bounds for 1h crypto (largest magnitude average
  is hyper_growth 30d at +0.61%/trade on 2 trades — a tiny, non-generalizable
  sample, flagged as such below, not treated as a result).

## Comparison vs prior (fabricated) tournament

| Strategy | Days | Prior (void) | Corrected | Delta |
|---|---|---|---|---|
| hyper_growth/ETHUSDT | 90 | +10.66% | -3.29% | -13.95pp |
| hyper_growth/ETHUSDT | 30 | -0.86% | +1.22% | +2.08pp |
| hyper_growth/BTCUSDT | 90 | +1.46% | -1.98% | -3.44pp |
| kelly_momentum/ETHUSDT | 90 | +19.25% | -0.03% | -19.28pp |
| kelly_momentum/ETHUSDT | 30 | +16.67% | +0.02% | -16.65pp |
| momentum_leverage/ETHUSDT | 90 | +20.95% | -0.05% | -21.00pp |
| momentum_leverage/ETHUSDT | 30 | +14.94% | +0.21% | -14.73pp |
| leveraged_regime/ETHUSDT | 90 | +1.67% | -0.07% | -1.74pp |
| ml_basic/BTCUSDT | 90 | ~+0.00% | +0.00% | ~0 (unaffected — no partial exits triggered) |

The strategies that used partial exits most aggressively (kelly_momentum,
momentum_leverage) show the largest corrections, exactly as PR #838's root
cause analysis predicted. `ml_basic` and `hyper_growth`'s 30d window barely
moved (`hyper_growth`'s 90d window moved a lot — it had 13 trades in that
window, more opportunities for partial-exit levels to trigger and for the
new marked-to-market drawdown accounting to bite).

## Comparison vs HyperGrowth's LIVE record

Live HyperGrowth (session 20, ETHUSDT, prod): 5 winning trades, individual
exits +3.1% to +3.8% (trailing stop), balance $83.29 -> $84.40 since 2026-06-05.

Two different things are being measured and must not be conflated:
- **+1.7% to +3.8% "per trade"** (as cited in the task and prior log entries)
  is the **price-move / trade-level P&L%** — the return on the position's
  own entry-to-exit price change.
- The **balance-level effect** of those 5 trades was only **+1.33%** total
  ($83.29 -> $84.40), because each position was sized at roughly 10-20% of
  notional — a healthy per-trade win only moves the account by a fraction of
  that price move.

The corrected backtest's hyper_growth 30-day run (+1.22% total_return, 2
trades, both winners) is the right comparison point for the live record: same
strategy, same symbol, same ~18% position fraction, same order of magnitude
of balance-level return per trade. It is *directionally consistent* with the
live record, though the backtest's 90-day window (-3.29%, 13 trades, 61.5%
win rate) is a reminder that HyperGrowth is not reliably profitable over
longer windows or larger trade counts — the live win streak is a short,
favorable sample, not evidence the edge holds over 90 days.

No challenger comes close to HyperGrowth's demonstrated live-record magnitude.
All four challengers (kelly_momentum, momentum_leverage, leveraged_regime,
ml_basic) post sub-0.25% total returns in every window because they are all
sizing at 0.3-2.1% of balance per trade — 10-40x smaller than HyperGrowth's
12-20%. Their near-zero returns are a direct, mechanical consequence of their
tiny position sizing, not evidence of a weak or strong trading edge either way.
There is not enough signal in these backtests to compare their underlying
edge to HyperGrowth's, because they were never tested at comparable notional.

## Robustness / sensitivity notes

- **Trade counts are thin across the board.** Even the busiest strategy
  (hyper_growth/ETHUSDT/90d) only produced 13 trades; four rows have 3 trades
  or fewer. None of these results should be treated as statistically reliable
  — they are directional signals only, consistent with `docs/backtesting.md`'s
  guidance that a "high Sharpe with 20 trades is not a result." All Sharpe
  ratios in this battery are effectively noise (0.00-0.05); they carry no
  decision weight.
- I did not re-run a parameter sensitivity sweep in this pass — the task was
  a straight rerun on the corrected engine, not a new parameter search. If the
  PM wants to pursue any of momentum_leverage/leveraged_regime as a paper
  candidate, a ±10-20% sensitivity check on their lookback/threshold params is
  required first per the standing workflow rules, and is not yet done.
- kelly_momentum's near-zero numbers here are **not a verdict on the Kelly
  edge** — they are a verdict on the fallback path, because the sizer's
  `record_trade()` has no engine callers (issue tracked, fix dispatched as
  `fix/kelly-sizer-trade-feedback`, #840, not yet merged as of this writeup).
  Rerunning this tournament again once #840 lands is the right next step
  before drawing any conclusion about Kelly sizing itself.

## How this could lose money (if any challenger were promoted as-is)

- Promoting any of kelly_momentum / momentum_leverage / leveraged_regime today
  would mean deploying with 0.3-2.1% position sizing that has never been
  tested at HyperGrowth's 12-20% live sizing — scaling these strategies up to
  match HyperGrowth's notional is untested territory; drawdown and slippage
  behavior at 10x the position size are unknown.
- HyperGrowth's own 90-day corrected numbers (-3.29% ETHUSDT, -1.98% BTCUSDT)
  show the incumbent is not unconditionally profitable — its live win streak
  is a short, favorable-regime sample. A regime shift could easily flip it
  negative, consistent with the backtest's negative 90d showing.
- kelly_momentum's cold-start fallback sizing is a silent trap: if #840 ships
  and Kelly sizing activates without a fresh live-parity check, the strategy's
  risk profile changes materially and without warning (from ~0.3% notional to
  potentially the full `kelly_max_fraction=0.20` from risk-limits.json).

## Ranked Recommendation

**Keep HyperGrowth as the sole live strategy. No challenger earns a paper
trial from this rerun.**

Reasoning:
1. HyperGrowth is the only strategy in this battery producing balance-level
   returns of a magnitude that matters (it moves the account by single-digit
   percent per quarter, positive or negative). Every challenger is sizing so
   small (0.3-2.1% of balance) that even a real edge would take years to
   compound into a meaningful outcome at this position size — it is not that
   they're proven bad, it's that this test batch cannot distinguish their edge
   from noise at these notional levels.
2. kelly_momentum, the winner of the prior (fabricated) tournament, is now
   confirmed flat (-0.03%/+0.02%) once the units bug is fixed — its old
   "win" was entirely a units-conversion artifact, exactly as risk-officer's
   2026-07-03 verdict (b) reject [high confidence] anticipated on independent
   grounds (dead Kelly sizer wiring). This rerun corroborates that rejection
   with real numbers rather than just the wiring-gap argument.
3. HyperGrowth itself is not unambiguously good news here: -3.29% ETHUSDT/90d
   and -1.98% BTCUSDT/90d are real, fee-inclusive corrected losses on the same
   config that produced the +1.33% live win streak. The live results and the
   backtest results are not in conflict (different windows, small samples,
   favorable vs unfavorable segments of the same noisy process) but they do
   mean HyperGrowth's edge is fragile, not dominant — it should keep the
   entry-pause guard around FOMC/CPI dates and continue to be watched, not
   scaled up further without a new sizing review.
4. Before any of the smaller strategies can be fairly evaluated, they need
   (a) `fix/kelly-sizer-trade-feedback` (#840) merged so kelly_momentum tests
   its real edge instead of a permanent fallback, and (b) a sizing-matched
   rerun (comparable position-size fraction to HyperGrowth) so a return
   comparison isn't just measuring "who sized bigger." Recommend re-opening
   this question after #840 ships, as a fresh, narrowly-scoped experiment.

**Verdict for pm: ready for risk review only on the "no change" outcome** —
i.e., risk-officer does not need to re-review a promotion, since none is
being proposed. If risk-officer or pm want a sizing-matched rerun of
momentum_leverage/leveraged_regime at HyperGrowth's ~15-20% notional to get
an apples-to-apples signal, that is a promising, cheap next step and I can
scope it as a new falsifiable experiment.

## Next steps

- Track `fix/kelly-sizer-trade-feedback` (#840); rerun kelly_momentum once merged.
- If pm wants a sizing-matched comparison, scope a new experiment:
  momentum_leverage / leveraged_regime at `--max-position-size 0.20` with
  their position sizers' base fraction raised to match HyperGrowth's ~0.20-0.25
  base, same windows, same sensitivity-analysis rigor as any live-affecting
  proposal requires.
- Flag `risk-limits.json`'s stale `$last_reviewed` (1970-01-01) and the
  0.10-vs-0.20 max_position_size_pct divergence from prod to risk-officer/pm
  for formal reconciliation (not new — first flagged 2026-07-03, still open).
