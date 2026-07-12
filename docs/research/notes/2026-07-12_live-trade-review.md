# Live Trade Review — ETHUSDT / HyperGrowth, 2026-06-02 → 2026-07-12

**Author:** quant-researcher · **Type:** periodic trade autopsy (`trade-review` skill) · **Feeds:** exit-geometry experiment (Lane B), future preregistrations

**Sample-size caveat up front:** this record is **12 closed trades** plus 1 open position. Per
the standing rule, no expectancy claim is made from this n. Everything below is direction-of-
evidence and falsifiable hypotheses, not a verdict.

## Data & method

- Prod Postgres via public proxy, `SET default_transaction_read_only = on;`, SELECT only
  (`RAILWAY_PRODUCTION_DATABASE_URL`). Tables: `trades`, `positions`, `account_balances`,
  `strategy_executions`.
- MFE/MAE reconstructed independently from the local ETHUSDT 1h parquet cache (34 files, Jan–Jul
  2026), **not** from `trades.mfe`/`trades.mae` — see Finding 1 below for why.
- Window: candles whose `[open, open+1h)` interval overlaps `[entry_time, exit_time]`; for trades
  under 1 hour (5 of the 12), this collapses to a single entry-hour candle. **Caveat:** this is an
  upper-bound approximation — it can include a sliver of pre-entry/post-exit price action in the
  boundary hour. For the five sub-1-minute "emergency close" trades the true intra-trade
  excursion is effectively unmeasurable at 1h granularity; those rows are flagged, not trusted.
- Live-vs-backtest spot check used the post-#838 corrected engine (`atb backtest hyper_growth`,
  current `develop`), same symbol/timeframe/window, default fees+slippage on.

## Per-trade MFE/MAE/capture table (closed trades)

| id | side | entry→exit (UTC) | duration | exit_reason (raw) | realized % | MFE % | MAE % | capture ratio | note |
|----|------|-------------------|----------|--------------------|-----------:|------:|------:|---------------:|------|
| 1 | LONG | 06-02 09:22:41 → 09:22:56 | 15s | Stop-loss placement failed - emergency close | +0.030 | 0.571 | -0.093 | 0.052 | sub-minute, candle-based MFE/MAE not meaningful (ops event) |
| 2 | LONG | 06-02 14:09:59 → 14:10:13 | 14s | Stop-loss placement failed - emergency close | -0.010 | 0.503 | -2.012 | -0.019 | same |
| 3 | LONG | 06-02 18:30:54 → 18:31:08 | 14s | Stop-loss placement failed - emergency close | -0.055 | 0.763 | -0.524 | -0.072 | same |
| 4 | LONG | 06-02 18:33:18 → 18:33:45 | 27s | Engine shutdown | -0.024 | 0.727 | -0.559 | -0.032 | ops event, not strategy |
| 5 | LONG | 06-04 20:07:00 → 06-05 15:01:18 | 18h54m | stop_loss | **-10.003** | 0.993 | -10.947 | -10.068 | see Finding 2 — pnl not independently ledger-verified |
| 6 | SHORT | 06-05 21:04:03 → 21:04:19 | 16s | Stop-loss placement failed - emergency close | -0.158 | 0.466 | -0.581 | -0.340 | sub-minute, ops event |
| 7 | LONG | 06-06 17:40:06 → 06-07 12:53:52 | 19h14m | Stop loss | +3.965 | 6.043 | -0.495 | **0.656** | win-streak #1 |
| 8 | LONG | 06-07 12:54:06 → 06-07 22:14:39 | 9h21m | Stop loss | +3.775 | 6.525 | -0.771 | **0.579** | win-streak #2 |
| 9 | LONG | 06-07 23:04:58 → 06-14 21:50:14 | 6d23h | Stop loss | +3.480 | 3.530 | -4.170 | **0.986** | win-streak #3, near-zero give-back |
| 10 | SHORT | 06-14 21:52:28 → 06-18 19:41:47 | 3d22h | Stop loss | +1.659 | 3.290 | -7.109 | **0.504** | win-streak #4, biggest give-back |
| 11 | SHORT | 06-18 19:41:57 → 06-23 08:24:50 | 4d13h | Stop loss | +3.201 | 3.682 | -4.812 | **0.869** | win-streak #5 |
| 12 | LONG | 06-23 08:25:02 → 07-02 13:34:13 | 9d05h | Stop loss | +3.136 | 4.263 | -8.012 | **0.736** | win-streak #6, closes the streak |

**Winners (7–12) average capture ratio ≈ 0.72** — live realizes roughly three-quarters of the
peak favorable move before the exit mechanism (labeled generically "Stop loss" in the DB but
economically behaving like a trailing stop) triggers. All six sit below the strategy's 8%
partial-exit target, consistent with the standing reference finding that partials have never
fired live.

**Trade 5 (the one big loser)** never built real favorable excursion (MFE 0.99% over 19 hours)
and bled down almost the entire width of the wide stop (MAE -10.95% vs a realized -10.00%,
~91% MAE-capture) — this looks like "wrong from the start," not "right then reversed."

## Finding 1 — `trades.mfe`/`mae` fields disagree with their own `mfe_price`/`mae_price` columns

For every multi-hour trade the DB's own `mfe_price` (or `mae_price`) implies a materially
different percentage than the `mfe` (`mae`) column stored alongside it, computed on the *same*
`entry_price`. Example, trade 9: `mfe_price=1727.84`, `entry_price=1673.21` →
(1727.84-1673.21)/1673.21 = **3.264%**, but the stored `mfe` column is **0.320%** — off by ~10.2x.
Trade 8: implied 3.321% vs stored 0.231% (~14.4x). Trade 11: implied 3.174% vs stored 0.306%
(~10.4x). Trade 12: implied 3.130% vs stored 0.137% (~22.8x). The ratio is **not constant**
across trades, so this isn't a simple unit/leverage scalar — the two columns appear to be
computed on different bases entirely (plausibly position/margin-relative vs price-relative), and
they're internally inconsistent with each other inside the same row. This is why the skill
directs reconstruction from cached OHLCV rather than trusting `trades.mfe`/`mae` directly — do
not use the raw `mfe`/`mae` columns for anything until this is understood; they don't currently
agree with their own `_price` companions. Flagged in the tracking issue for whoever picks up
live-tracking-field hygiene; not fixed here (out of scope for this review, no code touched).

## Finding 2 — Trade 5's realized loss is not independently visible in the ledger

`account_balances` has no `realized_pnl_ETHUSDT_stop_loss`-style row anywhere near trade 5's
exit (2026-06-05 15:01:18). The only ledger event in that window is a single
`margin_equity_sync_correction` row 40 minutes later (2026-06-05 15:41:14, -$17.12 total delta)
— this is the **same, already-documented** phantom-$100→true-$84 book correction from the prior
prod-forensics investigation (log.md 2026-06-08 finding), not a new incident. But it means trade
5's -10.00% `pnl_percent` in the `trades` table is not corroborated by an isolated ledger entry;
treat that one number as directionally right (it matches the candle-based MAE of -10.95% well)
but not ledger-verified in isolation.

## Exit-reason P&L decomposition (ledger-based, `account_balances`, since 2026-06-02)

| bucket | n (ledger rows) | net realized $ (ledger) | character |
|---|---:|---:|---|
| `Stop-loss placement failed - emergency close` | 4 | -0.048 | ops failure, not strategy signal |
| `Engine shutdown` | 1 | -0.010 | ops event |
| `Stop loss` (strategy exit, win-streak) | 6 | **+1.740** | strategy signal — this is the real P&L driver |
| (trade 5, not separately visible) | — | — | see Finding 2 |
| `margin_equity_sync_correction` | 2 | -17.116 | historical book correction, pre-dates this window's trading (already documented) |
| `entry_fee_ETHUSDT` | 18 | -0.164 | fees; 18 fee rows vs 8 distinct `position_id`s in `trades` — likely retry/partial-fill entries, not investigated further (small, out of scope) |

Net picture: the five ops-failure/shutdown closes are collectively noise (~-$0.06, negligible,
correctly separable from strategy behavior). The six win-streak "Stop loss" exits are the entire
positive story (+$1.74 net). Fee data is only trustworthy from trade 10 onward
(`trades.commission` populated post-#731); trades 1–9 show `commission=0.00000000` as a
placeholder, not a verified zero.

## Position #22 (open 9.8 days, still live as of this review)

- SHORT, entered 2026-07-02 13:34:24 @ 1696.83, confidence 0.250 / strength 0.208 (matches the
  prior #913 forensics reconstruction of this exact position — no new correction needed).
- Stop-loss 1864.654 (**+9.90%** from entry), take-profit 1186.598 (**-30.06%** from entry) — the
  wide 10%/30% asymmetric design, live, in one exhibit.
- Candle-reconstructed since entry: **MFE +2.127%**, **MAE -8.049%**. Current unrealized
  -1.016% (price 1804.91 vs entry 1696.83).
- The live-tracked `positions.mfe` field reads **0.00000000** (i.e., "never favorable") while the
  candle reconstruction shows a real +2.13% favorable dip at some point in the 9.8 days — the
  same kind of live-tracker-vs-price-reconstruction gap as Finding 1, just on the open-position
  side of the schema.
- In 9.8 days this position has used up roughly **81% of its stop-loss budget** (-8.05% of a
  9.90% stop) while its best moment was **only 7% of the way** to its take-profit target
  (+2.13% of 30.06%). It has not been close to profitable at any point since entry.

## Live-vs-backtest divergence (pass 4)

Ran `atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h --start 2026-06-02 --end 2026-07-12
--initial-balance 84` on current `develop` (post-#838 corrected engine), fees+slippage on:

| | live (this window) | backtest (matched config) |
|---|---:|---:|
| closed trades | 12 (5 ops-failure + 7 strategy) | 6 |
| net P&L (undiscounted sum of trade %) | ≈ **+9.0%** | **-0.78%** |
| win rate | 7/7 non-ops trades won except trade 5 (6/7 ≈ 86%) | 50% |
| avg winner | ~3.0% (win-streak) | 0.156% |
| avg loser | -10.0% (single trade) | -1.20% |
| avg trade duration | mixed, several multi-day | 99.8h (~4.2 days) |

This is a large divergence — trade count off by 2x, sign of total return flipped, and winner/
loser magnitudes an order of magnitude apart. It is **far outside** the charter's 15% parity
band. This is consistent with, not a new instance separate from, two things already on record:
(1) the 2026-07-06 forming-bar fliprate finding — live decides against a mutating tail candle and
can enter/exit on signals a closed-bar backtest never evaluates, and (2) the 2026-07-08
confidence-collapse finding — HyperGrowth's entries cluster right at a low-confidence boolean
gate (median confidence ~0.03–0.04, position #22 itself entered at 0.25), which is exactly where
small live/backtest microstructure differences are most likely to flip the gate. Not re-diagnosed
here; flagged as a concrete, quantified instance for whoever runs the next parity study.

## Hypotheses for `experiment-preregister` (not conclusions, not tweaks)

1. **H1 — trailing-stop tightening.** On this ETHUSDT 1h HyperGrowth window, winners captured
   ~72% of peak favorable excursion on average (range 50–99%). A trailing-stop calibrated to lock
   in ≥85% of realized MFE would improve realized return without a matching increase in loss
   frequency. Candidate metric: total return and max drawdown, backtest + OOS holdout, trailing-
   stop distance swept as the primary parameter.
2. **H2 — early-cut on near-zero MFE.** A trade that has built <1.5% MFE within its first ~12–18
   hours (as trade 5 did — 0.99% MFE over 19 hours before a 10% stop-out) is unlikely to recover;
   a time-boxed or MAE-triggered early-exit rule could cut such losses materially without
   truncating the win-streak trades (which also started with modest early MFE before compounding
   over days). Candidate metric: largest single loss and total return, same window + OOS.
3. **H3 — wide-stop cost on low-confidence entries.** Position #22 (confidence 0.25, near the
   noise floor) has ridden 81% of its stop-loss budget over 9.8 days while never getting past 7%
   of its take-profit distance — one data point consistent with (not proof of) the open #938
   finding that flat sizing/wide stops leave edge on the table at low confidence. Candidate
   metric: a preregistered sweep of stop width and/or confidence-conditioned sizing on a frozen
   exam window, scored against the current wide-stop baseline.
4. **H4 — parity gap is forming-bar-driven, testable.** The 6-vs-12-trade, sign-flipped-return
   divergence found here should shrink materially if the backtest harness is given forming-bar
   (intra-candle, mutating-tail) evaluation matching live's decision cadence. Candidate metric:
   trade count and total return delta between live and backtest, before/after a forming-bar-aware
   backtest variant, same window.

## What this review does NOT do

No parameter was changed. No strategy recommendation is made. These four hypotheses are handed
to `experiment-preregister` for the exit-geometry work already running in Lane B; this review's
job was evidence, not a tweak.
