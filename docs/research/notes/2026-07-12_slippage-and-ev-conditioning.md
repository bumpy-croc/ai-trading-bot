# Slippage Measurement + EV-Conditioning — GH #984

**Author:** quant-researcher · **Type:** measurement study (two parts) · **Feeds:** GH #984 (exam
cost recalibration), the audit's Hunt 4/5 findings, any future conditional-sizing preregistration.

**Executive summary (both verdicts up front):**

1. **Slippage.** Measured from every recoverable prod ETHUSDT fill (24 fills: 12 closed trades'
   entry+exit plus position #22's still-open entry; position #13's exit is unrecoverable — no
   `orders` row exists for it, see Data section). Median absolute per-fill slippage vs. a tight,
   look-ahead-free 1-minute reference is **5.1–5.8 bps per side**, mean **8.7–14.5 bps** (range
   reflects whether 3 extreme-volatility fills are trimmed). This is **close to the current
   exam default (5 bps/side), not the audit's proposed ~5x cut to ~1 bp/side.** Recommendation:
   do **not** cut `DEFAULT_SLIPPAGE_RATE` to the audit's suggested ~0.0001; the evidence does not
   support it. See Part 1 for the full honest-n treatment and a specific counter-proposal.
2. **EV-conditioning.** Across 136 control-arm HyperGrowth trades (ETHUSDT/1h, F1–F3, corrected
   for GH #997/#998's cross-symbol bug), **no conditioner shows a significant EV or win-rate
   gradient** at a Bonferroni-corrected threshold (predicted-return magnitude/confidence/strength
   collinear group, realized volatility at entry, regime, hour-of-day session — 4 independent
   conditioners, 8 tests, all p > 0.09 raw, let alone corrected). This extends the 2026-07-05
   confidence-calibration study's bar-level null to the trade level. **Verdict: flat sizing is not
   leaving conditionable edge on the table for this signal; the audit's Hunt-5 sizing-experiment
   premise is moot.** No conditional-sizing experiment is recommended.

---

## Data & method (shared)

- Prod Postgres via public proxy (`railway variables -e production -s Postgres --json` →
  `DATABASE_PUBLIC_URL`, `SET default_transaction_read_only = on;`, SELECT only). Tables: `trades`,
  `positions`, `orders`, `strategy_executions`.
- Fresh worktree `.claude/worktrees/slippage-ev-conditioning`, branch
  `docs/slippage-ev-conditioning-984`, off `origin/develop @ 3721a835`, `.agent-active` present.
- Cached ETHUSDT 1h OHLCV reused from sibling worktrees (content-addressed parquet cache, verified
  via `atb data cache-manager info` + a direct load) for Part 2's F1–F3 folds
  (2022-10-01→2025-07-05 continuous, 24,096–27,120 bars depending on script); a fresh **live**
  1-minute pull (public Binance klines endpoint, no credentials) for Part 1's May–Jul 2026 window,
  since the cache layer does not support sub-hourly timeframes (`CachedDataProvider` silently
  returns empty for `"1m"` — confirmed, not chased further, direct `BinanceProvider` calls work
  fine and are cheap for the narrow windows needed here).
- All code lives under `experiments/` (this repo's convention for research driver scripts, never
  imported by `src/`): `slippage_measurement.py`, `ev_conditioning_control_trades.py`,
  `ev_conditioning_features.py`, `ev_conditioning_analysis.py`. No `src/` file is modified by this
  study.

---

## Part 1 — Measuring real slippage from prod fills

### 1.1 Fill inventory (honest-n)

Query: `trades` (12 ETHUSDT rows, ids 1–12) joined to `positions` by matching entry
price/time, joined to `orders` filtered to `status IN ('FILLED','CONFIRMED') AND
actual_fill_price IS NOT NULL AND order_type IN ('ENTRY','FULL_EXIT')`, plus position #22
(currently OPEN, entry only).

- **12 closed trades → 24 expected fills (entry+exit).** Got **23**: position #13's exit
  (trade id 5, the −10.00% loser, 2026-06-05) has **no matching order row of any status** — its
  `trades.exit_price=1588.88` is the only surviving record of that exit. This coincides with the
  already-documented 2026-06-05 `margin_equity_sync_correction` incident (an SL-fail→emergency-close
  cascade), consistent with the Lane D trade-review's independent flag that this trade "has no
  isolated ledger row near its exit." **Excluded from slippage measurement** — I will not fabricate
  a reference for a fill that has no recoverable order record.
- **Position #22's entry** (still open, SHORT, 2026-07-02): **1 fill**, included per the task's
  scope.
- **Total: 24 fills, n=24.** 13 ENTRY, 11 FULL_EXIT. Notional range at these fills: $9.85–$19.90
  (0.0037–0.0079 ETH at $1554–$1997), matching the task's "$15-90 notionals" framing loosely — our
  actual live notionals in this sample sit at the low end of that range (**$10-20**, not $15-90;
  reported honestly rather than silently rounded up).

### 1.2 Two reference methods, and why the first one is a red herring

**Reference (a): the containing 1h bar's OHLC (as literally requested).** First pass used the 1h
bar's `open` as the price reference. Result: mean **−80.3 bps**, median **−54.99 bps**, std 130.75,
range −365 to +84 bps. **This number is not slippage — it's mostly ordinary intrabar drift.**
Fills execute within **2–6 seconds** of order creation (see 1.4), but the *reference* bar can be up
to 59 minutes stale relative to that instant. June 2026 (when 22 of 24 fills happened) was an
unusually volatile week for ETHUSDT (~$1968→$1554, ~21% decline in days), so a 59-minute-stale
1h-bar reference routinely differs from the fill by 50-300+ bps for reasons that have nothing to do
with execution quality. Reported for completeness (`slippage_bps_bar_open` /
`slippage_bps_decision_close_1h` in `experiments/slippage_fills.jsonl`) but **not used for the
constant recommendation** below.

**Reference (b): closest available reference at decision time — used a 1-minute close.** Fetched
live 1-minute OHLCV (Binance public klines, no credentials) in narrow ±20-minute windows around
each fill's `created_at`, and used the close of the **last fully-elapsed 1-minute bar strictly
before** `created_at` (not the bar *containing* `created_at` — see the note below on a bug I found
and fixed in my own script before trusting these numbers).

**A methodology bug I found and fixed while building this** (documented so it isn't silently
repeated): my first version of the 1-minute reference used the price bar *containing*
`created_at` and read its `close`. For a still-forming minute bar, `close` reflects price up to 60
seconds **after** the reference instant — i.e., look-ahead. This produced a spurious +137.98 bps
outlier for order 7418 (position 17's exit, filled 2s after creation) that traced directly to a
genuine ~4% 3-minute price spike (2026-06-07 22:11→22:14, volume 223→10,077) — the reference was
reading 25 seconds into the future relative to the actual decision instant. Fixed by stepping back
to the **prior closed** 1-minute bar (mirrors exactly the same "prior closed bar, not containing
bar" logic already used for reference (a) at 1h granularity). Verified the fix collapses that
specific outlier from +137.98 bps to −79.68 bps (i.e., it flips from a nonsensical extreme adverse
reading to a large-but-explicable favorable one — the position benefited from continuing to hold
into part of that same spike before its stop/target fired 2 seconds after order creation).

### 1.3 Results (reference b, the primary estimate)

| | n | mean (bps) | median (bps) | std |
|---|---|---|---|---|
| Full sample | 24 | −10.65 | −4.09 | 21.23 |
| Trimmed (drop 3 largest \|x\|) | 21 | −4.25 | −0.73 | 12.09 |

Sign convention: positive = adverse (worse than reference), negative = favorable (better than
reference). **Median and mean are both slightly favorable**, not adverse — i.e., on this sample,
real fills were not systematically worse than the last known pre-order price.

| | median \|slippage\| (bps) | mean \|slippage\| (bps) | p90 \|slippage\| (bps) |
|---|---|---|---|---|
| Full sample (n=24) | 5.76 | 14.51 | 42.83 |
| Trimmed (n=21, drop 3 largest) | 5.12 | 8.67 | — |

Per order type: ENTRY (n=13) mean −7.84 bps / median −3.36 bps; FULL_EXIT (n=11) mean −13.96 bps /
median −4.98 bps (exits show more dispersion, consistent with more of them being stop/target
triggers firing during active price moves rather than steady-state model-driven entries).

**Is the signed mean distinguishable from zero?** One-sample t-test: t=−2.405, p=0.0246. Wilcoxon
signed-rank: W=82, p=0.0526. **Borderline at n=24 — I am not claiming a confident "real fills are
favorable on average" finding**, just reporting that the data does not support "real fills are
systematically adverse" either, which is what would be needed to justify the *current* 5 bps
charge being too *low*, let alone motivate cutting it.

**Per-fill detail** (`experiments/slippage_fills.jsonl`), sorted by signed slippage:

| order_id | position | type | action | fill | ref (1m) | slippage (bps) |
|---|---|---|---|---|---|---|
| 7372 | 15 | FULL_EXIT | BUY | 1612.50 | 1610.34 | +13.41 |
| 7419 | 18 | ENTRY | BUY | 1673.21 | 1671.81 | +8.37 |
| 7416 | 16 | FULL_EXIT | SELL | 1616.38 | 1617.32 | +5.81 |
| 7268 | 10 | FULL_EXIT | SELL | 1912.36 | 1913.34 | +5.12 |
| 7417 | 17 | ENTRY | BUY | 1616.46 | 1615.95 | +3.16 |
| 7270 | 11 | FULL_EXIT | SELL | 1913.63 | 1914.22 | +3.08 |
| 7422 | 19 | FULL_EXIT | BUY | 1698.13 | 1697.61 | +3.06 |
| 7371 | 15 | ENTRY | SELL | 1609.95 | 1610.34 | +2.42 |
| 7256 | 5 | ENTRY | BUY | 1968.53 | 1968.14 | +1.98 |
| 7269 | 11 | ENTRY | BUY | 1914.08 | 1914.22 | −0.73 |
| 7267 | 10 | ENTRY | BUY | 1913.41 | 1913.55 | −0.73 |
| 7423 | 20 | ENTRY | SELL | 1698.18 | 1697.61 | −3.36 |
| 7374 | 16 | ENTRY | BUY | 1554.74 | 1555.49 | −4.82 |
| 7257 | 5 | FULL_EXIT | SELL | 1969.12 | 1968.14 | −4.98 |
| 7263 | 8 | FULL_EXIT | SELL | 1965.15 | 1964.03 | −5.70 |
| 7426 | 21 | FULL_EXIT | SELL | 1695.24 | 1693.86 | −8.15 |
| 7262 | 8 | ENTRY | BUY | 1965.34 | 1967.57 | −11.33 |
| 7421 | 19 | ENTRY | SELL | 1726.78 | 1724.52 | −13.11 |
| 7427 | 22 | ENTRY | SELL | 1696.83 | 1693.86 | −17.53 |
| 7340 | 13 | ENTRY | BUY | 1765.48 | 1769.48 | −22.61 |
| 7420 | 18 | FULL_EXIT | SELL | 1731.44 | 1724.08 | −42.69 |
| 7424 | 20 | FULL_EXIT | BUY | 1643.82 | 1650.90 | −42.89 |
| 7425 | 21 | ENTRY | BUY | 1643.70 | 1650.90 | −43.61 |
| 7418 | 17 | FULL_EXIT | SELL | 1677.48 | 1664.22 | −79.68 |

The 3 largest-magnitude rows (7418, 7424/7425, and 7420) all sit on days with visible short-term
volatility clusters in the 1-minute data (checked directly, not assumed) — real market moves
between the reference minute and the fill, not measurement error, not a repeat of the fixed
look-ahead bug (re-verified after the fix). 7424 and 7425 share the same reference price (1650.90)
because they are a same-second short-cover/long-flip on the same session — both benefited from the
same intervening price drop.

### 1.4 Order latency (why reference (b) is legitimate)

`filled_at − created_at` across all 24 fills: min ~2s, median ~5s, max ~6s (`describe()` in the
script output). This confirms the decision-to-execution latency is genuinely short — the 1-minute
reference is not "too coarse" for this population; if anything a sub-minute reference would be
better still, but 1-minute is the finest granularity Binance's public klines endpoint offers and is
already 60x tighter than the 1h alternative.

### 1.5 Fee cross-check (no double-counting)

`orders.actual_commission` is denominated in the **received** asset (base/ETH on buys, quote/USDT
on sells — confirmed directly, not just per the prod-forensics playbook's note). Spot-checked two
fills: order 7427 (SELL, position #22 entry) — notional $13.40, commission 0.013405 (USDT) →
0.1000% of notional exactly. Order 7256 (BUY, position 5 entry) — notional $11.02, commission
0.000006 ETH ≈ $0.0118 → 0.1002% of notional. **Both match Binance's standard 0.10% taker fee
exactly**, confirming (a) the exam's `DEFAULT_FEE_RATE=0.001` is correctly calibrated (agrees with
the audit's Hunt #4 finding) and (b) commission is a clean, separate, price-independent charge —
`actual_fill_price` is gross of commission, so the slippage numbers above and the fee numbers in
`experiments/slippage_fills.jsonl` are non-overlapping cost components by construction. No double
count.

### 1.6 Recommended exam-model constant

**Current:** `DEFAULT_SLIPPAGE_RATE = 0.0005` (5 bps/side, `src/config/constants.py`).
**Audit's Hunt #4 estimate:** ~0.0001 (1 bp/side), from a theoretical top-of-book half-spread
argument, no live fills examined.
**This measurement:** median absolute per-fill deviation **5.1–5.8 bps/side** (full vs. trimmed),
mean **8.7–14.5 bps/side**, signed mean borderline-favorable (not adverse) at n=24.

**Recommendation: do not adopt the audit's ~5x cut.** The empirical median sits almost exactly at
the current 5 bps assumption, and the distribution has a real fat right tail (three fills in the
20–80 bps range, all on genuinely volatile days) that argues for keeping some conservatism, not
less. If anything, a defensible reading of this data supports **leaving `DEFAULT_SLIPPAGE_RATE`
unchanged at 0.0005**, or at most a small downward nudge to **0.0004** reflecting the slightly
favorable signed mean — but that nudge is not statistically confident (Wilcoxon p=0.053) and I
would not spend a "candidate" on tuning it without more fills. **This is a direct, evidence-based
correction to the audit's Hunt #4 point estimate**, not a confirmation of it — the audit's number
was a plausible-sounding theoretical calculation that measured fills do not support.

**Consequence for GH #984's second clause** ("re-exam marginal verdicts... under corrected
costs"): since the evidence does not support materially lowering the slippage constant, **the
tp_06-class re-exam that #984 anticipated is not warranted**. The EXIT-GEOMETRY round 2 rerun
(`docs/research/experiments/2026-07-12_exit-geometry-round2.md`, already completed) ran under the
unchanged default 0.0005 assumption throughout (confirmed by grep — no slippage/CostCalculator
override anywhere in that study) and does not need to be redone under a different constant.

**Caveats (honest-n, stated plainly):** n=24, ~$10-20 notionals, drawn almost entirely from one
unusually volatile week. This is not enough to precisely pin a "true" constant — it is enough to
say the audit's specific proposed value is not supported and the current value is not obviously
wrong. A future re-measurement with more live fills (ideally spanning calmer regimes too) would
sharpen this; not blocking on it given the direction of the finding is already decisive against the
audit's proposed change.

### 1.7 Aside: an independently reconfirmed harness bug (GH #997/#998)

While building the control-arm rerun for Part 2 (below), I hit — independently, before reading
either issue — the exact cross-symbol scoring bug already documented in GH #997 (root cause:
`ExperimentRunner._load_strategy` never threads `config.symbol` into the ML strategy factory, so a
bare `create_hyper_growth_strategy()` call defaults to `MLBasicSignalGenerator.DEFAULT_SYMBOL =
"BTCUSDT"`) and GH #998 (round 1's exit-geometry-honest study, `docs/research/experiments/
2026-07-12_exit-geometry-honest.md`, scored ETHUSDT candles with the BTCUSDT model for all 6 arms
across all 3 folds). My own first `ev_conditioning_control_trades.py` run, invoked the same way
round 1's driver was (bare `create_hyper_growth_strategy()`, plus the separate `sys.path`
shadowing bug from the same addendum), reproduced a third, still-different result set — consistent
with stacking *both* known bugs. After adding `factory_kwargs={"symbol": "ETHUSDT"}` and the
`sys.path.insert` fix (same pattern as `exit_geometry_round2_sweep.py`), my F1 control rerun
reproduced round 2's independently-verified corrected baseline **exactly** (29 trades, −1.69%
return, PF 0.7971 — bit-for-bit match). This is reported here as an independent corroboration, not
a new finding — no new GH issue filed, both are already open and owned by quant-researcher.

---

## Part 2 — Does per-trade EV vary with anything observable at entry?

### 2.1 What data actually exists (checked before writing new code)

Dispatched a reconnaissance pass (general-purpose subagent, read-only) across every candidate
existing artifact before building anything new:

1. `experiments/exit_geometry_results.jsonl` (round 1): fold-level aggregates + a flat
   `trade_pnl_pcts` array only — **no per-trade `entry_time`, no confidence, no regime.**
2. `exit_geometry_round2_sweep.py`'s `trades_raw`: real per-trade objects, but only
   `{entry_time, pnl_percent, exit_reason, early_cut_window_mfe_pct}` — no predicted-return/
   confidence/regime; round-2 sweep was not running at the time of this check
   (`ps aux | grep exit_geometry` empty).
3. Target-redesign tournament / meta-labels: the worktree that produced `results.json`/
   `methods.md` no longer exists and those files were never git-tracked — unrecoverable.
4. Confidence-calibration study's diagnostic scripts: never git-committed, worktree gone —
   unrecoverable (only the doc's prose survives).
5. **The general mechanism**: `BaseTrade` (`src/engines/shared/models.py`) natively carries
   `entry_time`/`exit_time`/`side`/`pnl_percent` — no engine change needed for those. But it does
   **not** carry `confidence_score` or `regime` (those exist only on the DB-logging path,
   `engine.py:1580`, gated behind `log_to_database=True`, not on the in-memory `Trade` object the
   backtester returns). No `regime` field exists on `Trade` anywhere.

**Bottom line: no reusable per-trade entry-observable dataset exists on disk.** Everything below was
assembled fresh, independently of and without perturbing the round-2 sweep's compute budget.

### 2.2 Method

1. **`ev_conditioning_control_trades.py`** — a **light, control-arm-only** rerun of exactly
   `docs/research/experiments/2026-07-12_exit-geometry-honest.md`'s control config (`hyper_growth`,
   ETHUSDT/1h, F1/F2/F3, `factory_kwargs={"symbol": "ETHUSDT"}`, `sys.path` fix applied — see 1.7).
   3 backtest runs, not 21 — deliberately far lighter than the round-2 sweep it runs alongside.
   Extracts `BaseTrade`'s native fields (`entry_time`, `exit_time`, `side`, `pnl_percent`, `mfe`,
   `mae`) per trade. **136 trades** (F1=29, F2=40, F3=67 — these are the *corrected*-symbol trade
   counts, smaller than round 1's now-known-wrong 31/46/70).
2. **`ev_conditioning_features.py`** — for each trade's `entry_time`, computed (no `src/` change,
   no backtest rerun):
   - **predicted_return / confidence / strength**: recomputed via a standalone
     `MLBasicSignalGenerator(model_type="basic", symbol="ETHUSDT").generate_signal(df, idx)` call
     at the trade's entry bar — the exact backtest-live-parity signal-generation path (same method
     the confidence-calibration study used), against the same cached ETHUSDT 1h data. Sanity check:
     min confidence in the resulting sample is 0.05022 (≥ the `min_confidence=0.05` gate) and min
     `|predicted_return|` is 0.004185 (≈ 0.05/12 = 0.004167) — both match the strategy's own gate
     almost exactly, confirming correct alignment between my recomputed signal and the trades the
     backtest actually took.
   - **realized_vol_entry**: trailing 24-bar stdev of 1h log returns ending at the entry bar (plain
     descriptive statistic on cached OHLCV, not a duplicated trading calculation).
   - **regime**: `RegimeDetector().annotate()` (`src/regime/detector.py` — the same detector class
     `hyper_growth`'s `LeveragedPositionSizer` consumes at runtime) run **once** over the full
     continuous history (its hysteresis loop is sequential/stateful and must run start-to-finish),
     then read per-trade at the entry bar (`trend_label`: trend_up/trend_down/range).
   - **hour_utc / session**: from `entry_time` directly (asia/europe/us/late_us_early_asia
     4h-ish buckets).
3. **`ev_conditioning_analysis.py`** — pre-committed test design (written before reading any
   p-value): Spearman + Cochran-Armitage trend tests for the two continuous/ordinal conditioners
   (predicted-return magnitude, realized vol), Kruskal-Wallis + chi-square for the two categorical
   conditioners (regime, session).

### 2.3 Collinearity: 5 named conditioners are really 4 independent tests

The task named five candidate observables: predicted-return magnitude, realized vol, regime,
hour/session, and entry-signal strength. Checked before running any test:
`corr(confidence, strength) = 1.0` exactly, ratio `strength/confidence = 0.83333...` (=10/12, the
two signal-generator multipliers) to float precision, with **zero clipping** in this population
(max confidence 0.556, max strength 0.463 — both well below the 1.0 saturation point). Confidence,
strength, and `|predicted_return|` are the same underlying quantity under three linear rescalings
in HyperGrowth's operating range. **Treated as one conditioner, not three** — this is why the
Bonferroni correction below uses **4 conditioners × 2 outcomes = 8 tests**, not 10.

### 2.4 Results

| Conditioner | n | Q1/lowest group | Q4/highest group | pnl test (p) | win-rate test (p) |
|---|---|---|---|---|---|
| Predicted-return magnitude / confidence / strength | 136 | win 70.6%, mean_pnl −0.18% (Q1, mean conf 0.066) | win 61.8%, mean_pnl −0.17% (Q4, mean conf 0.266) | Spearman ρ=−0.066, p=0.446 | Cochran-Armitage Z=−0.48, p=0.630 |
| Realized volatility at entry | 136 | win 73.5%, mean_pnl −0.09% (Q1) | win 58.8%, mean_pnl −0.18% (Q4) | Spearman ρ=−0.113, p=0.192 | Cochran-Armitage Z=−1.12, p=0.261 |
| Regime (trend_label) | 136 | range: n=31, win 80.6%, mean_pnl −0.008% | trend_up: n=53, win 58.5%, mean_pnl −0.31% | Kruskal-Wallis H=4.62, p=0.099 | chi2=4.57, p=0.102 |
| Hour-of-day session | 136 | us: n=56, win 69.6% | europe: n=26, win 57.7% | Kruskal-Wallis H=0.75, p=0.862 | chi2=1.31, p=0.727 |

**Decision table (Bonferroni α = 0.05/8 = 0.00625, two-sided): no conditioner reaches significance
— none even reaches the raw, uncorrected 0.05 threshold.** Full output:
`experiments/ev_conditioning_results.json`.

**One near-miss worth flagging plainly (not overselling it):** regime shows the closest-to-notable
pattern — `range` regime trades win 80.6% of the time at essentially breakeven mean P&L, versus
`trend_up`/`trend_down` at 58-62% win rate and more negative mean P&L. Raw p=0.099-0.102, **8-16x**
above the Bonferroni bar and not even under the uncorrected 0.05 line. Reported per the
anti-p-hacking norm (show the near-miss, don't silently drop it), not treated as a finding.

### 2.5 Verdict (per the pre-stated interpretation)

Per the task's own pre-registered framing: *"a significant, monotone EV gradient in ANY observable
justifies preregistering a conditional-sizing experiment; no gradient anywhere = flat sizing is
actually optimal for this signal and the audit's Hunt-5 concern is moot — either answer is
valuable."*

**No conditioner cleared even the raw significance threshold, let alone the multiple-comparison-
corrected one. Flat sizing is not leaving conditionable edge on the table for this signal.** This
result is consistent with — and extends to the trade level — the 2026-07-05 confidence-calibration
study's bar-level null (`|predicted_return|` magnitude showed no OOS hit-rate gradient on 4,415
individual bars; here, the same null holds on the 136 actual *executed, post-gate* trades using
three additional conditioners the calibration study didn't test).

**Recommendation: do not preregister a conditional-sizing experiment off this signal.** The audit's
Hunt-5 concern ("FlatRiskManager's binary gate discretizes a real-but-thin edge — a predicted
+0.45% and +5% get identical size") remains true as a mechanical description of the code, but this
analysis finds no evidence that magnitude (or vol, or regime, or time-of-day) actually predicts
which trades are better, so there is nothing for a magnitude-aware sizer to condition on
profitably. Building one now would be sizing by noise — exactly the failure mode the calibration
study already diagnosed for the gate; this shows it would extend to a sizer too.

### 2.6 How this could be misleading (adversarial self-review)

- **n=136, single strategy/symbol/timeframe.** This is a real, if modest, sample — larger than the
  calibration study's Phase 3 exam (46-55 trades) but far smaller than its Phase 2 bar-level test
  (4,415 bars). A gradient could exist at a magnitude too small for 136 trades to detect; absence
  of evidence is not proof of absence, just a well-powered-enough null given the pre-committed
  effect size implicit in a Bonferroni-8 design.
- **Model-quality ceiling caveat carries over.** Per the signal-path audit and multiple prior
  studies, this model sits at a ~53% directional-accuracy ceiling. If the *entry* signal itself
  carries little information (established elsewhere, not re-litigated here), it would be
  unsurprising for none of its correlates (magnitude, vol, regime, time) to carry conditionable
  information either — this null is consistent with, not independent evidence against, that
  broader finding.
- **Regime detector reused as-is, not re-validated here.** `RegimeDetector`'s specific
  slope/ATR-window hyperparameters were not re-tuned or accuracy-checked in this study; the
  regime-label null could reflect an imprecise regime detector rather than a true absence of
  regime-conditioned EV. Flagged, not chased — out of scope for this measurement task.
- **One symbol, one live model snapshot, contiguous history windows.** F1-F3 are the same
  windows used across the rest of this research program for comparability, but that also means
  any single confound present across all three folds (e.g. the model's training-data leakage into
  every fold, already disclosed in the exit-geometry study) applies here unchanged.

---

## Artifacts

- `experiments/fills_scoped.csv` — the 24 fill rows (from `orders`).
- `experiments/slippage_measurement.py`, `experiments/slippage_fills.jsonl` — Part 1 driver +
  per-fill results.
- `experiments/ev_conditioning_control_trades.py`, `experiments/ev_conditioning_control_trades.jsonl`
  — Part 2 control-arm trade extraction (136 trades, corrected symbol config).
- `experiments/ev_conditioning_features.py`, `experiments/ev_conditioning_features.jsonl` — entry
  observables per trade.
- `experiments/ev_conditioning_analysis.py`, `experiments/ev_conditioning_results.json` — the
  statistical test suite and decision table.

## Recommendation to PM

**Part 1 (slippage constant): ready for a small, low-risk action** — do not adopt the audit's
proposed cut; optionally nudge `DEFAULT_SLIPPAGE_RATE` from 0.0005 to 0.0004 (weak evidence, not
worth spending review budget on alone) or leave unchanged (equally defensible, arguably safer given
the fat right tail observed). No re-exam of tp_06/marginal verdicts is warranted under a "corrected"
slippage constant, since the correction this study finds is small and in the opposite direction
from what was hypothesized. This does not touch live-affecting code — it is a research-harness
constant used by backtests/exams, not by live trading (live trading doesn't model slippage, it
just gets whatever price it gets). No `risk_review_required`.

**Part 2 (EV-conditioning): closed, not "promising but not ready" — a clean negative result.** No
conditional-sizing experiment should be preregistered off predicted-return magnitude, realized vol,
regime, or time-of-day for the current ETHUSDT/HyperGrowth signal. This closes the loop GH #984
opened on the audit's Hunt-5 concern.
