# Short-suppression counterfactual — does the SHORT-inventory guard cost or save returns? (GH #990)

**Date**: 2026-07-12
**Researcher**: quant-researcher
**Status**: PREREGISTERED — thresholds below (Sec. 3) locked before any backtest ran. Results appended below the `## RESULTS` marker.
**Worktree**: `.claude/worktrees/short-suppression-990`, branch `claude/short-suppression-990`, off `origin/develop @ 2f0dff5c` (`.agent-active` sentinel present).
**Related**:
- GH #990 — the finding this investigates (9 LONG vs 3 SHORT executed trades vs ~50/50 signal split).
- `docs/research/notes/2026-07-12_parity-gap-investigation.md` (PR #987) — Finding 2 established the funnel mechanism (SHORT-side margin/inventory guard, `execution_engine.py:663-706`, fail-closed rejects shorts whenever free ETH > $1 dust) and Finding 1 established the model-version confound this note must respect.
- GH #1006 (merged) — `--model-version`/`--model-as-of` point-in-time model pinning, used throughout this note.
- `docs/research/experiments/2026-07-12_exit-geometry-honest.md` — precedent for the "fixed-entries, in-sample-but-caveated relative comparison" methodology reused in Sec. 5.2, and source of the F1/F2/F3 fold windows reused here for comparability.

## 1. Hypothesis

**H1**: The SHORT-inventory guard's near-total suppression of short entries (GH #990: 9L/3S executed vs ~50/50 signal split) has a material, directionally consistent effect — either costing or saving — on HyperGrowth/ETHUSDT realized returns, relative to what the strategy would have produced with shorts allowed to enter freely (subject to existing risk sizing/stops).

**Falsifiable statement**: H1 is **supported in the "costing returns" direction** only if the shorts-enabled arm beats the long-only arm by a consistent, non-trivial margin across a majority of the windows tested (Sec. 3). It is **supported in the "saving returns" direction** only if the reverse holds. Otherwise H1 is **not supported at the current sample size** (recommendation iii).

## 2. Metric

Per window and pooled: total return delta (shorts-enabled − long-only), profit factor, max drawdown, trade count (total / short / long), and the short-side trades' standalone summed `pnl_percent`. Sharpe/Sortino are reported where the trade count is large enough to be non-degenerate (folds), omitted as uninformative for single-digit-trade windows.

## 3. Success threshold — PRE-COMMITTED before any backtest ran

Chosen against HyperGrowth's own realized scale: base risk per trade is 2% (`risk-limits.json`), and live's actual per-trade swings over the investigated window run from −10.00% to +3.97% (trades table, Sec. 4). A threshold below roughly one typical trade's worth of return is not distinguishable from noise; a threshold that only one dominant trade could produce is not a "consistent" finding.

- **"Costing returns" (recommendation i, → risk-officer guard-redesign review)**: shorts-enabled total return exceeds long-only by **≥2 percentage points** in a **majority of windows with any short trades at all**, AND short-side standalone P&L is net positive in those windows, AND profit factor is not worse for shorts-enabled.
- **"Saving returns" (recommendation ii, → propose long-only as official config)**: the mirror image — long-only exceeds shorts-enabled by ≥2pp in a majority of windows, and/or short-side standalone P&L is net negative across windows where shorts fired.
- **"Inconclusive" (recommendation iii)**: fewer than 3 windows produce any short trade at all; OR the sign of the per-window delta flips with no majority; OR every window's |delta| < 1pp (indistinguishable from fee/turnover noise). A single-window result — in either direction — is reported as suggestive only, never as the basis for a verdict, regardless of its magnitude.
- Any per-fold delta driven by 1–2 trades is flagged as anecdotal in the writeup even if it happens to clear the numeric bar.

## 4. Risks of false positive

- **Small samples everywhere.** The live-matched segment (Sec. 5.1) has at most a handful of trades; even the supplementary folds (Sec. 5.2) are single-digit-to-low-double-digit trade counts for HyperGrowth's lumpy signal. A clean-looking delta can be 1–2 trades' idiosyncratic outcome, not a real effect of direction-gating.
- **Model-version confound (already found once, GH #988/#1006).** Any comparison must pin the model actually in force for the dates tested — done throughout via `--model-version`/registry pin, never `latest` resolved fresh.
- **In-sample optimism (Sec. 5.2 only).** The supplementary folds reuse dates the live model's training cutoff (2026-07-04) has already seen. This inflates absolute numbers for both arms equally — inherited caveat from the exit-geometry-honest precedent — but does not bias the *relative* arm-vs-arm delta, which is the only thing Sec. 5.2 is used for.
- **Survivorship in "what would have happened."** The counterfactual assumes the exchange would have accepted every unblocked short at the modeled price/size with standard fees+slippage; real margin/borrow mechanics (the actual subject of the guard) are not modeled at all in backtest, by design (per the parity investigation). This note estimates the *signal-quality* side of the question, not the *margin-safety* side — that remains risk-officer's call regardless of this note's verdict.

## 5. Methodology

### 5.1 Live-matched segments (primary)

Live's model changed at the 2026-07-05 ETHUSDT/basic promotion (`docs/research/model-promotions.md`; only version ever registered: `2026-07-04_22h_v1`, `metadata.json created_at` 2026-07-04T22:44:32Z). The investigated live window (2026-06-02 → 2026-07-12, per GH #990/#987) is split at that boundary:

- **Segment A (2026-06-02 → 2026-07-05)**: live ran a cross-symbol substitute (BTCUSDT/basic scoring ETHUSDT bars, `FEATURE_ALLOW_CROSS_SYMBOL_MODEL`) — the parity investigation already established this is **not reconstructable** from current repo state (only one ETHUSDT/basic version has ever existed; the substitution code path itself only activates when no native bundle is registered, which is no longer true). Confirmed again here: `atb backtest hyper_growth --symbol ETHUSDT --start 2026-06-02 --end 2026-07-04 --model-as-of 2026-06-15` fails closed with `ModelNotAvailableError` (Sec. 6.1) — the harness correctly refuses to fabricate this comparison rather than silently mis-scoring it. Segment A is therefore **forensics-only** (Sec. 6.2) — no counterfactual return estimate is produced for it, to avoid exactly the mistake Finding 1 already flagged once.
- **Segment B (2026-07-05 → 2026-07-12)**: the native model was (per the promotion record) in force. Pinned via `--model-version 2026-07-04_22h_v1`, matched to live's actual initial balance at the boundary (~$84.40, `account_history`). This is the only segment where "the model live actually ran" is reproducible, so it is the only segment eligible for a live-matched counterfactual — reported in full in Sec. 7.1 regardless of how sparse it turns out (pre-committed: it will not be discarded for being inconvenient).

  Caveat carried into the results: `trading_sessions` shows the live process's session row (id 20) has been continuously open since 2026-06-05 with no visible restart, and `PredictionEngine.reload_models()` has zero callers (parity investigation, re-confirmed here) — so the 2026-07-05 promotion date is the *symlink* flip, not independently confirmed proof that the running live process picked up the new model at that exact moment. Used as given per the task framing; flagged as an open question, not silently assumed solid.

### 5.2 Supplementary power check (secondary, not live-matched)

Segment B's window is 7 days — too short to power any conclusion alone, and Segment A cannot be backtested at all. To get a non-trivial trade sample for the arm-vs-arm delta itself (holding the model fixed, not claiming to match live's specific history), this note reuses the exact F1/F2/F3 fold windows from `docs/research/experiments/2026-07-12_exit-geometry-honest.md` (same symbol, same strategy, same signal source), pinned to the same `2026-07-04_22h_v1` model:

- F1 = 2023-01-01 → 2023-06-30
- F2 = 2024-01-01 → 2024-06-30
- F3 = 2025-01-01 → 2025-06-30

Reusing an already-vetted, already-published set of dates (rather than picking new ones) avoids the appearance of window-shopping for a favorable result. **This is explicitly in-sample relative to the model's training cutoff (2026-07-04)** — inherits the exit-geometry-honest study's own caveat verbatim: absolute P&L is non-conservative, but the relative arm(a)-vs-arm(b) delta under an identical, fixed entry-generation pipeline is the thing being tested, and is not invalidated by in-sample optimism affecting both arms equally.

### 5.3 Arms

- **(a) As-designed**: `hyper_growth` strategy unmodified — `MLBasicSignalGenerator` sets `metadata["enter_short"]=True` on every SELL signal (existing code, `src/strategies/components/ml_signal_generator.py:1044-1045`), so SELL signals enter real short positions via the pre-existing, shared `enter_short` opt-in gate (`src/engines/shared/entry_utils.py:87`, identical in both engines).
- **(b) Long-only (suppression proxy)**: identical strategy, data, dates, and model pin, with a thin **research-only wrapper** (not committed to `src/`) around the constructed strategy's `signal_generator.generate_signal` that clears `enter_short` back to `False` whenever the underlying generator set it — i.e., it withdraws the opt-in the generator makes by default, falling back to the engines' own documented default ("Runtime engines only allow SELL decisions to enter shorts when strategies explicitly opt in via `enter_short=True` metadata. Default to long-only.", `src/strategies/components/strategy.py:965-966`). No engine code, no shared math, and no gate logic is touched or duplicated — the wrapper only decides what the strategy hands to the *existing* gate. This is a closer proxy to live's realized behavior (near-100% short rejection) than a naive "delete short trades from the trade log post-hoc" approach, because it lets the sizer/balance path evolve exactly as it would with no shorts ever taken (compounding effects included), rather than just filtering a finished trade list.

Both arms: fees and slippage on (`CostCalculator` defaults, never disabled), `--disable-engine-sl` not set (engine-level SL/TP active, matching live), no `--use-sentiment`, `--provider auto`, cached data.

## 6. Forensics (bounded — read-only prod DB + repo)

### 6.1 Segment-A irreproducibility, reconfirmed

`atb backtest hyper_growth --symbol ETHUSDT --start 2026-06-02 --end 2026-07-04 --model-as-of 2026-06-15` — see Sec. 7 for the exact failure. Confirms the parity investigation's own conclusion still holds after #1006 shipped point-in-time pinning: the tooling now *can* pin a version, but there is still no version to pin for this period, and the harness correctly refuses rather than silently falling back to `latest`.

### 6.2 What IS recoverable — the funnel, split at the promotion boundary

Read-only against prod Postgres (`RAILWAY_PRODUCTION_DATABASE_URL` public proxy, `SET default_transaction_read_only = on` first statement of every session, per `prod-forensics` skill). No writes. Query: `strategy_executions` for ETHUSDT/HyperGrowth entry actions, 2026-06-02 → 2026-07-12 17:10 UTC, cross-referenced against `positions` open/close intervals to classify each row as "flat" (a genuine opportunity) vs "logged while already in a position" (per the parity investigation, the entry coordinator logs this decision unconditionally regardless of whether a real attempt follows).

| segment | action | logged while flat | logged while already positioned |
|---|---|---:|---:|
| A (pre-2026-07-05) | opened_long | 288 | 4,875 |
| A (pre-2026-07-05) | opened_short | 182 | 4,465 |
| B (2026-07-05 onward) | opened_long | 0 | 447 |
| B (2026-07-05 onward) | opened_short | 0 | 727 |

**New finding not in the original parity note (which stopped at 2026-07-02 13:34, position #22's entry)**: position #22 — a **SHORT**, entered 2026-07-02 13:34:24, price 1696.83 — has been continuously **OPEN** through the end of this window (2026-07-12, last_update still ticking, current price 1820.82 → roughly **−7.3% unrealized against the short**). Because it never closed, **segment B has zero flat time at all** — every logged entry row in segment B (447 long, 727 short) fired while a position already existed, so the SHORT-inventory guard specifically had **no opportunity to fire or not fire** in segment B; there is nothing here about the guard, just an absence of trading opportunity. This is a materially different, and non-trivial, characterization from segment A, where flat-period opportunities existed in real volume (470 total) and the direction split (288 long / 182 short, 61/39) already tilts long even before the guard's differential rejection is applied — consistent with, though less extreme than, the ~50/50 split over the fuller pre-07-02 window the original investigation measured (methodology differs slightly: position-interval join here vs. the original's trade-interval CTE; the original's numbers are the authoritative ones for the mechanism itself, cited not re-derived).

Realized trades in segment A (from `trades`, confirmed directly): **9 LONG / 3 SHORT** — reconfirms the GH #990 headline exactly. Attrition from flat-period opportunity to realized trade: longs 288→9 (96.9%), shorts 182→3 (98.4%) — both enormous, shorts modestly worse, consistent with (not independently proof of, beyond the code-level confirmation already established) the asymmetric guard.

**Honest limitation, unchanged from the original investigation**: no per-asset historical balance ledger exists (`account_balances` is USD-only), and Railway logs don't retain history for these dates, so individual rejection instances beyond the one already-confirmed 2026-06-07 22:14–22:59 episode (30 consecutive `opened_short` log rows, zero matching SHORT orders, resolved 45 minutes later by a LONG) remain **not** independently re-verifiable balance-state-by-balance-state. This note does not attempt to re-litigate that — it is priced in as a known gap, per `docs/research/notes/2026-07-12_parity-gap-investigation.md` §7.

**Side note, out of scope for this note's verdict but material enough to flag**: position #22 being open, deeply underwater, and unresolved for 10 days coincides with a documented connectivity incident (`system_events`, 2026-07-05 18:47 → 2026-07-06 11:39, user-data-stream circuit-open, "REST-degraded; real-time fills/balance updates unavailable") and a day (2026-07-08) where every one of 204 logged confidence scores reads exactly 0. Neither is investigated further here — flagging for `live-ops`/`risk-officer` attention as a separate matter, not folded into this note's shorts-suppression analysis.

## 7. Results

All 8 runs completed sequentially (single Python process, one backtest at a time — no parallel runs). Parity check before trusting any number: the runner reuses `cli.commands.backtest._load_strategy`/`_resolve_model_pin` verbatim and replicates `_handle`'s `RiskParameters` construction **including** honoring `strategy.get_risk_overrides()["max_fraction"]` (HyperGrowth's real 0.25 position cap) — an early draft of this runner omitted that and silently ran both arms at the CLI's generic 0.10 default instead of HyperGrowth's real cap (caught via a smoke test disagreeing with expectations, fixed before any of the numbers below were generated; the fold reruns after the fix are the only ones reported).

### 7.1 Segment B (live-matched, 2026-07-05 → 2026-07-12, `$84.40` initial balance)

| arm | trades (S/L) | total return | max DD | note |
|---|---:|---:|---:|---|
| shorts-enabled (as-designed) | 0 (0/0) | −0.27% | 0.70% | 0 *closed* trades, but a position opened and was still open at backtest end (marked-to-market loss) |
| long-only | 1 (0/1) | +0.11% | 0.48% | one LONG, entry 07-10 06:00 → exit 07-11 23:00, stop loss, +0.0015% |

**As pre-committed: this segment is degenerate and not used to drive the verdict.** Both arms traded essentially nothing in 7 days — consistent with §6.2's finding that live itself had zero flat-period opportunities in this window. Worth noting only as color: the shorts-enabled arm's still-open position at backtest end is a SHORT losing money as ETH rallies — the same qualitative pattern as live's real, separately-opened position #22 (short, entered 07-02, ~−7.3% unrealized as of this writing). Not the same trade, not a quantitative data point, but a directionally consistent cross-check that the backtest and live are seeing the same kind of adverse move in this stretch.

### 7.2 Supplementary folds (in-sample relative to model training cutoff — Sec. 5.2 caveat applies to every number below)

| fold | arm | trades (S/L) | total return | PF | max DD | win rate | short-side Σpnl_pct |
|---|---|---:|---:|---:|---:|---:|---:|
| F1 2023H1 | shorts-enabled | 29 (19/10) | −3.21% | 0.727 | 6.59% | 72.4% | −0.037 |
| F1 2023H1 | long-only | 23 (0/23) | −3.99% | 0.584 | 7.12% | 69.6% | — |
| F2 2024H1 | shorts-enabled | 40 (28/12) | −12.59% | 0.409 | 13.75% | 65.0% | −0.106 |
| F2 2024H1 | long-only | 50 (0/50) | −8.92% | 0.558 | 12.06% | 72.0% | — |
| F3 2025H1 | shorts-enabled | 67 (29/38) | −19.94% | 0.359 | 22.10% | 61.2% | −0.078 |
| F3 2025H1 | long-only | 62 (0/62) | −18.79% | 0.355 | 20.31% | 64.5% | — |

**Delta (shorts-enabled − long-only), the number the pre-committed threshold (Sec. 3) is applied to:**

| fold | Δ return (pp) | Δ direction | clears ±2pp bar? | PF comparison | MaxDD comparison |
|---|---:|---|---|---|---|
| F1 | **+0.78** | shorts help | no | shorts-enabled better (0.727 vs 0.584) | shorts-enabled better (6.59% vs 7.12%) |
| F2 | **−3.67** | shorts hurt | **yes** | long-only better (0.558 vs 0.409) | long-only better (12.06% vs 13.75%) |
| F3 | **−1.15** | shorts hurt | no | roughly tied (0.359 vs 0.355) | long-only better (20.31% vs 22.10%) |

Short-side standalone P&L (sized `pnl_percent`, summed): **negative in all three folds** (F1 −0.037, F2 −0.106, F3 −0.078) — and not driven by one outlier trade: the per-trade short P&L distributions (19/28/29 trades) are broad, roughly balanced between small winners and small losers, tilted slightly negative throughout (min/max per fold: F1 −0.0198/+0.0052, F2 −0.024/+0.0107, F3 −0.0247/+0.0073 — checked explicitly to rule out a single blown-up loser driving the sign).

**Applying the pre-committed rule (Sec. 3) honestly:**
- "Costing returns" (i) is **not supported**: shorts-enabled only beats long-only in 1 of 3 folds (F1), that fold doesn't clear the ±2pp bar, and short-side standalone P&L is negative in every fold — there is no fold where allowing shorts both won by a material margin **and** the shorts themselves made money. Rule out (i) with reasonable confidence.
- "Saving returns" (ii): the strict ≥2pp-in-a-majority clause is **not** met (only F2 clears it, 1 of 3, not a majority) — but the OR-clause (short-side P&L net negative across every window where shorts fired) **is** met, 3 for 3. Per Sec. 3 as literally written, this qualifies as (ii). Read plainly, though, the evidence is **directionally consistent but not overwhelming**: 2 of 3 folds favor long-only, magnitude ranges from a clear miss (F2, −3.67pp) to marginal (F3, −1.15pp, inside the "could be noise" band), and F1 is a real, non-trivial exception in the other direction (though even there, the shorts taken lost money standalone — F1's aggregate win for shorts-enabled came from how removing shorts changed the *long* trades taken and their compounding path, not from the shorts themselves being profitable).
- (iii) inconclusive is not the right label either — three folds is admittedly few, but the direction is consistent enough (short-side P&L negative unanimously) to be more than noise, just not strong enough to be a confident, final answer.

**Verdict: leans (ii), moderate confidence, not a slam dunk.** Suppression is more likely accidentally neutral-to-beneficial than costly, for this specific model/asset/regime combination — but "propose long-only and ship it" would overstate what 3 folds and one degenerate live-matched segment can support.

## 8. How this could lose money (adversarial self-review)

1. **Regime-drift confound, not a model-quality finding.** 2023–2025 was a broadly bull-biased multi-year stretch for ETH. Shorting an asset with positive drift structurally loses money on average almost regardless of signal quality — the window tournament (`docs/research/experiments/2026-07-05_window-tournament.md`, #898) already found long+short HyperGrowth net-negative OOS over a 185-day *bear* market (−7.3% to −11.3%). If the observed "shorts underperform" pattern here is mostly ETH's multi-year upward drift rather than a genuine long/short model-quality asymmetry, a long-only regime with no short hedge could do *worse*, not better, in a real sustained bear market — the exact opposite of what this note would otherwise recommend. This is the single biggest reason not to treat this as settled.
2. **The underlying model's directional signal is barely above a coin flip (~51–53% DA, `docs/research/2026-07-12_returns-levers-synthesis.md`).** A long/short asymmetry built on a near-noise-level signal is not guaranteed to be stable — a retrain, a target-redesign, or a different symbol could flip which side looks "bad" in-sample without any real underlying change. Codifying long-only as permanent strategy config risks locking in a pattern that is itself close to statistical noise.
3. **In-sample optimism (Sec. 5.2 caveat), inherited from the exit-geometry-honest precedent.** The relative arm-vs-arm delta is not invalidated by the model having seen this history in training (both arms share the same model), but the exact *magnitude* of any fold's delta could still look different against genuinely-unseen future data — this note's numbers should not be read as a forward P&L estimate for either arm.
4. **Small samples, again.** 19–29 short trades per fold is enough to rule out a single-outlier artifact (checked directly, Sec. 7.2) but not enough to call a ~1–4pp return delta a statistically solid effect — no significance test is claimed here, deliberately, per the pre-registration.
5. **This note does not evaluate the guard's actual purpose (margin/inventory safety).** Even a strong version of "shorts don't help returns" says nothing about whether `execution_engine.py:663-706`'s specific implementation (net free-ETH-balance vs. $1 dust) is the right mechanism for a cross-margin account that may legitimately hold ETH inventory. That is a separate, still-open engineering question for risk-officer regardless of this note's verdict — this note is silent on it by design and should not be read as "the guard is fine, don't touch it."

## 9. Recommendation

**(ii) — leans toward "suppression is accidentally not costing returns, and plausibly helping modestly" — but flagged as promising, not proven.** Concretely:

- **Do not touch the SHORT-inventory guard** (`execution_engine.py:663-706`) on the strength of this note. Its margin-safety rationale is independent of whether shorts help or hurt returns, and nothing here evaluated whether its specific implementation (net free-ETH vs. $1 dust) is well-designed — that remains open, per Sec. 8 point 5.
- **Do propose formalizing long-only as an explicit, intentional HyperGrowth/ETHUSDT configuration choice**, replacing the current state where "almost no shorts happen" is an accidental byproduct of unrelated margin plumbing. See the companion proposal, `.claude/state/proposals/2026-07-12-01-hypergrowth-ethusdt-long-only.md` — `risk_review_required: true`, explicitly **not** requesting immediate live promotion, recommending a staging-paper confirmation window given the regime-drift confound in Sec. 8, point 1.
- **What would raise confidence, if pursued later**: (a) more out-of-sample folds as they become available (this model's usable OOS window grows by exactly the number of days that pass, one day at a time — Segment B-style live-matched windows will accumulate this way); (b) a fold or two from a period with clearer bear-market character, to directly test point 1's confound rather than only inferring it from a different study; (c) the go-forward observability already recommended in the parity investigation (a durable `system_event` on every guard rejection) would let a *future* version of this note replace segment A's forensics-only treatment with an actual reconstructed counterfactual, once enough guard-rejection history accumulates under the native (non-substitute) model.

## 10. Reproducibility

- Prod DB queries: SELECT-only against `RAILWAY_PRODUCTION_DATABASE_URL` (public proxy via `railway variables -e production -s Postgres --json`, itself read-only per `.claude/LESSONS.md` §3), `SET default_transaction_read_only = on` as the first statement of every session. No writes.
- Backtest runner: `run_counterfactual.py` (session scratchpad, not committed — a thin wrapper around `cli.commands.backtest`'s real construction path; see Sec. 5.3 for exactly what it changes). `atb backtest hyper_growth --symbol ETHUSDT --start 2026-06-02 --end 2026-07-04 --model-as-of 2026-06-15` reproduces the Sec. 6.1 fail-closed demonstration directly via the CLI.
- Fees/slippage: on throughout, `CostCalculator` defaults, never disabled.
- All 8 backtest runs executed sequentially in a single process (no parallel runs), per standing Mac-thermal guidance.
