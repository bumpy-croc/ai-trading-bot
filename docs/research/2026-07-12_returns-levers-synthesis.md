# Returns Levers Synthesis — 2026-07-12

**Author**: quant-researcher
**Audience**: Board (Alex) — where return improvement will and will not come from, given everything proven as of today
**Status**: Synthesis of completed research, not itself a new experiment. Every claim below is cited to its source doc; nothing here is a new number. Evidence tables live in the cited docs — this file does not reproduce them beyond the minimum needed to support a ranking.
**Scope note on source availability**: two of the cited reports are still on open PRs, not yet on `develop` — PR #970 (`docs/research/experiments/2026-07-12_exit-geometry-honest.md`) and PR #973 (`docs/research/experiments/2026-07-12_input-screening-nonlinear.md`). A third, `docs/research/experiments/2026-07-06_architecture-tournament.md`, was committed to a session branch (`claude/elated-vaughan-5625e7`) but was **never opened as its own PR and is not on `develop` today** — its findings are readable via GH issue #939 (which quotes the report's own headline numbers) or via that branch's git history. This is flagged here as a documentation-hygiene gap, not silently patched over.

---

## 1. The structural conclusion, stated plainly

Five independent experiments, run over the last two weeks, each varied exactly **one** lever on the ETHUSDT/1h `basic` model — training window, model architecture, target/label design, and (twice) the feature set itself across six alternative-input classes — while holding everything else fixed. All five converged on the same answer. In the nonlinear input-screen's own words, describing the full run of evidence to date:

> "four tournaments (window #898, architecture #939, target-redesign) plus two screens (linear, this nonlinear re-screen) — all finding the same ~51–53% DA ceiling under every lever tried: training window, model architecture, target/label design, and now both a linear and a nonlinear view of six alternative input classes."
> — `docs/research/experiments/2026-07-12_input-screening-nonlinear.md`

Concretely:
- **Window tournament** (`docs/research/experiments/2026-07-05_window-tournament.md`, #898): three training-window variants, all net-negative OOS (-7.3% to -11.3%) over a 185-day unseen bear market; window choice recovers a few points of drawdown/PF, not profitability.
- **Architecture tournament** (`docs/research/experiments/2026-07-06_architecture-tournament.md`, #939, unmerged — see scope note): 5 architectures (cnn_lstm, attention_lstm ×2, tcn ×2) cluster within a 1.29pp directional-accuracy (DA) band (53.16%–54.45%), no pairwise gap survives correction for 10 implicit comparisons.
- **Target-redesign tournament** (`docs/research/experiments/2026-07-10_target-redesign-tournament-results.md`, #933/#957): four target reformulations (binary direction, triple-barrier, smoothed-return, meta-labeling) — three collapse to predicting the unconditional/majority-class distribution (confirmed via probability-distribution diagnostics, across two different model families); the fourth (triple-barrier) "wins" on the L1 letter but is a label-imbalance artifact in 2 of 3 folds, and every entrant's money-exam profit factor sits at 0.31–0.58 (net-lossy on every fold, no exception). A same-contract linear baseline scores 53.2–53.6% DA, matching the deployed neural architecture almost bar-for-bar.
- **Linear input screen** (`docs/research/experiments/2026-07-12_input-screening-linear.md`, #967/#969): 6 alternative-input classes (realized-vol/range, calendar, BTC-cross, funding rate, basis/premium, Fear&Greed) + all-combined, tested with logistic regression — zero arms graduate against a pre-committed bar (McNemar-significant on ≥2/3 folds AND ≥0.5pp average DA gain). Largest average delta: funding_rate at +0.37pp, still short.
- **Nonlinear input re-screen** (`docs/research/experiments/2026-07-12_input-screening-nonlinear.md`, PR #973): identical arms/folds/bar, LightGBM instead of logistic regression — zero arms graduate again. One partial exception worth naming precisely (not smoothed into the clean null): `btc_cross` clears Bonferroni significance overwhelmingly in one fold (F1, Δ+3.84pp, p=6.9e-05) but not the other two (F2 p=0.226, F3 sign-reversed p=0.741) — real, regime-dependent, not persistent, correctly not graduating under the ≥2/3-fold rule.

**What this means economically at our fee/size level.** A ~51–53% DA ceiling is 1–3 percentage points above a coin flip. The `CostCalculator` defaults used throughout this research (`fee_rate=0.001`, `slippage_rate=0.0005` per side, never disabled — `docs/research/experiments/2026-07-12_exit-geometry-honest.md` §Metrics) are not a rounding error against an edge this thin, and the empirical L2 (money) exams bear this out directly rather than requiring a derived estimate: every tournament's profit factor, on every entrant, every fold, sits below 1.0 — window tournament's best variant PF 0.673, target-redesign's four entrants 0.31–0.58 across F1–F3, exit-geometry-honest's control PF 0.44–0.66 across F1–F3. **No lever tested so far — window, architecture, target shape, or feature set — has produced a model-quality-driven edge that clears round-trip transaction costs on ETHUSDT/1h.** The practical read for the Board: the "find a smarter model" well is dry for this specific symbol/timeframe/feature contract. Further return improvement will not come from another pass at entry-signal quality. It has to come from how a trade is managed once entered, how many independent instances of whatever edge exists we can harvest, closing measurement gaps that may mean the live picture is genuinely different from what these backtests show, or accepting a narrower, lower-confidence, regime-conditional edge.

---

## 2. The levers map, ranked

My working ranking below, evaluated against the evidence rather than accepted as given — I confirm three of the PM's five placements as stated, elevate one, and sharpen the caveats on the top-ranked lever, which is currently the single most action-adjacent item but is weaker than "the winning lever" framing implies.

### (a) Exit/trade-management, round 2 — confirmed #1, with a sharper caveat

**Evidence.** `docs/research/experiments/2026-07-12_exit-geometry-honest.md` (PR #970, issue #971) tested 6 exit-config arms against live prod control across F1–F3 (2023H1/2024H1/2025H1), fees/slippage on, Bonferroni-corrected bootstrap significance. **Verdict: NO-GO for all 6 arms** — none clears the pre-committed bar (return + PF improvement on every fold, Bonferroni-significant, MaxDD inside cap). Two findings matter for ranking:
1. **Stop-tightening is monotonically harmful, not helpful, on every fold** — this refutes the tightening direction of the prior (pre-#838/#867) hypothesis on corrected plumbing, and rules out the cheapest, most obvious "fix."
2. **`tp_06` is the only arm, across this entire research program, with a directionally-positive result on all 3 folds** (return Δ +0.49/+0.97/+1.39pp; PF 0.743 vs 0.662, 0.592 vs 0.528, 0.457 vs 0.446 vs control) — but every delta is statistically indistinguishable from zero (p=0.94/0.85/0.81) at the trade counts available (28–70/fold). This is exactly the report's own pre-committed "promising but not ready" category, not a win.

The mechanism has independent, real-fills corroboration: `docs/research/notes/2026-07-12_live-trade-review.md` (12 closed trades, sample-size caveat stated up front) found live winners capture ~72% of MFE on average and losers ride ~91% of MAE to the stop — and the exit-geometry-honest control's own backtest numbers (capture 0.73–0.82, MAE-ride 0.86–0.94 across folds) land in the same neighborhood as those live figures, which is the load-bearing check that the mechanism generalizes between live and backtest even though trade *frequency* does not (see lever (c)).

**Why the true trailing-stop/MFE-conditioned fix couldn't be tested yet.** The exit-geometry-honest expressibility audit found trailing-stop activation/distance, breakeven threshold, and the partial-exit ladder are all **hardcoded** in `hyper_growth.py`'s own `set_risk_overrides` call — not reachable via `RiskParameters` or any factory kwarg without a `src/` change. A true MFE-conditioned early-cut rule ("flatten if unrealized MFE hasn't reached X% within Y hours") doesn't exist as a policy anywhere in `ExitHandler`/`RiskManager` today. This is filed as **GH #971 (open)** — the closest proxy tested, an unconditional 18h time cutoff (`maxhold_18`), improved profit factor in 2 of 3 folds but made total return and MaxDD worse in *every* fold by multiplying trade count 7–10x (a "PF-improves-but-return-worsens" trap the report calls out explicitly, worth the Board knowing since a naive PF-only read would be misled).

**My critique of the PM's framing**: "the only lever with a directionally-positive result" is true but a low bar given five of six sibling arms in the same study got monotonically worse and every other lever this cycle found literally nothing. The effect size that does exist (~1pp return, ~0.1 PF) is also small in absolute terms at $87–$1,000 capital. **Confirmed as the top-ranked lever, but recommend budgeting it as a multi-week item, not a quick win**: it needs a real `src/engines/shared`/`ExitHandler` feature (money-path-adjacent, mandatory `architecture-reviewer` + `risk-officer` review), its own preregistration, and — separately, cheaply, and startable immediately since `tp_06` is *already* expressible today without any code change — a bigger-sample `tp_06`-focused follow-up (more folds, e.g. adding 2019–2022 half-years, for the statistical power this round's 28–70 trades/fold lacked). Note for that follow-up: earlier folds will have smaller training sets and the exit-geometry-honest report already found a non-differential ~2.8%-of-bars signal-generator failure rate in one tested window (Q1 2023) — worth checking it doesn't get worse further back before locking a prereg.

### (b) Trade frequency / symbol diversification — confirmed #2, but currently a scoping question, not a lever with numbers yet

**Evidence.** No experiment in this cycle quantified diversification's effect on expectancy variance or drawdown — this is a genuine, disclosed gap, not a result to cite. What *is* verified: a native BTCUSDT `basic` model registry already exists (`src/ml/models/BTCUSDT/basic/`), so this is not a from-scratch build.

**The honest logical point, not yet an established finding**: if ETHUSDT's edge is genuinely near-noise (as §1 establishes), adding an uncorrelated symbol helps only via the standard diversification argument — more independent draws can smooth aggregate P&L variance and drawdown timing. It does **not** raise any individual symbol's directional edge above whatever ceiling that symbol's own feature/model relationship has (untested for BTCUSDT) — and it is a double-edged sword: more instances of an edge that is currently net-negative after costs compounds losses exactly as readily as gains. **Any diversification proposal must be paired with BTCUSDT's own L1 (DA/ceiling) and L2 (money-exam, same fee/slippage regime) read before being treated as a return lever rather than only a variance-smoothing one** — every single-symbol exam run so far is net-negative, and there is no reason yet to assume BTCUSDT is different.

**Recommendation**: real, medium-tractability lever, correctly ranked #2, but the first deliverable is a scoping exam (does BTCUSDT show the same ~51–53% pattern, or an untested different one?), not a capital-allocation decision.

### (c) Live-vs-backtest parity gap — elevated above the PM's tentative #3

**Evidence.** `docs/research/notes/2026-07-12_live-trade-review.md`'s pass-4 comparison, matched config and window (2026-06-02→2026-07-12): live produced 12 closed trades / ≈+9.0% aggregate P&L vs. a matched backtest's 6 trades / -0.78% — 2x trade-count divergence, sign-flipped return, and an order-of-magnitude difference in average winner/loser size. This is **far outside the charter's 15% backtest/live parity KPI band** (`.claude/state/charter.md` KPI #2). The review explicitly ties this to two already-known mechanisms rather than treating it as new: the 2026-07-06 forming-bar fliprate finding (live decides against a mutating tail candle; a closed-bar backtest never evaluates those signals) and the 2026-07-08 confidence-collapse finding (HyperGrowth's entries cluster right at a low-confidence boolean gate — median ~0.03–0.04 — exactly where small live/backtest microstructure differences are most likely to flip a decision).

**Why this deserves a higher rank than "one lever among several."** If live behavior genuinely and persistently differs from backtest in ETHUSDT's favor — and the review is explicit that **no expectancy claim is licensed from n=12 trades** — then every backtest cited in §1, including all five convergent-null results, was run through the same closed-bar harness that may be measuring something systematically different from what the live engine actually does. That would not just add a lever; it would revise the confidence attached to every other result in this document. This is a measurement-validity question with unusually large potential impact if it resolves the way the live sample points, which is why I'd move it up from the PM's tentative #3 slot.

**Recommendation**: frame the next step exactly as `docs/research/notes/2026-07-12_live-trade-review.md`'s own H4 does — a scoped, preregistered forming-bar-aware backtest variant compared against the same live window, testable without first landing a money-path `src/` change (a backtest-harness variant is not itself a live-affecting change). Tractable to start now, moderate effort; the deliverable is "how big is the gap and why," not a proposal to change anything yet.

### (d) `btc_cross` regime-conditional lead-lag — confirmed #4, narrower than the brief implied

**Evidence.** The nonlinear screen (`docs/research/experiments/2026-07-12_input-screening-nonlinear.md`) found `btc_cross` clears Bonferroni significance by four orders of magnitude in F1 (Δ+3.84pp, p=6.9e-05) but not in F2 (Δ+1.16pp, p=0.226) or F3 (sign-reversed, Δ-0.33pp, p=0.741) — correctly failing the pre-committed ≥2/3-fold graduation rule. Feature-importance/gain shows `btc_ret_1h`/`btc_ret_6h` carry real, consistent gain in **all three** folds even where the DA improvement doesn't generalize — the tree finds a real, repeatable relationship that only translates into significant OOS accuracy in one of three regimes tested. This is **not yet preregistered** as its own experiment; it is a named, narrower open question the screen recommends but does not schedule.

**Recommendation**: confirmed at #4, exactly as the PM proposed — worth naming and preserving as a candidate, not worth prioritizing ahead of (a)–(c). If pursued, it needs its own regime-interaction-scoped prereg (which vol/trend axis plausibly gates the effect), not a rerun of the existing screen.

### (e) Longer horizons/timeframes (4h/1d) — confirmed #5, genuinely untested rather than falsified

**Evidence.** None of the five convergent experiments varied timeframe — every one is fixed at ETHUSDT/1h. This is the one lever in this map that has never been run, as opposed to run-and-failed.

**Tractability assessment.** The fold/embargo/McNemar harness used across the target-redesign tournament and both input screens is timeframe-agnostic by design (train-cutoff/embargo/eval-window logic doesn't hardcode 1h bars), so the statistical-testing scaffolding transfers cheaply. What does **not** transfer for free: `PriceOnlyFeatureExtractor`'s 120-bar lookback means 120 bars at 4h ≈ 20 days (plausible), but at 1d ≈ 120 days — a much longer lookback with correspondingly fewer independent training rows per fold, which could hurt the already-marginal significance tests further (current 1h folds already sit at n≈4,300 eval bars with single-fold SE≈0.4–0.8pp). Fewer bars per unit time also means far fewer trade opportunities per fold, changing the fee/slippage-drag-vs-signal-strength economics in an as-yet-unknown direction (fewer round trips to pay for, but also fewer compounding opportunities). The underlying OHLCV cache already covers full history at coarser granularities per the input-candidates audit, so there's no new data-acquisition cost.

**Recommendation**: confirmed as lowest priority given nothing has failed *enough* yet to justify a timeframe pivot, but it is the cheapest of the open, unfalsified levers to sanity-check — worth an opportunistic, off-critical-path first pass rather than active deprioritization.

---

## 3. What is formally retired (with the receipts)

| Lever | Verdict | Source |
|---|---|---|
| Input classes: realized-vol/range, calendar, funding rate, basis/premium, Fear & Greed | Retired for ETHUSDT/1h within this feature contract — zero arms graduate under a pre-committed bar in **both** a linear and a nonlinear detector family | `docs/research/experiments/2026-07-12_input-screening-linear.md` (#967/#969); `docs/research/experiments/2026-07-12_input-screening-nonlinear.md` (#973) |
| `btc_cross` (blanket 4-feature addition) | Retired **as a blanket addition** — not fully closed; narrowed to a separate, unscheduled regime-conditional question (lever (d) above) | `docs/research/experiments/2026-07-12_input-screening-nonlinear.md` |
| Target/label reformulation | Retired — no entrant proceeds to L3a staging; 3 of 4 collapse to unconditional-distribution prediction across two model families; money-exam PF 0.31–0.58, net-lossy on every fold, no exception | `docs/research/experiments/2026-07-10_target-redesign-tournament-results.md` (#933/#957) |
| Model architecture search | Retired — 5 entrants cluster within a 1.29pp DA band; no pairwise gap survives correction for 10 implicit comparisons; ensemble Phase 2 not justified | `docs/research/experiments/2026-07-06_architecture-tournament.md` (#939 — unmerged, see scope note) |
| Training-window curation | Retired — full history ties-or-beats shorter windows on return, wins on PF/MaxDD; all 3 variants net-negative OOS regardless | `docs/research/experiments/2026-07-05_window-tournament.md` (#898) |
| Stop-tightening (subset of exit geometry) | Retired — monotonically worse return/MaxDD on every fold, every tightening arm; refutes the pre-#838/#867 hypothesis on corrected plumbing | `docs/research/experiments/2026-07-12_exit-geometry-honest.md` (#970/#971) |

Not retired, and explicitly not authorized here: any live/staging change. All six results above are screening/tournament verdicts with "nothing to stress-test" recommendations in their own text — no proposal accompanies this synthesis.

---

## 4. Recommended next-session sequence

Ordered by what feeds the next preregistration fastest, not strictly by §2's impact ranking — items 1–2 are cheap scoping passes that inform whether 3–4 are worth their cost.

1. **BTCUSDT parallel L1+L2 scoping exam** (feeds lever (b)'s prereg). Reuses the existing tournament harness and an already-trained model; no new data acquisition. Est. ~1 session. Deliverable: does BTCUSDT show the same ~51–53% DA ceiling, or something genuinely different, at the same fee/slippage settings.
2. **Forming-bar-aware backtest variant, design/estimation pass** (feeds lever (c)'s prereg). Not a full implementation — first pass is sizing the harness change and confirming the mechanism against a second live window before committing to a build. Est. ~1 session.
3. **`tp_06` bigger-sample follow-up preregistration** (feeds lever (a)'s statistical-power gap). Already expressible today, no `src/` change needed — can start immediately, in parallel with item 4. Est. 1–2 sessions including fold selection and data-quality spot-checks on older windows.
4. **GH #971 build**: `ExitHandler` MFE-conditioned early-cut policy + real `RiskParameters`-driven trailing-stop/breakeven knobs for `hyper_growth`. Money-path-adjacent `src/` change — requires `architecture-reviewer` and `risk-officer` in the loop before any exam result from it counts. Multi-session; this is the long pole in lever (a).
5. **`btc_cross` regime-interaction preregistration** (lever (d)). Only after 1–4 stall, or as an explicitly low-cost side thread — needs its own hypothesis about which regime axis gates the effect before any run.
6. **4h/1d timeframe cheap sanity pass** (lever (e)). Opportunistic, off the critical path; reuses the existing fold harness.

**Documentation-hygiene item, not a research lever**: the 2026-07-06 architecture tournament report should be opened as its own PR against `develop` so it stops being institutional knowledge that only exists on a stale session branch — flagged here, not actioned in this document.

---

**For risk-officer**: nothing in this synthesis proposes a live-affecting change. Every underlying experiment already carries its own "nothing to stress-test" verdict. The one item that will eventually need risk-officer review is GH #971's build (item 4 above), when it exists.

**For pm**: recommend sequencing per §4. Lever (a)'s `tp_06` follow-up and lever (b)'s BTCUSDT scoping exam can both start next session with no prerequisite work; lever (c)'s parity-gap sizing pass is the one item I'd personally prioritize highest if session capacity is constrained, given its potential to revise how every other result here should be read.
