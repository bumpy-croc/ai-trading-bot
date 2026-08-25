# Experiment: `tp_06` Statistical-Power Follow-Up (Phase 1, Vol-Regime Program #3)

**Author**: quant-researcher
**Status**: PREREGISTERED — locked before any statistic is computed. Do not edit thresholds after seeing results; corrections are new dated sections.
**Program**: Vol/regime program (this doc's sibling `agents/research/vol-regime-program.md`).
**Blocked on**: `#1106` for any promotion decision — see §10. **Provenance check required before use as a baseline** — see §0.1.

---

## 0. Why this experiment, and what it is not

This is not a new idea — it is the pre-committed follow-up named by `docs/research/experiments/2026-07-12_exit-geometry-honest.md` itself. That study (6 exit-config arms vs. HyperGrowth control, F1=2023H1/F2=2024H1/F3=2025H1, Bonferroni-corrected bootstrap) found **`tp_06`** (`stop_loss_pct=0.10`, `take_profit_pct=0.06`, `max_holding_hours=336`) is the only arm across the entire returns-levers research program with a directionally-positive result on **all three folds** (return delta +0.49/+0.97/+1.39pp; profit factor 0.743 vs 0.662, 0.592 vs 0.528, 0.457 vs 0.446), but every delta was statistically indistinguishable from zero (p=0.94/0.85/0.81) at that study's available trade counts (28-70/fold). It is expressible today with **zero `src/` changes** — a config-only strategy-override change, same as HyperGrowth's own `take_profit_pct` override mechanism already documented.

This belongs in the vol-regime program, not as an odd inclusion, because of the mechanism: `tp_06`'s take-profit level (6%) sits below the first partial-exit tier (8%), so it fires a hard full-position close instead of engaging the partial-exit ladder/trailing-stop upside management. That is functionally a trade-management response to *how far price actually moves before reversing* — which is downstream of volatility, not direction. The exit-geometry-honest report's own mechanism cross-check found HyperGrowth's realized winning price moves are almost always far below even the tighter 15% TP level tested, meaning the TP knob's effect is really about **how much of the volatility-driven excursion gets captured before it's given back**, consistent with the same live-trade-review finding (winners capture ~72% of MFE, losers ride ~91% of MAE) that motivated this whole lever in the first place.

### 0.1 Provenance check (mandatory before this experiment starts)

`2026-07-12_exit-geometry-honest.md` is **not** on GH #1081's explicit "flagged for re-validation" list (that list names `2026-07-12_exit-geometry-round2.md` specifically), but it is also **not** on the "likely safe" list (only `2026-08-13_capital-sizing-knee.md` and the 2026-08-13 max-drawdown-cap review are named safe). Its provenance relative to the #1070 stale-import defect (which corrupted results from worktree-executed commands before the 2026-08-13 19:23 mitigation shim) is **unconfirmed, not cleared**. Before this experiment runs:

1. Check `git log` on the branch/worktree that produced `2026-07-12_exit-geometry-honest.md`'s numbers for evidence of `PYTHONPATH` / import-path discipline consistent with the #1080 fix agent's criteria, OR
2. Skip the check and simply **do not use the old F1/F2/F3 numbers as an anchor** — this preregistration's own control arm is re-run from scratch (§4) specifically so the power follow-up's verdict does not depend on trusting the original study's exact figures, only its qualitative arm definition (`tp_06`'s parameters, which are a fact about the config file, not a computed result, and therefore unaffected by the stale-import defect either way).

This experiment proceeds under option 2 by default: **the July numbers are cited above only as motivation for why `tp_06` is the arm being re-tested, never as data this experiment's verdict depends on.**

## 1. Hypotheses

**H0 (null)**: `tp_06` does not improve total return and profit factor vs. control on a properly powered sample (more folds, more history). The July directional consistency was a 3-fold coincidence.

**H1 (statistical-power hypothesis)**: `tp_06`'s effect is real but small (~1pp return, ~0.1 PF), and the July study's 28-70 trades/fold was underpowered to detect an effect of that size at the Bonferroni bar it correctly applied. A wider fold set (adding pre-2023 half-years, per the July report's own suggested follow-up) with correspondingly more trades will either (a) confirm the effect at significance, or (b) reveal it was noise once more independent draws are added — either way resolving the "promising but not ready" status the July study explicitly could not.

- *Mechanism if true*: `tp_06` captures a genuine, small, consistent edge in how HyperGrowth's realized trade population's price excursions relate to a tight, full-close TP level — not a directional edge (the entries are unchanged), a trade-management one.
- *Falsified if*: the effect does not hold its direction (positive on ≥5 of 6 folds, matching the "all 3 of 3" bar generalized proportionally) once the fold count increases, or a properly powered test still fails to clear Bonferroni significance.

## 2. Metric

**Primary**: total return and profit factor, `tp_06` vs. control, per fold and pooled, with fees/slippage on (`CostCalculator` defaults).

**Secondary**: MaxDD (measured mode), trade count per fold, MFE-capture ratio and MAE-ride fraction (the mechanism diagnostics the July study used, re-computed here with the same unit caveat disclosed in that study — sized fractions, not strictly bounded [0,1], reported as directional indicators).

Trade-count floor: ≥15 per fold (already comfortably cleared in every July fold; checked again here since fold composition changes).

## 3. Success threshold (pre-committed, numeric)

`tp_06` is a **promotion candidate** if:
- Total return improves vs. control on **every fold tested** (the July study's own bar, carried forward — a partial win on "most folds" is explicitly not sufficient, consistent with the multi-regime-robustness standard used throughout this research line), AND
- Profit factor improves vs. control on every fold, AND
- The **pooled** return-delta bootstrap (all folds' per-trade P&L differences combined) clears Bonferroni-corrected significance at alpha = 0.05 / (number of folds + 1 pooled test), two-sided, 10,000 resamples, fixed seed, AND
- MaxDD does not breach the 20% portfolio cap (checked under `--drawdown-cap-mode enforce` for the confirmatory run, §7).

If total return/PF improve on every fold but pooled significance is still not reached: reported as **"promising, still not ready — even a properly powered re-run keeps missing significance,"** a materially different (and more informative) finding than the July study's "not enough data" hedge, and one that would argue for retiring `tp_06` rather than running yet another power follow-up.

## 4. Arms and folds (pre-committed)

- **Control**: HyperGrowth as configured live (`stop_loss_pct` engine default, no override — matching the exit-geometry-honest control definition), re-run from scratch on this worktree.
- **`tp_06`**: `stop_loss_pct=0.10`, `take_profit_pct=0.06`, `max_holding_hours=336` — identical parameters to the July study, applied via the same `_risk_overrides` mechanism.
- **Folds**: F1=2023H1, F2=2024H1, F3=2025H1 (re-run, not reused from July, per §0.1), **plus** F0a=2021H2, F0b=2022H1, F0c=2022H2 — three additional half-years extending back through the 2021-2022 bull-to-bear transition and the Terra/FTX contagion window, per the July report's own suggested follow-up ("adding 2019-2022 half-years, for the statistical power this round's 28-70 trades/fold lacked") and per this program's diversification-for-variance thinking (more independent regime draws, not just more of the same regime). **Data-quality spot-check required first**: the July report found a non-differential ~2.8%-of-bars signal-generator failure rate in Q1 2023; before locking F0a-F0c as valid folds, run the same `MLBasicSignalGenerator` prediction-failure check on the extended window and report the failure rate per fold. If any extended fold shows a materially higher failure rate than the ~2.8% baseline, it is flagged and excluded from the pooled test (not silently included), and the primary verdict falls back to F1-F3 only.
- 2 arms x 6 folds = 12 primary runs + pooled bootstrap + 1 determinism recheck (control/F1, matching the July protocol).

## 5. Data window / protocol

- Symbol/timeframe: ETHUSDT/1h.
- Command: `PYTHONPATH="$(pwd)" atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h --start <fold start> --end <fold end> --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20 --initial-balance 85 --drawdown-cap-mode measure --log-to-db`, plus the `tp_06` override for the treatment arm, long-only enforced (matching GH #1020/current prod), run strictly sequentially. Explicit `--start`/`--end` calendar dates for every fold (never `--days N` relative to run-date) and `PYTHONPATH`-forced worktree execution with identity verification before trusting any number — both per GH #1070's finding that a naive worktree `atb backtest` invocation silently ran 131-commits-stale code and a relative `--days` window landed in an unrelated regime (`2026-08-13_hypergrowth-tier-restore-reproduction.md`).
- Worktree: new, disposable, from `develop` HEAD at preregistration-lock time; commit hash recorded in results.
- Determinism guard: control/F1 run twice, must match to full float precision (post-#923 deterministic inference, per the July protocol).

## 6. Decision each outcome triggers

- **Supported** (clears §3 on the full 6-fold set, or on F1-F3 if extended folds are excluded per the data-quality gate): recommend to pm as a staging-trial candidate for `hyper_growth`'s take-profit config, contingent on #1106 (§10). This would be the first arm in this entire multi-month research line to clear a Bonferroni bar on a money metric — treated with matching scrutiny, not fast-tracked because it's finally a "yes."
- **Rejected** (fails to hold direction across the extended fold set, or fails significance even pooled): full write-up. `tp_06` and the tight-TP/bypass-partial-ladder lever are retired from further follow-up — this was the one thread the July study explicitly left open, and this experiment is designed to close it either way, not to keep re-running until it works.
- **"Still not ready" outcome** (§3): reported as its own category, and this specific lever (tight fixed TP, no further power follow-up) is retired even though the direction never flipped — a small, never-significant effect across 6 folds is a different, weaker claim than across 3, and the honest read is that more data alone will not resolve it.
- **Inconclusive** (extended-fold data quality gate trips materially): report on F1-F3 only per the fallback above; flag the data-quality issue for a separate ticket rather than blocking this experiment's verdict on it.

## 7. Drawdown-truncation discipline (#1102)

Characterization runs use `--drawdown-cap-mode measure`; `early_stopped` checked and reported per fold per arm. If `tp_06` graduates per §6, a confirmatory `--drawdown-cap-mode enforce` run is required (live-representative) before the recommendation goes to risk-officer — `tp_06`'s wider effective stop distance combined with a tight TP changes the trade population enough that its measured-mode MaxDD is not assumed to equal its enforce-mode MaxDD without checking.

## 8. Sensitivity analysis

Only if `tp_06` graduates: test `take_profit_pct` in {0.05, 0.06, 0.07} on the training folds only (F1-F3 already spent as primary per §4 — this sensitivity check uses F0a-F0c if not excluded by the data-quality gate, otherwise is deferred to a fresh, separately preregistered window rather than reusing spent folds). Does the qualitative result survive a +-1pp wiggle around 6%, or is 6% a lucky specific value.

## 9. Risks of false positive

- **Effect size is already known to be small** (~1pp return, ~0.1 PF from the July study) — even a "significant" result here is a small edge in absolute terms at $87-$1,000 capital, and should not be oversold as a fix for HyperGrowth's broader -20.15%/365d honest profile.
- **`tp_06` changes trade *mechanism*, not just a parameter**: because it bypasses the partial-exit ladder entirely (fires before the first 8% partial-exit tier), a "win" here is really "a full-close-only exit policy beats the partial-exit-ladder policy for this trade population" — a more specific and more actionable claim than "tighter TP is better," and should be reported as such rather than generalized to "make TP tighter" broadly (the July study already showed stop-tightening is monotonically harmful, the opposite direction).
- **Extended folds (2021-2022) sit in a different market-structure era** — pre-dates some of the model's assumed regime variety and may include Binance API/listing changes for ETHUSDT specific to that period; spot-checked for gap count/size before being trusted, not silently interpolated over (same discipline as the model-free scan's §9).
- **Pooling folds for the bootstrap assumes each fold's trades are a reasonably independent draw** — they are not fully independent (adjacent folds can share open positions crossing the boundary, and market regimes autocorrelate across fold boundaries per the model-free scan's own regime-persistence finding) — this is disclosed as a residual risk, consistent with how the model-free scan treats its own BH-FDR independence assumption.

## 10. Parity status

Same framing as the sibling documents. `tp_06` changes the exit rule applied to entries generated by the closed-bar backtest; the *relative* control-vs-tp_06 comparison on a fixed entry sequence is parity-insensitive and may run now. Any promotion or staging-trial recommendation is parity-sensitive and waits for #1106 — a different live entry sequence (43.2% divergent at minute 5) could plausibly produce a trade population whose price-excursion distribution relative to a 6% TP differs from what this backtest characterizes, since entry timing changes where in a move a trade starts.

## 11. What this experiment explicitly does not do

- Does not build the MFE-conditioned early-cut policy that GH #971 describes (trailing-stop distance, breakeven threshold, and the partial-exit ladder are still hardcoded in `hyper_growth.py`, not reachable via `RiskParameters`) — that remains a separate, money-path-adjacent `src/` change requiring `architecture-reviewer` + `risk-officer` review, out of scope here.
- Does not modify the signal generator or sizing layer (those are the sibling documents' scope).
- Does not propose a live-affecting change by itself.

---

*Locked 2026-08-25, before any statistic is computed. Results appended below in a dated section per the anti-p-hacking rule.*
