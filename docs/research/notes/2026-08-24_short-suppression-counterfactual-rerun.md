# Short-suppression counterfactual — provenance-verified rerun (GH #1081 Step 2, item 1)

**Date**: 2026-08-24
**Researcher**: quant-researcher
**Status**: COMPLETE. Reproduces the original study and [D-2026-08-13-06] essentially exactly (once a
second, previously-undocumented engine-default confound is controlled for). Verdict: **genuinely
ambiguous across windows** at the decision-relevant (aggregate arm return) level, though the
underlying mechanism the long-only decision actually rests on (short trades are unprofitable
standalone) reproduces robustly in all four windows tested.
**Worktree**: `.claude/worktrees/short-suppression-rerun-1081`, branch
`claude/short-suppression-rerun-1081`, off `origin/develop @ a94dc8ae` (`.agent-active` present).
**Supersedes for provenance purposes**: `docs/research/notes/2026-07-12_short-suppression-counterfactual.md`
(#1019, rated AT-RISK by the #1081 Step-1 provenance audit) and, for the 12-month-window comparison,
`docs/research/experiments/2026-08-13_hypergrowth-tier-restore-reproduction.md` ([D-2026-08-13-06]).
**Related**: GH #1081 (parent), #1020 (long-only decision), #990 (original mechanism finding), #988/#1006
(model-version pinning), #1070/#1080 (stale-import defect + fix), #1073/#986/#1088 (RiskParameters
ratified hydration + position-cap default).

## Why this run

Proposal 2026-07-12-01 (long-only HyperGrowth/ETHUSDT) has been **live in production since
2026-08-13** (`fba64281`). Its sole evidence base, the short-suppression counterfactual, was rated
AT-RISK by the #1081 provenance audit: it ran through `atb backtest`/an in-process wrapper from a
worktree with no `PYTHONPATH` guard, on a tree that depended on `--model-version` pinning that
would not exist if misdirected to the stale primary checkout. Independently, a different
12-month window ([D-2026-08-13-06]) had already contradicted the original's conclusion (shorts
looked like a hedge there). This run re-executes both studies' arms from a clean, provenance-locked
worktree to settle whether the live config's evidence actually holds.

## Method

- One Python process per script invocation (`run_counterfactual_rerun.py`,
  `run_uncapped_supplement.py`), all 13 backtests run strictly sequentially, single process, no
  parallelism (standing Mac-thermal constraint).
- Strategy constructed in-process via `create_hyper_growth_strategy(symbol=..., model_version=...,
  allow_shorts=...)` — the same construction path `cli.commands.backtest._load_strategy` uses,
  diverging only to thread `allow_shorts` explicitly (no CLI flag exists for it; this mirrors both
  source documents' own methodology of an in-process bypass of the #1020 long-only lock for
  research purposes).
- Model pinned to `2026-07-04_22h_v1` throughout (the only version ever registered for
  ETHUSDT/basic — confirmed unchanged since both source studies; the 2026-08-09 and 2026-08-23
  retrains both retained this incumbent).
- Fees/slippage on throughout (`CostCalculator` defaults, never disabled). `enable_engine_risk_exits=True`
  (matches live). No sentiment.
- Fresh cache prefilled for ETHUSDT/1h, 2022–2026-08-24, inside this worktree (not shared with the
  primary checkout, to avoid any write path back into it under the #1087 write-guard).

### Provenance banner (printed at the top of every run's output; identical across both scripts and
### all 13 backtests — verified once per script invocation, not asserted)

```
======================================================================
PROVENANCE BANNER
  src.__file__ resolved root : /Users/alex/Sites/ai-trading-bot/.claude/worktrees/short-suppression-rerun-1081
  git rev-parse HEAD          : a94dc8aea26ef43af124fef485ae1c749f80decc
  git branch                  : claude/short-suppression-rerun-1081
  effective PYTHONPATH        : <unset>
  python executable           : /Users/alex/Sites/ai-trading-bot/.venv/bin/python
======================================================================
```

Verified with `python -P -c "import src; print(src.__file__)"` **and** plain `python -c ...` before
any run — both correctly resolve to this worktree (#1080's `verify_source_root()` hard guard is
live on `develop @ a94dc8ae` and would have raised `SourceRootMismatchError` on a mismatch, not
just warned). This is a materially different situation from the original study (`origin/develop @
2f0dff5c`, 2026-07-12) and the tier-restore study (`develop @ 9d900992`, 2026-08-13), both of which
predate the robust fix and had no such runtime assertion available.

### Controlling for #1088 (bare-`Backtester` position-cap default 0.10→0.20)

Every arm below either (a) uses HyperGrowth's own strategy-declared `max_fraction` override
(`min(base_fraction * max_leverage, 0.50)` = **0.25** with default params — resolved via
`resolve_strategy_max_position_size`, the same seam the CLI uses, and unaffected by #1088's bare-
`Backtester` default since it is an explicit strategy override, not the harness's fallback), for
the Segment B and F1/F2/F3 runs, matching the original counterfactual's own methodology exactly
(it explicitly checked and used this 0.25 cap, Sec. 7 of that doc); or (b) passes
`max_position_size=0.20` explicitly for the D-2026-08-13-06 window runs, matching the tier-restore
doc's explicit `--max-position-size 0.20` CLI flag. **Neither path touches the bare-`Backtester`
default #1088 changed**, so #1088 does not confound any number in this document. Verified per-run
via the `effective_max_position_size` field in each result (0.25 for Segment B/folds, 0.20 for the
D-0813-06 window, in every case).

### A second, more consequential confound found and controlled for (not #1088)

`Backtester.__init__` sets `self._early_stop_max_drawdown = risk_manager.params.max_drawdown` when
`risk_parameters is not None` (true for every run here). Neither source study passed
`--max-drawdown` explicitly, so both relied on `RiskParameters()`'s default. **PR #1073/#986**
("hydrate RiskParameters from ratified risk-limits.json"), merged **after both source studies ran**
(tier-restore: `9d900992`, 2026-08-13; short-suppression original: `2f0dff5c`, 2026-07-12) but
**before** this rerun, changed that default to hydrate from the ratified 20% cap. The engine halts
new entries once running drawdown exceeds this threshold (`engine.py:1276`) — the backtest analogue
of live's now-shipped max-drawdown hard-halt (#848/#849, also merged after both source studies).

Net effect: **any arm whose true (uncapped) drawdown would exceed 20% gets silently truncated**
under current defaults relative to the pre-#1073 originals — fewer trades, a different (smaller)
loss, because the run stops before the losses that would have followed. This first appeared as an
unexplained divergence: the shorts-enabled Segment-B/F1/F2/F3-long-only arms (whose true drawdown
never approaches 20%) reproduced the originals to 3–4 significant figures on the first pass, but
F3-shorts-enabled (true DD 22.1%) and the D-0813-06 long-only baseline (true DD 31.3%) did not —
both are exactly the arms whose uncapped drawdown crosses 20%. Confirmed by rerunning those three
arms with an explicit `max_drawdown=1.0` override (disabling the early-stop, i.e. the source
studies' effective configuration): all three then reproduce their originals essentially exactly
(see tables below). **Both the capped (current-default) and uncapped (apples-to-apples-with-the-
originals) numbers are reported below** — this is not a defect in the rerun, but a real, dated
behavior change in the harness that must be surfaced, not silently absorbed.

## Results

### 1. Segment B — live-matched (2026-07-05 → 2026-07-12, $84.40 initial balance)

| Arm | Trades (S/L) | Return | MaxDD | Original (07-12) | Match |
|---|---:|---:|---:|---|---|
| shorts-enabled | 0 (0/0) | −0.27% | 0.70% | −0.27% / 0.70% | **exact** |
| long-only | 1 (0/1) | +0.11% | 0.48% | +0.11% / 0.48% | **exact** |

Degenerate window, as pre-committed in the original (not used to drive the verdict either way).

### 2. Supplementary folds F1/F2/F3 (in-sample relative to model training cutoff — original's own caveat carries forward unchanged)

| Fold | Arm | Trades (S/L) | Return | PF | MaxDD | Win rate | Short Σpnl% | Original | Match |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| F1 2023H1 | shorts-enabled | 29 (19/10) | −3.21% | 0.727 | 6.59% | 72.4% | −0.037 | 29(19/10) −3.21%/PF.727/6.59% | **exact** |
| F1 2023H1 | long-only | 23 (0/23) | −3.99% | 0.584 | 7.12% | 69.6% | — | 23(0/23) −3.99%/PF.584/7.12% | **exact** |
| F2 2024H1 | shorts-enabled | 40 (28/12) | −12.59% | 0.409 | 13.75% | 65.0% | −0.106 | 40(28/12) −12.59%/PF.409/13.75% | **exact** |
| F2 2024H1 | long-only | 50 (0/50) | −8.92% | 0.558 | 12.06% | 72.0% | — | 50(0/50) −8.92%/PF.558/12.06% | **exact** |
| F3 2025H1 | shorts-enabled (**uncapped**, apples-to-apples) | 67 (29/38) | −19.94% | 0.359 | 22.10% | 61.2% | −0.078 | 67(29/38) −19.94%/PF.359/22.10% | **exact** |
| F3 2025H1 | shorts-enabled (current-default, capped at 20%) | 48 (21/27) | −21.47% | 0.242 | 21.85% | 54.2% | −0.089 | n/a — truncated by #1073's now-default early-stop | flagged, not comparable |
| F3 2025H1 | long-only | 62 (0/62) | −18.79% | 0.355 | 20.31% | 64.5% | — | 62(0/62) −18.79%/PF.355/20.31% | **exact** |

**Delta (shorts-enabled − long-only), the original study's own pre-committed metric (Sec. 3), using the apples-to-apples uncapped F3 number**:

| Fold | Δ return (pp) | Original Δ | Match | Clears ±2pp bar? |
|---|---:|---:|---|---|
| F1 | +0.78 | +0.78 | **exact** | no |
| F2 | −3.67 | −3.67 | **exact** | **yes**, long-only wins |
| F3 | −1.15 | −1.15 | **exact** | no |

Short-side standalone P&L: negative in all three folds (−0.037, −0.106, −0.078), matching the
original's "negative in every fold, not driven by an outlier" finding exactly.

**This is a clean, essentially byte-for-byte reproduction of the original study's fold-level
results and its own pre-registered delta metric.** The AT-RISK provenance rating from the #1081
audit was the correct call to make *before* re-running — the risk was real and the model-version
pinning it worried about was in fact load-bearing — but the underlying numbers turn out to have
been correct despite the unverified provenance. Confirmation, not correction.

### 3. The [D-2026-08-13-06] window (2025-07-04 → 2026-07-04, most recent 12 months, $85 initial balance, explicit 0.20 cap)

**Uncapped (apples-to-apples with the tier-restore doc's own methodology — no `--max-drawdown` flag, matching what that doc actually ran under, pre-#1073):**

| Arm | Trades (S/L) | Return | MaxDD | Win rate | Short Σpnl% | Tier-restore doc (08-13) | Match |
|---|---:|---:|---:|---:|---:|---|---|
| long-only | 89 (0/89) | −28.29% | 31.27% | 62.9% | — | −28.29% / 31.27%, 89 trades | **exact** |
| shorts-enabled | 116 (64/52) | −17.75% | 18.99% | 74.1% | −0.031 | −17.75% / 18.99%, 116 trades, 74.14% WR | **exact** |

Delta: shorts-enabled beats long-only by **+10.54pp return** and **−12.28pp MaxDD** — this
reproduces [D-2026-08-13-06]'s "shorts acted as a hedge" finding **exactly**.

**Current-default (capped at the ratified 20% max-drawdown, matching current `develop` and live's
actual #849 hard-halt behavior — not what either source doc ran, but what a fresh backtest today
actually reports without an explicit override):**

| Arm | Trades (S/L) | Return | MaxDD | Note |
|---|---:|---:|---:|---|
| long-only | 57 (0/57) | −18.19% | 21.14% | early-stopped once running DD crossed 20% |
| shorts-enabled | 116 (64/52) | −17.75% | 18.99% | never crosses 20%, not stopped |

Delta under current defaults: **+0.44pp return, −2.15pp MaxDD** — direction unchanged (shorts
still marginally ahead) but the magnitude **collapses by over 90%** once both arms are measured
under the risk control that's actually live today. The specific number [D-2026-08-13-06] cited
(+10.54pp) was correct *for the config it ran under*, but that config no longer matches what
either the backtest harness or live production actually enforces by default.

**Critical nuance, same mechanism the original study already flagged for F1**: even in this window,
where the shorts-enabled *arm* wins on aggregate return, the short trades **themselves lose money
standalone** (Σpnl% = −3.1%, negative — the fourth window in a row where this holds, 4/4). The
arm-level win comes from how allowing shorts changes the *timing and sizing of subsequent long
trades* through the shared balance/compounding path (fewer capital-lockups, different entries
taken) — not from the shorts being profitable. This is the exact mechanism the original study
identified for F1 ("F1's aggregate win for shorts-enabled came from how removing shorts changed the
long trades taken... not from the shorts themselves being profitable") and it generalizes to the
contested window too.

## Applying the original study's own pre-registered decision rule (Sec. 3) to all four windows

Treating the D-2026-08-13-06 window as a fourth data point alongside F1/F2/F3, using the uncapped
(apples-to-apples) numbers throughout:

| Window | Δ return (shorts − long) | Clears ±2pp? | Short-side Σpnl% | Direction favored |
|---|---:|---|---:|---|
| F1 2023H1 | +0.78pp | no | negative | shorts (weak) |
| F2 2024H1 | −3.67pp | **yes** | negative | long-only (clear) |
| F3 2025H1 | −1.15pp | no | negative | long-only (weak) |
| D-0813-06 (2025-07→2026-07) | +10.54pp | **yes** | negative | shorts (clear) |

**Two windows now clear the ±2pp bar in each direction** (F2 for long-only, D-0813-06 for shorts).
Under the original's own pre-committed rule, "costing returns" (shorts help) requires ≥2pp in a
**majority** of windows with short trades AND short-side P&L positive in those windows — not met
(short P&L is negative in literally every window tested, 4/4). "Saving returns" (long-only helps)
requires the mirror — also not met on the strict majority clause (2 of 4 windows, not a majority).
The rule's own fallback, "inconclusive" — "the sign of the per-window delta flips with no
majority" — is now the accurate label, more clearly than with 3 folds alone.

## Verdict

**Genuinely ambiguous across windows**, at the level the live decision actually operates on
(aggregate arm-level return comparison). This is not a hedge — it is the honest reading of a
50/50 split across four independent windows with no majority in either direction, exactly the
outcome the dispatching PM flagged as the most likely one going in.

What is **not** ambiguous, and reproduces with the highest confidence in this entire exercise:
**short trades taken by HyperGrowth/ETHUSDT under this model are unprofitable standalone in every
one of the four windows tested** (F1 −3.7%, F2 −10.6%, F3 −7.8%, D-0813-06 −3.1%, summed
`pnl_percent`, no outlier-driven sign in any fold). That specific claim — the one the original
counterfactual actually used to justify long-only, separate from the noisier "does the aggregate
arm do better" question — holds up completely under re-verification, including in the window that
was cited as contradicting it.

Reconciling with [D-2026-08-13-06]: that finding was correct on its own terms (reproduces exactly,
uncapped) but was, in retrospect, one favorable window among four, not an independent contradiction
of a settled result — and even in that window, the mechanism was reallocation-driven, not short-
side profitability. It should not have been read as strong evidence against the long-only decision,
and with the benefit of a fourth-window, apples-to-apples comparison, it looks more like sampling
variance in a signal that is barely above chance (per the original study's own adversarial
self-review, point 2: ~51–53% directional accuracy) than a real hedge effect.

### Which of the three outcomes this is

- **Not** "original conclusion holds" cleanly — the aggregate-return case for long-only was never
  as strong as 3 folds made it look, and a 4th window shows the opposite sign at a magnitude that
  clears the same bar.
- **Not** "original conclusion overturned" — nothing here shows shorts-enabled is *better*; the
  short-side-unprofitable mechanism reproduces in all four windows without exception.
- **Is** "genuinely ambiguous across windows" at the aggregate level, while the underlying causal
  mechanism (shorts lose money standalone) is robustly confirmed. Long-only was adopted on thinner
  evidence than the original 3-fold study implied — a real finding for the Board — but nothing here
  provides a basis to reverse it; if anything, the mechanism-level result is now confirmed across
  one more independent window than before.

## Recommendation

- **Do not revert long-only on the strength of this rerun.** The mechanism the decision actually
  rests on held up under re-verification in a fourth window, including the one previously thought
  to contradict it.
- **Do flag the ambiguity to the Board explicitly** — this closes #1020's evidence-provenance loop
  with a weaker, more honest confidence level than the original 3-fold study implied, not a clean
  confirmation. The original proposal's own risk-officer review already rated the evidence
  "adequate for a reversible config codification, not for anything irreversible" (2026-07-12) —
  this rerun does not change that calibration, it just re-grounds it in verified numbers.
- **[D-2026-08-13-06] should be annotated**, not withdrawn: its numbers are correct (reproduced
  exactly under its own methodology) but its framing ("shorts acted as a hedge") overstates what a
  single additional window can show, and significantly overstates the practical magnitude once
  today's actual 20%-drawdown hard-halt (current backtest default post-#1073, live post-#849) is
  applied to both arms — the effect shrinks from +10.54pp to +0.44pp.
- **Re-enable criteria from the original proposal remain the right test going forward**: model
  retrain/redesign, a bear-regime fold showing long-only materially worse, or short-side OOS DA
  beating long-side DA in ≥2 folds. None of those triggers has fired here.
- **Separately, flag #1073's RiskParameters ratified-hydration change as a live confound for any
  future backtest re-run against a pre-#1073 baseline** — this is exactly the kind of
  citation-chain risk #1081 exists to catch, and it is not specific to this study. Any other
  #1081 Step-2 item whose original run predates #1073 (2026-08-2x) and whose true drawdown
  approaches or exceeds 20% should expect the same truncation and control for it the same way.

## Reproducibility

- Backtest runners: `run_counterfactual_rerun.py` (10 arms: Segment B ×2, F1/F2/F3 ×2, D-0813-06
  window ×2 capped) and `run_uncapped_supplement.py` (3 arms: F3-shorts-enabled, D-0813-06 ×2,
  all with `max_drawdown=1.0` to disable the early-stop) — both session-scratchpad artifacts in
  this worktree, not committed to `src/`, mirroring the original study's own Sec. 10 convention.
- All 13 backtest runs executed sequentially in two single-process script invocations (no
  parallel runs), per standing Mac-thermal guidance.
- Fees/slippage: on throughout, `CostCalculator` defaults, never disabled.
- Cache: fresh ETHUSDT/1h prefill (2022–2026-08-24) inside this worktree, independent of the
  primary checkout's cache.
- Full raw JSON results: `rerun_results.json`, `rerun_results_uncapped.json` (this worktree,
  not committed — figures transcribed into the tables above).
