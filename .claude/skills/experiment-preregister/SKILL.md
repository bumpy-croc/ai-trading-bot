---
name: experiment-preregister
description: Pre-register every experiment BEFORE it runs — hypothesis, exam window, metrics, success thresholds, and the decision each outcome triggers, written to docs/research/experiments/ first. Use before ANY backtest study, tournament, calibration test, or parameter sweep; also the reference for anti-p-hacking rules (no post-hoc thresholds, candidate counting, negative results get full write-ups).
---

# Experiment Pre-registration

No experiment runs before its file exists. The un-preregistered path produced the fabricated
kelly "+16.67%" (real: +0.02%) that nearly earned a live swap, and the in-sample-contaminated
model validation (−1.31% leaked vs −7.43% clean). The preregistered path (window tournament,
calibration study) produced verdicts nobody could argue with — including negative ones. Files
are layer-2 record: append-only, corrections as new sections
(`docs/architecture/memory_system.md`).

## Before the first run: write `docs/research/experiments/YYYY-MM-DD_<name>.md`

Required sections (the template is `2026-07-05_window-tournament.md` — copy its shape):

1. **Hypothesis** — falsifiable H1 AND the competing H0, each with a mechanism ("shorter
   windows win because market structure shifted"), plus explicit "falsified if / supported if"
   conditions (the flip-rate study's are the model).
2. **Metric** — ONE primary (money metrics on the shared exam: OOS return / PF / MTM MaxDD /
   win rate, prod-matched flags), secondaries labeled as secondaries. Holdout RMSE is a health
   check, never the decision metric (it ranked the window tournament exactly backwards).
3. **Success threshold** — numeric, pre-committed, including minimum trade count (≥15; a high
   return on a handful of trades is not a result) and the statistical bar (n≈4,400 hourly bars
   → directional-accuracy SE ≈0.75pp; differences under ~1.5pp are noise; Wilson CIs + trend
   test for gradient claims).
4. **Decision each outcome triggers** — pre-commit ALL branches: "clears threshold → staging
   paper trial proposal; fails → keep incumbent, close issue; inconclusive → what specifically
   gets run next." An outcome without a pre-committed decision invites motivated re-analysis.
5. **Risks of false positive** — single-window/single-regime draw, trade-count risk, leakage
   vectors, cache/provider drift. Naming them up front is what let the window tournament call
   its own winner "promising but not ready."
6. **Protocol** — data windows with hard cutoffs, exam window strictly after every training
   cutoff, engine version (post-#838 only), worktree isolation, compute plan (sequential heavy
   jobs). For model comparisons the protocol IS `model-tournament` — reference it, don't fork it.

Then register: GH issue (`type:experiment`, owner label), and run.

## Anti-p-hacking rules (each traceable to a real save)

- **No post-hoc threshold moves.** The calibration study's Phase-3 gates were pre-selected from
  training-period data and held even when the vol-normalized variant came out "directionally
  favorable but sub-threshold" (+1.45pp vs required ≥3pp) — it was reported as noise, not
  shipped. If a threshold turns out wrong, write a NEW preregistration; never edit the old one
  after seeing results.
- **Candidate-count tracking per exam window.** Every candidate that faces a frozen exam
  increments its count; rotate/retire the window after ~10 (multiple-comparisons discipline,
  `model-tournament` rule 10, `docs/architecture/model_evaluation_system.md`).
- **Negative results get FULL write-ups.** The exit-geometry sweep (every variant strictly worse;
  NO-GO), tournament-v2 (all "winners" collapse to ~0%), and the calibration study (H0
  supported) are among the most-cited files in the record — they killed whole categories of
  wasted work. "Didn't work" without a write-up = the experiment will be re-run by someone else.
- **Determinism guard before trusting any result.** Re-run the first exam twice; identical or
  stop (the 0.1s inference-timeout bug produced 46 trades/−11.36% vs 55/−10.33% on IDENTICAL
  runs and threatened every frozen-exam comparison — #913).
- **Verify relayed numbers against artifacts.** Read the raw backtest JSONs / metadata.json
  yourself, "not taken from any relayed summary at face value" (window-tournament practice;
  a relayed "zero callers" claim once grepped the wrong class).

## After the run

Append (never rewrite) to the same file: results tables, verdict vs the pre-registered
thresholds, explicit answer to the hypothesis. Update status in the header line. Log a
track-record entry to log.md (the `2026-07-05 11:00 · track-record · quant-researcher` entry is
the reference shape: hypothesis → verdict → evidence path → recommendation). Scoreboard row if
model-related. Experiment verdicts that change what we do are material decisions —
`decision-record`.

## Red flags

- "Let's just run it quickly first to see if it's worth preregistering." That IS the p-hack.
- A success threshold that appears for the first time in the results section.
- An experiment whose file was created after its result timestamps.
- Reusing an exam window past its candidate budget because "it's convenient."
