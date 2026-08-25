# Experiment: Does the Deployed Model's Magnitude Signal Generalize as a Volatility Proxy? (Phase 0.5, Vol-Regime Program #0)

**Author**: quant-researcher
**Status**: PREREGISTERED — locked before any statistic is computed. Do not edit thresholds after seeing results; corrections are new dated sections.
**Program**: Vol/regime program (this doc's sibling `agents/research/vol-regime-program.md`).
**Blocked on**: nothing for the characterization question itself (pure signal-scoring, no backtest, no live-affecting change — same class of experiment as its precedent, §0). Any downstream sizing use is gated the same way the sibling sizing documents are (#1106).

---

## 0. Why this experiment, and what triggered it

GH #1067's Phase 1 gate (`docs/research/experiments/2026-08-25_conviction-sizing-phase1.md`, PR #1116) tested whether the deployed ETHUSDT model's confidence/`|predicted_return|` magnitude predicts realized *direction* — replicating GH #912 on a fresh 7-week window (2026-07-06 -> 2026-08-24, n=1,177 bars) against the model actually in production (`2026-07-04_22h_v1`, pinned). **The directional gate did not clear** (Cochran-Armitage p=0.967/0.996/0.967 across three bucket definitions, Spearman p=0.747/0.920/0.747) — a **replicated null** (matches #912's earlier null on a different window/model instance).

**But its secondary statistic is the one this program cares about**: `|predicted_return|` correlates with `|realized_return|` at **Spearman rho = 0.204, p = 1.5e-12** (n=1,177). Verified directly from the source document rather than taken on trust, per the anti-p-hacking rule's "verify relayed numbers against artifacts" instruction. Read plainly: the model has no idea which way the market is going to move, but it carries real, highly significant (not just non-zero) information about **how big** the move is going to be. That is a third independent line of evidence pointing the same direction as this whole program's other two inputs:

1. The model-free scan (#1105): ARCH volatility clustering significant in 20/20 cells, no directional edge clears cost in 20/20.
2. Six prior ETHUSDT/1h model tournaments (July synthesis): ~51-53% DA ceiling under every lever tried.
3. **This finding**: the deployed model, trained on a directional target, incidentally encodes a real second-moment (volatility-magnitude) signal even though its first-moment (direction) signal is noise.

**This is a candidate input, not an assumption.** rho=0.204 explains roughly 4% of variance (rho^2) — real and far from a coincidence at p=1.5e-12, but modest in effect size, and the p-value must not be read as if it were the effect size itself. This experiment does not assume the finding transfers beyond the exact model/symbol/window it was measured on; it tests that directly, which is what the coordinator's brief asked for.

## 1. Hypotheses

**H0 (null)**: the magnitude-vs-magnitude correlation is specific to the ETHUSDT `2026-07-04_22h_v1` model instance and its 7-week measurement window — it does not replicate on a different symbol's model, and/or does not hold up on a longer/different ETHUSDT window.

**H1 (generalization hypothesis)**: the effect reflects something structural about how a next-bar price-regression model's output magnitude relates to realized volatility (a plausible, disclosed mechanism below) rather than a coincidence of this one model/window, and holds — at comparable or better effect size — on (a) a longer/different ETHUSDT window using the same model, and (b) BTCUSDT's own trained `basic` model (`src/ml/models/BTCUSDT/basic/`, already exists, no new training required).

- *Mechanism if true*: a price-regression model trained to predict next-bar return will, even without any explicit magnitude-calibration objective, tend to output larger-magnitude predictions when its own input features (recent volatility, range, momentum) look more "extreme" — and those same extreme-feature conditions are mechanically associated with larger realized moves (this is close to tautological at the feature level, which is exactly why it is worth checking whether it survives out of sample rather than assuming it): if the correlation holds on fresh data and a different symbol's independently-trained model, that is evidence the model is doing something more than memorizing training-window associations.
- *Falsified if*: the correlation does not replicate (loses significance, or reverses sign, or the effect size collapses to a small fraction of 0.204) on BTCUSDT's model and/or a longer ETHUSDT window.

## 2. Metric

**Primary**: Spearman rank correlation of `|predicted_return|` vs `|realized_return|`, with exact p-value, on each of the two replication targets in §4, using the identical scoring methodology as `2026-08-25_conviction-sizing-phase1.md` (reuse `MLBasicSignalGenerator.generate_signal()` directly — no hand-rolled inference, no backtest engine, no `RiskManager`, no `CostCalculator` — this is a pure signal-quality question, fees/slippage are not applicable here, same as its precedent).

**Secondary**: effect-size stability check — bootstrap 95% CI on each replication's Spearman rho (2,000 resamples, fixed seed), so "does it generalize" is answered with a CI comparison, not a bare point estimate.

Trade-count floor is not applicable (this is a signal-scoring study, not a backtest); the sample-size floor instead is **n >= 500 scored bars** per replication target, matching the order of magnitude of the original 1,177-bar study closely enough to trust a comparable CI width — reported and checked, not assumed.

## 3. Success threshold (pre-committed, numeric)

The magnitude-signal is **"generalizes"** if, on **both** replication targets (§4):
- Spearman rho >= 0.10 (roughly half the original 0.204 — a deliberately generous floor, since even a smaller-but-real effect is still useful for sizing purposes, and this experiment is explicitly not requiring the effect to be as strong elsewhere as it was on the exact model/window it was discovered on), AND
- p < 0.01 (a stricter bar than the discovery study's own threshold, since this is now a confirmatory replication, not an exploratory finding — Bonferroni-style caution given the effect was found in the same broader research effort that is now testing it).

**"Partially generalizes"**: clears the bar on one target but not the other — reported precisely, with a stated hypothesis for why (e.g., BTCUSDT's model was trained under a different protocol/window than ETHUSDT's, which is a real, disclosed confound, not glossed over).

**"Does not generalize"**: fails both. This is reported as: the original finding stands on its own model/window (the p=1.5e-12 result is not retracted — it was real on that data), but is **not** treated as a general property of "ML models trained on this target," and the sequencing implication in §6 (skip building a new ATR-based signal, reuse the model output instead) does not apply — the sibling `2026-08-25_volatility-target-sizing-prereg.md` experiment proceeds on its already-built `EnhancedRegimeDetector`/ATR-percentile path unchanged.

## 4. Replication targets (pre-committed, no post-hoc target added)

- **Target A (window extension, same model)**: the exact `2026-07-04_22h_v1` ETHUSDT model, scored on a window immediately preceding the original study's (to avoid re-scoring the same bars): `2026-04-01 -> 2026-07-04` (the model's own training cutoff — deliberately in-sample-adjacent but not overlapping the original OOS window, disclosed as such; a genuinely fresh forward window beyond 2026-08-24 is preferred if enough new data has accumulated by run time and is used instead if available, recorded explicitly which window was actually used).
- **Target B (symbol extension, different model)**: BTCUSDT's own `latest`-pointed `basic` model, pinned by exact `model_version` (not `latest` resolution, per GH #988's point-in-time-pinning precedent), scored on the same relative OOS window shape as the original study (most recent ~7-8 weeks after that model's own training cutoff, recorded exactly).

Both targets reuse the discovery study's script (`docs/research/experiments/scripts/2026-08-25_confidence_bucket_oos_check.py`) with only the model-version pin and date range changed — no new methodology, so a difference in result is attributable to the model/window, not to a different measurement approach.

## 5. Data window / protocol

- No backtest engine involved; this is direct signal scoring against cached OHLCV.
- Data: `atb data cache-manager info` checked for staleness on both ETHUSDT and BTCUSDT 1h before running; `atb data prefill-cache` if needed, noted explicitly.
- Worktree import-path guard (GH #1070): `PYTHONPATH="$(pwd)" python docs/research/experiments/scripts/2026-08-25_confidence_bucket_oos_check.py ...`, worktree identity verified before trusting any number (same method as the discovery study: `python -P -c "import src; print(src.__file__)"` resolves under this session's worktree).
- No look-ahead: inherited directly from the discovery study's verified method (`current_price = df["close"].iloc[i]`, realized return compares `close[i+1]` vs `close[i]`) — re-verified by reading the script before running, not assumed correct because it's a reused script.
- This is a **single, short, sequential** run per target (no bootstrap loop heavy enough to need the thermal-discipline callout beyond the program's standing rule of never running two of this program's experiments concurrently).

## 6. Decision each outcome triggers

- **Generalizes**: this becomes a real, cheap, already-in-production candidate volatility signal. Sequencing implication for the sibling `2026-08-25_volatility-target-sizing-prereg.md` experiment: that experiment's `VolatilityTargetSizer` currently consumes `regime.metadata["atr_percentile"]` (a realized-volatility measure); a **follow-up** preregistration (not bundled here, per the no-post-hoc-arm rule) would test whether swapping in or blending `|predicted_return|` as an additional sizing input improves on ATR-percentile alone — named as the next natural step, not run in this document.
- **Partially generalizes**: usable for ETHUSDT specifically (if Target A clears but B does not), flagged to ml-engineer that BTCUSDT's model may need its own magnitude-calibration check before being trusted the same way, no broader claim made.
- **Does not generalize**: the finding is reported as real-but-narrow (true for the exact model/window it was found on, not a general property), full write-up, and the sibling vol-sizing experiment proceeds unchanged on ATR-percentile.
- **Inconclusive** (e.g., BTCUSDT's `latest` model turns out stale, mismatched training window, or the extended ETHUSDT window has data-quality gaps): flagged per-target, re-scoped rather than forcing a verdict.

## 7. Risks of false positive

- **Effect size framing**: rho=0.204 (~4% of variance) is being treated as "real and worth checking further," never as "a strong signal" — this document repeats that framing at every decision point specifically so a later reader skimming only the verdict doesn't overweight it.
- **Single-model, single-symbol origin** (n=1,177, ~7 weeks, one continuous regime segment) — the coordinator's own framing. This experiment is the direct answer to that caveat, not a way around it; if it fails to generalize, that is reported as fully as a positive result would be.
- **BTCUSDT model protocol difference**: whatever training protocol produced BTCUSDT's `latest` model is not audited here beyond recording its `metadata.json` (training window, architecture, target) — a difference in outcome between Target A and Target B could be attributable to a protocol difference rather than a symbol difference, and this is disclosed rather than collapsed into a single "generalizes/doesn't" headline.
- **Multiple-target correction**: two targets tested, both must independently clear §3's bar (not pooled) — this is a stricter standard than pooling would be, deliberately, since a pooled test could let one strong target compensate for one weak one and obscure exactly the "does it generalize" question being asked.
- **This does not retroactively validate #1067's directional gate.** A generalizing magnitude signal says nothing about direction — the replicated null on direction (per `2026-08-25_conviction-sizing-phase1.md`) stands regardless of this experiment's outcome, and this document does not reopen that question.

## 8. What this experiment explicitly does not do

- Does not train any new model. BTCUSDT's model already exists; no other symbol is tested here specifically because no other symbol in the #1105 grid has a trained model, and training one for LINKUSDT/SOLUSDT/DOGEUSDT to extend this check further is a materially larger ask (a real model-training project, not a cheap scoring script) — named explicitly as a disclosed scope boundary and a candidate for a separate, larger pm/ml-engineer decision if Target A and B both generalize, not silently expanded into here.
- Does not build a new sizer or sizing input. Any use of this signal in `VolatilityTargetSizer` or elsewhere is a distinct, deferred follow-up (§6).
- Does not touch `src/` production code, `RiskParameters`, or any live/paper trading path.
- Does not propose a live-affecting change.

---

*Locked 2026-08-25, before any statistic is computed. Results appended below in a dated section per the anti-p-hacking rule.*
