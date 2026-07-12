---
id: 2026-07-12-01-hypergrowth-ethusdt-long-only
from: quant-researcher
to: pm
status: open
risk_review_required: true
risk_verdict: null         # null | approve | approve-with-conditions | reject
code_review_required: true
board_required: true       # any change affecting live capital / strategy activation needs human approval per charter.md autonomy envelope
created: 2026-07-12T18:45:00Z
updated: 2026-07-12T18:45:00Z
---

## Ask

Formalize "long-only" as an explicit, intentional configuration choice for HyperGrowth/ETHUSDT, replacing the current state where near-total short suppression is an accidental byproduct of an unrelated margin-safety guard (`execution_engine.py:663-706`, GH #990). **This is NOT a request to loosen or otherwise change that guard** — its margin-safety rationale is independent of this proposal and remains risk-officer's call.

## Context

GH #990 found live HyperGrowth/ETHUSDT executed 9 LONG vs 3 SHORT trades despite a ~50/50 long/short signal split, traced to a SHORT-side inventory guard that fail-closed rejects shorts whenever free ETH exceeds a $1 dust threshold. The open question was whether this accidental suppression costs or saves returns. The counterfactual backtest in `docs/research/notes/2026-07-12_short-suppression-counterfactual.md` found: shorts-enabled beats long-only in only 1 of 3 out-of-sample-relative-to-training folds tested (F1 2023H1), and short trades' standalone P&L is negative in every fold tested (F1, F2, F3) with no single outlier driving that sign. The evidence leans toward suppression being neutral-to-mildly-beneficial rather than costly, but is not strong enough to be called conclusive (see note Sec. 7 for the full, honest accounting, including the one fold that disagrees).

## Proposed change

Add an explicit `allow_shorts: bool = True` (or equivalent) configuration point to `create_hyper_growth_strategy` / `MLBasicSignalGenerator` construction for the ETHUSDT HyperGrowth deployment specifically, defaulting to **whatever risk-officer and pm decide after review** — this proposal recommends defaulting it to `False` for ETHUSDT once approved, i.e., codifying the strategy as long-only for this symbol. No change to `execution_engine.py`'s margin guard. Exact implementation left to the assigned engineer/reviewer; the point of this proposal is the *decision* (should HyperGrowth/ETHUSDT be long-only by design), not a specific diff.

## Evidence

- `docs/research/notes/2026-07-12_short-suppression-counterfactual.md` — the counterfactual backtest (this note), pre-registered thresholds, 3 out-of-sample-relative-to-training folds (F1/F2/F3 2023H1/2024H1/2025H1) plus a degenerate live-matched segment (2026-07-05→2026-07-12).
- `docs/research/notes/2026-07-12_parity-gap-investigation.md` (PR #987) — the original mechanism finding (Finding 2) that motivated this investigation.
- `docs/research/2026-07-12_returns-levers-synthesis.md` — context that HyperGrowth's underlying model has only a ~51–53% directional-accuracy ceiling on ETHUSDT/1h under every lever tried so far; every fold in every cited study, including this one's, shows profit factor below 1.0 in absolute terms (this proposal is about the *relative* long-vs-short question, not a claim that HyperGrowth is profitable).

## How this could lose money

1. **Regime-drift confound.** 2023–2025 was broadly bull-biased for ETH; shorting a rising asset loses money on average regardless of signal quality. The window tournament (#898) already found HyperGrowth (both sides enabled) net-negative over a 185-day *bear* market. A hard-coded long-only config could underperform, not outperform, in a genuine sustained bear regime — the opposite of the intended effect. Mitigation: this is exactly what risk-officer should stress-test (see "call-out" below) before any live change.
2. **The signal being coded around is barely above noise (~51–53% DA).** A retrain, target redesign, or symbol change could flip which side looks "worse" without any real structural change — locking in long-only risks fossilizing a pattern close to statistical noise. Mitigation: revisit this decision if/when the underlying model changes materially (new architecture, retrain, or target redesign already in progress per the returns-levers synthesis).
3. **Sample size.** 3 folds, 19–29 short trades each; one fold (F1) disagrees with the other two. This is a real, disclosed limitation, not explained away — the proposal explicitly does not claim statistical significance.
4. **Opportunity cost is invisible in a backtest-only study.** If ETH enters a genuine, extended downtrend after this ships, a long-only HyperGrowth simply sits out any short-side profit opportunity entirely (not just "suppressed like today," but *intentionally* forever) — a stronger commitment than the current accidental, margin-state-dependent suppression, which at least occasionally lets a short through (3 in the live sample).

## Call-out for risk-officer

- Stress-test long-only HyperGrowth/ETHUSDT specifically against a simulated/extended bear-market scenario (the window-tournament's 185-day OOS bear window, #898, is a ready-made starting point) — does removing the short side make bear-market drawdown better or worse than the current both-sides config?
- Independently assess whether this proposal and the still-open guard-design question (Sec. 8, point 5 of the counterfactual note — is "free ETH > $1 dust" the right net-position check for a cross-margin account?) should be decided together or separately; this proposal takes no position on the guard itself.
- Confirm this does not interact adversarially with any in-flight drawdown-containment or exit-geometry work already in the review pipeline.

## Rollback plan

Configuration-level change (a boolean/flag), not a structural code change — revert the flag (or config value) to restore both-sides trading. No state migration, no open-position implications if flipped between restarts (existing open positions are unaffected by a signal-generation-time flag).

## Verdicts

### risk-officer
(not yet reviewed)

### code-reviewer
(not yet reviewed — implementation not yet written; this proposal is the decision request, not a diff)

### pm
(not yet reviewed)
