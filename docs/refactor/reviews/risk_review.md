# Risk Review — Backtest↔Live Parity Unification Plan (v1) — 2026-07-05 21:50 UTC

**Reviewer:** risk-officer (independent — no coordination with other reviewers)
**Target:** `docs/refactor/backtest_live_parity_plan.md` (DRAFT v1, 2026-06-15)
**Lens:** capital protection. This bot trades live capital; the plan's product is *trust in backtest numbers used to size real risk*. That trust is the risk surface.

**Overall verdict:** APPROVE-WITH-CONDITIONS (of the *plan*, not of any code). The engineering
is sound and the phase ordering (measure-before-refactor) is correct. But the plan has one
BLOCKER-class governance gap (the "100% confidence" framing is a live-sizing hazard) and
several MAJOR gaps around default-flip governance, threshold ownership, and the seam between
"backtest accuracy change" and "I just changed live sizing." Each is fixable with language and
process, not redesign.

Verified against source (not taken on faith):
- `annual_margin_interest_rate` default **is** `0.0` (`backtest/execution/exit_handler.py:117`) — margin backtests are silently optimistic today, as §2 claims.
- `orders.actual_commission` **exists** (`database/models.py:373`) — P4.3 fee-truth source is real.
- Determinism fingerprint **is** a real BLAS-pinned byte-identical test (`tests/integration/parity/test_backtest_determinism.py`).
- `DEFAULT_MAX_DRAWDOWN = 0.20` (`config/constants.py:128`) **matches** `risk-limits.json` `max_drawdown_pct: 0.20`. No divergence — good.
- The drawdown-guard / circuit-breaker hard halt lives in `src/engines/live/monitoring/` (live-only), but consumes sizing/fee/SL-detection from the code this plan moves into `engines/shared/`. This is the load-bearing fact behind finding #4.

---

## Findings (ranked most-severe first)

### 1. [BLOCKER] "100% confidence" / "byte-exact parity" framing will be read as "safe to size up" — the exact residual gap that bleeds capital is the part parity does NOT cover.

**Rationale:** §1 and §6 lead with "100% confidence," "byte-exact, CI-enforced equality,"
"the strongest form of 100% confidence that exists." A human (or a future PM/daemon) who reads
"parity proven" and up-sizes live risk is betting capital on a guarantee that covers only
*decisions + costs + accounting on candle-level data*. The money historically bled in exactly
the uncovered residual: intrabar SL fills, SL-placement failure → emergency-close cascade
(capital-erosion postmortem, ~15% loss), phantom balance, gap-through-SL, real slippage/partial
fills. §9 "Honest limits" is correct and well-written — but it is one section at the *bottom*,
while the headline oversell is at the *top* and in the proof table (§6). In a risk sense the
document's center of gravity is on the wrong side of the residual.

**Suggested revision:**
- **§1 (top of doc):** Replace every literal "100% confidence" with a **scoped** claim.
  Mandate the exact phrase everywhere the parity result is surfaced (CI output, replay report,
  docs, any dashboard): *"Decision/cost/accounting parity: proven. Fill/slippage/liquidity
  fidelity: bounded & monitored, NOT guaranteed."* Ban the bare token "100% confidence" and
  "byte-exact parity" as standalone phrases in any human-facing artifact.
- **Promote §9 to §1.5** (right after the definition), not buried at the end. The honest-limits
  fence must be co-located with the claim it fences.
- **Add an explicit sizing guardrail to §9:** a one-line rule that **"a parity-passing backtest
  is NOT authorization to increase live position size, risk-per-trade, leverage, or
  `max_concurrent_positions` beyond `risk-limits.json`. Sizing changes remain a separate,
  human-ratified decision gated by the charter's >$50/24h and irreversible-action rules."**
  This severs the mental link "parity green → size up" that the current framing invites.
- **Add a ledger row (§7) for the sizing hazard itself:** direction of bias = *optimistic*
  (candle backtest cannot see the worst intrabar fill), so the residual is asymmetric — it
  makes strategies look *safer* than live, which is the dangerous direction for up-sizing.

---

### 2. [MAJOR] The P2.1 / P2.2 / P2.4 "flip defaults" step can silently change every strategy's reported edge — and thus its already-live sizing — between two releases, with no per-strategy sign-off.

**Rationale:** §5 says fidelity flips "land behind config flags … then flip defaults in a
separate, loudly-labeled PR." Good, but "loudly-labeled PR" is not a control. ExchangeRules
rounding + `min_notional` rejection (P2.1) and non-zero financing (P2.2) will change *reported
returns and reported max-drawdown* for **every** margin/small-notional strategy at once. If a
live strategy's position size was ever justified by a pre-flip backtest number, the flip
retroactively invalidates that justification — and nobody re-checks, because the flip PR is
framed as "backtest accuracy," not "live risk re-baseline." The A/B report proves the *magnitude*
of the change but does not force anyone to *act* on it for live-sized strategies.

**Suggested revision (amend §5 and each P2 item):**
- The default-flip PR MUST include a **per-live-strategy delta table**: for every strategy
  currently running live capital, old vs new backtest CAGR, max-drawdown, and 99th-pct daily
  loss. Any strategy whose new max-drawdown crosses a `risk-limits.json` threshold
  (e.g. `max_drawdown_pct 0.20`) or whose edge drops >X% is **flagged for human sizing review
  before the flip merges** — the flip is *blocked* on that review, not merely annotated.
- Each flip PR carries a **git-recorded "backtest epoch" tag** (e.g. `parity-epoch: 2`) that is
  stamped into every backtest report artifact. This makes "which fidelity assumptions produced
  this number" auditable forever, and makes cross-epoch comparisons refuse-by-default.
- **Rollback must be a config flip, not a revert.** §5 should state the flag stays wired
  (not deleted) for ≥2 releases after the default flip, so a discovered regression is a
  one-line env change, not a code rollback under pressure. Matches LESSONS §4 "ship inert,
  flip a flag."

---

### 3. [MAJOR] Until T₁/T₂ are ratified, the P4.2 scheduled parity audit is a no-op alarm — and an un-tuned threshold is itself a capital risk in both directions.

**Rationale:** §P4.2 ships T₁ (bps/trade) and T₂ (%/30d) as "placeholders — human ratifies,"
suggesting 5 bps / 0.1%/30d. Two failure modes, both real: (a) if the audit ships *before*
ratification with placeholders, and nobody wires the halt-decision, the "continuous proof"
is decorative — it alerts into a channel no one owns, replicating the *double-blind alerting*
failure already in institutional memory (observability audit). (b) An un-tuned threshold either
fires constantly (alert fatigue → the one real divergence is ignored — this is how the
capital-erosion incident's early signals got lost) or is set so loose that a systematic
live-worse-than-backtest drift never trips. Critically, **the plan never says who owns the
kill-decision when replay shows live systematically worse than backtest.** That is the whole
point of the audit and it has no owner.

**Suggested revision (amend §P4.2 and add to §6 CI policy):**
- **Do not enable alerting until T₁/T₂ are ratified in `parity_gap_ledger.md` with a named
  human owner and a dated review.** Until then the audit runs in **dry-run/report-only** mode
  (compute + log the deltas, page no one) — same "ship inert" discipline as money-movers
  (LESSONS §4). An unratified audit must be explicitly a no-op, not a silent maybe.
- **Bootstrap the thresholds empirically, don't guess.** Run the replay over the last 30–90d
  of real sessions first; set T₁/T₂ from the observed p95/p99 of *explained* divergence + a
  margin, so day-one alerts mean "worse than the historical norm," not "hit an arbitrary 5 bps."
- **Name the escalation owner and the decision rule in the plan:** parity-audit alerts route
  to `risk-officer` (assessment) → `pm` (action). Add the decision rule explicitly: **"if
  replay shows live realized P&L systematically worse than backtest by > T₂ over the trailing
  window on a *live-capital* strategy, that is an incident (`type:incident`), risk-officer
  recommends entry-halt / size-down, human ratifies."** Directional asymmetry matters: live
  *better* than backtest is a modeling curiosity; live *worse* is money leaving. The threshold
  should be **tighter on the live-worse side.**

---

### 4. [MAJOR] The seam between "harmless backtest accuracy change" and "I just changed live sizing" is not drawn — and the safety rail (§5 fingerprint "identical except where fidelity improves") has a hole exactly there.

**Rationale:** The refactor moves fees, sizing, SL-fill detection, and financing into
`engines/shared/`, consumed by BOTH engines. The live drawdown-guard / circuit-breaker
(`src/engines/live/monitoring/`, hard halt at 2.5% daily / 15% drawdown per LESSONS §5.2) acts
on numbers produced by that shared code. So a P1/P2 PR that is *motivated* by backtest fidelity
can change a live number (a rounded quantity, a financing leg, an SL trigger comparison) that
shifts realized P&L, the drawdown-guard's peak/trough inputs, or the daily-loss baseline — i.e.
it changes *when the live kill-halt fires*. §5's guarantee is "fingerprint byte-identical
**except where fidelity improves**" — but the "except" clause is precisely the live-affecting
set. The determinism fingerprint is a **backtest** oracle; it says nothing about whether the
live path's behavior moved. There is currently no equivalent "live behavior unchanged" gate for
the shared extractions.

**Suggested revision (amend §5 and §6):**
- **Add a live-side invariant gate to L1/L2.** For every shared extraction that the live engine
  consumes, add a test that pins the *live* adapter's output byte-identical pre/post-refactor on
  a fixed input (the same discipline as the backtest fingerprint, applied to the live path).
  A "verbatim move" PR must prove *both* engines unchanged, not just the backtest.
- **Classify every PR in this plan as either `parity:refactor-only` (fingerprint identical BOTH
  sides — no live behavior change) or `parity:fidelity-change` (deliberately moves a number).**
  The two classes get different review + rollback treatment. A `fidelity-change` PR that touches
  a live-consumed symbol (fees, sizing, SL detection, financing) is a **live-capital change** and
  must carry the full money-path discipline: dual reviewer + codex-review-to-APPROVE
  (LESSONS §4) + explicit statement of the live behavioral delta and its effect on the
  drawdown-guard / circuit-breaker inputs. The plan currently reserves "dual review on money
  paths" (§P1) for extractions but does not name the fidelity-change → live-halt-timing link.
- **§5 should state explicitly:** *the determinism fingerprint proves backtest reproducibility,
  NOT live-path invariance; live invariance is proven separately by the live-adapter pin.*
  As written, §5 implies the fingerprint covers the safety concern. It does not.

---

### 5. [MAJOR] P4 scheduled replay + fee-truth jobs are an operational risk to the RUNNING bot if they share the prod DB or exchange rate-limit budget with live trading.

**Rationale:** §P4.1/P4.2/P4.3 run scheduled jobs that read the prod DB (`trading_sessions`,
`orders`, cached candles) and, for fee-truth (P4.3), normalize `orders.actual_commission`
against exchange-reported commissions — which implies either DB reads or exchange API calls.
The plan does not address contention. Institutional memory is emphatic that the live bot has
been taken down by infra it shared: a DB/DNS outage killed the loop while Railway reported
SUCCESS (bots-down-railway-dns); the user-stream degrades to REST under stress. A replay/audit
job that (a) runs a heavy read against the prod DB during a live trading cycle, (b) holds a
long transaction/lock, or (c) burns Binance REST weight that the live reconciler needs, can
*cause* the incident the audit is meant to detect. The plan's own §P4 requires "persisting the
live session's input snapshot" — writing to prod adds write contention too.

**Suggested revision (add a new bullet to §P4 and to §5 safety rails):**
- **P4 jobs run read-only against a replica / snapshot, never a lock-taking transaction on the
  primary during live hours.** If no replica exists, they run against a point-in-time export,
  and their DB user is read-only-scoped. State this as a hard constraint, not a preference.
- **Fee-truth (P4.3) must NOT call the exchange with the live API key/weight budget.** Source
  commissions from already-persisted `orders.actual_commission` (it exists — verified) or a
  separate read-only key with its own rate budget. Never let an audit job compete with the live
  reconciler for Binance weight.
- **Schedule P4 jobs in a low-activity window and make them pre-emptible** — if the live engine
  signals a busy/degraded cycle, the audit defers. A monitoring job must never be able to starve
  the thing it monitors.

---

### 6. [MAJOR] P2.1 ExchangeRules uses a *committed static filter fixture* for backtest — a stale fixture silently re-opens the precision bug class the plan claims to close, and directionally biases backtests optimistic.

**Rationale:** §P2.1 sources live filters from `exchangeInfo` (live truth) but backtest from
"a recorded/static filter set per symbol (committed fixture, refreshable via a CLI)." Binance
changes `step_size` / `tick_size` / `min_notional` over time. A stale fixture means the backtest
quantizes and rejects `min_notional` against *yesterday's* rules while live uses *today's* — so
the plan claims "removes caveat #1" but actually introduces a **new, silent** divergence that
looks like parity (the code path is shared) while the *data* diverges. LESSONS §1.1 is explicit
that precision bugs come in pairs and must be grepped as a class; a stale filter fixture is the
data-layer sibling of that bug. Direction of bias matters: if `min_notional` rose on the venue
but not in the fixture, the backtest will *accept* trades live now rejects → optimistic phantom
edge.

**Suggested revision (amend §P2.1):**
- **The fixture must carry a capture timestamp and the audit (§P4.2) must diff the committed
  fixture against live `exchangeInfo` on schedule**, filing drift as a `source:parity-audit`
  issue exactly like a threshold breach. A stale filter set is a parity failure, not a
  housekeeping chore.
- **Add a ledger row (§7):** "backtest symbol-filter fixture staleness — direction: optimistic
  if venue min_notional/step rose — monitor: scheduled fixture-vs-exchangeInfo diff."
- Extend the P3.2 precision grep-gate to also fail CI if a symbol is backtested with **no**
  committed filter fixture (fail-closed: no filters ≠ "raw floats OK", it means "reject").

---

### 7. [MINOR] `parity_gap_ledger.md` bounds have no owner, no review cadence, and no "unbounded until measured" default — a ledger row without a ratified number is false comfort.

**Rationale:** §7 is the right instrument, but a ledger whose `bound` column contains guesses or
blanks is worse than none — it *looks* governed. The seed rows (intrabar tick path, partial
fills, latency) are exactly the capital-bleed residuals from finding #1; if their bounds are
placeholders, the whole "measured and bounded" claim is unbacked.

**Suggested revision (amend §7):**
- Add columns: `owner`, `last_reviewed`, `ratified (y/n)`. A row with `ratified=n` renders as
  **UNBOUNDED** in every report (not as a small number), so unmeasured risk reads as risk, not
  as "≈0". Mirror the `risk-limits.json` convention (`$last_reviewed`, `$last_reviewer`) — that
  file's own `$last_reviewed` is `1970-01-01`, a cautionary example of an unreviewed control.

---

### 8. [MINOR] P2.3 multi-position is deferred to last, but the harness's `max_concurrent_positions == 1` enforcement (option b) must be a HARD, tested gate, not a documentation note — or single-position parity silently "covers" a multi-position live strategy.

**Rationale:** §P2.3 correctly flags that backtest is structurally single-position today and
recommends full multi-position (option a) sequenced last, with option (b) as the interim scope
fence. The risk is the *interim*: if any live strategy runs `max_concurrent_positions > 1` while
the backtest can only represent one, a parity-passing backtest is affirmatively misleading about
that strategy's real (correlated, concurrent) risk — the correlation-cluster exposure limit
(`max_correlated_exposure_pct: 0.15`) is unrepresentable in backtest.

**Suggested revision (amend §P2.3):**
- Until option (a) lands, the harness and the parity-audit MUST **hard-fail** (not warn) when
  asked to compare any strategy whose *live* config has `max_concurrent_positions > 1`, and the
  parity claim for that strategy is explicitly stamped **"NOT PARITY-COVERED — single-position
  backtest only."** No live strategy should be able to cite a parity pass it isn't covered by.

---

### 9. [MINOR] The plan has no explicit "parity check FAILS → what happens to live?" runbook. Silence here defaults to "keep trading," which is the wrong default for a capital system.

**Rationale:** The plan defines exhaustively what "parity proven" means, but not the negative:
when the scheduled audit (P4.2) shows a real, unexplained, live-worse divergence on a
live-capital strategy, what is the operational response? Finding #3 assigns the owner; this is
the missing *action*. Absent a stated runbook, the default is inertia (keep trading on a backtest
we now know is lying), which is the opposite of capital-protective.

**Suggested revision (add a short runbook subsection to §P4):**
- Tiered response: (1) divergence > T₁ single-trade → log + ledger classification;
  (2) cumulative > T₂ live-worse on a live strategy over the window → risk-officer incident +
  recommend `atb live-control halt` / entry-pause on that strategy pending root-cause;
  (3) divergence pattern matches a known capital-bleed signature (SL-fill mismatch, phantom
  balance) → escalate at MAXIMUM urgency per the risk charter, recommend immediate halt.
  The kill-switch stays human-triggered (per `risk-limits.json` `authorized_actors: [human]`),
  but the recommendation path must be pre-written, not improvised during an event.

---

## What I could not verify (data/scope limits)
- I did not run the existing parity suite or the determinism fingerprint (read-only review of
  the *plan*, not the code). I confirmed the tests *exist* and assert what the plan claims, not
  that they currently pass on this worktree.
- I could not confirm whether a prod DB read-replica exists (finding #5's cleanest mitigation
  depends on it) — the plan should state the answer.
- I did not audit whether any *currently-live* strategy runs `max_concurrent_positions > 1`
  or a margin/short config with non-zero real borrow (which would make findings #2/#6/#8 live
  today, not hypothetical). The PM should establish this before P2 flips any default.
- The suggested opener thresholds (T₁=5 bps, T₂=0.1%/30d) are unvalidated by me; finding #3
  requires empirical bootstrap before they mean anything.
```
