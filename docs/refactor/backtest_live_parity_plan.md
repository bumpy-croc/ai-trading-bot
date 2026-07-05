# Backtest ↔ Live Parity: Unification Plan

> **Status:** v2 — revised after a three-agent adversarial panel review (risk-officer,
> quant-researcher, architecture-reviewer; see §11). `/codex-review` was requested but is
> unavailable in the authoring environment; run it locally against this doc as a further pass.
> **Goal owner:** human. **Author:** Claude Code session `0188LNSixYW9Fa5hrJ7YWJoa` (v1
> 2026-06-15, v2 2026-07-05).
> **Prereq:** ✅ satisfied — the live-engine modularization (#486,
> `docs/refactor/live_engine_modularization.md`) is COMPLETE as of 2026-07-05.
> **Branch base:** `develop`. **Note:** parts of the v1 audit predate the #838 partial-exit
> accounting fixes; P0 includes a re-verification of the duplication map.

---

## 0. Decisions required from the human before/at phase boundaries

| # | Decision | Phase | Recommendation |
|---|---|---|---|
| D1 | **Closed-candle gating**: live currently acts on the *forming* candle (`kline_buffer` updates the in-progress bar on every WS event with no `is_closed` gate; the loop decides on `df.iloc[-1]`). Option (a): gate live decisions to closed candles — a **live behavior change** needing sign-off; likely also a correctness fix (deciding on mutating high/low/close). Option (b): keep intra-candle decisioning and model it as a permanent, ledgered divergence with its own scenarios. | before P1 | (a) |
| D2 | **Multi-position backtest**: live's production default is `max_concurrent_positions = 3` (`src/risk/risk_manager.py`). Single-position-only parity would not describe how the bot actually trades. Commit to the full multi-position backtest (P2.3a)? | before P2.3 | yes — (a); interim runs must banner "PARITY NOT VALIDATED: multi-position config" |
| D3 | **Futures/perpetuals scope**: funding-rate carry is unmodeled in both engines. In scope now, or explicitly deferred until futures go live-affecting? | P2.2 | design the financing interface to accommodate it; defer the implementation with a ledger row |
| D4 | **Divergence bounds ratification**: T₁/T₂ thresholds and their alert action must be ratified in the same human-reviewed artifact as `.claude/state/risk-limits.json` (currently epoch-dated/unreviewed). Alert action should match the charter's `breach_action` (halt new entries + page), not just file an issue. | before P4.2 | ratify alongside a first real risk-limits review |

---

## 1. Goal and what "parity" means operationally

**Ultimate goal:** the backtest engine is a faithful predictor of the live engine. Every
economically meaningful thing that happens in live — fees, slippage, exchange rounding,
margin interest, stop-order mechanics, partial operations — is either (a) **the same code**
in both engines, or (b) **explicitly modeled** in backtest with a **measured, bounded,
monitored** difference.

"100% confidence" cannot mean "identical outcomes" — live trading has irreducible
environmental facts (latency, intrabar tick paths, order-book liquidity, data-feed
revisions, exchange outages). It **can** mean, precisely:

1. **Decision parity** — given identical *closed-candle* inputs, both engines make
   identical entry/exit/sizing decisions. *Provable exactly* — **contingent on D1**; until
   the closed-candle question is resolved, live's decision inputs are not the backtest's.
2. **Execution parity** — given identical fill physics, both book identical prices, fees,
   slippage, quantities, P&L. *Provable exactly **against the simulated venue** —
   model-vs-model.* Whether the model matches the real venue is a separate, calibrated
   claim (L5), which is therefore a **required milestone**, not an optional tail.
3. **Accounting parity** — balances, fee legs, interest legs, quantization residuals,
   trade records evolve identically. *Provable exactly (same caveat as 2).*
4. **Environmental fidelity** — the residual live-vs-model gap (real ticks vs candles,
   spread/impact vs candle prices, real borrow/funding vs modeled, latency, data
   revisions) is *quantified continuously* against production and stays inside ratified
   bounds. *Provable statistically, never exactly.*

**Scope banner:** until P2.3(a) lands, all parity claims apply only to
single-position configurations; any run with `max_concurrent_positions > 1` (the
production default) must emit a loud "parity not validated" warning in harness and CLI.

**Sizing guardrail (hard rule — risk review finding #1):** the residual this plan cannot
close is biased *optimistic* (a candle backtest cannot see the worst intrabar fill, so a
pass makes a strategy look *safer* than live — the direction that historically bled capital:
SL-placement failure → emergency-close cascade with phantom balance). Therefore **a
parity-passing backtest is NOT authorization to increase live position size, risk-per-trade,
leverage, or `max_concurrent_positions` beyond `risk-limits.json`.** Sizing changes remain a
separate, human-ratified decision gated by the charter's `>$50 / 24h` and irreversible-action
rules. Parity green ⇏ size up.

**Language rule (governance):** the bare tokens **"100% confidence"** and **"byte-exact
parity"** are banned from human-facing artifacts (CI output, replay reports, dashboards).
Surface results only as the scoped claim: *"Decision/cost/accounting parity: proven
model-vs-model within declared tolerance. Fill/slippage/liquidity/concurrency fidelity:
bounded & monitored (L4/L5), NOT guaranteed."*

---

## 2. Current state (v1 audit 2026-06-15; env facts re-verified 2026-07-05)

### Already unified (single shared implementation, both engines call it)
- **Fees, slippage, fill price** — `src/engines/shared/cost_calculator.py`
  (`calculate_entry_costs` / `calculate_exit_costs`), `execution/ohlc_fill_model.py`,
  `fill_policy.py`. Both engines' execution engines
  (`src/engines/{backtest,live}/execution/execution_engine.py`) construct and delegate to
  the shared `CostCalculator` — verified by the panel; cite by symbol, not line number.
- **Models & P&L** — shared `models.py` (`Position`, `Trade`, `pnl_percent`).
- **Entry-plan extraction + dynamic-risk sizing** — `SharedEntryHandlerMixin` (both entry
  handlers subclass it; byte-identical by construction), `dynamic_risk_handler.py`.
- **Trailing stops, strategy-exit detection, partial-op *decisions*, correlation control,
  risk-config merging, validation, side utils** — all in `engines/shared/`.

### Duplicated (same concept, two implementations — drift risk)
- Entry orchestration: `backtest/execution/entry_handler.py` vs live
  `entry_handler.py` + `entry_coordinator.py`.
- Exit orchestration (SL→TP→trailing→strategy→time→partial ordering): backtest
  `exit_handler.py` (monolithic) vs live `exit_handler.py` + `exit_coordinator.py`.
- SL/TP high/low fill detection: inline in each exit handler, equal only by tests.
- Partial exit/scale-in **execution**: `shared/partial_exit_executor.py` is **not a stub** —
  it is ~205 lines of complete P&L/fee/slippage logic already wired into the backtest tracker,
  but it **duplicates fee/slippage math** (`exit_fee = abs(exit_notional * self.fee_rate)`,
  `slippage_cost = abs(exit_notional * self.slippage_rate)`) instead of delegating to
  `CostCalculator` — a live CODE.md "never duplicate financial logic" violation, and the real
  P1.4 target. ⚠️ Re-audit post-#838 in P0 — this area changed after the v1 audit.

### Live↔backtest divergences (union of code caveats + panel findings)
1. **Partial-candle decisioning (NEW, panel blocker)** — live acts on the forming candle
   (`src/engines/live/kline_buffer.py` updates the current bar on every WS event; no
   `is_closed` gate anywhere in `src/engines/live`); backtest sees only final bars.
   Decision-*input* mismatch. → D1, P1.0.
2. **Exchange quantity/price quantization** — live rounds to `step_size`/`tick_size` and
   enforces `min_qty`/`min_notional` (`_normalize_quantity`; LESSONS §1.1); backtest uses
   raw floats (documented caveat in `backtest/engine.py`). Additionally (panel): the
   rounding is *asymmetric* (round-to-nearest on entry, floor on exit) and the resulting
   **dust/residual is never booked back into balance or subsequent sizing** in either
   engine's model — systematic one-directional drift. → P2.1.
3. **Margin/borrow interest** — live queries the exchange (`margin_interest_tracker.py`);
   backtest models via `annual_margin_interest_rate` **defaulting to 0.0** (silently
   optimistic). Metadata stashing differs. → P2.2.
4. **Funding-rate carry (NEW)** — futures/perp funding is unmodeled and unmentioned in
   both engines. → D3, P2.2, ledger.
5. **Warmup/lookback (NEW)** — backtest hard-skips the first `warmup_period` indices;
   live has no equivalent gate (only NaN-drop). Early-session decisions can differ. → P2.6.
6. **Single- vs multi-position** — backtest holds one `ActiveTrade`; live holds N
   (production default 3). **Largest structural gap.** → D2, P2.3.
7. **Stop-loss as a real resting order** — live SL can fill mid-bar, partially fill, be
   cancelled/rejected (#741), reserves margin (#710); backtest checks per candle. → P0.1b/P2.4.
8. **Bid/ask spread & order-book impact (NEW)** — absent from *both* engines
   (`fill_policy.py` declares `"quote"`/`"order_book"` fidelity levels; only
   `"ohlc_conservative"` exists). The proof system can therefore reach perfect
   self-consistency while both engines are wrong vs real fills. → L5/P4.3+, ledger.
9. **Data source (NEW)** — live consumes WS klines (REST seed/fallback); backtest
   re-fetches REST history, which the venue can revise. Same timestamps ≠ same bytes. → P4.0.
10. **Partial-op cadence** — live evaluates every loop tick; backtest once per candle. → ledger.
11. **Sentiment freshness** — live overlays real-time sentiment; backtest is
    all-historical. Column-parity locked; value distributions differ by design. → ledger.
12. **Reconciliation / recovery / order tracking** — live-only by nature; their economic
    effects (external closes, restart-mid-position) are edge events the harness must
    *produce* and the ledger must bound. → P0.3 fault matrix.

---

## 3. Target architecture: thin drivers around shared **decision producers**

Panel-corrected shape: the engines' state models are irreconcilable (single-trade
synchronous loop vs N-position dict under locks with exchange round-trips). Forcing one
stateful "workflow object" over both would create adapter-soup and pressure live's
safety-critical I/O ordering. The shared surface is therefore **pure, ordered decision
functions** — state in, intents out — and each driver keeps its own effecting loop.

```
                 ┌──────────────────────────────────────────────────────┐
                 │           src/engines/shared/  (the core)             │
                 │                                                       │
                 │  CostCalculator · OHLCFillModel · FillPolicy     ✅   │
                 │  Models/PnL · DynamicRisk · Correlation          ✅   │
                 │  TrailingStops · StrategyExitChecker             ✅   │
                 │  PartialOpsManager (decisions)                   ✅   │
                 │  ─────────────── NEW in this plan ───────────         │
                 │  Clock protocol (injectable time)            🆕 P0.0  │
                 │  ExchangeRules (filters + quantization +              │
                 │    dust accounting)                          🆕 P2.1  │
                 │  ExitTriggerEvaluator (one high/low fill fn,          │
                 │    one same-bar tie-break)                   🆕 P1.1  │
                 │  ExitDecisionSequence — PURE ordered evaluator        │
                 │    (position+candle → ExitPlan | None)       🆕 P1.2  │
                 │  EntryDecisionSequence — PURE (context →              │
                 │    OrderIntent | None)                       🆕 P1.3  │
                 │  PartialExitAccounting (shared math; drivers          │
                 │    effect)                                   🆕 P1.4  │
                 │  FinancingCostProvider (borrow now; funding-          │
                 │    ready interface)                          🆕 P2.2  │
                 └──────────▲────────────────────────▲──────────────────┘
                            │ decisions/intents      │ decisions/intents
              ┌─────────────┴─────────┐   ┌──────────┴──────────────────┐
              │  Backtest driver      │   │  Live driver                │
              │  candle iteration     │   │  real-time loop, WS         │
              │  portfolio state      │   │  exchange I/O + the SAFETY  │
              │  (multi-position 🆕)  │   │  ORDERING: cancel-SL-before-│
              │  effects fills via    │   │  close (#710), deferred     │
              │  shared fill model    │   │  drain (#631), re-protect   │
              └───────────────────────┘   │  (#741), locks — NEVER      │
                                          │  moved into shared          │
                                          └─────────────────────────────┘
```

**Boundary rule (protected):** shared code decides *what* to do; drivers decide *how and
in what I/O order*. The live order-lifecycle safety steps are explicitly outside the
shared surface, listed in §5 as protected invariants with named tests.

---

## 4. Workstreams

**Ordering principle: build the measuring instrument first — against surfaces that won't
be refactored out from under it.**

### Phase P0 — Parity harness (measure, then refactor) — est. 4–6 PRs
- **P0.0 `Clock` protocol** *(new; the panel found the claimed seam doesn't exist —
  `loop_timing.py` calls `time`/`datetime` directly)*: injectable clock through
  `LiveLoopTimingCoordinator` and the freshness path (default real; scripted in tests).
  Genuine first PR; P0.2 and L3 depend on it.
- **P0.1a `SimulatedFillOracle`**: stateless fills for market/TP/time exits via the shared
  `OHLCFillModel` + `ExchangeRules`.
- **P0.1b `SimulatedExchange`**: stateful implementation of the full live
  `ExchangeInterface` (~20 abstract methods): resting SL orders that fire intrabar per a
  **fidelity tie-break committed here** (pulled forward from P2.4 — exact equality is
  unassertable without it), order status transitions, cancels, margin reserve (#710),
  `min_notional` rejection. **Plus a fault-injection mode**: order rejections (-2010),
  failed cancels, partial SL fills, WS drops, injected external closes, kill/restart.
- **P0.2 Side-by-side equality harness**: the *real live engine* (paper +
  `SimulatedExchange` + scripted `Clock`) vs the backtest engine on identical candles,
  compared as canonical `TradeRecord` tuples *(symbol, side, entry_ts, entry_px, qty,
  exit_ts, exit_px, reason, fee_entry, fee_exit, slippage, interest, quantization_residual,
  gross_pnl, net_pnl, balance_after)*. **Binds to the engine's public surface** (start →
  loop → records out), never handler internals, so P1's refactor cannot invalidate it.
  **Determinism statement required**: how live execution is pinned lock-step
  (single-threaded drive of the loop; WS callbacks delivered synchronously between candles)
  — or, failing that, an explicit tolerance/flake policy. Without this, L3 fuzzing produces
  false positives that erode trust.
- **P0.3 Scenario matrix** (golden fixtures): trend→TP; SL via low breach; SL+TP same bar
  (tie-break pinned); gap through SL (fill at open); trailing ratchet→trigger; time exits
  (max-holding/end-of-day/weekend); partial ladder + scale-in; short with N-day interest;
  `min_notional` rejection after quantization; maker vs taker; same-bar-entry protection;
  **session-start/warmup-boundary trades**; **fault sub-matrix** (each P0.1b fault fires
  the live fail-closed branch and the books still reconcile); **restart-mid-position**
  (kill live driver holding a position, restart, reconcile; balance delta compared against
  `reconciliation_balance_critical_pct` — the kill-switch condition); multi-position
  interleaving (activates with P2.3).
- **P0.4 Divergence report**: on inequality, per-trade field-level diff (first divergent
  field, both values, candle index). Also the P4 report format.
- **P0.5 Duplication re-audit**: refresh the §2 map post-#838 before P1 slicing.

*Exit criterion: harness green on current code for the happy-path matrix; every known
divergence either reproduced by a scenario or ledgered; fault matrix demonstrates the
live fail-closed branches actually run under the harness.*

### Phase P1 — Shared decision producers (decision parity by construction)
- **P1.0 Closed-candle gating (D1)**: implement the human's choice — (a) gate live
  decisions on closed candles (live behavior change: flag-gated, A/B'd, then default), or
  (b) model intra-candle decisioning as a ledgered divergence with dedicated scenarios.
  Decision parity claims are conditional until this lands.
- **P1.1 `ExitTriggerEvaluator`**: single shared SL/TP/liquidation trigger detection from
  candle high/low incl. same-bar tie-break and gap rules. Both exit handlers call it.
- **P1.2 `ExitDecisionSequence`** *(pure)*: ordered evaluation (SL → TP → trailing →
  strategy → time → partial) producing an `ExitPlan`; **owns the decision order only**.
  Live's cancel-before-close / deferred-drain / re-protect stay in the live driver.
- **P1.3 `EntryDecisionSequence`** *(pure)*: plan → dynamic-risk → correlation → sizing →
  `ExchangeRules` quantization → cost preview → `OrderIntent`.
- **P1.4 `PartialExitAccounting`**: shared math for partial exit/scale-in deltas
  (sizes, fees, realized P&L, remaining qty); drivers effect them.
- **Preconditions:** a short **threading & lock-ownership note** per shared producer
  (which thread calls it in live, what state it may read, why it takes no locks — pure
  functions should need none; that's the point), reviewed before code (CODE.md Thread
  Safety). Same #486 discipline otherwise: AST-assisted moves, fingerprint + harness
  byte-identical per PR, dual review on money paths.

### Phase P2 — Model live realities in backtest (execution & accounting parity)
- **P2.1 `ExchangeRules`**: symbol filters (`step_size`, `tick_size`, `min_qty`,
  `min_notional`) + `src/trading/precision.quantize_to_step`, sourced from `exchangeInfo`
  in live and from a committed, refreshable fixture in backtest. Backtest applies the same
  rounding and rejections as live **and books the quantization residual (dust) into
  balance/next-sizing identically in both engines** — sharing the rounding without the
  residual accounting would not close the drift. Flag-gated with an A/B report; see §5
  default-flip quarantine.
- **P2.2 `FinancingCostProvider`**: one interface — live queries the exchange; backtest
  accrues from a configured rate curve; **both stash identically to trade metadata**; one
  event-logger read path. Interface designed to also carry **funding-rate** cash flows
  (D3): implement when futures are live-affecting; ledger row until then. Calibration job:
  modeled vs exchange-reported interest per session; drift beyond threshold auto-files.
- **P2.3 Multi-position backtest (D2)** ⚠️ **highest-severity item in the plan** — it
  rewrites portfolio accounting for exactly the quantities `risk-limits.json` governs
  (drawdown, correlated exposure, leverage). Own design doc **with risk-officer sign-off
  before code**; the doc must show per-position SL orders + correlation caps composing
  without double-counting exposure. Option (b) (clamp to single-position) is acceptable
  only as an interim with the §1 scope banner — **not as an end state**, because live's
  default is 3 concurrent positions. Sequenced **before P4** (replaying real multi-position
  sessions requires it).
- **P2.4 Backtest resting-order tier**: backtest-side counterpart of the fidelity
  tie-break committed in P0.1b (config `execution_fidelity`): default candle-level
  conservative; optional resting-order mode mirroring live SL semantics (#710 reserve,
  cancel-before-close). Partial-fill modeling explicitly out of scope initially (ledgered).
- **P2.5 Cadence**: quantify live-tick vs per-candle partial-op gap in the ledger;
  optional sub-candle evaluation later.
- **P2.6 Warmup unification**: one shared warmup/readiness gate (backtest's index skip and
  live's context-readiness check driven by the same computed warmup), or an explicit
  ledger row + session-start scenario if unified gating is rejected.

### Phase P3 — Thin drivers + drift-proofing
- **P3.1** Backtest `engine.py` reduces to: data iteration, portfolio state, shared
  decision calls, fill effecting via the shared model, reporting. Live engine: loop + I/O
  + safety ordering + reconciliation around the same producers.
- **P3.2 Ownership guard** *(corrected: import-linter is absent from the repo and can't
  express this anyway)*: an AST-based pytest guard failing CI if
  `engines/{backtest,live}/**` **define** symbols owned by shared (fee/fill/trigger/sizing
  math), plus a regex CI gate for `round(x/step)*step` patterns outside
  `trading/precision` (LESSONS §1.1).
- **P3.3** Docs refresh; retire closed "Known parity caveats" docstring items into the ledger.

### Phase P4 — Continuous proof against production (environmental fidelity)
- **P4.0 Session input snapshot** *(panel: prerequisite, currently missing —
  `TradingSession` persists no candle snapshot, model version, resolved config, or
  point-in-time sentiment)*: persist per session the **raw candles actually consumed**
  (WS-fed values, not a re-fetchable range — venues revise history), model/feature-schema
  version + checksum, fully-resolved effective config, and sentiment values at decision
  time. Replay is not well-posed without this. *Partial reuse (verified):* the immutable,
  INSERT-only `strategy_executions` table already persists per-decision OHLCV
  (`indicators`) and `sentiment_data` FK'd to each trade — P4.0 should extend that record
  (add model `version_id` + config checksum) rather than build a parallel snapshot store,
  and must NOT rely on the mutable `cached_data_provider` parquet cache (overwritten in
  place via `os.replace`, no provenance) as the candle source.
- **P4.1 `atb parity replay --session <id>`**: replay the snapshot through the backtest
  engine; emit the P0.4 report.
- **P4.2 Scheduled parity audit** (weekly): replay last N sessions. **Metrics (panel-
  corrected):** signed *and* unsigned per-trade divergence (bps **and** absolute $),
  split by side/regime/symbol, plus a **non-resetting cumulative unexplained-P&L drift
  statistic since last human ratification** — rolling windows alone hide cancelling biases
  and slow bleeds. Alert action per charter `breach_action` (halt new entries + page), not
  merely a GitHub issue (D4).
- **P4.3 Fee-truth reconciliation** *(required milestone before "backtest costs = real
  costs" is claimed — L5 in §6)*: modeled fees vs exchange-reported commissions per order
  (`orders.actual_commission` is unit-ambiguous/async — normalize first), maker/taker
  misclassification, BNB-discount drift; corrections feed fee config.
- **P4.4 Slippage-vs-book spot check**: for orders above a notional threshold, compare
  modeled slippage against actual fill price vs best bid/ask at order time (data permitting)
  — the only layer that touches divergence #8 (spread/impact).

---

## 5. What must NOT change (safety rails)

- **Protected live invariants** (named tests; never moved into shared code; never
  reordered): cancel-resting-SL-before-market-close precedes every live close (#710);
  stop-loss fills are drained to the loop thread, never executed on the poll thread
  (#631); SL-cancel escalation re-protects or alerts (#741); base-asset lock scope on
  entry/exit (#703); fail-closed order-tracking-lost handling.
- All refactor PRs behavior-preserving; determinism fingerprint stays byte-identical
  **except** where a phase deliberately improves fidelity (P1.0a, P2.1, P2.2, P2.4) —
  those land flag-gated with an A/B report and human sign-off, then flip defaults in a
  separate, loudly-labeled PR.
- **Default-flip quarantine:** when P2.1/P2.2/P1.0 defaults flip, every
  `state:paper`/`state:building` strategy whose approval backtest predates the flip is
  auto-flagged "requires re-backtest before promotion"; CI blocks model/strategy promotion
  on a stale-parity-config check. Owner of the re-validation sweep: pm.
- **Performance:** benchmark gate on a 2,000+-candle fixture at **±5%** (±20% on a
  seconds-long run is noise), plus a structural de-vectorization assertion (bound
  per-candle object allocations/dispatch in the hot SL/TP scan). Perf regressions are
  fixed by optimization — **never by removing or thinning a fail-closed step**; benchmark
  PRs touching money paths get dual review.

---

## 6. The proof system (layer by layer, with honest claims)

| # | Layer | Mechanism | Proves | Does NOT prove | Status |
|---|---|---|---|---|---|
| L0 | Determinism | fingerprint test (BLAS pinned) | the oracle is reproducible | anything about live | ✅ |
| L1 | Component differential | identical inputs → identical outputs through both engines' adapters; wiring tests | no adapter drift | model correctness | partial → P1 |
| L2 | Scenario equality | P0.2 harness, happy-path matrix, exact `TradeRecord` equality | **the two drivers wire the shared core identically** | that the shared fill/cost model matches the real venue (common-mode bugs invisible — the sim *is* the model); any live fault path | 🆕 P0 |
| L2b | Fault equality | P0.1b fault injection + fault sub-matrix | live fail-closed branches fire and the books reconcile under faults | venue-realistic fault timing | 🆕 P0 |
| L3 | Property-based | Hypothesis: random OHLC × configs → (i) trade-sequence equality, (ii) balance conservation `final = initial + Σnet − Σfees − Σinterest − Σresiduals` in both, (iii) no orphan trades | parity isn't fixture-shaped | see L2 caveat; requires P0.2 determinism pinning | 🆕 P0/P1 |
| L4 | Production replay | P4.0–P4.2 scheduled divergence reports vs ratified bounds | the *model* tracks *production reality* within bounds | exactness | 🆕 P4 |
| L5 | Exchange-truth audit | P4.3 fees/interest + P4.4 slippage-vs-book | cost model calibrated to actual venue behavior — **required before relying on backtest cost realism** | future venue changes | 🆕 P4 |
| G | Ownership guard | AST definition guard + precision regex gate in CI | duplication can't silently return | — | 🆕 P3 |

**CI policy:** L0–L3 (incl. L2b) required on every PR touching `src/engines/**`. L4/L5
scheduled; alerts follow charter `breach_action`.

---

## 7. Parity Gap Ledger (living table — `docs/refactor/parity_gap_ledger.md`)

Row schema: `id | description | direction of bias | quantified impact (method + number) |
bound | monitor | status`. Seed rows: intrabar tick path vs candle SL fills; live partial
fills; **spread/order-book impact vs candle prices (both engines)**; **funding-rate carry
(if D3 defers)**; **partial-candle decisioning (if D1 chooses (b))**; **warmup boundary
(if P2.6 ledgers instead of unifies)**; **WS-vs-REST candle revisions**; quantization-dust
residual (until P2.1 books it); latency; partial-op cadence; sentiment freshness;
reconciler-booked external closes; exchange outages/restarts mid-position.
**Rule:** harness and replay reports may only show divergences that map to a ledger row;
anything unexplained is a failing check, not a shrug.

---

## 8. Sequencing, effort, dependencies

| Phase | PRs (est.) | Depends on |
|---|---|---|
| P0 harness (incl. Clock, sim, fault mode, scenarios, re-audit) | 4–6 | #486 done ✅ |
| P1.0 closed-candle decision (D1) | 1–2 | P0 harness, human D1 |
| P1 decision producers | 3–4 | P0, P1.0 |
| P2.1 ExchangeRules + dust accounting | 2 | P0 |
| P2.2 Financing provider (+ funding-ready interface) | 1–2 | P0 |
| P2.4 resting-order tier (backtest side) | 1–2 | P0.1b tie-break |
| P2.6 warmup unification | 1 | P0 |
| P2.3 multi-position (design doc + risk sign-off first) | 3–5 | P1; human D2 |
| P3 thin drivers + guards | 2 | P1, P2 |
| P4.0 session snapshot persistence | 1–2 | — (can start early) |
| P4.1–P4.4 replay + audits | 2–3 | P4.0, P2.2, **P2.3** (production runs multi-position) |

Total ≈ **21–29 PRs** over multiple sessions. Fastest confidence wins early: P0 harness +
P2.1 rounding + P4.3 fee-truth. Highest risk: P2.3 — own design review, sequenced after
the decision producers exist but **before** production replay can be trusted.

---

## 9. Honest limits (what even this plan cannot promise)

Candle-level simulation cannot see intrabar tick ordering (we pin a conservative tie-break
and measure the residual in L4), real order-book liquidity/partial fills, spread at the
moment of fill (L5/P4.4 samples it; nothing proves it continuously), venue latency, or
intra-session borrow/funding-rate changes. L2's exactness is model-vs-model by
construction; only L4/L5 connect the model to the venue, statistically. The claim this
plan supports when complete: *decisions, costs, and accounting are identical by
construction (model-vs-model, within declared tolerance); the remaining environment gap is
measured continuously against production and bounded by numbers a human ratified.* That is
the strongest honest form of confidence available in this domain (see the §1 language rule —
"100% confidence" is not a claim we make); anything stronger is marketing.

---

## 10. Relationship to existing docs
- `docs/refactor/live_engine_modularization.md` — completed prerequisite; its extraction
  discipline (AST moves, fingerprint, dual review, hardening-followups) carries over.
- `backtest/engine.py` "Known parity caveats" docstring — items migrate to the ledger as
  they close.
- `.claude/LESSONS.md` §1.1 — quantization rules; enforced by the P3.2 precision gate.

## 11. Review log

- **v1** (2026-06-15): initial draft.
- **v2** (2026-07-05): revised per three-agent adversarial panel (risk-officer,
  quant-researcher, architecture-reviewer), `/codex-review` being unavailable in the
  authoring environment. Substantive changes: partial-candle decisioning surfaced as a
  blocker (new D1/P1.0); multi-position re-scoped — option (b) demoted to interim-only,
  P2.3 re-ranked before P4 (live default is 3 concurrent positions); stateful shared
  workflows replaced by **pure decision producers** with live safety ordering explicitly
  protected (§5); L2 claims re-scoped (model-vs-model, circularity acknowledged) and a
  fault-equality layer (L2b) + fault-injection sim mode added; `Clock` protocol added as
  P0.0 (claimed seam did not exist); SimulatedExchange split and re-estimated, fidelity
  tie-break pulled into P0; harness re-bound to the engine public surface; P4 made
  well-posed via session input snapshots (P4.0) and upgraded divergence statistics
  (signed/unsigned, per-segment, non-resetting cumulative drift); spread/impact, funding,
  warmup, quantization-dust, and WS-vs-REST divergences added to §2/§7; default-flip
  quarantine + promotion gate added; benchmark gate tightened (2k candles, ±5%,
  de-vectorization assertion); import-linter replaced by an AST ownership guard; threshold
  ratification tied to a reviewed risk artifact with charter-consistent alerting; effort
  re-estimated 15–20 → 21–29 PRs. Panel also *verified*: shared cost path, shared entry
  mixin, duplication map, single-vs-multi as largest gap, #486 prereq complete.
- **v2 merge** (2026-07-05): two sessions independently ran the three-lens panel and drafted
  a v2 in parallel; this document is the reconciled union. Grafted from the sibling draft:
  the explicit **sizing guardrail** and "100% confidence" language ban (risk finding #1);
  the corrected `PartialExitExecutor` characterization (implemented-but-duplicates-fees, not
  a stub); and the `strategy_executions` reuse note for P4.0. The three code-verified panel
  reports are committed as durable artifacts under **`docs/refactor/reviews/`**
  (`quant_review.md`, `architecture_review.md`, `risk_review.md`) — cite these for the full
  file:line evidence behind each finding.

---

## PM adoption notes (2026-07-05, ownership moved to the PM daemon session)

The Board directed execution of this plan via the PM session's sub-agents. Amendments to
the decision table, applied as adopted policy:

- **D1 — ratified (a), argument strengthened**: the production model trains exclusively on
  closed bars, so forming-bar evaluation is out-of-distribution input at every live
  decision — this is model correctness, not just parity hygiene. Conditions: protection
  paths (SL/trailing/exit) remain tick-driven; only signal/entry evaluation gates on
  closed candles. Evidence: forming-bar flip-rate study (docs/research/experiments/
  2026-07-06_forming-bar-fliprate.md) quantifies the blast radius and feeds the A/B prior.
- **D2 — ratified end-state (a); sequencing amended**: design doc + risk-officer sign-off
  proceed now, but implementation is timed to precede the first second-live-symbol or
  portfolio-sizing change rather than blocking the single-symbol parity milestone —
  current prod reality is single-position, so P4 replays of real sessions do not yet
  require it. Interim banner-clamp stands, not as end state.
- **D3 — ratified as written** (design the financing interface, defer funding-rate
  implementation with a ledger row; zero live dollars touched today).
- **D4 — ratified, bundled**: T₁/T₂ thresholds ratify in the SAME Board sitting as the
  outstanding risk-limits.json corrections (max_position_size_pct 0.10→0.20 +
  $last_reviewed stamp). Breach action = charter breach_action (halt entries + page),
  wired through the alert-monitor/ALERT_WEBHOOK path — never just a filed issue.
- **Reporting**: A/B results and the P4.2 weekly parity audit publish through the frozen-
  exam/scoreboard system (docs/architecture/model_evaluation_system.md) — one append-only
  evidence discipline for both model and parity claims.

**D1 mechanism correction (2026-07-06, flip-rate study)**: the feature window is exclusive
of the forming bar — the model's prediction is FIXED within the hour; decisions flip because
the live reference price (predicted_return denominator) mutates tick-by-tick. The gate is
therefore "freeze the decision reference price to the closed bar," not "avoid OOD features"
(the adoption note above overstated the OOD mechanism). Study also shows neither mode beats
coin-flip on H+1 direction: the gate removes churn/whipsaw and buys input parity; it does not
claim edge. Evidence: docs/research/experiments/2026-07-06_forming-bar-fliprate.md.
