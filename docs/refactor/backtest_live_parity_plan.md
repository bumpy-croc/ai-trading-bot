# Backtest ↔ Live Parity: Unification Plan

> **Status:** DRAFT v1 (pending adversarial review — see §10)
> **Goal owner:** human. **Author:** Claude Code session `0188LNSixYW9Fa5hrJ7YWJoa` (2026-06-15).
> **Prereq:** the live-engine slim-down in `docs/refactor/live_engine_modularization.md` (finish first — shared extractions should pull from tidy code).
> **Branch base:** `develop`.

---

## 1. Goal and what "parity" means operationally

**Ultimate goal:** the backtest engine is a faithful predictor of the live engine. Every
economically meaningful thing that happens in live — fees, slippage, exchange rounding,
margin interest, stop-order mechanics, partial operations — is either (a) **the same code**
in both engines, or (b) **explicitly modeled** in backtest with a **measured, bounded,
monitored** difference.

"100% confidence" cannot mean "identical outcomes" — live trading has irreducible
environmental facts (latency, intrabar tick paths, order-book liquidity, exchange outages)
that no candle-level backtest can reproduce. It **can** mean:

1. **Decision parity** — given identical market data, both engines make identical
   entry/exit/sizing decisions. *Provable exactly* (same code + differential tests).
2. **Execution parity** — given identical fills-physics, both book identical prices, fees,
   slippage, quantities, P&L. *Provable exactly* (shared cost/fill model + equality tests).
3. **Accounting parity** — balances, fee legs, interest legs, trade records evolve
   identically. *Provable exactly.*
4. **Environmental fidelity** — the residual live-vs-model gap (real ticks vs candles,
   real borrow rates vs modeled, latency) is *quantified continuously* against production
   data and stays inside ratified bounds. *Provable statistically, not exactly.*

The plan drives 1–3 to **byte-exact, CI-enforced equality** and turns 4 into a
**monitored number** instead of an unknown.

---

## 2. Current state (audited 2026-06-15)

### Already unified (single shared implementation, both engines call it)
- **Fees, slippage, fill price** — `src/engines/shared/cost_calculator.py` (entry
  `:110-160`, exit `:162-214`), `execution/ohlc_fill_model.py`, `fill_policy.py`,
  `execution_model.py`. Both engines' execution engines delegate here
  (backtest `execution_engine.py:159,245,308`; live `execution_engine.py:241,338`).
- **Models & P&L** — shared `models.py` (`Position`, `Trade`, `pnl_percent`).
- **Entry-plan extraction + dynamic-risk sizing** — `SharedEntryHandlerMixin`
  (byte-identical by construction), `dynamic_risk_handler.py`.
- **Trailing stops, strategy-exit detection, partial-op *decisions*, correlation control,
  risk-config merging, validation, side utils** — all in `engines/shared/`.

≈65–70% of trading logic is shared. A real parity test suite exists
(`test_backtest_live_parity.py`, `fee_accounting_parity`, `exit_handler_parity`,
`sentiment_freshness_parity`, side-by-side integration tests, determinism fingerprint).

### Duplicated (same concept, two implementations — drift risk)
| Area | Backtest | Live |
|---|---|---|
| Entry orchestration | `backtest/execution/entry_handler.py` | `live/execution/entry_handler.py` + `entry_coordinator.py` |
| Exit orchestration (SL→TP→trailing→strategy→time→partial ordering) | `backtest/execution/exit_handler.py` | `live/execution/exit_handler.py` + `exit_coordinator.py` |
| SL/TP high/low fill detection | inline in each exit handler | (kept in sync only by tests) |
| Partial exit/scale-in **execution** | inline | `_execute_partial_exit()` (shared `partial_exit_executor.py` stub underused) |

### Live-only realities NOT modeled (or optionally modeled) in backtest
Documented in `backtest/engine.py:135-158` ("Known parity caveats") and found in audit:
1. **Exchange quantity/price quantization** — live rounds to `step_size`/`tick_size` and
   enforces `min_qty`/`min_notional` (`live/execution/execution_engine.py::_normalize_quantity`;
   see also `.claude/LESSONS.md` §1.1 — precision bugs come in pairs); backtest uses raw
   floats.
2. **Margin/borrow interest** — live queries the exchange (`margin_interest_tracker.py`);
   backtest models via `annual_margin_interest_rate` (default **0.0** — silently optimistic
   for margin strategies). Live does not stash interest to trade metadata the way backtest
   does; acquisition methods and defaults diverge.
3. **Single- vs multi-position** — backtest `PositionTracker` holds **one** `ActiveTrade`;
   live holds N (`max_concurrent_positions`). A multi-position strategy is structurally
   unrepresentable in backtest today. **Largest structural gap.**
4. **Stop-loss as a real resting order** — live SL can fill mid-bar at the exchange, can
   partially fill, can be cancelled/rejected (#741), reserves margin balance (#710);
   backtest checks the SL level once per candle against high/low.
5. **Partial-op cadence** — live evaluates every loop tick; backtest once per candle.
6. **Sentiment freshness** — live overlays real-time sentiment (4h window); backtest is
   all-historical (`sentiment_freshness=0`). Column-parity locked; value distributions
   differ by design.
7. **Reconciliation / recovery / order tracking** — live-only by nature (no broker to
   diverge from in backtest). Correctly out of scope for sharing; their *economic effects*
   (e.g. a reconciler-booked external close) are edge events, tracked in the ledger (§7).

---

## 3. Target architecture: thin drivers around a shared execution core

```
                 ┌───────────────────────────────────────────────────┐
                 │            src/engines/shared/  (the core)         │
                 │                                                    │
                 │  CostCalculator · OHLCFillModel · FillPolicy  ✅   │
                 │  Models/PnL · DynamicRisk · Correlation       ✅   │
                 │  TrailingStops · StrategyExitChecker          ✅   │
                 │  PartialOpsManager (decisions)                ✅   │
                 │  ──────────── NEW in this plan ────────────        │
                 │  ExchangeRules (stepSize/tickSize/minQty/          │
                 │    minNotional + quantize_to_step)            🆕   │
                 │  ExitTriggerEvaluator (one high/low fill fn,       │
                 │    one SL/TP same-bar tie-break policy)       🆕   │
                 │  SharedExitWorkflow (ordered pipeline)        🆕   │
                 │  SharedEntryWorkflow (plan→risk→corr→size→         │
                 │    rules→cost→intent)                         🆕   │
                 │  PartialExitExecutor (finish the stub)        🆕   │
                 │  FinancingCostProvider (margin interest,           │
                 │    pluggable: queried vs modeled)             🆕   │
                 └────────────▲──────────────────────▲───────────────┘
                              │                      │
              ┌───────────────┴───────┐   ┌──────────┴────────────────┐
              │  Backtest driver      │   │  Live driver              │
              │  candle iteration     │   │  real-time loop, WS       │
              │  portfolio state      │   │  exchange I/O, SL orders  │
              │  (multi-position 🆕)  │   │  reconciliation, recovery │
              └───────────────────────┘   └───────────────────────────┘
```

An engine driver may **feed** the core (data, clock, portfolio state) and **effect** its
decisions (simulated fill vs real order), but never re-implement a trading decision or a
cost computation. Enforced by CI (§6, guard G).

---

## 4. Workstreams

**Ordering principle: build the measuring instrument before touching the thing measured.**

### Phase P0 — Parity harness first (measure, then refactor)
- **P0.1 `SimulatedExchange`**: an implementation of the live `ExchangeInterface`
  (`place_order`, `place_stop_loss_order`, `get_order`, `get_open_orders`, `cancel_order`,
  balances…) whose fill physics are the **shared** `OHLCFillModel` + `ExchangeRules`, driven
  by a scripted candle feed and an **injected clock** (the `LiveLoopTimingCoordinator` seam
  makes the loop time-warpable; no wall-clock sleeps in tests).
- **P0.2 Side-by-side equality harness**: run the *real live engine* (paper mode +
  `SimulatedExchange`) and the backtest engine over identical candles; normalize both
  outputs to a canonical `TradeRecord` tuple *(symbol, side, entry_ts, entry_px, qty,
  exit_ts, exit_px, reason, fee_entry, fee_exit, slippage, interest, gross_pnl, net_pnl,
  balance_after)*; assert **exact equality** of the sequences. Extends the existing
  `tests/integration/parity/test_side_by_side_parity.py` from handler-level to
  engine-level.
- **P0.3 Scenario matrix** (each a golden fixture): trend→TP exit; SL via low breach;
  **SL and TP inside one bar** (tie-break policy pinned); **gap open through SL** (fill at
  open, not SL price); trailing ratchet→trigger; time exits (max-holding / end-of-day /
  weekend); partial-exit ladder + scale-in; short with N-day margin-interest accrual;
  entry rejected by `min_notional` after quantization; maker vs taker fee legs;
  same-bar-entry protection; multi-position interleaving (added in P2.3).
- **P0.4 Divergence report format**: when equality fails, the harness emits a per-trade
  field-level diff (first divergent field, both values, candle index) — debugging tool and
  later the production-replay report (§P4).

*Exit criterion: harness red/green on today's code; every currently-known divergence either
reproduced by a scenario or explicitly ledgered (§7).*

### Phase P1 — Collapse duplication (decision parity by construction)
- **P1.1 `ExitTriggerEvaluator`**: single shared function family for SL/TP/liquidation
  trigger detection from candle high/low incl. the same-bar tie-break and gap-fill rules.
  Both exit handlers call it. (The current duplicated logic is only held equal by tests.)
- **P1.2 `SharedExitWorkflow`**: one ordered pipeline (SL → TP → trailing → strategy →
  time → partial) with per-engine adapters. Backtest's monolithic `exit_handler.py` and
  live's `exit_handler.py` become thin.
- **P1.3 `SharedEntryWorkflow`**: plan → dynamic-risk → correlation → sizing →
  ExchangeRules quantization → cost → `OrderIntent`. Both entry paths drive it.
- **P1.4 `PartialExitExecutor`**: finish `shared/partial_exit_executor.py`; both engines
  execute partials through it (decision logic already shared).
- Discipline: same verbatim/parity method as #486 — AST-assisted moves, fingerprint +
  harness byte-identical per PR, dual review on money paths.

### Phase P2 — Model live realities in backtest (execution & accounting parity)
- **P2.1 `ExchangeRules`** (shared): symbol filters (`step_size`, `tick_size`, `min_qty`,
  `min_notional`) + `src/trading/precision.quantize_to_step`. Live sources it from
  `exchangeInfo` (as today); backtest from a recorded/static filter set per symbol
  (committed fixture, refreshable via a CLI). **Backtest applies the same rounding and
  rejections as live.** Removes caveat #1. *Note: this intentionally changes backtest
  results (more accurate); gated behind a config flag for one release with an A/B report,
  then default-on.*
- **P2.2 `FinancingCostProvider`**: one interface, two impls — live queries the exchange;
  backtest accrues from a configured rate curve. **Both stash identically to trade
  metadata** (extend the entry-fee metadata pattern), one event-logger read path. Add a
  **calibration job**: compare modeled vs exchange-reported interest on real sessions;
  auto-file drift beyond threshold.
- **P2.3 Multi-position backtest** ⚠️ *largest item; own design doc before code.* Upgrade
  backtest `PositionTracker`/portfolio to a dict of positions honoring
  `max_concurrent_positions`, correlation caps, per-position SL orders — reusing live's
  semantics via the shared workflows. **Decision point for the human:** (a) full
  multi-position backtest (correct, ~3–5 PRs, touches portfolio accounting throughout), or
  (b) explicitly scope parity claims to single-position strategies and have the harness
  *enforce* `max_concurrent_positions == 1` when comparing. (a) is the only path to
  "backtesting is representative" for multi-symbol; recommend (a), sequenced last.
- **P2.4 Stop-order semantics tier** (config: `execution_fidelity`): default candle-level
  (today, conservative tie-breaks); optional "resting-order" mode that models SL as an
  order that can fill at the stop price intrabar with the same
  reserve/cancel-before-close rules as live (#710) — closing caveat #4 to the extent
  candles allow. Partial-fill modeling explicitly **out of scope** initially (ledgered).
- **P2.5 Cadence alignment**: quantify the live-ticks-vs-per-candle partial-op gap (#5) in
  the ledger; optionally add sub-candle evaluation points to backtest later. Not
  parity-critical for candle-driven strategies (live decisions are gated per-candle too).

### Phase P3 — Thin drivers + drift-proofing
- **P3.1** Backtest `engine.py` (1,820 lines) reduces to: data iteration, portfolio state,
  calls into shared workflows, reporting. Live engine (post-#486 slim-down): loop + I/O +
  reconciliation around the same workflows.
- **P3.2 Architectural guard**: import-linter/CI contract — `engines/backtest/**` and
  `engines/live/**` may not define symbols in the "owned-by-shared" namespace (fees,
  fill detection, sizing, workflows); plus a grep-gate for `round(`-near-`step` patterns
  outside `trading/precision` (LESSONS §1.1 meta-rule).
- **P3.3** Update `docs/architecture.md` + this doc; retire the "Known parity caveats"
  docstring items as they close, moving residuals to the ledger.

### Phase P4 — Continuous proof against production (environmental fidelity)
- **P4.1 Production shadow replay**: `atb parity replay --session <id>` — replays a recorded
  live session's exact candle window + strategy/config through the backtest engine and
  emits the P0.4 divergence report (per-trade bps deltas on entry/exit/fees/interest;
  cumulative P&L delta). Requires persisting the live session's input snapshot (candles
  already cached; config already in `trading_sessions`).
- **P4.2 Scheduled parity audit** (weekly, existing daemon/workflow infra): run replay over
  the last N sessions; alert (webhook + GitHub issue, `source:parity-audit`) when any
  unexplained per-trade delta > **T₁ bps** or cumulative > **T₂ %** *(thresholds are
  placeholders — human ratifies in the ledger; suggest T₁=5 bps, T₂=0.1%/30d as openers)*.
- **P4.3 Fee-truth reconciliation**: compare `CostCalculator` modeled fees vs
  exchange-reported commissions per order (`orders.actual_commission`, unit-normalized);
  detect maker/taker misclassification and BNB-discount drift; feed corrections into fee
  config. This closes the loop between the *model* and *exchange truth* — the step that
  makes "fees in backtest" mean *actual* fees.

---

## 5. What must NOT change (safety rails)

- All refactor PRs are behavior-preserving; the determinism fingerprint
  (`tests/integration/parity/test_backtest_determinism.py`) stays byte-identical **except**
  where a phase deliberately improves fidelity (P2.1/P2.2/P2.4) — those land behind config
  flags with a before/after A/B report and a human sign-off, then flip defaults in a
  separate, loudly-labeled PR.
- Live-capital paths keep the #486 discipline: verbatim moves, dual reviewer, hardening
  follow-ups separate from extractions.
- Backtest **performance** is a feature: benchmark gate (golden 600-candle run wall-time
  budget ±20%) on every shared-workflow PR — shared code must not de-vectorize the hot loop.

---

## 6. The proof system (how we *know*, layer by layer)

| # | Layer | Mechanism | Proves | Status |
|---|---|---|---|---|
| L0 | Determinism | fingerprint test (BLAS pinned) | the oracle itself is reproducible | ✅ exists |
| L1 | Component differential | for every shared component: identical inputs → identical outputs via both engines' adapters; wiring tests pin both engines construct it with the same params | no adapter-level drift | partial → complete in P1 |
| L2 | Scenario equality | P0 harness: real live engine (paper + SimulatedExchange) vs backtest on the scenario matrix; **exact `TradeRecord` equality** | decision+execution+accounting parity end-to-end | 🆕 P0 |
| L3 | Property-based | Hypothesis: random OHLC walks × config space → invariants: (i) trade sequences equal, (ii) both satisfy balance conservation `final = initial + Σnet − Σfees − Σinterest`, (iii) no orphan trades either side | parity isn't fixture-shaped | 🆕 P0/P1 |
| L4 | Production replay | P4.1/P4.2 scheduled divergence reports vs ratified bounds | the *model* tracks *reality* | 🆕 P4 |
| L5 | Exchange-truth audit | P4.3 fee/interest reconciliation vs exchange-reported values | cost model calibrated to actual venue behavior | 🆕 P4 |
| G | Architectural guard | import contracts + precision grep-gate in CI | duplication cannot silently return | 🆕 P3 |

**CI policy:** L0–L3 are required checks on every PR touching `src/engines/**`. L4/L5 are
scheduled; their alerts create `type:incident`-adjacent issues (`source:parity-audit`).

---

## 7. Parity Gap Ledger (living table — `docs/refactor/parity_gap_ledger.md`)

Every irreducible or not-yet-closed difference gets a row:
`id | description | direction of bias | quantified impact (method + number) | bound | monitor | status`.
Seed rows: intrabar tick path vs candle SL fills; live partial fills; latency/slippage
beyond model; partial-op cadence; sentiment freshness; reconciler-booked external closes;
exchange outages/restarts mid-position. **Rule:** the harness and replay reports may only
show divergences that map to a ledger row; anything unexplained is a failing check, not a
shrug.

---

## 8. Sequencing, effort, dependencies

| Phase | PRs (est.) | Depends on |
|---|---|---|
| P0 harness + scenarios | 2–3 | live slim-down done (clock seam) |
| P1 shared workflows | 3–4 | P0 (measure first) |
| P2.1 ExchangeRules | 1–2 | P0 |
| P2.2 Financing provider | 1 | P0 |
| P2.4 stop-order tier | 1–2 | P1.1 |
| P2.3 multi-position | design doc + 3–5 | P1, human decision |
| P3 thin drivers + guards | 2 | P1, P2 |
| P4 replay + audits | 2–3 | P0.4, P2.2 |

Total ≈ **15–20 PRs** over multiple sessions. Highest risk: P2.3 (multi-position) — do it
last, behind its own design review. Fastest confidence wins: P0 + P2.1 + P4.3 (harness,
rounding, fee truth) deliver most of the "backtests don't lie about costs" goal early.

---

## 9. Honest limits (what even this plan cannot promise)

Candle-level simulation cannot see intrabar tick ordering (SL vs TP first when both are in
range — we pin a *conservative* tie-break and measure the residual in L4), real order-book
liquidity/partial fills, venue latency, or borrow-rate changes intra-session. The claim this
plan supports is: *decisions, costs, and accounting are identical by construction; the
remaining environment gap is measured weekly against production and bounded by numbers a
human ratified.* That is the strongest form of "100% confidence" that exists in this
domain; anything stronger is marketing.

---

## 10. Review log

- v1: initial draft (2026-06-15). Adversarial review pending; findings and revisions will
  be recorded here.
