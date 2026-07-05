# Adversarial Architecture Review — Backtest ↔ Live Parity Plan

> **Reviewer:** architecture / trading-system-safety lens (read-only; no source changed).
> **Target:** `docs/refactor/backtest_live_parity_plan.md` (DRAFT v1, 2026-06-15).
> **Base:** git worktree `backtest-live-parity` on top of #486 live-engine modularization.
> **Date:** 2026-07-05. Every citation below was verified against code in this worktree.

Findings are ranked most-severe first. Severity: **BLOCKER** (invalidates a P0 claim / must be
resolved before the plan is executed), **MAJOR** (significant architectural or safety gap that
needs a plan revision), **MINOR** (over-claim, mis-citation, or scope nit).

---

## 1. [BLOCKER] The P0 harness cannot exercise live's concurrency — no clock seam exists, and a single-threaded SimulatedExchange serializes away exactly the races that cause live-only bugs

**Rationale:** P0.1/§L2 rest on two premises that are both false against the code: (a) that a
`LiveLoopTimingCoordinator` "clock seam" makes the loop "time-warpable … no wall-clock sleeps,"
and (b) that a `SimulatedExchange` driven by a scripted candle feed can faithfully exercise the
live engine. It cannot reproduce the multi-threaded reality.

**Evidence — there is no injectable clock:** `LiveLoopTimingCoordinator`
(`src/engines/live/loop_timing.py:46`) is a *cadence/freshness helper*, not a clock. It calls
`time.time()`, `time.sleep()` and `datetime.now(UTC)` directly (`loop_timing.py:56,58,61,71,88,113,132`).
The trading loop sleeps on `stop_event.wait(seconds)` (wall-clock; `trading_engine.py:1481+`), and
the periodic reconciler sleeps on `time.sleep(1)` (`reconciliation.py:3234`). Nothing here accepts
an injected clock — "time-warpable via the coordinator seam" is not supported by the current code.

**Evidence — live is genuinely multi-threaded, and mutations happen OFF the loop:** the engine
spawns ≥6 long-lived threads: main loop (`startup.py:470`), WS-health (`ws_health.py:195`),
order-tracker poll (`order_tracker.py:201`), UserDataProcessor (`user_data_processor.py:15`),
periodic reconciler (`reconciliation.py:3190`), plus fire-and-forget alert threads
(`trading_engine.py:2318`). Position/balance/order state is mutated *off the trading-loop thread*:
order-tracker fill callbacks fire "outside any lock" on the poll thread
(`order_tracker.py:446,454`), WS user-stream fills dispatch on the UserDataProcessor thread
(`user_data_processor.py:73`), and the reconciler books trades and mutates balance on its own
thread (`reconciliation.py:1111,2146,2661,2813`). A scripted-clock, single-threaded driver
executes all of this in one deterministic order — the fill-vs-loop-drain interleave, the
WS-poll-vs-user-stream race on the same order (`order_tracker._order_locks`, line 126), and the
reconciler-vs-fill race (`_position_mutation_locks`, `reconciliation.py:3160`) simply never occur.
The harness would be **green while the live-only bug classes it is meant to catch (#741 reject,
#710 margin-reserve, partial-fill, cancel race) remain invisible.**

**Suggested plan revision (§P0.1, §6 L2):** Split the claim explicitly. (1) Rename what the harness
proves from "engine-level parity" to **"driver-level determinism parity"** — real financial
*calculations* run through both engines, single-threaded, on identical candles. That is valuable and
achievable. (2) Add a *separate* prerequisite workstream, **P0.0 "inject a Clock"**: introduce a
`Clock` protocol (`now()`, `sleep()`) threaded through `loop_timing`, the reconciler, and the
order-tracker poll loop, defaulting to a real-time impl. Without it, "no wall-clock sleeps in tests"
is unattainable and P0 will either hang on real sleeps or fake the timing. (3) State plainly in §9
that concurrency races are **out of scope for the equality harness** and belong to a distinct
thread-fuzzing/fault-injection suite (see Finding 4) — do not let §L2's "exact equality end-to-end"
imply the threading model is covered.

---

## 2. [BLOCKER] "Byte-exact TradeRecord equality" between the real live engine and backtest is unachievable as written, and the P0.2 normalization tuple is under-specified in a way that will either fail spuriously or mask real divergence

**Rationale:** §1 ("Provable exactly"), §L2 and §L3(i) promise *exact* `TradeRecord` equality
between a run of the **real live engine** and the backtest. Live mints non-deterministic identifiers
and wall-clock-derived fields that backtest cannot match; the canonical tuple in P0.2 must exclude
them — but the excluded fields are load-bearing, and excluding them silently removes real signal.

**Evidence:** paper order ids are `f"paper_{int(time.time()*1000)}"`
(`execution_engine.py:452,658`); live entry ids are
`f"atb_{timestamp_ms:x}_{uuid.uuid4().hex[:8]}"` (`execution_engine.py:746-748`) and exit ids
`f"atbx_{timestamp_ms:x}_{uuid.uuid4().hex[:8]}"` (`execution_engine.py:897-899`). These are pure
wall-clock + `uuid4` — never reproducible in a candle-driven backtest. The canonical tuple already
omits `order_id`, which is correct; but it *includes* `entry_ts`/`exit_ts`, and the live path's
timestamps are driven by loop wall-clock, whereas backtest's come from the candle index
(`exit_handler.py:173-174,188`). To make them comparable the harness must force the live clock to
the candle clock — which requires the Clock seam that does not exist (Finding 1). Absent that,
timestamps will differ and the "exact equality" check is a fiction.

**The masking risk is the real danger.** Once you normalize away `order_id` *and* timestamps *and*
whatever else refuses to match, the remaining tuple *(symbol, side, entry_px, qty, exit_px, reason,
fees, slippage, interest, gross/net pnl, balance)* can be byte-equal while a genuine divergence
hides in a field the normalizer dropped (e.g. a stop-order that filled at a *different time* within
the bar in live vs a candle-close in backtest — same price, same pnl, different exit_ts, silently
equal).

**Suggested plan revision (§P0.2, §1, §6 L2/L3):** Downgrade the language from "byte-exact
equality" to **"exact equality on a pinned economic-field set, with excluded fields enumerated and
justified in the doc."** Specifically: (a) list every excluded field (`order_id`,
`client_order_id`, `exchange_order_id`, and — until the Clock seam lands — the raw timestamps),
each with a one-line reason; (b) replace raw `entry_ts`/`exit_ts` equality with **candle-index
equality** (which bar the entry/exit booked on) so timing divergence is still caught at candle
resolution rather than silently normalized out; (c) add a positive assertion that no *other* field
is dropped — the normalizer is a closed, reviewed allowlist, not a "strip whatever differs" helper.

---

## 3. [MAJOR] The P3.2 import-linter "shared owns X" contract is violated by the shared layer *today* — `engines/shared` already reaches down into both `engines/backtest` and `engines/live`, and backtest imports live

**Rationale:** P3.2 proposes a CI contract that `engines/backtest/**` and `engines/live/**` may not
define "owned-by-shared" symbols, framed as making the shared core a clean upper layer. But the
actual layering is already inverted in places, so the guard as described would either fail on day
one or require moving code the live engine cannot cleanly give up — the plan does not acknowledge
this.

**Evidence:** `src/engines/shared/execution/snapshot_builder.py` imports **both**
`from src.engines.backtest.models import ActiveTrade` and
`from src.engines.live.execution.position_tracker import LivePosition, PositionSide`
(`snapshot_builder.py:19-20`, and again at `:193,215`). These are `TYPE_CHECKING`-guarded — but
import-linter analyzes `TYPE_CHECKING` imports by default, so a naïve "shared must not depend on
live/backtest" contract flags them immediately. Separately, backtest imports live at runtime:
`engines/backtest/engine.py:565-566` imports `RegimeStrategySwitcher` and `StrategyManager` from
`src.engines.live` (also `backtest/regime/regime_handler.py:21-22`). So the dependency graph today is
`backtest → live` and `shared → {backtest, live}`, the opposite of the plan's clean stack.

**Suggested plan revision (§P3.2, add a step to §P1):** Make the layering explicit and sequence the
guard *after* the coupling is removed, not before. Add a prerequisite: (a) invert
`snapshot_builder`'s dependency — define the position/trade shapes it needs as a `Protocol` in
`shared` and have both drivers satisfy it, so shared stops importing down; (b) move
`RegimeStrategySwitcher`/`StrategyManager` (currently in `live`, consumed by `backtest`) into
`shared` or a neutral module, or the `backtest → live` edge stays. Only then enable the
import-linter contract, and scope its first version to `--ignore` the known remaining edges with
tracking issues rather than presenting it as a clean green gate. Also note import-linter is not
currently a dependency (absent from `setup.cfg`/requirements) — P3.2 adds a new tool + config.

---

## 4. [MAJOR] The harness proves only the happy path — none of live's hardest bug classes (reject, partial fill, cancel race, margin-reserve, phantom-order UNKNOWN) are in the P0 scenario matrix except as passive ledger rows

**Rationale:** Principle §"Assume Malicious Markets" / CODE.md "Planning Complex Features" both
demand enumerating failure scenarios (timeout, partial fill, reject, crash-mid-op, external close).
The P0.3 matrix is entirely clean-execution scenarios; the failure modes are pushed to the §7 ledger
as "edge events … out of scope for sharing." But these are the exact paths that lost real capital
(capital-erosion postmortem; #741 order-reject; #710 margin-reserve) and where backtest most
dangerously over-promises returns.

**Evidence the code has rich failure handling worth testing:** `_execute_live_order` distinguishes
definitive reject (`ValueError` → mark FAILED, no position; `execution_engine.py:779-801`) from
ambiguous `None` (→ mark UNKNOWN, return client_order_id to block dupes;
`execution_engine.py:803-823`) — the phantom-order window. Partial fills are first-class
(`_is_filled_status` accepts `PARTIALLY_FILLED`, `execution_engine.py:199-207`; journal stays
SUBMITTED until fully filled, `:837-841`). `_normalize_quantity` rejects on `min_qty`/`min_notional`
(`:1035-1054`). The reconciler books balance-neutral external-close rows
(`reconciliation.py:2369-2375`). A `SimulatedExchange` implementing `ExchangeInterface` is the
*ideal* place to inject these — it is the seam through which every one of these paths runs.

**Argument for adding it (not against):** the marginal cost is low because the harness already has to
implement the full `ExchangeInterface` (`place_order`, `place_stop_loss_order`, `get_order`,
`cancel_order`, balances). Making fills *scriptable per order* (this order rejects; this one fills
40% then the rest next poll; this `place_order` returns `None`) is a small extension of a component
being built anyway, and it is the only way to prove the live engine's failure-recovery code without
a real exchange. Leaving it out means the plan's flagship safety artifact never touches the code
that actually loses money.

**Suggested plan revision (§P0.1 and §P0.3):** Add to `SimulatedExchange` a **fault-injection
script** — a per-`client_order_id` policy of `{fill_full | partial(fraction) | reject(reason) |
return_none_ambiguous | cancel_before_fill}`. Add four scenarios to P0.3: (i) entry rejected by
exchange (`ValueError`) → assert no position, FAILED journal; (ii) `place_order` returns `None` →
assert UNKNOWN journal + phantom-block, then reconciler resolves; (iii) partial entry fill →
assert position sized to filled qty, journal SUBMITTED; (iv) stop-loss `place_stop_loss_order`
returns a bare id then `get_order` shows fill (note the return-type asymmetry: `place_order` →
`Order | None` but `place_stop_loss_order` → `str | None`, `exchange_interface.py:242` vs `:285` —
the harness must model the follow-up `get_order`). These assert *live-only* behavior, so they run
against the live driver only (backtest has no exchange to reject), which is fine — the ledger then
records the residual, but the *code path* is now covered by a test.

---

## 5. [MAJOR] Sequencing risk: P0 builds the SimulatedExchange against the live `ExchangeInterface`, but P1/P3 reshape the surfaces the harness binds to — the "measure before you refactor" instrument will need rework mid-flight

**Rationale:** §"Ordering principle: build the measuring instrument before touching the thing
measured" is sound in spirit, but the instrument (P0) is coupled to two things that later phases
explicitly change: (a) the `ExchangeInterface` surface, and (b) the per-engine handler internals the
equality harness asserts against. The plan treats P0 as a stable foundation; it is partly built on
sand that P1/P2/P3 stir.

**Evidence:** P1.1-P1.4 replace both engines' exit/entry/partial handlers with shared workflows —
so the intermediate states the P0 harness observes (and the divergence-report field diff, P0.4)
shift underneath it. P2.1 (`ExchangeRules`) and P2.3 (multi-position `PositionTracker`) *deliberately
change backtest outputs* (P2.1 note: "intentionally changes backtest results"; the backtest tracker
is single-slot today — `current_trade: ActiveTrade | None`, `position_tracker.py:64`). So the
"golden fixtures" pinned in P0.3 must be re-baselined at P2.1, P2.3, and P2.4 by design. And the
`SimulatedExchange` binds to `ExchangeInterface`, whose `place_stop_loss_order`/`get_order` shape is
in scope for the stop-order-tier work (P2.4). This is not fatal — but "measure first, then refactor"
implies the measurement is stable, and it is not.

**Suggested plan revision (§4 preamble, §8 dependency table):** Reframe P0 as **two layers**: a
*stable* layer (balance-conservation invariants, L3 property tests, the divergence-report *format*)
that survives refactors, and a *volatile* layer (golden fixtures with exact values) that is
explicitly expected to re-baseline at each fidelity-improving phase. In §8, add the missing edges:
P0.1 `SimulatedExchange` → depends on a *frozen* `ExchangeInterface` (declare it frozen for the
duration, or accept P0 rework at P2.4); P0.3 golden fixtures → re-baseline gates at P2.1/P2.3/P2.4.
Consider building `SimulatedExchange` against a **thin adapter** you own rather than binding directly
to `ExchangeInterface`, so P2.4's interface changes touch one adapter, not the whole harness.

---

## 6. [MAJOR] `PartialExitExecutor` is mischaracterized as a "stub," and it computes fees with its own `fee_rate` rather than delegating to `CostCalculator` — a genuine parity hazard the plan misses while chasing a non-problem

**Rationale:** §2 calls `partial_exit_executor.py` a "stub underused" and P1.4 says "finish the
stub." That is factually wrong and, worse, it points effort at the wrong risk. The class is fully
implemented and *already wired into the backtest tracker*; the real hazard is that it duplicates fee/
slippage math instead of using the shared `CostCalculator`, which is precisely the drift the whole
plan exists to kill.

**Evidence:** `partial_exit_executor.py` is 205 lines of complete P&L/fee/slippage logic
(`execute_partial_exit`, `:91-176`), and the backtest `PositionTracker` already imports and uses it
(`position_tracker.py:23`). But it computes costs locally: `exit_fee = abs(exit_notional *
self.fee_rate)` and `slippage_cost = abs(exit_notional * self.slippage_rate)`
(`partial_exit_executor.py:154-155`) — a *second* implementation of fee/slippage, parallel to
`CostCalculator.calculate_exit_costs` (`cost_calculator.py:162`). CODE.md "Backtest-Live Parity":
"Never duplicate financial logic — use `src/engines/shared/` … (`cost_calculator`)." This is a live
CODE.md violation the plan should target, not a stub to finish.

**Suggested plan revision (§2 table and §P1.4):** Correct the description from "stub underused" to
"implemented but computes fees/slippage independently of `CostCalculator`." Rescope P1.4 from
"finish the stub" to **"route `PartialExitExecutor`'s cost math through `CostCalculator` and wire
both engines' partial paths through the executor."** Add a parity test that a full exit executed as
one 100% partial slice equals a normal exit through `CostCalculator` to the cent — that pins the two
fee paths together.

---

## 7. [MAJOR] §1 claims backtest models "every economically meaningful thing," but bid-ask spread (paid on every market fill) and perp funding are modeled by neither engine and are absent from the §7 ledger seed

**Rationale:** §1 enumerates "fees, slippage, exchange rounding, margin interest, stop-order
mechanics, partial operations" as the economically meaningful set. For a 24/7 crypto system trading
market orders, **bid-ask spread is a real per-fill cost** and (for any perpetual/futures venue)
**funding** is a real carry cost. Neither is in `CostCalculator`, and neither appears in the ledger's
seed rows — so the plan's own "every economically meaningful thing is modeled or ledgered" invariant
is already breached on the modeling side, not just the environmental side.

**Evidence:** `cost_calculator.py` contains no spread/bid/ask/funding logic (grep across the file:
zero hits for `spread|bid|ask|funding`); nothing in `engines/shared/` references funding. The §7
seed list names intrabar ticks, partial fills, latency, cadence, sentiment, reconciler closes,
outages — but not spread or funding. Slippage (`slippage_rate`) is modeled as a symmetric adverse
move, which is not the same as the half-spread crossed on entry *and* exit.

**Suggested plan revision (§1, §7, §P2):** Either (a) add spread and funding as explicit ledger rows
(`id | description | bias | quantified impact | bound | monitor | status`) with the fee-truth
reconciliation (P4.3) measuring realized spread from `orders.actual_commission`-adjacent fill data,
or (b) if a future phase will model half-spread inside `CostCalculator`, name it as a P2 item. What
the plan must not do is claim §1 completeness while two first-order market costs are silently
unmodeled and unledgered.

---

## 8. [MINOR] Over-claim: "byte-exact, CI-enforced equality" for accounting parity ignores float non-associativity across two independently-ordered summation paths

**Rationale:** §1(3) and §5 promise the determinism fingerprint stays "byte-identical" and accounting
evolves "identically." Even with identical inputs and shared cost code, the live driver and backtest
driver accumulate balance through *different call sequences* (live: async fill callbacks + reconciler
adjustments; backtest: in-loop). IEEE-754 addition is not associative, so `a+b+c` booked in a
different order can differ in the last ULP. CODE.md itself mandates `pytest.approx` for financial
calculations and epsilon tolerance for float comparison — "byte-exact" contradicts the house rule.

**Evidence:** CODE.md "Tests": "Use `pytest.approx` for financial calculations, not exact float
equality"; "Arithmetic": "Use epsilon tolerance for float comparisons." The live path applies fee
deltas incrementally via `_adjust_cost_totals` on out-of-order fills (`execution_engine.py:430,588`),
a different accumulation order than backtest's.

**Suggested plan revision (§1, §5, §6):** Replace "byte-exact equality" with **"equality within a
declared ULP/epsilon tolerance"** for monetary aggregates, reserving exact equality for integer/
identifier/enum fields and discrete decisions. Keep the determinism *fingerprint* byte-exact only for
a *single engine's* reproducibility (L0, which is legitimately exact); cross-engine (L2) should be
epsilon-based, consistent with CODE.md.

---

## 9. [MINOR] Line-reference drift in §2 and the docstring citation will rot; the plan hard-codes many `file:NNN` refs that are already slightly off

**Rationale:** The plan is dense with exact line citations, several of which have already drifted in
this worktree — a maintenance hazard for a doc meant to guide multi-session execution.

**Evidence:** plan cites the "Known parity caveats" docstring at `backtest/engine.py:135-158`; it is
actually at `:141-162`. Plan says `engine.py` is "1,820 lines"; it is 1,898. Plan cites live
execution delegation at "`execution_engine.py:241,338`"; the live entry cost call is at `:367` in
this worktree. The backtest delegations (`:159,245,308`) and the uuid/timestamp sites
(`:452,658,746-748,897-899`) *do* verify — so the refs are mostly right, which makes the stale ones
more misleading.

**Suggested plan revision (throughout):** Cite by **symbol** (`Backtester` docstring;
`LiveExecutionEngine._execute_live_order`; `PositionTracker.current_trade`) rather than line number,
or add a "verified against commit `<sha>`" stamp so readers know line numbers are point-in-time.

---

## 10. [MINOR] P0.1 implies a greenfield `SimulatedExchange`, but stateful/`Mock`-based exchange doubles already exist in the test tree and should be reconciled, not duplicated

**Rationale:** Building a third, parallel notion of "fake exchange" risks the same duplication the
plan is trying to eliminate — this time in test infrastructure.

**Evidence:** `tests/integration/live/test_reconciliation_integration.py:42` defines
`MockExchangeOrder` and drives `mock_exchange.place_order/get_order/get_order_by_client_id`
extensively; `tests/integration/live_trading/test_engine_core.py:112` wires a `MockExchange`. These
are per-test `Mock`s with no stateful fill engine — so `SimulatedExchange` is genuinely new (a
stateful, `OHLCFillModel`-backed implementation), but the plan should say it *supersedes/absorbs*
these doubles rather than adding a fourth pattern.

**Suggested plan revision (§P0.1):** Add a sentence: "the `SimulatedExchange` becomes the single
exchange test-double; migrate `MockExchangeOrder`/`MockExchange` call sites to it (or a thin
factory over it) so there is one fake-exchange implementation, not several."

---

## Verified Correct (parts of the plan that hold up against the code)

- **Shared-core inventory (§2 "Already unified") is accurate.** `engines/shared/` contains
  `cost_calculator.py`, `models.py`, `dynamic_risk_handler.py`, `correlation_handler.py`,
  `trailing_stop_manager.py`, `strategy_exit_checker.py`, `partial_operations_manager.py`,
  `validation.py`, `side_utils.py` — as claimed. Both engines' execution engines delegate cost/fill
  to `CostCalculator` (backtest `execution_engine.py:159,245,308`; live `execution_engine.py:367`).
- **The duplication table (§2) is real.** Backtest `exit_handler.py` (876 lines) contains inline
  SL/TP high/low fill detection (`~:546-625`) that mirrors the live exit path — held equal only by
  tests, exactly as the plan says. P1.1's `ExitTriggerEvaluator` targets a genuine hazard.
- **The single-position structural gap (§2 #3, P2.3) is verified and correctly flagged as the
  largest item.** Backtest `PositionTracker` holds `self.current_trade: ActiveTrade | None`
  (`position_tracker.py:64`, `POSITION_KEY = "active"`), while live holds N keyed by order_id and
  honors `get_max_concurrent_positions()` (`entry_coordinator.py:524`, `trading_engine.py:1608`).
  Recommending option (a) *sequenced last, behind its own design doc* is the right call.
- **The exchange-rounding gap (§2 #1, P2.1) is verified.** Live applies `step_size` rounding +
  `quantize_to_step` + `min_qty`/`min_notional` rejection (`execution_engine.py:1005-1054`); the
  backtest path calls `quantize_to_step` **nowhere** (zero hits under `engines/backtest/`) and uses
  raw floats. P2.1 addresses a real divergence, and gating it behind a flag with an A/B report before
  flipping defaults (§5) is the correct discipline.
- **The margin-interest asymmetry (§2 #2) is real.** Backtest defaults
  `annual_margin_interest_rate = 0.0` (silently optimistic for shorts), documented in the caveats
  docstring; live queries the exchange. P2.2's shared `FinancingCostProvider` + calibration job is
  the right shape.
- **Layered proof system (§6) and the parity-gap ledger (§7) are the right architecture** for a
  domain where exact reproduction is impossible — the honest framing in §9 ("anything stronger is
  marketing") is correct and should be preserved. The findings above tighten *which* claims can be
  called "exact"; they do not dispute the layered approach.
- **Existing parity test surface is as described.** `tests/integration/parity/` contains
  `test_side_by_side_parity.py`, `test_engine_parity.py`, `test_backtest_determinism.py`; the
  side-by-side suite already instantiates `LiveTradingEngine` (at the calculation/method level with
  `Mock`s) — so P0.2's "extend from handler-level to engine-level" starts from a real base, even
  though the end-to-end leap is larger than "extend" implies (Finding 1).

---

## Bottom line

The plan's *direction* is right — collapse the duplicated exit/entry/partial logic into shared
workflows, model the known backtest gaps (rounding, interest, multi-position), and prove it with
layered tests. The **fidelity gaps in three "Provable exactly" claims are the problem**: (1) the P0
harness cannot exercise live concurrency because no clock seam exists and a single-threaded driver
serializes the races (Finding 1); (2) "byte-exact TradeRecord equality" against a uuid/wall-clock-
minting live engine is unachievable and, as normalized, risks masking divergence (Finding 2); and
(3) the P3.2 layering guard is already violated by `shared → {live, backtest}` and `backtest → live`
edges (Finding 3). Add fault injection to the one component built to carry it (Finding 4), fix the
sequencing so the harness isn't rebuilt mid-refactor (Finding 5), and correct the
`PartialExitExecutor` and spread/funding characterizations (Findings 6-7). With those revisions the
plan is executable; as written, its headline safety guarantees over-promise.
