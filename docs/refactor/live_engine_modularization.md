# Live Trading Engine Modularization — Plan & Handover (#486)

> **Status:** living handover doc for the ongoing `LiveTradingEngine` modularization.
> **Audience:** an autonomous agent picking this up in a fresh session.
> **Author:** Claude Code session `0188LNSixYW9Fa5hrJ7YWJoa` (2026-06-15).
>
> ⚠️ **All the refactor work described below lives on the `develop` branch.** As of
> this writing `main` is ~45 commits behind `develop`, so if you are reading this on
> `main` the engine on this branch is still the pre-refactor monolith. **Branch from
> `develop`** for any continuation work, and read the engine there.

---

## 1. Goal

Make `src/engines/live/trading_engine.py` a **thin, modular, maintainable orchestrator**
that follows good software-engineering principles, **with functionality 100% intact**.
Any change that advances modularity / maintainability / readability while preserving
behavior is in scope. This is GitHub issue **#486**.

The North Star is **backtest↔live parity**: a deterministic backtest fingerprint must stay
**byte-identical** across every change (see §4).

---

## 2. Where we are now (post-2026-06-15 session)

`trading_engine.py`: **6,558 → 2,493 lines** across the #486 effort. Every cohesive
*behavior* block has been extracted behind a typed `Protocol` "engine-backref" and the
engine keeps thin delegating wrappers (so call sites and test mock points are unchanged).
**~80 of the engine's 98 methods are now ≤2-line wrappers or small helpers.**

### Coordinator / module family (all under `src/engines/live/`)

| Module | Class | Responsibility |
|---|---|---|
| `execution/entry_coordinator.py` | `LiveEntryCoordinator` | Entry decision + base-asset-locked order path (#703) |
| `execution/exit_coordinator.py` | `LiveExitCoordinator` | Exit decision + base-asset-locked close (#710, #657) |
| `execution/order_fill_coordinator.py` | `LiveOrderFillCoordinator` | OrderTracker poll-thread callbacks (fills/cancel/tracking-lost, #631/#741) |
| `execution/market_data_coordinator.py` | `LiveMarketDataCoordinator` | Per-candle data fetch, sentiment enrich, context-readiness, correlation context |
| `execution/stop_loss_manager.py` | `LiveStopLossManager` | All exchange-facing stop-loss lifecycle calls |
| `loop_timing.py` | `LiveLoopTimingCoordinator` | Loop cadence + data-freshness helpers |
| `dynamic_risk_coordinator.py` | `LiveDynamicRiskCoordinator` | Per-entry dynamic-risk sizing + audit logging |
| `strategy_runtime.py` | `StrategyRuntimeCoordinator` | Strategy normalization + per-candle runtime decision pipeline |
| `strategy_hot_swap.py` | `StrategyHotSwapCoordinator` | Hot-swap / model-update lifecycle |
| `ws_health.py` | `WebSocketHealthMonitor` | WS stream health + reconnect (lock-free single-writer) |
| `recovery.py` | `LiveSessionRecoverer` | Startup recovery (balance, position reload, reconciliation) |
| `monitoring/` | `LiveAccountMonitor` + extractors | Account snapshots, status lines, indicator/sentiment/ml extractors |
| `config.py` | `LiveEngineSettings` | Construction-time settings resolution (feature flags / env / config) |

### Largest blocks still in `trading_engine.py` (the remaining work)

| Method | Lines (develop) | Nature |
|---|---|---|
| `__init__` | ~534 (`183–717`) | Wiring hub — ~29 distinct phases (see §5). |
| `_trading_loop` | ~390 (`1540–1930`) | **Orchestrator core — stays.** Readability-only improvements optional. |
| `start` | ~326 (`992–1318`) | Bootstrap sequence (session recover/create, #668 carry-forward, #657 self-heal, account sync, WS streams, reconciler, loop kickoff). |
| `_init_modular_handlers` | ~112 (`743–855`) | Default-vs-injected handler construction. |
| `stop` | ~86 | Lifecycle teardown. |
| `_is_transient_db_error`, `_record_event`, `_send_alert`, `_create_exchange_provider` | small | Cross-cutting infra. |

(Re-grep current line numbers before editing — they shift between PRs.)

---

## 3. The proven extraction pattern (engine-backref `Protocol`)

Use this for every extraction. It is the established pattern across all coordinators above.

1. New class in **its own module**. `__init__` stores only `self._state = engine_state`.
2. Every method begins `state = self._state` and accesses **all** engine state/helpers via
   `state.X` at call time (never copies state).
3. Define a narrow `Protocol` (`Live<Thing>EngineState`) declaring **every** `state.X`
   attribute/method the module touches. Use **concrete types** where importable (under
   `if TYPE_CHECKING:` to avoid circular imports); `Any` only for genuinely loose
   interfaces. (Standard set by later coordinators: e.g. `db_manager: DatabaseManager`,
   `_kline_buffer: KlineBuffer | None`.)
4. The engine constructs `self.x_coordinator = XCoordinator(engine_state=self)` and keeps a
   **thin delegating wrapper** for every moved method, so all call sites and test mock
   points are unchanged.
5. **Verbatim moves:** the moved bodies must be byte-for-byte identical except a mechanical
   `self.` → `state.` rewrite. Use an AST-scripted slice+rewrite (don't hand-transcribe;
   it drifts). De-underscore moved method names in the coordinator; the engine wrapper
   keeps the original underscored name and delegates.

### Routing rules (critical for test-mock interception)

- **Coordinator-internal calls between two moved methods** → de-underscored `self.`
  (e.g. `execute_exit` → `self.execute_exit_locked`).
- **Calls to engine methods that tests mock** (`_record_event`, `_send_alert`,
  `_execute_exit`) → route through `state.` so engine-level `patch.object(engine, ...)`
  still intercepts. Before moving, grep tests for `patch.object`/`= Mock()` on the methods
  involved to decide routing.
- **Re-exports:** if a moved symbol is imported elsewhere from `trading_engine` (e.g. the
  `trade_close_accounting` helpers), keep a `# noqa: F401` re-export.
- After moving, run `ruff check --fix` on the engine to drop now-unused imports — but
  **verify** it didn't drop a name still used elsewhere or a needed re-export.

---

## 4. The parity discipline (the "Holy Grail")

- **Oracle:** `tests/integration/parity/test_backtest_determinism.py` —
  `run_deterministic_backtest()` + `_fingerprint()` (canonical JSON) /
  `_fingerprint_hash()`. This is committed and authoritative; prefer it over any ad-hoc script.
- **Current canonical value on `develop`:** `trades=14`,
  `final_balance=9964.469867425983`,
  `sha256=ee76fd681362f5a251f9bd34ee40c7177ca8697e9b29e70b2c0d1ed2afd03a87`.
- **Method:** capture the fingerprint before your change and after; assert **byte-identical**.
  A quick runner:
  ```python
  import sys; sys.path.insert(0, "tests/integration/parity")
  from test_backtest_determinism import run_deterministic_backtest, _fingerprint, _fingerprint_hash
  res = run_deterministic_backtest()
  open(sys.argv[1], "w").write(_fingerprint(res))
  print(res["total_trades"], repr(res["final_balance"]), _fingerprint_hash(res))
  ```
  then `diff` the before/after files.
- **Nuance:** live-only code (coordinators, WS health, loop timing, dynamic risk, entry/exit
  callbacks) is **not** exercised by the backtest, so changes confined to it won't move the
  fingerprint. Still keep behavior verbatim and test live paths directly. The fingerprint is
  the guard against accidentally perturbing **shared/backtest** code.
- Determinism was root-caused to multi-threaded BLAS; `Backtester.run` pins BLAS/OpenMP to 1
  thread (#811). **Do not undo that.**

---

## 5. Remaining work — prioritized plan

Each item is its own PR with the full workflow in §6. Re-grep line numbers first.

### Step A (highest value) — Construction slim-down
Decompose `__init__` (~534) + `_init_modular_handlers` (~112) into cohesive private
initializer helpers and/or a `LiveEngineBuilder`. The `__init__` phases are roughly:
input validation → settings resolution → coordinator construction → `_configure_strategy`
→ providers → risk-manager merge/bind → trailing-stop policy → dynamic-risk config →
timing config → balance/financial/flags → partial-operations policy → correlation
engine+handler → DB manager → dynamic-risk manager → exchange/account/order-tracker →
balance resume → strategy-manager (hot-swap) → trading-state → WS state → performance
tracker → error-handling → time-exit policy → threading → regime detector → signal
handlers → execution model → `_init_modular_handlers`.

Suggested helpers: `_validate_inputs`, `_resolve_risk_and_policies`
(risk-manager + trailing + dynamic-risk + time-exit + partial + correlation),
`_init_infrastructure` (DB / exchange / account-sync / order-tracker),
`_seed_runtime_state`, `_install_signal_handlers`. A builder that returns a wired
dependency bundle is the more testable end-state; phase-helper extraction is the
lower-risk first move and still hits the goal.

**Hard constraints (do not break):**
- Preserve the **full constructor signature** (35 params — see §7) exactly: names, order,
  defaults. The runner (`src/engines/live/runner.py:267–282`) and many tests depend on it.
- Preserve **every public attribute** the rest of the engine, `start()`, the coordinators,
  and the integration tests read.
- Preserve **construction ordering**: handlers have inter-deps (e.g. `live_execution_engine`
  needs `fee_rate`/`slippage_rate`; `live_entry_handler` needs `live_execution_engine`;
  dynamic-risk manager needs `db_manager`; coordinators are built before `_configure_strategy`
  which is their first caller).
- Consider moving OS **signal registration** out of `__init__` into `start()` (it's a side
  effect that complicates construction) — but only if behavior/tests allow; verify.

### Step B — `start()` bootstrap slim-down
Move the startup sequence into `LiveSessionRecoverer` and/or a new `LiveStartupSequencer`,
leaving `start()` a thin phase orchestration. Watch the capital-critical ordering: session
recover → create session → wire session/strategy/execution-engine/event-logger → #668
open-position carry-forward → #657 self-heal → account sync / exchange reconciliation → WS
streams → WS health monitor → periodic reconciler → loop kickoff.

### Step C (optional, behavior-preserving) — `_trading_loop` readability
Extract per-iteration phases (data fetch → freshness/context gate → entry/exit eval →
cadence/sleep → error handling) into named private helpers. The loop **stays** on the engine.

### Step D — Protocol-tightening + test-consistency sweep (tracked)
The **entry** and **exit** coordinators still use bare `Any` for some `Protocol` attrs
(e.g. `db_manager`) and their tests use `MagicMock(spec=...)`. Align them to the
concrete-typing + `create_autospec` standard the later coordinators set. (`create_autospec`
works fine on these `Protocol`s — verified — and enforces call-signature drift, not just
attribute names.)

### Step E — Document deliberate non-extractions
- **Keep `_record_event` / `_send_alert` on the engine.** They are cross-cutting infra used
  by *every* coordinator via `state.`; extracting them adds indirection without cohesion
  benefit. (A tiny `LiveEventEmitter` is possible but low value.)
- Record the **one-class-per-file** decision: the engine-state `Protocol`s are co-located
  with their coordinator. This is consistent across the family and acceptable; document it
  rather than churn.

---

## 6. Per-PR workflow

1. Branch off `develop`: `claude/live-trading-engine-refactor-09koj9-<topic>`.
2. Make the change (AST-sliced verbatim where extracting).
3. **Quality gate:** `ruff check`, `black --check`, `python -m mypy -p src.engines.live.<module>`
   (use the **`-p` package** form — the file-path form chokes on the `ai-trading-bot`
   metadata name). There is a pre-existing unused `type: ignore` on `import requests` in the
   engine that surfaces environment-dependently — **not yours to fix.**
4. **Parity:** capture the fingerprint before/after, assert byte-identical (§4).
5. **Run the full fast suite BEFORE pushing:** `pytest -m "not integration and not slow"`
   (~4,090 tests, ~2 min). Pushing first then fixing wastes the bot's ~8-min review cycle.
6. Update docs (§ below).
7. Push (`git push -u origin <branch>`; retry w/ backoff on network errors only).
8. Open PR (base **`develop`**). For money-touching extractions, dispatch
   `architecture-reviewer` + `code-reviewer` in parallel on `git diff develop...HEAD -- <files>`.
9. **CI required green:** the four `unit-tests` shards, `integration-tests`, and
   `claude-review` (the bot review; ~7–10 min; concludes `success` even with advisory
   comments). Unrelated scheduled jobs (Optimizer MVP, Weekly Documentation Scan, Heartbeat
   Monitor) are independent — ignore them.
10. **Webhooks do NOT deliver CI success / new pushes / merge-conflict transitions.** Re-poll
    via `mcp__github__pull_request_read get_check_runs`. `send_later` is unavailable; drive
    re-polls off a background `sleep` timer. Never foreground-`sleep`.
11. Merge with **squash**; title `refactor(live): … (#486) (#PR)` (or `fix(live): …` for a
    hardening follow-up).

### Docs to update every PR
- `docs/changelog.md` — `### Changed` (refactor) / `### Fixed` (hardening) under `[Unreleased]`.
- `docs/live_trading.md` — the coordinator / lock-ownership table (verify the table isn't
  mangled after editing).
- `docs/architecture.md` — the live engine **Key Files** bullet list and the directory-tree
  diagram (the tree is **selective** — don't bloat it; group root-level `live/*.py` files
  separately from `execution/`).

---

## 7. Constructor signature to preserve (public API)

```python
def __init__(
    self,
    strategy, data_provider, sentiment_provider=None, risk_parameters=None,
    check_interval=DEFAULT_CHECK_INTERVAL, initial_balance=DEFAULT_INITIAL_BALANCE,
    max_position_size=DEFAULT_MAX_POSITION_SIZE, enable_live_trading=False,
    log_trades=True, alert_webhook_url=None, enable_hot_swapping=True,
    resume_from_last_balance=True, database_url=None, max_consecutive_errors=10,
    account_snapshot_interval=DEFAULT_ACCOUNT_SNAPSHOT_INTERVAL, provider="binance",
    testnet=False, enable_dynamic_risk=DEFAULT_DYNAMIC_RISK_ENABLED,
    dynamic_risk_config=None, time_exit_policy=None, trailing_stop_policy=None,
    partial_manager=None, enable_partial_operations=True, fee_rate=DEFAULT_FEE_RATE,
    slippage_rate=DEFAULT_SLIPPAGE_RATE, use_high_low_for_stops=True,
    max_filled_price_deviation=DEFAULT_MAX_FILLED_PRICE_DEVIATION,
    position_tracker=None, execution_engine=None, entry_handler=None, exit_handler=None,
    market_data_handler=None, event_logger=None, health_monitor=None, settings=None,
): ...
```
Required: `strategy`, `data_provider`. The 7 `*_handler` / tracker / engine params + `settings`
are dependency-injection seams (defaults built when `None`). `settings` is injected by the
runner via `LiveEngineSettings.resolve()`; the engine self-resolves when omitted.

### Test guards to respect
- `tests/integration/live_trading/test_engine_core.py` asserts public attrs post-construction
  (`strategy`, `data_provider`, `current_balance`, `live_position_tracker`,
  `completed_trades`, `enable_live_trading`).
- No tests use `__new__`; coordinator unit tests mock the `Live*EngineState` `Protocol` backref.
- `tests/unit/live/test_order_execution.py`, `tests/unit/test_db_resilience.py`,
  `tests/unit/engines/live/test_record_event.py`,
  `tests/unit/engines/live/test_order_tracking_lost.py`,
  `tests/unit/engines/live/test_stop_loss_cancel_escalation_741.py` exercise the
  callback/observability methods directly via the engine wrappers — keep the wrappers.

---

## 8. The `claude[bot]` automated reviewer — policy & known traps

The bot reviews each PR and flags carried-over CODE.md nits: f-strings in logging,
`datetime.utcnow()`, missing `exc_info=True` on error-level logs, missing `-> None`,
`Any` where a concrete type exists, `MagicMock(spec=)` vs `create_autospec`, exact float
`==` on financial values (use `pytest.approx`), `if value:` on `0.0`-valid numeric fields
(use `is not None`), untested branches, missing docstrings.

**Always verify its suggestions against the code — some are wrong:**
- It suggests `datetime.now(UTC) - naive_dt` for the `utcnow()` fix → that **raises**
  (aware − naive). Correct fix: `datetime.now(UTC).replace(tzinfo=None)` when the original
  compared two naive-UTC datetimes.
- It has missed an off-hours wall-clock flaky test (a test asserting exact adaptive-interval
  values without pinning `datetime.now(UTC).hour`).

**Policy:**
- For a **verbatim money-logic extraction** that carried dual-reviewer + parity
  certification: **decline** carried-over nits in-place (post **one** concise PR comment),
  and fix them in a **dedicated hardening follow-up PR**. This preserves the verbatim
  guarantee the safety review rests on. (Precedent: #815→#816, #813→#814, #817→#818.)
- For **small, non-money-logic modules** with no dual-review certification: **fix valid
  findings in-place** (precedent: #820, #821). Add direct coordinator unit tests up front
  (autospec'd `Live*EngineState`) to preempt the "no tests" finding.

---

## 9. Git / environment conventions

- Commit author must be `noreply@anthropic.com` / `Claude`
  (`git config user.email noreply@anthropic.com && git config user.name Claude`); a stop-hook
  flags "Unverified" commits — fix the tip with `git commit --amend --no-edit --reset-author`.
- End commit messages with the session URL line. **Never** put model IDs in commits/PRs/code.
- GitHub ops are via `mcp__github__*` MCP tools (no `gh` CLI). Repo scope: `bumpy-croc/ai-trading-bot`.
- Read first: `CLAUDE.md`, `CODE.md`, `.claude/LESSONS.md`, `docs/live_trading.md`,
  `docs/architecture.md`.
- Work autonomously; escalate only on **parity drift** or a genuine confidence drop.

---

## 10. Session ledger (#486 PRs merged on `develop`)

| PR | What |
|---|---|
| #796 / #798 / #800 | monitoring + stop-loss; recovery; config-settings extractions |
| #809 / #810 / #812 / #813 | strategy-runtime; hot-swap; ws-health; entry pipeline |
| #811 | BLAS-thread pin for backtest determinism |
| #814 | entry-coordinator CODE.md hardening + tests |
| #815 / #816 | exit pipeline → `LiveExitCoordinator` + hardening |
| #817 / #818 | order-fill callbacks → `LiveOrderFillCoordinator` + hardening |
| #819 | market-data read path → `LiveMarketDataCoordinator` |
| #820 | loop-timing → `LiveLoopTimingCoordinator` (+ hardening folded in) |
| #821 | dynamic-risk → `LiveDynamicRiskCoordinator` (+ tests folded in) |

Engine: **6,558 → 2,493 lines**, parity byte-identical throughout.
