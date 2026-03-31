# CODE.md

**All instructions in this file must be followed when making code changes.**

---

## Before You Start Coding

### Planning Complex Features

For features involving exchange interaction, crash recovery, or position state mutation: enumerate all failure scenarios BEFORE writing code:

- Timeout (API returns None / no response)
- Partial fill (order partially executed)
- Crash mid-operation (bot dies between two state changes)
- Rejected order (exchange definitively refuses)
- Resumed session (bot restarts, finds stale state)
- External intervention (manual trade on exchange)
- Cancelled/expired orders (exchange cancels after timeout)
- Double-execution (same order processed twice)

Each scenario needs: a code path, a test, and a state recovery mechanism.

---

Follow standard SOLID, KISS, YAGNI, and composition-over-inheritance principles. Design for failure (timeouts, retries, circuit breakers) and observability (structured logging, metrics) by default.

---

## Coding Style

- Prioritize readability, simplicity, and maintainability.
- Use descriptive variable names, early returns, and guard clauses.
- Avoid deep nesting, overly large files, and within-file duplication.
- Remove debugging/temporary code before commits.
- Place user-customizable configuration at the beginning of scripts.
- Avoid nested regex quantifiers or ambiguous patterns (catastrophic backtracking).
- Maintain documentation in `docs/`.
- Remove unused imports, parameters, and dead code before submitting.
- Magic numbers need a one-line comment explaining why.
- Sort imports after changes.
- Don't shadow Python builtins (e.g. don't name a class `TimeoutError`).

### Functions
- Keep concise with a self-evident purpose. Always include a docstring.
- Don't override function arguments or overuse undocumented lambdas.

### Classes
- Single responsibility per class, one class per file.
- Use properties for public access to computed/delegated values. Never access `self._foo` from outside the class when a property exists.

### Types
- Avoid `Any` where possible. Use `Protocol` for duck-typed parameters.
- Type public APIs properly.

### Comments
- Explain the goal (why), not the mechanics (what).
- Never use "new", "updated", etc in comments or filenames.
- Present tense only — describe what the code *does*:
  - Bad: `# New enhanced v2 API.` / Good: `# Fetches user data from the v2 API.`

---

## Position Fields

**CRITICAL** — Using the wrong field is the single most common source of bugs in this codebase.

| Field | Meaning | Example | Use for |
|-------|---------|---------|---------|
| `quantity` | Asset amount | 0.5 BTC | Exchange orders, emergency close qty |
| `size` | Balance fraction at entry | 0.02 (2%) | Entry sizing decisions |
| `current_size` | Remaining after partial exits | 0.01 | SL re-placement, holdings checks |
| `original_size` | Size at entry (immutable) | 0.02 | Scaling ratios |

- These fields are **never interchangeable**.
- Scale expected holdings by `current_size / original_size` after partial exits.
- Never use `if value:` on these fields — `0.0` is valid state (flat position), not falsy. Use `if value is not None:`.
- Preserve `0.0` when deserializing from DB — don't convert to `None`.

---

## Exchange Mode & Account Type Safety

When adding support for a new exchange mode (margin, futures, etc.) or modifying account-type-sensitive logic:

- **Wire mode flags end-to-end.** If a class accepts a mode parameter (`use_margin`, `account_type`), verify every caller passes it. Dead-code flags that default to `False` are worse than no flag — they create false confidence. Search for every constructor call.
- **Audit all consumers of exchange data.** Balance, position, and order semantics change between modes. `free + locked` is equity in spot but not in cross-margin (ignores liabilities). `netAsset` per asset is not account-level equity (excludes cross-asset liabilities). Use the right field for the context.
- **Borrowed vs held.** For short position detection in margin, use raw `borrowed` amount, not `netAsset`. A short with `netAsset >= 0` still has outstanding debt if `borrowed > 0`.
- **Error codes differ across API variants.** Spot and margin APIs return different error codes for the same failure (e.g. insufficient balance). Add margin-specific codes (-3027, -3028, -3041, -3067) to definitive reject sets. An unrecognized margin reject falling through to `return None` creates phantom positions.
- **Config defaults must be safe for all deployment contexts.** If a mode enables borrowing or real-money operations, default to the safer option (`spot`, not `margin`). Dashboards, CLI tools, health checks, and Binance.US all instantiate providers without trading context.
- **Enforce assumptions about wallet state at startup.** If the implementation assumes USDT-only collateral, check for non-USDT holdings and fail fast. Undocumented assumptions become fund-loss paths when the wallet state changes.
- **Reconciliation must cover both startup and runtime.** A margin-aware check added only to `PositionReconciler` (startup) but not `PeriodicReconciler` (runtime) leaves a gap for positions closed during operation.
- **Fallback/stub paths must respect safety constraints.** If live margin mode must never use an offline stub, guard ALL paths to the stub — both "SDK unavailable" and "client init failed" — not just one.
- **Use quantity thresholds, not binary checks.** A position detection check that only asks "is borrowed > 0?" misses large partial external closes. Apply the same 50% tolerance threshold used in spot mode.

---

## Arithmetic & Financial Calculations

- Check `math.isfinite()` on any numeric value feeding into a trading decision.
- Validate divisors are non-zero before division. Use guard clauses or early returns.
- Use epsilon tolerance for float comparisons: `abs(a - b) < EPSILON`.
- Use consistent fee/slippage via shared modules — never duplicate financial logic.
- Track entry balance at position creation for accurate P&L.
- Protect against negative balance corruption with validation at every update.
- Never apply two reductions for the same partial close — subtract OR scale, not both.
- Refund only the unfilled portion of entry fee on order cancellation.
- Cap unnormalized position sizes at 1.0 when normalization metadata is missing.
- Use `result.position.quantity` (not size) in emergency close calculations.
- Guard division by zero in loops over positions that may close during iteration:
```python
current_fraction = position.current_size / position.original_size
if abs(current_fraction) < 1e-9:  # Position fully closed
    break
exit_of_current = exit_of_original / current_fraction
```
- Add maximum iteration guards to partial operation loops as defense-in-depth.

---

## Input Validation

- Validate at system boundaries: price > 0 and finite, qty >= 0, divisors != 0.
- Check JSON/dict API response types before accessing fields.
- Validate numeric ranges (e.g. percentages 0–1), string inputs used in queries.
- Validate array/list indices within bounds. Check DataFrame not empty before `.iloc[]`, `.min()`, `.max()`.
- Validate `split()` results have expected element count before indexing.
- Use `.get()` for dictionary access when key may not exist.
- Add `__post_init__` validation to dataclasses. Catch bad config at construction.
- Validate tick_size/step_size are non-zero before division.
- Validate ONNX metadata JSON is a dictionary before use.
- Add bounds validation for CLI arguments.
- Validate `entry_price > 0` before any P&L or stop-loss calculation.
- ML models require feature schema validation even when features appear unused.
- Validate parallel list lengths match at init (e.g. `exit_targets` and `exit_sizes`).
- Validate required dependencies in `__init__`, not at runtime:
```python
# Bad: fails at order execution time
def execute_order(self):
    if self.exchange_interface is None:  # Too late!
        return None

# Good: fail at init
def __init__(self, exchange_interface, enable_live_trading):
    if enable_live_trading and exchange_interface is None:
        raise ValueError("Cannot enable live trading without exchange interface")
```

---

## Error Handling

- No empty catch blocks. Prioritize specific exception types over generic ones.
- Log errors with sufficient context (relevant variables, operation attempted).
- Include `exc_info=True` in error-level log calls for stack traces.
- No bare `except Exception` with `logger.debug()`. Log at WARNING minimum.
- Use `%s` lazy formatting in logging: `logger.error("Failed: %s", e)`.
- Elevate critical initialization failures from DEBUG to WARNING/ERROR.
- Handle `OSError` around cache operations — TOCTOU between exists() and read().
- Wrap callback invocations in try/except so failures don't block state updates.

### Exchange `None` Returns

Distinguish "definitely rejected" from "ambiguous timeout":

```python
# Rejected — exchange explicitly refused. Safe to clean up.
if response and response.status == "REJECTED":
    logger.warning("Order rejected: %s", response.reason)
    return OrderResult.REJECTED

# Ambiguous — might have been placed. Enter close-only mode.
if response is None:
    logger.critical("Order result unknown — entering close-only mode")
    return OrderResult.UNKNOWN  # Blocks duplicate orders
```

Treating all `None` as "maybe placed" creates phantom positions. Treating all `None` as "rejected" orphans real inventory.

### DB/Memory State Divergence

On DB write failure after in-memory state change: rollback in-memory state or log CRITICAL. Silent divergence is a bug.

```python
previous_size = position.current_size
position.current_size = new_size
try:
    db_manager.update_position(position)
except Exception as e:
    position.current_size = previous_size  # Rollback
    logger.critical("DB/memory diverged for %s: %s", position.symbol, e)
```

---

## State Management & Recovery

- Every detection must mutate state to fix the problem. Detect-without-act is a bug.
- Write crash-recovery and restart paths alongside the happy path — they are part of the feature.
- A DB write without a corresponding exchange action is incomplete. If you write a stop-loss to DB, place it on the exchange too.
- Initialize session context (session_id, strategy_name) on ALL entry paths — new AND resumed.
- After recovering a partially-filled order, re-register it with the order tracker so post-recovery fills are monitored.
- When an exchange action fails after state was committed, either undo the state or escalate to CRITICAL/close-only.
- Emergency close must confirm the sell was accepted before removing the position from tracker and DB.
- Don't reuse inactive session IDs on clean restart. Create new sessions; recover balance from the most recent inactive one.
- Preserve paper positions across restarts instead of force-closing on shutdown.
- Serialize concurrent reconciliation when switching between redundant data paths (e.g. stream ↔ polling).
- Dead constants are misleading. If a constant describes intended behavior, import and use it.
- On terminal events (cancel/reject/expired), reconcile missed state deltas BEFORE firing the terminal callback.
- Don't mark a connection as healthy until the first successful event confirms it. Reset the flag on each reconnect.
- When handing off between redundant paths: stop old producer → drain in-flight work → enable replacement → catch-up → disable old path.

---

## Thread Safety & Concurrency

- Lock all shared mutable state. Release locks in finally blocks.
- Move callbacks and long-running operations outside lock scope to prevent deadlocks.
- Verify threads stopped after join timeout.
- Acquire lock before checking existence, not after. Check-then-act outside a lock is a TOCTOU race.
- Use `RLock` (not `Lock`) when methods may call each other (reentrant).
- Use atomic `pop()` for get-and-remove on shared dictionaries.
- Initialize locks at class/module level, not lazily inside methods. Lazy init is itself a race.
- Use copy-on-write for shared collections during reload (build new, swap atomically).
- Use `threading.Event` for shutdown signaling, not `time.sleep()` polling.
- Protect ALL access to mutable shared state with the same lock.
- Return defensive copies from properties that expose mutable internal state.
- Use `os.replace()` for atomic file/symlink updates (POSIX guarantee).
- Use `dict.pop(key, None)` for cache eviction, not `dict.get()` then delete.
- Redirect unused subprocess streams to DEVNULL to prevent pipe deadlocks.
- Use `pool_timeout` to prevent indefinite blocking on DB pool exhaustion.
- After snapshotting a collection for iteration, re-check membership under lock before mutating — items may have been removed concurrently.
- Snapshot mutable values before calling a mutation, then compare. The "before" reference may alias the mutated object.
- Lazy singleton init (check-then-create) requires a lock even for initially-`None` fields.
- If a worker thread doesn't stop after join timeout, stay degraded — don't proceed while it may still be mutating state.
- Gate producer→consumer queues with a lock-protected `_closed` flag. Set the flag and do a final drain atomically to prevent late items slipping in after the drain.
- On reconnect, pass a fresh callback — stale callbacks may reference stopped consumers.

---

## Resource Management

- Always close sessions, connections, and file handles (use context managers).
- Use `ExitStack` for managing multiple context managers.
- Close HTTP sessions explicitly (`provider.close()`) — don't rely on GC.
- Add `close()` / `__del__()` to classes holding ONNX sessions, DB connections, or HTTP sessions. Make `close()` idempotent.
- Stop tracking threads in test fixtures to prevent thread leaks.
- Cap unbounded collections (e.g. balance history) to prevent memory leaks.
- Clean up ONNX sessions and matplotlib figures in exception handlers.

---

## External API Calls

- Add timeout protection to all HTTP requests, DB queries, ONNX inference, and model loading.
- Implement circuit breakers for repeated API failures.
- Use exponential backoff for retries (3 attempts max).
- Validate response types and status codes before processing.
- Use `ConnectionError` (not broad `Exception`) when mocking network failures in tests.
- `dict.get(key, default)` returns `None` when the key exists with JSON `null`. Use `or`: `float(d.get("n") or 0)`.
- Map all known external statuses explicitly. Unknown values → no-op with warning, never terminal.
- Add a grace period before health-checking new connections to avoid false-positive failures.

---

## Event & Data Processing

- Separate dedup check from dedup marking. Only mark events as seen after successful processing.
- Only bump freshness timers when data was actually mutated, not for stale or replayed events.
- Detect gaps in sequential data (e.g. timestamp jumps) and trigger resync instead of silently appending.
- Guard data replacement against empty or stale responses — keep existing data and retry later.
- Only trust a source's freshness signal when the source is confirmed healthy, not just connected.

---

## Timezone Handling

- Use `datetime.now(UTC)` consistently, not `datetime.utcnow()` (deprecated).
- Store all timestamps in UTC; convert to local only for display.
- Never compare UTC-aware datetimes with timezone-naive pandas timestamps. Localize or use naive consistently.
- Extract shared timezone normalization helpers instead of duplicating tz logic.

---

## Security

- Never embed sensitive information directly in code. Use placeholders.
- Validate user-provided paths with `.resolve()` + parent directory checks to prevent path traversal.
- Use Path objects instead of string concatenation. Use atomic writes (write to temp, then rename).
- Prevent open redirect in admin login by validating redirect targets.
- Require authentication on ALL admin routes (Flask-Admin, `/db_info`, `/migrate`).
- Escape HTML when rendering DB-sourced values into templates.
- Validate numeric parameters like `timeout_ms` to prevent SQL injection.
- Redact credential keys in logging output.

---

## Database & Transactions

- Avoid `SELECT *` and redundant indexes.
- Write coupled fields (size + entry_price) in a single transaction. Split writes cause inconsistent state on partial failure.
- Scope DB lookups with symbol AND order_type, not just client_order_id.
- Add `db.rollback()` after `ProgrammingError` to prevent session poisoning.
- Use `db.expire_all()` before polling predicates to prevent stale reads.
- Add `for_update=True` (row-level locking) in reconciliation queries.
- Add query timeouts to balance operations — slow queries block the trading loop.
- Handle `session.commit()` errors everywhere, not just "important" paths.
- Verify DB column names match the actual schema before querying.

---

## Tests

- Keep tests stateless (fixtures, no global state). Use AAA pattern. FIRST principles.
- Assert each condition independently. Never `assert A or B`.
- Every new code branch needs a test before the PR.
- Use exact value assertions: `assert severity == Severity.LOW`, not `in (...)`.
- Add pytest markers (`@pytest.mark.fast`, `@pytest.mark.integration`) to every test.
- Integration tests must use unique IDs per test to prevent DB collisions.
- Use `autospec=True` on mocks to catch signature mismatches.
- Make tests deterministic — use dependency injection for randomness, timestamps, and external state.
- Don't write tautological tests that always pass regardless of behavior.
- Use `pytest.approx` for financial calculations, not exact float equality.
- Mock external providers in session recovery tests to prevent CI failures.
- Seed test data after `_reset_run_state()`, not before.

---

## Backtest-Live Parity

- Never duplicate financial logic — use `src/engines/shared/` for fee/slippage (`cost_calculator`), P&L (`pnl_percent`), position sizing, and SL/TP logic.
- Store gross P&L consistently in `Trade.pnl` across both engines.
- Write parity tests that run both engines with identical inputs and assert matching outputs.
- When adding a feature to one engine, check if the other needs the same change.

---

## Architecture

- Order tracker callbacks must handle ALL order types (entry AND stop-loss). Missing SL callback handling leaves positions unprotected.
- Use the strategy's `get_risk_overrides()` for backtest config, not global defaults.
- Wire parameters to the correct nested component (e.g. `position_sizer.base_sizer.fraction`, not `position_sizer.fraction`).
- Register new strategies in the engine's strategy managers — an unregistered strategy silently does nothing.

---

## Pre-Commit Checklist

Run through this before committing:

- [ ] `atb dev quality` passes (black, ruff, mypy, bandit).
- [ ] All new code paths have tests. Tests use AAA, exact assertions, and pytest markers.
- [ ] No `if value:` on numeric fields that can be zero — use `if value is not None:`.
- [ ] No bare `except Exception` with debug logging. Specific exceptions, WARNING minimum.
- [ ] Financial calculations use shared modules, not local reimplementations.
- [ ] DB writes and exchange actions are paired — no DB-only stop-loss, no unverified emergency close.
- [ ] All external calls have timeouts. All shared state access is locked.
- [ ] No unused imports, dead code, magic numbers without comments, or f-strings in logging.
- [ ] If you touched one engine, check if the other needs the same change.
- [ ] If you added a constructor parameter, verify every caller passes it (search for all instantiation sites).
- [ ] If you added exchange-mode logic, verify both startup and periodic reconciliation paths are covered.
- [ ] If you changed balance/position semantics, audit all consumers (sync, reconciliation, sizing, risk).
