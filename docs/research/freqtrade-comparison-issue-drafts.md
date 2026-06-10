# Issue drafts from freqtrade comparison audit (2026-06-10)

Status: **2 of 15 created** before GitHub auth started failing (401s from the API).

- ✅ Created: #777 `feat(experiments): Optuna hyperparameter search wrapped around backtester + walk-forward validation`
- ✅ Created: #778 `feat(backtest): multi-pair / multi-position backtesting to close the portfolio parity gap`
- ⏳ Remaining: the 13 drafts below. Also: enrich existing #486 (LiveTradingEngine orchestrator refactor) with the spec in section 6 instead of opening a duplicate.

Label taxonomy confirmed to exist in the repo: `state:proposed`, `priority:p0–p3`, `type:fix`, `type:research`, `type:post-mortem-action`, `area:backtest|data|live-ops|risk|strategy`, `enhancement`, `tech debt`, `code maintenance`, `source:parity-gap`, `training`, `bug`.

---

## 1. feat(live): Telegram remote control and trade notifications

Labels: `enhancement`, `state:proposed`, `priority:p2`, `area:live-ops`

### Problem

There is no remote control or push notification channel for the live bot. freqtrade ships Telegram + WebUI (stop bot, force-exit, P&L from a phone). Our DB schema already anticipates this (`alert_sent`, `alert_method` columns supporting telegram/email/slack) and `runner.py` has a `--webhook-url` stub, but nothing is implemented. During incidents (see `.claude/state/incidents/`), the only control surface is Railway/SSH.

### Scope of work

1. New module `src/notifications/` with a `Notifier` interface and a `TelegramNotifier` implementation (long-polling via `python-telegram-bot` or raw Bot API with `requests` — prefer the latter to avoid a heavy dependency; we already pin `requests`).
2. Outbound notifications (phase 1): trade opened/closed (symbol, side, size, price, PnL), emergency-control level changes (`src/strategies/components/emergency_controls.py` already has alert callbacks with 10s timeout — register the notifier there), reconciliation CRITICAL/HIGH events, bot start/stop.
3. Inbound commands (phase 2, same PR or follow-up): `/status` (open positions, balance, daily PnL), `/pause` (no new entries; existing positions managed), `/resume`, `/panic` (trigger close-only mode via existing EmergencyControls — must NOT bypass its approval workflow).
4. Config via env vars `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (through `src/config/config_manager.py`, never hardcoded). Feature is fully inert when unset.
5. Authorization: inbound commands accepted ONLY from the configured chat id; log and ignore everything else.
6. Update `alert_sent`/`alert_method` on the relevant DB rows when a notification is delivered.

### Acceptance criteria

- [ ] With env vars unset, zero behavior change and zero network calls (test asserts no notifier is constructed).
- [ ] Unit tests with mocked HTTP for all outbound message types and inbound command parsing/auth rejection.
- [ ] `/panic` goes through EmergencyControls (test proves close-only mode engaged, no direct order placement from the notifier module).
- [ ] Notification failures never block or crash the trading loop (fire-and-forget with timeout; test with a hanging mock).
- [ ] Secrets handled per CODE.md Security section; token never logged.
- [ ] Docs: new section in `docs/live_trading.md` + `docs/monitoring.md`.

### Out of scope

Slack/email implementations (interface should make them trivial later); WebUI changes.

### Required reading

CODE.md (External API Calls, Thread Safety, Security), `docs/live_trading.md`, `src/strategies/components/emergency_controls.py`.

---

## 2. feat(ml): model drift monitoring that files retraining proposals

Labels: `enhancement`, `state:proposed`, `priority:p2`, `training`

### Problem

Models are trained manually (`atb live-control train`) and promoted by hand. The weekly-training workflow has been failing for 8+ consecutive weeks (#605, #607, #614, #615, #618, #619, #622, #707) with nobody forced to care — there is no signal tying model staleness/drift to action. freqtrade's FreqAI retrains on a sliding window automatically; our charter forbids auto-promotion for live symbols, but we can automate *detection and proposal*.

### Scope of work

1. New module `src/prediction/drift.py`:
   - Prediction-quality drift: rolling directional accuracy / calibration of live predictions vs realized prices (prediction history is already cached/persisted via the prediction pipeline; persist it if not).
   - Feature drift: PSI or KS test on the live feature distribution vs the training distribution recorded in the model's `feature_schema.json` / `metadata.json` (extend training pipeline to store reference stats if missing).
   - Staleness: age of the `latest` symlink target vs a configurable threshold (default 45 days).
2. Wire a periodic drift check into the live engine's HealthMonitor (read-only; never blocks trading).
3. When thresholds breach: write a `system_events` row, send a notification (depends on Telegram issue, degrade gracefully to logs), and open a GitHub issue titled `[Automated] Model drift detected: {SYMBOL}/{TYPE}` with the metrics — pattern after the existing weekly-training failure automation in `.github/workflows/weekly-training.yml`.
4. CLI: `atb models drift-check --symbol BTCUSDT` for on-demand runs.
5. Explicitly NO auto-retrain and NO auto-promotion: the `latest` symlink for live symbols remains human-promoted (charter rule).

### Acceptance criteria

- [ ] Unit tests for each drift metric with synthetic drifted/non-drifted fixtures under `tests/data/`.
- [ ] Drift check failure (e.g., missing metadata) degrades to a logged warning — never affects the trading loop (test).
- [ ] Issue-filing path is idempotent: one open issue per symbol/type, not one per check.
- [ ] Reference-stats writing added to the training pipeline and covered by a test.
- [ ] Docs: `docs/prediction.md` section on drift thresholds and the response runbook.

### Out of scope

Fixing the weekly-training workflow failures themselves (that's #707 et al.); auto-retraining.

### Required reading

CODE.md (Input Validation, Resource Management, External API Calls), `docs/prediction.md`, charter rules in `CLAUDE.md`.

---

## 3. feat(data): complete Coinbase Advanced Trade support (stop orders + provider parity)

Labels: `enhancement`, `state:proposed`, `priority:p3`, `area:data`, `area:live-ops`

### Problem

The bot is effectively Binance-only. `CoinbaseProvider` exists but is incomplete: stop orders are a TODO (`src/data_providers/coinbase_provider.py:567`) and #762 shows every order is currently submitted as MARKET due to an order-type mapping bug. Single-exchange dependency is an availability and counterparty risk for live capital.

### Scope of work

1. Land or rebase on the #762 fix first (order-type mapping) — do not duplicate it.
2. Implement stop-loss / take-profit orders via Coinbase Advanced Trade API trigger orders, mapping to the `ExchangeInterface` enums (`OrderType.STOP_LOSS`, `TAKE_PROFIT`) in `src/data_providers/exchange_interface.py`.
3. Implement/verify: order status polling for all `OrderStatus` values, balance fetch, open-order listing, cancel — everything `PeriodicReconciler` needs.
4. Precision: apply `src/trading/precision.quantize_to_step()` to ALL exchange-bound numerics (qty, price, stop price) using Coinbase product metadata (LESSONS.md §1.1 — this bug class bit us twice on Binance; grep for `round(... / ...) *` patterns while in there).
5. Add a provider conformance test suite: a shared parametrized test class run against Binance + Coinbase mocks asserting identical semantics for order lifecycle, so future providers can't drift.

### Acceptance criteria

- [ ] Paper-trading session runs end-to-end against mocked Coinbase (entry, stop placement, fill, reconcile loop) in integration tests.
- [ ] Conformance suite passes for both providers.
- [ ] All exchange-bound numerics quantized (test with adversarial float artifacts, e.g. values that produce `1648.8200000000001`).
- [ ] No margin/futures pretense: Coinbase spot only; margin calls raise a clear `NotImplementedError` rather than silently no-op.
- [ ] `docs/data_pipeline.md` updated with Coinbase capability matrix.

### Out of scope

Live-capital rollout on Coinbase (separate proposal with `board_required: true`); additional exchanges.

### Required reading

CODE.md (Exchange Mode & Account Type Safety, External API Calls, Error Handling), `.claude/LESSONS.md` §1.1/§1.5, issue #762.

---

## 4. refactor(db): decompose DatabaseManager (3,940 lines, ~70 methods) into focused repositories

Labels: `tech debt`, `state:proposed`, `priority:p2`, `area:data`

### Problem

`src/database/manager.py` is 3,940 lines mixing connection/session lifecycle, CRUD for every model, business logic, analytics, and performance metrics. Several recent bugs live in this blast radius (#764 `recover_positions` calling a method that doesn't behave as expected, #766 dead code calling `get_first...`). Its size makes review and autonomous modification risky.

### Scope of work

1. Carve into repositories under `src/database/repositories/`: `TradeRepository`, `PositionRepository`, `SessionRepository`, `AccountHistoryRepository`, `MetricsRepository`, `EventRepository`. Each owns CRUD + queries for its aggregate.
2. `DatabaseManager` retains: engine/pool setup, session lifecycle/context managers, transaction helpers, current-session tracking, retry-on-`OperationalError` logic — and exposes the repositories as attributes (`db.trades`, `db.positions`, ...).
3. **Mechanical migration**: keep every public method as a thin delegating wrapper on `DatabaseManager` (with a deprecation note in the docstring) so the ~hundreds of call sites don't change in this PR. Call-site migration is follow-up work.
4. While moving, do NOT change behavior — even where behavior looks wrong (file an issue instead; #764/#766 are already tracked).
5. Split should be reviewable: one commit per repository extraction.

### Acceptance criteria

- [ ] `manager.py` under ~800 lines after extraction.
- [ ] Zero call-site changes outside `src/database/` in this PR; full test suite green with no test edits other than imports.
- [ ] No SQL/queries altered (diff review: moves only).
- [ ] Each repository has its own unit test file (existing manager tests may be split accordingly).
- [ ] Thread-safety semantics unchanged (session-local state preserved; document in module docstring).

### Out of scope

Behavior fixes (#764, #766); call-site migration; schema changes.

### Required reading

CODE.md (Database & Transactions, Error Handling, Architecture), `docs/database.md`.

---

## 5. refactor(engines): deduplicate live/backtest entry & exit handlers into engines/shared

Labels: `tech debt`, `state:proposed`, `priority:p1`, `area:backtest`, `area:live-ops`, `source:parity-gap`

### Problem

Entry/exit handling is duplicated between engines with ~70% identical logic:

- `src/engines/live/execution/entry_handler.py` (531 lines) vs `src/engines/backtest/execution/entry_handler.py` (655 lines)
- `src/engines/live/execution/exit_handler.py` (999 lines) vs `src/engines/backtest/execution/exit_handler.py` (819 lines)

Every backtest-live parity bug we've filed (#756, #757, #758, #748) lives in this seam: a fix applied to one copy can silently miss the other. `src/engines/shared/` already exists (entry_utils, execution models, fill policy, trailing stop manager) — the handlers just never finished migrating.

### Scope of work

1. Diff the four files and classify each block: (a) identical → move to `src/engines/shared/execution/`; (b) legitimately different (order submission vs simulated fill, snapshot builders) → leave behind small engine-specific adapters injected via composition (the audit recommendation: vary snapshot builders and order-side mappers, not whole handlers).
2. Target shape: `shared/execution/entry_logic.py` + `exit_logic.py` containing all decision logic (signal validation, sizing handoff, stop/TP computation, partial-exit tranche evaluation, time-exit checks); live/backtest handlers become <150-line adapters.
3. Document every intentional divergence in a module-level docstring table (the known parity caveats: qty rounding, margin interest, fees).
4. Sequence AFTER the open parity bug fixes (#756–#758) merge, or coordinate — otherwise this refactor will conflict with them.

### Acceptance criteria

- [ ] Combined handler LOC reduced ≥50%; no decision logic remains duplicated (reviewer spot-check: grep for any function body appearing in both engines).
- [ ] Existing parity tests (`tests/integration/parity/`) pass unchanged; add at least one new parity test that runs the SAME shared logic object under both adapters and asserts identical decisions on a fixture candle stream.
- [ ] Backtest snapshot regression: existing backtest results unchanged (byte-identical metrics on a reference run).
- [ ] Live engine coverage stays ≥95% (CI gate).
- [ ] No new behavior; pure refactor (one commit per extraction step).

### Out of scope

Multi-pair backtesting (#778); fixing the documented parity caveats themselves.

### Required reading

CODE.md (Backtest-Live Parity, Architecture, State Management), `docs/backtesting.md`, `docs/live_trading.md`.

---

## 6. SPEC COMMENT for existing #486 — decompose LiveTradingEngine (do not open a new issue)

Action: comment on #486 with the text below, and set labels `tech debt`, `state:proposed`, `priority:p1`, `area:live-ops` (it currently has none).

> Refreshed evidence from the 2026-06-10 architecture/quality audit, plus a concrete spec so this can be picked up autonomously.
>
> **Current state**: `src/engines/live/trading_engine.py` is **6,563 lines / 103 methods**, mixing the trading loop, config plumbing, entry/exit orchestration, monitoring, daily-PnL bookkeeping (TODO at ~line 5197), and reconciliation glue. It is the highest-risk file in the repo to modify, and most P0–P2 live-ops bugs (#734–#746 range) required changes inside it.
>
> **Scope of work**
> 1. Extract in this order (one PR each, smallest risk first):
>    a. monitoring/health glue → `src/engines/live/monitoring/` (PnL updater, balance snapshots, health hooks)
>    b. order/position lifecycle orchestration → thin calls into the existing `LiveExecutionEngine`, `OrderTracker`, `LivePositionTracker` (much of this is already half-delegated; finish the job and delete engine-local copies)
>    c. session/crash-recovery startup sequence → `src/engines/live/recovery.py` (coordinates with #743/#683 work)
>    d. config/feature-flag resolution → constructor-injected dataclass built in `runner.py`
> 2. `TradingEngine` ends as an orchestrator: main loop, component wiring, lifecycle (start/stop/pause) — target **<1,500 lines**.
> 3. Pure refactor discipline: no behavior changes; any bug found gets its own issue. Thread-safety: every moved block keeps its lock scope — document lock ownership per extracted module (CODE.md Thread Safety).
>
> **Acceptance criteria**
> - [ ] trading_engine.py <1,500 lines; no extracted module >800 lines
> - [ ] Live engine coverage ≥95% maintained; full suite green with no test semantic changes (import moves only)
> - [ ] A paper-trading smoke session (`atb live ml_basic --paper-trading`) runs cleanly start→entry→exit→shutdown
> - [ ] Lock-ownership table added to `docs/live_trading.md`
>
> Sequencing: do extraction (a) first; it unblocks the daily-PnL TODO. Coordinate with the entry/exit handler dedup issue (shared logic extraction) to avoid double-moves.

---

## 7. chore(deps): consolidate 3 requirements files into pyproject.toml optional-dependency groups

Labels: `tech debt`, `code maintenance`, `state:proposed`, `priority:p2`

### Problem

Four dependency manifests drift independently: `requirements.txt` (dev), `requirements-server.txt` (production/Railway), `requirements-github.txt` (CI), `pyproject.toml` (package metadata). Confirmed drift: `pandas==2.2.0` (dev) vs `pandas==2.1.4` (server) — dev/CI test against a different pandas than production trades with. TensorFlow (~500MB) correctly excluded from server, but the split is enforced by convention only.

### Scope of work

1. Move all deps into `pyproject.toml`: core deps in `[project.dependencies]` (exactly what the live bot needs = current requirements-server.txt set); extras: `[train]` (tensorflow, tf2onnx), `[dev]` (pytest, black, ruff, mypy, bandit, etc.), `[dashboards]` if separable.
2. Resolve every version conflict explicitly — production pin wins unless there's a reason; record decisions in the PR description. The pandas 2.1.4/2.2.0 split MUST be resolved and the chosen version tested.
3. Regenerate the three requirements*.txt as lock-style exports (`pip-compile` or `uv pip compile`) marked DO-NOT-EDIT with the generation command in a header — or delete them if Railway/CI/`Makefile`/`.github/workflows/*` and the sessionStart hook are all updated to `pip install .[group]`. Check ALL consumers: Dockerfile, docker-compose, railway config, Makefile (`make deps-server`, `make deps-dev`), every workflow, `.claude` session hooks.
4. CI guard: a check that fails if a dependency appears with different pins in two places.

### Acceptance criteria

- [ ] One source of truth; `pip install -e .[dev]` reproduces the dev env; server image builds without TensorFlow and stays within current size.
- [ ] All CI workflows green; Railway build verified on the development environment before merge to staging/main branches.
- [ ] Drift guard in CI demonstrated failing on an injected mismatch (then removed).
- [ ] `CLAUDE.md`/`Makefile`/`docs/` install instructions updated.

### Out of scope

Upgrading any dependency beyond conflict resolution.

---

## 8. fix(live): replace broad exception catches in trading-critical paths with typed handling + escalation

Labels: `tech debt`, `state:proposed`, `priority:p2`, `area:live-ops`, `type:fix`

### Problem

Audit found broad catches in trading-critical paths that log and continue: `except (AttributeError, TypeError, ValueError)` in `src/engines/live/trading_engine.py` and `src/engines/live/reconciliation.py`; `src/engines/live/order_tracker.py` logs errors after catches without re-raise or escalation. Given our incident history (silent failure modes are the documented theme of LESSONS.md — e.g., #738 retry decorators dead because wrapped methods swallow exceptions), swallowed exceptions in order execution/position tracking risk undetected position corruption. Related observability work: #669.

### Scope of work

1. Inventory every `except` in `src/engines/live/{trading_engine,reconciliation,order_tracker}.py` and `src/engines/live/execution/*.py`. Classify: (a) expected/recoverable — narrow the type, handle, continue; (b) unexpected in a money path — catch, write a `system_events` row, escalate (health monitor flag, and emergency-controls hook for order-execution failures), then re-raise or fail-closed; (c) `AttributeError`/`TypeError` catches that paper over the Decimal×float bug class (#675, LESSONS §1.2) — remove the catch and fix the type at the boundary instead.
2. Add module-level guidance comment per CODE.md Error Handling.
3. Coordinate with #738 (dead retry decorators) — that fix changes what propagates.

### Acceptance criteria

- [ ] No `except (AttributeError, TypeError, ValueError)` or bare `except Exception: log-and-continue` remains in order-execution, position-tracking, or reconciliation mutation paths (grep-verifiable).
- [ ] Every category-(b) path has a test proving: system_event written + escalation flag set + no silent continue.
- [ ] Chaos/reconciliation integration tests still pass; paper-trading smoke run clean.
- [ ] Findings that are actual latent bugs get filed as separate issues, not fixed inline.

### Required reading

CODE.md (Error Handling, State Management), `.claude/LESSONS.md` §1.2/§5, issues #669, #675, #738.

---

## 9. refactor(config): consolidate configuration into one layered loader

Labels: `tech debt`, `code maintenance`, `state:proposed`, `priority:p2`

### Problem

Six-plus config mechanisms with no defined precedence: `src/config/constants.py` (423 lines of defaults), `src/config/feature_flags.py` (169), `src/config/config_manager.py` (248, env+dotenv), plus domain configs (`src/infrastructure/logging/config.py`, `src/ml/training_pipeline/config.py`, `src/prediction/config.py`) and **56+ direct `os.getenv`/`os.environ` call sites** that bypass ConfigManager entirely. Direct env reads are untestable, invisible to documentation, and several control money-affecting behavior.

### Scope of work

1. Define and document the layering: hard defaults (constants) → feature flags → env/dotenv via ConfigManager → runtime overrides. Write it in `docs/configuration.md`.
2. Migrate the 56+ direct `os.getenv` call sites to ConfigManager accessors (mechanical, one area per commit: engines, data_providers, dashboards, ml, infra). EXCEPTION: process-bootstrap reads before config exists (e.g., `DATABASE_URL` in alembic env, `CLAUDE_CODE_REMOTE` in hooks) — list exceptions explicitly in the docs.
3. Add a lint guard: ruff custom rule or a CI grep that blocks new `os.getenv` outside `src/config/` and the documented exception list.
4. Generate a complete config reference table (name, type, default, layer, consumer) into `docs/configuration.md` — script it so it can be regenerated.

### Acceptance criteria

- [ ] `grep -rn "os.getenv\|os.environ" src/ --include="*.py"` returns only `src/config/` + documented exceptions.
- [ ] CI guard demonstrated failing on an injected violation.
- [ ] No default-value changes (diff review: every migrated site keeps its exact default; any discovered default-mismatch between call sites is filed as a separate issue).
- [ ] Full suite green; paper-trading smoke run clean.

### Out of scope

Changing any config value or flag semantics; merging the domain configs themselves (they may stay as typed sections).

---

## 10. fix(precision): audit & enforce centralized quantization on all exchange-bound numerics

Labels: `tech debt`, `state:proposed`, `priority:p2`, `area:live-ops`, `type:fix`

### Problem

`src/trading/precision.quantize_to_step()` is the canonical fix for the float-artifact bug class that produced real order rejections twice (Binance 51077 on quantity, then -1111 on price — LESSONS.md §1.1, whose meta-rule is "grep ALL `round(... / ...) * ...` sites, don't fix only the one in front of you"). The audit found ~590 mixed Decimal/float usages across 15 live-engine files and call sites still doing inline `round()`. #745 (raw-float qty serializing as scientific notation) and #675 (Decimal×float coercion) are siblings of this class. Nothing structurally prevents the next regression.

### Scope of work

1. Systematic grep audit: `round(`, `/ step`, `* step`, `float(round`, manual `f"{x:.8f}"` formatting in `src/engines/live/`, `src/engines/shared/`, `src/data_providers/`. Catalog every site that produces an exchange-bound number (quantity, price, stopPrice, limit price).
2. Route ALL of them through `quantize_to_step` (extend it if needed for formatting, covering #745's scientific-notation case if still open).
3. Add a single choke point: the exchange-facing order submission boundary (in `LiveExecutionEngine` / provider `place_order`) asserts/normalizes every numeric param as a final guard, logging loudly if an upstream site sent an unquantized value.
4. Property-based tests (hypothesis is acceptable as a dev dep) feeding adversarial floats through entry/exit/stop paths asserting the serialized order params are exactly on-step/on-tick.
5. Document the rule in CODE.md's Arithmetic section if not already explicit.

### Acceptance criteria

- [ ] Audit catalog in the PR description: every site found, fixed or justified.
- [ ] Boundary guard active for both paper and live order paths, with a test that an unquantized upstream value is corrected AND logged.
- [ ] Property tests pass; the two historical repro values from LESSONS §1.1 (`1648.82` / `0.01` family) are explicit regression cases.
- [ ] No behavior change to position sizing itself (only final snapping).

### Required reading

`.claude/LESSONS.md` §1.1/§1.2, `src/trading/precision.py`, issues #745, #675, PRs #695/#696/#699/#701.

---

## 11. chore(config): feature-flag registry with lifecycle metadata

Labels: `code maintenance`, `state:proposed`, `priority:p3`

### Problem

`src/config/feature_flags.py` (169 lines) holds flags that gate money-affecting behavior with no lifecycle metadata: `ENABLE_PARTIAL_EXITS` is hard-disabled after incident #734 with no sunset plan; `ENABLE_DYNAMIC_RISK`, `ENABLE_TIMEOUTS`, `ENABLE_CORRELATION_ENFORCEMENT` have no documented owner, default rationale, or removal criteria. Flags that never die become permanent forks of the codebase.

### Scope of work

1. Add structured metadata per flag (dataclass): name, default, owner area, created date, reason/linked issue, status (`permanent` | `temporary` | `kill-switch`), sunset criteria for temporaries.
2. `atb dev flags` (or extend `atb dev quality`) prints the registry; CI fails if a flag lacks metadata.
3. Document each existing flag's actual current state — especially `ENABLE_PARTIAL_EXITS`: it must reference #734 and state that re-enabling requires that fix plus a proposal (the live partial-exit path places no exchange orders).
4. Add the lifecycle policy to `docs/configuration.md`.

### Acceptance criteria

- [ ] Every existing flag has complete metadata; CI guard proven.
- [ ] No flag default changes.
- [ ] `ENABLE_PARTIAL_EXITS` entry cross-links #734 and blocks accidental re-enable (e.g., enabling it logs a CRITICAL warning referencing the issue until #734 is closed).

---

## 12. chore(db): remove migrations/versions_backup/ (15 stale files)

Labels: `code maintenance`, `state:proposed`, `priority:p3`

### Problem

`migrations/versions_backup/` contains 15 stale migration files, including a duplicate of `0001_initial_schema.py` that also exists in the active `migrations/versions/` (12 files). The duplicate revision ids confuse Alembic tooling and block reliable `--autogenerate` use. Git history already preserves everything; a backup directory in the working tree is pure noise.

### Scope of work

1. Verify nothing imports from or references `versions_backup` (grep repo + alembic.ini + `migrations/env.py` `version_locations`).
2. Verify the active chain is complete: `alembic history` resolves base→head using only `migrations/versions/`; `alembic upgrade head` on a fresh Postgres (docker compose) succeeds.
3. Delete the directory. Note the removal in `docs/database.md` (migrations are recoverable from git history) and document the intended migration workflow (when to autogenerate vs hand-write).

### Acceptance criteria

- [ ] Fresh-DB `alembic upgrade head` + `atb db verify` pass in CI/integration tests.
- [ ] `alembic history` shows a single linear (or documented-branch) chain.
- [ ] No references to versions_backup remain.

---

## 13. chore(ml): model artifact retention policy + `atb models cleanup`

Labels: `code maintenance`, `state:proposed`, `priority:p3`, `training`

### Problem

`src/ml/models/` holds 24MB across ~54 version dirs and keeps growing with weekly training; nothing ever prunes. Cruft observed: `BTCUSDT/basic/2025-09-16_legacy/`, `BTCUSDT/sentiment/2025-09-16_legacy/`, duplicate `2025-12-23_21h_v2` dirs under both BTCUSDT and ETHUSDT sentiment. These ship in every checkout/build.

### Scope of work

1. `atb models cleanup --keep N --older-than DAYS --dry-run` (dry-run default ON; require `--execute` to delete).
2. Safety rails: NEVER delete the `latest` symlink target nor any version referenced by an active trading session in the DB; always keep ≥N most recent per {symbol}/{type}; refuse to run if a `latest` symlink is dangling (that's an error to surface, not clean up).
3. One-time cleanup PR: run it to remove the `_legacy` and duplicate `_v2` dirs (after confirming `latest` symlinks resolve elsewhere) — list exactly what's deleted in the PR body.
4. Wire `--dry-run` output into the weekly-training workflow as a report step (no auto-delete in CI).
5. Document retention policy in `docs/prediction.md`.

### Acceptance criteria

- [ ] Unit tests: keeps latest-target under all flag combinations; dry-run deletes nothing; dangling symlink aborts.
- [ ] Repo size of `src/ml/models/` reduced; all `latest` symlinks still resolve; `atb live-control list-models` clean.
- [ ] Charter rule untouched: cleanup never modifies any `latest` symlink.

---

## 14. chore(docs): archive completed phase docs, refresh stale tech-debt doc, clarify optimizer.yml

Labels: `code maintenance`, `state:proposed`, `priority:p3`

### Problem

Doc drift identified by audit: `docs/PHASE1_INTEGRATION.md`, `docs/PHASE1_RESULTS_GUIDE.md`, `docs/PHASE1_TESTING_SUMMARY.md` describe a completed milestone; `docs/technical_debt.md` is explicitly marked "historical snapshot (2025-11-21)" and predates the #754 static-analysis sweep; `docs/execplans/` (7 plans) and `docs/issues/` (4 docs) have no lifecycle marking; `.github/workflows/optimizer.yml` (39 lines) has an unclear trigger/purpose.

### Scope of work

1. Create `docs/archive/`; move the three PHASE1 docs + completed execplans/issue docs there with a one-line header noting archival date and superseding doc. Update any inbound links (`docs/README.md` index, grep for references).
2. `docs/technical_debt.md`: either re-validate each entry against current code (mark fixed/still-open with issue links) or archive it and point to the GitHub `tech debt` label as the living source. Prefer the latter — a static debt doc will always rot.
3. Define execplan lifecycle (active/closed) in `docs/README.md`.
4. `optimizer.yml`: read it, determine trigger and whether its job still functions; either document it in `docs/README.md`/workflow comments, or disable it via a PR with reasoning if dead.
5. Update `docs/changelog.md` per living-docs policy.

### Acceptance criteria

- [ ] `docs/README.md` index accurate; no link in docs/ 404s (script-check with a link checker or grep).
- [ ] technical_debt.md no longer presents stale findings as current.
- [ ] optimizer.yml documented or disabled with rationale.
- [ ] No content deleted — archived only (git history aside, keep the files).
