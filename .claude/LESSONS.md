# LESSONS.md

Hard-won lessons for agents working on this trading bot. Read before touching live-trading,
margin, precision, or deployment code. Append new lessons as they're learned (newest at the
bottom of each section). Each entry: the trap → the rule.

---

## 1. Codebase bug-classes to watch for

### 1.1 `round(x / step) * step` leaves float artifacts → Binance rejects the order
Snapping a value to an exchange step/tick with `round(x / step) * step` re-introduces float
error: `round(1648.82 / 0.01) * 0.01 == 1648.8200000000001`. Binance then rejects with **51077**
("precision over maximum", on **quantity** / LOT_SIZE `stepSize`) or **-1111** ("price has too
much precision", on **stopPrice**/limit price / PRICE_FILTER `tickSize`).
- **Rule:** after any step/tick snap, quantize to the step's decimal count via
  `src/trading/precision.quantize_to_step(value, step)`. Applies to **every** exchange-bound
  numeric param: order quantity, `stopPrice`, and limit `price`.
- **Meta-rule:** this bug came in pairs. Fixing quantity (51077, #695/#696) let the order through,
  which then exposed the *price* version (-1111, #699/#701) on the very next step. **When you find
  a float-artifact precision bug, `grep` for ALL `round(... / ...) * ...` sites — don't fix only
  the one in front of you.**

### 1.2 Decimal × float `TypeError` from DB-loaded fields
SQLAlchemy `Numeric` columns load as `Decimal`. Mixing a DB-loaded value with a `float` raises
`unsupported operand type(s) for *: 'decimal.Decimal' and 'float'`. Fired in prod every cycle (#673)
and in SL placement (#15). **Rule:** coerce with `float(...)` at use sites when combining
DB-loaded position fields with floats.

### 1.3 Recovery paths must FULLY re-register state, not just the obvious bits
The active-session recovery path reused the session id but never set
`db_manager._current_session_id` → every balance update silently failed with "No active trading
session for balance update" (P0, #693). The *inactive* path created a new session (which sets it);
only the *active* path was broken. **Rule:** when a recovery/restart path reuses existing state,
audit every collaborator the normal-create path initializes and re-register all of it
(`db_manager.set_current_session(...)`).

### 1.4 Orphaned margin borrow is never repaid by close-time logic
Closing a short repays via the `AUTO_REPAY` sideEffect on the cover order. But a **flat** bot fires
no close order, so a borrow with no tracked position (an "orphan") sits forever — it blocks new
shorts (the SHORT-entry guard rejects while free base > $1) and accrues interest. The periodic
reconciler even `return`s early when flat, so it never looked. Fixed by a guarded reconciler
**sweep** (#702/#703). **Rule:** margin "dust" is often a **borrow remnant**, not free inventory —
see 1.5.

### 1.5 Margin "dust" may be an un-repaid LOAN — REPAY it, don't SELL it
Boot log: `holds 0.0029 ETH (borrowed=0.00282625)`. The wallet held ETH **and** owed ETH (a short
covered by buying back without repaying the loan). Net ≈ flat. **Selling the held ETH would create
a real naked short** (you'd owe borrowed ETH with none held). The correct cleanup is Binance
**Repay**, which nets the held against the borrow. **Rule:** on margin, always distinguish `free`
vs `borrowed` vs `netAsset` before acting; "clear the dust" usually means *repay*, not *sell*.

### 1.6 Use the modern Binance margin endpoint
`client.repay_margin_loan()` hits the **deprecated** `/sapi/v1/margin/repay`. Use
`client.margin_borrow_repay(asset=, amount=, type="REPAY", isIsolated="FALSE")`
(`/sapi/v1/margin/borrow-repay`); **no `symbol`** for cross margin (symbol is isolated-only).
**Rule:** WebSearch/verify exchange API currency before wiring a call (the codex-review skill
mandates this).

### 1.7 Margin liabilities are ASSET-scoped, not symbol-scoped
A borrowed `ETH` is not tied to `ETHUSDT` — `ETHUSDC` shares the same base asset and the same
borrow. A symbol-scoped safety check can miss a same-base position. **Rule:** for any margin
repay/borrow safety logic, key everything (gates, locks, cooldowns) by **base asset**, and require
the invariant to hold for **every configured symbol sharing that base**.

### 1.8 `get_open_orders` fails OPEN (returns `[]` on API error)
A safety gate that reads "no open orders" off `get_open_orders` can be fooled by a transient API
failure (looks like "none"). **Rule:** for a *safety* decision, use a **fail-CLOSED** accessor that
returns a distinct "unknown" (`None`) on lookup failure, and treat unknown as "an order may exist →
skip". (See `has_open_orders`.)

---

## 2. Process mistakes I made (avoid these)

### 2.1 Don't ship a HALF-fix of a bug-class
See 1.1 — fixing quantity precision but not price precision shipped a still-broken bot. Grep the
whole class.

### 2.2 "Deduping to a shared helper" can introduce a bug if the helper differs
I replaced a local `_base_asset_of` (which stripped `USDC`) with a delegate to
`PositionReconciler._extract_base_asset` (which did **not** strip `USDC`) → `ETHUSDC` mis-grouped →
a gate could be bypassed. codex caught it by *running pytest*. **Rule:** when consolidating to a
"single source of truth", confirm the survivor's behaviour is a **superset** of every caller's
needs; don't assume the canonical copy is the correct one (here it was *missing* a case).

### 2.3 Renaming a method breaks source-inspection tests
Renaming `_execute_entry` → `_execute_entry_locked` (thin-wrapper refactor) broke
`test_margin_side_effect::test_trading_engine_stop_loss_auto_repay`, which did
`inspect.getsource(LiveTradingEngine._execute_entry)`. **Rule:** after renaming/moving a method,
`grep` tests for the method name — especially tests that assert on `inspect.getsource(...)`.

### 2.4 Think through timing/races before recommending a "safe" sequence
I recommended "wait until the bot is flat, then flip the flag" — but the bot trades autonomously
and a Railway restart takes ~3 min, so you **can't reliably catch a flat window**. The actual
safety net for restart-with-a-position is **re-adoption (#677) + the per-symbol dedup guards**, not
flat-timing. **Rule:** before proposing a "do X only when state Y" sequence on a live autonomous
system, check whether you can actually *hold* state Y across the action; usually the resilient
mechanism (recovery/idempotency), not timing, is what makes it safe.

### 2.5 Verify agent/automation claims against live state
A recurring cron prompt asserted a "real ETH LONG orphaned at 19:12 (double-exposure risk!)". It
was a **phantom**: no such SL order existed, account sync showed `0 open orders`, the held ETH was
sub-threshold dust. **Rule:** treat automated/stale context as a hypothesis; confirm against live
state (`get_open_orders`, account sync, the actual order id) before acting or alarming.

---

## 3. Operational / tooling gotchas

- **Permission prompts:** the sandbox is already disabled (`.claude/settings.local.json`
  `sandbox.enabled:false`). Passing `dangerouslyDisableSandbox:true` then forces a redundant
  "dangerous override" prompt on *every* command. **Don't pass it.** Add recurring tools to
  `permissions.allow` (`railway`, `codex`, `black`, `ruff`, …). Keep live-deploy commands in
  `permissions.deny` (`railway variables --set`, `railway ssh`, `railway run`, `redeploy`/`up`/
  `down`/`delete`). Owner's rule: **only ask before deploying something live.**
- **`railway logs`:** `--since <N>m` hangs — use `railway logs -n <N>` (bounded). It shows **only
  the current deployment**, so a brand-new deploy's logs replace the old one's.
- **`railway variables --set` triggers a redeploy** (a restart). Setting a feature flag = a restart.
- **Feature flags** resolve from `FEATURE_<UPPER_SNAKE_KEY>` env vars (e.g.
  `FEATURE_ORPHANED_BORROW_SWEEP_MODE`); `get_flag(key, default)` returns the string; no
  `feature_flags.json` needed. Default lives in code (keep money-movers default-OFF).
- **Clock skew:** `ScheduleWakeup` displays **GMT+1**; prod logs are **UTC**. A wakeup labelled
  "01:31" fires at 00:31 UTC. Always compare `date -u` to the log timestamp before declaring the
  bot "down" (this caused a false alarm).
- **Bare numeric greps on timestamped logs false-positive:** `grep 51077` matched a nanosecond
  suffix `...243651077Z`. Anchor error-code greps (`code=-1111`, `(code=51077)`).
- **Surgical prod promote (NEVER wholesale `develop → main`):** branch off `origin/main`,
  `git cherry-pick <develop-squash>`, verify `git patch-id --stable` matches, PR → `main`, merge.
  `develop` carries unpromoted backlog; a wholesale merge would ship it. develop uses **squash**
  merges.
- **Primary checkout is live:** `/Users/alex/Sites/ai-trading-bot` is the `main` (production)
  checkout — never run git mutations there; work in a worktree under `.claude/worktrees/`.
- **codex CLI:** runs sometimes hang or crash (exit 144). Recover with `pkill -f codex` + re-run.
  **Always close stdin** (`< /dev/null` or pipe to `tail`) or it hangs on
  "Reading additional input from stdin…". Run scoped from a neutral cwd with
  `--skip-git-repo-check --model gpt-5.5`.
- **`railway` resolves the project link by cwd** — run `railway logs`/`status` from the linked repo
  dir (worktree root), NOT `/tmp` (an unlinked dir errors `No linked project found`). Redirect
  output to `/tmp`, but run the command from the repo.
- **Prod logs use 4-char level tags** `[ERRO]`/`[WARN]`/`[INFO]` — NOT `[ERROR]`. Grep `\[ERRO\]`
  or severity is undercounted.
- **Target prod explicitly:** `railway logs -e production -s "Trading Bot" -n 400` (env is
  `production`, the live bot service is `Trading Bot`).

---

## 4. Patterns that worked for risk-critical (money-moving) changes

- **codex-review loop until APPROVE.** For live-capital logic, run `/codex-review` (gpt-5.5) and
  iterate until clean. It found real bugs this session that local review + CI missed (cooldown
  lifetime, the USDC grouping regression, a fail-closed gap) — partly because **it runs pytest**.
  Don't merge money code on a single review pass.
- **Ship inert, validate on real data.** New money-movers default to **dry-run / OFF**. Promote the
  code (no behaviour change), then flip a flag to **dry-run** to *detect + log* against the real
  account ("would repay X — all gates passed") with **zero money moved**, and only then flip to
  **active**.
- **Fail-closed gates.** Refuse to act on uncertainty: unknown order state → skip; missing
  serialization lock in active mode → refuse to repay; `free < borrowed+interest` → skip. Use
  `Decimal` (from raw exchange strings) for any liability amount — never float-round a loan.
- **Thin lock-wrapper refactor.** To wrap a large method in a lock without a risky 300-line
  re-indent: rename the body to `_method_locked`, add a thin `_method` that acquires the lock and
  calls it. Use a **re-entrant** lock (`RLock`) so a held path can call a nested one (entry →
  SL-fail → emergency-close → exit) without deadlock.
- **Re-adoption (#677) makes restarts-with-a-position safe.** On restart the new process reloads
  open positions into the tracker, and the per-symbol + max-concurrent entry guards then prevent a
  duplicate. This — not "restart only when flat" — is the orphaning defense (#668 was the
  re-adoption-failure bug).
- **Plan-mode + multi-round codex on the plan** before coding a risk-critical feature. The
  orphaned-borrow sweep plan went 7 → 3 → 1 → 0 codex findings *before* a line was written, which
  caught the asset-vs-symbol scoping and the entry-vs-sweep TOCTOU up front.

---

## 5. Live-monitoring signatures (what to grep for)

The `bot-monitor-live` skill is the durable *method* for watching production; **this section is the
evolving list of *concrete* signatures it greps for.** Add new ones here as incidents teach them —
keep the skill generic and let the specifics live here.

### 5.1 Pull logs + judge liveness safely
- `railway logs -n 400` (bounded). **Never `--since <N>m`** — it hangs. Shows ONLY the current
  deployment, so a boot marker in your window may be an expected deploy OR a surprise restart.
- **Clock skew** (also §3): logs are **UTC**, your wall clock / wake scheduler may be **GMT+1**.
  Compute `date -u` minus the last log timestamp before declaring "down" — usually a 1h illusion
  (~2 min `Decision:` cadence).
- **Don't trust the deploy API for liveness.** Railway can show SUCCESS while the loop is dead (a
  DB/DNS outage killed it — see `MEMORY` bots-down-railway-dns). Ground truth = a recent
  `Decision:`/`Status:` log line **and** the hourly `account_history` heartbeat row in the DB.

### 5.2 Escalate immediately (critical markers)
- `emergency.close` / "Stop-loss placement failed" — opened a position it couldn't protect; repeated
  = capital-bleed churn.
- `CLOSE-ONLY MODE ACTIVATED` — entries halted (reconcile/DB problem).
- `code=-1111` (price precision) / `code=51077` (qty precision) — order precision rejection (should
  be fixed; recurrence = regression, see §1.1).
- `-2010` / "insufficient balance" on a stop-loss → unprotected position.
- "No active trading session for balance update" — balance updates silently failing (§1.3, #693).
- `Margin position check failed` — reconciler erroring every cycle (§1.2, #674).
- A **new** position opened while an **untracked**/orphaned position may be live → double-exposure.
- Unexpected `AI Trading Bot Starting` (a restart you didn't cause) — may re-orphan; watch recovery.
- kline WS churn that never returns to WS-primary; any `Traceback`; margin level drifting toward
  ~1.0–1.1 (liquidation) or reported balance diverging from true equity (phantom balance).

### 5.3 Watch / report (non-critical)
- A growing **orphaned margin borrow** (`borrowed=` with no tracked position) — blocks shorts +
  accrues interest (§1.4 / §1.5).
- `Task exception was never retrieved` — a swallowed async error; benign as a one-off at boot, report
  if it recurs / clusters.
- Sustained idleness when the bot *should* trade (sub-minimum sizing on a small account, #700).

### 5.4 Known-benign — do NOT alarm
- A **tracked** `Positions: 1` is normal trading, **not** double-exposure — only an *untracked*
  orphan is. A position surviving a restart (`new opens = 0`, `Positions` stays `1`) = re-adoption
  worked (#677) — that's GOOD.
- `🔍 DRY-RUN orphaned-borrow sweep: would repay …` ~every 5 min — expected `[WARN]`, log-only, no
  money moved.
- `Calculated quantity 0.00000000 below minimum` — a sizing *skip* on a small account (#700); logged
  at ERROR but it's the bot correctly declining a sub-minimum trade.
- `Cannot open short … margin wallet holds … ETH` (#697) — the SHORT guard refusing while base-asset
  dust sits in margin; expected until the borrow is repaid.
- `51077` matching a **timestamp nanosecond suffix** (`…243651077Z`) — anchor the grep (`code=51077`,
  `(code=51077)`); don't match bare digits in UTC timestamps.
- An expected deploy boot (one you / the operator just triggered).
- `[WARN] New order found on exchange: <id> … / [INFO] Skipping creation of new order <id> from
  sync` — `account_sync` re-detecting the resting stop-loss each cycle and (by design) not
  duplicating it into the DB. Benign; that order IS the protective SL.
- `[ERRO] Task exception was never retrieved → KeyError('margin_subscription:0')` (binance
  `threaded_stream.py:74`) at user-stream circuit-open — benign teardown noise (fix tracked in #716;
  disappears once it ships).
- The kline self-heal sequence (`Kline WS error: Connection closed` → `RESYNCING` → `KlineBuffer
  resynced … candles` → `Kline WebSocket recovered … WS primary again (#662)`) — the GOOD path,
  ~30 s, no data gap. Do NOT alarm.

### 5.5 Verify before alarming (phantom premises)
A recurring/cron prompt may assert e.g. "a real ETH LONG was ORPHANED at 19:12 → double-exposure!".
This has repeatedly been a **phantom**: no SL order existed, account sync showed `0 open orders`, the
held ETH was sub-threshold dust. Treat automated/stale context as a hypothesis; confirm against live
state — `get_open_orders`, account-sync open-order count, the actual SL order id, tracked `Positions`,
and `free` vs `borrowed` vs `netAsset` — before reporting an incident (see §2.5).
