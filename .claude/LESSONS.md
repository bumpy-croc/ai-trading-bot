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

### 1.9 An alerting/SLO budget reused as an execution/abort threshold silently corrupts output
`PredictionEngine` used `max_prediction_latency` — a **0.1s alerting SLO** — to *abort* inference
and to *invalidate* completed-but-slow predictions. Under CPU contention a fraction of bars silently
returned error results with `price=0.0`; both `_get_ml_prediction` impls read `result.price` without
checking `result.error`, so `price=0.0` → `predicted_return=-1.0` → **SELL strength 1.0 / confidence
1.0**. Latent LIVE risk (empirically reproduced against unmodified generator code), and the root of
backtest **non-determinism** (identical baseline → 46 vs 55 trades run-to-run). Fixed via #923
(`InferenceContext` LIVE/backtest split so the latency gate only applies live + a generator error
guard); the alerting-budget comment now lives at `engine.py:302`.
- **Rule (a):** never reuse an observability/alerting budget as a control/abort threshold — they
  have different owners and magnitudes; the real hang guard belongs elsewhere (`OnnxRunner` already
  carries a 30s guard, so the engine gate was a redundant double-gate on the *wrong* constant).
- **Rule (b):** a prediction consumer MUST check `result.error` before reading `result.price` — an
  error result's `price=0.0` is not a real price. Earned: #912 (addendum), #913, #923.

### 1.10 Silent wrong-source execution — exam/backtest numbers computed against the wrong code, cwd, or model
Three separate instances this week, one meta-class: a run produced numbers, but not from the source
it claimed.
- **(a) Shared-venv `atb` staleness.** The `atb` console script is an editable install pinned to
  ONE worktree; bare `atb` from any *other* worktree silently executes that pinned worktree's code
  (workaround + detail in §3). Sibling filed as GH #999 (script-path `sys.path` shadowing of the
  main checkout's stale `src/`).
- **(b) cwd-relative registry path.** `DEFAULT_MODEL_REGISTRY_PATH = "src/ml/models"`
  (`constants.py:25`) resolves against *process cwd* — an exam launched from the wrong directory
  silently resolved to an empty registry and produced an all-HOLD / 0-trade result that looked like
  a real (null) finding (2026-07-11). Still unfixed — anchor to module/repo root (GH #1023).
- **(c) unthreaded `config.symbol`.** `ExperimentRunner._load_strategy` never threaded
  `config.symbol` into the strategy factory → the entire EXIT-GEOMETRY round 1 study (#970/#971)
  and PR #976's own regression-evidence table scored ETHUSDT candles with **BTCUSDT's** model
  (#997, fixed #1004; #998 re-verifies round 1).
- **Meta-rule:** before trusting any exam/backtest number, verify its provenance — which code
  (worktree + venv), which cwd, which model **version and symbol** actually ran. A degenerate
  result (0 trades, all-HOLD, or **bit-identical blotters across supposedly-different inputs**) is a
  provenance-failure *signature*, not a finding. Round 2's 4-way isolation test — which reproduced
  round 1's exact published number ONLY when both the bug AND the wrong worktree were present — is
  the verification pattern to copy.

### 1.11 A sizing/confidence channel that silently ignores the signal makes a model-comparison exam measure the harness, not the model
Three instances converge on one trap: the exam's P&L was invariant to the model because the money
path never read the signal.
- **(a)** `FlatRiskManager.calculate_position_size` (`hyper_growth.py:97-118`) returns
  `balance × risk_fraction` with "no confidence/strength scaling" — position size is invariant to
  an 8x `predicted_return` range, so the architecture tournament's five genuinely-different models
  (max ONNX abs diff 0.31) produced a **bit-identical trade blotter** that looked like model
  equivalence but was a harness-validity defect (#938). This is a *deliberate* live design choice,
  not a bug to fix — the lesson is about **exam design**, not `hyper_growth.py`.
- **(b)** The `"confidence_weighted"` sizer-type string reads a `"prediction_confidence"` indicator
  **nothing in the codebase populates** → silently zeroed every trade for all four target-redesign
  entrants until caught by the end-to-end dry-run (#949/#950; fixed to `"fixed_fraction"`).
- **(c)** `confidence = clip(|predicted_return|×12, 0, 1)` feeds a boolean gate with
  `adjust_for_confidence=False` → recalibration can only change *which* bars trade, never their
  size (#912).
- **Rule:** any exam that ranks models by P&L must first prove its sizing path *varies with the
  signal* (sweep one model's predicted-return range and confirm position size responds), and the
  preregistration must **declare the exact RiskManager/PositionSizer** it scores through (#938
  implication). Ranking model quality through a flat/inert sizer is only ever a directional-sign
  test, never a magnitude one. Earned: #912, #938, #949/#950.
- **Live recurrence (signature, not yet root-caused):** staging emitted 30 minutes of BUY decisions
  at `Size: 0.00` with **no `gate_reason` or any other log line naming what zeroed them**, and went
  10 days without an entry (2026-07-25). Treat *"signals firing, size invariant at zero, nothing
  logged"* as this section's live signature — and note the second defect it exposes: a zero-size
  decision must never be silent (GH #1045).

### 1.12 A metadata key that switches a transform, absent, silently disables the transform
`atb train cloud` writes a `metadata.json` **without** `price_normalization` (the local training
path in `cli/commands/train_commands.py:280-281` is the only writer). The prediction path treats
that key as the denormalize switch and falls through to "return as-is" when it is missing —
`src/prediction/models/onnx_runner.py:534` (`if ...get("price_normalization"): pred =
self._denormalize_price(pred)`) and `src/prediction/engine.py:1183` (returns unchanged when
`method != "rolling_minmax"`). Neither raises. A model whose output is in normalized ~[0,1] space
is then compared against real ETH prices (~$3k): every prediction wrong, every artifact
structurally valid. Same silent-fabrication class as the pre-#838 partial-exit units bug — and
`cloud-promote --set-latest` would have pointed a live strategy at it with no external check,
since model promotion for a live symbol is autonomous under the charter.
- **Rule (a):** when a metadata/config key **selects a transform**, its absence must **fail loud**
  at load time, not fall through to identity. "Key missing → skip the step" is only safe when
  skipping is the documented default; for a denormalization it never is.
- **Rule (b):** when two producers write the same artifact schema (local vs cloud training), they
  share a **writer**, not a convention. A second copy of the literal drifts silently, and the
  consumer's fall-through hides the drift.
- **Detection:** diff the `metadata.json` key sets of two bundles of the same task type before
  trusting any head-to-head number. Identical `feature_schema.json` + `feature_names` does **not**
  mean the bundles are interchangeable. Earned: GH #1049 (found during the 2026-08-09 retrain,
  #1048 — it blocked the backtest rather than corrupting it, which is the good outcome).

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

### 2.6 Conflicted PRs run NO CI — and that looks like "CI is stuck"
A push to a PR branch whose merge with the base is conflicted (`mergeable_state: dirty`)
produces **zero** check runs: `pull_request`-triggered workflows run on the merge ref, which
GitHub cannot create. Nothing fails — checks simply never appear. **Rule:** when a PR push
shows no checks after a few minutes, check `mergeable_state` first (the base may have moved),
rebase onto the fresh base, and force-push; don't wait on or re-trigger phantom CI.

### 2.7 "Component-complete ≠ runnable" — scaffolding isn't done until each consumer runs end-to-end through the REAL CLI
PR #948's target-tournament Phase 2 scaffolding was fully unit-tested but **not reachable
end-to-end via any CLI entry point** — the tournament halted until #950 threaded the flags,
registered the exam strategies, and added one non-mocked, subprocess-based dry-run acceptance test
per consumer (`tests/integration/tournament/test_entrant_dry_runs.py`: synthetic OHLCV → train via
the real CLI → correctly-registered/timeframed artifact → exam backtest via the real CLI with
version pinning → ≥1 real trade). Those tests then caught **5 further real bugs** the unit suite
could not see (most consequential: the inert `"confidence_weighted"` sizer, §1.11b). **Rule:** the
definition-of-done for multi-piece scaffolding includes a **per-consumer end-to-end dry-run through
the actual CLI entrypoints**, not just green unit tests. Unit-green + CLI-unreachable is not done.
Earned: #948 → halt → #949/#950.

### 2.8 A review BOT's inline comment is a finding — harvest it, don't read only its pass/fail status
A real #838-class units bug sat as an inline `claude[bot]` comment on PR #948 while **both**
dispatched reviewer agents missed it; the merge flow read only the bot's overall check status
(green), so the finding was invisible until the #950 wiring round independently re-found it by
running the code. **Rule:** every merge flow and PR-review disposition runs
`gh api repos/OWNER/REPO/pulls/<N>/comments` and explicitly dispositions **each** bot finding — a
bot's green summary status does **not** mean zero inline findings. (Extends CLAUDE.md's "Handling PR
Review Comments".) Earned: #948.

### 2.9 Distillate deferred to someone else's PR is distillate lost — a lessons PR must land on its own
The 2026-07-13 retro (PR #1026) bundled its LESSONS/skill distillate with a 52-line PM-directed
`log.md` consolidation. The consolidation conflicted with every subsequent log append, so the PR sat
CI-green but unmergeable. The 2026-07-20 retro then *deliberately did not reproduce* that distillate
— to avoid a double-append if #1026 later merged — and instead recorded a per-item map plus "nothing
is lost **provided this PR lands**". #1026 was **closed unmerged** on 2026-07-21. Eleven agenda
items' worth of reviewed lessons (§1.9–1.11, §2.7–2.8, §3 bullets, the delegation-protocol and
deploy-prod amendments) evaporated, and only a line-by-line re-derivation at the next retro
recovered them.
- **Rule (a):** a retro/lessons PR is **self-contained**. Never make this week's distillate
  contingent on a PR you do not control merging. If a prior distillate PR is stuck, re-land its
  **distillate-only** subset in yours (a duplicate append is a 2-minute fix; a lost lesson is
  invisible forever) — the double-append hazard is strictly cheaper than the loss.
- **Rule (b):** "flagged on the PR + recommended NEXT ACTION for the PM" is not a disposition. An
  item is dispositioned only when the artifact exists on `develop`.
- **Rule (c):** **closing a PR does not dispose of its content.** Before closing any PR carrying
  reviewed distillate, diff its non-conflicting files against the target branch and confirm each one
  either landed elsewhere or is being deliberately dropped, in writing.
  Earned: #1026 (closed 2026-07-21), recovered by the 2026-07-27 retro.
- **Rule (d) — added 2026-08-10, after this failed a third time.** Rules (a)–(c) fix *recovery* and
  still assume someone eventually merges. They don't. #1026 closed unmerged; **#1047 (the retro that
  wrote rules (a)–(c)) then sat `CLEAN`, CI-green, zero conflicts, unmerged for 14 days** — so
  §2.9/§2.10 themselves were not on `develop` while the very failures they describe recurred, and
  the standup that re-detected them on 08-03/08-04 could not have read §2.10's escalation corollary.
  When a producer cannot merge its own output, "be more self-contained" is the wrong layer of fix:
  it is an **ownership defect**, not a discipline defect.
  - **Mechanical fix:** if the previous retro's PR is still open, **branch this retro off that PR's
    head** rather than off `develop`. The new PR is then a strict superset — it merges whether or
    not the old one lands, and git dedupes if both do. Costs one `git reset --hard origin/<branch>`.
  - **Escalation fix:** the *second* consecutive stranded distillate PR is no longer a retro finding
    to re-land quietly — **name it to the human in the completion summary as the top item.** The
    retro's only channel to a decision-maker is that summary; an unmerged PR queue is invisible
    everywhere else.

### 2.10 A monitoring run that writes nothing durable did not happen
Between 2026-07-20 and 2026-07-27 the scheduled fleet ran ~25 times (`daily-trading-standup` 8/8
days, `alert-monitor` 6-hourly, `staging-cohort-observer` 1–3x/day) and `log.md` gained **zero**
entries. Two real findings were surfaced and then lost: staging's phantom `OPEN` position rows —
independently re-observed **three times** (07-25T12:10Z, 07-25T20:11Z, 07-27T08:02Z) without
escalating — and 10 days of silent zero-size decisions. Both existed only in session memory
(claude-mem), which is **layer 4**: not swept by `/triage`, not read at PM session boot, and
consumed by exactly one thing — the weekly retro. Detection latency was therefore a full week.
- **Rule:** any monitoring/scheduled pass that surfaces a non-nominal finding must, **in the same
  run**, emit a layer-2 artifact — a `log.md` append (`decision-record`), an incident file, or a GH
  issue. A finding that lives only in the run's own summary or in session memory is not reported.
- **Corollary:** re-observing a finding a second time is an *escalation* trigger, not a re-report;
  if the previous run already saw it and nothing changed, that is now a process failure to name.
  Earned: 2026-07-21→27 log silence; GH #1044, #1045, #1046.

### 2.11 Filing an issue is not delegating the work
The 2026-07-27 retro filed #1044, #1045, #1046 and commented on #1041, #1038. Fourteen days later
**all five had zero activity** — no owner, no comment, no branch. This was already visible once (the
07-20 retro's "#1041 just needs a rebuild": accurate diagnosis, issue filed, still blocked at the
next retro) and was recorded as a calibration miss rather than a rule. It is now a five-for-five
pattern across two retros, against a backdrop of **zero code merged to `develop` in 27 days** (last
code commit 2f6c1fe8, 2026-07-14 — everything since is docs/state).
- **Rule:** an issue with no assignee and no dispatched agent is a **note to yourself**, not work in
  progress. Do not count it as a disposition, and do not report it as "handled". Per this repo's
  CLAUDE.md the intended pattern is *file the issue **and dispatch a subagent to it***; a filed-only
  issue is half of that.
- **Corollary:** when the same issue is still unowned at the next weekly pass, stop re-filing and
  re-describing it — escalate the *queue* (N issues, M days, no owner) as one item. The backlog
  depth is the finding, not any individual issue.
  Earned: #1041/#1038/#1044/#1045/#1046 untouched 2026-07-27→08-10.

### 2.12 A maintenance canary with no refill procedure inverts into a permanent CI tax
`test_default_config_has_upcoming_coverage` asserts `config/macro_events.json` lists an event within
the last 14 days — a deliberate canary from #962, whose own docstring says the guard otherwise
*"silently stops de-risking anything."* The calendar went stale on 2026-07-14; the canary began
failing around **07-28 and then failed every PR to `develop`**. It was a good test firing correctly,
on time, with a message naming the file and the fix. Nothing consumed it for 12 days, because it had
no owner and no refill procedure. Two costs, and the second is worse:
1. the real one — 26 days with **no upcoming macro de-risk coverage on live capital** (#1053);
2. **every** PR showed red CI, so red became the resting state and a genuinely broken PR was
   indistinguishable from the background failure. #1048 is red solely because of this and is
   otherwise a one-line docs change.
- **Rule:** a canary that gates **all** PRs needs a named owner and a scheduled refill, shipped *with
  the canary*. "It'll fail loudly and someone will fix it" is the assumption that fails — cf. §2.11.
- **Rule:** when a repo-wide check has been red for more than a couple of days, treat "is CI red for
  a reason unrelated to this PR?" as a first-class finding, not as noise to route around. A
  permanently-red check is a **disabled** check.
- **Design note:** prefer *warn* over *fail* when the staleness is in data the PR does not touch and
  the guarded code path is itself healthy — so calendar rot cannot block unrelated work.
  Earned: GH #1053 (found via #1048's `unit-tests (4)`), #962.

---

## 3. Operational / tooling gotchas

- **Permission prompts:** the sandbox is already disabled (`.claude/settings.local.json`
  `sandbox.enabled:false`). Passing `dangerouslyDisableSandbox:true` then forces a redundant
  "dangerous override" prompt on *every* command. **Don't pass it.** Add recurring tools to
  `permissions.allow` (`railway`, `codex`, `black`, `ruff`, …). Keep live-deploy commands in
  `permissions.deny` (`railway variables --set`, `railway ssh`, `railway run`, `railway domain`,
  `redeploy`/`up`/`down`/`delete`). Owner's rule: **only ask before deploying something live.**
  **Known gap (2026-07-09):** `.claude/settings.local.json` is gitignored/per-checkout, so this
  bullet is a *practice*, not an enforced control — as of the 2026-07-08 `railway domain`
  incident (GH #941) at least one checkout's `permissions.deny` was empty and separately
  *allowed* the mutating MCP tools `mcp__Railway__set-variables` / `mcp__Railway__deploy`. Verify
  your own checkout's `settings.local.json` matches this bullet before relying on it; don't
  assume it does.
- **`railway domain` is get-or-create, NOT read-only** — running it with no arguments to "check"
  whether a service already has a public URL instead *creates* one if none exists, with no
  dry-run and no confirmation prompt in non-interactive use. This created an unauthorized public
  domain on the production Trading Bot service on 2026-07-08 (incident
  `2026-07-08T2015-P2-unauthorized-public-domain`, GH #941) — served unauthenticated `/health`
  and `/status` for ~14h before the PM removed it via the Railway GraphQL API
  (`serviceDomainDelete`). No capital/trading impact, but an unapproved new internet-facing
  surface on a live-capital system. **To check for an existing domain, use `railway status
  --json`** and read
  `.environments.edges[].node.serviceInstances.edges[].node.domains.serviceDomains[]` — never
  `railway domain`.
  - **Canonical Railway CLI safe/prohibited list** (verified against `railway <cmd> --help`, CLI
    v4.30.5 — re-verify against a current `--version` before trusting this if the CLI has been
    upgraded):
    - **Safe / read-only:** `railway status [--json]`, `railway logs [-n N] [-e ENV] [-s SERVICE]
      [--json]`, `railway whoami [--json]`, `railway list [--json]`, `railway deployment list
      [...] [--json]`, `railway variable list` / bare `railway variables` (no `--set`/
      `--set-from-stdin`), `railway service status`, `railway service logs`, `railway
      environment config`, `railway project list`.
    - **Hard-prohibited for read-only/monitoring agents (confirmed mutating, no dry-run):**
      `railway domain` (any form), `railway up`/`deploy`/`redeploy`/`restart`/`down`/`delete`,
      `railway service` redeploy/restart/scale/link (or bare `service <NAME>`, a deprecated link
      form), `railway environment` new/delete/edit (or bare `environment <NAME>`, which links),
      `railway variable set`/`delete` (or the legacy `--set`/`--set-from-stdin` flags), `railway
      link`/`unlink`, `railway init`/`add`, `railway connect`/`ssh`/`run`/`shell` (opens a live
      shell or pulls prod credentials into a local process — treat as mutating-capable regardless
      of intent), `railway volume`/`functions`/`scale`.
    - **Rule:** before running any `railway` subcommand not on the safe list, run
      `railway <subcommand> --help` and confirm from the help text it cannot create, modify, or
      delete a resource. If in doubt, don't run it — escalate instead. `deploy-prod`,
      `deploy-staging`, and `kill-switch-drill` skills deliberately use mutating commands
      (`railway variables --set`, redeploy) as pre-committed, authorized actions within their own
      playbooks — that's a different, sanctioned use case from an agent reaching for a mutating
      command during what's supposed to be a read-only pass.
  - Mirrored in `.claude/agents/live-ops.md` and `.claude/skills/bot-monitor-live/SKILL.md`
    (the two places most likely to run a Railway command during a "just checking" pass).
- **`railway logs`:** `--since <N>m` hangs — use `railway logs -n <N>` (bounded). It shows **only
  the current deployment**, so a brand-new deploy's logs replace the old one's. Confirmed
  empirically (2026-07-06, #913 forensics): no `--since` value reaches prior containers, so
  **historical incident forensics must come from the Postgres tables** (`strategy_executions`,
  `system_events`, …) via the read-only public proxy — logs are gone once a deploy/restart lands.
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
- **Never write a credential/token to a file.** An agent saved a live ECR authorization token to a
  plaintext scratchpad file during a cloud-training pass (2026-07-10; PM caught + deleted it before
  it persisted). **Rule:** never persist a secret to disk, scratchpad, or a logged env-var — pipe it
  in one command: `aws ecr get-login-password --region <r> | docker login --username AWS
  --password-stdin <registry>`. No intermediate file, no echo, no copy.
- **Shared-venv `atb` staleness** (bug-class §1.10a): `make install` registers the `atb` console
  script as an editable install pinned to **whichever worktree ran it** — bare `atb` from any
  *other* worktree silently executes that pinned worktree's `src/`, not your cwd's. When running the
  CLI from a non-primary worktree, invoke it as `PYTHONPATH=<worktree-root> python3 -m cli.__main__`
  (or re-`make install` in the worktree, but that repins it for everyone). A warn-on-mismatch fix
  (cwd repo-root ≠ installed source root) is GH #1024; sibling GH #999 covers the script-path variant.
- **`ls ~/.claude/scheduled-tasks` is NOT the task list — the scheduler registry is.** The directory
  holds a `SKILL.md` per task and **keeps it after the task is deregistered**, so a dead task looks
  installed forever. On 2026-08-10: 19 directories, **13 registered tasks**. `alert-monitor`,
  `staging-cohort-observer`, `eod-worktree-prune` and `pm-fleet-watchdog` had vanished from the
  registry and last ran 2026-07-28/29 — including `alert-monitor`, the operator-alert watchdog for a
  **live-capital** bot, dark for 12 days. Three consecutive retros audited tasks with `ls` and
  reported "no task missed its schedule."
  **Rule:** audit with `mcp__scheduled-tasks__list_scheduled_tasks` and **diff registry ⇄ directory
  both ways** — a directory with no registry entry is a DEAD task; check `enabled` and `lastRunAt`,
  not just presence. (`prune-worktrees` is a *separate live task* from the dead `eod-worktree-prune`
  directory — don't read one as evidence for the other.) Earned: GH #1050.
- **A persisted model-provider selection silently kills every scheduled task.** `switch-model-provider`
  writes the model choice to settings, and scheduled runs inherit it. When the selection is
  unavailable the run dies **on turn 1** with `There's an issue with the selected model (<id>). It
  may not exist or you may not have access to it.` — no retry, no alert, and the transcript is ~20
  lines so it looks like a short successful run. Cost: the **2026-08-03 weekly retro** (`glm-5.2[1m]`)
  produced nothing and nobody noticed for a week; `daily-trading-standup` died the same way on 08-04
  and 08-05 (`glm-4.7`).
  **Rule:** after any provider switch, confirm the next scheduled run actually produced its artifact.
  When auditing tasks, grep transcripts for `may not exist or you may not have access` — a fired-but-
  died run is invisible in `lastRunAt`, which records the *attempt*. Earned: GH #1051.

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
- `ACCOUNT CIRCUIT BREAKER TRIPPED` / `error_code=ACCOUNT_CIRCUIT_BREAKER_TRIP` /
  `risk_event=account_circuit_breaker_trip` (#807) — the daily-loss (2.5% of the UTC-day baseline)
  or drawdown (15% peak-to-trough) hard halt fired; account is close-only for the day. Operator
  reviews & clears. `🟡 ... WOULD HALT (dry_run)` is the pre-enablement dry-run signal (report, not
  escalate).
- `code=-1111` (price precision) / `code=51077` (qty precision) — order precision rejection (should
  be fixed; recurrence = regression, see §1.1).
- `-2010` / "insufficient balance" on a stop-loss → unprotected position.
- `MANUAL SYSTEM HALT ENFORCED` / `error_code=SYSTEM_HALT` (#922) — the manual kill-switch
  (`atb live-control halt`) is in force: entries + scale-ins blocked, exits/stops continue.
  Expected if an operator just pulled it (a `SYSTEM_HALT_COMMAND` event precedes it); if nobody
  did, investigate WHO wrote the `system_control_flags.system_halt` row. `SYSTEM_HALT_CLEARED`
  = resumed.
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

### 5.6 Pre-2026-06-03 `account_history` equity is BOOK VALUE, not a live read — distinct-count check before trusting any peak
A drawdown/peak analysis over prod `account_history` reported "20% cap already breached (20.33% from
a $103.82 April peak)". **Phantom**: the `balance` base was software-pinned at the optimistic $100
`session_start` value through Mar–May 2026 (May: ONE distinct balance value across 451 hourly rows);
`equity` = frozen book + unrealized wiggle. True margin-equity reads only begin 2026-06-03 (#655 sync,
$84.14). Drawdown baseline policy (pm, 2026-07-04): peak = peak TRUE equity since the last reconciled
reset (2026-06-05 / session 20).
- **Rule:** before treating any `account_history` peak/trough as real, sanity-check that `balance`
  varies like a market-tracking value: `count(DISTINCT round(balance,4))` per month. A pinned or
  near-constant base means book value — do not compute drawdowns across it.

### 5.7 A stuck flag / frozen loop emits ZERO events — assert expected STATE, not just the absence of alarms
Prod `FEATURE_ENTRY_PAUSE` sat stuck `true` for ~95h (2026-07-13 → 07-17): a one-shot `cpi-pause-off`
task never fired (app closed at fire time), so entries stayed disabled long past the intended window.
The tell that every event-stream monitor **missed** it: prod wrote **0 `system_events` of any kind**
for the full 95h, and both `daily-trading-standup` and the 6-hourly `alert-monitor` scan the event
stream — so *silence read as health*. It was caught only by a manual status sweep.
- **Rule:** a health check must assert **expected positive state**, not merely the absence of error
  events. Concretely, per sweep verify: (a) `FEATURE_ENTRY_PAUSE=false` unless a live pause window is
  logged; (b) entries-per-window is non-zero when signals fired flat (0 trades + live signals + no
  pause = frozen, not calm); (c) `system_events` has a fresh heartbeat/tick row — an *empty* event
  window over a period that should have produced events is itself the alarm. "No news" is a null read
  on a possibly-dead channel, never a green light.
- Root cause (app-dependent one-shot scheduling silently no-ops safety-relevant transitions) is
  tracked in **GH #1038**; until it lands, treat every safety-relevant one-shot as best-effort and
  assert its resulting state in the sweep. Earned: 2026-07-17 stuck-entry-pause (log 2026-07-17
  ~11:35), #1038.
